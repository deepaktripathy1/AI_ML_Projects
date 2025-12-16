"""Prefect Deployment Pipeline Flow."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import re
import numpy as np
import pandas as pd
import tempfile
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow import MlflowClient, artifacts
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.runtime import flow_run
from prefect.variables import Variable

from src.model.deployment_manager import ModelDeploymentManager
from src.model.linear_regression import LinearRegressionModel
from src.utils.path_utils import to_serializable_path, from_serializable_path

from config.config import get_settings


@task
def validate_model_readiness() -> Dict[str, Any]:
    """Validate that trained model is ready for deployment."""
    logger = get_run_logger()

    try:
        # Read training report
        report_file = Path("/tmp/training_report.json")
        if not report_file.exists():
            raise Exception("Training report not found")

        with report_file.open("r") as f:
            training_report = json.load(f)

        # Extract validation results
        validation_summary = training_report.get("validation_summary", {})
        model_accepted = validation_summary.get("model_accepted", False)

        if not model_accepted:
            blockers = training_report.get("model_readiness", {}).get(
                "deployment_bockers", []
            )
            raise Exception(f"Model not ready for deployment: {blockers}")

        # Get model path from training summary
        training_summary = training_report.get("training_summary", {})
        model_path = training_summary.get("model_path")

        if not model_path or not Path(model_path).exists():
            raise Exception("Model file not found at expected path")

        readiness_info = {
            "model_ready": True,
            "model_path": model_path,
            "validation_timestamp": datetime.now().isoformat(),
            "training_metrics": training_summary.get("test_rmse", "unknown"),
            "model_type": training_summary.get(
                "model_type", "LinearRegression"),
            "training_completion": training_report.get(
                "report_metadata", {}).get(
                    "generation_timestamp"
            ),
        }

        # Create model readiness artifact
        create_markdown_artifact(
            key="model-readiness-validation",
            markdown=f"""
# Model Readiness Validation

## Status: READY FOR DEPLOYMENT

### Model Information
- **Model Type:** {readiness_info['model_type']}
- **Model Path:** {readiness_info['model_path']}
- **Training Metrics:** RMSE = {readiness_info['training_metrics']}

### Validation Checks
- **Model File Exists:** YES
- **Training Completed:** YES
- **Performance Acceptable:** YES
- **Validation Passed:** YES

### Deployment Details
- **Validation Time:** {readiness_info['validation_timestamp']}
- **Training Completion:** {readiness_info['training_completion']}
            """,
            description="Model readiness validation report for deployment"
        )

        logger.info(f"Model readiness validated: {readiness_info}")
        return readiness_info

    except Exception as e:
        logger.error(f"Error validating model readiness: {e}")
        raise e


@task
def prepare_deployment_environment(
        readiness_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare deployment environment and load trained model."""
    logger = get_run_logger()

    try:
        # Create deployment directories
        deployment_dir = Path("/tmp/deployment")
        deployment_dir.mkdir(parents=True, exist_ok=True)

        # Load the trained model
        model_uri = None
        model_version = None
        model_path = None
        model_loaded = None
        model_source = "local"

        try:
            client = MlflowClient(mlflow.get_tracking_uri())

            # Try loading model from MLflow registry
            model_name = "used_car_price_pred_model"
            versions = client.search_model_versions(
                filter_string=f"name='{model_name}'"
            )

            if versions:
                model_version = max(int(v.version) for v in versions)
                model_uri = f"models:/{model_name}/{model_version}"
                model_loaded = mlflow_sklearn.load_model(model_uri=model_uri)
                model_source = "mlflow"
            else:
                logger.warning(f"No versions found for model {model_name}")

        except Exception as e:
            logger.warning(f"Could not load from MLflow: {e}")

        # Load the model from the path specified in readiness_info
        if model_loaded is None:
            model_path = from_serializable_path(readiness_info["model_path"])

            try:
                model_loaded = LinearRegressionModel.load_model(
                    str(model_path)
                )
                model_source = "local"
                logger.info("Model loaded successfully")
            except AttributeError:
                # Load model components directly
                model_components = {
                    "model": joblib.load(model_path / "model.pkl"),
                    "scaler": joblib.load(model_path / "scaler.pkl"),
                    "features": joblib.load(model_path / "features.pkl"),
                    "metadata": joblib.load(
                        model_path / "metadata.pkl"
                    ) if (model_path / "metadata.pkl").exists() else {},
                }
                model_loaded = model_components
                logger.info(f"Model loaded directly from {model_path}")

        # Prepare deployment metadata
        deployment_metadata = {
            "deployment_preparation_timestamp": datetime.now().isoformat(),
            "model_source": model_source,
            "deployment_directory": to_serializable_path(deployment_dir),
            "model_ready": True,
            "components_validated": True,
            "readiness_info": readiness_info,
        }

        if model_source == "mlflow":
            deployment_metadata["model_uri"] = model_uri
            deployment_metadata["model_version"] = model_version
        elif model_source == "local" and model_path:
            deployment_metadata["model_path"] = to_serializable_path(
                model_path
            )

        # Save deployment metadata
        metadata_file = deployment_dir / "deployment_metadata.json"
        with metadata_file.open("w") as f:
            json.dump(deployment_metadata, f, indent=2, default=str)

        logger.info(
            f"Deployment environment prepared: {deployment_metadata}")
        return deployment_metadata

    except Exception as e:
        logger.error(f"Error preparing deployment environment: {e}")
        raise


@task
def deploy_to_bentoml(
        deployment_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Deploy model to BentoML model store."""
    logger = get_run_logger()
    settings = get_settings()

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    try:
        # Initialize deployment manager
        deployment_manager = ModelDeploymentManager()

        trained_model = None

        # Try to load model directly from MLFlow
        if "model_uri" in deployment_metadata:
            model_uri = deployment_metadata["model_uri"]
            mlflow_model = mlflow_sklearn.load_model(model_uri=model_uri)

            scaler = None
            features = None
            metadata = None

            # Try loading scaler, features and metadata from MLflow
            try:
                # Get MLflow experiment
                experiment = mlflow.get_experiment_by_name(
                    name="used_car_price_prediction"

                )

                if experiment is not None:
                    exp_id = experiment.experiment_id
                    runs = mlflow.search_runs(
                        experiment_ids=exp_id,
                        order_by=["start_time DESC"],
                        max_results=1,
                        output_format="list"
                    )

                    if runs is not None:
                        run_info = runs[0].info
                        artifact_uri = getattr(run_info, "artifact_uri", None)

                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Download scaler
                            scaler_path = artifacts.download_artifacts(
                                artifact_uri=(
                                    f"{artifact_uri}/scaler/scaler.pkl"
                                ),
                                dst_path=str(temp_dir)
                            )
                            scaler = joblib.load(scaler_path)

                            # Download features
                            features_path = artifacts.download_artifacts(
                                artifact_uri=(
                                    f"{artifact_uri}/features/features.pkl"
                                ),
                                dst_path=str(temp_dir)
                            )
                            features = joblib.load(features_path)

                            # Download metadata
                            metadata_path = artifacts.download_artifacts(
                                artifact_uri=(
                                    f"{artifact_uri}/metadata/metadata.pkl"
                                ),
                                dst_path=str(temp_dir)
                            )
                            metadata = joblib.load(metadata_path)

                            logger.info("All artifacts loaded from MLflow")

            except Exception as e:
                logger.warning("Could not load all artifacts from MLflow: {e}")

            # Assemble LinearRegressionModel Instance
            trained_model = LinearRegressionModel.from_mlflow(
                model=mlflow_model,
                scaler=scaler,
                features=features,
                metadata=metadata
            )

            logger.info("Model loaded from MLflow for BentoML deployment")

        # Load from local path
        elif "model_path" in deployment_metadata:
            model_path = from_serializable_path(
                deployment_metadata["model_path"]
            )
            try:
                trained_model = LinearRegressionModel.load_model(
                    str(model_path)
                )
                logger.info(
                    "Model loaded from local path for BentoML deployment"
                )

            except (AttributeError, FileNotFoundError):
                # Load components directly
                trained_model = LinearRegressionModel()
                trained_model.model = joblib.load(
                    model_path / "model.pkl"
                )
                trained_model.scaler = joblib.load(
                    model_path / "scaler.pkl"
                )
                trained_model.feature_names = joblib.load(
                    model_path / "features.pkl"
                )
                trained_model.model_metadata = joblib.load(
                    model_path / "metadata.pkl"
                ) if (model_path / "metadata.pkl").exists() else {}

                trained_model.is_trained = True
                logger.info("Model reconstructed from components")

            # Load scaler if available
            if trained_model is not None:
                scaler_path = model_path / "scaler.pkl"
                if scaler_path.exists() and (not hasattr(
                    trained_model, "scaler"
                ) or trained_model.scaler is None):
                    trained_model.scaler = joblib.load(scaler_path)
                    logger.info("Scaler loaded from local path.")

                # Load features if exists
                features_path = model_path / "features.pkl"
                if features_path.exists() and (not hasattr(
                    trained_model, "feature_names"
                ) or not trained_model.feature_names):
                    trained_model.feature_names = joblib.load(features_path)
                    logger.info("Features loaded from local path.")

                # Load metadata if available
                metadata_path = model_path / "metadata.pkl"
                if metadata_path.exists() and (not hasattr(
                    trained_model, "model_metadata"
                ) or not trained_model.model_metadata):
                    trained_model.model_metadata = joblib.load(metadata_path)
                    logger.info("Metadata loaded from local path")

                trained_model.is_trained = True

                logger.info("All model components verified and loaded")

        else:
            raise ValueError("Deployment metadata missing")

        # Generate deployment version
        deployment_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Deploy to BentoML
        bentoml_deployment_result = deployment_manager.deploy_model_to_bentoml(
            model=trained_model,
            model_version=deployment_version,
            metadata={
                "deployment_flow_run": str(flow_run.get_id()),
                "deployment_timestamp": datetime.now().isoformat(),
                "triggered_by": "deployment_pipeline",
                "source_training_run": deployment_metadata.get(
                    "readiness_info", {}).get(
                    "training_completion", "unknown"
                ),
            },
        )

        # Create BentoML deployment artifact
        create_markdown_artifact(
            key="bentoml-deployment-result",
            markdown=f"""
# BentoML Deployment Result

## Status: {
                'SUCCESS' if bentoml_deployment_result.get(
                    'deployment_status') == 'SUCCESS' else 'FAILED'
            }

### Deployment Details
- **Model Version:** {deployment_version}
- **BentoML Model Tag:** {bentoml_deployment_result.get(
                'bentoml_model_tag', 'N/A'
            )}
- **Model Size:** {bentoml_deployment_result.get('model_size_mb', 0)} MB

### Metadata
- **Flow Run ID:** {flow_run.get_id()}
- **Deployment Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """,
            description="BentoML model deployment results and metadata"
        )

        logger.info(
            f"BentoML deployment successful: {bentoml_deployment_result}")
        return bentoml_deployment_result

    except Exception as e:
        logger.error(f"BentoML deployment failed: {e}")
        raise


@task
def build_bentoml_service(
    _bentoml_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Build BentoML service for deployment."""
    logger = get_run_logger()

    try:
        # Initialize deployment manager
        deployment_manager = ModelDeploymentManager()

        # Generate service version
        service_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Build BentoML service
        build_result = deployment_manager.build_bento_service(
            service_version=service_version,
        )

        # Create service build artifact
        create_markdown_artifact(
            key="bentoml-service-build",
            markdown=f"""
# BentoML Service Build

## Status: {
                'SUCCESS' if build_result.get(
                    'build_status') == 'SUCCESS' else 'FAILED'
            }

### Build Details
- **Service Version:** {service_version}
- **Service Tag:** {build_result.get('service_tag', 'N/A')}
- **Build Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Next Steps
Use the following command to serve the model:
'''bash
{deployment_manager.serve_local(build_result.get('service_tag', 'N/A'))['serve_command']}
'''
            """,
            description="BentoML service build results and configuration"
        )

        logger.info(f"BentoML service successfully built: {build_result}")
        return build_result

    except Exception as e:
        logger.error(f"BentoML service build failed: {e}")
        raise


@task
def prepare_fastapi_service() -> Dict[str, Any]:
    """Prepare FastAPI service for deployment."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Check if FastAPI service files exist
        root_dir = settings.BASE_DIR
        api_dir = root_dir / "src" / "api"

        required_files = ["main.py", "models.py", "endpoints.py"]
        missing_files = [
            str(api_dir / f) for f in required_files if not (
                api_dir / f).exists()
        ]

        if missing_files:
            logger.warning(f"Missing FastAPI files: {missing_files}")
            # Fallback - create minimal service
            api_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal FastAPI service if files are missing
            simple_service = '''
from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Used Car Price Prediction API")

@app.get("/")
def root():
    return {
    "message": "FastAPI service is running",
    "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
'''
            with (api_dir / "simple_service.py").open("w") as f:
                f.write(simple_service)

            service_info = {
                "fastapi_ready": True,
                "service_type": "fallback",
                "service_file": str(api_dir / "simple_service.py"),
                "preparation_timestamp": datetime.now().isoformat(),
            }
        else:
            service_info = {
                "fastapi_ready": True,
                "service_type": "full",
                "api_directory": str(api_dir),
                "required_files_present": True,
                "preparation_timestamp": datetime.now().isoformat(),
            }

        # Create FastAPI preparation artifact
        create_markdown_artifact(
            key="fastapi-service-preparation",
            markdown=f"""
# FastAPI Service Preparation

## Status: READY

### Service Configuration
- **Service Type:** {service_info['service_type']}
- **API Directory:** {service_info.get('api_directory', 'N/A')}
- **Service File:** {service_info.get('service_file', 'N/A')}

### Files Status
- **Required Files Present:** {service_info.get('required_files_present', False)}
- **Missing Files:** {
                len(missing_files)
            } files {
                '(using fallback)' if missing_files else ''
            }

### Preparation Details
- **Preparation Time:** {service_info['preparation_timestamp']}
- **Ready for Deployment:** {service_info['fastapi_ready']}
            """,
            description="FastAPI service preparation status and configuration"
        )

        logger.info(f"FastAPI service prepared: {service_info}")
        return service_info

    except Exception as e:
        logger.error(f"FastAPI service preparation failed: {e}")
        raise


@task
def start_bentoml_service(build_result: Dict[str, Any]) -> Dict[str, Any]:
    """Start BentoML service locally."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Initialize deployment manager
        deployment_manager = ModelDeploymentManager()

        service_tag = build_result.get(
            "service_tag", "used_car_price_prediction_service:latest"
        )

        if "Bento(tag=" in service_tag:
            match = re.search(r'tag="([^"]+)"', service_tag)
            if match:
                service_tag = match.group(1)

        # Remove trailing periods, quotes or parentheses
        service_tag = service_tag.rstrip('.")').strip('"')

        logger.info(f"Cleaned service tag: {service_tag}")

        # Prepare serve configuration
        serve_info = deployment_manager.serve_local(
            service_tag=service_tag,
            host="0.0.0.0",
            port=3001,
            production=True,
            reload=False,
        )

        # Create a script to start the service
        root_dir = settings.BASE_DIR
        script_dir = root_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)

        # Create start script for Linux/macOS
        start_script_sh = script_dir / "start_bentoml_service.sh"
        script_content_sh = f"""#!/bin/bash
# =============================================================================
# BentoML Service Startup Script (Linux/macOS)
# Generated by: Prefect Deployment Pipeline
# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# =============================================================================
set -e

echo "========================================="
echo "Starting BentoML service..."
echo "========================================="
echo ""
echo "Service tag: {service_tag}"
echo "Host: 0.0.0.0"
echo "Port: 3001"
echo "Mode: Production"
echo ""

# Check if BentoML is installed
if ! command -v bentoml &> /dev/null; then
    echo "ERROR: BentoML is not installed"
    echo "Install with: pip install bentoml"
    exit 1
fi

# Check if service exists
echo "Checking if service exists..."
if bentoml list | grep -q "{service_tag}"; then
    echo "Service found: {service_tag}"
else
    echo "WARNING: Service {service_tag} not found in BentoML store"
    echo ""
    echo "Available services:"
    bentoml list
    echo ""
    read -p "Continue with latest service? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    SERVICE_TAG="used_car_price_prediction_service:latest"
fi

echo ""
echo "Starting BentoML service..."
echo "Access the service at: http://localhost:3001"
echo "API documentation: http://localhost:3001/docs"
echo "Metrics endpoint: http://localhost:3001/metrics"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start BentoML service
bentoml serve {service_tag} --host 0.0.0.0 --port 3001 --production
"""

        with start_script_sh.open("w", newline="\n") as f:
            f.write(script_content_sh)

        # Make executable on Unix systems
        try:
            start_script_sh.chmod(0o755)
            logger.info(f"Created executable .sh script: {start_script_sh}")
        except Exception as e:
            logger.warning(f"Could not set executable permission: {e}")

        # Creat start script for Windows
        start_script_bat = script_dir / "start_bentoml_service.bat"
        script_content_bat = f"""@echo off
REM ============================================================================
REM BentoML Service Startup Script (Windows)
REM Generated by: Prefect Deployment Pipeline
REM Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
REM ============================================================================

echo.
echo =========================================
echo Starting BentoML Service
echo =========================================
echo.
echo Service tag: {service_tag}
echo Host: 0.0.0.0
echo Port: 3001
echo Mode: Production
echo.

REM Check if BentoML is installed
bentoml --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: BentoML is not installed
    echo Install with: pip install bentoml
    pause
    exit /b 1
)

REM Check if service exists
echo Checking if service exists...
bentoml list | findstr "{service_tag}" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Service found: {service_tag}
    set "SERVICE_TAG={service_tag}"
) else (
    echo WARNING: Service {service_tag} not found in BentoML store
    echo.
    echo Available services:
    bentoml list
    echo.
    echo Using latest service instead...
    set "SERVICE_TAG=used_car_price_prediction_service:latest"
)

echo.
echo Starting BentoML service...
echo Access the service at: http://localhost:3001
echo API documentation: http://localhost:3001/docs
echo Metrics endpoint: http://localhost:3001/metrics
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start BentoML service
bentoml serve %SERVICE_TAG% --host 0.0.0.0 --port 3001 --production
"""
        with start_script_bat.open("w", newline="\r\n") as f:
            f.write(script_content_bat)

        logger.info(f"Created .bat script: {start_script_bat}")

        service_status = {
            "service_started": True,
            "service_tag": service_tag,
            "serve_info": serve_info,
            "start_script_sh": str(start_script_sh),
            "start_script_bat": str(start_script_bat),
            "start_timestamp": datetime.now().isoformat(),
            "service_ready": True
        }

        # Create service startup artifact
        create_markdown_artifact(
            key="bentoml-service-startup",
            markdown=f"""
# BentoML Service Startup

## Status: CONFIGURED

### Service Details
- **Service Tag:** {service_tag}
- **Port:** 3001
- **Host:** 0.0.0.0
- **Mode:** Production

### Service Endpoints
- **Main:** http://localhost:3001
- **Health:** http://localhost:3001/health
- **Documentation:** http://localhost:3001/docs
- **Metrics:** http://localhost:3001/metrics

### Start Scripts Generated

#### Linux/macOS:
**Script** `{start_script_sh}`

**Run with:**
```bash
bash {start_script_sh}
```

#### Windows:
**Script:** `{start_script_bat}`

**Run with:**
- Double-click the .bat file
- Or run: `{start_script_bat}`

Or double-click the .bat file!

### Manual Start Command
```bash
bentoml serve {service_tag} --host 0.0.0.0 --port 3001 --production
```
""",
            description="BentoML service startup configuration"
        )

        logger.info(f"BentoML service configured: {service_tag}")
        logger.info(f"Scripts created: .sh and .bat")
        return service_status

    except Exception as e:
        logger.error(f"BentoML service startup failed: {e}")
        raise


@task
def start_fastapi_service(
    fastapi_info: Dict[str, Any],
    _bentoml_status: Dict[str, Any]
) -> Dict[str, Any]:
    """Start FastAPI service."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Create script to start FastAPI service
        root_dir = settings.BASE_DIR
        script_dir = root_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)

        # Determine which service to start
        if fastapi_info["service_type"] == "full":
            service_module = "src.api.endpoints:app"
            service_description = "Full FastAPI service with all endpoints"
        else:
            service_module = "src.api.simple_service:app"
            service_description = "Fallback FastAPI service"

        # Create start script for Linux/macOS
        start_script_sh = script_dir / "start_fastapi_service.sh"
        script_content_sh = f"""#!/bin/bash
# =============================================================================
# FastAPI Service Startup Script (Linux/macOS)
# Generated by: Prefect Deployment Pipeline
# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# =============================================================================
set -e

echo "========================================="
echo "Starting FastAPI Service"
echo "========================================="
echo ""
echo "Service: {service_description}"
echo "Module: {service_module}"
echo "Host: 0.0.0.0"
echo "Port: 8000"
echo "Mode: Development (with auto-reload)"
echo ""

# Get project root (parent of scripts directory)
PROJECT_ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if uvicorn is installed
if ! command -v uvicorn &> /dev/null; then
    echo "ERROR: Uvicorn is not installed"
    echo "Install with: pip install uvicorn[standard]"
    exit 1
fi

# Check if the API module exists
if [ ! -f "src/api/main.py" ] && [ ! -f "src/api/endpoints.py" ]; then
    echo "ERROR: FastAPI module files not found"
    echo "Expected: src/api/main.py or src/api/endpoints.py"
    echo "Current directory: $(pwd)"
    exit 1
fi

echo "Starting FastAPI service..."
echo "Access the service at: http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo "Metrics endpoint: http://localhost:8000/metrics"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Start FastAPI service
uvicorn {service_module} --host 0.0.0.0 --port 8000 --reload
"""

        with start_script_sh.open("w", newline="\n") as f:
            f.write(script_content_sh)

        # Make executable on Unix systems

        try:
            start_script_sh.chmod(0o755)
            logger.info(f"Created executable .sh script: {start_script_sh}")
        except Exception as e:
            logger.warning(f"Could not set executable permission: {e}")

        # Create start script for Windows
        start_script_bat = script_dir / "start_fastapi_service.bat"
        script_content_bat = f"""@echo off
REM ============================================================================
REM FastAPI Service Startup Script (Windows)
REM Generated by: Prefect Deployment Pipeline
REM Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
REM ============================================================================

echo.
echo =========================================
echo Starting FastAPI Service
echo =========================================
echo.
echo Service: {service_description}
echo Module: {service_module}
echo Host: 0.0.0.0
echo Port: 8000
echo Mode: Development (with auto-reload)
echo.

REM Get project root (parent of scripts directory)
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

echo Project root: %PROJECT_ROOT%
echo.

REM Check if uvicorn is installed
uvicorn --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Uvicorn is not installed
    echo Install with: pip install uvicorn[standard]
    pause
    exit /b 1
)

REM Check if the API module exists
if not exist "src\\api\\main.py" (
    if not exist "src\\api\\endpoints.py" (
        echo ERROR: FastAPI module files not found
        echo Expected: src\\api\\main.py or src\\api\\endpoints.py
        echo Current directory: %CD%
        pause
        exit /b 1
    )
)

echo Starting FastAPI service...
echo Access the service at: http://localhost:8000
echo API documentation: http://localhost:8000/docs
echo Metrics endpoint: http://localhost:8000/metrics
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start FastAPI service
uvicorn {service_module} --host 0.0.0.0 --port 8000 --reload
"""
        with start_script_bat.open("w", newline="\r\n") as f:
            f.write(script_content_bat)

        logger.info(f"Created .bat script: {start_script_bat}")

        fastapi_status = {
            "service_ready": True,
            "service_module": service_module,
            "service_description": service_description,
            "start_script_sh": str(start_script_sh),
            "start_script_bat": str(start_script_bat),
            "port": 8000,
            "start_timestamp": datetime.now().isoformat(),
            "service_info": fastapi_info,
        }

        # Create FastAPI startup artifact
        create_markdown_artifact(
            key="fastapi-service-startup",
            markdown=f"""
# FastAPI Service Startup

## Status: CONFIGURED

### Service Details
- **Service Type:** {fastapi_info['service_type']}
- **Module:** {service_module}
- **Port:** 8000
- **Host:** 0.0.0.0
- **Mode:** Development (auto-reload enabled)

### Service Endpoints
- **Main:** http://localhost:8000
- **Health:** http://localhost:8000/health
- **Documentation:** http://localhost:8000/docs
- **Metrics:** http://localhost:8000/metrics

### Start Scripts Generated

#### Linux/macOS:
**Script:** `{start_script_sh}`

**Run with:**
```bash
bash {start_script_sh}
```

#### Windows:
**Script:** `{start_script_bat}`

**Run with:**
- Double-click the .bat file
- Or run: `{start_script_bat}`

### Manual Start Command
```bash
uvicorn {service_module} --host 0.0.0.0 --port 8000 --reload
```
""",
            description="FastAPI service configuration and startup scripts"
        )

        logger.info(f"FastAPI service configured: {service_module}")
        logger.info(f"Scripts created: .sh and .bat")
        return fastapi_status

    except Exception as e:
        logger.error(f"FastAPI service startup failed: {e}")
        raise


@task
def create_deployment_summary(
    bentoml_result: Dict[str, Any],
    build_result: Dict[str, Any],
    bentoml_status: Dict[str, Any],
    fastapi_status: Dict[str, Any],
) -> Dict[str, Any]:
    """Create comprehensive deployment summary."""
    logger = get_run_logger()

    try:
        summary = {
            "deployment_metadata": {
                "flow_run_id": str(flow_run.get_id()),
                "flow_run_name": flow_run.get_name(),
                "deployment_timestamp": datetime.now().isoformat(),
                "triggered_by": "training_pipeline_completion",
            },
            "bentoml_deployment": {
                "model_deployed": bentoml_result.get(
                    "deployment_status") == "SUCCESS",
                "model_tag": bentoml_result.get("bentoml_model_tag"),
                "service_built": build_result.get(
                    "build_status") == "SUCCESS",
                "service_tag": build_result.get("service_tag"),
                "service_ready": bentoml_status.get("service_ready", False),
            },
            "fastapi_deployment": {
                "service_prepared": fastapi_status.get(
                    "service_ready", False),
                "service_type": fastapi_status.get(
                    "service_info", {}).get("service_type", "unknown"),
            },
            "service_endpoints": {
                "bentoml": {
                    "url": "http://localhost:3001",
                    "docs": "http://localhost:3001/docs",
                    "health": "http://localhost:3001/health",
                },
                "fastapi": {
                    "url": "http://localhost:8000",
                    "docs": "http://localhost:8000/docs",
                    "health": "http://localhost:8000/health",
                },
            },
            "deployment_status": {
                "overall_success": (
                    bentoml_result.get("deployment_status") == "SUCCESS"
                    and build_result.get("build_status") == "SUCCESS"
                ),
                "services_configured": 2,
            },
            "next_steps": [
                f"Start BentoML: {bentoml_status.get(
                    'start_script', 'See configuration'
                )}",
                f"Start FastAPI: {fastapi_status.get(
                    'start_script', 'See configuration'
                )}",
                "Test prediction endpoints manually",
                "Set up monitoring and alerting",
            ],
        }

        # Save summary to file
        summary_file = Path("/tmp/deployment_summary.json")

        def to_serializable_paths(obj):
            """Recursively convert any Path objects to POSIX strings."""
            if isinstance(obj, Path):
                return to_serializable_path(obj)
            if isinstance(obj, dict):
                return {k: to_serializable_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serializable_paths(v) for v in obj]
            return obj

        with summary_file.open("w") as f:
            json.dump(to_serializable_paths(summary), f, indent=2, default=str)

        # Create comprehensive deployment summary artifact
        create_markdown_artifact(
            key="deployment-summary",
            markdown=f"""
# Deployment Summary

## Overall Status: {
                'SUCCESS' if summary[
                    'deployment_status'
                ]
                ['overall_success'
                 ] else 'PARTIAL/FAILED'
            }

### Deployment Results
- **Model Deployed:** {
                'YES' if summary['bentoml_deployment']['model_deployed'] else 'NO'
            }
- **Service Built:** {
                'YES' if summary['bentoml_deployment']['service_built'] else 'NO'
            }
- **Services Configured:** {
                summary['deployment_status']['services_configured']
            }/2

### Service Endpoints

#### BentoML Service
- **URL:** [http://localhost:3001](http://localhost:3001)
- **API Docs:** [http://localhost:3001/docs](http://localhost:3001/docs)
- **Health Check:** [http://localhost:3001/health](http://localhost:3001/health)

#### FastAPI Service
- **URL:** [http://localhost:8000](http://localhost:8000)
- **API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

### Deployment Details
- **Flow Run:** {summary['deployment_metadata']['flow_run_name']}
- **Deployment Time:** {summary['deployment_metadata']['deployment_timestamp']}
- **Model Tag:** {summary['bentoml_deployment']['model_tag']}
- **Service Tag:** {summary['bentoml_deployment']['service_tag']}

### Next Steps
{
                chr(10).join([f"{i+1}. {step}" for i,
                              step in enumerate(summary['next_steps'])])
            }
            """,
            description="Comprehensive deployment summary"
        )

        logger.info(f"Deployment summary created: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Failed to create deployment summary: {e}")
        raise


@task
def setup_monitoring_integration(
        _deployment_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Set up monitoring integration after successful deployment."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Create reference data for monitoring (if not exists)
        reference_data_path = Path("data/reference_data.csv")
        if not reference_data_path.exists():
            logger.info("Creating reference data for monitoring...")

            reference_data_path.parent.mkdir(parents=True, exist_ok=True)

            # Create a sample reference data
            np.random.seed(42)
            n_samples = 1000

            brands = ["Honda", "Toyota", "Ford", "Chevrolet", "BMW",
                      "Volkswagen", "Tesla", "Hyundai", "Kia", "Nissan"]
            fuel_types = ["Petrol", "Diesel", "Electric"]
            transmissions = ["Manual", "Automatic"]
            colors = ["Black", "Blue", "Gray", "Red", "Silver", "White"]

            reference_data = pd.DataFrame({
                "make_year": np.random.randint(2010, 2024, n_samples),
                "mileage_kmpl": np.random.normal(15, 3, n_samples).clip(8, 30),
                "engine_cc": np.random.randint(1000, 3000, n_samples),
                "accidents_reported": np.random.poisson(0.5, n_samples).clip(0, 5),
                "owner_count": np.random.randint(1, 4, n_samples),
                "brand": np.random.choice(brands, n_samples),
                "fuel_type": np.random.choice(fuel_types, n_samples),
                "transmission": np.random.choice(transmissions, n_samples),
                "color": np.random.choice(colors, n_samples),
                "insurance_valid": np.random.choice(["Yes", "No"], n_samples),
                "service_history": np.random.choice(
                    ["Full", "Partial", "None"], n_samples
                ),
            })

            # Add derived features
            reference_data["car_age"] = settings.CURRENT_YEAR - reference_data[
                "make_year"]
            reference_data["has_accident"] = (
                reference_data["accidents_reported"] > 0).astype(int)

            # Generate synthetic prices based on features
            price = (
                50000
                - reference_data["car_age"] * 2000
                + reference_data["mileage_kmpl"] * 500
                - reference_data["accidents_reported"] * 3000
                + np.where(
                    reference_data["brand"].isin(
                        ["BMW", "Volkswagen", "Tesla"]
                    ), 10000, 0
                )
                + np.random.normal(0, 2000, n_samples)
            ).clip(5000, 100000)

            reference_data["price_usd"] = price

            reference_data.to_csv(reference_data_path, index=False)
            logger.info(f"Reference data created at {reference_data_path}")

        monitoring_setup = {
            "reference_data_ready": reference_data_path.exists(),
            "reference_data_path": str(reference_data_path),
            "monitoring_ready": True,
            "setup_timestamp": datetime.now().isoformat(),
        }

        return monitoring_setup

    except Exception as e:
        logger.warning(f"Monitoring setup completed with warnings: {e}")
        return {"monitoring_ready": False, "error": str(e)}


@task
def cleanup_deployment_artifacts() -> Dict[str, Any]:
    """Clean up temporary deployment files."""
    logger = get_run_logger()

    try:
        temp_paths = [
            "/tmp/deployment/deployment_metadata.json",
            "/tmp/deployment",
        ]

        cleaned_files = []
        for path_str in temp_paths:
            path = Path(path_str)
            try:
                if path.is_file():
                    path.unlink()
                    cleaned_files.append(str(path))
                elif path.is_dir():
                    import shutil
                    shutil.rmtree(path)
                    cleaned_files.append(str(path))
            except Exception as e:
                logger.error(f"Failed to clean up {path}: {e}")

        cleanup_results = {
            "files_cleaned": len(cleaned_files),
            "cleaned_paths": cleaned_files,
            "cleanup_timestamp": datetime.now().isoformat(),
            "deployment_artifacts_preserved": True,
        }

        logger.info(f"Deployment cleanup completed: {cleanup_results}")
        return cleanup_results

    except Exception as e:
        logger.warning(f"Cleanup completed with warnings: {e}")
        return {"cleanup_status": "completed_with_warnings", "error": str(e)}


@flow(
    name="deployment-pipeline",
    description="Complete ML model deployment pipeline",
    flow_run_name=f"deployment-pipeline-{
        datetime.now().strftime('%Y%m%d-%H%M%S')}",
)
def deployment_pipeline() -> dict[str, Any]:
    """Complete ML model deployment pipeline flow."""
    logger = get_run_logger()

    try:
        # Validate model readiness
        model_readiness = validate_model_readiness()

        # Prepare deployment environment
        deployment_preparation = prepare_deployment_environment(
            readiness_info=model_readiness
        )

        # Deploy to BentoML
        bentoml_deployment = deploy_to_bentoml(
            deployment_metadata=deployment_preparation
        )

        # Build BentoML service
        bentoml_build = build_bentoml_service(
            _bentoml_result=bentoml_deployment
        )

        # Prepare FastAPI service
        fastapi_preparation = prepare_fastapi_service()

        # Start both services
        bentoml_start = start_bentoml_service(
            build_result=bentoml_build
        )
        fastapi_start = start_fastapi_service(
            fastapi_info=fastapi_preparation,
            _bentoml_status=bentoml_start
        )

        # Create deployment summary
        deployment_summary = create_deployment_summary(
            bentoml_result=bentoml_deployment,
            build_result=bentoml_build,
            bentoml_status=bentoml_start,
            fastapi_status=fastapi_start,
        )

        # Set up monitoring integration
        monitoring_setup = setup_monitoring_integration(deployment_summary)

        # Clean up artifacts
        cleanup_result = cleanup_deployment_artifacts()

        # Create final deployment state
        deployment_state = {
            "status": "SUCCESS" if deployment_summary.get(
                "deployment_status", {}
            ).get("overall_success", False) else "FAILED",
            "completion_time": datetime.now().isoformat(),
            "model_deployed": deployment_summary.get(
                "bentoml_deployment", {}).get(
                "model_deployed", False
            ),
            "service_built": deployment_summary.get(
                "bentoml_deployment", {}).get(
                "service_built", False
            ),
        }

        # Store state in Prefect variable
        Variable.set(
            "deployment_pipeline_state",
            json.dumps(deployment_state, default=str),
            overwrite=True
        )

        return deployment_state

    except Exception as e:
        logger.error(f"Deployment pipeline failed: {e}")

        # Clean up on failure
        cleanup_deployment_artifacts()

        # Store failure state
        failure_state = {
            "status": "FAILED",
            "completion_time": datetime.now().isoformat(),
            "error": str(e),
            "model_deployed": False,
            "service_built": False,
            "services_configured": 0,
            "monitoring_ready": False,
        }

        Variable.set(
            "deployment_pipeline_state",
            json.dumps(failure_state, default=str),
            overwrite=True
        )

        raise


if __name__ == "__main__":
    deployment_pipeline()
