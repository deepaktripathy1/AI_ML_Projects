"""Complete model deployment manager for BentoML integration."""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import bentoml

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.model.linear_regression import LinearRegressionModel


logger = LoggingConfig.get_logger(__name__)


class ModelDeploymentManager:
    """Model deployment manager with BentoML integration."""

    def __init__(self):
        """Initialize deployment manager."""
        settings = get_settings()

        self.bentoml_model_name = settings.BENTO_MODEL_NAME
        self.bentoml_service_name = settings.BENTO_SERVICE_NAME
        self.deployment_results = {}
        # Get the root directory i.e. model_production
        self.root_dir = settings.BASE_DIR
        self.deployment_dir = self.root_dir / "deployment" / "bentoml"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Model deployment initialized")

    def deploy_model_to_bentoml(
        self,
        model: LinearRegressionModel,
        model_version: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Deploy trained model to BentoML model store.

        Args:
            model: Trained LinearRegressionModel instance
            model_version: Version tag of the model
            metadata: Additional Metadata

        Returns:
            Dict containing deployment results
        """
        logger.info("Starting model deployment to BentoML")

        try:
            if not model.is_trained:
                raise ValueError("Model must be trained before deployment")

            # Generate version if not provided
            if model_version is None:
                model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Prepare model for BentoML
            sklearn_model = model.model
            scaler = model.scaler

            # Prepare metadata
            deployment_metadata = {
                "model_type": "LinearRegression",
                "creation_timestamp": datetime.now().isoformat(),
                "feature_names": model.feature_names,
                "normalize_features": model.normalize,
                "training_metrics": model.training_metrics,
                "model_metadata": model.model_metadata,
                "deployment_version": model_version,
            }

            if metadata:
                deployment_metadata.update(metadata)

            # Save main model to BentoML
            bento_model = bentoml.sklearn.save_model(
                name=self.bentoml_model_name,
                model=sklearn_model,
                labels={
                    "version": model_version,
                    "model_type": "LinearRegression",
                    "framework": "sklearn",
                    "stage": "production",
                },
                metadata=deployment_metadata,
                custom_objects={
                    "scaler": scaler,
                    "feature_names": model.feature_names,
                    "training_metrics": model.training_metrics,
                },
            )

            deployment_results = {
                "bentoml_model_tag": str(bento_model.tag),
                "model_name": self.bentoml_model_name,
                "model_version": model_version,
                "deployment_timestamp": datetime.now().isoformat(),
                "model_size_mb": self._get_model_size(bento_model),
                "deployment_status": "SUCCESS",
                "feature_count": len(model.feature_names),
                "model_path": str(bento_model.path),
            }

            self.deployment_results = deployment_results
            logger.info(
                f"Model successfully deployed to BentoML: {bento_model.tag}")

            return deployment_results

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            error_results = {
                "deployment_status": "FAILED",
                "error_message": str(e),
                "deployment_timestamp": datetime.now().isoformat(),
            }
            self.deployment_results = error_results
            raise

    def _get_model_size(self, bento_model) -> float:
        """Get the size of the BentoML model in MB."""
        try:
            model_path = Path(bento_model.path)
            if model_path.exists():
                size_bytes = sum(
                    f.stat().st_size for f in model_path.rglob("*") if f.is_file()
                )
                return round(size_bytes / (1024 * 1024), 2)
        except Exception as e:
            logger.error(f"Failed to calculate model size: {e}")
        return 0.0

    def create_bentofile(self) -> Path:
        """Create a BentoML bentofile.yaml file."""
        bentofile_path = self.deployment_dir / "bentofile.yaml"
        bentofile_content = f"""service: "service:svc"
labels:
    owner: ml-team
    stage: production
    version: "1.0.0"
    project: used-car-price-prediction

include:
    - "service.py"

python:
    packages:
        - "bentoml>=1.0.0"
        - "scikit-learn>=1.3.0"
        - "pandas>=2.0.0"
        - "numpy>=1.24.0"
        - "pydantic>=2.0.0"
    lock_packages: false

models:
    - "{self.bentoml_model_name}:latest"
"""
        with bentofile_path.open("w") as file:
            file.write(bentofile_content)

        logger.info(f"Created bentofile.yaml at {bentofile_path}")
        return bentofile_path

    def build_bento_service(
        self,
        service_version: str | None = None,
    ) -> Dict[str, Any]:
        """Build BentoML service for deployment.

        Args:
            service_version: Version for the service
            build_options: Additional build options

        Returns:
            Dict containing build results
        """
        logger.info("Building BentoML service")

        try:
            if not service_version:
                service_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Ensure bentofile exists
            bentofile_path = self.create_bentofile()

            # Ensure service.py exists in deployment directory
            service_source = self.deployment_dir / "service.py"
            if not service_source.exists():
                # Copy from the main directory if it exists there
                main_service = self.root_dir / "service.py"
                if main_service.exists():
                    import shutil
                    shutil.copy2(main_service, service_source)
                    logger.info(f"Copied service.py to {service_source}")
                else:
                    raise FileNotFoundError(
                        f"service.py not found. Please ensure it exists at {
                            service_source}"
                    )

            # Build command
            build_cmd = [
                "bentoml",
                "build",
                str(self.deployment_dir),
                "--version",
                service_version,
            ]

            # Execute build
            logger.info(f"Executing: {' '.join(build_cmd)}")
            result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.deployment_dir)
            )

            # Parse the build output to get service tag
            service_tag = self._parse_service_tag_from_output(
                result.stdout,
                service_version
            )

            build_results = {
                "service_name": self.bentoml_service_name,
                "service_version": service_version,
                "service_tag": service_tag,
                "build_timestamp": datetime.now().isoformat(),
                "build_status": "SUCCESS",
                "build_output": result.stdout,
            }

            logger.info(f"BentoML service built successfully: {service_tag}")
            return build_results

        except subprocess.CalledProcessError as e:
            logger.error(f"BentoML service build failed: {e}")
            logger.error(f"Build output: {e.stdout}")
            logger.error(f"Build errors: {e.stderr}")
            raise Exception(f"Service build failed: {e.stderr}") from e

        except Exception as e:
            logger.error(f"Service build failed: {e}")
            raise

    def _parse_service_tag_from_output(
            self,
            output: str,
            version: str
    ) -> str:
        """Parse service tag from build output."""
        try:
            lines = output.split("\n")
            for line in lines:
                if "Successfully built" in line or "Built Bento" in line:
                    # Look for pattern like "service_name:version"
                    parts = line.split()
                    for part in parts:
                        if ":" in part:
                            return part.strip()

            # Fallback: construct tag from service name and version
            return f"{self.bentoml_service_name}:{version}"

        except Exception as e:
            logger.error(f"Failed to parse service tag: {e}")
            return f"{self.bentoml_service_name}:{version}"

    def serve_local(
        self,
        service_tag: str,
        host: str = "0.0.0.0",
        port: int = 3001,
        production: bool = True,
        reload: bool = False,
    ) -> Dict[str, Any]:
        """Prepare configuration for serving BentoML service locally.

        Args:
            service_tag: BentoML service tag
            host: Host to bind to
            port: Port to bind to
            production: Whether to run in production mode
            reload: Whether to enable auto-reload

        Returns:
            Dict containing serve command and configuration
        """
        logger.info(f"Preparing to serve configuration for {service_tag}")

        # Build serve command
        serve_cmd = [
            "bentoml",
            "serve",
            service_tag,
            "--host",
            host,
            "--port",
            str(port),
        ]

        if production:
            serve_cmd.append("--production")

        if reload:
            serve_cmd.append("--reload")

        serve_info = {
            "service_tag": service_tag,
            "host": host,
            "port": port,
            "production": production,
            "reload": reload,
            "serve_command": " ".join(serve_cmd),
            "endpoint_url": f"http://{host}:{port}",
            "docs_url": f"http://{host}:{port}/docs",
            "health_url": f"http://{host}:{port}/health",
        }

        logger.info(
            f"Service configuration ready: {serve_info['endpoint_url']}"
        )

        return serve_info

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get current deployment information."""
        return {
            "deployment_results": self.deployment_results,
            "model_name": self.bentoml_model_name,
            "service_name": self.bentoml_service_name,
            "deployment_directory": str(self.deployment_dir),
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List BentoML models."""
        try:
            result = subprocess.run(
                ["bentoml", "models", "list", "--output", "json"],
                capture_output=True,
                text=True,
                check=True,
            )

            models = json.loads(result.stdout) if result.stdout.strip() else []
            return models

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def list_services(self) -> List[Dict[str, Any]]:
        """List BentoML services."""
        try:
            result = subprocess.run(
                ["bentoml", "list", "--output", "json"],
                capture_output=True,
                text=True,
                check=True,
            )

            services = json.loads(
                result.stdout) if result.stdout.strip() else []
            return services

        except Exception as e:
            logger.error(f"Failed to list services: {e}")
            return []

    def cleanup_old_models(self, keep_latest: int = 3) -> Dict[str, Any]:
        """Clean up old models, keeping only the latest N versions.

        Args:
            keep_latest: Number of latest models to keep

        Returns:
            Dict containing cleanup results
        """
        try:
            # List models
            models = self.list_models()

            # Filter models by name
            our_models = [
                m for m in models if m.get(
                    "name") == self.bentoml_model_name
            ]

            # Sort by creation time (newest first)
            our_models.sort(key=lambda x: x.get(
                "creation_time", ""), reverse=True)

            # Keep only the latest N models
            models_to_delete = our_models[keep_latest:]

            deleted_models = []
            for model in models_to_delete:
                tag = f"{model['name']}:{model['version']}"
                try:
                    subprocess.run(
                        ["bentoml", "models", "delete", tag],
                        check=True,
                        capture_output=True,
                    )
                    deleted_models.append(tag)
                    logger.info(f"Deleted old model: {tag}")
                except Exception as e:
                    logger.error(f"Failed to delete model {tag}: {e}")

            cleanup_results = {
                "total_models": len(our_models),
                "kept_models": len(our_models) - len(deleted_models),
                "deleted_models": deleted_models,
                "cleanup_timestamp": datetime.now().isoformat(),
            }

            return cleanup_results

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"error": str(e)}
