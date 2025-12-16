"""Prefect Training Pipeline Flow."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import mlflow
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import flow_run
from prefect.variables import Variable

from src.feature_store.feature_store_client import FeatureStoreClient
from src.model.model_trainer_pipeline import ModelTrainer
from config.config import get_settings


@task(cache_policy=TASK_SOURCE + INPUTS, cache_expiration=timedelta(minutes=30))
def check_feature_store_readiness() -> Dict[str, Any]:
    """Check if feature store has fresh data for training."""
    logger = get_run_logger()

    try:
        feature_store_client = FeatureStoreClient()
        client_type = feature_store_client.get_client_type()

        logger.info(f"Using Feature Store Client: {client_type}")

        # check if feature group exists and has recent data
        try:
            feature_group = feature_store_client.get_feature_group(
                name="used_car_features", version=1
            )

            # Get basic statistics about the feature group
            primary_key = getattr(feature_group, "primary_key", [])
            fg_stats = {
                "feature_store": client_type,
                "name": getattr(
                    feature_group, "name", "used_car_features"
                ),
                "version": getattr(feature_group, 'version', 1),
                "features_count": len(
                    getattr(feature_group, "features", []) if hasattr(
                        feature_group, "features") else []
                ),
                "primary_key": primary_key,
            }
            logger.info(f"Feature store check passed: {fg_stats}")
            return fg_stats

        except Exception as e:
            logger.error(
                f"Feature group not found or inaccessible: {e}"
            )
            raise Exception("Feature store not ready for training") from e

    except Exception as e:
        logger.error(f"Failed to check feature store readiness: {e}")
        raise


@task
def validate_training_prerequisites(
    feature_store_meta: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate model training prerequisites."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Try to create/get experiment
        try:
            experiment = mlflow.get_experiment_by_name(
                settings.MLFLOW_EXPERIMENT_NAME
            )
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=settings.MLFLOW_EXPERIMENT_NAME
                )
                logger.info(
                    f"Created new MLflow experiment: {experiment_id}"
                )
            else:
                logger.info(
                    f"Using existing MLflow experiment: {
                        experiment.experiment_id}"
                )
        except Exception as e:
            logger.error(f"MLflow connectivity check failed: {e}")
            raise

        # Check feature store readiness
        if not feature_store_meta or "features_count" not in feature_store_meta:
            raise ValueError("Invalid feature store metadata")

        min_features_required = 5
        if feature_store_meta["features_count"] < min_features_required:
            raise ValueError(
                "Insufficient features:"
                "{feature_store_meta['features_count']} < {min_features_required}"
            )

        validation_results = {
            "mlflow_ready": True,
            "feature_store_ready": True,
            "features_available": feature_store_meta["features_count"],
            "validation_timestamp": datetime.now().isoformat(),
            "triggered_by_data_pipeline": True,
        }

        logger.info(
            f"Training prerequisites validated: {validation_results}"
        )

        # Create prerequisites validation artifact
        create_markdown_artifact(
            key="training-prerequisites",
            markdown=f"""
# Training Prerequisites Validation

## Status: PASSED

### MLflow Connection
- **Status:** Connected
- **Tracking URI:** {settings.MLFLOW_TRACKING_URI}
- **Experiment:** {settings.MLFLOW_EXPERIMENT_NAME}

### Feature Store Readiness
- **Features Available:** {validation_results['features_available']}
- **Minimum Required:** {min_features_required}
- **Status:** Ready

### Data Pipeline Integration
- **Triggered by Data Pipeline:** {
                validation_results['triggered_by_data_pipeline']}
    """,
            description="Training prerequisites validation report"
        )

        return validation_results

    except Exception as e:
        logger.error(f"Failed to validate training prerequisites: {e}")
        raise


@task
def prepare_training_environment(
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Prepare training environment."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Ensure model directories exist
        settings.models_dir.mkdir(parents=True, exist_ok=True)

        # Clear any previous temporary training files
        temp_paths = [
            "/tmp/training_data.pkl",
            "/tmp/model_artifacts.pkl",
            "/tmp/training_results.json",
        ]

        for temp_path in temp_paths:
            path = Path(temp_path)
            if path.exists():
                path.unlink()
                logger.info(f"Cleaned up previous temp file: {temp_path}")

        environment_info = {
            "models_dir": str(settings.models_dir),
            "temp_files_cleaned": len(
                [p for p in temp_paths if not Path(p).exists()]
            ),
            "preparation_timestamp": datetime.now().isoformat(),
            "random_state": settings.RANDOM_STATE,
            "triggered_by_external_flow": True,
            "prerequisites": validation_results,
        }

        logger.info(f"Training environment prepared: {environment_info}")
        return environment_info

    except Exception as e:
        logger.error(f"Failed to prepare training environment: {e}")
        raise


@task
def execute_model_training(
    _environment_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute model training."""
    logger = get_run_logger()

    try:
        # Initalize model trainer
        model_trainer = ModelTrainer()

        logger.info("Training triggered by data pipeline completion")

        # Execute training pipeline
        training_results = model_trainer.train_model(
            use_feature_store=True,
            validation_enabled=True,
            save_model=True,
        )

        # Save training results to temporary location
        results_path = Path("/tmp/training_results.json")
        with results_path.open("w") as f:
            json.dump(training_results, f, indent=2, default=str)

        # Extract key metrics for task return
        training_summary = {
            "training_status": training_results.get(
                "training_status", "UNKNOWN"
            ),
            "model_type": training_results.get(
                "model_type", "LinearRegression"
            ),
            "training_timestamp": training_results.get("training_timestamp"),
            "model_path": training_results.get("model_path"),
            "results_path": str(results_path),
            "triggered_by_external": True,
        }

        # Add performance metrics if available
        if "performance_summary" in training_results:
            perf = training_results["performance_summary"]
            training_summary.update(
                {
                    "test_rmse": perf.get("test_rmse"),
                    "test_mape": perf.get("test_mape"),
                    "test_r2": perf.get("test_r2"),
                    "overfitting_risk": perf.get("overfitting_risk"),
                }
            )

        # Create training results artifact
        create_markdown_artifact(
            key="training-results",
            markdown=f"""
# Model Training Results

## Training Status: {
                'SUCCESS' if training_summary[
                    'training_status'] == 'SUCCESS' else 'FAILED'
            }

### Model Information
- **Model Type:** {training_summary['model_type']}
- **Model Path:** {training_summary.get('model_path', 'N/A')}

### Performance Metrics
- **Test RMSE:** {training_summary.get('test_rmse', 'N/A')}
- **Test MAPE:** {training_summary.get('test_mape', 'N/A')}
- **Test R²:** {training_summary.get('test_r2', 'N/A')}
- **Overfitting Risk:** {training_summary.get('overfitting_risk', 'N/A')}

### Training Configuration
- **Feature Store Used:** Yes
- **Validation Enabled:** Yes
- **Model Saved:** Yes
- **Triggered By:** Data Pipeline
""",
            description="Model training execution results and performance metrics"
        )

        logger.info(f"Model training completed: {training_summary}")
        return training_summary

    except Exception as e:
        logger.error(f"Failed to execute model training: {e}")
        raise


@task
def validate_trained_model(
    training_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate trained model performance."""
    logger = get_run_logger()

    try:
        # Load training results
        results_path = str(training_summary.get("results_path"))
        path = Path(results_path)

        if not path or not Path(path).exists():
            raise ValueError("Training results file not found")

        with path.open("r") as f:
            training_results = json.load(f)

        # Perform validation checks
        validation_checks = {
            "model_saved": bool(
                training_summary.get("model_path") and Path(
                    training_summary["model_path"]).exists()
            ),
            "training_successful": training_summary.get(
                "training_status") == "SUCCESS",
            "performance_acceptable": False,
            "validation_passed": False,
            "external_trigger_success": training_summary.get(
                "triggered_by_external", False
            ),
        }

        # Check performance metrics
        if "performance_summary" in training_results:
            perf = training_results["performance_summary"]
            test_r2 = perf.get("test_r2", 0)
            test_rmse = perf.get("test_rmse", float("inf"))

            # Define acceptance criteria
            validation_checks["performance_acceptable"] = (
                test_r2 > 0.7 and test_rmse < 10000
            )

        # Check validation results from training
        if "validation_results" in training_results:
            val_results = training_results["validation_results"]
            validation_checks["validation_passed"] = val_results.get(
                "overall_passed", False
            )

        # Overall model acceptance
        model_accepted = all(
            [
                validation_checks["model_saved"],
                validation_checks["training_successful"],
                validation_checks["performance_acceptable"],
            ]
        )

        validation_summary = {
            "model_accepted": model_accepted,
            "validation_checks": validation_checks,
            "validation_timestamp": datetime.now().isoformat(),
            "model_path": training_summary.get("model_path"),
            "recommendations": training_results.get("recommendations", []),
            "trigger_info": training_results.get("trigger_info", {}),
        }

        status_message = (
            "Ready for Deployment" if model_accepted else 'Not Ready - Fix Issues Above'
        )

        # Create model validation artifact
        create_markdown_artifact(
            key="model-validation",
            markdown=f"""
# Model Validation Report

## Overall Status: {'ACCEPTED' if model_accepted else 'REJECTED'}

### Validation Checks
- **Model Saved:** {'YES' if validation_checks['model_saved'] else 'NO'}
- **Training Successful:** {'YES' if validation_checks['training_successful'] else 'NO'}
- **Performance Acceptable:** {
                'YES' if validation_checks[
                    'performance_acceptable'] else 'NO'
            }
- **Validation Passed:** {
                'YES' if validation_checks['validation_passed'] else 'NO'
            }

### Model Information
- **Model Path:** {validation_summary.get('model_path', 'N/A')}
- **Validation Time:** {validation_summary['validation_timestamp']}

### Recommendations
{chr(10).join([f"- {rec}" for rec in validation_summary.get('recommendations', [])])
                or '- No specific recommendations'}

### Deployment Readiness
**Status:** {status_message}
""",
            description="Model validation report"
        )

        if model_accepted:
            logger.info(
                "Model validation passed - model ready for deployment")
        else:
            logger.warning(f"Model validation failed: {validation_checks}")

        return validation_summary

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        raise


@task
def generate_training_report(
    training_summary: Dict[str, Any],
    validation_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate training report."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Load full training results
        results_path = training_summary.get("results_path")

        if results_path and Path(results_path).exists():
            with Path(results_path).open("r") as f:
                full_results = json.load(f)
        else:
            full_results = {}

        # Create comprehensive report
        report = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "flow_run_id": str(flow_run.get_id()),
                "flow_run_name": str(flow_run.get_name()),
                "triggered_by": "used_car_price_data_pipeline",
                "trigger_type": "automatic",
                "feature_store_type": training_summary.get(
                    "feature_store_type", "Unknown"
                )
            },
            "training_summary": training_summary,
            "validation_summary": validation_summary,
            "full_training_results": full_results,
            "model_readiness": {
                "ready_for_deployment": validation_summary.get(
                    "model_accepted", False
                ),
                "deployment_blockers": [],
            },
            "pipeline_integration": {
                "data_pipeline_triggered": True,
                "automatic_trigger_success": True,
                "trigger_info": full_results.get("trigger_info", {}),
            },
        }

        # Identify deployment blockers
        if not validation_summary.get("model_accepted", False):
            blockers = []
            checks = validation_summary.get("validation_checks", {})

            if not checks.get("model_saved", False):
                blockers.append("Model not saved properly")
            if not checks.get("training_successful", False):
                blockers.append("Training did not complete successfully")
            if not checks.get("performance_acceptable", False):
                blockers.append(
                    "Model performance below acceptance criteria")

            report["model_readiness"]["deployment_blockers"] = blockers

        # Save report
        logs_dir = settings.logs_dir
        report_path = logs_dir / "training_report.json"
        # report_path = Path("/tmp/training_report.json")
        with report_path.open("w") as f:
            json.dump(report, f, indent=2, default=str)

        report_summary = {
            "report_path": str(report_path),
            "model_ready": report[
                "model_readiness"]["ready_for_deployment"],
            "blockers_count": len(
                report["model_readiness"]["deployment_blockers"]
            ),
            "report_timestamp": report["report_metadata"]["generation_timestamp"],
            "external_trigger_success": True,
            "feature_store_type": report["report_metadata"]["feature_store_type"],
        }

        # Create training report artifact
        create_markdown_artifact(
            key="comprehensive-training-report",
            markdown=f"""
# Comprehensive Training Report

## Report Summary
- **Generated:** {report['report_metadata']['generation_timestamp']}
- **Flow Run:** {report['report_metadata']['flow_run_name']}
- **Triggered By:** {report['report_metadata']['triggered_by']}
- **Feature Store:** {report['report_metadata']['feature_store_type']}

## Model Readiness for Deployment
- **Status:** {'Ready' if report_summary['model_ready'] else 'Not Ready'}
- **Blockers:** {report_summary['blockers_count']}

## Training Performance
- **Status:** {training_summary.get('training_status', 'Unknown')}
- **Model Type:** {training_summary.get('model_type', 'Unknown')}
- **Test RMSE:** {training_summary.get('test_rmse', 'N/A')}
- **Test R²:** {training_summary.get('test_r2', 'N/A')}

## Deployment Blockers
{chr(10).join([f"- {blocker}" for blocker in report['model_readiness']
               ['deployment_blockers']]) or "- None"}

## Next Steps
{'- Proceed to deployment pipeline' if report_summary['model_ready']
                else '- Fix identified issues before deployment'}
- Monitor model performance
- Set up automated retraining triggers
""",
            description="Comprehensive training report"
        )
        logger.info(f"Training report generated: {report_summary}")
        return report_summary

    except Exception as e:
        logger.error(f"Failed to generate training report: {e}")
        raise


@task
def cleanup_training_artifacts() -> Dict[str, Any]:
    """Cleanup temporary training artifacts."""
    logger = get_run_logger()

    try:
        temp_path = "/tmp/training_results.json"
        cleaned_files = []
        path = Path(temp_path)
        if path.exists:
            path.unlink()
            cleaned_files.append(path)
            logger.info(f"Cleaned up: {path}")

        cleanup_summary = {
            "cleanup_timestamp": datetime.now().isoformat(),
            "cleaned_files": cleaned_files,
            "external_trigger_cleanup": True,
        }

        logger.info(f"Cleanup completed: {cleanup_summary}")
        return cleanup_summary

    except Exception as e:
        logger.warning(f"Cleanup completed with warnings: {e}")
        return {"cleanup_status": "completed_with_warnings", "error": str(e)}


@flow(
    name="training-pipeline",
    description="Used car price prediction model training pipeline",
    flow_run_name=f"training-pipeline-{
        datetime.now().strftime('%Y%m%d-%H%M%S')
    }",
)
def training_pipeline() -> Dict[str, Any]:
    """Model training pipeline flow triggered by data pipeline completion."""
    logger = get_run_logger()
    try:
        # Check feature store readiness
        feature_store_result = check_feature_store_readiness()

        # Validate training prerequisites
        prerequisites_result = validate_training_prerequisites(
            feature_store_meta=feature_store_result
        )

        # Prepare training environment
        environment_result = prepare_training_environment(
            validation_results=prerequisites_result
        )

        # Execute model training
        training_result = execute_model_training(
            _environment_info=environment_result
        )

        # Validate trained model
        validation_result = validate_trained_model(
            training_summary=training_result
        )

        # Generate training report
        report_result = generate_training_report(
            training_summary=training_result, validation_summary=validation_result
        )
        logger.info(
            "\nTraining Report Path: ", {
                report_result.get('report_path')
            })

        # Update training completion timestamp for monitoring
        Variable.set(
            "last_training_date",
            datetime.now().isoformat(),
            overwrite=True
        )

        # Clean up artifacts
        cleanup_result = cleanup_training_artifacts()

        # Create final training state
        training_state = {
            "status": training_result.get("training_status", "FAILED"),
            "feature_store_type": training_result.get(
                "feature_store_type", "Unknown"
            ),
            "completion_time": datetime.now().isoformat(),
            "model_accepted": validation_result.get("model_accepted", False),
            "model_path": training_result.get("model_path"),
            "performance_metrics": {
                "test_rmse": training_result.get("test_rmse"),
                "test_r2": training_result.get("test_r2"),
                "test_mape": training_result.get("test_mape"),
            },
            "deployment_ready": validation_result.get("model_accepted", False),
            "cleanup_status": cleanup_result.get("cleanup_status", "completed"),
        }

        # Store state in Prefect variable
        Variable.set(
            "training_pipeline_state",
            json.dumps(
                training_state, default=str
            ),
            overwrite=True
        )

        return training_state

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")

        # Clean up on failure
        cleanup_training_artifacts()

        # Store failure state
        failure_state = {
            "status": "FAILED",
            "completion_time": datetime.now().isoformat(),
            "error": str(e),
            "model_accepted": False,
            "deployment_ready": False,
        }
        Variable.set(
            "training_pipeline_state",
            json.dumps(
                failure_state, default=str),
            overwrite=True
        )
        raise


if __name__ == "__main__":
    training_pipeline()
