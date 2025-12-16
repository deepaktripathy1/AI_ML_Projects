import mlflow
import mlflow.tracking.fluent as fluent
import logging

logger = logging.getLogger(__name__)


def reset_mlflow_state():
    """Forcefully end and reset MLflow's internal state to prevent active run conflicts."""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
            logger.info("Closed an existing MLflow run.")
    except Exception as e:
        logger.warning(f"Error ending active MLflow run: {e}")

    logger.info("Reset MLflow internal state.")
