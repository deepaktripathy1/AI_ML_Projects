"""Complete feature pipeline."""

from datetime import datetime
from typing import Any, Dict, List, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.data_ingestion.data_loader import Dataloader
from src.data_preprocessing.data_cleaner import DataCleaner
from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.data_preprocessing.feature_selection import FeatureSelector
from src.feature_store.feature_store_client import FeatureStoreClient


logger = LoggingConfig().get_logger(__name__)


class FeaturePipeline:
    """Complete feature engineering and feature store pipeline."""

    def __init__(self) -> None:
        """Initialize feature pipeline."""
        settings = get_settings()

        # Initialize components
        self.data_loader = Dataloader()
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector(target_column="price_usd")
        self.feature_store_client = FeatureStoreClient()

        # Initialize MLFlow tracking with error handling
        self._setup_mlflow()

        # Pipeline results storage
        self.pipeline_results: Dict[str, Any] = {}

        logger.info("Feature pipeline initialized.")

    def _setup_mlflow(self) -> None:
        """Setup MLflow."""
        settings = get_settings()
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            logger.info(
                f"MLflow tracking URI set to: {settings.MLFLOW_TRACKING_URI}"
            )

            # Initialize client
            client = MlflowClient(
                tracking_uri=settings.MLFLOW_TRACKING_URI
            )

            experiment_name = settings.MLFLOW_EXPERIMENT_NAME

            # Try to get experiment by name
            try:
                experiment = client.get_experiment_by_name(experiment_name)

                if experiment is None:
                    # Experiment doesn't exist, create it
                    logger.info(f"Creating new experiment: {experiment_name}")
                    mlflow.set_experiment(experiment_name)

                elif experiment.lifecycle_stage == "deleted":
                    # Experiment is deleted, permanently delete it and create new
                    logger.warning(
                        f"Experiment '{experiment_name}' is in deleted state. "
                        "Removing and creating fresh experiment."
                    )
                    try:
                        client.delete_experiment(experiment.experiment_id)
                    except Exception as e:
                        logger.warning(
                            f"Could not permanently delete experiment: {e}")

                    # Create new experiment
                    mlflow.set_experiment(experiment_name)

                else:
                    # Experiment exists and is active
                    mlflow.set_experiment(experiment_name)
                    logger.info(
                        f"Using existing experiment: {experiment_name}")

            except MlflowException as e:
                if "RESOURCE_DOES_NOT_EXIST" in str(e):
                    logger.warning(
                        f"Experiment reference is corrupted. Creating fresh experiment: {e}"
                    )
                    mlflow.set_experiment(experiment_name)
                else:
                    raise

        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            logger.warning("Continuing without MLflow tracking")

    def run_complete_pipeline(
        self,
        create_feature_group: bool = True,
        create_feature_view: bool = True,
        feature_selection_method: str = "rfecv",
    ) -> Dict[str, Any]:
        """Run the complete feature pipeline.

        Args:
            create_feature_group: Whether to create a feature group in Hopsworks.
            create_feature_view: Whether to create a feature view in Hopsworks.
            feature_selection_method: Feature selection method to use.

        Returns:
            Dictionary containing pipeline results
        """
        logger.info("Running complete feature pipeline...")

        with mlflow.start_run(
            run_name=f"feature_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            try:
                # Load raw data
                logger.info("Loading raw data...")
                raw_data = self.data_loader.load_raw_data()
                mlflow.log_metric("raw_data_records", len(raw_data))

                # Data cleaning
                logger.info("Data cleaning...")
                cleaned_data = self.data_cleaner.clean_data(raw_data)
                mlflow.log_metric("cleaned_data_records", len(cleaned_data))

                # Feature engineering
                logger.info("Feature engineering...")
                engineered_data = self.feature_engineer.engineer_features(
                    cleaned_data)
                mlflow.log_metric("engineered_features", len(engineered_data))

                # Feature Selection
                logger.info("Feature selection...")
                final_data, selected_features = self.feature_selector.select_features(
                    engineered_data, method=feature_selection_method
                )
                mlflow.log_metric("selected_features", len(selected_features))
                mlflow.log_param("feature_selection_method",
                                 feature_selection_method)

                # Create feature group in feature store
                if create_feature_group:
                    logger.info(
                        f"Creating feature group in {
                            self.feature_store_client.get_client_type()}"
                    )
                    self._create_feature_group(final_data)

                # Create feature view in feature store
                if create_feature_view:
                    logger.info(
                        f"Creating feature view in {
                            self.feature_store_client.get_client_type()}"
                    )
                    self._create_feature_view()

                # Compile results
                logger.info("Step 7: Compiling pipeline results")
                self.pipeline_results = self._compile_pipeline_results(
                    raw_data,
                    cleaned_data,
                    engineered_data,
                    final_data,
                    selected_features,
                )

                # Log pipeline metrics to MLflow
                self._log_pipeline_metrics()

                logger.info("Feature pipeline completed successfully")
                return self.pipeline_results

            except Exception as e:
                logger.error(f"Feature pipeline failed: {e}")
                mlflow.log_param("pipeline_status", "FAILED")
                mlflow.log_param("error_message", str(e))
                raise

    def _create_feature_group(self, data: pd.DataFrame) -> None:
        """Create feature group in feature store."""
        try:
            # Add car_id if not present
            if "car_id" not in data.columns:
                data = data.copy()
                data["car_id"] = range(1, len(data) + 1)

            # Create feature group
            # Check if feature group already exists
            feature_group = self.feature_store_client.get_feature_group(
                name="used_car_features",
                version=1
            )
            if feature_group is not None:
                mlflow.log_param("feature_group_exists", True)
                logger.info(
                    f"Feature group already exists in {
                        self.feature_store_client.get_client_type()
                    }. Skipping creation"
                )
                return  # Skip creation

            self.feature_store_client.create_feature_group(
                df=data,
                name="used_car_features",
                version=1,
                description="Used car features for price prediction",
                primary_key=["car_id"],
            )

            mlflow.log_param("feature_group_created", True)
            logger.info(
                f"Feature group created successfully in {
                    self.feature_store_client.get_client_type()}"
            )

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning("Feature group already exists. Skipping.")
                mlflow.log_param("feature_group_exists", True)
                return
            logger.error(f"Failed to create feature group: {e}")
            mlflow.log_param("feature_group_created", False)
            raise

    def _create_feature_view(self) -> None:
        """Create feature view in feature store."""
        settings = get_settings()
        try:
            # Check if feature view already exists
            feature_view = self.feature_store_client.get_feature_view(
                name="used_car_price_features",
                version=1
            )
            if feature_view is not None:
                mlflow.log_param("feature_view_exists", True)
                logger.info(
                    f"Feature view already exists in {
                        self.feature_store_client.get_client_type()}"
                )
                return

            self.feature_store_client.create_feature_view(
                name="used_car_price_features",
                version=1,
                description="Feature view for used car price prediction",
                labels=[settings.TARGET_COLUMN],
            )

            mlflow.log_param("feature_view_created", True)
            logger.info(
                f"Feature view created successfully in {
                    self.feature_store_client.get_client_type()}"
            )

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.warning("Feature view already exists. Skipping.")
                mlflow.log_param("feature_view_exists", True)
                return
            logger.error(f"Failed to create feature view: {e}")
            mlflow.log_param("feature_view_created", False)
            raise

    def _compile_pipeline_results(
        self,
        raw_data: pd.DataFrame,
        cleaned_data: pd.DataFrame,
        engineered_data: pd.DataFrame,
        final_data: pd.DataFrame,
        selected_features: List[str],
    ) -> Dict[str, Any]:
        """Compile comprehensive pipeline results."""
        results = {
            "pipeline_timestamp": datetime.now().isoformat(),
            "feature_store_type": self.feature_store_client.get_client_type(),
            "data_shapes": {
                "raw": raw_data.shape,
                "cleaned": cleaned_data.shape,
                "engineered": engineered_data.shape,
                "final": final_data.shape,
            },
            "feature_counts": {
                "raw": len(raw_data.columns),
                "cleaned": len(cleaned_data.columns),
                "engineered": len(engineered_data.columns),
                "final": len(final_data.columns),
                "selected": len(selected_features),
            },
            "selected_features": selected_features,
            "data_quality": {
                "null_values": final_data.isnull().sum().sum(),
                "duplicate_rows": final_data.duplicated().sum(),
                "memory_usage_mb": final_data.memory_usage(deep=True).sum()
                / 1024
                / 1024,
            },
            "reports": {
                "cleaning": self.data_cleaner.get_cleaning_report(),
                "feature_engineering": self.feature_engineer.get_feature_report(),
                "feature_selection": self.feature_selector.get_selection_report(),
            },
        }

        # Calculate quality score
        results["quality_score"] = self._calculate_quality_score(results)

        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        score = 100.0

        # Deduct points for data quality issues
        null_values = results.get("data_quality", {}).get(
            "null_values", 0
        )
        duplicate_rows = results.get("data_quality", {}).get(
            "duplicate_rows", 0
        )
        total_rows = results.get("data_shapes", {}).get(
            "final", (0, 0)
        )[0]

        num_features = results.get("feature_counts", {}).get(
            "final", 1
        )

        if total_rows > 0 and num_features > 0:
            # Deduct for null values
            null_percentage = (null_values / (total_rows * num_features)) * 100
            score -= min(null_percentage * 2, 30)  # Max 30 points deduction

            # Deduct for duplicates
            duplicate_percentage = (duplicate_rows / total_rows) * 100
            # Max 20 points deduction
            score -= min(duplicate_percentage * 3, 20)

        return round(max(score, 0.0), 2)

    def _generate_recommendations(
            self,
            results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on pipeline results."""
        recommendations = []

        # Check data quality issues
        null_values = results.get("data_quality", {}).get(
            "null_values", 0
        )
        if null_values > 0:
            recommendations.append(
                f"Consider addressing {null_values} null values in the final dataset"
            )

        duplicate_rows = results.get("data_quality", {}).get(
            "duplicate_rows", 0
        )
        if duplicate_rows > 0:
            recommendations.append(
                f"Remove {duplicate_rows} duplicate rows to improve data quality"
            )

        # Check feature counts
        engineered_features = results.get("feature_counts", {}).get(
            "engineered", 0
        )
        final_features = results.get(
            "feature_counts", {}).get("final", 0)

        if engineered_features > 0 and final_features > 0:
            reduction_ratio = (
                engineered_features - final_features
            ) / engineered_features
            if reduction_ratio > 0.7:
                recommendations.append(
                    "High feature reduction ratio - "
                    "consider reviewing feature selection criteria"
                )
            elif reduction_ratio < 0.1:
                recommendations.append(
                    "Low feature reduction - consider more aggressive feature selection"
                )

        # Check quality score
        quality_score = results.get("quality_score", 0)
        if quality_score < 80:
            recommendations.append(
                "Overall data quality score is below 80% - "
                "review data cleaning processes"
            )

        return recommendations

    def _log_pipeline_metrics(self) -> None:
        """Log pipeline metrics to MLflow."""
        if not self.pipeline_results:
            return

        # Log data shape metrics
        for stage, shape in self.pipeline_results["data_shapes"].items():
            mlflow.log_metric(f"data_rows_{stage}", shape[0])
            mlflow.log_metric(f"data_cols_{stage}", shape[1])

        # Log feature count metrics
        for stage, count in self.pipeline_results["feature_counts"].items():
            mlflow.log_metric(f"feature_count_{stage}", count)

        # Log data quality metrics
        for metric, value in self.pipeline_results["data_quality"].items():
            mlflow.log_metric(f"quality_{metric}", float(value))

        # Log quality score
        mlflow.log_metric(
            "overall_quality_score", self.pipeline_results["quality_score"]
        )

        # Log selected features as parameter
        mlflow.log_param(
            "selected_features", ",".join(
                self.pipeline_results["selected_features"])
        )

    def get_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Get training data from feature store."""
        try:
            X_train, X_test, y_train, y_test = self.feature_store_client.get_training_data()
            logger.info(
                f"Retrieved training data from {self.feature_store_client.get_client_type()}: "
                f"train = {len(X_train)}, test = {len(X_test)}"
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Failed to retrieve training data: {e}")
            raise

    def get_pipeline_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        return self.pipeline_results
