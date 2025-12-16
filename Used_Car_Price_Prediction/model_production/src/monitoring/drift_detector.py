"""Data and model drift detection using Evidently AI."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Regression, Report
from evidently.generators.column import ColumnMetricGenerator
from evidently.metrics import ValueDrift, DriftedColumnsCount
from evidently.metrics.dataset_statistics import (
    DatasetMissingValueCount,
)
from evidently.presets.drift import DataDriftPreset
from evidently.presets.regression import RegressionPreset

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.utils.path_utils import from_serializable_path


logger = LoggingConfig.get_logger(__name__)


class DriftDetector:
    """Drift detection system for used car price prediction model.

    Monitors data drift, concept drift, and model performance degradation.
    """

    def __init__(
        self,
        reference_data_path: str | None = None,
        drift_threshold: float = 0.1,
        output_dir: str | None = None,
    ):
        """Initialize drift detector.

        Args:
            reference_data_path: Path to reference dataset
            drift_threshold: Threshold for drift detection
            output_dir: Directory to save reports
        """
        settings = get_settings()

        self.drift_threshold = drift_threshold
        self.reference_data = None
        self.drift_dir = settings.logs_dir / "drift_reports"
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring_dir = settings.logs_dir / "monitoring_reports"
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Define data definition for car price data
        self.data_definition = DataDefinition(
            # column types
            numerical_columns=[
                "mileage_kmpl",
                "engine_cc",
                "owner_count",
                "has_accident",
                "car_age",
                "service_history_encoded",
                "price_usd",
                "fuel_type_electric",
                "fuel_type_petrol",
                "brand_chevrolet",
                "brand_ford",
                "brand_honda",
                "brand_hyundai",
                "brand_kia",
                "brand_nissan",
                "brand_tesla",
                "brand_toyota",
                "brand_volkswagen",
                "transmission_manual",
                "color_blue",
                "color_gray",
                "color_red",
                "color_silver",
                "color_white",
                "insurance_valid_yes"
            ],
            # column roles
            regression=[
                Regression(target="price_usd", prediction="price_usd_pred"),
            ],
        )

        # Load reference data if provided
        if reference_data_path is not None:
            self.load_reference_data(reference_data_path)

        logger.info("Drift Detector initialized")

    def load_reference_data(self, data_path: str) -> None:
        """Load reference dataset for comparison."""
        try:
            if data_path.endswith(".csv"):
                self.reference_data = pd.read_csv(
                    from_serializable_path(data_path)
                )
            elif data_path.endswith(".pkl"):
                self.reference_data = pd.read_pickle(
                    from_serializable_path(data_path)
                )
            else:
                raise ValueError("Unsupported file format. Use CSV or PKL.")

            logger.info(f"Reference data loaded: {self.reference_data.shape}")

        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            raise

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Detect data drift using Evidently AI.

        Args:
            current_data: Current production data
            reference_data: Reference data (training data)

        Returns:
            Dict containing drift detection results
        """
        settings = get_settings()
        try:
            if self.reference_data is None:
                raise ValueError("Reference data not available")
            else:
                # Create Evidently reference dataset object
                reference_data = self.reference_data.copy()
                # Convert boolean to int
                reference_data = self._prepare_data_for_evidently(
                    reference_data
                )
                reference_dataset = Dataset.from_pandas(
                    reference_data,
                    data_definition=self.data_definition
                )

            # Create Evidently current dataset
            # Convert boolean to int
            current_data = self._prepare_data_for_evidently(
                current_data.copy()
            )
            current_dataset = Dataset.from_pandas(
                current_data,
                data_definition=self.data_definition
            )

            # Create drift report
            reference_columns = [
                "mileage_kmpl", "engine_cc", "owner_count", "car_age", "has_accident", "service_history_encoded", "fuel_type_electric", "fuel_type_petrol", "brand_chevrolet", "brand_ford", "brand_honda", "brand_hyundai", "brand_kia", "brand_nissan", "brand_tesla", "brand_toyota", "brand_volkswagen", "transmission_manual", "color_blue", "color_gray", "color_red", "color_silver", "color_white", "insurance_valid_yes", "price_usd"
            ]

            columns_to_check = list(
                reference_data.columns if reference_data is not None else reference_columns)

            report = Report(
                [
                    DataDriftPreset(),
                    DriftedColumnsCount(),
                    DatasetMissingValueCount(),
                    ColumnMetricGenerator(
                        ValueDrift, columns=columns_to_check),
                ]
            )

            # Run the report
            report_dict = report.run(
                reference_data=reference_dataset,
                current_data=current_dataset
            ).dict()

            # Save report as html
            drift_html_path = (self.drift_dir / f"data_drift_report_{
                datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            ).resolve()

            report.run(
                reference_data=reference_dataset,
                current_data=current_dataset
            ).save_html(filename=str(drift_html_path))

            # Parse drift results
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "dataset_drift_detected": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "drift_details": {},
                "data_quality_issues": [],
            }

            metrics = report_dict.get("metrics", [])

            # Check dataset drift
            for metric in metrics:
                if metric["metric_id"].startswith("DriftedColumnsCount"):
                    drift_info = metric["value"]
                    drift_results["drift_score"] = drift_info.get(
                        "share", 0.0
                    )
                    drift_results["dataset_drift_detected"] = drift_info.get(
                        "share", 0.0) > 0.3

            # Check individual feature drift
            for metric in metrics:
                if metric["metric_id"].startswith("ValueDrift(column="):
                    feature_name = metric["metric_id"].split(
                        "ValueDrift(column=")[1][:-1]
                    p_value = metric["value"]
                    drift_detected = float(p_value) < 0.05

                    drift_results["drift_details"][feature_name] = {
                        "drift_score": float(p_value),
                        "drift_detected": drift_detected,
                    }

                    if drift_detected:
                        drift_results["drifted_features"].append(feature_name)

            # Check data quality
            for metric in metrics:
                if metric["metric_id"].startswith("DatasetMissingValueCount"):
                    missing_values_info = metric["value"]
                    if missing_values_info.get("count", 0) > 0:
                        drift_results["data_quality_issues"].append(
                            f"Missing values detected in {
                                missing_values_info['count']
                            } columns"
                        )

            logger.info(
                f"Data drift analysis completed. Drift detected: {
                    drift_results['dataset_drift_detected']
                }"
            )

            return drift_results

        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            raise

    def detect_model_drift(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        current_predictions: np.ndarray | None = None,
        reference_predictions: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Detect model performance drift.

        Args:
            current_data: Current production data with targets
            reference_data: Reference data with targets
            current_predictions: Current model predictions
            reference_predictions: Reference model predictions

        Returns:
            Dict containing model drift results
        """
        try:
            # Prepare data
            if self.reference_data is None:
                raise ValueError("Reference data not available")
            else:
                reference_data_copy = self.reference_data.copy()
                # Convert boolean to int
                reference_data_copy = self._prepare_data_for_evidently(
                    reference_data_copy
                )
                # Add sample predictions for testing
                reference_data_copy["price_usd_pred"] = reference_data_copy[
                    "price_usd"
                ] * 0.98

            current_data_copy = current_data.copy()
            # Convert boolean to int
            current_data_copy = self._prepare_data_for_evidently(
                current_data_copy
            )
            # Add sample predictions for testing
            current_data_copy["price_usd_pred"] = current_data_copy[
                "price_usd"
            ] * 1.05

            # Add predictions to dataframes if provided
            if current_predictions is not None:
                current_data["predicted_price"] = current_predictions

            if reference_predictions is not None and reference_data is not None:
                reference_data["predicted_price"] = reference_predictions

            # Create datasets
            if reference_data is not None:
                reference_dataset = Dataset.from_pandas(
                    reference_data_copy,
                    data_definition=self.data_definition
                )

            current_dataset = Dataset.from_pandas(
                current_data_copy,
                data_definition=self.data_definition
            )

            # Create model performance report
            report = Report(
                metrics=[
                    RegressionPreset(),
                ],
                include_tests=True
            )

            # Run the report
            report_dict = report.run(
                reference_data=reference_dataset,
                current_data=current_dataset
            ).dict()

            # Save html
            model_drift_html_path = (self.drift_dir / f"model_drift_report_{
                datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            ).resolve()

            report.run(
                reference_data=reference_dataset,
                current_data=current_dataset
            ).save_html(filename=str(model_drift_html_path))

            # Run report only for reference metrics
            ref_report = Report(
                metrics=[
                    RegressionPreset()
                ]
            )
            reference_metrics = ref_report.run(
                reference_data=None,
                current_data=reference_dataset
            ).dict()

            # Save as html for viewing
            reference_model_drift_path = (self.drift_dir / f"reference_model_drift_report_{
                datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            ).resolve()

            ref_report.run(
                reference_data=None,
                current_data=reference_dataset
            ).save_html(filename=str(reference_model_drift_path))

            # Parse model performance results
            model_results = {
                "timestamp": datetime.now().isoformat(),
                "performance_degradation": False,
                "prediction_drift_detected": False,
                "current_metrics": {},
                "reference_metrics": {},
                "performance_change": {},
                "drift_details": {},
                "test_failures": []
            }

            # Extract regression metrics
            metrics = report_dict.get("metrics", [])
            tests = report_dict.get("tests", [])

            # Collect key performance metrics
            metric_values = {}
            for metric in metrics:
                metric_id = metric.get("metric_id", "")
                value = metric.get("value", None)
                if "R2Score" in metric_id:
                    metric_values["r2_score"] = value
                elif "MAE" in metric_id:
                    metric_values["mae"] = value["mean"] if isinstance(
                        value, dict) else value
                elif "RMSE" in metric_id:
                    metric_values["rmse"] = value
                elif "MAPE" in metric_id:
                    metric_values["mape"] = value["mean"] if isinstance(
                        value, dict) else value
                elif "MeanError" in metric_id:
                    metric_values["mean_error"] = value["mean"] if isinstance(
                        value, dict) else value

            model_results["current_metrics"] = metric_values

            # Collect reference data metrics for regression
            metrics_ref = reference_metrics.get("metrics", [])
            previous_values = {}
            for metric in metrics_ref:
                metric_id = metric.get("metric_id", "")
                value = metric.get("value", None)
                if "R2Score" in metric_id:
                    previous_values["r2_score"] = value
                elif "MAE" in metric_id:
                    previous_values["mae"] = value["mean"] if isinstance(
                        value, dict) else value
                elif "RMSE" in metric_id:
                    previous_values["rmse"] = value
                elif "MAPE" in metric_id:
                    previous_values["mape"] = value["mean"] if isinstance(
                        value, dict) else value
                elif "MeanError" in metric_id:
                    previous_values["mean_error"] = value["mean"] if isinstance(
                        value, dict) else value

            model_results["reference_metrics"] = previous_values

            # Calculate performance change
            if model_results["current_metrics"] and model_results["reference_metrics"]:
                current_r2 = model_results["current_metrics"].get(
                    "r2_score", 0
                )
                reference_r2 = model_results["reference_metrics"].get(
                    "r2_score", 0
                )
                current_mae = model_results["current_metrics"].get(
                    "mae", 0
                )
                reference_mae = model_results["reference_metrics"].get(
                    "mae", 0
                )

                model_results["performance_change"]["r2_change"] = (
                    current_r2 - reference_r2
                )
                model_results["performance_change"]["mae_change"] = (
                    current_mae - reference_mae
                )
                model_results["performance_degradation"] = (
                    current_r2 - reference_r2
                ) < -0.05

            # Parse Evidently "tests" results
            failed_tests = []
            for test in tests:
                status = str(test.get("status"))
                name = test.get("name", "Unnamed test")
                description = test.get("description", "")
                if "FAIL" in status:
                    failed_tests.append({
                        "name": name, "description": description
                    })

            # Store failed test information
            model_results["test_failures"] = failed_tests

            # Detect if overall performance drift based on test failures
            model_results["prediction_drift_detected"] = len(failed_tests) > 0

            # Store detailed test-level drift results
            model_results["drift_details"]["tests"] = [
                {
                    "metric_id": test["metric_config"]["metric_id"],
                    "status": str(test["status"]),
                    "name": test["name"],
                    "description": test["description"],
                }
                for test in tests
            ]

            logger.info(
                f"Model drift analysis complete. R2: {
                    model_results['performance_change']['r2_change']:.3f}, "
                f"Drift detected: {
                    model_results['prediction_drift_detected']}, "
                f"Failed tests: {len(failed_tests)}"
            )

            return model_results

        except Exception as e:
            logger.error(f"Model drift detection failed: {e}")
            raise

    def _prepare_data_for_evidently(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert boolean columns to integers.

        Args:
            data: Input dataframe

        Returns:
            Dataframe with boolean cols converted to int
        """
        data_copy = data.copy()

        # Boolean columns
        bool_cols = [
            "has_accident",
            "fuel_type_electric",
            "fuel_type_petrol",
            "brand_chevrolet",
            "brand_ford",
            "brand_honda",
            "brand_hyundai",
            "brand_kia",
            "brand_nissan",
            "brand_tesla",
            "brand_toyota",
            "brand_volkswagen",
            "transmission_manual",
            "color_blue",
            "color_gray",
            "color_red",
            "color_silver",
            "color_white",
            "insurance_valid_yes"
        ]

        # Convert boolean columns to int
        for col in bool_cols:
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].astype(int)

        return data_copy

    def create_dataset_with_predictions(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray | None = None,
        target_column: str | None = None,
    ) -> Dataset:
        """Helper method to create Dataset object with predictions.

        Args:
            data: Input dataframe
            predictions: Model predictions
            target_column: Name of target column

        Returns:
            Evidently Dataset object
        """
        data_copy = data.copy()

        if predictions is not None:
            data_copy["predicted_price"] = predictions

        # Update data definition if needed
        data_definition = self.data_definition
        if target_column and target_column != "price_usd":
            data_definition = DataDefinition(
                numerical_columns=self.data_definition.numerical_columns,
                categorical_columns=self.data_definition.categorical_columns,
                regression=[
                    Regression(target=target_column,
                               prediction="predicted_price")
                ],
            )

        return Dataset.from_pandas(data=data_copy, data_definition=data_definition)

    def generate_monitoring_summary(
        self,
        drift_results: dict[str, Any],
        model_results: dict[str, Any],
        quality_results: list
    ) -> dict[str, Any]:
        """Generate comprehensive monitoring summary.

        Args:
            drift_results: Data drift results
            model_results: Model drift results

        Returns:
            Dict containing monitoring summary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "HEALTHY",
            "alerts": [],
            "metrics": {
                "data_drift_score": drift_results.get("drift_score", 0.0),
                "model_performance_change": model_results.get(
                    "performance_change", {}
                ).get("r2_change", 0.0),
            },
            "quality_issues": drift_results.get("data_quality_issues", []),
            "recommendations": [],
        }

        # Check for alerts
        if drift_results.get("dataset_drift_detected", False):
            summary["overall_status"] = "WARNING"
            summary["alerts"].append("Data drift detected")
            summary["recommendations"].append(
                "Investigate feature distributions and consider model retraining"
            )

        if model_results.get("performance_degradation", False):
            summary["overall_status"] = "CRITICAL"
            summary["alerts"].append("Model performance degradation detected")
            summary["recommendations"].append(
                "Urgent: Model retraining required")

        if len(quality_results) > 0:
            if summary["overall_status"] == "HEALTHY":
                summary["overall_status"] = "WARNING"
            summary["alerts"].append(
                f"{len(quality_results)} data quality issues found"
            )
            summary["recommendations"].append(
                "Address data quality issues in pipeline")

        # Add detailed metrics
        summary["detailed_results"] = {
            "data_drift": drift_results,
            "model_drift": model_results,
            "data_quality": quality_results,
        }

        return summary

    def save_monitoring_results(
        self,
        results: dict[str, Any],
        filename: str | None = None
    ) -> str:
        """Save monitoring results to file."""
        if filename is None:
            filename = (
                f"monitoring_results{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = self.monitoring_dir / filename

        with filepath.open("w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Monitoring results saved to {filepath}")
        return str(filepath)

    def load_monitoring_results(self, filepath: Path) -> dict[str, Any]:
        """Load monitoring results from file."""
        with filepath.open("r") as f:
            results = json.load(f)

        return results

    def get_drift_history(self, days: int = 30) -> list[dict[str, Any]]:
        """Get drift detection history for the last N days."""
        history = []

        # Look for monitoring result files
        for file_path in self.monitoring_dir.glob("monitoring_results_*.json"):
            try:
                results = self.load_monitoring_results(filepath=file_path)
                timestamp = datetime.fromisoformat(results["timestamp"])

                if timestamp >= datetime.now() - timedelta(days=days):
                    history.append(results)

            except Exception as e:
                logger.warning(
                    f"Could not load monitoring file {file_path}: {e}")

        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])

        return history

    def run_comprehensive_monitoring(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        reference_predictions: np.ndarray | None = None,
        current_predictions: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Run complete monitoring pipeline: data drift, model drift, and quality tests.

        Args:
            current_data: Current production data
            current_predictions: Current model predictions
            reference_data: Reference data
            reference_predictions: Reference model predictions

        Returns:
            Complete monitoring summary
        """
        try:
            logger.info("Starting comprehensive monitoring...")

            # Run data drift detection
            drift_results = self.detect_data_drift(
                current_data=current_data,
                reference_data=reference_data
            )
            quality_results = drift_results.get("data_quality_issues", [])

            # Run model drift detection if predictions are available
            if (
                current_predictions is not None
                or "price_usd_pred" in current_data.columns
            ):
                model_results = self.detect_model_drift(
                    current_data=current_data,
                    reference_data=reference_data,
                    current_predictions=current_predictions,
                    reference_predictions=reference_predictions,
                )
            else:
                logger.warning(
                    "No predictions provided, skipping model drift detection"
                )
                model_results = {
                    "timestamp": datetime.now().isoformat(),
                    "performance_degradation": False,
                    "prediction_drift_detected": False,
                    "current_metrics": {},
                    "reference_metrics": {},
                    "performance_change": {},
                    "drift_details": {},
                }

            # Generate comprehensive summary
            summary = self.generate_monitoring_summary(
                drift_results,
                model_results,
                quality_results
            )

            # Save results
            results_path = self.save_monitoring_results(summary)
            summary["results_saved_to"] = results_path

            logger.info(
                f"Comprehensive monitoring completed. Status: {
                    summary['overall_status']
                }"
            )

            return summary

        except Exception as e:
            logger.error(f"Comprehensive monitoring failed: {e}")
            raise
