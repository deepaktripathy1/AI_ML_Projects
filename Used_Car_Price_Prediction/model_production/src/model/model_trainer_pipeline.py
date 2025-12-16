"""Model training pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import joblib
import mlflow
import mlflow.sklearn as mlflow_sklearn
import mlflow.shap as mlflow_shap
import shap
import shap.maskers as shap_maskers
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.feature_store.feature_pipeline import FeaturePipeline
from src.feature_store.feature_store_client import FeatureStoreClient
from src.model.linear_regression import LinearRegressionModel
from src.utils.mlflow_utils import reset_mlflow_state
from src.utils.path_utils import to_serializable_path


logger = LoggingConfig.get_logger(__name__)


class ModelTrainer:
    """Model training pipeline."""

    def __init__(self):
        """Initialize model trainer."""

        self.feature_pipeline = FeaturePipeline()
        self.feature_client = FeatureStoreClient()
        self.model = None
        self.training_results = {}
        self.validation_results = {}
        self.shap_explainer = None

        logger.info("Model trainer initialized")

    def train_model(
        self,
        use_feature_store: bool = True,
        validation_enabled: bool = True,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """Execute complete model training pipeline.

        Args:
            use_feature_store: Whether to load data from feature store
            validation_enabled: Whether to perform model validation
            save_model: Whether to save trained model
            run_name: Name of the MLflow run

        Reforms:
            Dict containing training results
        """
        reset_mlflow_state()
        settings = get_settings()

        logger.info("Starting model training pipeline..")

        with mlflow.start_run(
            run_name=f"AutoTraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:

            mlflow.set_tags({
                "pipeline": "training",
                "component": "LinearRegressionModel",
                "framework": "scikit-learn",
                "dataset_source": "Hopsworks Feature Store",
                "data_version": "v1",
                "env": "production",
                "triggered_by": "Prefect Flow",
                "owner": "Deepak Tripathy",
                "timestamp": datetime.now().isoformat()
            })

            try:
                # Load training data
                logger.info("Load training data")
                X_train, X_test, y_train, y_test = self._load_training_data(
                    use_feature_store
                )

                # Log data information
                self._log_data_info(X_train, X_test, y_train, y_test)

                # Initialize and train model
                logger.info("Training model")
                self.model = LinearRegressionModel(normalize=True)

                training_metrics = self.model.train(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                )

                logger.info("Logging training metrics to MLflow")

                # Log paramters
                mlflow.log_param("model_type", "Linear Regression")
                mlflow.log_param("normalize_features", self.model.normalize)
                mlflow.log_param("n_features", len(self.model.feature_names))
                mlflow.log_param("n_training_samples", len(X_train))
                mlflow.log_param("random_state", settings.RANDOM_STATE)
                mlflow.log_param("training_status", "SUCCESS")

                # Log feature names
                mlflow.log_param("feature_names", ",".join(
                    self.model.feature_names
                ))

                # Log metrics
                for key, value in training_metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                    elif isinstance(value, dict):
                        # flatten cross validation metrics
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (int, float)):
                                mlflow.log_metric(f"{key}_{subkey}", subvalue)

                # Log model coefficients as metrics
                for _i, (feature, coef) in enumerate(zip(
                    self.model.feature_names,
                    self.model.model.coef_,
                    strict=False
                )):
                    mlflow.log_metric(f"coef_{feature}", coef)

                intercept = self.model.model.intercept_
                if isinstance(intercept, np.ndarray):
                    intercept_val = float(intercept.item())
                else:
                    intercept_val = float(intercept)

                mlflow.log_metric("intercept", intercept_val)

                # Generate SHAP explanations
                logger.info("Generating SHAP model explanations")
                shap_results = self._generate_shap_explanations(
                    X_train, X_test
                )

                # Validate model performance
                if validation_enabled:
                    logger.info("Validating model performance")
                    validation_results = self._validate_model_performance(
                        X_train, y_train, X_test, y_test
                    )
                    self.validation_results = validation_results

                # Generate training report
                logger.info("Generating training report")
                training_results = self._generate_training_report(
                    training_metrics,
                    self.validation_results if self.validation_results else {},
                )
                self.training_results = training_results
                self.training_results["shap_results"] = shap_results

                # Infer model signature
                X_train_processed, _ = self.model._prepare_data(
                    X_train, X_test
                )
                y_pred_sample = self.model.model.predict(X_train_processed[:5])
                signature = infer_signature(X_train_processed, y_pred_sample)

                # Input example
                input_example = X_train.iloc[:2]

                # Log model
                mlflow_sklearn.log_model(
                    sk_model=self.model.model,
                    name="model",
                    registered_model_name="used_car_price_pred_model",
                    signature=signature,
                    input_example=input_example
                )

                logger.info("Model logged to MLflow")

                # Register the model
                try:
                    logger.info(
                        "Registering model in MLflow Model Registry...")

                    # Retrieve the latest logged model URI
                    model_uri = f"runs:/{run.info.run_id}/model"

                    # Register model
                    registered_model = mlflow.register_model(
                        model_uri=model_uri,
                        name="used_car_price_pred_model"
                    )

                    client = MlflowClient()

                    # Tag the registered model
                    client.set_model_version_tag(
                        name="used_car_price_pred_model",
                        version=registered_model.version,
                        key="training_pipeline",
                        value="Prefect AutoPipeline"
                    )

                    client.set_model_version_tag(
                        name="used_car_price_pred_model",
                        version=registered_model.version,
                        key="data_source",
                        value="Hopsworks v1"
                    )

                    client.set_model_version_tag(
                        name="used_car_price_pred_model",
                        version=registered_model.version,
                        key="run_id",
                        value=run.info.run_id
                    )

                    client.set_model_version_tag(
                        name="used_car_price_pred_model",
                        version=registered_model.version,
                        key="experiment_name",
                        value=settings.MLFLOW_EXPERIMENT_NAME
                    )

                    logger.info(
                        f"Model registered successfully: version {
                            registered_model.version
                        }"
                    )

                    # Tag alias to the model
                    client.set_registered_model_alias(
                        name="used_car_price_pred_model",
                        alias="production",
                        version=registered_model.version
                    )

                    logger.info(f"Model alias set to 'production'")

                except Exception as e:
                    logger.warning(f"Model registration failed: {e}")

                # Log Artifacts

                with tempfile.TemporaryDirectory() as tmp_dir:
                    temp_dir = Path(tmp_dir)

                    # Log scaler if used
                    if hasattr(
                        self.model,
                        "scaler"
                    ) and self.model.scaler is not None:
                        scaler_temp_path = temp_dir / "scaler.pkl"
                        joblib.dump(self.model.scaler, scaler_temp_path)
                        mlflow.log_artifact(
                            local_path=str(scaler_temp_path),
                            artifact_path="scaler"
                        )

                        logger.info("Scaler logged to MLflow")

                    # Log features
                    if hasattr(
                        self.model,
                        "feature_names"
                    ) and self.model.feature_names:
                        features_temp_path = temp_dir / "features.pkl"
                        joblib.dump(
                            self.model.feature_names,
                            features_temp_path
                        )
                        mlflow.log_artifact(
                            local_path=str(features_temp_path),
                            artifact_path="features"
                        )

                        logger.info("Feature names logged to MLflow")

                        # Log model metadata to MLflow
                        metadata_temp_path = temp_dir / "metadata.pkl"

                        metadata_dict = {
                            "model_source": "mlflow",
                            "model_type": "LinearRegression",
                            "mlflow_run_id": run.info.run_id,
                            "model_uri": f"runs:/{run.info.run_id}/model",
                            "scaler_uri": mlflow.get_artifact_uri(
                                artifact_path="scaler/scaler.pkl"
                            ),
                            "features_uri": mlflow.get_artifact_uri(
                                artifact_path="features/features.pkl"
                            ),
                            "shap_explainer_path": mlflow.get_artifact_uri(
                                artifact_path="shap_explanation/shap_explainer.pkl"
                            ),
                            "training_timestamp": (
                                datetime.now().isoformat()
                            ),
                            "feature_names": self.model.feature_names,
                            "n_features": len(
                                self.model.feature_names
                            ),
                            "n_training_samples": len(X_train),
                            "training_results": (
                                self.training_results
                            ),
                        }

                        joblib.dump(metadata_dict, metadata_temp_path)

                        mlflow.log_artifact(
                            local_path=str(metadata_temp_path),
                            artifact_path="metadata"
                        )

                        logger.info("Model metadata logged to MLflow")

                # Save model locally if requested
                if save_model:
                    logger.info("Saving model")
                    model_path = self._save_trained_model(run.info.run_id)
                    training_results["model_path"] = str(model_path)

                # Log training results to MLflow
                self._log_training_results(training_results)

                logger.info("Model training completed successfully")
                return training_results

            except Exception as e:
                logger.error(f"Model training failed: {e}")
                mlflow.log_param("training_status", "FAILED")
                mlflow.log_param("error message", str(e))
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
                raise

            finally:
                if mlflow.active_run():
                    mlflow.end_run(status="FINISHED")

    def _generate_shap_explanations(
            self,
            X_train: pd.DataFrame,
            X_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate SHAP model explanations and log to MLflow.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Dict containing SHAP analysis results
        """
        try:
            # Prepare data for SHAP
            if self.model is not None:
                if self.model.normalize and self.model.scaler is not None:
                    X_train_processed = pd.DataFrame(
                        data=self.model.scaler.fit_transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                    X_test_processed = pd.DataFrame(
                        data=self.model.scaler.transform(X_test),
                        columns=X_test.columns,
                        index=X_test.index
                    )
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test

                # Create SHAP explainer for linear model
                self.shap_explainer = shap.LinearExplainer(
                    model=self.model.model,
                    masker=X_train_processed,
                    feature_names=list(X_train_processed.columns)
                )

                # Log SHAP explainer
                mlflow_shap.log_explainer(
                    explainer=self.shap_explainer,
                    name="shap_explainer",
                    registered_model_name="used_car_model_explainer"
                )

                # Calculate SHAP values for test dataset
                shap_values = self.shap_explainer.shap_values(X_test_processed)

                # SHAP Explanation Object
                shap_explanation = shap.Explanation(
                    values=shap_values,
                    base_values=self.shap_explainer.expected_value,
                    data=X_test_processed.values,
                    feature_names=list(X_test_processed.columns)
                )

                # Log SHAP explanation
                mlflow_shap.log_explanation(
                    predict_function=self.model.predict,
                    features=X_test,
                    artifact_path="shap_explanation"
                )

                if shap_values is not None:

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_dir = Path(tmp_dir)

                        # Summary Plot
                        plt.figure(figsize=(10, 6))
                        shap.summary_plot(
                            shap_values=shap_values,
                            feature_names=list(X_test_processed.columns),
                            show=False
                        )
                        summary_path = temp_dir / "summary.png"
                        plt.savefig(summary_path, bbox_inches="tight", dpi=150)
                        plt.close()
                        # Log plot to MLflow
                        mlflow.log_artifact(
                            local_path=str(summary_path),
                            artifact_path="shap_plots"
                        )

                        # Global Feature Importance Plot
                        plt.figure(figsize=(10, 6))
                        shap_mean_abs = np.abs(shap_values).mean(axis=0)
                        shap_importance_sorted = pd.DataFrame({
                            "feature": list(X_test_processed.columns),
                            "importance": shap_mean_abs
                        }).sort_values("importance", ascending=True)

                        plt.barh(
                            shap_importance_sorted["feature"], shap_importance_sorted["importance"]
                        )
                        plt.xlabel("Mean |SHAP value|")
                        plt.title("Feature Importance (SHAP)")
                        plt.tight_layout()
                        bar_path = temp_dir / "global_feat_imp.png"
                        plt.savefig(bar_path, bbox_inches="tight", dpi=150)
                        plt.close()
                        # Log plot to MLflow
                        mlflow.log_artifact(
                            local_path=str(bar_path),
                            artifact_path="shap_plots"
                        )

                        # Waterfall Plot for first prediction
                        plt.figure(figsize=(10, 6))
                        shap.waterfall_plot(
                            shap_values=shap_explanation[0],
                            show=False
                        )
                        waterfall_path = temp_dir / "waterfall.png"
                        plt.savefig(
                            waterfall_path,
                            bbox_inches="tight",
                            dpi=150
                        )
                        plt.close()
                        # Log plot to MLflow
                        mlflow.log_artifact(
                            local_path=str(waterfall_path),
                            artifact_path="shap_plots"
                        )

                        # Heatmap
                        plt.figure(figsize=(10, 6))
                        shap.heatmap_plot(
                            shap_explanation,
                            feature_values=shap_explanation.abs.max(0),
                            show=False
                        )
                        heatmap_path = temp_dir / "shap_heatmap.png"
                        plt.savefig(heatmap_path, bbox_inches="tight", dpi=150)
                        plt.close()
                        # Log plot to MLflow
                        mlflow.log_artifact(
                            local_path=str(heatmap_path),
                            artifact_path="shap_plots"
                        )

                        # Store shap values in a Dataframe
                        shap_df = pd.DataFrame(
                            shap_values,
                            columns=X_test_processed.columns
                        )
                        shap_df["base_value"] = self.shap_explainer.expected_value
                        csv_path = temp_dir / "shap_values.csv"
                        shap_df.to_csv(csv_path, index=False)
                        mlflow.log_artifact(
                            str(csv_path),
                            artifact_path="shap_values"
                        )

                        logger.info("SHAP plots generated and logged")

                    # Calculate feature importance from SHAP
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    feature_importance_df = pd.DataFrame({
                        "feature": list(X_test_processed.columns),
                        "shap_importance": shap_importance
                    }).sort_values("shap_importance", ascending=False)

                    # Log top 10 feature importances as metrics
                    for _, row in feature_importance_df.head(10).iterrows():
                        mlflow.log_metric(
                            f"shap_importance_{row['feature']}", row[
                                "shap_importance"]
                        )

                    # Log summary metrics
                    mean_abs_shap = float(np.abs(shap_values).mean())
                    mlflow.log_metric("shap_mean_abs", mean_abs_shap)
                    mlflow.log_metric("shap_samples", len(X_test_processed))
                    mlflow.log_param("shap_explainer_type", "LinearExplainer")

                    shap_results = {
                        "explainer_type": "LinearExplainer",
                        "n_samples_explained": len(X_test_processed),
                        "top_features": feature_importance_df.head(5).to_dict(
                            "records"
                        ),
                        "mean_abs_shap": float(
                            np.abs(shap_values).mean()
                        ),
                    }

                    logger.info(
                        "SHAP explanations generated and logged successfully"
                    )

                    return shap_results

                else:
                    logger.info("No SHAP values found")

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return {"error": str(e)}

        return {"error": "Error generating SHAP explanations"}

    def _load_training_data(
        self,
        use_feature_store: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load training data from feature store or process raw data."""

        settings = get_settings()

        if use_feature_store:
            try:
                X_train, X_test, y_train, y_test = (
                    self.feature_client.get_training_data()
                )
                logger.info(
                    f"Loaded training data: train = {
                        len(X_train)}, test = {
                            len(X_test)}"
                )
                return X_train, X_test, y_train, y_test

            except Exception as e:
                logger.error(
                    f"Failed to load training data from feature store: {e}."
                    "Processing raw data instead."
                )
                raise

        # Process raw data through feature pipeline
        _ = self.feature_pipeline.run_complete_pipeline(
            create_feature_group=True,
            create_feature_view=True,
            feature_selection_method="rfecv",
        )

        # Get processed data and split
        final_data = (
            pd.read_pickle("/tmp/final_data.pkl")
            if Path("/tmp/final_data.pkl").exists()
            else None
        )

        if final_data is None:
            raise ValueError("Failed to process training data")

        # Prepare features and target
        X = final_data.drop(columns=[settings.TARGET_COLUMN])
        y = final_data[settings.TARGET_COLUMN]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.TEST_SIZE,
            stratify=None,
            random_state=settings.RANDOM_STATE,
        )
        logger.info(
            f"Processed and split training data: train = {
                len(X_train)}, test = {
                    len(X_test)}"
        )
        return X_train, X_test, y_train, y_test

    def _log_data_info(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Log data information to MLflow."""
        settings = get_settings()

        # Log data shapes
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("test_size", settings.TEST_SIZE)

        # Log feature names
        mlflow.log_param("feature_names", ",".join(X_train.columns.tolist()))

        # Log target statistics
        mlflow.log_metric("target_train_mean", y_train.mean())
        mlflow.log_metric("target_train_std", y_train.std())
        mlflow.log_metric("target_test_mean", y_test.mean())
        mlflow.log_metric("target_test_std", y_test.std())
        mlflow.log_metric("target_min", min(y_train.min(), y_test.min()))
        mlflow.log_metric("target_max", max(y_train.max(), y_test.max()))

    def _validate_model_performance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Validate model performance."""

        logger.info("Validating model performance")

        # Define performance thresholds
        performance_thresholds = {
            "max_rmse": 10000,  # Maximum acceptable RMSE
            "min_r2": 0.7,  # Minimum acceptable R²
            "max_mape": 15,  # Maximum acceptable MAPE (%)
            "max_cv_std": 1000,  # Maximum acceptable CV standard deviation
        }

        # Get model metrics
        if self.model is not None:
            metrics = self.model.training_metrics
        else:
            metrics = {}
            logger.warning(
                "Model has not been initialized. No metrics available.")

        # Validation checks
        validation_results = {
            "thresholds": performance_thresholds,
            "checks": {
                "rmse_check": {
                    "value": metrics["test_rmse"],
                    "threshold": performance_thresholds["max_rmse"],
                    "passed": metrics["test_rmse"] < performance_thresholds["max_rmse"],
                },
                "r2_check": {
                    "value": metrics["test_r2"],
                    "threshold": performance_thresholds["min_r2"],
                    "passed": metrics["test_r2"] > performance_thresholds["min_r2"],
                },
                "mape_check": {
                    "value": metrics["test_mape"],
                    "threshold": performance_thresholds["max_mape"],
                    "passed": metrics["test_mape"] < performance_thresholds["max_mape"],
                },
            },
        }
        # Add cross-validation check if available
        if "cv_rmse_std" in metrics:
            validation_results["checks"]["cv_stability_check"] = {
                "value": metrics["cv_rmse_std"],
                "threshold": performance_thresholds["max_cv_std"],
                "passed": metrics["cv_rmse_std"] < performance_thresholds["max_cv_std"],
            }

        # Overall validation status
        all_checks_passed = all(
            check["passed"] for check in validation_results["checks"].values()
        )
        validation_results["overall_passed"] = all_checks_passed

        # Additional diagnostics
        validation_results["diagnostics"] = self._run_model_diagnostics(
            X_train, y_train, X_test, y_test
        )

        # Log validation results
        for check_name, check_data in validation_results["checks"].items():
            mlflow.log_metric(
                f"validation_{check_name}_passed", int(check_data["passed"])
            )
            mlflow.log_metric(
                f"validation_{check_name}_value", check_data["value"])

        mlflow.log_param("validation_overall_passed", all_checks_passed)

        if all_checks_passed:
            logger.info("Model validation passed all checks")
        else:
            failed_checks = [
                name
                for name, check in validation_results["checks"].items()
                if not check["passed"]
            ]
            logger.warning(f"Model validation failed checks: {failed_checks}")

        return validation_results

    def _run_model_diagnostics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Run additional model diagnostics."""
        diagnostics = {}

        try:
            # Prediction Analysis
            if self.model is not None:
                y_train_pred = self.model.predict(X=X_train)
                y_test_pred = self.model.predict(X=X_test)

                # Residual analysis
                train_residuals = y_train - y_train_pred
                test_residuals = y_test - y_test_pred

                diagnostics["residual_analysis"] = {
                    "train_residual_mean": float(train_residuals.mean()),
                    "test_residual_mean": float(test_residuals.mean()),
                    "train_residual_skewness": train_residuals.skew()
                    if len(train_residuals) > 0
                    else 0,
                    "test_residual_skewness": test_residuals.skew()
                    if len(test_residuals) > 0
                    else 0,
                }

                # Feature importance analysis
                feature_importance = self.model.get_feature_importance()
                top_features = feature_importance.head(5)

                diagnostics["feature_importance"] = {
                    "top_5_features": top_features[
                        ["feature", "abs_coefficient"]
                    ].to_dict("records"),
                    "feature_count": len(feature_importance),
                    "max_coefficient": float(
                        feature_importance["abs_coefficient"].max()
                    ),
                    "min_coefficient": float(
                        feature_importance["abs_coefficient"].min()
                    ),
                }

                # Prediction range analysis
                diagnostics["prediction_analysis"] = {
                    "train_pred_min": float(y_train_pred.min()),
                    "train_pred_max": float(y_train_pred.max()),
                    "test_pred_min": float(y_test_pred.min()),
                    "test_pred_max": float(y_test_pred.max()),
                    "train_actual_range": float(y_train.max() - y_train.min()),
                    "test_actual_range": float(y_test.max() - y_test.min()),
                }
        except Exception as e:
            logger.error(f"Failed to run model diagnostics: {e}")
            diagnostics["error"] = {"error": str(e)}

        return diagnostics

    def _generate_training_report(
        self,
        training_metrics: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate training report."""
        report = {
            "training_timestamp": datetime.now().isoformat(),
            "model_type": "LinearRegression",
            "training_metrics": training_metrics,
            "validation_results": validation_results,
            "model_info": self.model.get_model_info() if self.model else {},
            "training_status": "SUCCESS",
        }

        # Add performance summary
        if training_metrics:
            report["performance_summary"] = {
                "test_rmse": training_metrics.get("test_rmse", 0),
                "test_r2": training_metrics.get("test_r2", 0),
                "test_mape": training_metrics.get("test_mape", 0),
                "cv_rmse_mean": training_metrics.get("cv_rmse_mean", 0),
                "overfitting_risk": self._assess_overfitting_risk(training_metrics),
            }

        # Add recommendations
        report["recommendations"] = self._generate_recommendations(
            training_metrics, validation_results
        )

        return report

    def _assess_overfitting_risk(self, metrics: Dict[str, Any]) -> str:
        """Assess overfitting risk based on training vs test performance."""
        if "train_rmse" not in metrics or "test_rmse" not in metrics:
            return "unknown"

        train_rmse = metrics["train_rmse"]
        test_rmse = metrics["test_rmse"]

        if test_rmse > train_rmse * 1.5:
            return "high"
        elif test_rmse > train_rmse * 1.2:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        training_metrics: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []

        if not training_metrics:
            return recommendations

        # Performance-based recommendations
        if training_metrics.get("test_r2", 0) < 0.8:
            recommendations.append(
                "Consider feature engineering or polynomial "
                "features to improve R² score"
            )

        if training_metrics.get("test_mape", 100) > 10:
            recommendations.append(
                "High MAPE indicates prediction errors - consider data quality review"
            )

        # Validation-based recommendations
        if validation_results and not validation_results.get(
            "overall_passed", True
        ):
            failed_checks = [
                name
                for name, check in validation_results.get(
                    "checks", {}
                ).items() if not check.get("passed", True)
            ]
            if failed_checks:
                recommendations.append(
                    f"Address validation failures in: {
                        ', '.join(failed_checks)}"
                )

        # Model-specific recommendations
        if self.model and hasattr(self.model, "model"):
            feature_importance = self.model.get_feature_importance()
            if len(feature_importance) > 20:
                recommendations.append(
                    "Consider additional feature selection to reduce model complexity"
                )

        # Cross-validation recommendations
        cv_std = training_metrics.get("cv_rmse_std", 0)
        if cv_std > 1000:
            recommendations.append(
                "High cross-validation variance - consider regularization or more data"
            )

        if not recommendations:
            recommendations.append(
                "Model performance looks good - ready for deployment"
            )

        return recommendations

    def _save_trained_model(self, run_id: str) -> Path:
        """Save trained model to disk."""
        settings = get_settings()

        if not self.model or not self.model.is_trained:
            raise ValueError("No trained model to save")

        # Get active MLflow run ID
        try:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else "local_run"
        except Exception:
            run_id = "local_run"

        # Create model directory
        model_dir = (
            settings.models_dir
            / f"linear_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model.pkl"
        self.model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        if hasattr(self.model, "scaler") and self.model.scaler is not None:
            scaler_path = model_dir / "scaler.pkl"
            joblib.dump(self.model.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        else:
            logger.warning("No scaler found")

        # Save features
        if hasattr(self.model, "feature_names") and self.model.feature_names:
            features_path = model_dir / "features.pkl"
            joblib.dump(self.model.feature_names, features_path)
            logger.info(f"Features saved to {features_path}")
        else:
            logger.warning("No feature names found to save.")

        # Save SHAP explainer
        if self.shap_explainer is not None:
            explainer_path = model_dir / "shap_explainer.pkl"
            joblib.dump(self.shap_explainer, explainer_path)
            logger.info(f"SHAP explainer saved to {explainer_path}")

        # Save metadata
        metadata_path = model_dir / "metadata.pkl"
        metadata_dict = {
            "model_source": "local",
            "mlflow_run_id": run_id,
            "model_type": "LinearRegression",
            "training_timestamp": datetime.now().isoformat(),
            "model_path": to_serializable_path(model_path),
            "scaler_path": to_serializable_path(
                model_dir / "scaler.pkl"
            ) if (model_dir / "scaler.pkl").exists() else None,
            "features_path": to_serializable_path(
                model_dir / "features.pkl"
            ) if (model_dir / "features.pkl").exists() else None,
            "shap_explainer_path": to_serializable_path(
                model_dir / "shap_explainer.pkl"
            ) if (model_dir / "shap_explainer.pkl").exists() else None,
            "training_results": self.training_results,
        }

        joblib.dump(metadata_dict, metadata_path)

        logger.info(f"Model artifacts saved to {model_dir}")

        # Update pointer file for model folder name
        try:
            pointer_file = settings.models_dir / "current.txt"
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            tmp_pointer = settings.models_dir / f".current_tmp_{ts}"

            # write the folder name
            tmp_pointer.write_text(model_dir.name, encoding="utf-8")

            # auto replace
            tmp_pointer.replace(pointer_file)
            logger.info(
                f"Updated pointer file: {pointer_file} -> {model_dir.name}"
            )
        except Exception as e:
            logger.info(f"Failed to update current.txt pointer: {e}")

        return model_path

    def _log_training_results(self, results: Dict[str, Any]) -> None:
        """Log comprehensive training results to MLflow."""

        # Log performance summary
        if "performance_summary" in results:
            for metric, value in results["performance_summary"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"summary_{metric}", value)
                else:
                    mlflow.log_param(f"summary_{metric}", str(value))

        # Log recommendations count
        if "recommendations" in results:
            mlflow.log_metric("recommendations_count",
                              len(results["recommendations"]))
            mlflow.log_param("recommendations", "; ".join(
                results["recommendations"]))

        # Log training status
        mlflow.log_param(
            "training_pipeline_status", results.get(
                "training_status", "UNKNOWN")
        )

        logger.info("Training results logged to MLflow")

    def evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        log_to_mlflow: bool = True
    ) -> Dict[str, Any]:
        """Evaluate trained model on test data.

        Args:
            X_test: Test features
            y_test: Test target
            log_to_mlflow: Whether to log results to MLflow

        Returns:
            Dict containing evaluation metrics
        """
        if not self.model or not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model performance")

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate comprehensive metrics

        evaluation_metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            "evaluation_timestamp": datetime.now().isoformat(),
            "n_samples": len(X_test),
        }

        # Add prediction statistics
        evaluation_metrics.update(
            {
                "pred_min": float(y_pred.min()),
                "pred_max": float(y_pred.max()),
                "pred_mean": float(y_pred.mean()),
                "pred_std": float(y_pred.std()),
                "actual_min": float(y_test.min()),
                "actual_max": float(y_test.max()),
                "actual_mean": float(y_test.mean()),
                "actual_std": float(y_test.std()),
            }
        )

        # Log to MLflow if requested
        if log_to_mlflow:
            for metric, value in evaluation_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{metric}", value)
                else:
                    mlflow.log_param(f"eval_{metric}", str(value))

        logger.info(
            f"Model evaluation completed - RMSE: {
                evaluation_metrics['rmse']:.2f}, R²: {
                    evaluation_metrics['r2']:.4f}"
        )

        return evaluation_metrics

    def get_training_results(self) -> Dict[str, Any]:
        """Get training results."""
        return self.training_results

    def get_model(self) -> LinearRegressionModel | None:
        """Get trained model."""
        return self.model

    def load_trained_model(self, model_path: str) -> LinearRegressionModel:
        """Load a previously trained model.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded LinearRegressionModel instance
        """
        self.model = LinearRegressionModel.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
