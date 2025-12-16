"""Linear Regression model implementation."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from config.logging_config import LoggingConfig


logger = LoggingConfig.get_logger(__name__)


class LinearRegressionModel:
    """Linear Regression model with MLflow tracking."""

    def __init__(self, normalize: bool = True) -> None:
        """Initialize Linear Regression Model.

        Args:
            normalize: Whether to normalize features using StandardScaler.
        """
        self.model = LinearRegression()
        self.scaler = StandardScaler() if normalize else None
        self.normalize = normalize
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
        self.model_metadata = {}

        logger.info(
            f"Linear Regression model initialized (normalize = {normalize})")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train the Linear Regression model.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target

        Returns:
            Dict containing training metrics
        """
        logger.info("Starting Linear Regression model training")

        try:
            # Store feature names
            self.feature_names = list(X_train.columns)

            # Prepare data
            X_train_processed, X_test_processed = self._prepare_data(
                X_train, X_test
            )

            # Train model
            logger.info("Training Linear Regression model...")
            self.model.fit(X_train_processed, y_train)

            # Make predictions
            y_train_pred = self.model.predict(X_train_processed)
            y_test_pred = self.model.predict(X_test_processed)

            # Calculate metrics
            metrics = self._calculate_metrics(
                y_train, y_train_pred, y_test, y_test_pred
            )
            self.training_metrics = metrics

            # Perform cross-validation
            cv_scores = self._perform_cross_validation(
                X_train_processed, y_train
            )
            metrics.update(cv_scores)

            # Store model metadata
            self._store_model_metadata(X_train)

            self.is_trained = True
            logger.info("Model training completed successfully")

            return metrics

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            mlflow.log_param("training_status", "FAILED")
            mlflow.log_param("error message", str(e))
            raise

    def _prepare_data(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training and testing."""

        if self.normalize and self.scaler is not None:
            logger.info("Normalizing features using Standard Scaler")
            X_train_processed = self.scaler.fit_transform(X_train)
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_train_processed = X_train.values
            X_test_processed = X_test.values

        return X_train_processed, X_test_processed

    def _calculate_metrics(
        self,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        y_test: pd.Series,
        y_test_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate model metrics."""

        metrics = {
            # Training metrics
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_mae": mean_absolute_error(y_train, y_train_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            # Testing metrics
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "test_mae": mean_absolute_error(y_test, y_test_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        }

        # Calculate percentage errors
        metrics["train_mape"] = (
            np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
        )

        metrics["test_mape"] = np.mean(
            np.abs((y_test - y_test_pred) / y_test)) * 100

        # Calculate residual statistics
        train_residuals = y_train - y_train_pred
        test_residuals = y_test - y_test_pred

        metrics["train_residual_std"] = np.std(train_residuals)
        metrics["test_residual_std"] = np.std(test_residuals)

        logger.info(
            f"Model metrics - Test RMSE: {metrics['test_rmse']:.2f}",
            "Test R²: {metrics['test_r2']:.4f}",
        )
        return metrics

    def _perform_cross_validation(
        self,
        X: np.ndarray,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation and return scores."""

        logger.info(f"Performing {cv}-fold cross validation")

        cv_scores = cross_val_score(
            estimator=self.model,
            X=X,
            y=y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        cv_results = {
            "cv_mean_rmse": np.sqrt(-cv_scores.mean()),
            "cv_std_rmse": np.sqrt(cv_scores.std()),
            "cv_scores": cv_scores.tolist(),
        }

        logger.info(
            f"Cross-validation RMSE: {
                cv_results['cv_mean_rmse']:.2f} +- {
                    cv_results['cv_std_rmse']:.2f}"
        )
        return cv_results

    def _store_model_metadata(self, X_train: pd.DataFrame) -> None:
        """Store model metadata for deployment."""

        self.model_metadata = {
            "model_type": "LinearRegression",
            "training_timestamp": pd.Timestamp.now().isoformat(),
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "n_training_samples": len(X_train),
            "training_metrics": self.training_metrics,
        }

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Validate features
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected: {
                    self.feature_names}, Got: {
                        list(X.columns)}"
            )

        # Prepare data
        if self.normalize and self.scaler is not None:
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values

        # Make predictions
        predictions = self.model.predict(X_processed)
        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on coefficients."""
        if not self.is_trained:
            raise ValueError("Model must be trained")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_,
                "abs_coefficient": abs(self.model.coef_),
            }
        ).sort_values(by="abs_coefficient", ascending=False)

        return importance_df

    def save_model(self, model_path: str) -> None:
        """Save model to disk.

        Args:
            model_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "normalize": self.normalize,
            "metadata": self.model_metadata,
            "training_metrics": self.training_metrics,
        }

        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load_model(cls, model_path: str) -> LinearRegressionModel:
        """Load model from disk.

        Args:
            model_path: Path to load model from

        Returns:
            Loaded LinearRegressionModel instance
        """
        model_data = joblib.load(model_path)

        # Create instance
        instance = cls(normalize=model_data["normalize"])

        # Restore model state
        instance.model = model_data["model"]
        instance.scaler = model_data["scaler"]
        instance.feature_names = model_data["feature_names"]
        instance.model_metadata = model_data["metadata"]
        instance.training_metrics = model_data["training_metrics"]
        instance.is_trained = True

        logger.info(f"Model loaded from {model_path}")
        return instance

    @classmethod
    def from_mlflow(
        cls,
        model: Any,
        scaler: Optional[StandardScaler] = None,
        features: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LinearRegressionModel:
        """Reconstruct a LinearRegressionModel from MLflow."""

        # Create instance
        instance = cls(normalize=False)
        instance.model = model
        instance.scaler = scaler
        instance.feature_names = features or []
        instance.model_metadata = metadata
        instance.is_trained = True

        logger.info(f"Model loaded from MLflow")
        return instance

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "model_type": "LinearRegression",
            "normalize_features": self.normalize,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "metadata": self.model_metadata,
        }
