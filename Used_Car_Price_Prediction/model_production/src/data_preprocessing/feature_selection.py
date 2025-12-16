"""Feature selection using RFECV."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from config.config import get_settings
from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)


class FeatureSelector:
    """Feature selection pipeline using multiple methods."""

    def __init__(self, target_column: str):
        """Initialize feature selector.

        Args:
            target_column: Name of the target column
        """
        settings = get_settings()
        self.target_column = target_column or settings.TARGET_COLUMN
        self.selection_report: dict[str, Any] = {
            "selection_steps": []
        }
        self.selected_features: list[str] = []
        self.feature_rankings: dict[str, int] = {}
        logger.info("Feature selector initialized")

    def select_features(
        self, df: pd.DataFrame, method: str = "rfecv", cv_folds: int = 5
    ) -> tuple[pd.DataFrame, list[str]]:
        """Select optimal features using specified method.

        Args:
            df: DataFrame with engineered features
            method: Feature selection method ("rfecv", "correlation", "combined")
            cv_folds: Number of cross-validation folds

        Returns:
            Tuple of (selected_features_df, selected features)
        """
        logger.info(f"Starting feature selection using method: {method}")

        # Prepare data
        X, y = self._prepare_data(df)
        initial_features = list(X.columns)

        # Initialize selection report
        self.selection_report = {
            "method": method,
            "initial_features_count": len(initial_features),
            "initial_features": initial_features,
            "selection_steps": [],
        }

        if method == "rfecv":
            selected_features = self._rfecv_selection(X, y, cv_folds)
        elif method == "correlation":
            selected_features = self._correlation_selection(X, y)
        elif method == "combined":
            selected_features = self._combined_selection(X, y, cv_folds)
        else:
            raise ValueError(f"Unsupported feature selection method: {method}")

        # Create final dataset
        df_selected = df[selected_features + [self.target_column]].copy()

        # Finalize selection report
        self.selected_features = selected_features
        self.selection_report["final_features_count"] = len(selected_features)
        self.selection_report["final_features"] = selected_features
        self.selection_report["features_removed"] = len(initial_features) - len(
            selected_features
        )
        self.selection_report["reduction_percentage"] = (
            (len(initial_features) - len(selected_features))
            / len(initial_features)
            * 100
        )

        logger.info(
            f"Feature selection completed. Features: {
                len(initial_features)} -> {
                    len(selected_features)}"
        )
        return df_selected, selected_features

    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data for feature selection."""
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column {self.target_column} not found in dataframe"
            )

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Handle categorical columns if any remain
        categorical_cols = X.select_dtypes(
            include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            logger.warning(
                f"Found categorical columns: {
                    list(categorical_cols)}. Converting to dummy variables."
            )
            X = pd.get_dummies(X, columns=list(
                categorical_cols), drop_first=True)

        # Handle missing values
        if X.isnull().any().any():
            logger.warning(
                "Found missing values in features. Filling with median/mode."
            )
            for col in X.columns:
                if X[col].dtype in ["int64", "float64"]:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(
                        X[col].mode()[0] if len(X[col].mode()) > 0 else 0
                    )

        logger.info(f"Prepared data: {X.shape[1]} features, {len(y)} samples")
        return X, y

    def _rfecv_selection(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> list[str]:
        """Recursive Feature Elimination with Cross-Validation."""
        logger.info("Applying RFECV feature selection")

        # Initialize estimator
        estimator = LinearRegression()

        # Apply RFECV
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_folds,
            scoring="r2",
            n_jobs=-1,
        )

        # Fit selector
        selector.fit(X, y)

        # Get selected features
        selected_features = X.columns[selector.support_].tolist()

        # Store feature rankings
        self.feature_rankings = dict(
            zip(X.columns, selector.ranking_, strict=False))

        # Get cross-validation scores
        cv_scores = selector.cv_results_["mean_test_score"]
        optimal_features = selector.n_features_

        step_info = {
            "method": "RFECV",
            "optimal_features": optimal_features,
            "cv_folds": cv_folds,
            "best_cv_score": max(cv_scores),
            "feature_rankings": self.feature_rankings,
            "selected_features": selected_features,
        }
        self.selection_report["selection_steps"].append(step_info)

        logger.info(
            f"RFECV selected {
                len(selected_features)} features with CV score: {
                    max(cv_scores):.4f}"
        )
        return selected_features

    def _correlation_selection(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.1
    ) -> list[str]:
        """Correlation-based feature selection."""
        logger.info("Applying correlation-based feature selection")

        # Calculate correlations with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)

        # Select features with correlation above threshold
        selected_features = correlations[correlations >=
                                         threshold].index.tolist()

        # Remove highly correlated features among themselves
        selected_features = self._remove_multicollinear_features(
            X[selected_features])

        step_info = {
            "method": "Correlation",
            "threshold": threshold,
            "correlations": correlations.to_dict(),
            "selected_features": selected_features,
        }
        self.selection_report["selection_steps"].append(step_info)

        logger.info(
            f"Correlation method selected {len(selected_features)} features")
        return selected_features

    def _combined_selection(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> list[str]:
        """Combined feature selection using multiple methods."""
        logger.info("Applying combined feature selection")

        # Step 1: Correlation pre-filtering
        correlation_features = self._correlation_selection(
            X, y, threshold=0.05)

        # Step 2: RFECV on pre-filtered features
        X_filtered = X[correlation_features]
        rfecv_features = self._rfecv_selection(X_filtered, y, cv_folds)

        step_info = {
            "method": "Combined",
            "step1_features": len(correlation_features),
            "step2_features": len(rfecv_features),
            "final_features": rfecv_features,
        }
        self.selection_report["selection_steps"].append(step_info)

        logger.info(f"Combined method selected {len(rfecv_features)} features")
        return rfecv_features

    def _remove_multicollinear_features(
        self, X: pd.DataFrame, threshold: float = 0.9
    ) -> list[str]:
        """Remove highly correlated features."""
        correlation_matrix = X.corr().abs()

        # Get upper triangle of correlation matrix
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        high_corr_features = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        # Keep features not in high correlation list
        selected_features = [
            col for col in X.columns if col not in high_corr_features]

        logger.info(
            f"Removed {len(high_corr_features)} highly correlated features")
        return selected_features

    def get_feature_importance_scores(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, float]:
        """Get feature importance scores using linear regression coefficients."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit linear regression
        lr = LinearRegression()
        lr.fit(X_scaled, y)

        # Get absolute coefficients as importance scores
        importance_scores = dict(
            zip(X.columns, np.abs(lr.coef_), strict=False))

        # Sort by importance
        importance_scores = dict(
            sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_scores

    def get_selection_report(self) -> dict[str, Any]:
        """Get detailed feature selection report."""
        return self.selection_report
