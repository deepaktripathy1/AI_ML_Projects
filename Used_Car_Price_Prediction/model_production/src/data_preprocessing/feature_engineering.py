"""Feature Engineering pipeline."""

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import get_settings
from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)


class FeatureEngineer:
    """Feature engineering pipeline."""

    def __init__(self):
        """Initialize feature engineer."""
        self.feature_engineering_report: dict[str, Any] = {
            "engineering_steps": []
        }
        self.encoders: dict[str, Any] = {}
        logger.info("Feature engineer initialized")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features and return dataframe.

        Args:
            df: Cleaned dataframe

        Returns:
            pd.DataFrame: Feature-engineered dataframe
        """
        logger.info("Starting feature engineering pipeline")
        initial_features = list(df.columns)

        df_engineered = df.copy()

        # Initialize feature report
        self.feature_engineering_report = {
            "initial_features": initial_features,
            "engineering_steps": [],
        }

        # Step 1: Create car_age feature
        df_engineered = self._create_car_age(df_engineered)

        # Step 2: Create has_accident feature
        df_engineered = self._create_has_accident(df_engineered)

        # Step 3: Ordinal encoding of service_history
        df_engineered = self._encode_service_history(df_engineered)

        # Step 4: One-hot encoding of categorical features
        df_engineered = self._encode_categorical_features(df_engineered)

        # Step 5: Dtop unnecessary columns
        df_engineered = self._drop_unnecessary_columns(df_engineered)

        # Finalize feature report
        self.feature_engineering_report["final_features"] = list(
            df_engineered.columns)
        self.feature_engineering_report["features_added"] = len(
            df_engineered.columns
        ) - len(initial_features)

        logger.info(
            f"Feature engineering completed. Features: {len(initial_features)} -> {len(df_engineered.columns)}"
        )
        return df_engineered

    def _create_car_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create car_age feature by subtracting make_year from current_year."""
        settings = get_settings()
        if "make_year" in df.columns:
            df["car_age"] = settings.CURRENT_YEAR - df["make_year"]

            step_info = {
                "step": "create_car_age",
                "action": f"Created car_age = {settings.CURRENT_YEAR} - make_year",
                "feature_added": "car_age",
                "statistics": {
                    "min_age": df["car_age"].min(),
                    "max_age": df["car_age"].max(),
                    "mean_age": df["car_age"].mean(),
                },
            }
            self.feature_engineering_report["engineering_steps"].append(
                step_info)

            logger.info(
                f"Created car_age feature. Range: {df['car_age'].min()} - {df['car_age'].max()} years"
            )

        return df

    def _create_has_accident(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create has_accident binary feature from accidents_reported."""
        if "accidents_reported" in df.columns:
            df["has_accident"] = df["accidents_reported"].apply(
                lambda x: 1 if x > 0 else 0
            )

            accident_distribution = df["has_accident"].value_counts().to_dict()

            step_info = {
                "step": "create_has_accident",
                "action": "Created binary has_accident feature",
                "feature_added": "has_accident",
                "distribution": accident_distribution,
            }
            self.feature_engineering_report["engineering_steps"].append(
                step_info)

            logger.info(
                f"Created has_accident feature distribution: {accident_distribution}"
            )

        return df

    def _encode_service_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ordinal encoding to service_history column."""
        settings = get_settings()
        if "service_history" in df.columns:
            mapping = settings.SERVICE_HISTORY_MAPPING
            df["service_history_encoded"] = df["service_history"].map(mapping)

            # Handle any unmapped values
            unmapped_mask = df["service_history_encoded"].isna()
            if unmapped_mask.any():
                logger.warning(
                    f"Found {unmapped_mask.sum()} unmapped service_history values"
                )
                # Default to "None"
                df.loc[unmapped_mask, "service_history_encoded"] = 0

            # Convert to integer
            df["service_history_encoded"] = df["service_history_encoded"].astype(
                int)

            distribution = df["service_history_encoded"].value_counts(
            ).to_dict()

            step_info = {
                "step": "encode_service_history",
                "action": "Applied ordinal encoding to service_history",
                "feature_added": "service_history_encoded",
                "mapping": mapping,
                "distribution": distribution,
            }
            self.feature_engineering_report["engineering_steps"].append(
                step_info)

            logger.info(
                f"Encoded service_history. Distribution: {distribution}")

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features."""
        settings = get_settings()
        categorical_features = [
            col for col in settings.CATEGORICAL_FEATURES if col in df.columns
        ]

        if categorical_features:
            # Apply one hot encoding
            df_encoded = pd.get_dummies(
                df,
                columns=categorical_features,
                drop_first=True,
                prefix=categorical_features,
            )

            # Get new columns names
            new_cols = [
                col for col in df_encoded.columns if col not in df.columns]

            step_info = {
                "step": "encode_categorical_features",
                "action": "Applied one-hot encoding to categorical features",
                "features_encoded": categorical_features,
                "new_features": new_cols,
                "new_features_count": len(new_cols),
            }
            self.feature_engineering_report["engineering_steps"].append(
                step_info)

            logger.info(
                f"One-hot encoded {len(categorical_features)} categorical features, created {len(new_cols)} new features"
            )
            return df_encoded

        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""
        settings = get_settings()
        cols_to_drop = [
            col for col in settings.FEATURES_TO_DROP if col in df.columns]

        # Also drop original service_history column if encoded version exists
        if "service_history_encoded" in df.columns and "service_history" in df.columns:
            cols_to_drop.append("service_history")

        if cols_to_drop:
            df_final = df.drop(columns=cols_to_drop)

            step_info = {
                "step": "drop_unnecessary_columns",
                "action": "Dropped unnecessary columns",
                "columns_dropped": cols_to_drop,
                "columns_remaining": list(df_final.columns),
            }
            self.feature_engineering_report["engineering_steps"].append(
                step_info)

            logger.info(f"Dropped {len(cols_to_drop)} unnecessary columns")
            return df_final

        return df

    def get_feature_importance_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get data for feature importance analysis."""
        numerical_features = df.select_dtypes(
            include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        return {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "total_features": len(df.columns),
            "feature_types": {
                "numerical": len(numerical_features),
                "categorical": len(categorical_features),
            },
        }

    def get_feature_report(self) -> dict[str, Any]:
        """Get detailed feature engineering report."""
        return self.feature_engineering_report
