"""Hopsworks feature store client."""

from typing import Any, Optional, Tuple

import hopsworks
import pandas as pd

from config.config import get_settings
from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)


class HopsworksClient:
    """Hopsworks feature store client for data operations."""

    def __init__(self) -> None:
        """Initialize  Hopsworks client."""
        self.project = None
        self.fs = None
        self.feature_group = None
        self.feature_view = None
        self._connect()
        logger.info("Hopsworks client initialized")

    def _connect(self) -> None:
        """Connect to Hopsworks project and feature store."""
        settings = get_settings()
        try:
            self.project = hopsworks.login(
                api_key_value=settings.HOPSWORKS_API_KEY,
                project=settings.HOPSWORKS_PROJECT,
                host=settings.HOPSWORKS_HOST,
            )
            # Get feature store
            self.fs = self.project.get_feature_store()

            logger.info(
                f"Connected to Hopsworks project: {settings.HOPSWORKS_PROJECT}")

        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            raise ConnectionError("Failed to connect to Hopsworks") from e

    def create_feature_group(
        self,
        df: pd.DataFrame,
        name: str = "used_car_features",
        version: int = 1,
        description: str = "Used car features for price prediction ",
        primary_key: Optional[list[str]] = None,
        event_time: Optional[str] = None,
    ) -> None:
        """Create and insert data into a feature group.

        Args:
            df: DataFrame with features
            name: Feature group name
            version: Feature group version
            description: Feature group description
            primary_key: List of primary key columns
            event_time: Event time column name
        """
        try:
            if primary_key is None:
                primary_key = ["car_id"] if "car_id" in df.columns else []

            # Create feature group
            if self.fs:
                self.feature_group = self.fs.create_feature_group(
                    name=name,
                    version=version,
                    description=description,
                    primary_key=primary_key,
                    event_time=event_time,
                    online_enabled=True,
                    statistics_config=True,
                )

                # Insert data
                self.feature_group.insert(df, wait=True)

                # Update statistics
                self.feature_group.update_statistics_config()

                logger.info(
                    f"Successfully created feature group '{name}' with {
                        len(df)} records"
                )

        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")
            raise

    def create_feature_view(
        self,
        name: str = "used_car_price_features",
        version: int = 1,
        description: str = "Feature view for used car price prediction",
        labels: Optional[list[str]] = None,
    ) -> None:
        """Create a feature view from the feature group.

        Args:
            name: Feature view name
            version: Feature view version
            description: Feature view description
            labels: List of label columns for the feature view
        """
        settings = get_settings()
        try:
            if self.feature_group is None:
                raise ValueError(
                    "Feature group not created. Create feature group first."
                )

            # Set default labels if not provided
            if labels is None:
                labels = [settings.TARGET_COLUMN]

            # Create feature view
            if self.fs:
                self.feature_view = self.fs.create_feature_view(
                    name=name,
                    version=version,
                    description=description,
                    query=self.feature_group.select_all(),
                    labels=labels,
                )
                logger.info(f"Successfully created feature view '{name}'")

        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            raise

    def get_feature_group(self, name: str, version: int):
        """Get existing feature group."""
        try:
            self.feature_group = self.fs.get_feature_group(  # type: ignore
                name, version
            )
            logger.info(f"Retrieved feature group '{name}' version {version}")
            return self.feature_group
        except Exception as e:
            logger.error(f"Failed to get feature group: {e}")
            raise

    def get_feature_view(self, name: str, version: int = 1):
        """Get existing feature view."""
        try:
            self.feature_view = self.fs.get_feature_view(  # type: ignore
                name, version
            )
            logger.info(f"Retrieved feature view '{name}' version {version}")
            return self.feature_view
        except Exception as e:
            logger.error(f"Failed to get feature view: {e}")
            raise

    def get_training_data(
        self,
        feature_view_name: str = "used_car_price_features",
        version: int = 1,
        test_size: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Get training and testing data from the feature view.

        Args:
            feature_view_name: Name of the feature view
            version: Version of the feature view
            test_size: Test set size ratio

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        settings = get_settings()
        try:
            # Get feature view if not already loaded
            if self.feature_view is None:
                self.feature_view = self.get_feature_view(
                    feature_view_name, version
                )

            # Set default  test size
            if test_size is None:
                test_size = settings.TEST_SIZE

            # Create training data
            version, _ = self.feature_view.create_train_test_split(
                test_size=settings.TEST_SIZE,
                description="Create metadata for training dataset",
                data_format="csv"
            )

            logger.info(
                f"Created metadata for training with version {version}")

            # Get train-test split
            X_train, X_test, y_train, y_test = self.feature_view.get_train_test_split(
                training_dataset_version=version,
            )

            # Convert y_train/y_test to Series
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()

            logger.info(
                f"Retrieved training data: train={
                    len(X_train)}, test={
                        len(X_test)}"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            raise

    def update_feature_group_data(
            self,
            df: pd.DataFrame,
            feature_group=None
    ) -> None:
        """Update feature group with new data."""
        try:
            fg = feature_group or self.feature_group
            if fg is None:
                raise ValueError(
                    "Feature group not loaded. Load or create feature group first."
                )

            # Insert new data
            fg.insert(df, wait=True)

            # Update statistics
            fg.update_statistics_config()

            logger.info(f"Updated feature group with {len(df)} new records")

        except Exception as e:
            logger.error(f"Failed to update feature group: {e}")
            raise

    def get_feature_statistics(self) -> dict[str, Any]:
        """Get feature group statistics."""

        if self.feature_group is None:
            raise ValueError("Feature group not loaded.")

        try:
            stats = self.feature_group.get_statistics()
            return stats

        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}
