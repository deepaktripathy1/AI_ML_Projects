"""Unified feature store client."""

from typing import Any, Optional, Tuple

import pandas as pd
import requests

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.feature_store.hopsworks_client import HopsworksClient
from src.feature_store.feast_client import FeastClient


logger = LoggingConfig().get_logger(__name__)


class FeatureStoreClient:
    """Unified feature store client."""

    def __init__(self) -> None:
        """Initialize unified feature store client."""
        self.primary_client: Optional[HopsworksClient] = None
        self.fallback_client: Optional[FeastClient] = None
        self.active_client = None
        self.client_type = None

        self._initialize_clients()
        logger.info(
            f"Unified feature store initialized using: {self.client_type}"
        )

    def _initialize_clients(self) -> None:
        """Initialize feature store clients with fallback logic."""
        settings = get_settings()

        # Try Hopsworks
        try:
            logger.info("Attempting to connect with Hopsworks...")

            # Check if Hopsworks is accessible
            if self._check_hopsworks_availability():
                self.primary_client = HopsworksClient()
                self.active_client = self.primary_client
                self.client_type = "Hopsworks"
                logger.info("Successfully connected to Hopsworks")
                return
            else:
                logger.warning("Hopsworks server is not accessible")

        except Exception as e:
            logger.warning(f"Failed to connect to Hopsworks: {e}")

        # Fallback to Feast
        try:
            logger.info("Falling back to Feast feature store...")
            self.fallback_client = FeastClient()
            self.active_client = self.fallback_client
            self.client_type = "Feast"
            logger.info("Successfully initialized Feast as fallback")

        except Exception as e:
            logger.error(f"Failed to initialize Feast fallback: {e}")
            raise RuntimeError(
                "Failed to initialize both Hopsworks and Feast feature stores"
            ) from e

    def _check_hopsworks_availability(self) -> bool:
        """Check if Hopsworks server is available."""
        settings = get_settings()
        try:
            if not settings.HOPSWORKS_HOST:
                return False

            # Check if host is reachable
            response = requests.get(
                f"https://{settings.HOPSWORKS_HOST}",
                timeout=10,
                verify=True
            )

            if response.status_code == 503:
                logger.warning("Hopsworks returned 503 service unavailable")
                return False

            return True

        except requests.exceptions.Timeout:
            logger.warning("Hopsworks connection timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning("Cannot connect to Hopsworks server")
            return False
        except Exception as e:
            logger.warning(f"Hopsworks availability check failed: {e}")
            return False

    def create_feature_group(
        self,
        df: pd.DataFrame,
        name: str = "used_car_features",
        version: int = 1,
        description: str = "Used car features for price prediction",
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
            if self.active_client:
                self.active_client.create_feature_group(
                    df=df,
                    name=name,
                    version=version,
                    description=description,
                    primary_key=primary_key,
                    event_time=event_time,
                )
            logger.info(f"Feature group created using {self.client_type}")

        except Exception as e:
            logger.error(f"Failed to create feature group: {e}")

            # If Hopsworks fails, try Feast fallback
            if self.client_type == "Hopsworks" and self.fallback_client is None:
                logger.info("Attempting to create Feast client...")
                try:
                    self.fallback_client = FeastClient()
                    self.active_client = self.fallback_client
                    self.client_type = "Feast"

                    self.active_client.create_feature_group(
                        df=df,
                        name=name,
                        version=version,
                        description=description,
                        primary_key=primary_key,
                        event_time=event_time,
                    )
                    logger.info(
                        "Successfully created feature group using Feast fallback")
                except Exception as fallback_error:
                    logger.error(
                        f"Feast fallback also failed: {fallback_error}")
                    raise
            else:
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
            if self.active_client:
                self.active_client.create_feature_view(
                    name=name,
                    version=version,
                    description=description,
                    labels=labels
                )
            logger.info(f"Feature view created using {self.client_type}")

        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")

            # Try Feast fallback
            if self.client_type == "Hopsworks" and self.fallback_client is None:
                try:
                    self.fallback_client = FeastClient()
                    self.active_client = self.fallback_client
                    self.client_type = "Feast"

                    self.active_client.create_feature_view(
                        name=name,
                        version=version,
                        description=description,
                        labels=labels,
                    )
                    logger.info(
                        "Successfully created feature view using Feast fallback")
                except Exception as fallback_error:
                    logger.error(
                        f"Feast fallback also failed: {fallback_error}")
                    raise
            else:
                raise

    def get_feature_group(self, name: str, version: int):
        """Get existing feature group."""
        try:
            if self.active_client:
                return self.active_client.get_feature_group(name, version)
        except Exception as e:
            logger.error(f"Failed to get feature group: {e}")
            raise

    def get_feature_view(self, name: str, version: int = 1):
        """Get existing feature view."""
        try:
            if self.active_client:
                return self.active_client.get_feature_view(name, version)
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
            Tuple of (X_train, X_test, y_train, y_test)
        """
        settings = get_settings()
        try:
            if test_size is None:
                test_size = settings.TEST_SIZE

            # For Feast, use the feature group name instead
            if self.active_client and self.client_type == "Feast":
                feature_view_name = "used_car_features"

                return self.active_client.get_training_data(
                    feature_view_name=feature_view_name,
                    version=version,
                    test_size=test_size,
                )
            # Hopsworks
            elif self.active_client:
                return self.active_client.get_training_data(
                    feature_view_name=feature_view_name,
                    version=version,
                    test_size=test_size,
                )
            else:
                raise ValueError(
                    "Active feature store client is not initialized."
                )
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            raise

    def update_feature_group_data(
        self, df: pd.DataFrame, feature_group=None
    ) -> None:
        """Update feature group with new data."""
        try:
            if self.active_client:
                self.active_client.update_feature_group_data(df, feature_group)
                logger.info(f"Feature group updated using {self.client_type}")

        except Exception as e:
            logger.error(f"Failed to update feature group: {e}")
            raise

    def get_feature_statistics(self) -> dict[str, Any]:
        """Get feature group statistics."""
        try:
            if self.active_client:
                return self.active_client.get_feature_statistics()
            else:
                # Fallback if client is not initialized
                logger.warning(
                    "Active client not initialized. Returning empty statistics."
                )
            return {}
        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}

    def get_client_type(self) -> str:
        """Get the active client type."""
        return self.client_type or "Unknown"

    def is_using_hopsworks(self) -> bool:
        """Check if currently using Hopsworks."""
        return self.client_type == "Hopsworks"

    def is_using_feast(self) -> bool:
        """Check if currently using Feast."""
        return self.client_type == "Feast"
