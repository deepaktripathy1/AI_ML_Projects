"""Data Loader with caching and validation capabilities."""

from pathlib import Path
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.config import get_settings
from config.logging_config import LoggingConfig
from src.data_ingestion.mongodb_connector import MongoDBConnector


logger = LoggingConfig().get_logger(__name__)


class Dataloader:
    """Data Loader with caching and validation."""

    def __init__(self, cache_enabled: bool = True):
        """Initialize data loader.

        Args:
            cache_enabled: Whether to enable caching
        """
        settings = get_settings()

        self.cache_enabled = cache_enabled
        self.cache_dir = settings.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mongodb_connector = MongoDBConnector(
            connection_string=settings.MONGODB_URL,
            database_name=settings.MONGODB_NAME,
            collection_name=settings.MONGODB_COLLECTION
        )
        logger.info("Data loader initialized")

    def load_raw_data(
        self, use_cache: bool = True, cache_expiry_hours: int = 24
    ) -> pd.DataFrame:
        """Load raw data from MongoDB and cache it if enabled.

        Args:
            use_cache: Whether to use cache
            cache_expiry_hours: Number of hours before cache expires

        Returns:
            pandas.DataFrame: Raw data
        """
        cache_file = self.cache_dir / "raw_data.pkl"

        #  # Check if cached data exists and is valid
        if (
            use_cache
            and self.cache_enabled
            and self._is_cache_valid(cache_file, cache_expiry_hours)
        ):
            logger.info("Using cached data")
            return joblib.load(cache_file)

        # Load data from MongoDB
        logger.info("Loading raw data from MongoDB")

        try:
            self.mongodb_connector.connect()

            # Test connection first
            if not self.mongodb_connector.test_connection():
                raise ConnectionError("Failed to connect to MongoDB")

            # Fetch data
            df = self.mongodb_connector.fetch_data()

            if df.empty:
                raise ValueError("No data found in MongoDB")

            # Validate data structure
            self._validate_raw_data(df)

            # Cache data
            if self.cache_enabled:
                joblib.dump(df, cache_file)
                logger.info("Cached data successfully to {cache_file}")

            logger.info("Successfully loaded {len(df)} records from MongoDB")

            return df

        except Exception as e:
            logger.error(f"Failed to load raw data from MongoDB: {e}")
            raise

    def _validate_raw_data(self, df: pd.DataFrame) -> None:
        """Validate raw data structure and content.

        Args:
            df: DataFrame
        """
        settings = get_settings()
        required_cols = [
            "price_usd",
            "make_year",
            "mileage_kmpl",
            "engine_cc",
            "fuel_type",
            "owner_count",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
            "accidents_reported",
        ]

        # Check required cols
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check data types and ranges
        current_year = int(settings.CURRENT_YEAR)
        validations = [
            ((df["price_usd"] > 0).fillna(False), "Price must be positive"),
            (
                (df["make_year"] >= 1900).fillna(
                    False), "Make year must be at least 1900"
            ),
            (
                (df["make_year"] <= current_year).fillna(False),
                f"Make year must be less than or equal to {current_year}",
            ),
            ((df["accidents_reported"] >= 0).fillna(False),
             "Accidents reported must be non-negative"),
        ]
        for condition, message in validations:
            if not condition.all():
                logger.warning(f"Data Validation warning: {message}")

        logger.info("Raw Data validation complete")

    def _is_cache_valid(self, cache_file: Path, expiry_hours: int) -> bool:
        """Check if cache file is valid and not expired.

        Args:
            cache_file: Path to cache file
            expiry_hours: Number of hours before cache expires

        Returns:
            bool: True if cache file is valid and not expired, False otherwise
        """
        if not cache_file.exists():
            return False

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age.total_seconds() < (expiry_hours * 3600)

    def get_data_info(self, df: pd.DataFrame) -> dict[str, Any]:
        """Get comprehensive data information.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict[str, Any]: Data information
        """
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "null_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "duplicated_rows": df.duplicated().sum(),
        }

        # Numerical column statistics
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            info["numerical_stats"] = df[num_cols].describe().to_dict()

        # Categorical column statistics
        cat_cols = df.select_dtypes(include=["object"]).columns
        if len(cat_cols) > 0:
            info["cat_stats"] = {}
            for col in cat_cols:
                info["cat_stats"][col] = {
                    "unique_count": df[col].nunique(),
                    "unique_vaues": df[col].unique().tolist()[:10],
                }

        return info

    def clear_cache(self) -> None:
        """Clear cached data."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")

    def close_connections(self) -> None:
        """Close all database connections."""
        self.mongodb_connector.close()
