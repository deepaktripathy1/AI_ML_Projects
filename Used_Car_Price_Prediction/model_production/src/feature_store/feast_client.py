"""Feast feature store client."""

from datetime import datetime
from typing import Any, Optional, Dict
import json

import pandas as pd
from feast import FeatureStore, Entity, FeatureView, Field
from feast import FileSource
from feast.types import Float32, Int64, String
from sklearn.model_selection import train_test_split

from config.config import get_settings
from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)


class FeastClient:
    """Feast feature store client for data operations."""

    def __init__(self) -> None:
        """Initialize  Feast client."""
        settings = get_settings()
        self.repo_path = settings.BASE_DIR / "feature_repo"
        self.data_path = self.repo_path / "data"
        self.meta_path = self.repo_path / "metadata"
        self.meta_file = self.meta_path / "feature_groups.json"
        self.store = None
        self._initialize_repo()
        logger.info("Feast client initialized")

    def _initialize_repo(self) -> None:
        """Initialize feast repository."""
        try:
            # Create feature repo directory if it doesn't exist
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.meta_path.mkdir(parents=True, exist_ok=True)

            # Create feature_store.yaml
            feature_store_yaml = self.repo_path / "feature_store.yaml"
            if not feature_store_yaml.exists():
                yaml_content = """
project: used_car_price_prediction
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
entity_key_serialization_version: 2
"""
                feature_store_yaml.write_text(yaml_content)
                logger.info("Created Feast repository structure")

            # Initialize or load feature store
            try:
                self.store = FeatureStore(repo_path=str(self.repo_path))
                logger.info("Loaded existing Feast feature store")
            except Exception:
                # Create new feature store
                self.store = FeatureStore(repo_path=str(self.repo_path))
                logger.info("Created new Feast feature store")

        except Exception as e:
            logger.error(f"Failed to initialize Feast repository: {e}")
            raise

    def create_feature_group(
        self,
        df: pd.DataFrame,
        name: str = "used_car_features",
        version: int = 1,
        description: str = "Used car features for price prediction",
        primary_key: Optional[list[str]] = None,
        event_time: Optional[str] = None
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
            full_name = f"{name}_v{version}"

            if primary_key is None:
                primary_key = ["car_id"] if "car_id" in df.columns else []

            # Add event_time if not present
            if event_time is None or event_time not in df.columns:
                df = df.copy()
                df["event_timestamp"] = pd.Timestamp.now()
                event_time = "event_timestamp"

            # Ensure primary key exists
            if not primary_key or primary_key[0] not in df.columns:
                df = df.copy()
                df["car_id"] = range(1, len(df) + 1)
                primary_key = ["car_id"]

            # Save data to parquet for Feast
            data_path = self.data_path / f"{full_name}.parquet"
            df.to_parquet(data_path, index=False)

            # Create entity
            entity = Entity(
                name="car",
                join_keys=primary_key,
                description=f"Car entity (version {version})"
            )

            # Dynamically create schema based on DataFrame columns
            schema_fields = []
            for col in df.columns:
                if col not in primary_key and col != event_time and col != "price_usd":
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        schema_fields.append(Field(name=col, dtype=Int64))
                    elif pd.api.types.is_float_dtype(dtype):
                        schema_fields.append(Field(name=col, dtype=Float32))
                    else:
                        schema_fields.append(Field(name=col, dtype=String))

            # Define data source
            file_source = FileSource(
                path=str(data_path),
                event_timestamp_column=event_time,
            )

            # Define feature view
            feature_view = FeatureView(
                name=full_name,
                entities=[entity],
                schema=schema_fields,
                source=file_source,
                description=description,
            )

            # Apply entities and feature views
            if self.store:
                self.store.apply([entity, feature_view])

            # Materialize features
            if self.store:
                self.store.materialize(
                    start_date=datetime.now(),
                    end_date=datetime.now(),
                )

            # Record metadata
            meta = {
                "name": name,
                "version": version,
                "feature_view": full_name,
                "description": description,
                "path": str(data_path),
                "primary_key": primary_key,
                "event_time": event_time,
                "created_at": datetime.now().isoformat(),
                "row_count": len(df),
                "columns": list(df.columns)
            }

            if not self.meta_file.exists():
                with self.meta_file.open("w") as f:
                    json.dump(meta, f, indent=4)

            logger.info(f"✅ Created feature view '{full_name}' with {
                len(df)} records"
            )

        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            raise

    def create_feature_view(
        self,
        name: str = "used_car_price_features",
        version: int = 1,
        description: str = "Feature view for used car price prediction",
        labels:  Optional[list[str]] = None,
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
            full_name = f"{name}_v{version}"
            labels = [settings.TARGET_COLUMN]

            logger.info(
                f"Feature view '{full_name}' created with label: {
                    labels
                } and description: {
                    description
                }"
            )

        except Exception as e:
            logger.error(f"Failed to create feature view: {e}")
            raise

    def get_feature_group(self, name: str, version: int):
        """Get existing feature group."""
        full_name = f"{name}_v{version}"

        try:
            if self.store:
                feature_view = self.store.get_feature_view(full_name)
                logger.info(
                    f"Retrieved feature group or view '{full_name}'"
                )
                return feature_view

        except Exception as e:
            logger.error(f"Failed to get feature group or view: {e}")
            raise

    def get_feature_view(self, name: str, version: int = 1):
        """Get existing feature view."""
        full_name = f"{name}_v{version}"
        try:
            if self.store:
                feature_view = self.store.get_feature_view(full_name)
                logger.info(
                    f"Retrieved feature view '{name}' version {version}"
                )
                return feature_view
        except Exception as e:
            logger.error(f"Failed to get feature view: {e}")
            raise

    def get_training_data(
        self,
        feature_view_name: str = "used_car_price_features",
        version: int = 1,
        test_size: Optional[float] = None,
    ) -> tuple:
        """Get training and testing data from the feature view.

        Args:
            feature_view_name: Name of the feature view
            version: Version of the feature view
            test_size: Test set size ratio

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        settings = get_settings()
        full_name = f"{feature_view_name}_v{version}"
        try:
            # Set default test size
            if test_size is None:
                test_size = settings.TEST_SIZE

            # Load data
            data_path = self.data_path / f"{full_name}.parquet"
            if not data_path.exists():
                raise FileNotFoundError(
                    f"Feature data not found at {data_path}")

            df = pd.read_parquet(data_path)

            # Remove metadata columns
            metadata_cols = ["car_id", "event_timestamp"]
            feature_cols = [
                col for col in df.columns if col not in metadata_cols
            ]

            # Separate features and target
            target_col = settings.TARGET_COLUMN
            if target_col not in df.columns:
                raise ValueError(
                    f"Target columns '{target_col}' not found in data"
                )

            X = df[[col for col in feature_cols if col != target_col]]
            y = df[target_col]

            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=settings.RANDOM_STATE
            )

            logger.info(
                f"Retrieved training data: train = {
                    len(X_train)}, test = {len(X_test)}"
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            raise

    def update_feature_group_data(
        self,
        df: pd.DataFrame,
        feature_group: None
    ) -> None:
        """Append or update data for a given feature group."""
        try:
            feature_view_name = feature_group if isinstance(
                feature_group, str
            ) else "used_car_features_v1"

            # Add event_time if not present
            if "event_timestamp" not in df.columns:
                df = df.copy()
                df["event_timestamp"] = pd.Timestamp.now()

            # Ensure primary key exists
            if "car_id" not in df.columns:
                df = df.copy()
                df["car_id"] = range(1, len(df) + 1)

            # Append to existing parquet file
            data_path = self.data_path / f"{feature_view_name}.parquet"

            if data_path.exists():
                existing_df = pd.read_parquet(data_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.drop_duplicates(
                    subset=["car_id"],
                    keep="last"
                )
                combined_df.to_parquet(data_path, index=False)

                # Materialize updated features
                if self.store:
                    self.store.materialize(
                        start_date=datetime.now(),
                        end_date=datetime.now(),
                    )
                logger.info(
                    f"Updated feature group with {
                        len(df)} new records"
                )

        except Exception as e:
            logger.error(f"Failed to update feature group data: {e}")
            raise

    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get feature group statistics."""
        try:
            # Load data and compute basic statistics
            data_path = self.data_path / "used_car_features_v1.parquet"
            if not data_path.exists():
                return {}

            df = pd.read_parquet(path=data_path)

            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "null_counts": df.isnull().sum().to_dict()
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get feature statistics: {e}")
            return {}
