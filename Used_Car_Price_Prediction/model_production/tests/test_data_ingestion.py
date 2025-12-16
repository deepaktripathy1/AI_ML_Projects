"""Test data ingestion components."""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from pymongo.errors import ConnectionFailure, OperationFailure

from src.data_ingestion.data_loader import Dataloader
from src.data_ingestion.mongodb_connector import MongoDBConnector


class TestDataLoader:
    """Test cases for DataLoader class."""

    def test_initialization(self, mock_settings, temp_cache_dir):
        """Test dataloader initialization."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings_patch:
            mock_settings_patch.data_dir = temp_cache_dir

            # Test with cache enabled
            loader = Dataloader(cache_enabled=True)
            assert loader.cache_enabled is True
            assert loader.cache_dir == temp_cache_dir / "cache"
            assert loader.cache_dir.exists()

            # Test with cache disables
            loader_no_cache = Dataloader(cache_enabled=False)
            assert loader_no_cache.cache_enabled is False

    @patch("src.data_ingestion.data_loader.MongoDBConnector")
    def test_load_raw_data_from_mongodb(
        self, mock_connector_class, sample_raw_data, temp_cache_dir
    ):
        """Test loading raw data from MongoDB."""
        # Set up mocks
        mock_connector = Mock()
        mock_connector.test_connector.return_value = True
        mock_connector.fetch_data.return_value = sample_raw_data
        mock_connector_class.return_value = mock_connector

        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            loader = Dataloader()
            result_df = loader.load_raw_data(use_cache=False)

            # Assertions
            assert isinstance(result_df, pd.DataFrame)
            assert not result_df.empty
            assert len(result_df) == len(sample_raw_data)
            mock_connector.test_connection.assert_called_once()
            mock_connector.fetch_data.assert_called_once()

    def test_load_raw_data_with_valid_cache(self, sample_raw_data, temp_cache_dir):
        """Test loading data from valid cache."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            # Create cache file
            cache_dir = temp_cache_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "raw_data.pkl"
            joblib.dump(sample_raw_data, cache_file)

            # Mock MongoDB connector to ensure it"s not called
            with patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class:
                mock_connector = Mock()
                mock_connector_class.return_value = mock_connector

                loader = Dataloader()
                result_df = loader.load_raw_data(
                    use_cache=True, cache_expiry_hours=24)

                # Should load from cache, not call MongoDB
                assert isinstance(result_df, pd.DataFrame)
                assert len(result_df) == len(sample_raw_data)
                mock_connector.test_connection.assert_not_called()
                mock_connector.fetch_data.assert_not_called()

    def test_load_raw_data_with_expired_cache(self, sample_raw_data, temp_cache_dir):
        """Test loading data with expired cache."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            # Create expired cache file
            cache_dir = temp_cache_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "raw_data.pkl"
            joblib.dump(sample_raw_data, cache_file)

            # Make cache file appear old
            old_time = datetime.now() - timedelta(hours=25)
            os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))

            with patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class:
                mock_connector = Mock()
                mock_connector.test_connection.return_value = True
                mock_connector.fetch_data.return_value = sample_raw_data
                mock_connector_class.return_value = mock_connector

                loader = Dataloader()
                result_df = loader.load_raw_data(
                    use_cache=True, cache_expiry_hours=24)

                # Should load from MongoDB, not cache
                assert isinstance(result_df, pd.DataFrame)
                mock_connector.test_connection.assert_called_once()
                mock_connector.fetch_data.assert_called_once()

    def test_load_raw_data_mongodb_connection_failure(self, temp_cache_dir):
        """Test handling MongoDB connection failure."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            with patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class:
                mock_connector = Mock()
                mock_connector.test_connection.return_value = False
                mock_connector_class.return_value = mock_connector

                loader = Dataloader()

                with pytest.raises(
                    ConnectionError, match="Failed to connect to MongoDB"
                ):
                    loader.load_raw_data(use_cache=False)

    def test_load_raw_data_empty_dataframe(self, temp_cache_dir):
        """Test handling empty DataFrame from MongoDB."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            with patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class:
                mock_connector = Mock()
                mock_connector.test_connection.return_value = True
                mock_connector.fetch_data.return_value = (
                    pd.DataFrame()
                )  # Empty DataFrame
                mock_connector_class.return_value = mock_connector

                loader = Dataloader()

                with pytest.raises(ValueError, match="No data found in MongoDB"):
                    loader.load_raw_data(use_cache=False)

    def test_validate_raw_data_success(self, sample_raw_data, temp_cache_dir):
        """Test successful data validation."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            loader = Dataloader()
            # Should not raise exception
            loader._validate_raw_data(sample_raw_data)

    def test_validate_raw_data_missing_columns(self, temp_cache_dir):
        """Test data validation with missing required columns."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            # Create DataFrame missing required columns
            incomplete_df = pd.DataFrame({"price_usd": [1000, 2000]})

            loader = Dataloader()
            with pytest.raises(ValueError, match="Missing required columns"):
                loader._validate_raw_data(incomplete_df)

    def test_is_cache_valid(self, temp_cache_dir):
        """Test cache validation logic."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            loader = Dataloader()
            cache_file = temp_cache_dir / "test_cache.pkl"

            # Non-existent file should be invalid
            assert not loader._is_cache_valid(cache_file, 24)

            # Create fresh cache file
            cache_file.touch()
            assert loader._is_cache_valid(cache_file, 24)

            # Make file old
            old_time = datetime.now() - timedelta(hours=25)
            os.utime(cache_file, (old_time.timestamp(), old_time.timestamp()))
            assert not loader._is_cache_valid(cache_file, 24)

    def test_get_data_info(self, sample_raw_data, temp_cache_dir):
        """Test data information extraction."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            loader = Dataloader()
            info = loader.get_data_info(sample_raw_data)

            # Check required keys
            required_keys = [
                "shape",
                "columns",
                "dtypes",
                "null_counts",
                "null_percentages",
                "memory_usage",
                "duplicated_rows",
            ]
            for key in required_keys:
                assert key in info

            # Check data types
            assert "numerical_stats" in info
            assert "cat_stats" in info

            # Validate content
            assert info["shape"] == sample_raw_data.shape
            assert set(info["columns"]) == set(sample_raw_data.columns)

    def test_clear_cache(self, temp_cache_dir):
        """Test cache clearing functionality."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            # Create cache files
            cache_dir = temp_cache_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "test1.pkl").touch()
            (cache_dir / "test2.pkl").touch()
            (cache_dir / "not_cache.txt").touch()  # Should not be deleted

            loader = Dataloader()
            loader.clear_cache()

            # Check that only .pkl files are deleted
            assert not (cache_dir / "test1.pkl").exists()
            assert not (cache_dir / "test2.pkl").exists()
            assert (cache_dir / "not_cache.txt").exists()

    def test_close_connections(self, temp_cache_dir):
        """Test connection closing."""
        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir

            with patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class:
                mock_connector = Mock()
                mock_connector_class.return_value = mock_connector

                loader = Dataloader()
                loader.close_connections()

                mock_connector.close_connection.assert_called_once()


class TestMongoDBConnector:
    """Test cases for MongoDBConnector class."""

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_initialization(self, mock_config_class):
        """Test MongoDB connector initialization."""

        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # with patch("src.data_ingestion.mongodb_connector.settings") as #mock_settings:
        # mock_settings.MONGODB_URL = "test_url"
        # mock_settings.MONGODB_DATABASE = "test_db"
        # mock_settings.MONGODB_COLLECTION = "test_collection"

        connector = MongoDBConnector()

        assert connector.db_config == mock_config
        mock_config_class.assert_called_once_with(
            connection_string="mongodb://localhost:27017",
            database_name="test_db",
            collection_name="test_collection",
        )

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_fetch_data_success(self, mock_config_class, sample_raw_data):
        """Test successful data fetching."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_cursor = MagicMock()

        # Setup mock data with _id field
        test_data = sample_raw_data.to_dict("records")
        for i, record in enumerate(test_data):
            record["_id"] = f"id_{i}"

        mock_cursor.__iter__.return_value = iter(test_data)
        mock_collection.find.return_value = mock_cursor
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        result_df = connector.fetch_data()

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_raw_data)
        assert "_id" not in result_df.columns  # Should be removed
        mock_collection.find.assert_called_once_with({}, None)

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_fetch_data_with_parameters(self, mock_config_class):
        """Test data fetching with query, projection, and limit."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter(
            [{"field1": "value1", "_id": "test_id"}]
        )
        mock_cursor.limit.return_value = mock_cursor
        mock_collection.find.return_value = mock_cursor
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        query = {"field1": "value1"}
        projection = {"field1": 1}
        limit = 10

        result_df = connector.fetch_data(
            query=query, projection=projection, limit=limit
        )

        assert isinstance(result_df, pd.DataFrame)
        mock_collection.find.assert_called_once_with(query, projection)
        mock_cursor.limit.assert_called_once_with(limit)

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_fetch_data_empty_result(self, mock_config_class):
        """Test fetching with no data found."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_cursor = MagicMock()
        mock_cursor.__iter__.return_value = iter([])
        mock_collection.find.return_value = mock_cursor
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        result_df = connector.fetch_data()

        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_fetch_data_connection_failure(self, mock_config_class):
        """Test handling connection failure during fetch."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection.find.side_effect = ConnectionFailure(
            "Connection failed")
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()

        with pytest.raises(ConnectionFailure):
            connector.fetch_data()

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_fetch_data_operation_failure(self, mock_config_class):
        """Test handling operation failure during fetch."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection.find.side_effect = OperationFailure("Operation failed")
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()

        with pytest.raises(OperationFailure):
            connector.fetch_data()

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_insert_data_success(self, mock_config_class, sample_raw_data):
        """Test successful data insertion."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_result = Mock()
        mock_result.inserted_ids = ["id1", "id2", "id3"]
        mock_collection.insert_many.return_value = mock_result
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        result = connector.insert_data(sample_raw_data)

        assert result is True
        mock_collection.insert_many.assert_called_once()
        # Check that DataFrame was converted to dict records
        call_args = mock_collection.insert_many.call_args[0][0]
        assert len(call_args) == len(sample_raw_data)

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_insert_data_failure(self, mock_config_class, sample_raw_data):
        """Test handling data insertion failure."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection.insert_many.side_effect = Exception("Insert failed")
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        result = connector.insert_data(sample_raw_data)

        assert result is False

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_get_collection_statistics(self, mock_config_class):
        """Test getting collection statistics."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection.count_documents.return_value = 100
        mock_collection.name = "test_collection"
        mock_collection.database.name = "test_database"
        mock_collection.find_one.return_value = {
            "_id": "test_id",
            "field1": "value1",
            "field2": "value2",
        }
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        stats = connector.get_collection_statistics()

        expected_stats = {
            "document_count": 100,
            "collection_name": "test_collection",
            "database_name": "test_database",
            "sample_fields": ["_id", "field1", "field2"],
        }

        assert stats == expected_stats
        mock_collection.count_documents.assert_called_once_with({})
        mock_collection.find_one.assert_called_once()

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_get_collection_statistics_failure(self, mock_config_class):
        """Test handling statistics retrieval failure."""
        mock_config = Mock()
        mock_collection = Mock()
        mock_collection.count_documents.side_effect = Exception("Stats failed")
        mock_config.get_collection.return_value = mock_collection
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        stats = connector.get_collection_statistics()

        assert stats == {}

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_test_connection(self, mock_config_class):
        """Test connection testing."""
        mock_config = Mock()
        mock_config.test_connection.return_value = True
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        result = connector.test_connection()

        assert result is True
        mock_config.test_connection.assert_called_once()

    @patch("src.data_ingestion.mongodb_connector.MongoDBConfig")
    def test_close_connection(self, mock_config_class):
        """Test connection closing."""
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        connector = MongoDBConnector()
        connector.close_connection()

        mock_config.close_connection.assert_called_once()


class TestIntegrationDataIngestion:
    """Integration tests for data ingestion components."""

    @patch("src.data_ingestion.data_loader.MongoDBConnector")
    def test_end_to_end_data_loading(
        self, mock_connector_class, sample_raw_data, temp_cache_dir
    ):
        """Test complete data loading workflow."""
        # Setup MongoDB mock
        mock_connector = Mock()
        mock_connector.test_connection.return_value = True
        mock_connector.fetch_data.return_value = sample_raw_data
        mock_connector_class.return_value = mock_connector

        with patch("src.data_ingestion.data_loader.settings") as mock_settings:
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            # Initialize loader and load data
            loader = Dataloader(cache_enabled=True)

            # First load should fetch from MongoDB and cache
            df1 = loader.load_raw_data(use_cache=True)
            assert len(df1) == len(sample_raw_data)
            assert mock_connector.fetch_data.call_count == 1

            # Second load should use cache
            df2 = loader.load_raw_data(use_cache=True)
            assert len(df2) == len(sample_raw_data)
            assert mock_connector.fetch_data.call_count == 1  # Should not increase

            # Get data info
            info = loader.get_data_info(df2)
            assert info["shape"] == df2.shape
            assert len(info["columns"]) == len(df2.columns)

            # Clear cache and close connections
            loader.clear_cache()
            loader.close_connections()

            mock_connector.close_connection.assert_called_once()

    @pytest.mark.integration
    def test_mongodb_connector_with_dataloader_integration(
        self, sample_raw_data, temp_cache_dir
    ):
        """Test MongoDB connector integration with dataloader."""
        with (
            patch(
                "src.data_ingestion.mongodb_connector.MongoDBConfig"
            ) as mock_config_class,
            patch(
                "src.data_ingestion.data_loader.settings"
            ) as mock_settings
        ):
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            # Setup MongoDB config mock
            mock_config = Mock()
            mock_collection = Mock()

            # Setup data with _id fields
            test_data = sample_raw_data.to_dict("records")
            for i, record in enumerate(test_data):
                record["_id"] = f"id_{i}"

            mock_cursor = MagicMock()
            mock_cursor.__iter__.return_value = iter(test_data)
            mock_collection.find.return_value = mock_cursor
            mock_config.get_collection.return_value = mock_collection
            mock_config.test_connection.return_value = True
            mock_config_class.return_value = mock_config

            # Test dataloader using MongoDB connector
            # Disable cache for this test
            loader = Dataloader(cache_enabled=False)
            result_df = loader.load_raw_data(use_cache=False)

            # Verify results
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(sample_raw_data)
            assert "_id" not in result_df.columns

            # Verify MongoDB calls
            mock_config.test_connection.assert_called()
            mock_collection.find.assert_called()

    @pytest.mark.slow
    def test_large_dataset_handling(self, temp_cache_dir):
        """Test handling of large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = {
            "price_usd": np.random.uniform(10000, 50000, 10000),
            "make_year": np.random.randint(2000, 2024, 10000),
            "fuel_type": np.random.choice(["Petrol", "Diesel", "Electric"], 10000),
            "brand": np.random.choice(
                ["Toyota", "Honda", "BMW", "Tesla", "Kia"], 10000
            ),
            "transmission": np.random.choice(["Manual", "Automatic"], 10000),
            "color": np.random.choice(
                ["White", "Black", "Silver", "Red", "Blue"], 10000
            ),
            "insurance_valid": np.random.choice(["Yes", "No"], 10000),
            "service_history": np.random.choice(["Complete", "Partial", "None"], 10000),
            "accidents_reported": np.random.randint(0, 5, 10000),
        }
        large_df = pd.DataFrame(large_data)

        with (
            patch(
                "src.data_ingestion.data_loader.MongoDBConnector"
            ) as mock_connector_class,
            patch(
                "src.data_ingestion.data_loader.settings"
            ) as mock_settings,
        ):
            mock_settings.data_dir = temp_cache_dir
            mock_settings.CURRENT_YEAR = 2025

            mock_connector = Mock()
            mock_connector.test_connection.return_value = True
            mock_connector.fetch_data.return_value = large_df
            mock_connector_class.return_value = mock_connector

            loader = Dataloader(cache_enabled=True)
            result_df = loader.load_raw_data(use_cache=False)

            assert len(result_df) == 10000
            assert isinstance(result_df, pd.DataFrame)

            # Test data info on large dataset
            info = loader.get_data_info(result_df)
            assert info["shape"][0] == 10000
            assert "memory_usage" in info
            assert info["memory_usage"] > 0
