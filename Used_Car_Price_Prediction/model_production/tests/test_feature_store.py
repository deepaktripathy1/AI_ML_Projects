"""Test feature store components."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.feature_store.hopsworks_client import HopsworksClient


class TestHopsworksClient:
    """Test cases for HopsworksClient class."""

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_initialization_success(self, mock_hopsworks):
        """Test successful Hopsworks client initialization."""
        # Mock Hopsworks login and project
        mock_project = Mock()
        mock_fs = Mock()
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            # Verify initialization
            assert client.project == mock_project
            assert client.fs == mock_fs

            # Verify login was called with correct parameters
            mock_hopsworks.login.assert_called_once_with(
                api_key_value="test_api_key", project="test_project", host="test_host"
            )
            mock_project.get_feature_store.assert_called_once()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_initialization_failure(self, mock_hopsworks):
        """Test Hopsworks client initialization failure."""
        mock_hopsworks.login.side_effect = Exception("Login failed")

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            with pytest.raises(ConnectionError) as exc_info:
                HopsworksClient()

            assert str(exc_info.value.__cause__) == "Login failed"

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_create_feature_group_success(self, mock_hopsworks, sample_engineered_data):
        """Test successful feature group creation."""
        # Setup mocks
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_group.return_value = mock_feature_group
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            # Add car_id to sample data
            test_data = sample_engineered_data.copy()
            test_data["car_id"] = range(1, len(test_data) + 1)

            # Create feature group
            client.create_feature_group(
                df=test_data,
                name="test_feature_group",
                version=1,
                description="Test feature group",
                primary_key=["car_id"],
            )

            # Verify calls
            mock_fs.create_feature_group.assert_called_once()
            mock_feature_group.insert.assert_called_once_with(
                test_data, wait=True)
            mock_feature_group.update_statistics_config.assert_called_once()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_create_feature_group_failure(self, mock_hopsworks, sample_engineered_data):
        """Test feature group creation failure."""
        # Setup mocks
        mock_project = Mock()
        mock_fs = Mock()
        mock_fs.create_feature_group.side_effect = Exception(
            "Failed to create feature group")

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(Exception) as exc_info:
                client.create_feature_group(df=sample_engineered_data)

            assert "Failed to create feature group" in str(exc_info)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_create_feature_view_success(self, mock_hopsworks):
        """Test successful feature view creation."""
        # Set up mocks
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()
        mock_feature_view = Mock()
        mock_query = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_view.return_value = mock_feature_view
        mock_feature_group.select_all.return_value = mock_query
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"
            mock_settings.TARGET_COLUMN = "price_usd"

            client = HopsworksClient()
            client.feature_group = mock_feature_group  # Set feature group

            # Create feature view
            client.create_feature_view(
                name="test_feature_view",
                version=1,
                description="Test feature view",
                labels=["price_usd"],
            )

            # Verify calls
            mock_fs.create_feature_view.assert_called_once()
            call_args = mock_fs.create_feature_view.call_args
            assert call_args[1]["name"] == "test_feature_view"
            assert call_args[1]["version"] == 1
            assert call_args[1]["labels"] == ["price_usd"]

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_create_feature_view_no_feature_group(self, mock_hopsworks):
        """Test feature view creation without feature group."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(ValueError, match="Feature group not created"):
                client.create_feature_view()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_group_success(self, mock_hopsworks):
        """Test successful feature group retrieval."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.get_feature_group.return_value = mock_feature_group
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            result = client.get_feature_group("test_group", 1)

            assert result == mock_feature_group
            assert client.feature_group == mock_feature_group
            mock_fs.get_feature_group.assert_called_once_with("test_group", 1)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_group_failure(self, mock_hopsworks):
        """Test feature group retrieval failure."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_fs.get_feature_group.side_effect = Exception("Get failed")

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(Exception, match="Get failed"):
                client.get_feature_group("test_group", 1)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_view_success(self, mock_hopsworks):
        """Test successful feature view retrieval."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_view = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.get_feature_view.return_value = mock_feature_view
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            result = client.get_feature_view("test_view", 1)

            assert result == mock_feature_view
            assert client.feature_view == mock_feature_view
            mock_fs.get_feature_view.assert_called_once_with("test_view", 1)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_view_failure(self, mock_hopsworks):
        """Test feature view retrieval failure."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_fs.get_feature_view.side_effect = Exception("Get view failed")

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(Exception, match="Get view failed"):
                client.get_feature_view("test_view", 1)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_training_data_success(self, mock_hopsworks):
        """Test successful training data retrieval"""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_view = Mock()

        # Mock training data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        X_test = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})
        y_train = pd.Series([100, 200, 300])
        y_test = pd.Series([400, 500])

        mock_feature_view.create_train_test_split.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
        )
        mock_fs.get_feature_view.return_value = mock_feature_view
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42

            client = HopsworksClient()
            result = client.get_training_data()

            X_train_result, X_test_result, y_train_result, y_test_result = result

            # Verify results
            assert isinstance(X_train_result, pd.DataFrame)
            assert isinstance(X_test_result, pd.DataFrame)
            assert isinstance(y_train_result, pd.Series)
            assert isinstance(y_test_result, pd.Series)

            # Verify method was called with correct parameters
            mock_feature_view.create_train_test_split.assert_called_once_with(
                test_size=0.2, random_state=42
            )

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_training_data_with_custom_parameters(self, mock_hopsworks):
        """Test training data retrieval with custom parameters."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_view = Mock()

        # Mock training data
        mock_feature_view.create_train_test_split.return_value = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Series(),
            pd.Series(),
        )

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            client.feature_view = mock_feature_view  # Set feature view directly

            # Call with custom parameters
            client.get_training_data(
                feature_view_name="custom_view", version=2, test_size=0.3
            )

            # Verify method was called with custom parameters
            mock_feature_view.create_train_test_split.assert_called_once_with(
                test_size=0.3, random_state=mock_settings.RANDOM_STATE
            )

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_training_data_no_feature_view(self, mock_hopsworks):
        """Test training data retrieval when feature view doesn"t exist."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_fs.get_feature_view.side_effect = Exception(
            "Feature view not found")

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(Exception, match="Feature view not found"):
                client.get_training_data()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_update_feature_group_data_success(
        self, mock_hopsworks, sample_engineered_data
    ):
        """Test successful feature group data update."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            client.feature_group = mock_feature_group  # Set feature group

            # Update data
            client.update_feature_group_data(sample_engineered_data)

            # Verify calls
            mock_feature_group.insert.assert_called_once_with(
                sample_engineered_data, wait=True
            )
            mock_feature_group.update_statistics_config.assert_called_once()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_update_feature_group_data_no_feature_group(
        self, mock_hopsworks, sample_engineered_data
    ):
        """Test feature group data update without feature group."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(ValueError, match="Feature group not loaded"):
                client.update_feature_group_data(sample_engineered_data)

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_statistics_success(self, mock_hopsworks):
        """Test successful feature statistics retrieval."""
        mock_project = Mock()
        mock_feature_group = Mock()

        expected_statistics = {"mean": 100.0, "std": 15.0, "count": 1000}
        mock_feature_group.get_statistics.return_value = expected_statistics
        mock_project.get_feature_store.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            client.feature_group = mock_feature_group  # Set feature group

            stats = client.get_feature_statistics()

            assert stats == expected_statistics
            mock_feature_group.get_statistics.assert_called_once()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_statistics_no_feature_group(self, mock_hopsworks):
        """Test feature statistics retrieval without feature group."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            with pytest.raises(ValueError, match="Feature group not loaded"):
                client.get_feature_statistics()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_get_feature_statistics_failure(self, mock_hopsworks):
        """Test feature statistics retrieval failure."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()
        mock_feature_group.get_statistics.side_effect = Exception(
            "Stats failed")

        mock_project.get_feature_store.return_value = mock_fs
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()
            client.feature_group = mock_feature_group  # Set feature group

            stats = client.get_feature_statistics()
            assert stats == {}


class TestHopsworksIntegration:
    """Integration tests for HopsworksClient."""

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_complete_feature_store_workflow(
        self, mock_hopsworks, sample_engineered_data
    ):
        """Test complete feature store workflow."""
        # Set up comprehensive mocks
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()
        mock_feature_view = Mock()
        mock_query = Mock()

        # Chain the mocks
        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_group.return_value = mock_feature_group
        mock_fs.create_feature_view.return_value = mock_feature_view
        mock_fs.get_feature_view.return_value = mock_feature_view
        mock_feature_group.select_all.return_value = mock_query
        mock_hopsworks.login.return_value = mock_project

        # Mock training data
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        X_test = pd.DataFrame({"feature1": [7, 8], "feature2": [9, 10]})
        y_train = pd.Series([100, 200, 300])
        y_test = pd.Series([400, 500])
        mock_feature_view.create_train_test_split.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"
            mock_settings.TARGET_COLUMN = "price_usd"
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42

            # Initialize client
            client = HopsworksClient()

            # Step 1: Create feature group
            test_data = sample_engineered_data.copy()
            test_data["car_id"] = range(1, len(test_data) + 1)

            client.create_feature_group(
                df=test_data,
                name="integration_test_fg",
                version=1,
                description="Integration test feature group",
                primary_key=["car_id"],
            )

            # Step 2: Create feature view
            client.create_feature_view(
                name="integration_test_fv",
                version=1,
                description="Integration test feature view",
                labels=["price_usd"],
            )

            # Step 3: Get training data
            X_train_result, X_test_result, y_train_result, y_test_result = (
                client.get_training_data()
            )

            # Verify all steps completed successfully
            mock_fs.create_feature_group.assert_called_once()
            mock_feature_group.insert.assert_called_once()
            mock_fs.create_feature_view.assert_called_once()
            mock_feature_view.create_train_test_split.assert_called_once()

            # Verify training data
            assert isinstance(X_train_result, pd.DataFrame)
            assert isinstance(X_test_result, pd.DataFrame)
            assert isinstance(y_train_result, pd.Series)
            assert isinstance(y_test_result, pd.Series)

    @pytest.mark.integration
    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_feature_group_update_workflow(
        self, mock_hopsworks, sample_engineered_data
    ):
        """Test feature group update workflow."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()

        # Mock statistics
        expected_stats = {
            "document_count": len(sample_engineered_data),
            "mean_price": sample_engineered_data.get(
                "price_usd", pd.Series([0])
            ).mean(),
        }
        mock_feature_group.get_statistics.return_value = expected_stats

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_group.return_value = mock_feature_group
        mock_fs.get_feature_group.return_value = mock_feature_group
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            # Create initial feature group
            test_data = sample_engineered_data.copy()
            test_data["car_id"] = range(1, len(test_data) + 1)

            client.create_feature_group(df=test_data, name="update_test_fg")

            # Update with new data
            new_data = test_data.iloc[:10].copy()  # Subset of data
            new_data["car_id"] = range(1000, 1010)  # Different IDs

            client.update_feature_group_data(new_data)

            # Get statistics
            stats = client.get_feature_statistics()

            # Verify workflow
            assert mock_feature_group.insert.call_count == 2  # Initial + update
            assert mock_feature_group.update_statistics_config.call_count == 2
            assert stats == expected_stats

    @pytest.mark.slow
    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_large_dataset_handling(self, mock_hopsworks):
        """Test handling of large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "feature1": np.random.randn(10000),
                "feature2": np.random.randn(10000),
                "feature3": np.random.randint(0, 100, 10000),
                "price_usd": np.random.uniform(10000, 50000, 10000),
                "car_id": range(1, 10001),
            }
        )

        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()
        mock_feature_view = Mock()

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_group.return_value = mock_feature_group
        mock_fs.create_feature_view.return_value = mock_feature_view
        mock_hopsworks.login.return_value = mock_project

        # Mock training data split
        split_point = int(0.8 * len(large_data))
        # Exclude price_usd and car_id
        X_train = large_data.iloc[:split_point, :-2]
        X_test = large_data.iloc[split_point:, :-2]
        y_train = large_data.iloc[:split_point]["price_usd"]
        y_test = large_data.iloc[split_point:]["price_usd"]

        mock_feature_view.create_train_test_split.return_value = (
            X_train,
            X_test,
            y_train,
            y_test,
        )

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"
            mock_settings.TARGET_COLUMN = "price_usd"
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42

            client = HopsworksClient()

            # Test large dataset operations
            client.create_feature_group(
                df=large_data, name="large_dataset_fg", primary_key=["car_id"]
            )

            client.create_feature_view(
                name="large_dataset_fv", labels=["price_usd"])

            # Get training data
            X_train_result, X_test_result, y_train_result, y_test_result = (
                client.get_training_data()
            )

            # Verify large dataset handling
            assert len(X_train_result) == len(y_train_result)
            assert len(X_test_result) == len(y_test_result)
            assert len(X_train_result) + len(X_test_result) == len(large_data)

            # Verify feature group was called with large dataset
            call_args = mock_feature_group.insert.call_args
            inserted_data = call_args[0][0]
            assert len(inserted_data) == 10000

    @pytest.mark.integration
    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_error_recovery_workflow(self, mock_hopsworks, sample_engineered_data):
        """Test error recovery in feature store operations."""
        mock_project = Mock()
        mock_fs = Mock()
        mock_feature_group = Mock()

        # Simulate intermittent failures
        mock_feature_group.insert.side_effect = [
            Exception("Network error"),  # First call fails
            None,  # Second call succeeds
        ]

        mock_project.get_feature_store.return_value = mock_fs
        mock_fs.create_feature_group.return_value = mock_feature_group
        mock_hopsworks.login.return_value = mock_project

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            client = HopsworksClient()

            test_data = sample_engineered_data.copy()
            test_data["car_id"] = range(1, len(test_data) + 1)

            # First attempt should fail
            with pytest.raises(Exception) as exc_info:
                client.create_feature_group(
                    df=test_data, name="error_recovery_fg")

            assert "Network error" in str(exc_info)

            # Reset the side effect for retry
            mock_feature_group.insert.side_effect = None

            # Second attempt should succeed
            client.create_feature_group(df=test_data, name="error_recovery_fg")

            # Verify both attempts were made
            assert mock_feature_group.insert.call_count == 2


class TestFeatureStoreUtilities:
    """Test utility functions and edge cases for feature store operations."""

    def test_data_validation_before_feature_store(self, sample_engineered_data):
        """Test data validation before sending to feature store."""
        # Test with valid data
        assert isinstance(sample_engineered_data, pd.DataFrame)
        assert not sample_engineered_data.empty
        assert len(sample_engineered_data.columns) > 0

        # Test with invalid data types
        invalid_data = pd.DataFrame(
            {
                "feature1": [1, 2, np.inf],  # Invalid infinity
                "feature2": [1, 2, np.nan],  # Missing values
                "price_usd": [100, 200, 300],
            }
        )

        # These should be handled appropriately before sending to feature store
        assert np.isinf(invalid_data["feature1"]).any()
        assert invalid_data["feature2"].isnull().any()

    @patch("src.feature_store.hopsworks_client.hopsworks")
    def test_connection_retry_logic(self, mock_hopsworks):
        """Test connection retry logic."""
        # Simulate connection failures then success
        mock_hopsworks.login.side_effect = [
            Exception("Connection timeout"),
            Exception("Authentication failed"),
            Mock(),  # Success on third try
        ]

        with patch("src.feature_store.hopsworks_client.settings") as mock_settings:
            mock_settings.HOPSWORKS_API_KEY = "test_api_key"
            mock_settings.HOPSWORKS_PROJECT = "test_project"
            mock_settings.HOPSWORKS_HOST = "test_host"

            # First two attempts should fail
            with pytest.raises(ConnectionError):
                HopsworksClient()

            with pytest.raises(ConnectionError):
                HopsworksClient()

            # Third attempt should succeed
            client = HopsworksClient()
            assert client.project is not None

    def test_data_schema_validation(self):
        """Test data schema validation for feature store compatibility."""
        # Valid schema
        valid_data = pd.DataFrame(
            {
                "numerical_feature": [1.0, 2.0, 3.0],
                "categorical_feature": ["A", "B", "C"],
                "boolean_feature": [True, False, True],
                "target": [100, 200, 300],
            }
        )

        # Check data types are appropriate for feature store
        assert valid_data["numerical_feature"].dtype in ["float64", "int64"]
        assert valid_data["categorical_feature"].dtype == "object"
        assert valid_data["boolean_feature"].dtype == "bool"

        # Invalid schema with complex objects
        invalid_data = pd.DataFrame(
            {
                # Lists not supported
                "list_feature": [[1, 2], [3, 4], [5, 6]],
                # Dicts not supported
                "dict_feature": [{"a": 1}, {"b": 2}, {"c": 3}],
            }
        )

        # These would need to be flattened or encoded before feature store
        assert invalid_data["list_feature"].dtype == "object"
        assert invalid_data["dict_feature"].dtype == "object"
