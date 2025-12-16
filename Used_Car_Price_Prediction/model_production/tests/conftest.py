"""Pytest configuration file with fixtures and test setup."""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "price_usd": np.random.uniform(10000, 50000, n_samples),
        "make_year": np.random.randint(2000, 2025, n_samples),
        "mileage_kmpl": np.random.normal(15, 3, n_samples).clip(8, 30),
        "engine_cc": np.random.randint(1000, 3000, n_samples),
        "fuel_type": np.random.choice(["Petrol", "Diesel", "Electric"], n_samples),
        "owner_count": np.random.randint(1, 4, n_samples),
        "brand": np.random.choice(
            [
                "Honda",
                "Toyota",
                "Ford",
                "Chevrolet",
                "BMW",
                "Volkswagen",
                "Tesla",
                "Hyundai",
                "Kia",
                "Nissan",
            ],
            n_samples,
        ),
        "transmission": np.random.choice(["Manual", "Automatic"], n_samples),
        "color": np.random.choice(
            ["White", "Black", "Silver", "Red", "Gray", "Blue"], n_samples
        ),
        "insurance_valid": np.random.choice(["Yes", "No"], n_samples),
        "service_history": np.random.choice(["Complete", "Partial", "None"], n_samples),
        "accidents_reported": np.random.randint(0, 5, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_cleaned_data():
    """Create sample cleaned data for testing."""
    np.random.seed(42)
    n_samples = 95

    data = {
        "price_usd": np.random.uniform(10000, 50000, n_samples),
        "make_year": np.random.randint(2000, 2025, n_samples),
        "mileage_kmpl": np.random.normal(15, 3, n_samples).clip(8, 30),
        "engine_cc": np.random.randint(1000, 3000, n_samples),
        "fuel_type": np.random.choice(["Petrol", "Diesel", "Electric"], n_samples),
        "owner_count": np.random.randint(1, 4, n_samples),
        "brand": np.random.choice(
            [
                "Honda",
                "Toyota",
                "Ford",
                "Chevrolet",
                "BMW",
                "Volkswagen",
                "Tesla",
                "Hyundai",
                "Kia",
                "Nissan",
            ],
            n_samples,
        ),
        "transmission": np.random.choice(["Manual", "Automatic"], n_samples),
        "color": np.random.choice(
            ["White", "Black", "Silver", "Red", "Gray", "Blue"], n_samples
        ),
        "insurance_valid": np.random.choice(["Yes", "No"], n_samples),
        "service_history": np.random.choice(["Complete", "Partial", "None"], n_samples),
        "accidents_reported": np.random.randint(0, 4, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_engineered_data():
    """Create sample feature engineered data for testing."""
    np.random.seed(42)
    n_samples = 95

    data = {
        "price_usd": np.random.uniform(10000, 50000, n_samples),
        "make_year": np.random.randint(2000, 2024, n_samples),
        "car_age": np.random.randint(0, 24, n_samples),
        "has_accident": np.random.choice([0, 1], n_samples),
        "service_history_encoded": np.random.choice([0, 1, 2], n_samples),
        # One-hot encoded features
        "fuel_type_Diesel": np.random.choice([0, 1], n_samples),
        "fuel_type_Electric": np.random.choice([0, 1], n_samples),
        "fuel_type_Petrol": np.random.choice([0, 1], n_samples),
        "brand_BMW": np.random.choice([0, 1], n_samples),
        "brand_Honda": np.random.choice([0, 1], n_samples),
        "brand_Ford": np.random.choice([0, 1], n_samples),
        "brand_Toyota": np.random.choice([0, 1], n_samples),
        "brand_Chevrolet": np.random.choice([0, 1], n_samples),
        "brand_Volkswagen": np.random.choice([0, 1], n_samples),
        "brand_Tesla": np.random.choice([0, 1], n_samples),
        "brand_Hyundai": np.random.choice([0, 1], n_samples),
        "brand_Kia": np.random.choice([0, 1], n_samples),
        "transmission_Manual": np.random.choice([0, 1], n_samples),
        "color_Black": np.random.choice([0, 1], n_samples),
        "color_Red": np.random.choice([0, 1], n_samples),
        "color_White": np.random.choice([0, 1], n_samples),
        "color_Silver": np.random.choice([0, 1], n_samples),
        "color_Gray": np.random.choice([0, 1], n_samples),
        "insurance_valid_Yes": np.random.choice([0, 1], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_mongodb_config():
    """Mock MongoDB configuration."""
    mock_config = Mock()
    mock_config.connection_string = "mongodb://localhost:27017"
    mock_config.database_name = "test_db"
    mock_config.collection_name = "test_collection"
    mock_config.get_collection.return_value = Mock()
    mock_config.test_connection.return_value = True
    return mock_config


@pytest.fixture
def mock_hopsworks_project():
    """Mock Hopsworks project and feature store."""
    mock_project = Mock()
    mock_fs = Mock()
    mock_feature_group = Mock()
    mock_feature_view = Mock()

    # Setup mock chain
    mock_project.get_feature_store.return_value = mock_fs
    mock_fs.create_feature_group.return_value = mock_feature_group
    mock_fs.create_feature_view.return_value = mock_feature_view
    mock_fs.get_feature_group.return_value = mock_feature_group
    mock_fs.get_feature_view.return_value = mock_feature_view

    # Mock feature group methods
    mock_feature_group.insert.return_value = None
    mock_feature_group.update_statistics_config.return_value = None
    mock_feature_group.select_all.return_value = Mock()
    mock_feature_group.get_statistics.return_value = {"count": 100}

    # Mock feature view methods
    mock_feature_view.create_train_test_split.return_value = (
        pd.DataFrame({"feature1": [1, 2, 3]}),  # X_train
        pd.DataFrame({"feature1": [4, 5]}),  # X_test
        pd.Series([100, 200, 300]),  # y_train
        pd.Series([400, 500]),  # y_test
    )

    return mock_project, mock_fs, mock_feature_group, mock_feature_view


@pytest.fixture
def mock_settings():
    """Mock settings configuration."""
    mock_settings = Mock()
    mock_settings.data_dir = Path("/tmp/test_data")
    mock_settings.CURRENT_YEAR = 2025
    mock_settings.TARGET_COLUMN = "price_usd"
    mock_settings.TEST_SIZE = 0.2
    mock_settings.RANDOM_STATE = 42
    mock_settings.MONGODB_URL = "mongodb://localhost:27017"
    mock_settings.MONGODB_NAME = "test_db"
    mock_settings.MONGODB_COLLECTION = "test_collection"
    mock_settings.HOPSWORKS_API_KEY = "test_api_key"
    mock_settings.HOPSWORKS_PROJECT = "test_project"
    mock_settings.HOPSWORKS_HOST = "c.app.hopsworks.ai"
    mock_settings.MLFLOW_TRACKING_URI = "http://localhost:5000"
    mock_settings.MLFLOW_EXPERIMENT_NAME = "test_experiment"
    mock_settings.LOG_LEVEL = "INFO"
    mock_settings.LOG_FILE = "logs/test.log"
    mock_settings.SERVICE_HISTORY_MAPPING = {
        "None": 0, "Partial": 1, "Complete": 2}
    mock_settings.CATEGORICAL_FEATURES = [
        "fuel_type",
        "brand",
        "transmission",
        "color",
        "insurance_valid",
    ]
    mock_settings.FEATURES_TO_DROP = [
        "make_year",
        "accidents_reported",
        "fuel_type",
        "brand",
        "transmission",
        "color",
        "insurance_valid",
    ]
    return mock_settings


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    return Mock()


@pytest.fixture
def mock_logging_config(mock_logger):
    """Mock LoggingConfig instance."""
    mock_config = Mock()
    mock_config.get_logger.return_value = mock_logger
    return mock_config


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, mock_settings, mock_logging_config):
    """Setup test environment with mocked dependencies."""

    # Mock the entire config module initialization
    monkeypatch.setattr("config.__init__.settings", mock_settings)
    monkeypatch.setattr("config.logging_config", mock_logging_config)

    # Mock the LoggingConfig class constructor
    monkeypatch.setattr(
        "config.logging_config.LoggingConfig",
        lambda **kwargs: mock_logging_config)

    # Mock imports in source module
    monkeypatch.setattr(
        "src.data_ingestion.data_loader.settings",
        mock_settings
    )
    monkeypatch.setattr(
        "src.data_ingestion.mongodb_connector.settings",
        mock_settings
    )

    # Create test data directory
    mock_settings.data_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_dirty_data():
    """Create sample dirty data with various issues for testing cleaning."""
    np.random.seed(42)
    n_samples = 110

    data = {
        "Price USD ": np.concatenate(
            [
                np.random.uniform(10000, 50000, 90),  # Valid prices
                [-1000, -500],  # Invalid negative prices
                [np.nan] * 5,  # Missing values
                [1000000, 2000000] * 6,  # Outliers
                [0, 0, 0],  # Edge cases
            ],
        )[:n_samples],
        " Make Year ": np.concatenate(
            [
                np.random.randint(2000, 2024, 90),
                [1800, 1850],  # Invalid years
                [np.nan] * 5,
                [2030, 2040] * 6,  # Future years
                [1999, 1998, 1997],
            ],
        )[:n_samples],
        "FUEL TYPE": np.concatenate(
            [
                np.random.choice(["Petrol", "Diesel", "Electric"], 90),
                np.array([None, None], dtype=object),
                np.array([np.nan] * 5, dtype=object),
                np.array(["Unknown"] * 12, dtype=object),
                np.array(["petrol"], dtype=object),  # Case issue
            ],
        )[:n_samples],
        "Brand": np.random.choice(
            [
                "Honda",
                "Toyota",
                "Ford",
                "Chevrolet",
                "BMW",
                "Volkswagen",
                "Tesla",
                "Hyundai",
                "Kia",
                "Nissan",
            ],
            n_samples,
        ),
        "transmission": np.random.choice(
            np.array(["Manual", "Automatic", None], dtype=object), n_samples
        ),
        "Color": np.random.choice(
            np.array(["White", "Black", "Silver", "Red",
                     "Blue", None], dtype=object),
            n_samples,
        ),
        "insurance valid": np.random.choice(
            np.array(["Yes", "No", "Y", "N", None], dtype=object), n_samples
        ),
        "Service History": np.random.choice(
            np.array(["Complete", "Partial", "None", None, ""],
                     dtype=object), n_samples
        ),
        "accidents_reported": np.concatenate(
            [
                np.random.randint(0, 5, 100),
                [-1, -2] * 5,  # Invalid negative values
            ],
        )[:n_samples],
    }

    # Add duplicate rows
    df = pd.DataFrame(data)
    df_with_duplicates = pd.concat([df, df.iloc[:5]], ignore_index=True)

    return df_with_duplicates


class TestDataBase:
    """Base class for test data and utilities."""

    @staticmethod
    def assert_dataframe_valid(df, required_columns=None):
        """Assert that dataframe is valid"""
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        if required_columns:
            assert all(col in df.columns for col in required_columns)

    @staticmethod
    def assert_no_missing_values(df, columns=None):
        """Assert no missing values in specified columns"""
        if columns is None:
            columns = df.columns
        for col in columns:
            assert not df[col].isnull().any(
            ), f"Missing values found in column {col}"

    @staticmethod
    def assert_data_types(df, expected_types):
        """Assert expected data types"""
        for col, expected_type in expected_types.items():
            if col in df.columns:
                assert (
                    df[col].dtype == expected_type
                    or str(df[col].dtype) == expected_type
                ), f"Expected {col} to be {expected_type}, but got {df[col].dtype}"


@pytest.fixture
def test_data_base():
    """Provide TestDataBase utilities to tests"""
    return TestDataBase()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark tests based on their location or name
        if "integration" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.name or "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name or "test_complete_pipeline" in item.name:
            item.add_marker(pytest.mark.slow)
