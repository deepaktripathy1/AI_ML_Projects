"""Test data preprocessing components."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.data_cleaner import DataCleaner
from src.data_preprocessing.feature_engineering import FeatureEngineer
from src.data_preprocessing.feature_selection import FeatureSelector
from src.feature_store.feature_pipeline import FeaturePipeline


class TestDataCleaner:
    """Test cases for DataCleaner class."""

    def test_initialization(self):
        """Test data cleaner initialization."""
        cleaner = DataCleaner()
        assert hasattr(cleaner, "cleaning_report")
        assert isinstance(cleaner.cleaning_report, dict)

    def test_clean_data_complete_workflow(self, sample_dirty_data):
        """Test complete data cleaning workflow."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(sample_dirty_data)

        # Basic assertions
        assert isinstance(cleaned_df, pd.DataFrame)
        assert not cleaned_df.empty
        assert len(cleaned_df) <= len(sample_dirty_data)

        # Check cleaning report
        report = cleaner.get_cleaning_report()
        assert "initial_shape" in report
        assert "final_shape" in report
        assert "cleaning_steps" in report
        assert "rows_removed" in report

    def test_clean_column_names(self, sample_dirty_data):
        """Test column name cleaning."""
        cleaner = DataCleaner()
        cleaned_df = cleaner._clean_column_names(sample_dirty_data)

        # Check that column names are standardized
        expected_columns = [
            "price_usd",
            "make_year",
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
            "accidents_reported",
        ]

        # Verify that the columns are cleaned
        for col in cleaned_df.columns:
            assert " " not in col  # No spaces
            assert col.islower()  # lowercase
            assert list(cleaned_df.columns) == expected_columns

    def test_handle_missing_values(self):
        """Test missing value handling."""
        # Create data with missing values
        data_with_nulls = pd.DataFrame(
            {
                "numeric_col": [1, 2, np.nan, 4, 5],
                "categorical_col": ["A", "B", None, "A", "B"],
                "service_history": ["Complete", None, "Partial", "None", "Complete"],
            }
        )

        cleaner = DataCleaner()
        cleaned_df = cleaner._handle_missing_values(data_with_nulls)

        # Check that there are no null values
        assert not cleaned_df.isnull().any().any()

        # Check that nulls in service_history columns are filled with None
        assert "None" in cleaned_df["service_history"].values

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        # Create data with duplicates
        data_with_dups = pd.DataFrame(
            {"col1": [1, 2, 3, 1, 2], "col2": ["A", "B", "C", "A", "B"]}
        )

        cleaner = DataCleaner()
        cleaned_df = cleaner._remove_duplicates(data_with_dups)

        # Check for 3 unique rows
        assert len(cleaned_df) == 3
        assert not cleaned_df.duplicated().any()

    def test_handle_outliers(self):
        """Test outlier handling."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(100, 15, 100)
        outlier_data = np.array([300, 350, -50, -100])
        full_data = np.concatenate([normal_data, outlier_data])

        df_with_outliers = pd.DataFrame(
            {"price_usd": full_data, "other_col": range(len(full_data))}
        )

        cleaner = DataCleaner()
        cleaned_df = cleaner._handle_outliers(df_with_outliers)

        # Check for fewer rows
        assert len(cleaned_df) < len(df_with_outliers)

        # Extreme outliers should be removed
        assert cleaned_df["price_usd"].max() < 300
        assert cleaned_df["price_usd"].min() > -50

    def test_correct_data_types(self):
        """Test data type correction."""
        # Create data with wrong types
        df_wrong_types = pd.DataFrame(
            {
                "make_year": ["2020", "2021", "2019"],  # Should be int
                "accidents_reported": ["0", "1", "2"],  # Should be int
                "fuel_type": [1, 2, 3],  # Should be string
                "brand": [10, 20, 30],  # Should be string
                "transmission": ["Manual", "Auto", "Manual"],
                "color": ["Red", "Blue", "Green"],
                "insurance_valid": ["Yes", "No", "Yes"],
                "service_history": ["Complete", "None", "Partial"],
            }
        )

        cleaner = DataCleaner()
        correct_df = cleaner._correct_data_types(df_wrong_types)

        # Check integer columns
        assert correct_df["make_year"].dtype in ["int64", "Int64"]
        assert correct_df["accidents_reported"].dtype in ["int64", "Int64"]

        # Check categorical columns are objects
        categorical_cols = [
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
        ]
        for col in categorical_cols:
            if col in correct_df.columns:
                assert (
                    correct_df[col].dtype == "object"
                    or str(correct_df[col].dtype) == "string"
                )

    @patch("src.data_preprocessing.data_cleaner.settings")
    def test_apply_business_validations(self, mock_settings):
        """Test business validation rules."""
        mock_settings.CURRENT_YEAR = 2025

        # Create data with business rule violations
        invalid_data = pd.DataFrame(
            {
                "price_usd": [10000, -5000, 30000, 0, 25000],  # Negative price
                "make_year": [2020, 1850, 2025, 2022, 2010],  # Invalid years
                # Negative accidents
                "accidents_reported": [0, 1, -1, 2, 3],
                "other_col": range(5),
            }
        )

        cleaner = DataCleaner()
        validated_df = cleaner._apply_business_validations(invalid_data)

        # Should have fewer rows due to validation
        assert len(validated_df) < len(invalid_data)

        # Check remaining data follows business rules
        if "price_usd" in validated_df.columns:
            assert all(validated_df["price_usd"] >= 0)
        if "make_year" in validated_df.columns:
            assert all(validated_df["make_year"] >= 1900)
            assert all(validated_df["make_year"] <= 2024)
        if "accidents_reported" in validated_df.columns:
            assert all(validated_df["accidents_reported"] >= 0)

    def test_get_cleaning_report(self, sample_dirty_data):
        """Test cleaning report generation."""
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(sample_dirty_data)
        report = cleaner.get_cleaning_report()

        # Check report structure
        required_keys = [
            "initial_shape",
            "final_shape",
            "cleaning_steps",
            "rows_removed",
            "cleaning_percentage",
        ]
        for key in required_keys:
            assert key in report

        # Check data types
        assert isinstance(report["cleaning_steps"], list)
        assert isinstance(report["rows_removed"], int)
        assert isinstance(report["cleaning_percentage"], float)

        assert cleaned_df.shape == report["final_shape"]


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def test_initialization(self):
        engineer = FeatureEngineer()
        assert hasattr(engineer, "feature_engineering_report")
        assert hasattr(engineer, "encoders")

    @patch("src.data_preprocessing.feature_engineering.settings")
    def test_engineer_features_complete_workflow(
        self, mock_settings, sample_cleaned_data
    ):
        """Test complete feature engineering workflow."""
        # Setup mock settings
        mock_settings.CURRENT_YEAR = 2025
        mock_settings.SERVICE_HISTORY_MAPPING = {
            "None": 0, "Partial": 1, "Complete": 2}
        mock_settings.CATEGORICAL_FEATURES = [
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
        ]
        mock_settings.FEATURES_TO_DROP = ["make_year"]

        engineer = FeatureEngineer()
        engineered_df = engineer.engineer_features(sample_cleaned_data)

        # Basic assertions
        assert isinstance(engineered_df, pd.DataFrame)
        assert not engineered_df.empty
        assert len(engineered_df.columns) >= len(sample_cleaned_data.columns)

        # Check new features are created
        expected_new_features = ["car_age",
                                 "has_accident", "service_history_encoded"]
        for feature in expected_new_features:
            if "make_year" in sample_cleaned_data.columns and feature == "car_age":
                assert feature in engineered_df.columns
            if (
                "accidents_reported" in sample_cleaned_data.columns
                and feature == "has_accident"
            ):
                assert feature in engineered_df.columns

    @patch("src.data_preprocessing.feature_engineering.settings")
    def test_create_car_age(self, mock_settings):
        """Test car age feature creation."""
        mock_settings.CURRENT_YEAR = 2025

        df_with_year = pd.DataFrame(
            {
                "make_year": [2020, 2018, 2022, 2010],
                "price_usd": [25000, 20000, 35000, 15000],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._create_car_age(df_with_year)

        assert "car_age" in result_df.columns
        expected_ages = [5, 7, 3, 15]
        assert list(result_df["car_age"]) == expected_ages

    def test_create_has_accident(self):
        """Test has_accident feature creation."""
        df_with_accidents = pd.DataFrame(
            {
                "accidents_reported": [0, 1, 3, 0, 2],
                "price_usd": [25000, 20000, 15000, 30000, 18000],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._create_has_accident(df_with_accidents)

        assert "has_accident" in result_df.columns
        expected_has_accident = [0, 1, 1, 0, 1]
        assert list(result_df["has_accident"]) == expected_has_accident

    @patch("src.data_preprocessing.feature_engineering.settings")
    def test_encode_service_history(self, mock_settings):
        """Test service history encoding."""
        mock_settings.SERVICE_HISTORY_MAPPING = {
            "None": 0, "Partial": 1, "Complete": 2}

        df_with_service = pd.DataFrame(
            {
                "service_history": ["Complete", "None", "Partial", "Complete"],
                "price_usd": [25000, 20000, 22000, 28000],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._encode_service_history(df_with_service)

        assert "service_history_encoded" in result_df.columns
        expected_encoded = [2, 0, 1, 2]
        assert list(result_df["service_history_encoded"]) == expected_encoded

    @patch("src.data_preprocessing.feature_engineering.settings")
    def test_encode_categorical_features(self, mock_settings):
        """Test one-hot encoding of categorical features"""
        mock_settings.CATEGORICAL_FEATURES = ["fuel_type", "brand"]

        df_categorical = pd.DataFrame(
            {
                "fuel_type": ["Petrol", "Diesel", "Petrol", "Electric"],
                "brand": ["Toyota", "BMW", "Toyota", "Tesla"],
                "price_usd": [25000, 35000, 24000, 45000],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._encode_categorical_features(df_categorical)

        # Check that new columns are created
        expected_new_cols = [
            "fuel_type_Diesel",
            "fuel_type_Electric",
            "fuel_type_Petrol",
            "brand_BMW",
            "brand_Tesla",
            "brand_Toyota",
        ]

        # At least some of these should be present (depending on drop_first)
        new_cols = [
            col for col in result_df.columns if col not in df_categorical.columns
        ]
        assert len(new_cols) > 0
        assert len(new_cols) < len(expected_new_cols)

    @patch("src.data_preprocessing.feature_engineering.settings")
    def test_drop_unnecessary_columns(self, mock_settings):
        """Test dropping unnecessary columns."""
        mock_settings.FEATURES_TO_DROP = ["unnecessary_col"]

        df_with_unnecessary = pd.DataFrame(
            {
                "price_usd": [25000, 30000],
                "important_col": [1, 2],
                "unnecessary_col": [10, 20],
                "service_history": ["Complete", "Partial"],
                "service_history_encoded": [2, 1],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._drop_unnecessary_columns(df_with_unnecessary)

        # Should drop unnecessary_col and service_history (since encoded version exists)
        assert "unnecessary_col" not in result_df.columns
        assert "service_history" not in result_df.columns
        assert "service_history_encoded" in result_df.columns

    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        df_for_interaction = pd.DataFrame(
            {
                "car_age": [2, 5, 10, 15],
                "has_accident": [0, 1, 1, 0],
                "price_usd": [30000, 25000, 18000, 12000],
            }
        )

        engineer = FeatureEngineer()
        result_df = engineer._create_interaction_features(df_for_interaction)

        # Should create age_group and accident_age_interaction
        assert "age_group" in result_df.columns
        assert "accident_age_interaction" in result_df.columns

        # Check interaction calculation
        expected_interaction = [0, 5, 10, 0]  # has_accident * car_age
        assert list(result_df["accident_age_interaction"]
                    ) == expected_interaction

    def test_get_feature_importance_data(self, sample_engineered_data):
        """Test feature importance data extraction"""
        engineer = FeatureEngineer()
        importance_data = engineer.get_feature_importance_data(
            sample_engineered_data)

        required_keys = [
            "numerical_features",
            "categorical_features",
            "total_features",
            "feature_types",
        ]
        for key in required_keys:
            assert key in importance_data

        assert isinstance(importance_data["numerical_features"], list)
        assert isinstance(importance_data["categorical_features"], list)
        assert importance_data["total_features"] == len(
            sample_engineered_data.columns)


class TestFeatureSelector:
    """Test cases for FeatureSelector class."""

    def test_initialization(self):
        """Test feature selector initialization."""
        selector = FeatureSelector(target_column="price_usd")
        assert selector.target_column == "price_usd"
        assert hasattr(selector, "selection_report")
        assert hasattr(selector, "selected_features")

    def test_prepare_data(self, sample_engineered_data):
        """Test data preparation for feature selection"""
        selector = FeatureSelector(target_column="price_usd")
        X, y = selector._prepare_data(sample_engineered_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert "price_usd" not in X.columns
        assert y.name == "price_usd" or len(y) > 0

    def test_prepare_data_missing_target(self, sample_engineered_data):
        """Test data preparation with missing target column."""
        selector = FeatureSelector(target_column="missing_column")

        with pytest.raises(ValueError, match="Target column missing_column not found"):
            selector._prepare_data(sample_engineered_data)

    @patch("src.data_preprocessing.feature_selection.RFECV")
    def test_rfecv_selection(self, mock_rfecv_class, sample_engineered_data):
        """Test RFECV feature selection"""
        # Mock RFECV
        mock_rfecv = Mock()
        mock_rfecv.support_ = np.array(
            [True, False, True, False, True]
            + [False] * (len(sample_engineered_data.columns) - 6)
        )
        mock_rfecv.ranking_ = np.arange(1, len(sample_engineered_data.columns))
        mock_rfecv.n_features_ = 3
        mock_rfecv.cv_results_ = {"mean_test_score": [0.8, 0.85, 0.82]}
        mock_rfecv_class.return_value = mock_rfecv

        selector = FeatureSelector(target_column="price_usd")
        X, y = selector._prepare_data(sample_engineered_data)
        selected_features = selector._rfecv_selection(X, y, cv_folds=5)

        assert isinstance(selected_features, list)
        assert len(selected_features) > 0
        mock_rfecv.fit.assert_called_once()

    def test_correlation_selection(self, sample_engineered_data):
        """Test correlation-based feature selection."""
        selector = FeatureSelector(target_column="price_usd")
        X, y = selector._prepare_data(sample_engineered_data)
        selected_features = selector._correlation_selection(
            X, y, threshold=0.01)

        assert isinstance(selected_features, list)
        # Should select features with some correlation
        assert len(selected_features) >= 1

    def test_remove_multicollinear_features(self):
        """Test multicollinear feature removal."""
        # Create data with high correlation
        np.random.seed(42)
        feature1 = np.random.normal(0, 1, 100)
        # Highly correlated
        feature2 = feature1 + np.random.normal(0, 0.1, 100)
        feature3 = np.random.normal(0, 1, 100)  # Independent

        correlated_df = pd.DataFrame(
            {"feature1": feature1, "feature2": feature2, "feature3": feature3}
        )

        selector = FeatureSelector(target_column="price_usd")
        selected_features = selector._remove_multicollinear_features(
            correlated_df, threshold=0.8
        )

        # Should remove one of the highly correlated features
        assert len(selected_features) == 2
        assert "feature3" in selected_features  # Independent feature should remain

    def test_get_feature_importance_scores(self, sample_engineered_data):
        """Test feature importance score calculation."""
        selector = FeatureSelector(target_column="price_usd")
        X, y = selector._prepare_data(sample_engineered_data)
        importance_scores = selector.get_feature_importance_scores(X, y)

        assert isinstance(importance_scores, dict)
        assert len(importance_scores) == len(X.columns)
        # Check all scores are non-negative
        assert all(score >= 0 for score in importance_scores.values())

    def test_select_features_rfecv_method(self, sample_engineered_data):
        """Test complete feature selection with RFECV method."""
        with patch.object(FeatureSelector, "_rfecv_selection") as mock_rfecv:
            mock_rfecv.return_value = ["car_age",
                                       "insurance_valid_Yes", "service_history_encoded"]

            selector = FeatureSelector(target_column="price_usd")
            selected_df, selected_features = selector.select_features(
                sample_engineered_data, method="rfecv"
            )

            assert isinstance(selected_df, pd.DataFrame)
            assert isinstance(selected_features, list)
            assert len(selected_features) == 3
            assert "price_usd" in selected_df.columns  # Target should be included
            mock_rfecv.assert_called_once()

    def test_select_features_invalid_method(self, sample_engineered_data):
        """Test feature selection with invalid method."""
        selector = FeatureSelector(target_column="price_usd")

        with pytest.raises(ValueError, match="Unsupported feature selection method"):
            selector.select_features(
                sample_engineered_data, method="invalid_method")

    def test_get_selection_report(self, sample_engineered_data):
        """Test selection report generation."""
        with patch.object(FeatureSelector, "_rfecv_selection") as mock_rfecv:
            mock_rfecv.return_value = ["car_age",
                                       "insurance_valid_Yes", "service_history_encoded"]

            selector = FeatureSelector(target_column="price_usd")
            selected_df, selected_features = selector.select_features(
                sample_engineered_data, method="rfecv"
            )

            report = selector.get_selection_report()

            required_keys = [
                "method",
                "initial_features_count",
                "final_features_count",
                "features_removed",
                "reduction_percentage",
            ]
            for key in required_keys:
                assert key in report

            assert isinstance(report["reduction_percentage"], float)
            assert report["final_features_count"] == len(selected_features)


class TestIntegrationPreprocessing:
    """Integration tests for preprocessing components."""

    @pytest.mark.integration
    def test_data_cleaning_to_feature_engineering_integration(self, sample_dirty_data):
        """Test integration between data cleaning and feature engineering."""
        # Clean data first
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(sample_dirty_data)

        # Then engineer features
        with patch(
            "src.data_preprocessing.feature_engineering.settings"
        ) as mock_settings:
            mock_settings.CURRENT_YEAR = 2025
            mock_settings.SERVICE_HISTORY_MAPPING = {
                "None": 0,
                "Partial": 1,
                "Complete": 2,
            }
            mock_settings.CATEGORICAL_FEATURES = [
                "fuel_type",
                "brand",
                "transmission",
                "color",
                "insurance_valid",
            ]
            mock_settings.FEATURES_TO_DROP = ["make_year"]

            engineer = FeatureEngineer()
            engineered_df = engineer.engineer_features(cleaned_df)

            # Verify the flow worked
            assert isinstance(engineered_df, pd.DataFrame)
            assert not engineered_df.empty
            # May have fewer rows after feature engineering
            assert len(engineered_df) <= len(cleaned_df)

    @pytest.mark.integration
    def test_feature_engineering_to_selection_integration(self, sample_cleaned_data):
        """Test integration between feature engineering and selection."""
        # Engineer features first
        with patch(
            "src.data_preprocessing.feature_engineering.settings"
        ) as mock_fe_settings:
            mock_fe_settings.CURRENT_YEAR = 2025
            mock_fe_settings.SERVICE_HISTORY_MAPPING = {
                "None": 0,
                "Partial": 1,
                "Complete": 2,
            }
            mock_fe_settings.CATEGORICAL_FEATURES = [
                "fuel_type",
                "brand",
                "transmission",
                "color",
                "insurance_valid",
            ]
            mock_fe_settings.FEATURES_TO_DROP = ["make_year"]

            engineer = FeatureEngineer()
            engineered_df = engineer.engineer_features(sample_cleaned_data)

            # Then select features
            with patch.object(FeatureSelector, "_rfecv_selection") as mock_rfecv:
                # Mock selection to return some features
                available_features = [
                    col for col in engineered_df.columns if col != "price_usd"
                ]
                # Select first 3 features
                mock_rfecv.return_value = available_features[:3]

                selector = FeatureSelector(target_column="price_usd")
                selected_df, selected_features = selector.select_features(
                    engineered_df, method="rfecv"
                )

                # Verify the flow worked
                assert isinstance(selected_df, pd.DataFrame)
                assert isinstance(selected_features, list)
                assert "price_usd" in selected_df.columns
                assert (
                    len(selected_features) <= len(engineered_df.columns) - 1
                )  # Excluding target

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_preprocessing_workflow(self, sample_dirty_data):
        """Test complete preprocessing workflow from dirty data to selected features."""
        # Step 1: Clean data
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(sample_dirty_data)

        # Step 2: Engineer features
        with patch(
            "src.data_preprocessing.feature_engineering.settings"
        ) as mock_fe_settings:
            mock_fe_settings.CURRENT_YEAR = 2025
            mock_fe_settings.SERVICE_HISTORY_MAPPING = {
                "None": 0,
                "Partial": 1,
                "Complete": 2,
            }
            mock_fe_settings.CATEGORICAL_FEATURES = [
                "fuel_type",
                "brand",
                "transmission",
                "color",
                "insurance_valid",
            ]
            mock_fe_settings.FEATURES_TO_DROP = ["make_year"]

            engineer = FeatureEngineer()
            engineered_df = engineer.engineer_features(cleaned_df)

        # Step 3: Select features
        if (
            "price_usd" not in engineered_df.columns
            and "price" in engineered_df.columns
        ):
            # Ensure target exists
            engineered_df["price_usd"] = engineered_df["price"]

        with patch.object(FeatureSelector, "_correlation_selection") as mock_corr:
            # Mock correlation selection
            available_features = [
                col for col in engineered_df.columns if col != "price_usd"
            ]
            # Select first 5 features
            mock_corr.return_value = available_features[:5]

            selector = FeatureSelector(target_column="price_usd")
            final_df, selected_features = selector.select_features(
                engineered_df, method="correlation"
            )

        # Verify end-to-end workflow
        assert isinstance(final_df, pd.DataFrame)
        assert not final_df.empty
        # Should be smaller due to cleaning
        assert len(final_df) <= len(sample_dirty_data)
        assert "price_usd" in final_df.columns
        assert len(selected_features) > 0

        # Verify reports are available
        cleaning_report = cleaner.get_cleaning_report()
        feature_report = engineer.get_feature_report()
        selection_report = selector.get_selection_report()

        assert isinstance(cleaning_report, dict)
        assert isinstance(feature_report, dict)
        assert isinstance(selection_report, dict)

        # Each report should contain required keys
        assert "initial_shape" in cleaning_report
        assert "final_shape" in cleaning_report
        assert "cleaning_steps" in cleaning_report
