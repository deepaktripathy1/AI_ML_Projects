"""Data Cleaning pipeline."""

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from janitor.functions import clean_names

from config.config import get_settings
from config.logging_config import LoggingConfig


logger = LoggingConfig().get_logger(__name__)


class DataCleaner:
    """Data cleaning pipeline."""

    def __init__(self):
        """Initialize data cleaner."""
        self.cleaning_report: dict[str, Any] = {
            "cleaning_steps": []
        }
        logger.info("Data cleaner initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and return cleaned dataframe.

        Args:
            df: Raw dataframe to be cleaned

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Cleaning data")
        initial_shape = df.shape

        # Create a copy of the dataframe
        df_clean = df.copy()

        # Cleaning report
        self.cleaning_report = {
            "initial_shape": initial_shape,
            "cleaning_steps": [],
        }

        # Clean column names
        df_clean = self._clean_column_names(df_clean)

        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)

        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)

        # Handle outliers
        df_clean = self._handle_outliers(df_clean)

        # Correct column data types
        df_clean = self._correct_data_types(df_clean)

        # Business logic validations
        df_clean = self._apply_business_validations(df_clean)

        # Finalize cleaning report
        self.cleaning_report["final_shape"] = df_clean.shape
        self.cleaning_report["rows_removed"] = initial_shape[0] - \
            df_clean.shape[0]
        self.cleaning_report["cleaning_percentage"] = (
            self.cleaning_report["rows_removed"] / initial_shape[0] * 100
        )

        logger.info(
            "Data cleaning completed. Shape : {initial_shape} -> {df_clean.shape}"
        )
        return df_clean

    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names using pyjanitor."""
        df.columns = df.columns.str.strip()
        df_clean = clean_names(df=df)

        step_info = {
            "step": "clean_column_names",
            "action": "Standardize column names",
            "columns_before": df.columns.tolist(),
            "columns_after": df_clean.columns.tolist(),
        }
        self.cleaning_report["cleaning_steps"].append(step_info)

        logger.info("Column names cleaned and standardized")
        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values."""
        initial_nulls = df.isnull().sum().sum()

        # Fill null values in service_history with None
        if "service_history" in df.columns:
            df["service_history"] = df["service_history"].fillna("None")
            logger.info(
                "Filled out null values in service_history column with None")

        # Handle other missing values based on column type
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ["int64", "float64"]:
                    # For numerical columns, use median
                    df[column] = df[column].fillna(df[column].median())
                    logger.info(
                        f"Filled out null values in {column} column with median: {df[column].median()}"
                    )
                elif df[column].dtype == "object":
                    # For categorical columns, use mode
                    mode_val = df[column].mode()
                    fill_value = mode_val[0] if len(
                        mode_val) > 0 else "Unknown"
                    df[column] = df[column].fillna(fill_value)
                    logger.info(
                        f"Filled out null values in {column} column with mode: {fill_value}"
                    )

        final_nulls = df.isnull().sum().sum()

        step_info = {
            "step": "handle_missing_values",
            "action": f"Handle missing values: {initial_nulls} -> {final_nulls}",
            "nulls_before": initial_nulls,
            "nulls_after": final_nulls,
        }

        self.cleaning_report["cleaning_steps"].append(step_info)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        final_rows = len(df_clean)

        duplicates_removed = initial_rows - final_rows

        step_info = {
            "step": "remove_duplicates",
            "action": f"Remove duplicates: {initial_rows} -> {final_rows}",
            "rows_before": initial_rows,
            "rows_after": final_rows,
        }
        self.cleaning_report["cleaning_steps"].append(step_info)

        logger.info(f"Removed {duplicates_removed} duplicate rows")
        return df_clean

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        outliers_info = {}

        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            initial_count = len(df)
            outlier_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[outlier_mask]
            final_count = len(df)

            outliers_removed = initial_count - final_count
            outliers_info[col] = {
                "outliers_removed": outliers_removed,
                "bounds": (lower_bound, upper_bound),
            }
            logger.info(
                f"Removed {outliers_removed} outliers from {col} column")

            step_info = {
                "step": "handle_outliers",
                "action": "Removed outliers using IQR",
                "outliers_info": outliers_info,
            }
            self.cleaning_report["cleaning_steps"].append(step_info)
        return df

    def _correct_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Correct data types."""
        type_corrections = {}

        # Ensure integer columns are of correct type
        int_cols = ["make_year", "accidents_reported"]
        for col in int_cols:
            original_type = df[col].dtype
            if df[col].dtype != "int64":
                df[col] = pd.to_numeric(
                    df[col], errors="coerce").astype("Int64")
                type_corrections[col] = {
                    "before": original_type,
                    "after": "int64",
                }
                logger.info(f"Corrected {col} column to int64")

        # Ensure categorical columns are strings
        categorical_columns = [
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
        ]
        for col in categorical_columns:
            original_type = df[col].dtype
            if df[col].dtype != "object":
                df[col] = df[col].astype("string")
                type_corrections[col] = {
                    "before": original_type,
                    "after": "object",
                }
                logger.info(f"Corrected {col} column to object")

        step_info = {
            "step": "correct_data_types",
            "action": "Corrected data types",
            "type_corrections": type_corrections,
        }
        self.cleaning_report["cleaning_steps"].append(step_info)
        logger.info("Data types corrected")
        return df

    def _apply_business_validations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply business validations."""
        settings = get_settings()
        initial_rows = len(df)

        # Remove records with invalid price i.e. remove negative values if any
        if "price_usd" in df.columns:
            df = df[df["price_usd"] >= 0]
            logger.info("Removed records with invalid price (negative values)")

        # Remove records with invalid make_year
        if "make_year" in df.columns:
            current_year = settings.CURRENT_YEAR
            df = df[(df["make_year"] >= 1900) & (
                df["make_year"] <= current_year)]
            logger.info("Removed records with invalid make_year")

        # Remove records with negative accidents
        if "accidents_reported" in df.columns:
            df = df[df["accidents_reported"] >= 0]
            logger.info("Removed records with invalid accidents_reported")

        final_rows = len(df)
        invalid_removed = initial_rows - final_rows

        step_info = {
            "step": "apply_business_validations",
            "action": f"Removed {invalid_removed} records with invalid business logic",
            "rows_before": initial_rows,
            "rows_after": final_rows,
        }
        self.cleaning_report["cleaning_steps"].append(step_info)
        return df

    def get_cleaning_report(self) -> dict[str, Any]:
        """Get detailed cleaning report."""
        return self.cleaning_report
