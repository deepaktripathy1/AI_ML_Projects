"""Pydantic Model for FastAPI."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.data_preprocessing.data_cleaner import DataCleaner
from src.data_preprocessing.feature_engineering import FeatureEngineer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models


class RawCarData(BaseModel):
    """Input schema for raw data."""

    make_year: int = Field(
        ..., ge=1900, le=2024, description="Year the car was manufactured"
    )
    accidents_reported: int = Field(
        ..., ge=0, le=10, description="Number of accidents reported"
    )
    fuel_type: str = Field(
        ..., description="Type of fuel: Diesel, Diesel, Electric, Petrol"
    )
    brand: str = Field(
        ...,
        description=(
            "Car brand: Honda, Toyota, Ford, Chevrolet,"
            "BMW, Volkswagen, Tesla, Hyundai, Kia, Nissan"
        ),
    )
    transmission: str = Field(...,
                              description="Transmission type: Automatic, Manual")
    color: str = Field(
        ..., description="Car color: Black, Blue, Gray, Red, Silver, White"
    )
    insurance_valid: str = Field(...,
                                 description="Insurance validity: Yes, No")
    service_history: str | None = Field(
        "None",
        description="Service history: Full Service History, "
        "Partial Service History, None",
    )
    mileage_kmpl: float | None = Field(
        ..., ge=0, le=100000, description="Mileage in kilometers per liter"
    )
    engine_cc: int = Field(..., ge=0, le=100000,
                           description="Engine capacity in cc")

    owner_count: int = Field(..., ge=0, le=100, description="Number of owners")

    @field_validator("fuel_type")
    def validate_fuel_type(cls, v):
        valid_types = ["Diesel", "Electric", "Petrol"]
        if v not in valid_types:
            raise ValueError(f"fuel_type must be one of {valid_types}")
        return v

    @field_validator("brand")
    def validate_brand(cls, v):
        valid_brands = [
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
        ]
        if v not in valid_brands:
            raise ValueError(f"brand must be one of {valid_brands}")
        return v

    @field_validator("transmission")
    def validate_transmission(cls, v):
        valid_transmissions = ["Automatic", "Manual"]
        if v not in valid_transmissions:
            raise ValueError(
                f"transmission must be one of {valid_transmissions}")
        return v

    @field_validator("color")
    def validate_color(cls, v):
        valid_colors = ["Black", "Blue", "Gray", "Red", "Silver", "White"]
        if v not in valid_colors:
            raise ValueError(f"color must be one of {valid_colors}")
        return v

    @field_validator("insurance_valid")
    def validate_insurance(cls, v):
        valid_insurance = ["Yes", "No"]
        if v not in valid_insurance:
            raise ValueError(
                f"insurance_valid must be one of {valid_insurance}")
        return v


# Define output schema


class PredictionOutput(BaseModel):
    """Output schema for used car price prediction."""

    predicted_price: float = Field(..., description="Predicted car price")
    confidence_score: float = Field(..., description="95% confidence interval")
    input_validation: dict[str, Any] = Field(
        ..., description="Input validation results"
    )
    preprocessing_info: dict[str, Any] = Field(
        ..., description="Preprocessing steps applied"
    )
    prediction_metadata: dict[str,
                              Any] = Field(..., description="Prediction metadata")


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    cars: list[RawCarData] = Field(
        ..., min_length=1, max_length=1000, description="Car data for price prediction"
    )


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: list[PredictionOutput] = Field(
        ..., description="List of prediction results"
    )
    batch_metadata: dict[str,
                         Any] = Field(..., description="Batch processing metadata")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Current timestamp")
    features_count: int = Field(..., description="Number of features")
    preprocessing_enabled: bool = Field(...,
                                        description="Preprocessing status")
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfoResponse(BaseModel):
    """Model information response schema."""

    model_info: dict[str, Any] = Field(..., description="Model information")
    input_schema: dict[str,
                       Any] = Field(..., description="Input schema details")
    preprocessing_steps: list[str] = Field(...,
                                           description="Preprocessing steps")
    api_endpoints: list[dict[str, str]] = Field(
        ..., description="Available API endpoints"
    )


class CarPriceProcessor:
    """Preprocessing pipeline wrapper for production deployment."""

    def __init__(
        self,
        selected_features: list[str],
        scaler=None,
        current_year: int = 2025
    ):
        """Initialize preprocessor with selected features.

        Args:
            selected_features: List of features selected during training
            scaler: Scaler object for feature scaling
            current_year: Current year for age calculation
        """
        self.selected_features = selected_features
        self.scaler = scaler
        self.current_year = current_year
        self.data_cleaner = DataCleaner()
        self.feature_engineer = FeatureEngineer()

        # Service history mapping
        self.service_history_mapping = {
            "None": 0,
            "Partial Service History": 1,
            "Full Service History": 2,
        }

        logger.info(
            f"Preprocessor initialized with"
            f"{len(selected_features)} selected features"
            f", scaler: {'enabled' if scaler else 'disabled'}"
        )

    def preprocess_single(self, raw_data: RawCarData) -> np.ndarray:
        """Preprocess single raw input to model-ready format.

        Args:
            raw_data: Raw car data

        Returns:
            numpy.ndarray: Preprocessed features
        """
        # Convert to DataFrame
        df = pd.DataFrame([raw_data.model_dump()])

        # Apply preprocesing pipeline
        df_processed = self._apply_preprocessing_pipeline(df)

        # Ensure all selected features are present
        df_final = self._ensure_feature_alignment(df_processed)

        # Apply scaling if scaler is available
        if self.scaler is not None:
            df_final_scaled = self.scaler.transform(df_final.values)
            return df_final_scaled[0]  # Return first row as array

        return df_final.values[0]  # Return first row as array

    def preprocess_batch(self, raw_data_list: list[RawCarData]) -> np.ndarray:
        """Preprocess batch of raw inputs.

        Args:
            raw_data_list: List of raw car data

        Returns:
            numpy.ndarray: Preprocessed features matrix
        """
        # Convert to DataFrame
        df = pd.DataFrame([item.model_dump() for item in raw_data_list])

        # Apply preprocessing pipeline
        df_processed = self._apply_preprocessing_pipeline(df)

        # Ensure all selected features are present
        df_final = self._ensure_feature_alignment(df_processed)

        # Apply scaling if scaler is available
        if self.scaler is not None:
            df_final_scaled = self.scaler.transform(df_final.values)
            return df_final_scaled[0]  # Return first row as array

        return df_final.values

    def _apply_preprocessing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the complete preprocessing pipeline."""
        # Create a copy
        df_clean = df.copy()

        # Handle missing service_history
        df_clean["service_history"] = df_clean["service_history"].fillna(
            "None")

        # Feature engineering
        # Create car_age
        df_clean["car_age"] = self.current_year - df_clean["make_year"]

        # Create has_accident
        df_clean["has_accident"] = (
            df_clean["accidents_reported"] > 0).astype(int)

        # Encode service_history
        df_clean["service_history_encoded"] = df_clean["service_history"].map(
            self.service_history_mapping
        )
        df_clean["service_history_encoded"] = (
            df_clean["service_history_encoded"].fillna(0).astype(int)
        )

        # One-hot encode categorical features
        categorical_features = [
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
        ]
        df_encoded = pd.get_dummies(
            df_clean, columns=categorical_features, drop_first=True, dtype=int
        )

        # Drop unnecessary columns
        # price might not be in inference data
        columns_to_drop = [
            "make_year",
            "accidents_reported",
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
        ]
        existing_columns_to_drop = [
            col for col in columns_to_drop if col in df_encoded.columns
        ]
        df_final = df_encoded.drop(columns=existing_columns_to_drop)

        return df_final

    def _ensure_feature_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has exactly the features the model expects.

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with aligned features
        """
        # Add missing features with default value 0
        for feature in self.selected_features:
            if feature not in df.columns:
                df[feature] = 0
                logger.warning(
                    f"Added missing feature '{feature} with default value 0")

        # Remove extra features not in selected_features
        extra_features = [
            col for col in df.columns if col not in self.selected_features
        ]
        if extra_features:
            df = df.drop(columns=extra_features)
            logger.warning(f"Removed extra features: {extra_features}")

        # Reorder columns to match training order
        df = df[self.selected_features]

        return df

    def get_preprocessing_info(self, raw_data: RawCarData) -> dict[str, Any]:
        """Get information about preprocessing steps applied."""
        return {
            "car_age_calculated": self.current_year - raw_data.make_year,
            "has_accident": raw_data.accidents_reported > 0,
            "service_history_encoded": self.service_history_mapping,
            "categorical_features_encoded": [
                "fuel_type",
                "brand",
                "transmission",
                "color",
                "insurance_valid",
            ],
            "selected_features_count": len(self.selected_features),
            "preprocessing_timestamp": datetime.now().isoformat(),
        }
