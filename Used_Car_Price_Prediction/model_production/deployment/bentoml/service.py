"""BentoML service for model deployment."""

import logging
import time
from datetime import datetime
from typing import Any

import bentoml
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, Gauge

from config.config import get_settings
from src.data_preprocessing import DataCleaner, FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prediction Metrics
BENTOML_PREDICTION_COUNT = Counter(
    "bentoml_model_predictions_total",
    "Total number of predictions made",
    ["model_version", "endpoint"]
)

BENTOML_PREDICTION_ERRORS = Counter(
    "bentoml_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type", "endpoint"]
)

BENTOML_PREDICTION_DURATION = Histogram(
    "bentoml_prediction_duration_seconds",
    "Time spent processing predictions",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

# Model Metrics
BENTOML_CONFIDENCE_SCORE = Histogram(
    "bentoml_confidence_score",
    "Distribution of model confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
)

BENTOML_PREDICTED_PRICE = Histogram(
    "bentoml_predicted_car_price_usd",
    "Distribution of predicted car prices",
    buckets=[5000, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000]
)

# Business Metrics
BENTOML_PREDICTION_BY_BRAND = Counter(
    "bentoml_predictions_by_brand_total",
    "Total predictions by car brand",
    ["brand"]
)

BENTOML_PREDICTION_BY_FUEL_TYPE = Counter(
    "bentoml_predictions_by_fuel_type_total",
    "Total predictions by fuel type",
    ["fuel_type"]
)

BENTOML_ACTIVE_REQUESTS = Gauge(
    "bentoml_active_prediction_requests",
    "Number of prediction requests currently being processed"
)


# Define input schema for raw data


class RawCarData(BaseModel):
    """Input schema for raw data."""

    make_year: int = Field(
        ..., ge=1900, le=2024, description="Year the car was manufactured"
    )
    accidents_reported: int = Field(
        ..., ge=0, le=10, description="Number of accidents reported"
    )
    fuel_type: str = Field(...,
                           description="Type of fuel: Diesel, Electric, Petrol")
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
        description=(
            "Service history: Full Service History, Partial Service History, None"
        ),
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
            scaler: The scaler to use for feature scaling
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
            return df_final_scaled  # Return first row as array

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


# Create BentoML service
settings = get_settings()


@bentoml.service(name=str(settings.BENTO_SERVICE_NAME))
class CarPricePredictionService:
    """Class-based BentoML service for car price prediction."""

    def __init__(self):
        settings = get_settings()
        # Load model during initialization
        try:
            bento_model = bentoml.models.get("used_car_price_model:latest")
            self.model = bentoml.sklearn.load_model(
                bento_model=bento_model)
            self.scaler = bento_model.custom_objects.get("scaler", None)
            self.selected_features = bento_model.custom_objects.get(
                "feature_names", []
            )
            logger.info(
                "Model, scaler and features loaded from BentoML model store"
            )
        except Exception as e:
            logger.warning(f"Could not load from BentoML model store: {e}")
            # Load directly from file
            model_dir = (
                settings.models_dir
                / f"linear_regression_{datetime.now().strftime(
                    '%Y%m%d_%H%M%S')}"
            )
            self.model = joblib.load(model_dir / "model.pkl")
            self.scaler = joblib.load(model_dir / "scaler.pkl")
            self.selected_features = joblib.load(model_dir / "features.pkl")
            logger.info("Model loaded directly from file")

        # Initialize preprocessor
        self.preprocessor = CarPriceProcessor(
            selected_features=self.selected_features,
            scaler=self.scaler,
            current_year=2025
        )
        logger.info("Service initialized successfully")

    @bentoml.api
    def predict_single(self, input_data: RawCarData) -> PredictionOutput:
        """Predict car price for a single car.

        Args:
            input_data: Raw car data

        Returns:
            PredictionOutput: Prediction with metadata
        """
        start_time = time.time()

        # Track active requests
        BENTOML_ACTIVE_REQUESTS.inc()

        try:
            logger.info(
                f"Received prediction request for {
                    input_data.brand} {
                        input_data.make_year}"
            )

            # Increment prediction counter
            BENTOML_PREDICTION_COUNT.labels(
                model_version="latest",
                endpoint="predict_single"
            ).inc()

            # Track business metrics
            BENTOML_PREDICTION_BY_BRAND.labels(brand=input_data.brand).inc()
            BENTOML_PREDICTION_BY_FUEL_TYPE.labels(
                fuel_type=input_data.fuel_type).inc()

            # Input validation
            validation_result = self._validate_input(input_data)

            # Preprocess the raw input
            processed_features = self.preprocessor.preprocess_single(
                raw_data=input_data
            )

            # Make prediction
            prediction = self.model.predict(  # type: ignore
                processed_features.reshape(1, -1)
            )
            prediction = float(np.ravel(prediction)[0])

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                input_data, prediction
            )

            # Track model metrics
            BENTOML_CONFIDENCE_SCORE.observe(confidence_score)
            BENTOML_PREDICTED_PRICE.observe(prediction)

            # Get preprocessing info
            preprocessing_info = self.preprocessor.get_preprocessing_info(
                raw_data=input_data
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Track prediction duration
            BENTOML_PREDICTION_DURATION.labels(
                endpoint="predict_single"
            ).observe(processing_time)

            # Create prediction metadata
            prediction_metadata = {
                "model_version": "latest",
                "prediction_timestamp": datetime.now().isoformat(),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "features_used": len(self.selected_features),
                "car_summary": (
                    f"{input_data.brand} {input_data.make_year}"
                ),
            }

            logger.info(
                "Prediction completed: "
                "${prediction:.2f} (confidence: {confidence_score:.3f})"
            )

            return PredictionOutput(
                predicted_price=float(prediction),
                confidence_score=confidence_score,
                input_validation=validation_result,
                preprocessing_info=preprocessing_info,
                prediction_metadata=prediction_metadata,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")

            # Track error metrics
            BENTOML_PREDICTION_ERRORS.labels(
                error_type=type(e).__name__,
                endpoint="predict_single"
            ).inc()

            raise

        finally:
            BENTOML_ACTIVE_REQUESTS.dec()

    @bentoml.api
    def predict_batch(
        self,
        input_data: BatchPredictionInput
    ) -> BatchPredictionOutput:
        """Predict car prices for multiple cars with complete processing pipeline.

        Args:
            input_data: Batch of raw car data

        Returns:
            BatchPredictionOutput: Batch predictions with metadata
        """
        start_time = time.time()

        # Track active requests
        BENTOML_ACTIVE_REQUESTS.inc()

        try:
            logger.info(
                f"Received batch prediction request for {len(input_data.cars)} cars"
            )

            predictions = []
            successful_count = 0
            failed_count = 0

            # Process each car
            for i, car_data in enumerate(input_data.cars):
                try:
                    # Track business metrics
                    BENTOML_PREDICTION_BY_BRAND.labels(
                        brand=car_data.brand).inc()
                    BENTOML_PREDICTION_BY_FUEL_TYPE.labels(
                        fuel_type=car_data.fuel_type).inc()

                    # input validation
                    validation_result = self._validate_input(car_data)

                    # Preprocess the raw input
                    processed_features = self.preprocessor.preprocess_single(
                        raw_data=car_data
                    )

                    # Predict
                    prediction = self.model.predict(  # type: ignore
                        processed_features.reshape(1, -1)
                    )
                    prediction = float(np.ravel(prediction)[0])

                    # Calculate confidence score
                    confidence_score = self._calculate_confidence_score(
                        car_data, prediction
                    )

                    # Track model metrics
                    BENTOML_CONFIDENCE_SCORE.observe(confidence_score)
                    BENTOML_PREDICTED_PRICE.observe(prediction)

                    # Get preprocessing info
                    preprocessing_info = self.preprocessor.get_preprocessing_info(
                        raw_data=car_data
                    )

                    # Create prediction metadata
                    prediction_metadata = {
                        "batch_index": i,
                        "model_version": "latest",
                        "prediction_timestamp": datetime.now().isoformat(),
                        "features_used": len(self.selected_features),
                        "car_summary": (
                            f"{car_data.brand} {car_data.make_year}"
                        )
                    }

                    predictions.append(
                        PredictionOutput(
                            predicted_price=float(prediction),
                            confidence_score=confidence_score,
                            input_validation=validation_result,
                            preprocessing_info=preprocessing_info,
                            prediction_metadata=prediction_metadata,
                        )
                    )

                    successful_count += 1

                except Exception as e:
                    logger.error(f"Prediction failed for car {i}: {str(e)}")

                    # Track individual error
                    BENTOML_PREDICTION_ERRORS.labels(
                        error_type=type(e).__name__,
                        endpoint="predict_batch"
                    ).inc()

                    predictions.append(
                        PredictionOutput(
                            predicted_price=0.0,
                            confidence_score=0.0,
                            input_validation={
                                "status": "error", "error": str(e)},
                            preprocessing_info={},
                            prediction_metadata={
                                "batch_index": i, "error": str(e)},
                        )
                    )

                    failed_count += 1

            # Track batch prediction count
            BENTOML_PREDICTION_COUNT.labels(
                model_version="latest",
                endpoint="predict_batch"
            ).inc(successful_count)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Track prediction duration
            BENTOML_PREDICTION_DURATION.labels(
                endpoint="predict_batch"
            ).observe(processing_time)

            # Create batch metadata
            batch_metadata = {
                "total_cars": len(input_data.cars),
                "successful_predictions": sum(
                    1 for p in predictions if p.confidence_score > 0
                ),
                "failed_predictions": sum(
                    1 for p in predictions if p.confidence_score == 0
                ),
                "batch_processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "model_version": "latest",
                "batch_timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"Batch prediction completed: "
                f"{batch_metadata['successful_predictions']}/"
                f"{batch_metadata['total_cars']} successful"
            )

            return BatchPredictionOutput(
                predictions=predictions, batch_metadata=batch_metadata
            )

        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")

            # Track error
            BENTOML_PREDICTION_ERRORS.labels(
                error_type=type(e).__name__,
                endpoint="predict_batch"
            ).inc()

            raise

        finally:
            BENTOML_ACTIVE_REQUESTS.dec()

    @bentoml.api
    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "timestamp": datetime.now().isoformat(),
            "features_count": len(self.selected_features),
            "preprocessing_enabled": True,
        }

    @bentoml.api
    def model_info(self) -> dict[str, Any]:
        """Get model information and feature details."""
        return {
            "model_info": {
                "model_version": "latest",
                "model_type": "Linear Regression",
                "framework": "sklearn",
                "features_count": len(self.selected_features),
                "selected_features": self.selected_features,
                "model_loaded": self.model is not None,
            },
            "input_schema": {
                "required_fields": [
                    "make_year",
                    "mileage_kmpl",
                    "engine_cc",
                    "accidents_reported",
                    "fuel_type",
                    "brand",
                    "transmission",
                    "color",
                    "insurance_valid",
                    "owner_count",
                    "service_history",
                ],
                "optional_field": ["price_usd"],
                "valid_values": {
                    "fuel_type": ["Diesel", "Electric", "Petrol"],
                    "brand": [
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
                    "transmission": ["Automatic", "Manual"],
                    "color": ["Black", "Blue", "Gray", "Red", "Silver", "White"],
                    "insurance_valid": ["Yes", "No"],
                    "service_history": [
                        "None",
                        "Partial Service History",
                        "Full Service History",
                    ],
                },
            },
            "preprocessing_steps": [
                "Calculate car_age from make_year",
                "Create has_accident binary feature",
                "Encode service_history ordinally",
                "One-hot encode categorical features",
                "Select trained features only",
            ],
        }

    def _validate_input(self, input_data: RawCarData) -> dict[str, Any]:
        """Validate input data and return validation results."""
        validation_result = {"status": "valid", "warnings": [], "errors": []}

        try:
            # Check car age
            car_age = datetime.now().year - input_data.make_year
            if car_age > 30:
                validation_result["warnings"].append(
                    f"Car is {car_age} years old - "
                    "prediction may be less accurate for very old cars"
                )
            # Check accident count
            if input_data.accidents_reported > 5:
                validation_result["warnings"].append(
                    "High accident count may affect prediction accuracy"
                )

            # Check for unusual combinations
            if input_data.brand in ["Tesla", "BMW", "Volkswagen"] and car_age > 20:
                validation_result["warnings"].append(
                    "Luxury cars with high age - unusual combination"
                )

            if validation_result["errors"]:
                validation_result["status"] = "error"
            elif validation_result["warnings"]:
                validation_result["status"] = "warning"

        except Exception as e:
            validation_result["status"] = "error"
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    def _calculate_confidence_score(
        self, input_data: RawCarData, prediction: float
    ) -> float:
        """Calculate a confidence score for the prediction."""
        try:
            confidence_score = 0.8  # 80% base confidence

            # Car Age
            car_age = datetime.now().year - input_data.make_year
            if car_age <= 5:
                confidence_score += 0.1
            elif car_age > 20:
                confidence_score -= 0.2

            # Brands
            common_brands = ["Honda", "Toyota", "Ford", "Chevrolet"]
            if input_data.brand in common_brands:
                confidence_score += 0.05

            # Accidents
            if input_data.accidents_reported == 0:
                confidence_score += 0.05
            elif input_data.accidents_reported > 3:
                confidence_score -= 0.1

            # Predictions
            if prediction < 1000 or prediction > 100000:
                confidence_score -= 0.1

            confidence_score = max(0.1, min(0.95, confidence_score))
            return round(confidence_score, 3)

        except Exception:
            return 0.5  # Default confidence score

    def test_preprocessing(self):
        """Test the preprocessing pipeline with sample data."""
        sample_data = RawCarData(
            make_year=2020,
            mileage_kmpl=12.14,
            engine_cc=1500,
            accidents_reported=1,
            fuel_type="Petrol",
            owner_count=3,
            brand="Toyota",
            transmission="Automatic",
            color="White",
            insurance_valid="Yes",
            service_history="Full Service History",
        )

        processed = self.preprocessor.preprocess_single(sample_data)
        print(f"Processed features shape: {processed.shape}")
        print(f"Processed features: {processed}")

        return processed


if __name__ == "__main__":
    # Test preprocessing
    service = CarPricePredictionService()
    service.test_preprocessing()

svc = CarPricePredictionService
