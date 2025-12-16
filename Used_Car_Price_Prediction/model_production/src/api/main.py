"""FastAPI main app."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import bentoml
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.api.models import CarPriceProcessor, RawCarData
from config.config import get_settings
from src.monitoring.monitoring_middleware import MonitoringMiddleware


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model storage
model_store = {}


# Model loader


async def load_model():
    """Load trained model and preprocessor."""
    settings = get_settings()

    try:
        # Load from BentoML
        try:
            bento_model = bentoml.models.get("used_car_price_model:latest")
            model = bentoml.sklearn.load_model(
                bento_model=bento_model)

            # Extract scaler and selected features from custom object if available
            scaler = bento_model.custom_objects.get("scaler", None)
            selected_features = bento_model.custom_objects.get(
                "feature_names", [])
            logger.info(
                "Model, scaler and features loaded from BentoML model store"
            )
        except Exception as e:
            logger.warning(f"Could not load from BentoML model store: {e}")
            # Load directly from pointer file
            try:
                pointer_file = settings.models_dir / "current.txt"
                if not pointer_file.exists():
                    raise FileNotFoundError(
                        f"Pointer file {pointer_file} not found"
                    )
                folder_name = pointer_file.read_text(encoding="utf-8").strip()
                model_dir = settings.models_dir / folder_name
                if not model_dir.exists():
                    raise FileNotFoundError(
                        f"Model folder {model_dir} doesn't exist"
                    )

                model_dict = joblib.load(model_dir / "model.pkl")
                model = model_dict.get("model")
                scaler = joblib.load(model_dir / "scaler.pkl")
                selected_features = joblib.load(model_dir / "features.pkl")
                logger.info("Model loaded directly from local file")
            except Exception as e:
                logger.warning("Pointer based model loading failed {e}")

                # Find latest model folder by mtime
                folder_list = list(
                    (settings.models_dir).glob("linear_regression_*")
                )
                if not folder_list:
                    raise FileNotFoundError(
                        f"No model directories found in {settings.models_dir}"
                    )

                latest = max(folder_list, key=lambda p: p.stat().st_mtime)

                model_dict = joblib.load(latest / "model.pkl")
                model = model_dict.get("model")
                scaler = joblib.load(latest / "scaler.pkl")
                selected_features = joblib.load(latest / "features.pkl")
                logger.info("Model loaded directly based on creation time")

        # Initialize preprocessor
        preprocessor = CarPriceProcessor(
            selected_features=selected_features,
            scaler=scaler,
            current_year=2025
        )

        model_store["model"] = model
        model_store["selected_features"] = selected_features
        model_store["scaler"] = scaler
        model_store["preprocessor"] = preprocessor
        model_store["loaded_at"] = datetime.now()

        logger.info("Model and processor loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Lifespan context manager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Car Price Prediction API")
    await load_model()
    yield
    # Shutdown
    logger.info("Shutting down Car Price Prediction API")
    model_store.clear()


# Create FastAPI app
app = FastAPI(
    title="Used Car Price Prediction API",
    description="FastAPI service for predicting used car prices",
    version="1.0.0",
    contact={"name": "ML Team", "email": "deepaktripathy1@gmail.com"},
    license_info={"name": "MIT License",
                  "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://streamlit:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(middleware_class=MonitoringMiddleware)

# Dependency to get model


def get_model():
    """Dependency to get the loaded model."""
    if "model" not in model_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return model_store["model"]


def get_preprocessor():
    """Dependency to get the preprocessor."""
    if "preprocessor" not in model_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Preprocessor not loaded",
        )
    return model_store["preprocessor"]


def get_scaler():
    """Dependency to get the scaler."""
    if "scaler" not in model_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Scaler not loaded"
        )
    return model_store["scaler"]


def validate_input(input_data: RawCarData) -> dict[str, Any]:
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


def calculate_confidence_score(input_data: RawCarData, prediction: float) -> float:
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
