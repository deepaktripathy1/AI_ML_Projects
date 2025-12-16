"""API endpoints."""

import logging
import time
from datetime import datetime
from typing import Any

import numpy as np
import uvicorn
from fastapi import Depends, HTTPException, status
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

from src.api.main import (
    app,
    calculate_confidence_score,
    get_model,
    get_preprocessor,
    get_scaler,
    model_store,
    validate_input,
)
from src.api.models import (
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    ModelInfoResponse,
    PredictionOutput,
    RawCarData,
)
from src.monitoring.monitoring_middleware import add_monitoring_endpoints


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== Prometheus Metrics Definition ====

# Request Metrics
FASTAPI_REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total FastAPI requests",
    ["method", "endpoint", "status"]
)

FASTAPI_REQUEST_DURATION = Histogram(
    "fastapi_request_duration_seconds",
    "FastAPI request duration",
    ["method", "endpoint"]
)

# Prediction Metrics
FASTAPI_PREDICTION_COUNT = Counter(
    "fastapi_model_predictions_total",
    "Total model predictions",
    ["model_version", "endpoint"]
)

FASTAPI_PREDICTION_ERRORS = Counter(
    "fastapi_model_prediction_errors_total",
    "Total prediction errors",
    ["error_type", "endpoint"]
)

FASTAPI_PREDICTION_DURATION = Histogram(
    "fastapi_model_prediction_duration_seconds",
    "Prediction processing time",
    ["endpoint"]
)

# Model Metrics
FASTAPI_MODEL_CONFIDENCE = Histogram(
    "fastapi_model_confidence_score",
    "Distribution of confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)

FASTAPI_PREDICTED_PRICE = Histogram(
    "fastapi_predicted_car_price_usd",
    "Distribution of predicted car prices",
    buckets=[5000, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000]
)

# Business metrics
FASTAPI_PREDICTION_BY_BRAND = Counter(
    "fastapi_predictions_by_brand_total",
    "Total predictions by car brand",
    ["brand"]
)

FASTAPI_PREDICTION_BY_FUEL_TYPE = Counter(
    "fastapi_predictions_by_fuel_type_total",
    "Total predictions by fuel type",
    ["fuel_type"]
)

# System metrics
FASTAPI_ACTIVE_REQUESTS = Gauge(
    "fastapi_active_requests",
    "Number of requests currently being processed"
)

# ==== Prometheus Metrics Endpoint ====


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint.

    This endpoint exposes metrics in Prometheus format for scraping
    """
    return Response(
        content=generate_latest(REGISTRY),
        media_type="text/plain; charset=utf-8"
    )

# ==== Dependencies =====

model_dependency = Depends(get_model)
scaler_dependency = Depends(get_scaler)
preprocessor_dependency = Depends(get_preprocessor)

# API endpoints


@app.get("/", response_model=dict[str, Any])
async def root():
    """Root endpoint with API Information."""
    return {
        "message": "Used Car Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model_info",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    FASTAPI_REQUEST_COUNT.labels(
        method="GET",
        endpoint="/health",
        status="200"
    ).inc()

    model_loaded = "model" in model_store
    loaded_at = model_store.get("loaded_at", datetime.now())
    uptime_seconds = (datetime.now() - loaded_at).total_seconds()
    features_count = len(model_store.get("selected_features", []))

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
        features_count=features_count,
        preprocessing_enabled=True,
        version="1.0.0",
        uptime_seconds=uptime_seconds,
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information and API details."""
    FASTAPI_REQUEST_COUNT.labels(
        method="GET",
        endpoint="/model_info",
        status="200"
    ).inc()

    selected_features = model_store.get("selected_features", [])

    return ModelInfoResponse(
        model_info={
            "model_version": "latest",
            "model_type": "Linear Regression",
            "framework": "sklearn",
            "features_count": len(selected_features),
            "selected_features": selected_features,
            "model_loaded": "model" in model_store,
        },
        input_schema={
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
            "valid_values": {
                "fuel_type": ["Diesel", "Electric", "Hybrid", "Petrol"],
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
        preprocessing_steps=[
            "Calculate car_age from make_year",
            "Create has_accident binary feature",
            "Encode service_history ordinally",
            "One-hot encode categorical features",
            "Select trained features only",
        ],
        api_endpoints=[
            {
                "endpoint": "/predict",
                "method": "POST",
                "description": "Single car prediction",
            },
            {
                "endpoint": "/predict-batch",
                "method": "POST",
                "description": "Batch car predictions",
            },
            {
                "endpoint": "/health",
                "method": "GET",
                "description": "Health check"},
            {
                "endpoint": "/model_info",
                "method": "GET",
                "description": "Model information",
            },
            {
                "endpoint": "/docs",
                "method": "GET",
                "description": "API documentation"
            },
            {
                "endpoint": "/metrics",
                "method": "GET",
                "description": "Prometheus metrics"
            },

        ],
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(
    input_data: RawCarData,
    model=model_dependency,
    scaler=scaler_dependency,
    preprocessor=preprocessor_dependency,
):
    """Predict car price for a single car."""
    start_time = time.time()

    # Track active requests
    FASTAPI_ACTIVE_REQUESTS.inc()

    try:
        logger.info(
            f"Received prediction request for {input_data.brand} {input_data.make_year}"
        )

        # Increment prediction counter
        FASTAPI_PREDICTION_COUNT.labels(
            model_version="latest",
            endpoint="/predict"
        ).inc()

        # Track business metrics
        FASTAPI_PREDICTION_BY_BRAND.labels(brand=input_data.brand).inc()
        FASTAPI_PREDICTION_BY_FUEL_TYPE.labels(
            fuel_type=input_data.fuel_type).inc()

        # Input validation
        validation_result = validate_input(input_data)

        # Preprocess the raw input
        processed_features = preprocessor.preprocess_single(
            raw_data=input_data)

        # Scale the raw input
        # scaled_features = scaler.transform(processed_features)

        # Make prediction
        prediction = model.predict(processed_features.reshape(1, -1))
        prediction = float(np.ravel(prediction)[0])

        # Calculate confidence score
        confidence_score = calculate_confidence_score(input_data, prediction)

        # Track prediction metrics
        FASTAPI_MODEL_CONFIDENCE.observe(confidence_score)
        FASTAPI_PREDICTED_PRICE.observe(prediction)

        # Get preprocessing info
        preprocessing_info = preprocessor.get_preprocessing_info(
            raw_data=input_data)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track request duration
        FASTAPI_REQUEST_DURATION.labels(
            method="POST",
            endpoint="/predict"
        ).observe(processing_time)

        FASTAPI_PREDICTION_DURATION.labels(
            endpoint="/predict"
        ).observe(processing_time)

        # Create predicton metadata
        prediction_metadata = {
            "model_version": "latest",
            "prediction_timestamp": datetime.now().isoformat(),
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "features_used": len(model_store["selected_features"]),
            "car_summary": (
                f"{input_data.brand} {input_data.make_year} ({input_data.color})"
            ),
        }

        logger.info(
            "Prediction completed: "
            "${prediction:.2f} (confidence: {confidence_score:.3f})"
        )

        # Track successful request
        FASTAPI_REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status="200"
        ).inc()

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
        FASTAPI_PREDICTION_ERRORS.labels(
            error_type=type(e).__name__,
            endpoint="/predict"
        ).inc()

        FASTAPI_REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            status="500"
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        ) from None


@app.post("/predict-batch", response_model=BatchPredictionOutput)
async def predict_batch(
    input_data: BatchPredictionInput,
    model=model_dependency,
    scaler=scaler_dependency,
    preprocessor=preprocessor_dependency,
):
    """Predict car prices for multiple cars."""
    start_time = time.time()

    # Track active requests
    FASTAPI_ACTIVE_REQUESTS.inc()

    try:
        logger.info(
            f"Received batch prediction request with {len(input_data.cars)} cars"
        )

        predictions = []
        successful_predictions = 0
        failed_predictions = 0

        # Process each car
        for i, car_data in enumerate(input_data.cars):
            try:
                # Track business metrics
                FASTAPI_PREDICTION_BY_BRAND.labels(brand=car_data.brand).inc()
                FASTAPI_PREDICTION_BY_FUEL_TYPE.labels(
                    fuel_type=car_data.fuel_type).inc()

                # Input validation
                validation_result = validate_input(car_data)

                # Preprocess the raw input
                processed_features = preprocessor.preprocess_single(
                    raw_data=car_data)

                # Scale the features
                # scaled_features = scaler.transform(processed_features)

                # Predict
                prediction = model.predict(processed_features.reshape(1, -1))
                prediction = float(np.ravel(prediction)[0])

                # Calculate confidence score
                confidence_score = calculate_confidence_score(
                    car_data, prediction)

                # Track prediction metrics
                FASTAPI_MODEL_CONFIDENCE.observe(confidence_score)
                FASTAPI_PREDICTED_PRICE.observe(prediction)

                # Get preprocessing info
                preprocessing_info = preprocessor.get_preprocessing_info(
                    raw_data=car_data
                )

                # Create prediction metadata
                prediction_metadata = {
                    "batch_index": i,
                    "model_version": "latest",
                    "prediction_timestamp": datetime.now().isoformat(),
                    "features_used": len(model_store["selected_features"]),
                    "car_summary": (
                        f"{car_data.brand} {car_data.make_year} ({car_data.color})"
                    ),
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

                successful_predictions += 1

            except Exception as e:
                logger.error(f"Prediction failed for car {i}: {str(e)}")

                # Track individual prediction error
                FASTAPI_PREDICTION_ERRORS.labels(
                    error_type=type(e).__name__,
                    endpoint="/predict-batch"
                ).inc()

                predictions.append(
                    PredictionOutput(
                        predicted_price=0.0,
                        confidence_score=0.0,
                        input_validation={"status": "error", "error": str(e)},
                        preprocessing_info={},
                        prediction_metadata={
                            "batch_index": i, "error": str(e)},
                    )
                )

                failed_predictions += 1

        # Track batch prediction count
        FASTAPI_PREDICTION_COUNT.labels(
            model_version="latest",
            endpoint="/predict-batch"
        ).inc(successful_predictions)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Track request duration
        FASTAPI_REQUEST_DURATION.labels(
            method="POST",
            endpoint="/predict-batch"
        ).observe(processing_time)

        FASTAPI_PREDICTION_DURATION.labels(
            endpoint="/predict-batch"
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
            f"{batch_metadata['successful_predictions']}/{batch_metadata['total_cars']} successful"
        )

        # Track successful request
        FASTAPI_REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict-batch",
            status="200"
        ).inc()

        return BatchPredictionOutput(
            predictions=predictions, batch_metadata=batch_metadata
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")

        # Track error metrics
        FASTAPI_PREDICTION_ERRORS.labels(
            error_type=type(e).__name__,
            endpoint="/predict-batch"
        ).inc()

        FASTAPI_REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict-batch",
            status="500"
        ).inc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        ) from None

    finally:
        FASTAPI_ACTIVE_REQUESTS.dec()


# Custom OpenAPI schema


def custom_openapi():
    """Custom Open API schema with additional information."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Used Car Price Prediction API",
        version="1.0.0",
        description="Fast API service for predicting used car prices",
        routes=app.routes,
    )

    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

add_monitoring_endpoints(app=app)

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api.endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="INFO"
    )
