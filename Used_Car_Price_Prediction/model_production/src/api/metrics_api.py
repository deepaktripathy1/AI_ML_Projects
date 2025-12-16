"""FastAPI service for ML model metrics with HTTP POST interface."""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    generate_latest, CollectorRegistry
)
from starlette.responses import Response
import threading

app = FastAPI(title="ML Model metrics API", version="1.0.0")

# Create a custom registry for this FastAPI instance
registry = CollectorRegistry()
metrics_lock = threading.Lock()

# Initialize Prometheus metrics
# Prediction Counters
predictions_total = Counter(
    "model_predictions_total",
    "Total number of predictions made",
    ["model_version", "endpoint"],
    registry=registry,
)

# Prediction latency
prediction_latency = Histogram(
    "model_prediction_duration_seconds",
    "Time spent on predictions",
    ["model_version"],
    buckets=[
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
    ],
    registry=registry,
)

# Prediction values
prediction_value = Histogram(
    "model_prediction_value_dollars",
    "Distribution of predicted car prices",
    buckets=[
        1000,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        40000,
        50000,
        75000,
        100000,
    ],
    registry=registry,
)

# Confidence scores
confidence_score = Histogram(
    "model_confidence_score",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    registry=registry,
)

# Current prediction statistics
current_prediction_mean = Gauge(
    "model_prediction_mean_dollars",
    "Current mean of predictions",
    registry=registry,
)

current_prediction_std = Gauge(
    "model_prediction_std_dollars",
    "Current standard deviation of predictions",
    registry=registry,
)

# Batch prediction metrics
batch_size = Histogram(
    "model_batch_prediction_size",
    "Size of batch predictions",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
    registry=registry,
)

# Drift detection status
data_drift_detected = Gauge(
    "model_data_drift_detected",
    "Whether data drift is detected (1 for yes, 0 for no)",
    registry=registry,
)

model_drift_detected = Gauge(
    "model_performance_drift_detected",
    "Whether model performance drift is detected (1 for yes, 0 for no)",
    registry=registry,
)

prediction_drift_detected = Gauge(
    "model_prediction_drift_detected",
    "Whether prediction drift is detected (1 for yes, 0 for no)",
    registry=registry,
)

# Drift scores
data_drift_score = Gauge(
    "model_data_drift_score",
    "Current data drift score",
    registry=registry
)

model_performance_change = Gauge(
    "model_performance_change",
    "Change in model performance (R² score change)",
    registry=registry,
)

# Feature drift
feature_drift_count = Gauge(
    "model_drifted_features_count",
    "Number of features with detected drift",
    registry=registry,
)

# Individual feature drift scores
feature_drift_score = Gauge(
    "model_feature_drift_score",
    "Drift score for each feature",
    ["feature"],
    registry=registry
)

# Drift check frequency
drift_checks_total = Counter(
    "model_drift_checks_total",
    "Total number of drift checks performed",
    ["drift_type"],
    registry=registry,
)

# Model test failures
model_test_failures = Gauge(
    "model_test_failures_count",
    "Number of failed model performance tests",
    registry=registry,
)

# Performance change metrics
performance_r2_change = Gauge(
    "model_performance_r2_change",
    "Change in R² score compared to reference",
    registry=registry,
)

performance_mae_change = Gauge(
    "model_performance_mae_change",
    "Change in MAE compared to reference",
    registry=registry,
)

# System metrics
# CPU and Memory
cpu_usage_percent = Gauge(
    "system_cpu_usage_percent",
    "Current CPU usage percentage",
    registry=registry,
)

memory_usage_percent = Gauge(
    "system_memory_usage_percent",
    "Current memory usage percentage",
    registry=registry,
)

disk_usage_percent = Gauge(
    "system_disk_usage_percent",
    "Current disk usage percentage",
    ["mount_point"],
    registry=registry,
)

# Service health
service_uptime_seconds = Gauge(
    "service_uptime_seconds",
    "Service uptime in seconds",
    registry=registry,
)

service_health_status = Gauge(
    "service_health_status",
    "Service health status (1 for healthy, 0 for unhealthy)",
    registry=registry,
)

# Model info
model_info = Info(
    "model_info",
    "Information about the current model",
    registry=registry
)

# Pydantic models for API


class PredictionMetric(BaseModel):
    prediction_value: float
    confidence_score: float
    latency_seconds: float
    batch_size: int
    successful_prediction: int
    failed_predictions: int
    total_latency: float
    model_version: str = "latest"
    endpoint: str = "predict"
    batch_endpoint: str = "batch_predict"


class DriftMetric(BaseModel):
    drift_results: Dict[str, Any]
    model_results: Optional[Dict[str, Any]] = None


class SystemMetric(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    mount_point: str = "/"


# API endpoints
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "service": "ML Model Metrics API",
        "status": "running",
        "metrics_endpoint": "/metrics",
        "post_endpoints": {
            "prediction": "/api/metrics/prediction",
            "drift": "/api/metrics/drift",
            "system": "/api/metrics/system"
        }
    }


@app.get("/health")
def health():
    """Health check for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(registry=registry),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post("/api/metrics/prediction")
def record_prediction(metric: PredictionMetric):
    """Record a prediction metric."""
    try:
        with metrics_lock:
            predictions_total.labels(
                model_version=metric.model_version,
                endpoint=metric.endpoint
            ).inc()
            prediction_latency.labels(
                model_version=metric.model_version).observe(
                metric.latency_seconds
            )
            prediction_value.observe(metric.prediction_value)
            confidence_score.observe(metric.confidence_score)
            batch_size.observe(metric.batch_size)

        return {
            "status": "success",
            "message": "Prediction metric recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metrics/drift")
def record_drift(metric: DriftMetric):
    """Record drift detection metrics."""
    try:
        with metrics_lock:
            drift_results = metric.drift_results
            model_results = metric.model_results

            # Update drift counters
            drift_checks_total.labels(drift_type="data").inc()
            if model_results:
                drift_checks_total.labels(drift_type="model").inc()

            # Update drift status
            data_drift_detected.set(
                1 if drift_results.get("dataset_drift_detected", False) else 0
            )

            # Update drift score
            data_drift_score.set(drift_results.get("drift_score", 0.0))

            # Update drifted features count
            drifted_features = drift_results.get("drifted_features", [])
            feature_drift_count.set(len(drifted_features))

            # Update individual feature drift scores
            drift_details = drift_results.get("drift_details", {})
            for feature, details in drift_details.items():
                drift_score = details.get("drift_score", 0.0)
                feature_drift_score.labels(feature).set(drift_score)

            # Model drift metrics
            if model_results:
                model_drift_detected.set(
                    1 if model_results.get(
                        "performance_degradation", False) else 0
                )
                prediction_drift_detected.set(
                    1 if model_results.get(
                        "prediction_drift_detected", False) else 0
                )

                # Update performance change metrics
                performance_change = model_results.get(
                    "performance_change", {})
                r2_change = performance_change.get("r2_change", 0.0)
                mae_change = performance_change.get("mae_change", 0.0)

                performance_r2_change.set(r2_change)
                performance_mae_change.set(mae_change)

                # Update test failures count
                test_failures = model_results.get("test_failures", [])
                model_test_failures.set(len(test_failures))

                model_info.info(
                    {
                        "last_updated": datetime.now().isoformat(),
                        "model_type": "LinearRegression",
                        "performance_degradation": str(model_results.get(
                            "performance_degradation", False)),
                        "prediction_drift": str(model_results.get(
                            "prediction_drift_detected", False)),
                        "failed_tests": str(len(test_failures)),
                        "r2_change": str(r2_change),
                        "mae_change": str(mae_change),
                    }
                )

        return {
            "status": "success",
            "message": "Drift metrics recorded",
            "drift_detected": drift_results.get("dataset_drift_detected", False),
            "drift_score": drift_results.get("drift_score", 0.0)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/metrics/system")
def record_system(metric: SystemMetric):
    """Record system metrics."""
    try:
        with metrics_lock:
            cpu_usage_percent.set(metric.cpu_percent)
            memory_usage_percent.set(metric.memory_percent)
            disk_usage_percent.labels(mount_point=metric.mount_point).set(
                metric.disk_percent)
            service_health_status.set(1)

        return {
            "status": "success",
            "message": "System metrics recorded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Metrics API Server on http://localhost:8001")
    print("Metrics endpoint: http://localhost:8001/metrics")
    print("=" * 60)

    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
