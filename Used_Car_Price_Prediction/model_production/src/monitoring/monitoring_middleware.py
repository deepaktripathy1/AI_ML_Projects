"""FastAPI monitoring middleware integration."""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.logging_config import LoggingConfig


logger = LoggingConfig.get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic model monitoring."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_predictions": 0,
            "total_errors": 0,
            "response_times": deque(maxlen=1000),
            "predictions_per_minute": deque(maxlen=60),
            "last_prediction": None,
            "error_types": defaultdict(int),
        }
        self.start_time = time.time()

    async def dispatch(self, request: Request, call_next):
        """Process request and response with monitoring."""
        start_time = time.time()
        path = request.url.path
        method = request.method

        # Increment total requests
        self.metrics["total_requests"] += 1

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time
            self.metrics["response_times"].append(process_time)

            # Monitor prediction endpoints
            if path in ["/predict", "/predict-batch"] and method == "POST":
                if response.status_code == 200:
                    self.metrics["total_predictions"] += 1
                    self.metrics["last_prediction"] = datetime.now(
                    ).isoformat()
                    # Track predictions per minute
                    self.metrics["predictions_per_minute"].append(
                        datetime.now())
                else:
                    self.metrics["total_errors"] += 1

            # Add timing header
            response.headers["X-Process-Time"] = str(round(process_time, 4))

            logger.info(
                f"{method} {path} - {response.status_code}-{process_time:.4f}s")

            return response

        except Exception as e:
            # Record error
            self.metrics["total_errors"] += 1
            self.metrics["error_types"][type(e).__name__] += 1

            logger.error(f"Request processing failed: {method} {path} - {e}")
            raise

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        # Calculate average response time
        response_times = list(self.metrics["response_times"])
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Calculate recent predictions per minute
        now = datetime.now()
        recent_predictions = [
            dt
            for dt in self.metrics["predictions_per_minute"]
            if (now - dt).total_seconds() <= 60
        ]

        return {
            "total_requests": self.metrics["total_requests"],
            "total_predictions": self.metrics["total_predictions"],
            "total_errors": self.metrics["total_errors"],
            "error_rate": (
                self.metrics["total_errors"] /
                max(1, self.metrics["total_requests"])
            )
            * 100,
            "avg_response_time": round(avg_response_time, 4),
            "predictions_per_minute": len(recent_predictions),
            "last_prediction": self.metrics["last_prediction"],
            "error_types": dict(self.metrics["error_types"]),
            "uptime_minutes": (time.time() - self.start_time) / 60,
        }


class MonitoringManager:
    """Monitoring Manager."""

    def __init__(self):
        self.middleware = None
        self.start_time = time.time()
        self.health_checks = []

        logger.info("Monitoring manager initialized")

    def get_health_status(self) -> dict[str, Any]:
        """Get basic health status."""
        if not self.middleware:
            return {"status": "monitoring_not_initialized"}

        metrics = self.middleware.get_metrics()

        # Determine health status based on simple rules
        status = "healthy"
        if metrics["error_rate"] > 10:  # More than 10% errors
            status = "degraded"
        if metrics["error_rate"] > 25:  # More than 25% errors
            status = "unhealthy"

        return {
            "status": status,
            "uptime_minutes": round(metrics["uptime_minutes"], 2),
            "total_requests": metrics["total_requests"],
            "total_predictions": metrics["total_predictions"],
            "error_rate": round(metrics["error_rate"], 2),
            "avg_response_time": metrics["avg_response_time"],
            "last_check": datetime.now().isoformat(),
        }


# Global monitoring manager instance
monitoring_manager = MonitoringManager()

# Simple lifespan event handlers for FastAPI


@asynccontextmanager
async def monitoring_lifespan(_app: FastAPI):
    """Simple lifespan context manager for monitoring."""
    # Start up
    logger.info("Starting simple monitoring...")

    yield

    # shutdown
    logger.info("Simple monitoring shutdown")


# FastAPI dependency


async def get_monitoring_manager() -> MonitoringManager:
    """Dependency to get monitoring manager."""
    return monitoring_manager


# Add simple monitoring endpoints


def add_monitoring_endpoints(app: FastAPI):
    """Add simple monitoring endpoints to FastAPI app."""

    @app.get("/monitoring/health")
    async def monitoring_health():
        """Get monitoring system health status."""
        return monitoring_manager.get_health_status()

    @app.get("/monitoring/metrics")
    async def get_metrics():
        """Get simple metrics."""
        if not monitoring_manager.middleware:
            return {"error": "Monitoring not initialized"}

        return monitoring_manager.middleware.get_metrics()

    @app.get("/monitoring/status")
    async def get_status():
        """Get simple status overview."""
        health = monitoring_manager.get_health_status()
        metrics = (
            monitoring_manager.middleware.get_metrics()
            if monitoring_manager.middleware
            else {}
        )

        return {
            "service": "ML Model API",
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "summary": {
                "requests_processed": metrics.get("total_requests", 0),
                "predictions_made": metrics.get("total_predictions", 0),
                "current_status": health.get("status", "unknown"),
            },
        }
