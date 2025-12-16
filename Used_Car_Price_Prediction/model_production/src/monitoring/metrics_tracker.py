"""Model monitoring metrics tracker for Prometheus integration."""

import json
import threading
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import psutil
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
    CollectorRegistry
)

from config.config import get_settings
from config.logging_config import LoggingConfig
from monitoring.drift_detector import DriftDetector


logger = LoggingConfig.get_logger(__name__)

# Global singleton instance
_METRICS_TRACKER_INSTANCE = None
_METRICS_TRACKER_LOCK = threading.Lock()


class ModelMetricsTracker:
    """Metrics tracker for model monitoring with Prometheus integration.

    Tracks prediction metrics, data quality, drift detection, and system health.
    """

    def __init__(
        self,
        metrics_port: int = 8001,
        # metrics_port: int = 8090,
        drift_detector: DriftDetector | None = None,
        enable_system_metrics: bool = True,
        metrics_api_url: str = "http://localhost:8001"
    ):
        """Initialize metrics tracker.

        Args:
            metrics_port: Port for Prometheus metrics endpoint
            drift_detector: DriftDetector instance
            enable_system_metrics: Whether to collect system metrics
        """
        self.metrics_port = metrics_port
        self.drift_detector = drift_detector
        self.enable_system_metrics = enable_system_metrics
        self.metrics_api_url = metrics_api_url
        self.start_time: float | None = None

        # Set up HTTP session for posting metrics
        self._setup_http_session()

        # HTTP server state
        # self.http_server_started = False
        # self.http_server_thread = None

        # Create custom registry
        self.registry = CollectorRegistry()

        # Initialize metrics
        self._init_prediction_metrics()
        self._init_drift_metrics()
        if enable_system_metrics:
            self._init_system_metrics()

        # Data storage for calculations
        self.prediction_history = deque(
            maxlen=10000)  # Store last 10k predictions
        self.drift_history = deque(maxlen=100)  # Store last 100 drift checks

        # Thread safety
        self.lock = threading.Lock()

        # Background monitoring
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        logger.info(f"Metrics tracker initialized on port {metrics_api_url}")

    def _setup_http_session(self):
        """Setup requests session with retry."""
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def _init_prediction_metrics(self):
        """Initialize prediction-related metrics."""
        # Prediction counters
        self.predictions_total = Counter(
            "model_predictions_total",
            "Total number of predictions made",
            ["model_version", "endpoint"],
            registry=self.registry,
        )
        self.prediction_errors = Counter(
            "model_prediction_errors_total",
            "Total number of prediction errors",
            ["error_type"],
            registry=self.registry,
        )

        # Prediction latency
        self.prediction_latency = Histogram(
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
            registry=self.registry,
        )

        # Prediction values
        self.prediction_value = Histogram(
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
            registry=self.registry,
        )

        # Confidence scores
        self.confidence_score = Histogram(
            "model_confidence_score",
            "Distribution of prediction confidence scores",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=self.registry,
        )

        # Current prediction statistics
        self.current_prediction_mean = Gauge(
            "model_prediction_mean_dollars",
            "Current mean of predictions",
            registry=self.registry,
        )

        self.current_prediction_std = Gauge(
            "model_prediction_std_dollars",
            "Current standard deviation of predictions",
            registry=self.registry,
        )

        # Batch prediction metrics
        self.batch_size = Histogram(
            "model_batch_prediction_size",
            "Size of batch predictions",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
            registry=self.registry,
        )

        # Model info
        self.model_info = Info(
            "model_info",
            "Information about the current model",
            registry=self.registry
        )

    def _init_drift_metrics(self):
        """Initialize drift detection metrics."""

        # Drift detection status
        self.data_drift_detected = Gauge(
            "model_data_drift_detected",
            "Whether data drift is detected (1 for yes, 0 for no)",
            registry=self.registry,
        )

        self.model_drift_detected = Gauge(
            "model_performance_drift_detected",
            "Whether model performance drift is detected (1 for yes, 0 for no)",
            registry=self.registry,
        )

        self.prediction_drift_detected = Gauge(
            "model_prediction_drift_detected",
            "Whether prediction drift is detected (1 for yes, 0 for no)",
            registry=self.registry,
        )

        # Drift scores
        self.data_drift_score = Gauge(
            "model_data_drift_score",
            "Current data drift score",
            registry=self.registry
        )

        self.model_performance_change = Gauge(
            "model_performance_change",
            "Change in model performance (RÂ² score change)",
            registry=self.registry,
        )

        # Feature drift
        self.feature_drift_count = Gauge(
            "model_drifted_features_count",
            "Number of features with detected drift",
            registry=self.registry,
        )

        # Individual feature drift scores
        self.feature_drift_score = Gauge(
            "model_feature_drift_score",
            "Drift score for each feature",
            ["feature"],
            registry=self.registry
        )

        # Drift check frequency
        self.drift_checks_total = Counter(
            "model_drift_checks_total",
            "Total number of drift checks performed",
            ["drift_type"],
            registry=self.registry,
        )

        # Model test failures
        self.model_test_failures = Gauge(
            "model_test_failures_count",
            "Number of failed model performance tests",
            registry=self.registry,
        )

        # Performance change metrics
        self.performance_r2_change = Gauge(
            "model_performance_r2_change",
            "Change in RÂ² score compared to reference",
            registry=self.registry,
        )

        self.performance_mae_change = Gauge(
            "model_performance_mae_change",
            "Change in MAE compared to reference",
            registry=self.registry,
        )

    def _init_system_metrics(self):
        """Initialize system health metrics."""

        # CPU and Memory
        self.cpu_usage_percent = Gauge(
            "system_cpu_usage_percent",
            "Current CPU usage percentage",
            registry=self.registry,
        )

        self.memory_usage_percent = Gauge(
            "system_memory_usage_percent",
            "Current memory usage percentage",
            registry=self.registry,
        )

        self.disk_usage_percent = Gauge(
            "system_disk_usage_percent",
            "Current disk usage percentage",
            ["mount_point"],
            registry=self.registry,
        )

        # Service health
        self.service_uptime_seconds = Gauge(
            "service_uptime_seconds",
            "Service uptime in seconds",
            registry=self.registry,
        )

        self.service_health_status = Gauge(
            "service_health_status",
            "Service health status (1 for healthy, 0 for unhealthy)",
            registry=self.registry,
        )

    def record_prediction(
        self,
        prediction_value: float,
        confidence_score: float,
        latency_seconds: float,
        model_version: str = "latest",
        endpoint: str = "predict",
    ):
        """Record a prediction event."""
        with self.lock:
            # Store locally
            self.prediction_history.append({
                "value": prediction_value,
                "confidence": confidence_score,
                "timestamp": datetime.now(),
                "latency": latency_seconds,
            })

            # Update local statistics
            self._update_prediction_statistics()

            # POST to FastAPI
            try:
                response = self.session.post(
                    f"{self.metrics_api_url}/api/metrics/prediction",
                    json={
                        "prediction_value": prediction_value,
                        "confidence_score": confidence_score,
                        "latency_seconds": latency_seconds,
                        "model_version": model_version,
                        "endpoint": endpoint
                    },
                    timeout=5
                )
                if response.status_code != 200:
                    logger.warning(
                        f"Failed to post prediction metric: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to post prediction to FastAPI: {e}")

    def record_drift_check(
        self,
        drift_results: dict[str, Any],
        model_results: dict[str, Any] | None = None,
    ):
        """Record drift detection results with updated API support."""
        with self.lock:
            try:
                # Post metrics to FastAPI
                response = self.session.post(
                    url=f"{
                        self.metrics_api_url
                    }/api/metrics/drift",
                    json={
                        "drift_results": drift_results,
                        "model_results": model_results
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    logger.info("Drift metrics posted to FastAPI successfully")
                else:
                    logger.warning(f"Failed to post drift metrics: {
                        response.status_code}"
                    )

                # Update drifted features count
                drifted_features = drift_results.get("drifted_features", [])
                self.feature_drift_count.set(len(drifted_features))

                # Update individual feature drift scores
                drift_details = drift_results.get("drift_details", {})
                for feature, details in drift_details.items():
                    drift_score = details.get("drift_score", 0.0)

                    self.feature_drift_score.labels(feature).set(drift_score)

                # Store drift history
                self.drift_history.append(
                    {
                        "timestamp": datetime.now(),
                        "data_drift": drift_results,
                        "model_drift": model_results,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to post drift metrics to FastAPI: {e}")

    def _update_prediction_statistics(self):
        """Update running prediction statistics."""
        if not self.prediction_history:
            return

        # Calculate recent statistics (last 1000 predictions)
        recent_predictions = list(self.prediction_history)[-1000:]
        values = [p["value"] for p in recent_predictions]

    def update_system_metrics(self):
        """Update system health metrics."""
        if not self.enable_system_metrics:
            return

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage("/")

            # POST to FastAPI
            try:
                response = self.session.post(
                    f"{self.metrics_api_url}/api/metrics/system",
                    json={
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "disk_percent": (disk.used / disk.total) * 100,
                        "mount_point": "/"
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    logger.debug(
                        "System metrics posted to FastAPI successfully")
                else:
                    logger.warning(
                        f"Failed to post system metrics: {
                            response.status_code
                        }"
                    )

            except requests.exceptions.Timeout:
                logger.debug("Timeout posting system metrics")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error posting system metrics: {e}")

        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")

    def start_monitoring(self, update_interval: int = 60):
        """Start background monitoring thread."""

        def monitor():
            self.start_time = time.time()

            while not self.stop_monitoring.is_set():
                try:
                    self.update_system_metrics()
                    time.sleep(update_interval)

                except Exception as e:
                    logger.error(f"Error in monitoring thread: {e}")
                    time.sleep(5)

        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=monitor, daemon=True)
            self.monitoring_thread.start()
            logger.info(
                f"Background monitoring started (update interval: {update_interval}s)"
            )

    # def _update_loop(self, update_interval: int):
        # while not self._stop_event.is_set():
            # try:
            # self.cpu_usage_percent.set(psutil.cpu_percent())
            # self.memory_usage_percent.set(psutil.virtual_memory().percent)
            # self.service_uptime_seconds.set(time.time() - self.start_time)
            # self.service_health_status.set(1)
            # except Exception as e:
            # self.service_health_status.set(0)
            # logger.warning(f"System metrics update failed: {e}")
            # time.sleep(update_interval)

    def stop_monitoring_thread(self):
        """Stop background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=5)
            logger.info("Background monitoring stopped")

    # def start_metrics_server(self):
        # """Start Prometheus metrics server in a separate thread."""
        # try:
            # def run_server():
            # try:
            # start_http_server(self.metrics_port,
            # registry=self.registry)
            # self.service_health_status.set(1)
            # logger.info(
            # f"Metrics HTTP server started on http://localhost:{
            # self.metrics_port}/metrics"
            # )
            # except OSError as e:
            # if "Address already in use" in str(e):
            # logger.warning(f"Port {
            # self.metrics_port
            # } already in use - metrics server may already be running"
            # )
            # else:
            # raise
            # except Exception as e:
            # logger.error(f"Failed to start metrics server: {e}")
            # raise

            # Verify server is accessible
            # import requests
            # try:
            # response = requests.get(
            # f"http://localhost:{self.metrics_port}/metrics", timeout=2
            # )
            # if response.status_code == 200:
            # logger.info(
            # f"Metrics endpoint verified at http://localhost:{
            # self.metrics_port}/metrics"
            # )
            # else:
            # logger.warning(
            # f"Metrics endpoint returned status {
            # response.status_code}"
            # )
            # except Exception as e:
            # logger.warning(f"Could not verify metrics endpoint: {e}")

        # except Exception as e:
            # logger.error(f"Failed to start metrics server: {e}")
            # raise

    def get_prediction_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get prediction summary for the last N hours."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_predictions = [
                p for p in self.prediction_history if p["timestamp"] >= cutoff_time
            ]

            if not recent_predictions:
                return {
                    "total_predictions": 0,
                    "avg_prediction_value": 0,
                    "avg_confidence": 0,
                    "avg_latency_ms": 0,
                }

            values = [p["value"] for p in recent_predictions]
            confidences = [p["confidence"] for p in recent_predictions]
            latencies = [p["latency"] for p in recent_predictions]

            return {
                "total_predictions": len(recent_predictions),
                "avg_prediction_value": np.mean(values),
                "min_prediction_value": np.min(values),
                "max_prediction_value": np.max(values),
                "std_prediction_value": np.std(values),
                "avg_confidence": np.mean(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "avg_latency_ms": np.mean(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000,
                "p99_latency_ms": np.percentile(latencies, 99) * 1000,
            }

    def get_drift_summary(self) -> dict[str, Any]:
        """Get drift detection summary."""
        with self.lock:
            if not self.drift_history:
                return {
                    "last_check": None,
                    "data_drift_detected": False,
                    "model_drift_detected": False,
                    "prediction_drift_detected": False,
                    "drift_score": 0.0,
                    "drifted_features": [],
                    "failed_tests": 0
                }

            latest_drift = self.drift_history[-1]

            return {
                "last_check": latest_drift["timestamp"].isoformat(),
                "data_drift_detected": latest_drift["data_drift"].get(
                    "dataset_drift_detected", False
                ),
                "model_drift_detected": latest_drift["model_drift"].get(
                    "performance_degradation", False
                )
                if latest_drift["model_drift"]
                else False,
                "prediction_drift_detected": latest_drift["model_drift"].get(
                    "prediction_drift_detected", False
                )
                if latest_drift["model_drift"]
                else False,
                "drift_score": latest_drift["data_drift"].get(
                    "drift_score", 0.0
                ),
                "drifted_features": latest_drift["data_drift"].get(
                    "drifted_features", []
                ),
                "drifted_features_count": len(
                    latest_drift["data_drift"].get("drifted_features", [])
                ),
                "failed_tests": len(
                    latest_drift["model_drift"].get("test_failures", [])
                ) if latest_drift["model_drift"] else 0,
                "total_checks": len(self.drift_history),
            }

    def export_monitoring_data(self, output_file: str | None = None) -> str:
        """Export monitoring data to JSON file."""
        settings = get_settings()

        try:
            collected_metrics = list(self.feature_drift_score.collect())
            if collected_metrics:
                metric_samples = collected_metrics[0].samples
                feature_scores = {
                    sample.labels["feature"]: sample.value for sample in metric_samples if "feature" in sample.labels
                }
            else:
                feature_scores = {}
        except Exception as e:
            logger.error(f"Failed to collect feature drift scores: {e}")
            feature_scores = {}

        monitoring_data = {
            "export_timestamp": datetime.now().isoformat(),
            "prediction_summary_24h": self.get_prediction_summary(24),
            "drift_summary": self.get_drift_summary(),
            "recent_predictions": [
                {
                    "timestamp": p["timestamp"].isoformat(),
                    "value": p["value"],
                    "confidence": p["confidence"],
                    "latency": p["latency"],
                }
                # Last 100 predictions
                for p in list(self.prediction_history)[-100:]
            ],
            "recent_drift_checks": [
                {
                    "timestamp": d["timestamp"].isoformat(),
                    "data_drift_detected": d["data_drift"].get(
                        "dataset_drift_detected", False
                    ),
                    "drift_score": d["data_drift"].get("drift_score", 0.0),
                    "drifted_features": d["data_drift"].get("drifted_features", []),
                    "model_drift_detected": d["model_drift"].get(
                        "performance_degradation", False
                    )
                    if d["model_drift"]
                    else False,
                    "failed_tests": len(d["model_drift"].get(
                        "test_failures", []))
                    if d["model_drift"]
                    else 0,
                }
                for d in list(self.drift_history)[-10:]  # Last 10 drift checks
            ],
            "feature_drift_scores": feature_scores
        }

        if output_file is None:
            output_file = (
                f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        monitor_path = settings.logs_dir / "monitoring_reports"
        output_path = monitor_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(monitoring_data, f, indent=2)

        logger.info(f"Monitoring data exported to {output_path}")
        return str(output_path)

    # def reset_metrics(self):
        # """Reset all metrics (use with caution)."""
        # with self.lock:
        # Clear histories
        # self.prediction_history.clear()
        # self.drift_history.clear()

        # Reset gauges
        # self.data_drift_detected.set(0)
        # self.model_drift_detected.set(0)
        # self.prediction_drift_detected.set(0)
        # self.data_drift_score.set(0)
        # self.model_performance_change.set(0)
        # self.feature_drift_count.set(0)
        # self.current_prediction_mean.set(0)
        # self.current_prediction_std.set(0)
        # self.model_test_failures.set(0)
        # self.performance_r2_change.set(0)
        # self.performance_mae_change.set(0)

        # Reset feature drift scores
        # try:
        # collected_metrics = list(self.feature_drift_score.collect())
        # if collected_metrics:
        # metric_samples = collected_metrics[0].samples
        # for sample in metric_samples:
        # if "feature" in sample.labels:
        # self.feature_drift_score.labels(
        # feature=sample.labels["feature"]
        # ).set(0.0)
        # except Exception as e:
        # logger.error(f"Failed to reset feature drift scores: {e}")

        # logger.warning("All metrics have been reset")

    @classmethod
    def get_instance(
        cls,
        metrics_port: int = 8001,
        drift_detector=None,
        enable_system_metrics: bool = True,
        metrics_api_url: str = "http://localhost:8001"
    ):
        """Get or create singleton instance of metrics tracker."""
        global _METRICS_TRACKER_INSTANCE

        with _METRICS_TRACKER_LOCK:
            if _METRICS_TRACKER_INSTANCE is None:
                _METRICS_TRACKER_INSTANCE = cls(
                    metrics_port=metrics_port,
                    drift_detector=drift_detector,
                    enable_system_metrics=enable_system_metrics,
                    metrics_api_url=metrics_api_url
                )
                logger.info("Created new metrics tracker singleton instance")
            else:
                logger.info(
                    "Reusing existing metrics tracker singleton instance")
                # Update drift detector if provided
                if drift_detector is not None:
                    _METRICS_TRACKER_INSTANCE.drift_detector = drift_detector

            return _METRICS_TRACKER_INSTANCE


class MonitoringOrchestrator:
    """Orchestrates the monitoring workflow."""

    def __init__(
        self,
        drift_detector: DriftDetector,
        metrics_tracker: ModelMetricsTracker,
        monitoring_interval: int = 3600,  # 1 hour
    ):
        """Initialize monitoring orchestrator.

        Args:
            drift_detector: DriftDetector instance
            metrics_tracker: ModelMetricsTracker instance
            monitoring_interval: Interval between drift checks in seconds
        """
        self.drift_detector = drift_detector
        self.metrics_tracker = metrics_tracker
        self.monitoring_interval = monitoring_interval

        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()

        logger.info("Monitoring orchestrator initialized")

    def run_comprehensive_monitoring(
        self,
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Run comprehensive monitoring including drift detection and metrics update.

        Args:
            current_data: Current production data
            reference_data: Reference data for comparison

        Returns:
            Dict containing comprehensive monitoring results
        """
        try:
            logger.info("Starting comprehensive monitoring check")

            # Run drift detection
            drift_results = self.drift_detector.detect_data_drift(
                current_data, reference_data
            )

            # Run model drift detection
            model_results = self.drift_detector.detect_model_drift(
                current_data, reference_data
            )

            # Update metrics tracker
            self.metrics_tracker.record_drift_check(
                drift_results, model_results)

            # Generate monitoring summary
            quality_results = drift_results.get("data_quality_issues", [])
            monitoring_summary = self.drift_detector.generate_monitoring_summary(
                drift_results, model_results or {}, quality_results
            )

            # Save results
            _ = self.drift_detector.save_monitoring_results(
                monitoring_summary
            )

            logger.info(
                f"Comprehensive monitoring completed. Status: {
                    monitoring_summary['overall_status']}"
            )

            return monitoring_summary

        except Exception as e:
            logger.error(f"Comprehensive monitoring failed: {e}")
            raise

    def start_automated_monitoring(
        self,
        data_source_callback: Callable[[], pd.DataFrame],
        reference_data: pd.DataFrame,
    ):
        """Start automated monitoring with periodic checks.

        Args:
            data_source_callback: Function that returns current data
            reference_data: Reference data for comparison
        """

        def monitor():
            while not self.stop_monitoring.is_set():
                try:
                    # Get current data
                    current_data = data_source_callback()

                    if current_data is not None and not current_data.empty:
                        # Run monitoring
                        self.run_comprehensive_monitoring(
                            current_data, reference_data)
                    else:
                        logger.warning(
                            "No current data available for monitoring")

                    # Wait for next interval
                    self.stop_monitoring.wait(self.monitoring_interval)

                except Exception as e:
                    logger.error(f"Error in automated monitoring: {e}")
                    # Wait a bit before retrying
                    self.stop_monitoring.wait(
                        min(300, self.monitoring_interval))

        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(
                target=monitor, daemon=True)
            self.monitoring_thread.start()
            logger.info(
                f"Automated monitoring started with {
                    self.monitoring_interval}s interval"
            )

    def stop_automated_monitoring(self):
        """Stop automated monitoring."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=10)
            logger.info("Automated monitoring stopped")
