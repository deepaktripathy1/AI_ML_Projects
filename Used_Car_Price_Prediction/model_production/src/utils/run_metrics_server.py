"""Run FastAPI metrics server and background metrics collection."""

import time
import sys

from src.monitoring.metrics_tracker import ModelMetricsTracker
from src.monitoring.drift_detector import DriftDetector
from config.config import get_settings


if __name__ == "__main__":
    settings = get_settings()

    print("=" * 60)
    print("ML MODEL METRICS SERVICE")
    print("=" * 60)
    print("\n1. Start FastAPI metrics server in separate terminal:")
    print("   python metrics_api.py")
    print("\n2. Then run this script to start background collection")
    print("=" * 60)

    # Wait for user confirmation that FastAPI is running
    input("\nPress Enter once FastAPI metrics server is running...")

    # Verify FastAPI is accessible
    import requests
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("FastAPI metrics server is running\n")
        else:
            print("FastAPI metrics server not responding correctly")
            sys.exit(1)
    except Exception as e:
        print(f"Cannot connect to FastAPI metrics server: {e}")
        print("Please start it first: python metrics_api.py")
        sys.exit(1)

    try:
        # Initialize drift detector
        drift_detector = DriftDetector(
            reference_data_path=settings.monitoring_reference_data_path,
            drift_threshold=getattr(
                settings, "MONITORING_DRIFT_THRESHOLD", 0.1)
        )

        # Initialize metrics tracker (will POST to FastAPI)
        tracker = ModelMetricsTracker.get_instance(
            metrics_port=8001,
            drift_detector=drift_detector,
            enable_system_metrics=True,
            metrics_api_url="http://localhost:8001"
        )

        # Start background monitoring
        tracker.start_monitoring(update_interval=30)

        print("=" * 60)
        print("Background metrics collection running")
        print("=" * 60)
        print(f"Metrics endpoint: http://localhost:8001/metrics")
        print(f"Health check:     http://localhost:8001/health")
        print("\nPress Ctrl+C to stop.")
        print("=" * 60 + "\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nShutting down background collection...")
        tracker.stop_monitoring_thread()  # type: ignore
        print("✓ Background collection stopped")
