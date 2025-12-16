#!/usr/bin/env bash
# ===========================================
#  Start ML Metrics Tracker Service (port 8090)
# ===========================================

echo "-------------------------------------------"
echo "Starting ML metrics tracker on port 8090..."
echo "-------------------------------------------"

# Activate your virtual environment
source ../.venv/Scripts/activate

# Run your monitoring pipeline continuously
python - <<'EOF'
from src.monitoring.metrics_tracker import ModelMetricsTracker
from prometheus_client import start_http_server
import time, signal

metrics_tracker = ModelMetricsTracker()
metrics_tracker.start_metrics_server()
print("✅ Metrics tracker running at http://localhost:8090/metrics")

# Keep alive indefinitely
signal.signal(signal.SIGINT, lambda s, f: exit(0))
while True:
    time.sleep(5)
EOF
