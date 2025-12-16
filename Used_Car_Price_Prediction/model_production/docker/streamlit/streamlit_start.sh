#!/bin/bash

# Streamlit startup script

set -e # exit on error

echo "=========================================="
echo "Starting Streamlit Service"
echo "=========================================="

# Environment variables with defaults
export ENVIRONMENT=${ENVIRONMENT:-production}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8501}
export USE_LOCAL_MODEL=${USE_LOCAL_MODEL:-True}
export API_BASE_URL=${FASTAPI_URL:-http://fastapi:8000}
export MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-120}
export SLEEP_INTERVAL=${SLEEP_INTERVAL:-3}

# Workers configuration
# Formula: (2 x $num_cores) + 1
export WORKERS=${WORKERS:-4}

USE_LOCAL_MODEL_LC="$(printf '%s' "$USE_LOCAL_MODEL" | tr '[:upper:]' '[:lower:]')"

echo "STARTUP: USE_LOCAL_MODEL=${USE_LOCAL_MODEL_LC} FASTAPI_URL=${API_BASE_URL}"

wait_for_fastapi() {
    start_ts=$(date +%s)
    end_ts=$((start_ts + MAX_WAIT_SECONDS))

echo "Waiting for FastAPI at ${API_BASE_URL}/health (timeout ${MAX_WAIT_SECONDS}s)..."

while [ "$(date +%s)" -le "$end_ts" ]; do
    # Try HTTP call
    # Prefer jq if available to parse JSON; fallback to grep
    if command -v jq >/dev/null 2>&1; then
      resp=$(curl -s --max-time 5 "${API_BASE_URL}/health" || true)
      # ensure non-empty
      if [ -n "$resp" ]; then
        status_code=$(printf '%s' "$resp" | jq -r '.status_code? // empty' 2>/dev/null || echo "")
        model_loaded=$(printf '%s' "$resp" | jq -r '.model_loaded? // false' 2>/dev/null || echo "false")
        if [ "$model_loaded" = "true" ]; then
          echo "FastAPI reports model_loaded = true"
          return 0
        fi
      fi
    else
      # fallback: grep for "model_loaded": true or "model_loaded":true
      resp=$(curl -s --max-time 5 "${API_BASE_URL}/health" || true)
      if printf '%s' "$resp" | grep -E '"model_loaded"[[:space:]]*:[[:space:]]*true' >/dev/null 2>&1; then
        echo "FastAPI reports model_loaded = true (grepped)"
        return 0
      fi
    fi

    # If response not ready, sleep and retry
    echo "FastAPI not ready yet — sleeping ${SLEEP_INTERVAL}s..."
    sleep "${SLEEP_INTERVAL}"
  done

  echo "Timed out waiting for FastAPI readiness after ${MAX_WAIT_SECONDS} seconds" >&2
  return 1
}

start_streamlit() {
    echo "Starting Streamlit..."
    exec streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0
}

echo "Configuration:"
echo " Environment: $ENVIRONMENT"
echo " Host: $HOST"
echo " Port: $PORT"
echo " Use Local Model: $USE_LOCAL_MODEL"
echo "=========================================="

if [ "${USE_LOCAL_MODEL_LC}" = "true" ] || [ "${USE_LOCAL_MODEL_LC}" = "1" ] || [ "${USE_LOCAL_MODEL_LC}" = "yes" ]; then
  echo "Using local model — will not wait for FastAPI."
  start_streamlit
else
  echo "Configured to use remote API. Checking FastAPI status before starting Streamlit..."
  if wait_for_fastapi; then
    echo "FastAPI ready — launching Streamlit (will use API for predictions)."
    start_streamlit
  else
    echo "FastAPI did not become ready in time. Exiting with non-zero status." >&2
    exit 2
  fi
fi