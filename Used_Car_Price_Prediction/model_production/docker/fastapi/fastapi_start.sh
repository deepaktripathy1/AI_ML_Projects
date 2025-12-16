#!/bin/bash

# FastAPI startup script

set -e # exit on error

echo "=========================================="
echo "Starting FastAPI Service"
echo "=========================================="

# User log level
RAW_LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Loguru
LOGURU_LEVEL="$(printf '%s' "$RAW_LOG_LEVEL" | tr '[:lower:]' '[:upper:]')"

# Uvicorn log level
UVICORN_LEVEL="$(printf '%s' "$RAW_LOG_LEVEL" | tr '[:upper:]' '[:lower:]')"

# Environment variables with defaults
export ENVIRONMENT=${ENVIRONMENT:-production}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export LOG_LEVEL="$LOGURU_LEVEL"

# Workers configuration
# Formula: (2 x $num_cores) + 1
export WORKERS=${WORKERS:-4}

echo "Configuration:"
echo " Environment: $ENVIRONMENT"
echo " Host: $HOST"
echo " Port: $PORT"
echo " Workers: $WORKERS"
echo " Log Level: $LOGURU_LEVEL"
echo " UvicornLevel: $UVICORN_LEVEL"
echo "=========================================="

# Check mode
if [ "$ENVIRONMENT" = "development" ]; then
    echo "Running in DEVELOPMENT mode with auto-reload"
    exec uvicorn src.api.endpoints:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$UVICORN_LEVEL"
else
    echo "Running in PRODUCTION mode with $WORKERS workers"
    exec uvicorn src.api.endpoints:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$UVICORN_LEVEL" \
        --timeout-keep-alive 65 \
        --access-log \
        --no-use-colors
fi