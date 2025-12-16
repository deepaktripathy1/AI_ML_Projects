# ===== Stage 1: Builder =====
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements file
COPY docker/fastapi/requirements.txt /app/requirements.txt

# Install dependencies to a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt

# ===== Stage 2: Runtime =====
FROM python:3.11-slim

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment to use venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application files
COPY config/ ./config/
COPY src/__init__.py ./src/
COPY src/api/ ./src/api/
COPY src/utils/ ./src/utils/
COPY src/data_preprocessing/data_cleaner.py ./src/data_preprocessing/data_cleaner.py
COPY src/data_preprocessing/feature_engineering.py ./src/data_preprocessing/feature_engineering.py
COPY src/monitoring/monitoring_middleware.py ./src/monitoring/monitoring_middleware.py
COPY models/ ./models/

# Copy startup script
COPY docker/fastapi/fastapi_start.sh /app/fastapi_start.sh
RUN chmod +x /app/fastapi_start.sh

COPY docker/bentoml/ /bentoml/
ENV BENTOML_HOME=/bentoml

# ===== FastAPI Service Stage =====
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/fastapi_start.sh"]