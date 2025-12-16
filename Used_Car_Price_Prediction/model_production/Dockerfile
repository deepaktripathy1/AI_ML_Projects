# Base stage with common dependencies
FROM python:3.11-slim AS base

# Set working dir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY . .

# Install the project in editable mode
RUN uv pip install --system -e .

# Create logs directory
RUN mkdir -p logs/drift_reports logs/monitoring_reports

# ===== FastAPI Service Stage =====
FROM base AS fastapi

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.endpoints:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ===== BentoML Service Stage =====
FROM base AS bentoml

EXPOSE 3001

WORKDIR /app

CMD ["bentoml", "serve", "deployment.bentoml.service:svc", "--host", "0.0.0.0", "--port", "3001"]

# ===== Streamlit Service Stage =====
FROM base AS streamlit

EXPOSE 8501

WORKDIR /app

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# ===== Metrics Server Stage =====
FROM base AS metrics

EXPOSE 8001

WORKDIR /app

CMD ["python", "run_metrics_server.py"]

# ===== Prefect Worker Stage =====
FROM base AS prefect-worker

WORKDIR /app

# Set Prefect API URL
ENV PREFECT_API_URL=http://prefect-server:4200/api

# Create entrypoint script for auto-deployment using pipeline_controller
RUN echo '#!/bin/bash\n\
    echo "Waiting for Prefect server to be ready..."\n\
    sleep 10\n\
    \n\
    echo "Deploying all Prefect flows using pipeline_controller..."\n\
    cd /app\n\
    \n\
    # Deploy all pipelines using your controller\n\
    python prefect_automation/pipeline_controller.py --deploy || true\n\
    \n\
    echo "Starting Prefect worker..."\n\
    prefect worker start --pool default-agent-pool\n\
    ' > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]