<p align="center">
  <a href="" rel="noopener">
 <img width=400px height=300px src="streamlit_app\assets\car_image.png" alt="Project logo"></a>
</p>

<h3 align="center">Used Car Price Prediction</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/deepaktripathy1/AI_ML_Projects/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/deepaktripathy1/AI_ML_Projects/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="left"> A complete end-to-end machine learning production system for predicting used car prices, featuring automated pipelines, model monitoring, drift detection, and comprehensive MLOps practices.
    <br> 
</p>

## 📝 Table of Contents

- [Overview](#overview)
- [ML Lifecycle Flow](#ml_lifecycle_flow)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project_structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick_start)
- [Component Usage](#component_usage)
- [Github Actions CI/CD](#github_actions_ci_cd)
- [Monitoring & Observability](#monitoring_obs)
- [API Documentation](#api_documentation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview <a name = "overview"></a>

This project demonstrates a production-grade machine learning system that predicts used car prices using Linear Regression. It implements MLOps best practices including:

- Automated data pipelines with Prefect orchestration
- Feature store integration with Hopsworks/Feast
- Experiment tracking with MLflow
- Model deployment with FastAPI and BentoML
- Real-time monitoring with Prometheus and Grafana
- Drift detection using Evidently AI
- CI/CD automation with GitHub Actions
- Interactive UI with Streamlit

## 🔄 ML Lifecycle Flow <a name = "ml_lifecycle_flow"></a>

<p align="center">
  <a href="" rel="noopener">
 <img width=1000px height=600px src="images/ML_Lifecycle.png" alt="ML Lifecycle"></a>
</p>

Key Components:
- Orchestration: Prefect manages all pipeline stages (Data, Training, Deployment, Monitoring)
- Tracking: MLflow tracks experiments, models, and artifacts
- Serving: FastAPI provides REST API endpoints for predictions
- Monitoring: Prometheus + Grafana + Evidently for observability
- Automation: GitHub Actions orchestrates the entire workflow
- User Interface: Streamlit provides interactive prediction interface

### ✨ Features <a name = "features"></a>

Core ML Capabilities

- ✅ Linear Regression model for price prediction
- ✅ Feature engineering and selection
- ✅ Automated hyperparameter tuning
- ✅ Model versioning and experiment tracking

MLOps Infrastructure

- ✅ Feature store integration (Hopsworks/Feast)
- ✅ Automated data pipelines with Prefect
- ✅ Model registry with MLflow
- ✅ REST API with FastAPI
- ✅ Real-time predictions
- ✅ Model monitoring and drift detection
- ✅ Docker containerization
- ✅ CI/CD with GitHub Actions

Monitoring & Observability

- ✅ Prometheus metrics collection
- ✅ Grafana dashboards
- ✅ Data drift detection
- ✅ Model performance tracking
- ✅ Automated alerting

### 🏗️ Architecture <a name = "architecture"></a>

Technology Stack

<p align="left">
  <a href="" rel="noopener">
 <img width=700px height=400px src="images/features_table.png" alt="Features Table"></a>
</p>

## 📁 Project Structure <a name = "project_structure"></a>

<p align="left">
  <a href="" rel="noopener">
 <img width=400px height=400px src="images/project_structure.png" alt="Project Structure"></a>
</p>

### 🔧 Prerequisites <a name = "prerequisites"></a>

- Python: 3.11 or higher
- UV: 0.1.0 or higher
- Docker: 20.10 or higher
- Docker Compose: 2.0 or higher
- Git: Latest version

Required Accounts & Services

- Hopsworks account (for feature store)
- MLflow tracking server (can be local or remote)
- MongoDB instance (MongoDB Atlas, local Docker, or any MongoDB service)

### 📦 Installation <a name = "installation"></a>

1. Clone the Repository

- git clone https://github.com/deepaktripathy1/AI_ML_Projects.git
- cd ai_ml_projects/used-car-price-prediction/model_production

2. Install UV

    #### Install UV
    curl -LsSf https://astral.sh/uv/install.sh | sh

    #### Add to PATH (if not automatically added)
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc

    #### Verify installation
    uv --version

3. Install Dependencies

    #### Create virtual environment
    uv venv

    #### Activate virtual environment
      ##### On Unix/Linux/macOS:
      source .venv/bin/activate

      ##### On Windows (Command Prompt):
      .venv\Scripts\activate

      ##### On Windows (PowerShell):
      .venv\Scripts\Activate.ps1

    #### Install project dependencies
    uv pip install -e .

    #### Install development dependencies
    uv pip install -e ".[dev]"

4. Setup Environment Variables

    cp .env.example .env

    Edit .env file with your configurations:

    #### MongoDB (use your MongoDB Atlas URI or local instance)
    - MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
    #### Or for local: mongodb://localhost:27017/
    - MONGODB_DATABASE=used_car_raw_data
    - MONGODB_COLLECTION=raw_csv
    #### Hopsworks
    - HOPSWORKS_API_KEY=your_api_key_here
    - HOPSWORKS_PROJECT=your_project_name
    #### MLflow
    - MLFLOW_TRACKING_URI=http://localhost:8080
    #### API
    - API_HOST=0.0.0.0
    - API_PORT=8000
    #### Monitoring
    - PROMETHEUS_PORT=9090
    - GRAFANA_PORT=3000

5. Start Infrastructure Services

    - uv venv
    - source .venv/bin/activate
    - uv pip install mlflow

    mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlruns \
      --host 0.0.0.0 \
      --port 8080

MongoDB Setup Options:
- MongoDB Atlas
  - Create a free cluster at MongoDB Atlas
  - Get your connection string and add it to .env
  - No local setup required!
 
## 🚀 Quick Start <a name="quick_start"></a>

#### Option 1: Using Docker

##### Build Docker images
    docker buildx build -f docker/fastapi/Dockerfile -t used-car-price-prediction-fastapi:latest
    docker buildx build -f docker/streamlit/Dockerfile -t used-car-price-prediction-streamlit:latest

  ##### OR
    docker compose build fastapi
    docker compose build streamlit (for fastapi API dependency)

##### Start FastAPI service
    docker run -d name fastapi -p 8000:8000 --env-file .env used-car-price-prediction-fastapi:latest

##### Start Streamlit service (for local model dependency)
    docker run -d name streamlit -p 8501:8501 --env-file .env used-car-price-prediction-streamlit:latest

  #### OR (for FastAPI model dependency in Streamlit)
    docker compose up -d --no-deps fastapi streamlit 

##### Check running containers
    docker ps

#### Option 2: Manual Setup
Start each component individually (see Component Usage section).

## 🎮 Component Usage <a name = "component_usage"></a>

#### 1. MLflow Server
    Start MLflow Tracking Server:
    mlflow server --host 127.0.0.1 --port 8080
    docker compose up mlflow-server

    Access MLflow UI:
    http://localhost:8080

    Key Features:

    Experiment tracking
    Model registry
    Artifact storage
    Run comparison

#### 2. Prefect Server & Workers
    Start Prefect Server:
    prefect server start

    Start Prefect Workers:
    prefect work-pool create --type process default-pool
    prefect worker start --pool default-agent-pool

    Deploy All Prefect Flows:
    python pipeline_controller.py --deploy

    Create Prefect Automation:
    python pipeline_controller.py --automate

    Trigger data pipeline run:
    python pipeline_controller.py --run

    Trigger individual deployments:
    prefect deployment run "data-pipeline/data-pipeline"
    prefect deployment run "training-pipeline/training-pipeline"
    prefect deployment run "deployment-pipeline/deployment-pipeline"
    prefect deployment run "monitoring-pipeline/monitoring-pipeline"

    Access Prefect UI:
    http://localhost:4200

#### 3. FastAPI Server
    Start API Server:
    # Development mode with auto-reload
    uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

    # Production mode with multiple workers
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # Using.bat script
    cd scripts
    start start_fastapi_service.bat

    # Docker
    docker run -d name fastapi -p 8000:8000 --env-file .env used-car-price-prediction-fastapi:latest

    # View Docker logs
    docker logs -f fastapi

    Access API Documentation:
    Swagger UI: http://localhost:8000/docs
    ReDoc: http://localhost:8000/redoc

    API Endpoints:
    # Health check
    curl http://localhost:8000/health

    # Make prediction
    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "make_year": 2020,
        "accidents_reported": 1,
        "fuel_type": "Petrol",
        "brand": "Toyota",
        "transmission": "Automatic",
        "color": "Black",
        "insurance_valid": "Yes",
        "service_history": "Full Service History",
        "mileage_kmpl": 11,
        "engine_cc": 1500,
        "owner_count": 2
      }'

    # Get model info
    curl http://localhost:8000/model/info

    # Get metrics
    curl http://localhost:8000/metrics

    Run Metrics Server:
    python run_metrics_server.py

    Access metrics at: http://localhost:9090/metrics

#### 4. Streamlit App
    Start Streamlit UI:
    # Standard mode
    uv run streamlit run streamlit_app/app.py

    # Custom port
    uv run streamlit run streamlit_app/app.py --server.port 8501

    # Using Docker for local model dependency
    docker run -d name streamlit -p 8501:8501 --env-file .env used-car-price-prediction-streamlit:latest

    # Using Docker for FastAPI model dependency
    docker compose up -d --no-deps fastapi streamlit

    # View Docker logs
    docker logs -f streamlit

    Access Streamlit:
    http://localhost:8501

    Features:

    Interactive price prediction
    Data visualization
    Model performance metrics
    Historical predictions view

#### 5. Prometheus and Alertmanager
    Start Prometheus:
    # Create prometheus.yml configuration file first

    # Using Docker
    docker run -d \
    --name prometheus \
    -p 9090:9090 \
    -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus

    # Using .bat script to start both prometheus and alertmanager
    cd scripts
    start start_prometheus_service.bat (put the prometheus and alertmanager folder in C:/)

    Access Prometheus and alertmanager:
    http://localhost:9090
    http://localhost:9093

    Key Metrics:

    prediction_request_total - Total prediction requests
    prediction_duration_seconds - Prediction latency
    model_prediction_value - Predicted values
    data_drift_score - Drift detection scores

#### 6. Grafana
    Start Grafana:
    # Using Docker
    docker run -d \
    --name grafana \
    -p 3000:3000 \
    -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
    -v grafana_data:/var/lib/grafana \
    grafana/grafana

    Access Grafana:
    http://localhost:3000

    Default Credentials:

    Username: admin
    Password: admin (change on first login)

    Setup Steps:

    Add Prometheus Data Source:

    Go to Configuration → Data Sources
    Click "Add data source"
    Select Prometheus
    URL: http://host.docker.internal:9090 (or http://172.17.0.1:9090 on Linux)
    Click "Save & Test"

    Import Dashboard:
    Located in: monitoring/grafana/dashboards/ml_monitoring.json

    Key Panels:

    Prediction request rate
    Model latency percentiles
    Error rate
    Data drift alerts
    Feature distribution

    Create Custom Dashboard:
    -- Prediction Rate
    rate(prediction_request_total[5m])

    -- Average Latency
    avg(prediction_duration_seconds)

    -- P95 Latency
    histogram_quantile(0.95, prediction_duration_seconds)

    -- Drift Score
    data_drift_score

#### 7. Monitoring Setup

    Initialize Monitoring:
    uv run python monitoring_setup.py

    Run Drift Detection:
    # Manual drift detection
    uv run python -c "from src.monitoring.drift_detector import DriftDetector; DriftDetector().check_drift()"
    # Automated via Prefect
    uv run prefect deployment run "monitoring-pipeline/monitoring-pipeline"

    View Reports:
    # Drift reports
    ls logs/drift_reports/

    # Monitoring reports
    ls logs/monitoring_reports/

#### 8. Clear Logs
    Clear all log files:
    uv run python src/utils/clear_logs.py

## 🤖 GitHub Actions CI/CD <a name = "github_actions_ci_cd"></a>

The project includes comprehensive GitHub Actions workflows that automate the entire ML lifecycle.

#### Workflow Triggers

- Push to main/develop: Full pipeline execution
- Pull requests: Code quality checks and testing
- Manual: Workflow dispatch for specific pipelines

#### Pipeline Jobs

##### 1. Code Quality (code-quality)

  - Black formatting check
  - Flake8 linting
  - MyPy type checking
  - Pytest with coverage

##### 2. Build Docker (build-docker)

  - Build and push Docker images
  - Tag with latest and commit SHA

##### 3. Setup Infrastructure (setup-infrastructure)

  - Start MongoDB and MLflow
  - Health checks

##### 4. Data Pipeline (data-pipeline)

  - Run Prefect data ingestion
  - Data preprocessing
  - Feature engineering


##### 5. Training Pipeline (training-pipeline)

  - Model training
  - Experiment tracking
  - Model validation

##### 6. Deployment Pipeline (deployment-pipeline)

  - Model deployment
  - API smoke tests

##### 7. Monitoring Pipeline (monitoring-pipeline)

  - Drift detection
  - Performance monitoring
  - Alert generation

##### 8. Deploy Production (not covered in this project)

  - Production deployment (main branch only)
  - Slack notifications

##### 9. Performance Testing (not covered in this project)

  - Load testing with Locust

#### Manual Trigger
    # Via GitHub CLI

    gh workflow run ml-pipeline.yml -f pipeline_type=training

#### Required Secrets
    Configure in GitHub Repository Settings → Secrets:

    DOCKER_USERNAME
    DOCKER_PASSWORD
    MONGODB_URI
    HOPSWORKS_API_KEY
    MLFLOW_TRACKING_URI
    SLACK_WEBHOOK (optional)

#### Workflow Status Badges
<div align="left">

[![CI/CD Pipeline]((https://github.com/deepaktripathy1/AI_ML_Projects/workflows/ML%20Pipeline%20CI/CD/badge.svg))](https://github.com/deepaktripathy1/AI_ML_Projects/actions)

</div>

## 📊 Monitoring & Observability <a name = "monitoring_obs"></a>

#### Key Metrics

<p align="left">
  <a href="" rel="noopener">
 <img width=500px height=200px src="images/key_metrics.png" alt="Key Metrics"></a>
</p>

#### Drift Detection
    The system uses Evidently AI for drift detection:

    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()
    report = detector.detect_data_drift()

    if report['drift_detected']:
      print("⚠️  Drift detected! Retraining recommended.")

#### Alerts
Alerts are triggered when:

- Data drift score exceeds threshold
- Model performance drops below target
- API latency exceeds SLA
- Error rate spikes

## 📖 API Documentation <a name = "api_documentation"></a>

#### Prediction Endpoint

    curl -X 'POST' \
      'http://localhost:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "make_year": 2010,
      "accidents_reported": 1,
      "fuel_type": "Diesel",
      "brand": "Kia",
      "transmission": "Manual",
      "color": "Black",
      "insurance_valid": "Yes",
      "service_history": "Partial Service History",
      "mileage_kmpl": 12,
      "engine_cc": 1000,
      "owner_count": 1
    }'

#### Response:

    {
      "predicted_price": 5483.663305955514,
      "confidence_score": 0.8,
      "input_validation": {
        "status": "valid",
        "warnings": [],
        "errors": []
      },
      "preprocessing_info": {
        "car_age_calculated": 15,
        "has_accident": true,
        "service_history_encoded": {
          "None": 0,
          "Partial Service History": 1,
          "Full Service History": 2
        },
        "categorical_features_encoded": [
          "fuel_type",
          "brand",
          "transmission",
          "color",
          "insurance_valid"
        ],
        "selected_features_count": 25,
        "preprocessing_timestamp": "2025-12-16T15:58:36.043146"
      },
      "prediction_metadata": {
        "model_version": "latest",
        "prediction_timestamp": "2025-12-16T15:58:36.043146",
        "processing_time_ms": 521.75,
        "features_used": 25,
        "car_summary": "Kia 2010 (Black)"
      }
    }

#### Batch Prediction

    curl -X 'POST' \
      'http://localhost:8000/predict-batch' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "cars": [
        {
          "make_year": 2005,
          "accidents_reported": 3,
          "fuel_type": "Petrol",
          "brand": "Tesla",
          "transmission": "Automatic",
          "color": "White",
          "insurance_valid": "No",
          "service_history": "Full Service History",
          "mileage_kmpl": 14,
          "engine_cc": 1500,
          "owner_count": 2
        },
        { 
          "make_year": 2010,
          "accidents_reported": 1,
          "fuel_type": "Diesel",
          "brand": "Kia",
          "transmission": "Manual",
          "color": "Black",
          "insurance_valid": "Yes",
          "service_history": "Partial Service History",
          "mileage_kmpl": 12,
          "engine_cc": 1000,
          "owner_count": 1
        }
      ]
    }'

## 🧪 Testing <a name = "testing"></a>

#### Run Tests

    # Run all tests
    uv run pytest

    # Run with coverage
    uv run pytest --cov=src --cov-report=html

    # Run specific test file
    uv run pytest tests/test_data_ingestion.py

    # Run with verbose output
    uv run pytest -v

#### Test Coverage

    # Generate coverage report
    uv run pytest --cov=src --cov-report=term-missing

    # View HTML report
    open htmlcov/index.html

#### Load Testing

    # Install locust
    uv pip install locust

    # Run load test
    locust -f tests/load_test.py --host http://localhost:8000

## 🚢 Deployment <a name = "deployment"></a>

#### Docker Deployment

    # Build FastAPI image
    docker build -f docker/fastapi/Dockerfile -t used-car-price-prediction-fastapi:latest .
    docker compose build fastapi

    # Build Streamlit image
    docker build -f docker/streamlit/Dockerfile -t used-car-price-prediction-streamlit:latest .
    docker compose build streamlit

    # Tag images for registry
    docker tag used-car-price-prediction-fastapi:latest yourusername/used-car-price-prediction-fastapi:latest
    docker tag used-car-price-prediction-streamlit:latest yourusername/used-car-price-prediction-streamlit:latest

    # Push to Docker Hub
    docker push yourusername/used-car-price-prediction-fastapi:latest
    docker push yourusername/used-car-price-prediction-streamlit:latest

    # Run containers
    docker run -d -p 8000:8000 --env-file .env used-car-price-prediction-fastapi:latest
    docker run -d -p 8501:8501 --env-file .env used-car-price-prediction-streamlit:latest

#### BentoML Deployment

    # Build Bento
    cd deployment/bentoml
    bentoml build

    # Serve locally
    bentoml serve service:svc
    cd scripts
    start start_bentoml_service.bat

    # Containerize
    bentoml containerize car_price_predictor:latest

    # Deploy to BentoCloud
    bentoml deployment create car-price-api --bento car_price_predictor:latest

## 🐛 Troubleshooting <a name = "troubleshooting"></a>

#### MongoDB Connection Error:

    # If using local MongoDB with Docker:
    # Check MongoDB is running
    docker ps | grep mongodb

    # Verify connection
    docker exec -it mongodb mongosh --eval "db.adminCommand('ping')"

    # Check logs
    docker logs mongodb

    # If using MongoDB Atlas:
    # Verify your connection string in .env
    # Check if your IP is whitelisted in Atlas Network Access
    # Verify username and password are correct

#### MLflow Not Starting:

    # Check port availability
    lsof -i :5000

    # Kill process using port 5000 if needed
    kill -9 $(lsof -t -i:5000)

    # Restart MLflow
    uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 8080

#### Docker Container Issues:

    # Check all running containers
    docker ps -a

    # Check container logs
    docker logs fastapi
    docker logs streamlit

    # Restart container
    docker restart fastapi
    docker restart streamlit

    # Remove and recreate container
    docker rm -f fastapi
    docker run -d -p 8000:8000 --env-file .env used-car-price-prediction-fastapi:latest

#### Prefect Worker Issues:

    # Reset Prefect database
    uv run prefect database reset

    # Recreate work pool
    uv run prefect work-pool create --type process default-pool

#### Import Errors:

    # Reinstall dependencies
    uv pip install -e . --no-cache

    # Update dependencies
    uv pip install -e . --upgrade

#### Debug Mode
Enable debug logging:

    # config/logging_config.py
    import logging
    logging.basicConfig(level=logging.DEBUG)

## 🤝 Contributing <a name = "contributing"></a>

Contributions are welcome! Please follow these steps:

- Fork the repository
- Create a feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

#### Code Style

    # Format code
    uv run black .

    # Lint
    uv run flake8 src/

    # Type check
    uv run mypy src/

## 📝 License
This project is licensed under the MIT License.

## 👥 Authors

Deepak Tripathy - [GitHub](https://github.com/deepaktripathy1/AI_ML_Projects)

## 🙏 Acknowledgments

- Hopsworks for feature store
- MLflow for experiment tracking
- Prefect for orchestration
- Evidently AI for drift detection

## 📞 Support
For support, email deepaktripathy1@gmail.com or open an issue on GitHub.

#### Built using MLOps best practices

