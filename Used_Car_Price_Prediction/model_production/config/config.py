"""Main configuration file."""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        validate_assignment=True,
        extra="allow",
    )

    # Project settings
    PROJECT_NAME: str = "Used Car Price Prediction"
    PROJECT_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, description="Debug mode flag")

    # UTF-8
    PYTHONIOENCODING: str = Field(default="utf-8")

    # Warnings
    PYTHONWARNINGS: str = Field(default="ignore::DeprecationWarning")

    # Database settings
    MONGODB_URL: str = Field(
        default="", description="MongoDB connection URL"
    )
    MONGODB_NAME: str = Field(
        default="used_car_raw_data", description="MongoDB database name"
    )
    MONGODB_COLLECTION: str = Field(
        default="raw_csv", description="MongoDB collection name"
    )

    # Hopsworks settings
    HOPSWORKS_API_KEY: str = Field(
        default="", description="Hopsworks API key"
    )
    HOPSWORKS_PROJECT: str = Field(
        default="", description="Hopsworks project name")
    HOPSWORKS_HOST: str = Field(
        default="c.app.hopsworks.ai", description="Hopsworks host"
    )

    # Mlflow settings
    MLFLOW_TRACKING_URI: str = Field(
        default="http://127.0.0.1:8080", description="MLflow tracking URI"
    )
    MLFLOW_EXPERIMENT_NAME: str = Field(
        default="used_car_price_prediction", description="MLflow experiment name"
    )
    MLFLOW_ARTIFACT_ROOT: str = Field(
        default="./mlruns", description="MLflow artifact root directory"
    )

    # FastAPI settings
    API_HOST: str = Field(default="0.0.0.0", description="API host address")
    API_PORT: int = Field(default=8000, ge=1, le=65535,
                          description="API port number")
    API_WORKERS: int = Field(
        default=1, ge=1, description="Number of API workers")

    # BentoML settings
    BENTO_MODEL_NAME: str = Field(
        default="used_car_price_model", description="BentoML model name"
    )
    BENTO_SERVICE_NAME: str = Field(
        default="used_car_price_prediction_service", description="BentoML service name"
    )
    BENTOML_CONFIG_PATH: str = Field(
        default="./deployment/bentoml/bentoml_configuration.yaml",
        description="BentoML configuration file path",
    )

    # Prefect settings
    PREFECT_HOME: str = Field(
        default="./prefect_automation", description="Prefect home directory"
    )
    PREFECT_API_URL: str = Field(
        default="", description="Prefect URL"
    )
    PREFECT_API_KEY: str = Field(
        default="", description="Prefect API Key"
    )
    PREFECT_UI_URL: str = Field(
        default="http://127.0.0.1:4200", description="Prefect UI"
    )
    DB_TRIGGER_INTERVAL_DAYS: int = Field(
        default=60, description="Interval in days for trigger file to run"
    )

    # Model settings
    TARGET_COLUMN: str = Field(
        default="price_usd", description="Target column for prediction"
    )
    TEST_SIZE: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Test set size ratio"
    )
    RANDOM_STATE: int = Field(
        default=42, description="Random state for reproducibility"
    )
    USE_LOCAL_MODEL: bool = Field(
        default=True, description="For Streamlit app"
    )

    # Feature Engineering Settings
    CURRENT_YEAR: int = Field(
        default=2025, ge=2000, le=2030, description="Current year for age calculation"
    )
    SERVICE_HISTORY_MAPPING: dict[str, int] = Field(
        default={"None": 0, "Partial": 1, "Full": 2},
        description="Service history mapping",
    )

    # Categorical Features for Encoding
    CATEGORICAL_FEATURES: list[str] = Field(
        default=["fuel_type", "brand", "transmission",
                 "color", "insurance_valid"],
        description="List of categorical features for encoding",
    )

    # Features to drop after encoding
    FEATURES_TO_DROP: list[str] = Field(
        default=[
            "make_year",
            "accidents_reported",
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
        ],
        description="List of features to drop after encoding",
    )

    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")

    # =================================================================
    # MONITORING SETTINGS
    # =================================================================

    # Prometheus settings
    MONITORING_PROMETHEUS_METRICS_PORT: int = Field(
        default=8090, description="Prometheus metrics port"
    )
    MONITORING_PROMETHEUS_SCRAPE_INTERVAL: int = Field(
        default=30, description="Prometheus scrape interval in seconds"
    )

    # Drift Detection Settings
    MONITORING_DRIFT_CHECK_INTERVAL_HOURS: int = Field(
        default=1, description="Drift check interval in hours"
    )
    MONITORING_DRIFT_THRESHOLD: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Data drift threshold"
    )
    MONITORING_DATA_QUALITY_THRESHOLD: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Data quality threshold"
    )

    # Alert Settings
    MONITORING_ENABLE_ALERTS: bool = Field(
        default=True, description="Enable monitoring alerts"
    )
    MONITORING_ALERT_EMAIL_RECIPIENTS: str = Field(
        default="deepaktripathy1@gmail.com", description="Alert email recipients"
    )
    MONITORING_SLACK_WEBHOOK_URL: str | None = Field(
        default=None, description="Slack webhook URL for alerts"
    )

    # System Monitoring Settings
    MONITORING_ENABLE_SYSTEM_METRICS: bool = Field(
        default=True, description="Enable system metrics collection"
    )
    MONITORING_SYSTEM_METRICS_INTERVAL: int = Field(
        default=60, description="System metrics collection interval in seconds"
    )

    # Storage Settings
    MONITORING_DATA_RETENTION_DAYS: int = Field(
        default=30, description="Data retention period in days"
    )
    MONITORING_MAX_PREDICTION_BUFFER_SIZE: int = Field(
        default=10000, description="Maximum prediction buffer size"
    )

    # Performance Thresholds
    MONITORING_MAX_PREDICTION_LATENCY_SECONDS: float = Field(
        default=1.0, description="Maximum acceptable prediction latency"
    )
    MONITORING_MAX_ERROR_RATE_PERCENT: float = Field(
        default=5.0, description="Maximum acceptable error rate percentage"
    )
    MONITORING_MIN_CONFIDENCE_THRESHOLD: float = Field(
        default=0.5, description="Minimum confidence threshold for predictions"
    )

    # File Paths
    MONITORING_DRIFT_REPORTS_DIR: str = Field(
        default="logs/drift_reports", description="Directory for drift reports"
    )
    MONITORING_EXPORTS_DIR: str = Field(
        default="logs/monitoring_reports",
        description="Directory for monitoring exports",
    )

    # Retraining automation control
    MONITORING_TEST_MODE: bool = Field(
        default=True,
        description="Control for auto retraining in Prefect"
    )

    # Grafana Settings (for Docker setup)
    GF_SECURITY_ADMIN_USER: str = Field(
        default="admin", description="Grafana admin username"
    )
    GF_SECURITY_ADMIN_PASSWORD: str = Field(
        default="admin", description="Grafana admin password"
    )

    # Paths
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        """Data directory path."""
        return self.BASE_DIR / "data"

    @property
    def logs_dir(self) -> Path:
        """Logs directory path."""
        return self.BASE_DIR / "logs"

    @property
    def models_dir(self) -> Path:
        """Models directory path."""
        return self.BASE_DIR / "models"

    @property
    def monitoring_reference_data_path(self) -> str:
        """Absolute path to monitoring reference data."""
        return str(self.BASE_DIR / "data" / "monitoring" / "reference_data.csv")

    @property
    def streamlit_image_path(self) -> Path:
        """Absolute path to streamlit background image."""
        return self.BASE_DIR / "streamlit_app" / "assets"

    @property
    def monitoring_drift_reports_path(self) -> Path:
        """Monitoring drift reports directory path."""
        return self.BASE_DIR / self.MONITORING_DRIFT_REPORTS_DIR

    @property
    def monitoring_exports_path(self) -> Path:
        """Monitoring exports directory path."""
        return self.BASE_DIR / self.MONITORING_EXPORTS_DIR

    @property
    def api_url(self) -> str:
        host = "127.0.0.1" if self.API_HOST == "0.0.0.0" else self.API_HOST
        return f"http://{host}:{self.API_PORT}"

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Create necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create monitoring directories
        self.monitoring_drift_reports_path.mkdir(parents=True, exist_ok=True)
        self.monitoring_exports_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get settings instance with lazy loading."""
    from pathlib import Path
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

    return Settings()


settings = get_settings()
