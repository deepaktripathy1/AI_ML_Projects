"""Configuration package initialization.

Provides centralized access to all configuration objects.
"""

from .config import Settings, get_settings
from .logging_config import LoggingConfig


# Initialize settings instance
settings = get_settings()

# Initialize logging configuration
logging_config = LoggingConfig(
    log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)

__all__ = ["Settings", "get_settings", "settings", "logging_config"]
