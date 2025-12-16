"""Logging configuration for the project."""

import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    """Centralized logging configuration."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/app.log", description="Log file path")

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up the logging configuration for the application."""
        # Remove default handler
        logger.remove()

        # Ensure log directory exists
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Console handler
        logger.add(
            sys.stdout,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            colorize=True,
        )

        # File handler
        logger.add(
            self.log_file,
            level=self.log_level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            rotation="10MB",
            retention="1 month",
            compression="zip",
        )

    @staticmethod
    def get_logger(name: str):
        """Get logger instance."""
        return logger.bind(name=name)
