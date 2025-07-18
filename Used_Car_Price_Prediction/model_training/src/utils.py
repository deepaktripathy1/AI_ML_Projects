import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import mlflow
import mlflow.sklearn


def setup_logging(log_level: str = "INFO", log_dir: str = "logs/") -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'app.log')),
            logging.StreamHandler()
        ]
    )


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object as pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Saved object to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logging.info(f"Loaded object from {filepath} ")
    return obj


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:  # type: ignore
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded JSON from {filepath}")
    return data


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:  # type: ignore
    """Calculate regression metrics."""
    metrics = {
        "r2_score": r2_score(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "mean_squared_error": mean_squared_error(y_true, y_pred),
        "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        "mean_absolute_percentage_error": np.mean(np.abs((y_true - y_pred)/y_true))
    }
    return metrics


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def log_metrics_to_mlflow(metrics: Dict[str, float],
                          step: int = None) -> None:  # type: ignore
    """Log metrics to mlflow."""
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value, step=step)


def save_plot(fig, filename: str, plot_dir: str, formats: List[str] = ["png"]) -> None:
    """Save plot in specified formats."""
    os.makedirs(plot_dir, exist_ok=True)
    for fmt in formats:
        filepath = os.path.join(plot_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        logging.info(f"Saved plot to {filepath}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, plot_dir: str,
                   formats: List[str] = ["png"]) -> None:
    """Create residual plot."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Residuals vs predicted
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color="red", linestyle="--")
    ax1.set_xlabel("Predicted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted Values")
    # Histogram
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of residuals")
    plt.tight_layout()
    save_plot(fig=fig, filename="residual_analysis",
              plot_dir=plot_dir, formats=formats)
    plt.close()


def plot_prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, plot_dir: str,
                              formats: List[str] = ["png"]) -> None:
    """Create prediction vs actual plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Predicted vs Actual Values")
    # Add R2 score to plot
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes,
            bbox=dict(boxstyle="round, pad=0.3", facecolor="yellow", alpha=0.7))
    plt.tight_layout()
    save_plot(fig=fig, filename="prediction vs actual",
              plot_dir=plot_dir, formats=formats)
    plt.close()


def print_data_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print basic information about the dataframe."""
    logging.info(f"\n{name} Info:")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    logging.info(f"Data types:\n{df.dtypes}")
    logging.info(f"Missing values:\n{df.isnull().sum()}")
    logging.info(f"Basic statistics:\n{df.describe()}")


def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://127.0.0.1:5000") -> None:
    """Set up MLflow experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)
    logging.info(f"MLflow experiment '{experiment_name}' setup complete")
