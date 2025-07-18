import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import hydra
from omegaconf import DictConfig
import joblib
import mlflow
import mlflow.sklearn

from utils import (
    setup_logging, save_pickle, save_json, load_pickle,
    calculate_metrics, setup_mlflow_experiment, log_metrics_to_mlflow
)


class TrainingPipeline:
    """Training pipeline for a regression model."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.feature_names = None

    def load_data(
            self) -> Tuple[pd.DataFrame, pd.Series]:  # type: ignore
        """Load preprocessed training data."""
        try:
            train_data = pd.read_csv(self.config.paths.data.train_path)
            self.feature_names = load_pickle(
                os.path.join(self.config.paths.data.processed_data_path,
                             'feature_names.pkl'))
            target_col = "price_usd"

            # Split the data into features and target variable
            X_train = train_data[self.feature_names]
            y_train = train_data[target_col]

            logging.info(f"Loaded training data with shape : {X_train.shape}")
            logging.info(f"Target variable: {target_col}")
            logging.info(f"Feature names: {self.feature_names}")

            return X_train, y_train
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            raise

    def create_model_pipeline(self, model_type: str) -> Pipeline:
        """Create a machine learning model pipeline."""
        try:
            # Create a pipeline with standard scaling and linear regression
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ])
            logging.info("Model pipeline created successfully.")
            return pipeline
        except Exception as e:
            logging.error(f"Error creating model pipeline: {e}")
            raise

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the regression model."""
        model_type = self.config.training.model_type
        logging.info(f"Training {model_type} model...")
        try:
            self.model = self.create_model_pipeline(model_type)
            self.model.fit(X_train, y_train)

            # Store training data for later use
            self.X_train = X_train
            self.y_train = y_train

            # Calculate and log training metrics
            y_train_pred = self.model.predict(X_train)
            train_metrics = calculate_metrics(
                y_train.to_numpy(), np.array(y_train_pred))

            # Log metrics
            logging.info(f"Training metrics:")
            for metric, value in train_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

            # Log to MLflow
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            logging.info("Model training completed successfully.")
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def save_model(self) -> None:
        """Save the trained model."""
        model_dir = self.config.paths.model.save_path
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(
            model_dir, self.config.paths.model.model_name)
        joblib.dump(self.model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(self.model, "model")  # type: ignore

        logging.info(f"Model saved to {model_path}")

        # Save training metrics
        if self.model is not None and hasattr(self, "X_train") and hasattr(self, "y_train"):
            y_train_pred = self.model.predict(self.X_train)
            train_metrics = calculate_metrics(
                self.y_train.to_numpy(), np.array(y_train_pred))
            save_json(train_metrics, "logs/training_metrics.json")
        else:
            logging.warning(
                "Model has not been trained yet. Skipping saving training metrics.")

    def run_training(self) -> None:
        """Run the complete training pipeline."""
        try:
            # Load data
            self.X_train, self.y_train = self.load_data()

            # Train model
            self.train_model(self.X_train, self.y_train)

            # Save model
            self.save_model()

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the training pipeline."""

    # Set up logging
    setup_logging(log_level=cfg.logging.level, log_dir=cfg.logging.log_dir)

    # Set up MLflow
    setup_mlflow_experiment(experiment_name=cfg.mlflow.experiment_name,
                            tracking_uri=cfg.mlflow.tracking_uri)

    with mlflow.start_run(run_name="model_training"):
        # Log parameters
        mlflow.log_params({
            "model_type": cfg.train.model_type,
            "target_column": cfg.train.target_column,
        })

        # Initialize and run training pipeline
        training_pipeline = TrainingPipeline(cfg)

        # Run training pipeline
        training_pipeline.run_training()

        logging.info("Model Training completed successfully.")


if __name__ == "__main__":
    main()
