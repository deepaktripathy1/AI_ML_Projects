import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import scipy.stats as stats
import joblib
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.sklearn

from utils import (
    setup_logging, save_json, load_pickle, calculate_metrics,
    setup_mlflow_experiment, log_metrics_to_mlflow, save_plot,
    plot_prediction_vs_actual, plot_residuals
)


class Evaluator:
    """Model evaluation pipeline."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.feature_names = None

    def load_model_and_data(self) -> tuple:
        """Load trained model and test data."""
        try:
            # Load model
            model_path = os.path.join(
                self.config.model.save_path, self.config.model.model_name)
            self.model = joblib.load(model_path)

            # Load test data
            test_data = pd.read_csv(self.config.data.test_data_path)

            # Load feature names
            self.feature_names = load_pickle(
                os.path.join(
                    self.config.data.processed_data_path, 'feature_names.pkl')
            )

            target_col = "price_usd"

            # Split data into features and target
            X_test = test_data[self.feature_names]
            y_test = test_data[target_col]

            logging.info(f"Model loaded from {model_path}")
            logging.info(
                f"Test data loaded with {len(X_test)} samples and shape {X_test.shape}.")
            logging.info(f"Model type: {type(self.model)}")

            return X_test, y_test

        except Exception as e:
            logging.error(f"Error loading model or data: {e}")
            raise

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[Dict[str, float], Any]:
        """Evaluate model on test data."""
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Call load_model_and_data() before evaluate_model().")
        # Make predictions
        y_pred = self.model.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test.to_numpy(), y_pred)

        # Log metrics
        logging.info(f"Test set evaluation metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Log to MLflow
        for metric, value in metrics.items():
            mlflow.log_metric(f"test_{metric}", value)

        return metrics, y_pred

    def perform_cross_validation(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Perform cross-validation evaluation."""
        if self.model is None:
            raise ValueError(
                "Model is not loaded. Call load_model_and_data() before perform_cross_validation().")

        # Perform cross-validation
        cv_config = self.config.evaluation.cross_validation
        if not cv_config.enabled:
            return {}

        # Perform cross-validation
        cv_scores = cross_val_score(
            self.model,
            X_test,
            y_test,
            cv=cv_config.cv_folds,
            scoring=cv_config.scoring,
            n_jobs=-1
        )

        cv_metrics = {
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "min_cv_score": np.min(cv_scores),
            "max_cv_score": np.max(cv_scores)
        }

        # Log cross-validation scores
        logging.info(f"Cross-validation results:")
        for metric, value in cv_metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Log to MLflow
        log_metrics_to_mlflow(cv_metrics)

        return cv_metrics

    def create_evaluation_plots(self, y_test: pd.Series, y_pred: np.ndarray) -> None:
        """Create evaluation plots."""
        plots_config = self.config.evaluation.plots
        if not plots_config.save_plots:
            return

        plot_dir = plots_config.plot_dir
        plot_formats = plots_config.plot_formats

        # Create plot directory if it doesn't exist
        os.makedirs(plot_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl", 8)

        # Plot residuals
        if plots_config.residual_plot:
            plot_residuals(
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                plot_dir=plot_dir,
                formats=plot_formats
            )

        # Plot prediction vs actual
        if plots_config.prediction_vs_actual:
            plot_prediction_vs_actual(
                y_true=y_test.to_numpy(),
                y_pred=y_pred,
                plot_dir=plot_dir,
                formats=plot_formats
            )

        # Q-Q plot
        if plots_config.qq_plot:
            self.plot_qq_plot(y_test, y_pred,
                              plot_dir, plot_formats)

        # Error Distribution plot
        self.plot_error_distribution(
            y_test, y_pred, plot_dir, plot_formats)

        # Prediction intervals
        self.plot_prediction_intervals(
            y_test, y_pred, plot_dir, plot_formats)

        logging.info("Evaluation plots created and saved.")

    def plot_qq_plot(self, y_test: pd.Series, y_pred: np.ndarray,
                     plot_dir: str, formats: List[str]) -> None:
        """Create Q-Q plot of residuals."""
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot of Residuals")
        ax.grid(True)
        plt.tight_layout()
        save_plot(fig=fig, filename="qq_plot",
                  plot_dir=plot_dir, formats=formats)
        plt.close()

    def plot_error_distribution(self, y_test: pd.Series, y_pred: np.ndarray,
                                plot_dir: str, formats: List[str]) -> None:
        """Plot error distribution."""
        errors = y_test - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of errors
        ax1.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel("Prediction Errors")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Prediction Errors")
        ax1.axvline(0, color='red', linestyle='--', alpha=0.7)

        # Box plot of errors
        ax2.boxplot(errors, vert=True)
        ax2.set_ylabel("Prediction Errors")
        ax2.set_title("Box Plot of Prediction Errors")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(fig=fig, filename="error_distribution",
                  plot_dir=plot_dir, formats=formats)
        plt.close()

    def plot_prediction_intervals(self, y_test: pd.Series, y_pred: np.ndarray,
                                  plot_dir: str, formats: List[str]) -> None:
        """Plot prediction intervals."""
        residuals = y_test - y_pred
        std_residuals = np.std(residuals)

        # sort by actual values for better visualization
        sorted_indices = np.argsort(y_test)
        y_test_sorted = y_test.iloc[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot actual vs predicted values
        ax.scatter(range(len(y_test_sorted)), y_test_sorted,
                   alpha=0.6, label="Actual", s=20)
        ax.scatter(range(len(y_pred_sorted)), y_pred_sorted,
                   alpha=0.6, label="Predicted", s=20)

        # Plot prediction intervals
        lower_bound = y_pred_sorted - 1.96 * std_residuals
        upper_bound = y_pred_sorted + 1.96 * std_residuals

        ax.fill_between(range(len(y_pred_sorted)), lower_bound, upper_bound,
                        color='gray', alpha=0.2, label="95% Prediction Interval")

        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Price")
        ax.set_title("Prediction Intervals")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(fig=fig, filename="prediction_intervals",
                  plot_dir=plot_dir, formats=formats)
        plt.close()

    def generate_evaluation_report(self, metrics: Dict[str, float],
                                   cv_metrics: Optional[Dict[str, float]] = None) -> str:
        """Generate evaluation report."""
        report = f"""
        MODEL EVALUATION REPORT
        =======================

        Test Set Metrics:
        {'-' * 20}
        """
        for metric, value in metrics.items():
            report += f"{metric.replace('-', '' '').title()}:{value:.4f}\n"

        if cv_metrics:
            report += f"""
        Cross-Validation Metrics:
        {'-' * 25}
        """
            for metric, value in cv_metrics.items():
                report += f"{metric.replace('-', '').title()}:{value:.4f}\n"

            report += f"""
        Model Performance Summary:
        {"-" * 25}
        - R² Score of {metrics['r2_score']:.4f} indicates the model explains {metrics['r2_score']*100:.2f}% of the variance
        - Average prediction error: {metrics['mean_absolute_error']:.2f} units
        - Average percentage error: {metrics['mean_absolute_percentage_error']:.2f}%
        """

        return report

    def save_evaluation_results(self, metrics: Dict[str, float],
                                cv_metrics: Optional[Dict[str, float]] = None) -> None:
        """Save evaluation results."""
        # Combine all metrics
        all_metrics = {**metrics}
        if cv_metrics:
            all_metrics.update(cv_metrics)

        # Save metrics to JSON
        save_json(
            data=all_metrics,
            filepath="logs/evaluation_metrics.json"
        )

        # Generate and save evaluation report
        report = self.generate_evaluation_report(
            metrics=metrics,
            cv_metrics=cv_metrics
        )

        # Save report
        report_path = "logs/evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logging.info(f"Evaluation report saved to {report_path}")

        # Log report to MLflow
        mlflow.log_text(report, "evaluation_report.txt")

    def run_evaluation(self) -> None:
        """Run the complete evaluation pipeline."""
        # Load model and data
        X_test, y_test = self.load_model_and_data()

        # Evaluate model
        metrics, y_pred = self.evaluate_model(X_test, y_test)

        # Perform cross-validation
        cv_metrics = self.perform_cross_validation(X_test, y_test)

        # Create evaluation plots
        self.create_evaluation_plots(y_test, y_pred)

        # Save evaluation results
        self.save_evaluation_results(metrics, cv_metrics)

        logging.info("Model evaluation completed successfully.")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""

    # Setup logging
    setup_logging(cfg.logging.level, cfg.logging.log_dir)

    # Setup MLflow experiment
    setup_mlflow_experiment(cfg.mlflow.experiment_name,
                            cfg.mlflow.tracking_uri)

    with mlflow.start_run(run_name="model_evaluation"):
        # Log parameters
        mlflow.log_params({
            "evaluation_with_cv": cfg.evaluation.cross_validation.enabled,
            "cv_folds": cfg.evaluation.cross_validation.cv_folds,
            "plots_enabled": cfg.evaluation.plots.save_plots
        })

        # Initialize evaluator
        evaluator = Evaluator(cfg)

        # Run evaluation
        evaluator.run_evaluation()

        logging.info("Model Evaluation completed successfully.")


if __name__ == "__main__":
    main()
