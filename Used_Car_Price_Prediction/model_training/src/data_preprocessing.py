import pandas as pd
import numpy as np
import logging
import os
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn


from utils import (setup_logging, setup_mlflow_experiment, save_json,
                   save_pickle, print_data_info, create_directories,
                   log_metrics_to_mlflow)


class DataPreprocessor:
    """Data preprocessing pipeline"""

    def __init__(self, config: DictConfig):
        self.config = config
        self.current_year = config.current_year
        self.feature_names = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully from {filepath}")
            print_data_info(df=df, name="Raw Data")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()

        # Fill null values in the service_history column with "None"
        values = {"service_history": "None"}
        df.fillna(value=values, inplace=True)

        logging.info("Missing vaues handled")

        return df

    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        df = df.copy()

        # Create car_age column
        df["car_age"] = self.current_year - df["make_year"]

        # Create has_accident column
        df["has_accident"] = df["accidents_reported"].apply(
            lambda x: 1 if x > 0 else 0)

        logging.info("New features created: car_age, has_accident")

        return df

    def encode_cat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()

        # Ordinal encoding for service_history
        service_history_mapping = self.config.preprocessing.service_history_mapping
        df["service_history"] = df["service_history"].map(
            service_history_mapping)

        # One-hot encoding for categorical features
        cat_cols = self.config.preprocessing.categorical_columns
        cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

        # Concatenate with original dataframe
        df_final = pd.concat([df, cat_encoded], axis=1)

        logging.info(f"Categorical features encoded: {cat_cols}")

        return df_final

    def drop_unnecessary_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns."""
        df = df.copy()

        cols_to_drop = self.config.preprocessing.drop_columns
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # Export final dataframe to csv
        filename = "preprocessed_data.csv"
        df.to_csv(os.path.join(self.config.data.processed_data_path, filename),
                  index=False)

        logging.info(f"Dropped columns: {cols_to_drop}")

        return df

    def perform_feat_selection(self,
                               X: pd.DataFrame,
                               y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Perform feature selection using RFECV."""
        fs_config = self.config.feature_selection
        if fs_config.method == "rfecv":
            estimator = LinearRegression()
            cv = KFold(n_splits=fs_config.cv_folds,
                       shuffle=True,
                       random_state=42)

            # Perform RFECV
            rfecv = RFECV(
                estimator=estimator,
                step=1,
                cv=cv,
                scoring=fs_config.scoring,
                min_features_to_select=fs_config.min_features_to_select,
                n_jobs=-1
            )
            rfecv.fit(X, y)

            # Get selected features
            selected_features = X.columns[rfecv.support_].tolist()
            X_selected = X[selected_features]

            # Log results
            logging.info(
                f"Feature selection completed. Selected {len(selected_features)} features")
            logging.info(f"Selected features: {selected_features}")
            logging.info(f"Optimal number of features: {rfecv.n_features_}")

            # Log to MLflow
            mlflow.log_param("n_features_selected", len(selected_features))
            mlflow.log_param("optimal_num_features", rfecv.n_features_)
            return X_selected, selected_features
        else:
            logging.warning(
                "Feature selection method not recognized. Using all features.")
            return X, X.columns.tolist()

    def split_data(self, df: pd.DataFrame,
                   target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.preprocessing.test_size,
            random_state=self.config.preprocessing.random_state
        )

        logging.info(
            f"Data split into Train : {X_train.shape} and Test : {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, df: pd.DataFrame,
                            target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Data Pre-processing Pipeline."""
        # Handle missing values
        df = self.handle_missing_values(df=df)

        # Create new features
        df = self.create_new_features(df=df)

        # Encode categorical features
        df = self.encode_cat_features(df=df)

        # Drop unnecessary columns
        df = self.drop_unnecessary_cols(df=df)

        print_data_info(df=df, name="Preprocessed Data")

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(
            df=df, target_col=target_col)

        # Perform feature selection
        X_train_selected, selected_features = self.perform_feat_selection(
            X=X_train, y=y_train)
        X_test_selected = X_test[selected_features]

        # Store feature names
        self.feature_names = selected_features

        return X_train_selected, X_test_selected, y_train, y_test

    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> None:
        """Save processed data to files."""
        # Create processed data directory
        processed_dir = self.config.data.processed_data_path
        create_directories(directories=[processed_dir])

        # Combine features and target for saving
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Save data
        train_data.to_csv(self.config.data.train_data_path, index=False)
        test_data.to_csv(self.config.data.test_data_path, index=False)

        # Save feature names
        save_pickle(self.feature_names, os.path.join(
            processed_dir, "feature_names.pkl"))

        logging.info("Processed data saved successfully")

        # Log metrics
        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(self.feature_names) if self.feature_names is not None else 0,
            "train_target_mean": y_train.mean(),
            "train_target_std": y_train.std(),
            "test_target_mean": y_test.mean(),
            "test_target_std": y_test.std()
        }

        save_json(metrics, filepath="logs/preprocessing_metrics.json")
        log_metrics_to_mlflow(metrics=metrics)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main preprocessing function."""
    # Set up logging
    setup_logging(log_level=cfg.logging.level, log_dir=cfg.logging.log_dir)

    # Set up MLflow
    setup_mlflow_experiment(experiment_name=cfg.mlflow.experiment_name,
                            tracking_uri=cfg.mlflow.tracking_uri)

    with mlflow.start_run(run_name="data_preprocessing"):
        # Log parameters
        mlflow.log_params({
            "current_year": cfg.current_year,
            "test_size": cfg.preprocessing.test_size,
            "random_state": cfg.preprocessing.random_state,
            "feature_selection_method": cfg.feature_selection.method
        })

        # Initialize preprocessor
        preprocessor = DataPreprocessor(cfg)

        # Load data
        df = preprocessor.load_data(cfg.data.raw_data_path)

        # Target column
        target_col = "price_usd"

        # Preprocess data
        X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(df=df,
                                                                            target_col=target_col)

        # Save processed data
        preprocessor.save_processed_data(X_train=X_train, X_test=X_test,
                                         y_train=y_train, y_test=y_test)

        logging.info("Data preprocessing completed successfully..")


if __name__ == "__main__":
    main()
