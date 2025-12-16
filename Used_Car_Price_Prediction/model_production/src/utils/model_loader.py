import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import streamlit as st
from config.logging_config import LoggingConfig

logger = LoggingConfig.get_logger(__name__)


class LocalModelPreprocessor:
    """Handles preprocessing for local model predictions."""

    def __init__(self, current_year: int = 2025):
        self.current_year = current_year
        # Service history mapping
        self.service_history_mapping = {
            "None": 0,
            "Partial Service History": 1,
            "Full Service History": 2,
        }

    def preprocess(
            self,
            car_data: Dict[str, Any],
            selected_features: List[str]
    ) -> pd.DataFrame:
        """Preprocess car data for prediction.

        Args:
            car_data: Raw car data dictionary
            selected_features: List of features the model expects

        Returns:
            Preprocessed dataframe with selected features
        """

        # Convert to DataFrame
        df = pd.DataFrame([car_data])

        # Feature engineering
        df["car_age"] = self.current_year - df["make_year"]
        df["has_accident"] = (df["accidents_reported"] > 0).astype(int)

        # Handle service_history
        if "service_history" in df.columns:
            df["service_history_encoded"] = df["service_history"].map(
                self.service_history_mapping
            ).fillna(0).astype(int)

        # One-hot encode categorical features
        categorical_features = [
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
        ]

        df_encoded = pd.get_dummies(
            df,
            columns=categorical_features,
            drop_first=True,
            dtype=int
        )

        # Drop columns not needed for prediction
        columns_to_drop = [
            "make_year",
            "accidents_reported",
            "fuel_type",
            "brand",
            "transmission",
            "color",
            "insurance_valid",
            "service_history",
        ]
        existing_columns_to_drop = [
            col for col in columns_to_drop if col in df_encoded.columns
        ]
        df_final = df_encoded.drop(columns=existing_columns_to_drop)

        # Ensure all selected features are present
        for feature in selected_features:
            if feature not in df_final.columns:
                df_final[feature] = 0

        # Remove extra features and reorder
        df_final = df_final[selected_features]

        return df_final


class LocalModelLoader:
    """Loads and manages local ML model."""

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str] = None,
        selected_features_path: Optional[str] = None,
        current_year: int = 2025
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self.selected_features_path = Path(
            selected_features_path) if selected_features_path else None
        self.current_year = current_year

        self.model = None
        self.scaler = None
        self.selected_features = None
        self.preprocessor = LocalModelPreprocessor(current_year=current_year)

    def load_model(self):
        """Load the trained model from pickle file."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}"
                )
            with open(self.model_path, "rb") as f:
                loaded = joblib.load(f)

            # Extract estimator
            self.model = loaded["model"]

            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error(f"Failed to load model: {str(e)}")
            return False

    def load_scaler(self):
        """Load the feature scaler if available."""
        if not self.scaler_path or not self.scaler_path.exists():
            logger.info(
                "No scaler file found, predictions will use unscaled features")
            return True

        try:
            with open(self.scaler_path, "rb") as f:
                self.scaler = joblib.load(f)

            logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            st.warning(f"Failed to load scaler: {str(e)}")
            return True

    def load_selected_features(self):
        """Load the list of selected features."""
        if not self.selected_features_path \
                or not self.selected_features_path.exists():
            logger.warning("No selected features file found")
            # Infer from model
            if self.model:
                if hasattr(self.model, "n_features_in_"):
                    logger.info(
                        f"Model expects {self.model.n_features_in_} features"
                    )
            return True

        try:
            with open(self.selected_features_path, "rb") as f:
                self.selected_features = joblib.load(f)

            logger.info(
                f"Loaded {len(self.selected_features)} selected features"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading selected features: {str(e)}")
            st.warning(f"Failed to load selected features: {str(e)}")
            return False

    def initialize(self) -> bool:
        """Initialize all components."""
        success = True
        success &= self.load_model()
        success &= self.load_scaler()
        success &= self.load_selected_features()
        return success

    def predict(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on car data.

        Args:
            car_data: Dictionary containing car features

        Returns:
            Dictionary with prediction results
        """
        if not self.model:
            raise ValueError("Model not loaded. Call initialize() first.")

        if not self.selected_features:
            raise ValueError("Selected features not loaded.")

        try:
            # Preprocess data
            X = self.preprocessor.preprocess(
                car_data=car_data,
                selected_features=self.selected_features
            )

            # Apply scaling if available
            if self.scaler:
                X_scaled = self.scaler.transform(X.values)
            else:
                X_scaled = X.values

            # Make prediction
            prediction = self.model.predict(X_scaled)

            # Calculate confidence score
            confidence_score = 0.8  # 80% base confidence

            # Car Age
            car_age = datetime.now().year - car_data["make_year"]
            if car_age <= 5:
                confidence_score += 0.1
            elif car_age > 20:
                confidence_score -= 0.2

            # Brands
            common_brands = ["Honda", "Toyota", "Ford", "Chevrolet"]
            if car_data["brand"] in common_brands:
                confidence_score += 0.05

            # Accidents
            if car_data["accidents_reported"] == 0:
                confidence_score += 0.05
            elif car_data["accidents_reported"] > 3:
                confidence_score -= 0.1

            # Predictions
            if prediction < 1000 or prediction > 100000:
                confidence_score -= 0.1

            confidence_score = round(
                (max(0.1, min(0.95, confidence_score))), 3)

            # Prepare preprocessing info
            preprocessing_info = {
                "car_age_calculated": self.current_year - car_data["make_year"],
                "has_accident": car_data["accidents_reported"] > 0,
                "features_used": len(self.selected_features),
                "scaling_applied": self.scaler is not None
            }

            # Prepare result
            result = {
                "predicted_price": float(prediction),
                "confidence_score": confidence_score,
                "input_validation": {
                    "warnings": [],
                    "errors": []
                },
                "preprocessing_info": preprocessing_info,
                "prediction_metadata": {
                    "model_type": type(self.model).__name__,
                    "features_count": len(self.selected_features),
                    "prediction_mode": "local"
                }
            }

            # Add validation warnings
            car_age = self.current_year - car_data["make_year"]
            if car_age > 15:
                result["input_validation"]["warnings"].append(
                    "Car is older than 15 years - prediction may be less accurate"
                )

            if car_data["accidents_reported"] > 3:
                result["input_validation"]["warnings"].append(
                    "High number of accidents reported - may significantly affect value"
                )

            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model:
            return {}

        return {
            "model_type": type(self.model).__name__,
            "model_loaded": True,
            "framework": "scikit-learn",
            "features_count": len(self.selected_features)
            if self.selected_features else 0,
            "selected_features": self.selected_features or [],
            "scaler_loaded": self.scaler is not None
        }


@st.cache_resource(show_spinner=False)
def get_model_loader(
    model_path: str,
    scaler_path: Optional[str] = None,
    selected_features_path: Optional[str] = None,
    current_year: int = 2025
) -> Optional[LocalModelLoader]:
    """Get cached model loader instance.

    Args:
        model_path: Path to model pickle file
        scaler_path: Path to scaler pickle file
        selected_features_path: Path to selected features pickle file
        current_year: Current year for age calculation

    Returns:
        LocalModelLoader instance or None if loading failed
    """
    try:
        loader = LocalModelLoader(
            model_path=model_path,
            scaler_path=scaler_path,
            selected_features_path=selected_features_path,
            current_year=current_year
        )

        if loader.initialize():
            logger.info("Model loader initialized successfully")
            return loader
        else:
            logger.error("Failed to initialize model loader")
            return None

    except Exception as e:
        logger.error(f"Error creating model loader: {str(e)}")
        st.error(f"Failed to initialize model: {str(e)}")
        return None
