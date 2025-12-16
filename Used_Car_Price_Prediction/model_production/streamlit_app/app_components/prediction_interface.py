"""Prediction interface for the Streamlit application."""

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config.config import get_settings
from src.utils.model_loader import get_model_loader


class PredictionInterface:
    """Handles the used car price prediction interface and logic."""

    def __init__(self):
        self.settings = get_settings()
        self.api_base_url = self.settings.api_url
        self.use_local_model = self.settings.USE_LOCAL_MODEL

        # Initialize local model if needed
        self.local_model_loader = None
        if self.use_local_model:
            self._initialize_local_model()

    def _initialize_local_model(self):
        """Initialize local model loader."""
        try:
            import contextlib
            import io

            pointer_file = self.settings.models_dir / "current.txt"
            folder_name = pointer_file.read_text(encoding="utf-8").strip()
            model_dir = self.settings.models_dir / folder_name

            model_path = model_dir / "model.pkl"
            scaler_path = model_dir / "scaler.pkl"
            selected_features_path = model_dir / "features.pkl"
            current_year = self.settings.CURRENT_YEAR

            with st.spinner("⏳ Loading Model..."):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.local_model_loader = get_model_loader(
                        model_path=str(model_path),
                        scaler_path=str(scaler_path),
                        selected_features_path=str(selected_features_path),
                        current_year=current_year
                    )

            if self.local_model_loader:
                st.markdown(
                    """
                <div style="
                    background-color: transparent;
                    border: none;
                    padding: 1rem;
                    border-radius: none;
                    margin-bottom: 1rem;
                ">
                    <strong style="color: #f7d77e;text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Local model loaded successfully</span>
                </div>
                """,
                    unsafe_allow_html=True
                )
            else:
                st.error("Failed to load local model")
                self.use_local_model = False

        except Exception as e:
            st.error(f"Error initializing local model: {str(e)}")
            self.use_local_model = False

    def check_api_status(self) -> bool:
        """Check if FastAPI service is running."""
        if self.use_local_model:
            st.markdown(
                """
                <div style="
                    background-color: transparent;
                    border: none;
                    padding: 1rem;
                    border-radius: none;
                    margin-bottom: 1rem;
                ">
                    <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Service:</strong> 
                    <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Not Required (Using local model)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            return True
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                # API Service Online
                st.markdown(
                    """
                    <div style="
                        background-color: transparent;
                        border: none;
                        padding: 1rem;
                        border-radius: none;
                        margin-bottom: 1rem;
                    ">
                        <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Service:</strong> 
                        <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Online</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                health_data = response.json()
                if health_data.get("model_loaded"):
                    # ML Model: Loaded
                    st.markdown(
                        """
                        <div style="
                            background-color: transparent;
                            border: none;
                            padding: 1rem;
                            border-radius: none;
                            margin-bottom: 1rem;
                        ">
                            <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">ML Model:</strong> 
                            <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Loaded</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    return True
                else:
                    # ML Model: Not Loaded
                    st.markdown(
                        """
                        <div style="
                            background-color: transparent;
                            border: none;
                            padding: 1rem;
                            border-radius: none;
                            margin-bottom: 1rem;
                        ">
                            <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">ML Model:</strong> 
                            <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Not Loaded</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    return False
            else:
                # API Service Offline
                st.markdown(
                    """
                    <div style="
                        background-color: transparent;
                        border: none;
                        padding: 1rem;
                        border-radius: none;
                        margin-bottom: 1rem;
                    ">
                        <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Service:</strong> 
                        <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Offline</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return False
        except requests.exceptions.ConnectionError as ce:
            # API Service Unavailable
            st.markdown(
                """
                <div style="
                    background-color: transparent;
                    border: none;
                    padding: 1rem;
                    border-radius: none;
                    margin-bottom: 1rem;
                ">
                    <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Service:</strong> 
                    <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Unavailable</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    background-color: transparent;
                    border: none;
                    padding: 1rem;
                    border-radius: none;
                    margin-bottom: 1rem;
                ">
                    <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">{str(ce)}. Make sure your FastAPI server is running on port 8000
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )
            return False
        except Exception as e:
            # API Error
            st.markdown(
                f"""
                <div style="
                    background-color: transparent;
                    border: none;
                    padding: 1rem;
                    border-radius: none;
                    margin-bottom: 1rem;
                ">
                    <strong style="color: #f7d77e; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Error:</strong> 
                    <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">{str(e)}</span>
                </div>
                    """,
                unsafe_allow_html=True
            )
            return False

    def get_valid_options(self) -> Dict[str, list]:
        """Get valid options for dropdowns from API if available."""
        # Try API
        if not self.use_local_model:
            try:
                response = requests.get(
                    f"{self.api_base_url}/model_info", timeout=5)
                if response.status_code == 200:
                    model_info = response.json()
                    valid_values = model_info.get(
                        "input_schema", {}).get("valid_values", {})
                    if valid_values:
                        return valid_values
            except requests.exceptions.RequestException:
                pass

        # Default options if API not available
        return {
            "brand": [
                "Honda",
                "Toyota",
                "Ford",
                "Chevrolet",
                "BMW",
                "Volkswagen",
                "Tesla",
                "Hyundai",
                "Kia",
                "Nissan",
            ],
            "fuel_type": ["Petrol", "Diesel", "Hybrid", "Electric"],
            "transmission": ["Manual", "Automatic"],
            "color": ["White", "Black", "Silver", "Red", "Blue", "Green"],
            "insurance_valid": ["Yes", "No"],
            "service_history": [
                "Full Service History",
                "Partial Service History",
                "None",
            ],
        }

    def render_input_form(self) -> Optional[Dict[str, Any]]:
        """Render the input form for user to enter car specifications."""
        st.markdown(
            '<h2 style="color: #f7d77e !important; margin-bottom: 1rem; text-shadow: 0 0 10px rgba(247, 215, 126, 0.5);">Car Specifications</h2>',
            unsafe_allow_html=True
        )

        # Get valid options
        valid_options = self.get_valid_options()

        with st.form("car_prediction_form"):
            # Create input form
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    '<h4 style="color: #f7d77e; font-weight: 600; font-size: 1.1rem;">Basic Information</h4>', unsafe_allow_html=True
                )
                make_year = st.number_input(
                    label="Manufacturing Year",
                    min_value=1990,
                    max_value=2024,
                    value=2020,
                    help="Year the car was manufactured",
                )

                brand = st.selectbox(
                    label="Brand",
                    options=valid_options.get("brand", ["Honda"]),
                    help="Car manufacturer",
                )

                fuel_type = st.selectbox(
                    label="Fuel Type",
                    options=valid_options.get("fuel_type", ["Petrol"]),
                    help="Type of fuel the car uses",
                )

                transmission = st.selectbox(
                    label="Transmission",
                    options=valid_options.get("transmission", ["Manual"]),
                    help="Transmission type",
                )

            with col2:
                st.markdown(
                    '<h4 style="color: #f7d77e; font-weight: 600; font-size: 1.1rem;">Technical Specifications</h4>', unsafe_allow_html=True
                )
                engine_cc = st.number_input(
                    label="Engine Capacity (CC)",
                    min_value=500,
                    max_value=8000,
                    value=1800,
                    help="Engine displacement in cubic centimeters",
                )

                mileage_kmpl = st.number_input(
                    label="Mileage (KMPL)",
                    min_value=5.0,
                    max_value=50.0,
                    value=15.0,
                    step=0.1,
                    help="Fuel efficiency in kilometers per liter",
                )

                owner_count = st.number_input(
                    label="Number of Owners",
                    min_value=0,
                    max_value=10,
                    value=1,
                    help="Number of previous owners",
                )

                accidents_reported = st.number_input(
                    label="Accidents Reported",
                    min_value=0,
                    max_value=10,
                    value=0,
                    help="Number of reported accidents",
                )

            with col3:
                st.markdown(
                    '<h4 style="color: #f7d77e; font-weight: 600; font-size: 1.1rem;">Additional Details</h4>', unsafe_allow_html=True
                )
                color = st.selectbox(
                    label="Color",
                    options=valid_options.get("color", ["White"]),
                    help="Car color",
                )

                insurance_valid = st.selectbox(
                    label="Insurance Valid",
                    options=valid_options.get("insurance_valid", ["Yes"]),
                    help="Is the insurance currently valid?",
                )

                service_history = st.selectbox(
                    label="Service History",
                    options=valid_options.get(
                        "service_history", ["Full Service History"]),
                    help="Maintenance service record",
                )

                # Spacer for alignment
                st.empty()

            # Submit button
            submit_button = st.form_submit_button(
                label="Predict Price",
                use_container_width=True,
                type="primary"
            )

            if submit_button:
                car_data = {
                    "make_year": make_year,
                    "accidents_reported": accidents_reported,
                    "fuel_type": fuel_type,
                    "brand": brand,
                    "transmission": transmission,
                    "color": color,
                    "insurance_valid": insurance_valid,
                    "service_history": service_history,
                    "mileage_kmpl": mileage_kmpl,
                    "engine_cc": engine_cc,
                    "owner_count": owner_count,
                }
                return car_data

        return None

    def make_prediction(
            self,
            car_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make prediction using either local model or FastAPI service."""
        try:
            with st.spinner("Analyzing your car specifications..."):
                # Use local model if enabled
                if self.use_local_model and self.local_model_loader:
                    return self._predict_local(car_data)
                else:
                    return self._predict_api(car_data)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

    def _predict_local(
        self,
        car_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make prediction using local model."""
        try:
            if self.local_model_loader:
                result = self.local_model_loader.predict(car_data=car_data)
                st.markdown(
                    """
                    <div style="
                        background-color: transparent;
                        border: none;
                        padding: 0.5rem 0;
                        color: #f7d77e;
                    ">
                        <p style="
                            color: #f7d77e;
                            font-weight: 600;
                            margin: 0;
                            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
                        ">
                            Prediction completed using local model
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return result
        except Exception as e:
            st.error(f"Local prediction failed: {str(e)}")
            return None

    def _predict_api(
            self,
            car_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make prediction using FastAPI service."""

        try:
            response = requests.post(
                f"{self.api_base_url}/predict",
                json=car_data,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                st.markdown(
                    """
                    <div style="
                        background-color: transparent;
                        border: none;
                        padding: 0.5rem 0;
                        color: #f7d77e;
                    ">
                        <p style="
                            color: #f7d77e;
                            font-weight: 600;
                            margin: 0;
                            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
                    ">
                        Prediction completed using API
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                return result
            else:
                st.error(f"API Prediction failed: {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the prediction service. "
                "Please make sure the FastAPI server is running."
            )
        except requests.exceptions.Timeout:
            st.error("Request timeout. The service might be busy.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        return None

    def display_prediction_results(
        self,
        result: Dict[str, Any],
        car_data: Dict[str, Any]
    ):
        """Display the prediction results."""
        predicted_price = result.get("predicted_price", 0)
        confidence_score = result.get("confidence_score", 0)

        # Main prediction display
        st.markdown(
            f"""
        <div class="prediction-box">
            <h2>Predicted Price</h2>
            <h1>${predicted_price:,.2f}</h1>
            <p>Confidence Score: {confidence_score:.1%}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)

        car_age = datetime.now().year - car_data["make_year"]

        with col1:
            st.metric("Car Age", f"{car_age} years")

        with col2:
            st.metric("Confidence", f"{confidence_score:.1%}")

        with col3:
            st.metric("Engine", f"{car_data['engine_cc']} CC")

        with col4:
            st.metric("Mileage", f"{car_data['mileage_kmpl']} KMPL")

        # Validation results
        validation = result.get("input_validation", {})
        if validation.get("warnings"):
            st.warning("Validation Warnings:")
            for warning in validation["warnings"]:
                st.write(f"• {warning}")

        if validation.get("errors"):
            st.error("Validation Errors:")
            for error in validation["errors"]:
                st.write(f"• {error}")

    def display_insights(
        self,
        result: Dict[str, Any],
        car_data: Dict[str, Any],
        predicted_price: float
    ):
        """Display additional insights and analysis."""
        st.markdown(
            '<h3 style="color: #f7d77e; text-shadow: none;">Analysis & Insights</h3>', unsafe_allow_html=True
        )

        # Create tabs for different insights
        tab1, tab2, tab3 = st.tabs(
            ["Price Insights", "Car Analysis", "Market Comparison"]
        )

        with tab1:
            self.show_price_insights(car_data, predicted_price)

        with tab2:
            self.show_car_analysis(result, car_data)

        with tab3:
            self.show_market_comparison(car_data, predicted_price)

    def show_price_insights(
            self,
            car_data: Dict[str, Any],
            predicted_price: float
    ):
        """Show price related insights."""
        car_age = datetime.now().year - car_data["make_year"]

        insights = []

        # Age based insights
        if car_age <= 3:
            insights.append(
                "This is a relatively new car, which positively impacts its value"
            )
        elif car_age <= 7:
            insights.append(
                "This car is in the optimal age range for resale value")
        else:
            insights.append("Older cars typically have lower market values")

        # Brand based insights
        luxury_brands = ["BMW", "Volkswagen", "Tesla"]
        reliable_brands = ["Honda", "Toyota"]

        if car_data["brand"] in luxury_brands:
            insights.append(
                "Luxury brand vehicles often maintain better resale value")
        elif car_data["brand"] in reliable_brands:
            insights.append(
                "This brand is known for reliability and lower maintenance costs"
            )

        # Fuel Type Insights
        if car_data["fuel_type"] == "Electric":
            insights.append("Electric vehicles are gaining market popularity")
        elif car_data["fuel_type"] == "Diesel":
            insights.append(
                "Diesel vehicles typically offer better fuel efficiency")

        # Display insights
        for insight in insights:
            st.write(f". {insight}")

        # Price Range Estimation
        price_range_low = predicted_price * 0.9
        price_range_high = predicted_price * 1.1

        st.markdown(
            f"""
            <div style="
                background-color: transparent;
                border: 2px solid #f7d77e;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            ">
                <p style="color: #ffffff; margin-bottom: 0.5rem;">
                    <strong style="color: #ffffff;">Estimated Price Range:</strong>
                </p>
                <p style="color: #f7d77e; font-size: 1.3rem; font-weight: 700; margin: 0;">
                ${price_range_low:,.2f} - ${price_range_high:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def show_car_analysis(
            self,
            result: Dict[str, Any],
            car_data: Dict[str, Any]
    ):
        """Show detailed car analysis."""
        preprocessing_info = result.get("preprocessing_info", {})

        st.write("**Preprocessing Applied**")

        # Calculate car age
        car_age = datetime.now().year - car_data["make_year"]
        st.write(f"• Car age calculated: {car_age} years")

        # Check accident status
        has_accident = car_data["accidents_reported"] > 0
        accident_status = "Yes" if has_accident else "No"
        st.write(f"• Accident history: {accident_status}")

        # Show scaling info
        st.write(f"• Feature scaling applied: Yes")

        # If preprocessing_info exists from API, show additional details
        if preprocessing_info:
            if "car_age_calculated" in preprocessing_info:
                st.write(
                    f"• Car age (from API): {preprocessing_info['car_age_calculated']} years"
                )
            if "has_accident" in preprocessing_info:
                accident_status_api = "Yes" if preprocessing_info["has_accident"] else "No"
                st.write(
                    f"• Accident history (from API): {accident_status_api}"
                )

        st.markdown("---")

        # Create a summary table
        summary_data = {
            "Feature": [
                "Brand",
                "Year",
                "Fuel Type",
                "Transmission",
                "Engine CC",
                "Mileage",
                "Owners",
                "Accidents",
                "Color",
                "Insurance",
            ],
            "Value": [
                car_data["brand"],
                car_data["make_year"],
                car_data["fuel_type"],
                car_data["transmission"],
                f"{car_data['engine_cc']} CC",
                f"{car_data['mileage_kmpl']} KMPL",
                car_data["owner_count"],
                car_data["accidents_reported"],
                car_data["color"],
                car_data["insurance_valid"],
            ],
        }

        df_summary = pd.DataFrame(summary_data)
        st.table(df_summary)

    def show_market_comparison(
            self,
            car_data: Dict[str, Any],
            predicted_price: float
    ):
        """Show market comparison and trends."""
        # Create a comparison chart
        brands = [
            "Honda",
            "Toyota",
            "Ford",
            "Chevrolet",
            "BMW",
            "Volkswagen",
            "Tesla",
            "Hyundai",
            "Kia",
            "Nissan",
        ]

        brand_price_ranges = {
            "Honda": [1000, 17500],
            "Toyota": [1000, 17000],
            "Ford": [1000, 15600],
            "Chevrolet": [1000, 17000],
            "BMW": [1000, 17000],
            "Volkswagen": [1000, 17000],
            "Tesla": [1000, 18000],
            "Hyundai": [1000, 18000],
            "Kia": [1000, 17500],
            "Nissan": [1000, 15500],
        }

        # Create comparison chart
        fig = go.Figure()

        for brand in brands:
            price_range = brand_price_ranges[brand]
            color = "#f7d77e" if brand == car_data["brand"] else "#6b6b6b"

            fig.add_trace(
                go.Bar(
                    name=brand,
                    x=[brand],
                    y=[price_range[1] - price_range[0]],
                    base=price_range[0],
                    marker={"color": color},
                    text=[f"${price_range[0]:,} - ${price_range[1]:,}"],
                    textposition="inside",
                    hovertemplate=f"<b>{brand}</b><br>Range: ${price_range[0]:,} - ${price_range[1]:,}<extra></extra>",
                )
            )

        fig.add_hline(
            y=predicted_price,
            line_dash="dash",
            line_color="#dc3545",
            annotation={
                "text": f"Your Car: ${predicted_price:,.2f}",
                "font": {"color": "#f7d77e"}
            },
        )

        fig.update_layout(
            title={
                "text": "Price Comparison by Brand",
                "font": {"color": "#f7d77e", "size": 20}
            },
            xaxis_title="Brand",
            yaxis_title="Price Range ($)",
            showlegend=False,
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,20)",
            font=dict(color="white", size=12),
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333'),
            # Remove text shadows
            annotations=[],
            hoverlabel=dict(
                bgcolor="rgba(30, 30, 30, 0.95)",
                font_size=13,
                font_color="white",
                bordercolor="#f7d77e"
            )
        )

        # Remove text shadows from all text elements
        fig.update_xaxes(title_font=dict(color='white'),
                         tickfont=dict(color='white'))
        fig.update_yaxes(title_font=dict(color='white'),
                         tickfont=dict(color='white'))

        st.plotly_chart(fig, use_container_width=True)

        # Market Insights
        st.write("**Market Insights:**")
        st.write(
            f"• Your {car_data['brand']} falls within the typical price range for this brand"
        )
        st.write("• Consider checking multiple sources before finalizing the price")
        st.write("• Market conditions and location can affect actual selling price")

    def render(self):
        """Render the prediction interface."""
        # Show prediction mode
        if self.use_local_model:
            st.markdown(
                """
                <div style="
                background-color: transparent;
                border: none;
                padding: 1rem;
                border-radius: none;
                margin-bottom: 1rem;
            ">
                <strong style="color: #f7d77e;">Prediction Mode:</strong> 
                <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">Local Model</span>
            </div>
            """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="
                background-color: transparent;
                border: none;
                padding: 1rem;
                border-radius: none;
                margin-bottom: 1rem;
            ">
                <strong style="color: #f7d77e;">Prediction Mode:</strong> 
                <span style="color: #ffffff; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);">API Service</span>
            </div>
            """,
                unsafe_allow_html=True
            )

        # Check API status
        api_status = self.check_api_status()

        # Render input form and get data
        car_data = self.render_input_form()

        # If form was submitted, make prediction
        if car_data:
            result = self.make_prediction(car_data)

            if result:
                predicted_price = result.get("predicted_price", 0)
                self.display_prediction_results(result, car_data)
                self.display_insights(result, car_data, predicted_price)
