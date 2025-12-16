"""Data viewer component for analytics and model insights."""

from typing import Any, Dict, Optional
import json
import os

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config.config import get_settings


class DataViewer:
    """Handles data analytics and insights visualization."""

    def __init__(self):
        self.settings = get_settings()
        self.api_base_url = self.settings.api_url
        self.training_report_path = self.settings.logs_dir / "training_report.json"
        self.use_local_model = self.settings.USE_LOCAL_MODEL

    def load_training_report(self) -> Optional[Dict[str, Any]]:
        """Load training report from JSON."""
        try:
            if os.path.exists(self.training_report_path):
                with open(self.training_report_path, "r") as f:
                    return json.load(f)
            else:
                st.warning(f"Training report not found at: {self.training_report_path}"
                           )
        except Exception as e:
            st.error(f"Error loading training report: {str(e)}")
            return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Fetch model information from the API or local model info."""
        if self.use_local_model:
            pointer_file = self.settings.models_dir / "current.txt"
            folder_name = pointer_file.read_text(encoding="utf-8").strip()
            model_dir = self.settings.models_dir / folder_name
            if not model_dir.exists():
                st.error(
                    f"Model folder {model_dir} doesn't exist"
                )
            info_path = model_dir / "model_info.json"
            with open(info_path, "rb") as f:
                model_info = json.load(f)
                return model_info

        try:
            response = requests.get(
                f"{self.api_base_url}/model_info", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"Failed to fetch model information: {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API service.")
            return None
        except Exception as e:
            st.error(f"Error fetching model info: {str(e)}")
            return None

    def render_model_overview(
            self,
            model_info: Dict[str, Any],
            training_report: Optional[Dict[str, Any]]
    ):
        """Render model overview section."""
        st.header("Model overview")

        model_data = model_info.get("model_info", {})

        # Get additional info from training report if available
        if training_report:
            full_results = training_report.get("full_training_results", {})
            model_info_from_report = full_results.get("model_info", {})
            metadata = model_info_from_report.get("metadata", {})
        else:
            metadata = {}

        # Model metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Model Type",
                model_data.get("model_type", "Unknown")
            )

        with col2:
            st.metric(
                "Framework",
                model_data.get("framework", "scikit-learn")
            )

        with col3:
            features_count = model_data.get("features_count") or metadata.get(
                "n_features", 0)
            st.metric(
                "Features Count",
                features_count
            )

        with col4:
            model_loaded = (
                "Loaded" if model_data.get(
                    "model_loaded", False) else "Not Loaded"
            )
            st.metric(
                "Status",
                model_loaded
            )

        # Training information from report
        if training_report:
            training_summary = training_report.get("training_summary", {})
            st.subheader("Training Information")

            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    f"**Training Status:** {training_summary.get('training_status', 'Unknown')}"
                )
                st.write(
                    f"**Training Timestamp:** {training_summary.get('training_timestamp', 'Unknown')}"
                )
                st.write(
                    f"**Triggered By External:** {training_summary.get('triggered_by_external', 'Unknown')}"
                )

            with col2:
                st.write(
                    f"**Overfitting Risk:** {training_summary.get('overfitting_risk', 'Unknown')}"
                )
                st.write(
                    f"**Training Samples:** {metadata.get('n_training_samples', 'Unknown')}"
                )

        # Selected features
        selected_features = model_data.get(
            "selected_features", []) or metadata.get("feature_names", [])
        if selected_features:
            st.subheader("Selected Features")

            html_content = """
                <style>
                .features-container {
                    background-color: transparent;
                    border: 2px solid #f7d77e;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem 2rem;
                }
                .feature-item {
                    color: white;
                    font-size: 0.95rem;
                    padding: 0.4rem 0;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
                    line-height: 1.6;
                }
                </style>
                <div class="features-container">
                """

            for _, feature in enumerate(selected_features):
                html_content += f'<div class="feature-item"><span class="feature-number"></span>{feature}</div>'

            html_content += "</div>"

            st.markdown(html_content, unsafe_allow_html=True)

            # Feature categories visualization
            self.visualize_feature_categories(selected_features)

    def visualize_feature_categories(self, features):
        """Visualize feature categories."""
        # Categorize features
        categories = {
            "Numerical": [],
            "Categorical_Encoded": [],
            "Engineered": []
        }

        for feature in features:
            if any(keyword in feature.lower() for keyword in ["cc", "age", "mileage", "owner", "accident"]
                   ):
                categories["Numerical"].append(feature)
            elif any(keyword in feature for keyword in ["_", "encoded"]):
                categories["Categorical_Encoded"].append(feature)
            else:
                categories["Engineered"].append(feature)

        # Create pie chart
        category_counts = {k: len(v) for k, v in categories.items() if v}

        if category_counts:
            colors = ["#f7d77e", "#4a4a4a", "#6b6b6b"]

            fig = px.pie(
                names=list(category_counts.keys()),
                values=list(category_counts.values()),
                title="Feature Distribution by Category",
                color_discrete_sequence=colors
            )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", size=12),
                title_font=dict(color="white", size=18),
                showlegend=True,
                legend=dict(
                    bgcolor="rgba(30,30,30,0.8)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(color="white"),
                    x=1.05,
                    y=1,
                    xanchor="left",
                    yanchor="top"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(30, 30, 30, 0.95)",
                    font_size=13,
                    font_color="white",
                    bordercolor="#f7d77e"
                ),
                margin=dict(t=50, b=20, l=20, r=80)
            )

            fig.update_traces(
                textfont=dict(color="white", size=14, weight="bold"),
                marker=dict(
                    line=dict(color="rgba(0,0,0,0)", width=0)
                ),
                pull=[0.05 if i == 0 else 0 for i in range(
                    len(category_counts))]
            )

            st.plotly_chart(fig, use_container_width=True)

    def render_input_schema(self, model_info: Dict[str, Any]):
        """Render input schema information."""
        st.header("Input Schema")

        input_schema = model_info.get("input_schema", {})

        # Required fields
        required_fields = input_schema.get("required_fields", [])
        if required_fields:
            st.subheader("Required Fields")

            html_content = """
                <style>
                .features-container {
                    background-color: transparent;
                    border: 2px solid #f7d77e;
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin-bottom: 1.5rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem 2rem;
                }
                .feature-item {
                    color: white;
                    font-size: 0.95rem;
                    padding: 0.4rem 0;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
                    line-height: 1.6;
                }
                </style>
                <div class="features-container">
                """

            for _, feature in enumerate(required_fields):
                html_content += f'<div class="feature-item"><span class="feature-number"></span>{feature}</div>'

            html_content += "</div>"

            st.markdown(html_content, unsafe_allow_html=True)

        # Valid values
        valid_values = input_schema.get("valid_values", {})
        if valid_values:
            st.subheader("Valid Values for Categorical Fields")

            for field, values in valid_values.items():
                with st.expander(f"{field.replace('_', ' ').title()}"):
                    st.write(f"{', '.join(values)}")

    def render_preprocessing_info(self, model_info: Dict[str, Any]):
        """Render preprocessing pipeline information."""
        st.header("Preprocessing Pipeline")

        preprocessing_steps = model_info.get("preprocessing_steps", [])

        if preprocessing_steps:
            st.subheader("Processing Steps")

            for i, step in enumerate(preprocessing_steps, 1):
                st.write(f"{i}. {step}")

            # Visualize preprocessing flow
            self.visualize_preprocessing_flow(preprocessing_steps)
        else:
            st.info("No preprocessing information available.")

            # Show default preprocessing steps
            st.subheader("Standard Preprocessing Steps")
            default_steps = [
                "Feature Engineering (car_age, has_accident)",
                "Categorical Encoding (fuel_type, brand, transmission, color, insurance_valid)",
                "Service History Encoding",
                "Feature Selection",
                "Feature Normalization"
            ]
            for i, step in enumerate(default_steps, 1):
                st.write(f"{i}. {step}")

    def visualize_preprocessing_flow(self, steps):
        """Create a flow diagram for preprocessing steps."""
        # Create a simple flow visualization
        fig = go.Figure()

        # Add nodes for each step
        for i, step in enumerate(steps):
            fig.add_trace(
                go.Scatter(
                    x=[i],
                    y=[0],
                    mode="markers+text",
                    marker={"size": 60, "color": "#616161"},
                    text=[f"Step {i + 1}"],
                    textposition="middle center",
                    name=step,
                    hovertemplate=f"<b>{step}</b><extra></extra>",
                )
            )

            # Add arrows between steps
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=i + 0.5,
                    y=0,
                    ax=i + 0.3,
                    ay=0,
                    axref="x",
                    ayref="y",
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="white",
                    showarrow=True,
                )

        fig.update_layout(
            title={
                "text": "Preprocessing Pipeline Flow",
                "font": {"color": "#ffffff", "size": 20}
            },
            xaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "range": [-0.5, len(steps) - 0.5],
                "gridcolor": "#333",
            },
            yaxis={
                "showgrid": False,
                "zeroline": False,
                "showticklabels": False,
                "range": [-0.5, -0.5],
                "gridcolor": "#333"
            },
            showlegend=False,
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            font={"color": "#f7d77e", "size": 12},
            hoverlabel=dict(
                bgcolor="rgba(30, 30, 30, 0.95)",
                font_size=13,
                font_color="white",
                bordercolor="#f7d77e"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_model_performance(
            self,
            training_report: Optional[Dict[str, Any]] = None
    ):
        """Render model performance metrics from training report."""
        st.header("Model Performance")

        if not training_report:
            st.warning(
                "Training report not available. Please ensure training_report.json exists.")
            return

        # Extract metrics from training report
        full_results = training_report.get("full_training_results", {})
        training_metrics = full_results.get("training_metrics", {})
        training_summary = training_report.get("training_summary", {})

        if training_metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "MAE",
                    f"{training_metrics.get('test_mae', 0):.2f}",
                    help="Mean Absolute Error on test set"
                )

            with col2:
                st.metric(
                    "RMSE",
                    f"{training_metrics.get('test_rmse', 0):.2f}",
                    help="Root Mean Squared Error on test set"
                )

            with col3:
                st.metric(
                    "R² Score",
                    f"{training_metrics.get('test_r2', 0):.4f}",
                    help="Coefficient of determination"
                )

            with col4:
                st.metric(
                    "MAPE",
                    f"{training_metrics.get('test_mape', 0):.2f}%",
                    help="Mean Absolute Percentage Error"
                )

            # Additional metrics
            st.subheader("Detailed Metrics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Training Set Performance**")

                # Create styled dataframe HTML
                train_metrics_html = f"""
                <style>
                .metrics-table {{
                    background-color: #2a2a2a;
                    border: 1px solid #f7d77e;
                    border-radius: 8px;
                    overflow: hidden;
                    width: 100%;
                    margin: 1rem 0;
                }}
                .metrics-table table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .metrics-table th {{
                    background-color: #2a2a2a;
                    color: #f7d77e;
                    padding: 0.75rem;
                    text-align: left;
                    font-weight: 700;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                }}
                .metrics-table td {{
                    background-color: #2a2a2a;
                    color: white;
                    padding: 0.75rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }}
                .metrics-table tr:last-child td {{
                    border-bottom: none;
                }}
                </style>
                <div class="metrics-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>MSE</td>
                                <td>{training_metrics.get(
                    'train_mse', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>MAE</td>
                                <td>{training_metrics.get(
                        'train_mae', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>R²</td>
                                <td>{training_metrics.get(
                            'train_r2', 0):.4f}</td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>{training_metrics.get(
                                'train_rmse', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>MAPE</td>
                                <td>{training_metrics.get(
                                    'train_mape', 0):.2f}%</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(train_metrics_html, unsafe_allow_html=True)

            with col2:
                st.markdown("**Test Set Performance**")

                # Create styled dataframe HTML
                test_metrics_html = f"""
                <div class="metrics-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>MSE</td>
                                <td>{training_metrics.get('test_mse', 0):.2f}
                                </td>
                            </tr>
                            <tr>
                                <td>MAE</td>
                                <td>{training_metrics.get('test_mae', 0):.2f}
                                </td>
                            </tr>
                            <tr>
                                <td>R²</td>
                                <td>{training_metrics.get('test_r2', 0):.4f}
                                </td>
                            </tr>
                            <tr>
                                <td>RMSE</td>
                                <td>{training_metrics.get('test_rmse', 0):.2f}
                                </td>
                            </tr>
                            <tr>
                                <td>MAPE</td>
                                <td>{training_metrics.get('test_mape', 0):.2f}%
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(test_metrics_html, unsafe_allow_html=True)

            # Cross-validation results
            if "cv_mean_rmse" in training_metrics:
                st.subheader("Cross-Validation Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "CV Mean RMSE",
                        f"{training_metrics.get('cv_mean_rmse', 0):.2f}"
                    )

                with col2:
                    st.metric(
                        "CV Std RMSE",
                        f"{training_metrics.get('cv_std_rmse', 0):.2f}"
                    )

                # Visualize CV scores
                cv_scores = training_metrics.get("cv_scores", [])
                if cv_scores:
                    # Convert negative scores to positive RMSE
                    cv_rmse_scores = [
                        np.sqrt(abs(score)) for score in cv_scores
                    ]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(cv_rmse_scores) + 1)),
                        y=cv_rmse_scores,
                        mode="lines+markers",
                        name="CV RMSE",
                        line=dict(color="#f7d77e", width=2),
                        marker=dict(size=10)
                    ))

                    fig.add_hline(
                        y=training_metrics.get("cv_mean_rmse", 0),
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Mean RMSE"
                    )

                    fig.update_layout(
                        title={
                            "text": "Cross-Validation RMSE Scores",
                            "font": {"color": "#ffffff", "size": 20}
                        },
                        xaxis_title="Fold",
                        yaxis_title="RMSE",
                        height=400,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(20,20,20,0.8)",
                        font=dict(color="white"),
                        xaxis=dict(gridcolor="#333"),
                        yaxis=dict(gridcolor="#333"),
                        hoverlabel=dict(
                            bgcolor="rgba(30, 30, 30, 0.95)",
                            font_size=13,
                            font_color="white",
                            bordercolor="#f7d77e"
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Performance visualization
            self.visualize_performance_metrics(training_metrics)

            # Overfitting analysis
            st.subheader("Overfitting Analysis")
            overfitting_risk = training_summary.get(
                "overfitting_risk", "unknown"
            )

            if overfitting_risk == "low":
                st.markdown(
                    f"""
                    <p style="color: #5dff5d; font-weight: 700; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">✅ Overfitting Risk: <strong>{overfitting_risk.upper()}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            elif overfitting_risk == "medium":
                st.markdown(
                    f"""
                    <p style="color: #ffc107; font-weight: 700; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">⚠️ Overfitting Risk: <strong>{overfitting_risk.upper()}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <p style="color: #ff6b6b; font-weight: 700; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">❌ Overfitting Risk: <strong>{overfitting_risk.upper()}</strong>
                    </p>
                    """,
                    unsafe_allow_html=True
                )

            train_r2 = training_metrics.get("train_r2", 0)
            test_r2 = training_metrics.get("test_r2", 0)
            r2_diff = abs(train_r2 - test_r2)

            st.write(f"**R² Difference (Train vs Test):**  {r2_diff:.4f}")
            if r2_diff < 0.05:
                st.write("✅ Model generalizes well")
            elif r2_diff < 0.1:
                st.write("⚠️ Slight overfitting detected")
            else:
                st.write("❌ Significant overfitting detected")

        else:
            st.info("No performance metrics available.")

    def visualize_performance_metrics(self, training_metrics: Dict[str, Any]):
        """Visualize model performance metrics."""
        metrics = ["MAE", "RMSE", "R² Score"]
        train_values = [
            training_metrics.get("train_mae", 0),
            training_metrics.get("train_rmse", 0),
            training_metrics.get("train_r2", 0),
        ]

        test_values = [
            training_metrics.get("test_mae", 0),
            training_metrics.get("test_rmse", 0),
            training_metrics.get("test_r2", 0),
        ]

        # Create grouped bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    name="Train",
                    x=metrics,
                    y=train_values,
                    marker=dict(color="#f7d77e"),
                    text=[
                        f"{val:.2f}" if i < 2 else f"{val:.4f}"
                        for i, val in enumerate(train_values)
                    ],
                    textposition="auto",
                ),
                go.Bar(
                    name="Test",
                    x=metrics,
                    y=test_values,
                    marker=dict(color="#6b6b6b"),
                    text=[
                        f"{val:.2f}" if i < 2 else f"{val:.4f}" for i, val in enumerate(test_values)
                    ],
                    textposition="auto"
                )
            ]
        )

        fig.update_layout(
            title={
                "text": "Train vs Test Performance Metrics",
                "font": {"color": "#ffffff", "size": 20}
            },
            yaxis_title="Value",
            barmode="group",
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(20,20,20,0.8)",
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            font=dict(color="white"),
            hoverlabel=dict(
                bgcolor="rgba(30, 30, 30, 0.95)",
                font_size=13,
                font_color="white",
                bordercolor="#f7d77e"
            ),
            legend=dict(
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
                font=dict(color="white"),
                x=1.05,
                y=1,
                xanchor="left",
                yanchor="bottom"
            ),
            margin=dict(t=50, b=20, l=20, r=80)
        )

        st.plotly_chart(fig, use_container_width=True)

    def render_feature_importance(
            self,
            training_report: Optional[Dict[str, Any]]
    ):
        """Render feature importance analysis from SHAP values."""
        st.header("Feature Importance")

        if not training_report:
            st.warning(
                "Training report not available for feature importance analysis."
            )
            return

        # Extract SHAP results from training report
        full_results = training_report.get("full_training_results", {})
        shap_results = full_results.get("shap_results", {})

        if not shap_results or not shap_results.get("top_features"):
            st.info("SHAP feature importance not available in training report.")

            # Fallback to coefficient-based feature importance
            validation_results = full_results.get("validation_results", {})
            diagnostics = validation_results.get("diagnostics", {})
            feature_importance = diagnostics.get("feature_importance", {})
            top_features = feature_importance.get("top_5_features", [])

            if top_features:
                st.subheader("Feature Importance - Coefficient Based")

                features = [f["feature"] for f in top_features]
                importance_scores = [f["abs_coefficient"]
                                     for f in top_features]

                table_rows = ""
                for feature, score in zip(features, importance_scores):
                    table_rows += "<tr><td>{}</td><td>{:.2f}</td></tr>".format(
                        feature,
                        score
                    )

                # Create styled table HTML
                importance_html = """
                <style>
                .importance-table {
                    background-color: #2a2a2a;
                    border: 1px solid #f7d77e;
                    border-radius: 8px;
                    overflow: hidden;
                    width: 100%;
                    margin: 1rem 0;
                }
                .importance-table table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .importance-table th {
                    background-color: #2a2a2a;
                    color: #f7d77e;
                    padding: 0.75rem;
                    text-align: left;
                    font-weight: 700;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                }
                .importance-table td {
                    background-color: #2a2a2a;
                    color: white;
                    padding: 0.75rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                }
                .importance-table tr:last-child td {
                    border-bottom: none;
                }
                .importance-table tr:hover td {
                    background-color: #3a3a3a;
                }
                </style>
                <div class="importance-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>Importance</th>
                            </tr>
                        </thead>
                        <tbody>
                        """ + table_rows + """
                        </tbody>
                    </table>
                </div>
                """
                st.markdown(importance_html, unsafe_allow_html=True)

                # Create bar chart
                importance_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance_scores
                }).sort_values("Importance", ascending=False)

                # Create colors: gold for highest, gray for rest
                colors = [
                    "#f7d77e" if i == 0 else "#6b6b6b" for i in range(len(importance_df))
                ]

                # Create bar chart
                fig = px.bar(
                    data_frame=importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 5 Feature Importance (Coefficient Magnitude)",
                )

                # Update with custom colors
                fig.update_traces(
                    marker_color=colors,
                    marker_line_color="rgba(0,0,0,0)",
                    marker_line_width=0
                )

                fig.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(20,20,20,0.8)",
                    font=dict(color="white"),
                    title_font=dict(color="#4da6ff", size=18),
                    xaxis=dict(gridcolor="#333"),
                    yaxis=dict(
                        gridcolor="#333",
                        categoryorder="total ascending"
                    ),
                    hoverlabel=dict(
                        bgcolor="rgba(30, 30, 30, 0.95)",
                        font_size=13,
                        font_color="white",
                        bordercolor="#f7d77e"
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            return

        # Display SHAP information
        st.write(
            f"**Explainer Type:** {shap_results.get('explainer_type', 'Unknown')}"
        )
        st.write(
            f"**Samples Explained:** {shap_results.get('n_samples_explained', 0)}"
        )
        st.write(
            f"**Mean Absolute SHAP Value:** {shap_results.get('mean_abs_shap', 0):.4f}"
        )

        # Extract top features
        top_features = shap_results.get("top_features", [])

        if top_features:
            features = [f["feature"] for f in top_features]
            shap_importance = [f["shap_importance"] for f in top_features]

            # Create DataFrame
            importance_df = pd.DataFrame({
                "Feature": features,
                "SHAP Importance": shap_importance
            }).sort_values("SHAP Importance", ascending=False)

            # Display top features table
            st.subheader("Top Important Features (SHAP)")

            table_rows = ""
            for feature, score in zip(features, shap_importance):
                table_rows += "<tr><td>{}</td><td>{:.2f}</td></tr>".format(
                    feature,
                    score
                )

            # Create styled table HTML
            shap_table_html = """
            <style>
            .shap-table {
                background-color: #2a2a2a;
                border: 1px solid #f7d77e;
                border-radius: 8px;
                overflow: hidden;
                width: 100%;
                margin: 1rem 0;
            }
            .shap-table table {
                width: 100%;
                border-collapse: collapse;
            }
            .shap-table th {
                background-color: #2a2a2a;
                color: #f7d77e;
                padding: 0.75rem;
                text-align: left;
                font-weight: 700;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            .shap-table td {
                background-color: #2a2a2a;
                color: white;
                padding: 0.75rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .shap-table tr:last-child td {
                border-bottom: none;
            }
            .shap-table tr:hover td {
                background-color: #3a3a3a;
            }
            </style>
            <div class="shap-table">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>SHAP Importance</th>
                        </tr>
                    </thead>
                    <tbody>
                    """ + table_rows + """
                    </tbody>
                </table>
            </div>
            """
            st.markdown(shap_table_html, unsafe_allow_html=True)

            # Create colors: gold for highest, gray for rest
            colors = [
                "#f7d77e" if i == 0 else "#6b6b6b" for i in range(len(importance_df))
            ]

            # Create horizontal bar chart
            fig = px.bar(
                importance_df,
                x="SHAP Importance",
                y="Feature",
                orientation="h",
                title="Top Feature Importance Scores (SHAP)",
            )

            # Update with custom colors
            fig.update_traces(
                marker_color=colors,
                marker_line_color="rgba(0,0,0,0)",
                marker_line_width=0
            )

            # Update layout
            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(20,20,20,0.8)",
                font=dict(color="white"),
                title_font=dict(color="#ffffff", size=18),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(
                    gridcolor="#333",
                    categoryorder="total ascending"
                ),
                hoverlabel=dict(
                    bgcolor="rgba(30, 30, 30, 0.95)",
                    font_size=13,
                    font_color="white",
                    bordercolor="#f7d77e"
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance insights
            st.subheader("Feature Insights")

            top_feature = features[0]
            top_importance = shap_importance[0]

            st.write(f"🎯 **Most Important Feature:** {top_feature}")
            st.write(f"   SHAP Importance: {top_importance:.2f}")

            st.write("\n**Top 3 Features Contributing to Predictions:**")
            for i in range(min(3, len(features))):
                st.write(
                    f"{i+1}. **{features[i]}** - Importance: {shap_importance[i]:.2f}")

        else:
            st.info("No SHAP feature importance data available.")

    def render_prediction_insights(
            self,
            training_report: Optional[Dict[str, Any]]
    ):
        """Render prediction insights and statistics."""
        st.header("Prediction Insights")

        if not training_report:
            st.warning("Training report not available for prediction analysis.")
            return

        # Extract prediction analysis from training report
        full_results = training_report.get("full_training_results", {})
        validation_results = full_results.get("validation_results", {})
        diagnostics = validation_results.get("diagnostics", {})
        prediction_analysis = diagnostics.get("prediction_analysis", {})
        residual_analysis = diagnostics.get("residual_analysis", {})

        if prediction_analysis:
            st.subheader("Prediction Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Min Prediction (Train)",
                    f"${prediction_analysis.get('train_pred_min', 0):,.2f}"
                )

            with col2:
                st.metric(
                    "Max Prediction (Train)",
                    f"${prediction_analysis.get('train_pred_max', 0):,.2f}"
                )

            with col3:
                st.metric(
                    "Min Prediction (Test)",
                    f"${prediction_analysis.get('test_pred_min', 0):,.2f}"
                )

            with col4:
                st.metric(
                    "Max Prediction (Test)",
                    f"${prediction_analysis.get('test_pred_max', 0):,.2f}"
                )

            # Actual value ranges
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Actual Price Range (Train)",
                    f"${prediction_analysis.get('train_actual_range', 0):,.2f}"
                )

            with col2:
                st.metric(
                    "Actual Price Range (Test)",
                    f"${prediction_analysis.get('test_actual_range', 0):,.2f}"
                )

        # Residual analysis
        if residual_analysis:
            st.subheader("Residual Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Training Set Residuals**")
                st.write(
                    f"Mean: {residual_analysis.get('train_residual_mean', 0):.6f}")
                st.write(
                    f"Skewness: {residual_analysis.get('train_residual_skewness', 0):.6f}")

                if abs(residual_analysis.get("train_residual_skewness", 0)) < 0.5:
                    st.markdown(
                        """
                        <p style="color: #5dff5d; font-weight: 600; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">✅ Residuals are approximately normally distributed
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ Residuals show some skewness")

            with col2:
                st.write("**Test Set Residuals**")
                st.write(
                    f"Mean: {residual_analysis.get('test_residual_mean', 0):.6f}")
                st.write(
                    f"Skewness: {residual_analysis.get('test_residual_skewness', 0):.6f}")

                if abs(residual_analysis.get('test_residual_skewness', 0)) < 0.5:
                    st.markdown(
                        """
                        <p style="color: #5dff5d; font-weight: 600; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">✅ Residuals are approximately normally distributed
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("⚠️ Residuals show some skewness")

        # Recommendations
        recommendations = full_results.get("recommendations", [])
        if recommendations:
            st.subheader("Model Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")

        # Model readiness
        model_readiness = training_report.get("model_readiness", {})
        if model_readiness:
            st.subheader("Deployment Readiness")

            ready = model_readiness.get("ready_for_deployment", False)
            blockers = model_readiness.get("deployment_blockers", [])

            if ready and not blockers:
                st.markdown(
                    """
                    <p style="color: #5dff5d; font-weight: 700; font-size: 1.1rem; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);">✅ Model is ready for deployment
                    </p>             
                    """,
                    unsafe_allow_html=True,
                )
            elif blockers:
                st.error("❌ **Deployment Blockers Found:**")
                for blocker in blockers:
                    st.write(f"• {blocker}")
            else:
                st.warning("⚠️ **Model readiness status unclear**")

    def render(self):
        """Render the data viewer component."""
        settings = get_settings()
        st.markdown(
            '<h2 style="color: #f7d77e !important; margin-bottom: 1rem; text-shadow: 0 0 10px rgba(247, 215, 126, 0.5);">Data Analytics and Model Insights</h2>',
            unsafe_allow_html=True
        )

        # Load training report
        training_report = self.load_training_report()

        # Fetch model information
        model_info = self.get_model_info()

        if not model_info and not training_report:
            st.error(
                "Unable to fetch model information from API and training report not found.")
            st.info("Please ensure:")
            st.write(
                "1. Your FastAPI server is running on http://localhost:8000"
            )
            st.write(
                "2. Training report exists at: logs/training_report.json"
            )
            return

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Model Overview",
                "Input Schema",
                "Preprocessing",
                "Performance",
                "Feature Importance",
            ]
        )

        with tab1:
            if model_info:
                self.render_model_overview(
                    model_info=model_info,
                    training_report=training_report
                )
            else:
                st.warning("Model info not available from API")

        with tab2:
            if model_info:
                self.render_input_schema(model_info=model_info)
            else:
                st.info(
                    "Input schema not available. Please start the FastAPI server.")

        with tab3:
            if model_info:
                self.render_preprocessing_info(model_info=model_info)
            else:
                st.info("Preprocessing info not available from API")

        with tab4:
            self.render_model_performance(training_report=training_report)

        with tab5:
            self.render_feature_importance(training_report=training_report)

        # Additional section for prediction insights
        st.markdown("---")
        self.render_prediction_insights(training_report=training_report)
