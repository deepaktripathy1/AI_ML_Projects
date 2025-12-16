"""Prefect Monitoring Pipeline Flow."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.deployments import run_deployment
from prefect.runtime import flow_run
from prefect.events import emit_event
from prefect.variables import Variable

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.metrics_tracker import ModelMetricsTracker
from src.utils.path_utils import from_serializable_path

from config.config import get_settings

_GLOBAL_METRICS_TRACKER = None
_GLOBAL_DRIFT_DETECTOR = None


@task
def check_deployment_status() -> Dict[str, Any]:
    """Check if there is a deployed model to monitor."""
    logger = get_run_logger()

    try:
        deployment_info = {
            "has_deployed_model": False,
            "services_healthy": False,
            "deployment_timestamp": None,
            "monitoring_timestamp": datetime.now().isoformat(),
        }

        # Look for deployment summary
        deployment_summary_path = Path("/tmp/deployment_summary.json")

        if deployment_summary_path.exists():
            with deployment_summary_path.open("r") as f:
                deployment_summary = json.load(f)

            deployment_status = deployment_summary.get(
                "deployment_status", {})
            deployment_info.update({
                "has_deployed_model": True,
                "services_healthy": deployment_status.get(
                    "overall_success", False),
                "services_running": deployment_status.get(
                    "services_configured", 0),
                "deployment_timestamp": deployment_summary.get(
                    "deployment_metadata", {}
                ).get("deployment_timestamp"),
                "bentoml_healthy": deployment_summary.get(
                    "bentoml_deployment", {}).get(
                        "service_ready", False),
                "fastapi_healthy": deployment_summary.get(
                    "fastapi_deployment", {}).get(
                        "service_prepared", False),
            })

        # Check if services are actually running (fallback check)
        if not deployment_info["services_healthy"]:
            try:
                bentoml_health = False
                fastapi_health = False

                try:
                    response = requests.get(
                        "http://localhost:3001/health", timeout=5)
                    bentoml_health = response.status_code == 200
                except requests.exceptions.RequestException:
                    pass

                try:
                    response = requests.get(
                        "http://localhost:8000/health", timeout=5)
                    fastapi_health = response.status_code == 200
                except requests.exceptions.RequestException:
                    try:
                        response = requests.get(
                            "http://localhost:8000/", timeout=5)
                        fastapi_health = response.status_code == 200
                    except requests.exceptions.RequestException:
                        pass

                deployment_info.update({
                    "services_healthy": bentoml_health or fastapi_health,
                    "bentoml_healthy": bentoml_health,
                    "fastapi_healthy": fastapi_health,
                    "live_check_performed": True,
                })

            except Exception as e:
                logger.warning(
                    f"Could not perform live service check: {e}")

        # Create deployment status artifact
        create_table_artifact(
            key="deployment-status-check",
            table=[
                {"Service": "BentoML",
                 "Status": "Healthy" if deployment_info.get(
                     "bentoml_healthy") else "Unhealthy"
                 },
                {"Service": "FastAPI",
                 "Status": "Healthy" if deployment_info.get(
                     "fastapi_healthy") else "Unhealthy"
                 },
                {"Overall": "Services",
                 "Status": "Deployed" if deployment_info[
                     "services_healthy"
                 ] else "Not Available"
                 }
            ],
            description=f"Service deployment status for monitoring - {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        logger.info(f"Deployment status check: {deployment_info}")
        return deployment_info

    except Exception as e:
        logger.error(f"Failed to check deployment status: {e}")
        return {
            "has_deployed_model": False,
            "services_healthy": False,
            "deployment_timestamp": None,
            "monitoring_timestamp": datetime.now().isoformat(),
            "check_error": str(e),
        }


@task(
    cache_policy=TASK_SOURCE + INPUTS,
    cache_expiration=timedelta(minutes=15)
)
def initialize_monitoring_components(
        deployment_info: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize drift detector and metrics tracker components."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        if not deployment_info.get("services_healthy", False):
            logger.info(
                "No healthy services found, initializing monitoring components anyway")

        global _GLOBAL_METRICS_TRACKER, _GLOBAL_DRIFT_DETECTOR
        reference_data_path = settings.monitoring_reference_data_path
        drift_threshold = getattr(
            settings, "MONITORING_DRIFT_THRESHOLD", 0.1
        )

        # Initialize drift detector
        if _GLOBAL_DRIFT_DETECTOR is None:
            _GLOBAL_DRIFT_DETECTOR = DriftDetector(
                reference_data_path=reference_data_path,
                drift_threshold=drift_threshold
            )
            logger.info("Drift detector initialized")

        # Initialize Trackers
        if _GLOBAL_METRICS_TRACKER is None:
            _GLOBAL_METRICS_TRACKER = ModelMetricsTracker.get_instance(
                metrics_port=8001,
                drift_detector=_GLOBAL_DRIFT_DETECTOR,
                enable_system_metrics=True
            )

            _GLOBAL_METRICS_TRACKER.start_monitoring(update_interval=60)
            logger.info(
                "Metrics tracker initialized and server started"
            )
        else:
            logger.info("Connected to existing metrics server")

        # Verify FastAPI accessibility
        try:
            import requests
            response = requests.get("http://localhost:8001/health", timeout=5)
            if response.status_code == 200:
                logger.info("FastAPI metrics service is healthy")
            else:
                logger.warning(f"FastAPI metrics service returned status {
                    response.status_code}"
                )
        except Exception as e:
            logger.warning(f"Could not verify FastAPI metrics service: {e}")
            logger.warning("Make sure to run: python run_metrics_server.py")

        components_info = {
            "drift_detector_ready": True,
            "metrics_tracker_ready": True,
            "reference_data_available": Path(
                reference_data_path
            ).exists() if reference_data_path else False,
            "drift_threshold": drift_threshold,
            "metrics_port": 8090,
            "timestamp": datetime.now().isoformat(),
            "components_initialized": True,
        }

        # Store components initialization signal
        Variable.set(
            name="monitoring_components_initialized",
            value="true",
            overwrite=True
        )

        # Create monitoring components initialization artifact
        create_markdown_artifact(
            key="monitoring-components-init",
            markdown=f"""
# Monitoring Components Initialization

## Status: INITIALIZED

### Components Status

- **Drift Detector:** {
                'Ready' if components_info[
                    'drift_detector_ready'
                ] else 'Failed'
            }
- **Metrics Tracker:** {
                'Ready' if components_info[
                    'metrics_tracker_ready'
                ] else 'Failed'
            }
- **Reference Data:** {
                'Available' if components_info[
                    'reference_data_available'
                ] else 'Not Available'
            }

### Configuration
- **Drift Threshold:** {components_info['drift_threshold']}
- **Metrics Port:** {components_info['metrics_port']}
- **Update Interval:** 60 seconds

### Reference Data
- **Path:** {reference_data_path if reference_data_path else 'Not configured'}
- **Available:** {components_info['reference_data_available']}

### Initialization Time
**Timestamp:** {components_info['timestamp']}
            """,
            description="Monitoring components initialization status and configuration"
        )

        logger.info(f"Monitoring components initialized: {components_info}")
        return components_info

    except Exception as e:
        logger.error(f"Failed to initialize monitoring components: {e}")
        return {
            "components_initialized": False,
            "reason": "initialization_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task
def collect_current_data(components_info: Dict[str, Any]) -> Dict[str, Any]:
    """Collect current data for drift analysis."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        if not components_info.get("components_initialized", False):
            logger.info(
                "Components not initialized, skipping data collection")
            return {
                "data_collected": False,
                "reason": "components_not_initialized",
                "timestamp": datetime.now().isoformat()
            }

        # Look for recent prediction data from various sources
        recent_data_sources = [
            "logs/recent_predictions.csv",
            "data/recent_predictions.csv",
            "/tmp/recent_predictions.csv",
            "data/monitoring/current_data.csv"
        ]

        current_data = None
        data_source = None

        for source in recent_data_sources:
            if Path(source).exists():
                try:
                    current_data = pd.read_csv(source)
                    data_source = source
                    logger.info(
                        f"Loaded current data from {source}: {
                            current_data.shape
                        }"
                    )
                    break
                except Exception as e:
                    logger.warning(f"Could not read {source}: {e}")

        if current_data is None or len(current_data) == 0:
            logger.info("No recent data found, generating sample data")
            # Generate sample current data for testing
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            n_samples = 100

            transmissions = ["Manual", "Automatic"]
            colors = ["Black", "Blue", "Gray", "Red", "Silver", "White"]

            sample_data = pd.DataFrame({
                # slightly higher mileage
                "mileage_kmpl": np.random.uniform(10, 28, n_samples),
                # bigger engines trending
                "engine_cc": np.random.randint(1000, 3200, n_samples),
                "owner_count": np.random.choice(
                    [1, 2, 3], n_samples, p=[0.7, 0.25, 0.05]
                ),
                # newer cars on average
                "car_age": np.random.randint(0, 12, n_samples),
                "has_accident": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                # more serviced cars
                "service_history_encoded": np.random.randint(1, 3, n_samples)
            })

            # Fuel type drift: more EVs
            fuel_types = np.random.choice(
                ["electric", "petrol", "diesel"],
                n_samples, p=[0.25, 0.6, 0.15]
            )
            sample_data["fuel_type_electric"] = (fuel_types == "electric")
            sample_data["fuel_type_petrol"] = (fuel_types == "petrol")

            # Brand drift: more Tesla, Hyundai, Toyota
            brands = ["chevrolet", "ford", "honda", "hyundai",
                      "kia", "nissan", "tesla", "toyota", "volkswagen"]
            brand_probs = [0.05, 0.07, 0.08, 0.15, 0.08, 0.1, 0.2, 0.15, 0.12]
            brand_choices = np.random.choice(brands, n_samples, p=brand_probs)
            for b in brands:
                sample_data[f"brand_{b}"] = (brand_choices == b)

            # Color drift: more white and gray
            colors = ["blue", "gray", "red", "silver", "white"]
            color_probs = [0.05, 0.25, 0.10, 0.20, 0.40]
            color_choices = np.random.choice(colors, n_samples, p=color_probs)
            for c in colors:
                sample_data[f"color_{c}"] = (color_choices == c)

            # Transmission drift: more automatics (so fewer manuals)
            sample_data["transmission_manual"] = np.random.choice(
                [True, False], n_samples, p=[0.4, 0.6])

            # Insurance validity remains high
            sample_data["insurance_valid_yes"] = np.random.choice(
                [True, False], n_samples, p=[0.92, 0.08])

            # === Drifted target generation ===
            base_price = (
                65000  # increased average market value
                + (sample_data["engine_cc"] - 1000) * 5.5
                + sample_data["mileage_kmpl"] * 450
                - sample_data["car_age"] * 1400
                - sample_data["owner_count"] * 1800
                - sample_data["has_accident"] * 2500
                + np.where(sample_data["brand_tesla"], 15000, 0)
                + np.where(sample_data["brand_volkswagen"], 5000, 0)
                + np.random.normal(0, 3000, n_samples)
            ).clip(5000, 130000)

            sample_data["price_usd"] = base_price

            # Maintain reference order
            reference_columns = [
                "mileage_kmpl", "engine_cc", "owner_count", "car_age", "has_accident", "service_history_encoded", "fuel_type_electric", "fuel_type_petrol", "brand_chevrolet", "brand_ford", "brand_honda", "brand_hyundai", "brand_kia", "brand_nissan", "brand_tesla", "brand_toyota", "brand_volkswagen", "transmission_manual", "color_blue", "color_gray", "color_red", "color_silver", "color_white", "insurance_valid_yes", "price_usd"
            ]

            sample_data = sample_data[reference_columns]

            # Convert bool to int
            bool_cols = [
                "has_accident",
                "fuel_type_electric",
                "fuel_type_petrol",
                "brand_chevrolet",
                "brand_ford",
                "brand_honda",
                "brand_hyundai",
                "brand_kia",
                "brand_nissan",
                "brand_tesla",
                "brand_toyota",
                "brand_volkswagen",
                "transmission_manual",
                "color_blue",
                "color_gray",
                "color_red",
                "color_silver",
                "color_white",
                "insurance_valid_yes"
            ]

            for col in bool_cols:
                if col in sample_data.columns:
                    sample_data[col] = sample_data[col].astype(int)

            current_data = sample_data
            data_source = "generated_sample"
            logger.info("Generated sample current data for monitoring")

        data_info = {
            "data_collected": True,
            "data_source": data_source,
            "records_count": len(current_data),
            "features_count": len(current_data.columns),
            "timestamp": datetime.now().isoformat(),
            "data_shape": current_data.shape,
        }

        # Save current data
        data_dir = settings.data_dir
        current_data_path = data_dir / "monitoring" / "current_data.csv"

        current_data.to_csv(
            from_serializable_path(str(current_data_path)),
            index=False
        )
        data_info["current_data_path"] = str(current_data_path)
        logger.info(f"Current data collected: {data_info}")
        return data_info

    except Exception as e:
        logger.error(f"Failed to collect current data: {e}")
        return {
            "data_collected": False,
            "reason": "collection_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task
def run_drift_analysis(
    data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Run comprehensive drift analysis using DriftDetector."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        if not data_info.get("data_collected", False):
            logger.info("Skipping drift analysis - data not collected")
            return {
                "drift_analysis_completed": False,
                "reason": "data_not_collected",
                "timestamp": datetime.now().isoformat(),
            }

        # Load current data
        path = str(data_info.get("current_data_path"))
        current_data_path = from_serializable_path(path_str=path)
        if not current_data_path.exists():
            raise Exception("Current data file not found")

        current_data = pd.read_csv(current_data_path)

        analysis_results = {}
        model_results = {}

        global _GLOBAL_DRIFT_DETECTOR

        if _GLOBAL_DRIFT_DETECTOR is not None:
            drift_detector = _GLOBAL_DRIFT_DETECTOR

            # Perform drift analysis
            if drift_detector.reference_data is not None:
                logger.info("Running data drift analysis with reference data")
                drift_results = drift_detector.detect_data_drift(
                    current_data=current_data,
                    reference_data=drift_detector.reference_data
                )

                logger.info("Running model drift analysis")
                model_drift_results = drift_detector.detect_model_drift(
                    current_data=current_data,
                    reference_data=drift_detector.reference_data,
                    current_predictions=None,
                    reference_predictions=None
                )

                analysis_results = {
                    "drift_analysis_completed": True,
                    "drift_results": drift_results,
                    "model_results": model_drift_results,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "records_analyzed": len(current_data),
                }

                model_results = model_drift_results
            else:
                logger.warning("No reference data available")

            # Store drift results
            results_dir = settings.logs_dir / "drift_reports"
            results_dir.mkdir(parents=True, exist_ok=True)

            results_file = results_dir / \
                f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with results_file.open("w") as f:
                json.dump(analysis_results, f, indent=2, default=str)

            analysis_results["results_file"] = str(results_file)

            # Create drift analysis artifact
            drift_detected = analysis_results["drift_results"].get(
                "dataset_drift_detected", False
            )
            drift_score = analysis_results["drift_results"].get(
                "drift_score", 0.0
            )
            quality_results = analysis_results["drift_results"].get(
                "data_quality_issues", []
            )
            performance_degradation = model_results.get(
                "performance_degradation", False
            )
            prediction_drift = model_results.get(
                "prediction_drift_detected", False
            )

            recommendations = '\n'.join(
                filter(
                    None, [
                        '- Consider retraining' if drift_detected else '- Continue',
                        '- Investigate data quality issues' if quality_results else ''
                    ]
                ))

            create_markdown_artifact(
                key="drift-analysis-results",
                markdown=f"""
# Drift Analysis Results

## Overall Status: {'DRIFT DETECTED' if drift_detected else 'NO DRIFT'}

### Drift Detection
- **Dataset Drift:** {'Yes' if drift_detected else 'No'}
- **Drift Score:** {drift_score:.3f}

### Model Drift Detection
- **Performance Degradation:** {'Yes' if performance_degradation else 'No'}
- **Prediction Drift:** {'Yes' if prediction_drift else 'No'}
- **Failed Tests:** {len(model_results.get('test_failures', []))}

### Data Quality
- **Quality Issues:** {len(quality_results)}
- **Records Analyzed:** {analysis_results['records_analyzed']:,}

### Drifted Features
{
                    chr(10).join(
                        [f"- {feature}" for feature in analysis_results[
                            'drift_results'
                        ].get('drifted_features', [])]
                    ) or '- None detected'
                }

### Analysis Details
- **Analysis Time:** {analysis_results['analysis_timestamp']}
- **Results File:** {results_file.name}
- **Reference Data Used:** {
                    'Yes' if drift_detector is not None else 'No'
                }

### Recommendations
{recommendations}
""",
                description=(
                    f"Drift analysis results - {
                        'Drift detected' if drift_detected or performance_degradation else 'No drift'
                    } (Score: {drift_score:.3f})"
                )
            )

            logger.info(
                f"Drift analysis completed: drift={
                    'detected' if drift_detected else 'none'
                }, model_drift = {
                    'detected' if performance_degradation else 'none'
                }")
            return analysis_results

        else:
            return {
                "drift_analysis_completed": False,
                "reason": "analysis_failed",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logger.error(f"Drift analysis failed: {e}")
        return {
            "drift_analysis_completed": False,
            "reason": "analysis_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task
def update_metrics_tracker(
        analysis_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Update metrics tracker with analysis results."""
    logger = get_run_logger()

    try:
        if not analysis_results.get("drift_analysis_completed", False):
            logger.info("Skipping metrics update - analysis not completed")
            return {
                "metrics_updated": False,
                "reason": "analysis_not_completed",
                "timestamp": datetime.now().isoformat(),
            }

        global _GLOBAL_METRICS_TRACKER

        if _GLOBAL_METRICS_TRACKER is not None:
            metrics_tracker = _GLOBAL_METRICS_TRACKER

            # Update drift metrics
            drift_results = analysis_results.get("drift_results", {})
            model_results = analysis_results.get("model_results", {})

            # metrics_tracker.start_metrics_server()
            metrics_tracker.start_monitoring()

            metrics_tracker.record_drift_check(
                drift_results=drift_results,
                model_results=model_results
            )
            logger.info("Drift metrics updated")

            # Export current monitoring data
            export_path = metrics_tracker.export_monitoring_data()

            metrics_update_result = {
                "metrics_updated": True,
                "drift_metrics_updated": bool(drift_results),
                "model_metrics_updated": bool(model_results),
                "system_metrics_updated": True,
                "export_path": export_path,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(f"Metrics tracker updated: {metrics_update_result}")
            return metrics_update_result
        else:
            return {
                "metrics_updated": False,
                "reason": "metrics tracker missing",
                "timestamp": datetime.now().isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to update metrics tracker: {e}")
        return {
            "metrics_updated": False,
            "reason": "update_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@task
def evaluate_retraining_need(
        analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate if model retraining is needed based on comprehensive analysis."""
    logger = get_run_logger()

    try:
        needs_retraining = False
        reasons = []
        severity = "low"
        flow_id = flow_run.get_id()

        if analysis_results.get("drift_analysis_completed", False):
            drift_results = analysis_results.get("drift_results", {})
            model_results = analysis_results.get("model_resuts", {})

            if drift_results.get("dataset_drift_detected", False):
                drift_score = drift_results.get("drift_score", 0)
                if drift_score > 0.2:  # High drift threshold
                    needs_retraining = True
                    severity = "high"
                    reasons.append(
                        f"Significant data drift detected (score: {
                            drift_score:.3f})")
                    emit_event(
                        event="used_car_model.retraining_required",
                        resource={
                            "prefect.resource.id": f"prefect.flow.{flow_id}"
                        },
                        payload={"drift_score": drift_score}
                    )
                elif drift_score > 0.1:  # Medium drift threshold
                    severity = "medium"
                    reasons.append(
                        f"Moderate data drift detected (score: {
                            drift_score:.3f})")

            # Check for model performance degradation
            if model_results.get("performance_degradation", False):
                needs_retraining = True
                severity = "high"
                r2_change = model_results.get("performance_change", {}).get(
                    "r2_change", 0
                )
                reasons.append(
                    f"Model performance degradation detected (R² change: {
                        r2_change:.3f
                    })")
                emit_event(
                    event="used_car_model.retraining_required",
                    resource={
                        "prefect.resource.id": f"prefect.flow.{flow_id}"
                    },
                    payload={
                        "performance_degradation": True,
                        "r2_change": r2_change
                    }
                )

            # Check test failures
            failed_tests = model_results.get("test_failures", [])
            if len(failed_tests) > 3:
                needs_retraining = True
                severity = "high"
                reasons.append(
                    f"Multiple test failures: {len(failed_tests)} tests failed")
                emit_event(
                    event="used_car_model.retraining_required",
                    resource={
                        "prefect.resource.id": f"prefect.flow.{flow_id}"
                    },
                    payload={"failed_tests": len(failed_tests)}
                )

            # Check individual feature drift
            drifted_features = drift_results.get("drifted_features", [])
            if len(drifted_features) > 3:  # More than 3 features drifted
                needs_retraining = True
                severity = "high"
                reasons.append(
                    f"Multiple features drifted: {drifted_features}")
                emit_event(
                    event="used_car_model.retraining_required",
                    resource={
                        "prefect.resource.id": f"prefect.flow.{flow_id}"
                    },
                    payload={"drifted_features": len(drifted_features)}
                )

        # Check time since last training to prevent too frequent retraining
        if needs_retraining:
            try:
                last_training_date_str = Variable.get(
                    "last_training_date", default=None)
                if last_training_date_str:
                    last_training = datetime.fromisoformat(
                        str(last_training_date_str))
                    hrs_passed = (
                        datetime.now() - last_training).total_seconds() / 3600

                    # Don't retrain more than once per day
                    if hrs_passed < 24:
                        needs_retraining = False
                        reasons.append(
                            f"Recent training complete {hrs_passed:.1f} hours ago"
                        )
            except Exception as e:
                logger.warning(f"Could not check last training date: {e}")

        retraining_result = {
            "needs_retraining": needs_retraining,
            "reasons": reasons,
            "severity": severity,
            "analysis_summary": {
                "drift_detected": analysis_results.get(
                    "drift_results", {}
                ).get("dataset_drift_detected", False),
                "drift_score": analysis_results.get(
                    "drift_results", {}
                ).get("drift_score", 0),
                "performance_degradation": analysis_results.get(
                    "model_results", {}
                ).get("performance_degradation", False),
                "drifted_features_count": len(
                    analysis_results.get("drift_results", {}).get(
                        "drifted_features", [])
                ),
                "failed_tests_count": len(
                    analysis_results.get("model_results", {}).get(
                        "test_failures", []
                    )
                )
            },
            "evaluation_timestamp": datetime.now().isoformat(),
        }

        recommendations_list = []

        if needs_retraining:
            if severity == "high":
                recommendations_list.append("- Schedule retraining")
            elif severity == "medium":
                recommendations_list.append(
                    "- Plan retraining within 24-48 hours")
        else:
            recommendations_list.append("- Continue regular monitoring")

        recommendations = "\n".join(recommendations_list)

        # Create retraining evaluation artifact
        create_markdown_artifact(
            key="retraining-evaluation",
            markdown=f"""
# Model Retraining Evaluation

## Decision: {'RETRAINING NEEDED' if needs_retraining else 'NO RETRAINING NEEDED'}

### Evaluation Summary
- **Severity Level:** {severity.upper()}
- **Decision:** {'Retrain Model' if needs_retraining else 'Continue Monitoring'}

### Analysis Results
- **Drift Detected:** {retraining_result['analysis_summary']['drift_detected']}
- **Drift Score:** {retraining_result['analysis_summary']['drift_score']:.3f}
- **Performance Degradation:** {retraining_result['analysis_summary'][
                'performance_degradation']}
- **Drifted Features:** {
                retraining_result[
                    'analysis_summary'
                ]['drifted_features_count']
            }
- **Failed Tests:** {
                retraining_result[
                    'analysis_summary'
                ]['failed_tests_count']
            }

### Reasons
{chr(10).join([f"- {reason}" for reason in reasons])
                or '- No critical issues detected'}

### Recommendations
{recommendations}
- Review data sources for quality improvements
- Monitor prediction performance closely

### Evaluation Time
**Timestamp:** {retraining_result['evaluation_timestamp']}
            """,
            description=(
                f"Model retraining evaluation - {
                    'Needed' if needs_retraining else 'Not needed'
                } (Severity: {severity})"
            )
        )

        logger.info(f"Retraining evaluation: {retraining_result}")
        return retraining_result

    except Exception as e:
        logger.error(f"Retraining evaluation failed: {e}")
        return {
            "needs_retraining": False,
            "reasons": [f"Evaluation failed: {str(e)}"],
            "severity": "low",
            "evaluation_timestamp": datetime.now().isoformat(),
        }


@task
def send_monitoring_alerts(
    deployment_info: Dict[str, Any],
    analysis_results: Dict[str, Any],
    metrics_result: Dict[str, Any],
    retraining_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Send monitoring alerts based on comprehensive analysis."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        enable_alerts = settings.MONITORING_ENABLE_ALERTS

        if not enable_alerts:
            logger.info("Monitoring alerts disabled in settings")
            return {"alerts_sent": 0, "alerts_disabled": True}

        # Collect alert messages
        alert_messages = []
        alert_severity = "low"

        # Service health alerts
        if not deployment_info.get("services_healthy", False):
            alert_messages.append("Model serving services are not healthy")
            alert_severity = "high"

        # Drift alerts
        if analysis_results.get("drift_analysis_completed", False):
            drift_results = analysis_results.get("drift_results", {})
            if drift_results.get("dataset_drift_detected", False):
                score = drift_results.get("drift_score", 0)
                alert_messages.append(
                    f"Data drift detected (score: {score:.3f})")
                if score > 0.15:
                    alert_severity = "high"
                elif score > 0.1 and alert_severity == "low":
                    alert_severity = "medium"

            # Model performance alerts
            model_results = analysis_results.get("model_results", {})
            if model_results.get("performance_degradation", False):
                alert_messages.append("Model performance degradation detected")
                alert_severity = "high"

            if model_results.get("prediction_drift_detected", False):
                alert_messages.append("Prediction drift detected")
                if alert_severity == "low":
                    alert_severity = "medium"

        # Retraining alerts
        if retraining_result.get("needs_retraining", False):
            reasons = retraining_result.get("reasons", [])
            alert_messages.append(
                f"Model retraining recommended: {', '.join(reasons)}")
            if retraining_result.get("severity") == "high":
                alert_severity = "high"

        # Metrics alerts
        if not metrics_result.get("metrics_updated", False):
            alert_messages.append("Metrics tracking issues detected")
            if alert_severity == "low":
                alert_severity = "medium"

        # Prepare alert summary
        alert_summary = {
            "alert_count": len(alert_messages),
            "severity": alert_severity,
            "messages": alert_messages,
            "services_healthy": deployment_info.get("services_healthy", False),
            "drift_detected": analysis_results.get("drift_results", {}).get(
                "dataset_drift_detected", False),
            "performance_degradation": analysis_results.get(
                "model_results", {}).get(
                "performance_degradation", False),
            "retraining_needed": retraining_result.get(
                "needs_retraining", False),
            "alert_timestamp": datetime.now().isoformat(),
        }

        if alert_messages:
            # Create monitoring alert artifact
            create_markdown_artifact(
                key="monitoring-alerts",
                markdown=f"""
# Monitoring Alert Summary

## Alert Level: {alert_severity.upper()}

### Issues Detected ({len(alert_messages)})
{chr(10).join([f"- {msg}" for msg in alert_messages])}

### System Status
- **Services Healthy:** {deployment_info.get("services_healthy", False)}
- **BentoML:** {'Healthy' if deployment_info.get("bentoml_healthy") else 'Unhealthy'}
- **FastAPI:** {'Healthy' if deployment_info.get("fastapi_healthy") else 'Unhealthy'}

### Analysis Summary
- **Records Analyzed:** {analysis_results.get("records_analyzed", 0):,}
- **Drift Score:** {analysis_results.get("drift_results", {}).get("drift_score", 0):.3f}
- **Performance Degradation:** {analysis_results.get(
                    "model_results", {}
                ).get("performance_degradation", False)}
- **Failed Tests:** {len(analysis_results.get(
                    "model_results", {}
                ).get("test_failures", []))}

### Monitoring Resources
- **Metrics Endpoint:** http://localhost:8090/metrics
- **Service Health:** http://localhost:3000/health, http://localhost:8000/health

**Alert Time:** {alert_summary['alert_timestamp']}
                """,
                description=(
                    f"Monitoring alerts - {
                        alert_severity.upper()
                    } severity ({len(alert_messages)} issues)"
                )
            )

            logger.warning(
                f"MONITORING ALERTS ({
                    alert_severity.upper()}): {
                        len(alert_messages)
                } issues detected"
            )
        else:
            logger.info("No monitoring alerts to send - all systems healthy")

        return alert_summary

    except Exception as e:
        logger.error(f"Alert processing failed: {e}")
        return {
            "alerts_sent": 0,
            "alert_error": str(e),
            "alert_timestamp": datetime.now().isoformat(),
        }


@task
def update_monitoring_variables(
    analysis_results: Dict[str, Any],
    retraining_result: Dict[str, Any],
    alert_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Update Prefect variables with monitoring state."""
    logger = get_run_logger()

    try:
        # Update monitoring state variables
        monitoring_state = {
            "last_monitoring_run": datetime.now().isoformat(),
            "last_drift_check": datetime.now().isoformat(),
            "drift_detected": analysis_results.get(
                "drift_results", {}).get(
                    "dataset_drift_detected", False),
            "drift_score": analysis_results.get(
                "drift_results", {}).get("drift_score", 0),
            "performance_degradation": analysis_results.get(
                "model_results", {}).get("performance_degradation", False),
            "retraining_recommended": retraining_result.get(
                "needs_retraining", False),
            "retraining_severity": retraining_result.get("severity", "low"),
            "alert_count": alert_summary.get("alert_count", 0),
            "overall_status": "HEALTHY" if alert_summary.get(
                "alert_count", 0) == 0 else "ATTENTION_NEEDED",
        }

        # Update Prefect Variables
        for key, value in monitoring_state.items():
            Variable.set(key, str(value), overwrite=True)

        logger.info(f"Updated monitoring variables: {monitoring_state}")
        return monitoring_state

    except Exception as e:
        logger.warning(f"Failed to update monitoring variables: {e}")
        return {"update_error": str(e)}


@flow(
    name="monitoring-pipeline",
    description="Model monitoring pipeline with drift detection",
    flow_run_name=f"monitoring-pipeline-{
        datetime.now().strftime('%Y%m%d-%H%M%S')}",
)
def monitoring_pipeline() -> Dict[str, Any]:
    """Model monitoring pipeline flow with comprehensive drift detection."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        # Check deployment status
        deployment_check = check_deployment_status()

        # Initialize monitoring components
        component_init = initialize_monitoring_components(deployment_check)

        # Collect current data for analysis
        data_collection = collect_current_data(
            components_info=component_init
        )

        # Run drift analysis
        drift_analysis = run_drift_analysis(
            data_info=data_collection,
        )

        # Update metrics tracker
        metrics_update = update_metrics_tracker(
            analysis_results=drift_analysis,
        )

        # Evaluate retraining needs
        retraining_evaluation = evaluate_retraining_need(
            analysis_results=drift_analysis
        )

        # Send alerts if needed
        alert_task = send_monitoring_alerts(
            deployment_info=deployment_check,
            analysis_results=drift_analysis,
            metrics_result=metrics_update,
            retraining_result=retraining_evaluation,
        )

        # Update monitoring variables
        variable_update = update_monitoring_variables(
            analysis_results=drift_analysis,
            retraining_result=retraining_evaluation,
            alert_summary=alert_task,
        )

        # Trigger retraining if needed
        if retraining_evaluation.get("needs_retraining", False):
            if getattr(settings, "MONITORING_TEST_MODE", False):
                logger.warning(
                    "Test mode ON — retraining skipped to preserve metrics server.")
            else:
                logger.info(
                    "Retraining needed, triggering training pipeline..."
                )
                try:
                    retraining_run = run_deployment(
                        name="training-pipeline/training-pipeline",
                        timeout=10800  # 3 hours timeout
                    )
                    logger.info(
                        f"Retraining pipeline triggered: {retraining_run}"
                    )
                except Exception as e:
                    logger.error(f"Failed to trigger retraining pipeline: {e}")

        # Create final monitoring state
        monitoring_state = {
            "status": "SUCCESS",
            "completion_time": datetime.now().isoformat(),
            "services_healthy": deployment_check.get(
                "services_healthy", False),
            "drift_detected": drift_analysis.get(
                "drift_results", {}).get(
                    "dataset_drift_detected", False),
            "drift_score": drift_analysis.get(
                "drift_results", {}).get("drift_score", 0),
            "performance_degradation": drift_analysis.get(
                "model_results", {}).get(
                    "performance_degradation", False),
            "retraining_recommended": (
                retraining_evaluation.get(
                    "needs_retraining", False
                )),
            "alert_count": alert_task.get("alert_count", 0),
            "overall_status": variable_update.get(
                "overall_status", "HEALTHY"),
            "records_analyzed": drift_analysis.get("records_analyzed", 0),
        }

        # Store state in Prefect variable
        Variable.set(
            "monitoring_pipeline_state",
            json.dumps(monitoring_state, default=str),
            overwrite=True
        )

        return monitoring_state

    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {e}")

        # Store failure state
        failure_state = {
            "status": "FAILED",
            "completion_time": datetime.now().isoformat(),
            "error": str(e),
            "services_healthy": False,
            "drift_detected": False,
            "overall_status": "FAILED",
        }

        Variable.set(
            "monitoring_pipeline_state",
            json.dumps(failure_state, default=str),
            overwrite=True
        )

        raise


if __name__ == "__main__":
    monitoring_pipeline()
