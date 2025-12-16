"""Prefect data pipeline flow handles data ingestion and preprocessing."""

import json
from datetime import datetime
from typing import Any, Dict

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.runtime import flow_run
from prefect.variables import Variable

from src.feature_store.feature_pipeline import FeaturePipeline
from src.data_ingestion.mongodb_connector import MongoDBConnector

from config.config import get_settings


@task()
def detect_db_data() -> dict[str, Any]:
    """Check for raw data in MongoDB collection."""
    logger = get_run_logger()
    settings = get_settings()

    try:
        mongodb_connector = MongoDBConnector(
            connection_string=settings.MONGODB_URL,
            database_name=settings.MONGODB_NAME,
            collection_name=settings.MONGODB_COLLECTION
        )

        mongodb_connector.connect()

        if not mongodb_connector.test_connection():
            raise RuntimeError("Cannot connect to MongoDB")

        # Get collection statistics
        stats = mongodb_connector.get_collection_info()
        document_count = stats.get("document_count", 0)
        logger.info(f"Current document count: {document_count}")

        if document_count == 0:
            raise RuntimeError("No Data Found")

        # Close connection
        mongodb_connector.close()

        return {
            "has_data": True,
            "document_count": document_count,
            "stats": stats,
            "detection_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to detect MongoDB data: {e}")
        return {
            "has_data": False,
            "document_count": None,
            "stats": None,
            "detection_time": datetime.now().isoformat()
        }


@task
def run_feature_pipeline(
    db_check_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute the complete feature pipeline."""
    logger = get_run_logger()

    if not db_check_result.get("has_data"):
        raise RuntimeError("No data available in MongoDB")

    try:
        """Initialize and run feature pipeline."""
        feature_pipeline = FeaturePipeline()
        logger.info("Initialized FeaturePipeline")

        # Run complete pipeline with feature store creation
        pipeline_results = feature_pipeline.run_complete_pipeline(
            create_feature_group=True,
            create_feature_view=True,
            feature_selection_method="rfecv"
        )

        logger.info("Feature pipeline completed successfully")
        return pipeline_results

    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        raise


@task
def create_pipeline_artifacts(
    pipeline_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Create Prefect artifacts for visualization and documentation."""
    logger = get_run_logger()

    try:
        # Extract key metrics
        data_shapes = pipeline_results.get("data_shapes", {})
        feature_counts = pipeline_results.get("feature_counts", {})
        data_quality = pipeline_results.get("data_quality", {})
        selected_features = pipeline_results.get("selected_features", [])
        recommendations = pipeline_results.get("recommendations", [])
        quality_score = pipeline_results.get("quality_score", 0)

        # Create data processing flow artifact
        create_markdown_artifact(
            key="data-processing-flow",
            markdown=f"""
# Data Processing Flow

## Pipeline Status: SUCCESS

### Data Shape Transformations
| Stage | Records | Features |
|-------|---------|----------|
| Raw Data | {
                data_shapes.get('raw', (0, 0))[0]
            } | {
                data_shapes.get('raw', (0, 0))[1]
            } |
| Cleaned Data | {
                data_shapes.get('cleaned', (0, 0))[0]
            } | {
                data_shapes.get('cleaned', (0, 0))[1]
            } |
| Engineered Data | {
                data_shapes.get('engineered', (0, 0))[0]
            } | {
                data_shapes.get('engineered', (0, 0))[1]
            } |
| Final Data | {
                data_shapes.get('final', (0, 0))[0]
            } | {
                data_shapes.get('final', (0, 0))[1]
            } |

### Feature Counts by Stage
- **Raw Features:** {feature_counts.get('raw', 0)}
- **Cleaned Features:** {feature_counts.get('cleaned', 0)}
- **Engineered Features:** {feature_counts.get('engineered', 0)}
- **Final Features:** {feature_counts.get('final', 0)}
- **Selected Features:** {feature_counts.get('selected', 0)}

### Data Quality Metrics
- **Quality Score:** {quality_score:.2f}%
- **Null Values:** {data_quality.get('null_values', 0)}
- **Duplicate Rows:** {data_quality.get('duplicate_rows', 0)}
- **Memory Usage:** {data_quality.get('memory_usage_mb', 0):.2f} MB

### Feature Store
- **Type:** {pipeline_results.get('feature_store_type', 'Unknown')}
- **Timestamp:** {pipeline_results.get('pipeline_timestamp', 'N/A')}
""",
            description="Complete data processing flow summary"
        )

        # Create selected features artifact
        if selected_features:
            features_table = [
                # Show first 20
                {"Feature": feature, "Selected": True} for feature in selected_features[:20]
            ]
            create_table_artifact(
                key="selected-features",
                table=features_table,
                description=f"Selected {
                    len(selected_features)
                } features for training"
            )

        # Create recommendations artifact
        if recommendations:
            recommendations_md = "\n".join([
                f"- {rec}" for rec in recommendations
            ])
            create_markdown_artifact(
                key="pipeline-recommendations",
                markdown=f"""
# Pipeline Recommendations

{recommendations_md}


---
*Generated by FeaturePipeline on {
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }*
""",
                description="Data quality and pipeline improvement recommendations"
            )

        # Create detailed reports artifact
        reports = pipeline_results.get("reports", {})
        cleaning_report = reports.get("cleaning", {})
        feature_engineering_report = reports.get(
            "feature_engineering", {}
        )
        feature_selection_report = reports.get(
            "feature_selection", {}
        )

        create_markdown_artifact(
            key="detailed-report",
            markdown=f"""
# Detailed Pipeline Reports

## Data Cleaning Report
- ** # Steps Applied:** {len(cleaning_report.get('cleaning_steps', []))}
- **Cleaning Steps:** {cleaning_report.get('cleaning_steps', [])}

## Feature Engineering Report
- **# Engineering Steps:** {
                len(feature_engineering_report.get('engineering_steps', 'N/A'))
            }
- **Engineering Steps:** {
                feature_engineering_report.get('engineering_steps', [])
            }

## Feature Selection Report
- **Selection Method:** {feature_selection_report.get('method', 'N/A')}
- **Features Selected:** {feature_selection_report.get('final_features', "N/A")}
- **Importance Scores:** Available in MLflow

""",
            description="Detailed reports from each pipeline stage"
        )

        logger.info("Created all pipeline artifacts")

        return {
            "artifacts_created": True,
            "artifact_count": 3,
            "creation_time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to create artifacts: {e}")
        raise


@task()
def validate_and_summarize(
        pipeline_results: Dict[str, Any],
        artifact_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate pipeline results and create completion summary."""
    logger = get_run_logger()

    # Extract metrics
    data_shapes = pipeline_results.get("data_shapes", {})
    feature_counts = pipeline_results.get("feature_counts", {})
    quality_score = pipeline_results.get("quality_score", 0)

    # Artifact info
    artifacts_created = artifact_info.get("artifacts_created", False)
    num_artifacts = artifact_info.get("artifact_count", None)

    # Validation checks
    validation_warnings = []
    final_records = data_shapes.get("final", (0, 0))[0]
    selected_features_count = feature_counts.get("selected", 0)

    if final_records < 100:
        validation_warnings.append(
            f"Only {final_records} records in final data (expected >= 100)"
        )

    if selected_features_count < 5:
        validation_warnings.append(
            f"Only {selected_features_count} features selected (expected >= 5)"
        )

    if quality_score < 80:
        validation_warnings.append(
            f"Quality score {quality_score:.2f}% is below 80%"
        )

    pipeline_summary = {
        "execution_date": datetime.now().isoformat(),
        "flow_run_id": str(flow_run.get_id()),
        "flow_run_name": flow_run.get_name(),
        "feature_store_type": pipeline_results.get(
            "feature_store_type", "Unknown"
        ),
        "data_shape": {
            "raw": data_shapes.get("raw", (0, 0)),
            "cleaned": data_shapes.get("cleaned", (0, 0)),
            "engineered": data_shapes.get("engineered", (0, 0)),
            "final": data_shapes.get("final", (0, 0)),
        },
        "selected_features_count": selected_features_count,
        "quality_score": quality_score,
        "pipeline_status": "SUCCESS" if not validation_warnings else "SUCCESS_WITH_WARNINGS",
        "validation_warnings": validation_warnings,
        "recommendations": pipeline_results.get("recommendations", None),
        "artifact_info": {
            "artifacts_created": artifacts_created,
            "num_artifacts": num_artifacts,
        }
    }

    logger.info(f"Pipeline summary: {json.dumps(pipeline_summary, indent=2)}")

    return pipeline_summary


@task()
def cleanup_on_completion() -> Dict[str, Any]:
    """Cleanup and prepare for next pipeline stage."""
    logger = get_run_logger()

    try:
        logger.info("Pipeline cleanup completed")
        return {
            "cleanup_status": "completed",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"cleanup_status": "failed", "error": str(e)}


@flow(
    name="data-pipeline",
    description="Used car price prediction data pipeline",
    flow_run_name=f"data-pipeline-{
        datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
def data_pipeline() -> Dict[str, Any]:
    """Data Pipeline flow for used car prediction."""
    logger = get_run_logger()

    try:
        # Detect data in MongoDB
        logger.info("Starting data pipeline...")
        data_detection_result = detect_db_data()

        # Run complete feature pipeline
        pipeline_results = run_feature_pipeline(
            db_check_result=data_detection_result
        )

        # Create Prefect artifacts for monitoring
        artifact_info = create_pipeline_artifacts(
            pipeline_results=pipeline_results
        )

        # Validate and summarize
        pipeline_summary = validate_and_summarize(
            pipeline_results=pipeline_results,
            artifact_info=artifact_info
        )

        # Cleanup
        cleanup_result = cleanup_on_completion()

        # Store final state in Prefect variable for downstream workflows
        final_state = {
            "status": pipeline_summary.get("pipeline_status", "FAILED"),
            "feature_store_type": pipeline_summary.get(
                "feature_store_type", "Unknown"
            ),
            "completion_time": datetime.now().isoformat(),
            "records_processed": pipeline_summary.get(
                "data_shape", {}
            ).get("final", [0])[0],
            "features_selected": pipeline_summary.get(
                "selected_features_count", 0
            ),
            "quality_score": pipeline_summary.get("quality_score", 0),
            "validation_warnings": pipeline_summary.get(
                "validation_warnings", []
            ),
        }

        Variable.set(
            "data_pipeline_state",
            json.dumps(final_state, default=str),
            overwrite=True
        )

        logger.info(f"Data pipeline completed successfully")
        return final_state

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")

        # Store failure state
        failure_state = {
            "status": "FAILED",
            "completion_time": datetime.now().isoformat(),
            "error": str(e),
        }

        Variable.set(
            "data_pipeline_state",
            json.dumps(failure_state, default=str),
            overwrite=True
        )
        raise


if __name__ == "__main__":
    data_pipeline()
