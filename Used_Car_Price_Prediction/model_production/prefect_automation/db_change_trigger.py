"""Triggers the complete ML pipeline depending on database changes."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_markdown_artifact
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.deployments import run_deployment
from prefect.runtime import flow_run

from src.data_ingestion.mongodb_connector import MongoDBConnector
from config.config import get_settings


@task(
    cache_policy=TASK_SOURCE + INPUTS,
    cache_expiration=timedelta(minutes=30))
def detect_database_changes(
    force_trigger: bool = False
) -> dict[str, Any]:
    """Check various conditions to determine if pipeline should be triggered.

    If force_trigger = True, always indicate pipeline should run.
    """
    logger = get_run_logger()
    settings = get_settings()
    result = {
        "should_trigger": False,
        "reason": None
    }

    if force_trigger:
        logger.info("Force trigger enabled - skipping DB check")
        result.update(
            should_trigger=True,
            reason="Force triggered by user"
        )
        return result

    try:
        # Initialize MongoDB connector
        connector = MongoDBConnector(
            connection_string=settings.MONGODB_URL,
            database_name=settings.MONGODB_NAME,
            collection_name=settings.MONGODB_COLLECTION
        )

        connector.connect()

        # Test MongoDB connection
        if not connector.test_connection():
            logger.error("MongoDB connection failed: {e}")
            result["reason"] = "MongoDB connection failed"
            result["check_timestamp"] = datetime.now().isoformat()
            result["mongodb_connected"] = False
            return result

        # Get collection statistics
        stats = connector.get_collection_info()
        document_count = stats.get("document_count", 0)

        logger.info(f"Current document count: {document_count}")

        if document_count == 0:
            logger.info("No documents found in MongoDB collection")
            connector.close()
            result["reason"] = "No documents found in MongoDB collection"
            result["mongodb_connected"] = True
            result["check_timestamp"] = datetime.now().isoformat()
            return result

        # Create content hash for change detection
        try:
            sample_data = connector.fetch_data(limit=500)

            if sample_data.empty:
                result["reason"] = "Fetched data is empty"
                content_hash = "empty"
                stats_hash = "empty"
                return result

            # Hash first 100, last 100, and 100 random rows
            first_100 = sample_data.head(
                100).sort_index(axis=1).to_string()
            last_100 = sample_data.tail(
                100).sort_index(axis=1).to_string()

            # Random sample from middle
            if len(sample_data) > 200:
                middle_sample = sample_data.iloc[100:-100].sample(
                    min(100, len(sample_data) - 200)
                )
                middle_100 = middle_sample.sort_index(
                    axis=1).to_string()
            else:
                middle_100 = ""

            # Combine all samples for comprehensive hash
            combined_content = first_100 + middle_100 + last_100
            content_hash = hashlib.md5(
                combined_content.encode()).hexdigest()

            # Statistical fingerprint
            numeric_cols = sample_data.select_dtypes(
                include=["number"]).columns
            stats_signature = {}
            for col in numeric_cols:
                if not sample_data[col].empty:
                    stats_signature[col] = {
                        "mean": float(sample_data[col].mean()),
                        "std": float(sample_data[col].std() or 0),
                        "min": float(sample_data[col].min()),
                        "max": float(sample_data[col].max()),
                    }
            stats_hash = hashlib.md5(
                json.dumps(stats_signature, sort_keys=True).encode()
            ).hexdigest()

        except Exception as e:
            logger.warning(f"Could not create content hash: {e}")
            content_hash = f"error_{datetime.now().timestamp()}"
            stats_hash = f"error_{datetime.now().timestamp()}"

        # DB change metadata file
        prefect_root = Path(__file__).resolve().parents[1]
        db_change_metadata_file = prefect_root / \
            "prefect_automation" / "db_change_metadata.json"
        current_time = datetime.now()

        if db_change_metadata_file.exists():
            with db_change_metadata_file.open("r") as f:
                previous_metadata = json.load(f)

            prev_count = previous_metadata.get("document_count", 0)
            prev_hash = previous_metadata.get("content_hash", "")
            prev_stats_hash = previous_metadata.get("stats_hash", "")
            last_trigger_time = None

            if "last_trigger_time" in previous_metadata:
                last_trigger_time = datetime.fromisoformat(
                    previous_metadata["last_trigger_time"]
                )

            # Calculate time differences
            time_since_trigger = None
            if last_trigger_time:
                time_since_trigger = (
                    current_time - last_trigger_time
                ).total_seconds() / 3600

            # Enhanced change detection
            count_changed = document_count != prev_count
            content_changed = content_hash != prev_hash
            stats_changed = stats_hash != prev_stats_hash

            # Trigger conditions
            has_data_changes = count_changed or content_changed or stats_changed
            # Time passed > 6 hours
            enough_time_passed = (
                time_since_trigger is None or time_since_trigger >= 6
            )

            # Detailed change reasons
            change_reasons = []
            if count_changed:
                change_reasons.append(
                    f"count: {prev_count}→{document_count}")
            if content_changed:
                change_reasons.append("content modified")
            if stats_changed:
                change_reasons.append("statistics changed")

            # Check for trigger conditions
            if not has_data_changes and enough_time_passed:
                result["should_trigger"] = False
                result["reason"] = "No data changes detected"
            elif has_data_changes and not enough_time_passed:
                result["should_trigger"] = False
                result["reason"] = (
                    f"Changes detected ({', '.join(change_reasons)}) "
                    f"but triggered recently"
                    f"({time_since_trigger:.1f}h ago)"
                )
            elif has_data_changes and enough_time_passed:
                result["should_trigger"] = True
                result["reason"] = (
                    f"Data changes detected:{', '.join(change_reasons)}"
                )
            else:
                result["should_trigger"] = False
                result["reason"] = "No changes and recent trigger"
                f"({time_since_trigger:.1f}h ago)"
        else:
            logger.warning(
                "Could not read previous db change metadata")
            result["reason"] = (
                "Could not verify previous state - triggering pipeline"
            )

        # Current database change metadata
        current_metadata = {
            "document_count": document_count,
            "content_hash": content_hash,
            "stats_hash": stats_hash,
            "last_check_time": current_time.isoformat(),
            "collection_stats": stats,
        }

        # Add trigger time if triggering
        if result["should_trigger"]:
            current_metadata[
                "last_trigger_time"] = current_time.isoformat()

        # Update database change metadata file
        with db_change_metadata_file.open("w") as f:
            json.dump(current_metadata, f, indent=2)

        # Close MongoDB connection
        connector.close()

        result["document_count"] = document_count
        # First 10 chars for logging
        result["content_hash"] = content_hash[:10]
        result["stats"] = stats

        logger.info("Pipeline trigger conditions:")
        return result

    except Exception as e:
        logger.error(f"Failed to check trigger conditions: {e}")
        result["should_trigger"] = False
        result["reason"] = f"Error checking conditions: {str(e)}"
        return result


@task
def send_db_change_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Send a DB change summary."""
    logger = get_run_logger()

    summary = {
        "db_change_trigger_run": {
            "flow_run_id": str(flow_run.get_id()),
            "flow_run_name": str(flow_run.get_name()),
            "check_timestamp": result.get("check_timestamp"),
        },
        "mongodb_status": {
            "connected": result.get("mongodb_connected", False),
            "document_count": result.get("document_count", 0),
            "content_hash": result.get("content_hash", "unknown"),
        },
        "trigger_evaluation": {
            "should_trigger": result.get("should_trigger", False),
            "reason": result.get("reason", "unknown")
        },
        "summary_timestamp": datetime.now().isoformat(),
    }

    # Create Prefect artifacts
    create_markdown_artifact(
        key="controller-summary",
        markdown=f"""
# ML Pipeline Controller Summary

**Timestamp:** {summary['summary_timestamp']}
**Flow Run:** {summary['db_change_trigger_run']['flow_run_name']}

## MongoDB Status
- **Connected:** {summary['mongodb_status']['connected']}
- **Document Count:** {summary['mongodb_status']['document_count']:,}
- **Content Hash:** {summary['mongodb_status']['content_hash']}

## Trigger Decision
- **Should Trigger:** {summary['trigger_evaluation']['should_trigger']}
- **Reason:** {summary['trigger_evaluation']['reason']}
""",
        description="DB change trigger execution summary"
    )

    logger.info(f"DB Change summary: {summary}")
    return summary


@flow(
    name="db-change-trigger-pipeline",
    description="Triggers data pipeline if DB changes are detected",
    flow_run_name=f"db-change-run-{
        datetime.now().strftime('%Y%m%d-%H%M%S')}",
)
def db_change_trigger_pipeline(force_trigger: bool = False) -> dict[str, Any]:
    """DB change detection for triggering data pipeline."""
    logger = get_run_logger()

    # Main pipeline logic
    detect_db_changes = detect_database_changes(
        force_trigger=force_trigger
    )

    summary_result = send_db_change_summary(
        result=detect_db_changes
    )

    if not summary_result["trigger_evaluation"]["should_trigger"]:
        logger.info(f"Skipping data pipeline: {
            summary_result['trigger_evaluation']['reason']}"
        )
        return {"triggered": False, "reason": summary_result[
            'trigger_evaluation']['reason']}

    try:
        logger.info(
            "✅ Database change detected, triggering data pipeline deployment..."
        )
        data_pipeline_run = run_deployment(
            name="data-pipeline/data-pipeline",
            parameters={
                "triggered_by": "db_change_trigger_pipeline",
                 "trigger_timestamp": datetime.now().isoformat(),
                "trigger_reason": summary_result[
                    "trigger_evaluation"
                 ]["reason"]
            }
        )

        logger.info(f"Data pipeline triggered: {data_pipeline_run}")
        return {"triggered": True, "deployment_run": str(data_pipeline_run)}

    except Exception as e:
        logger.error(f"Failed to trigger data pipeline: {e}")
        return {"triggered": False, "error": str(e)}


if __name__ == "__main__":
    db_change_trigger_pipeline(force_trigger=False)
