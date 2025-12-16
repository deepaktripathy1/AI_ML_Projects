"""One-command controller for ML pipelines using Prefect.

Usage:
    
    # Start server
    prefect server start

    # Start local worker
    prefect worker start --pool default-agent-pool

    # Deploy all flows
    python pipeline_controller.py --deploy

    # Create automations
    python pipeline_controller.py --automate

    # Just trigger data pipeline run
    python pipeline_controller.py --run

    python pipeline_controller.py --delete
    # Delete all deployments

    mlflow server run
    mlflow server --host 127.0.0.1 --port 8080
"""

import asyncio
import warnings
from datetime import timedelta

from prefect.automations import Automation
from prefect.client.orchestration import get_client
from prefect.events.schemas.events import ResourceSpecification
from prefect.events.actions import RunDeployment
from prefect.events.schemas.automations import EventTrigger, Posture

from data_pipeline import data_pipeline
from training_pipeline import training_pipeline
from deployment_pipeline import deployment_pipeline
from monitoring_pipeline import monitoring_pipeline
from db_change_trigger import db_change_trigger_pipeline

warnings.filterwarnings(
    "ignore",
    message="numpy.core is deprecated and has been renamed to numpy._core.",
    category=DeprecationWarning
)
warnings.filterwarnings(
    "ignore",
    message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated",
    category=DeprecationWarning
)

# Deployment Configuration
# List of pipeline details for deployment

PIPELINES = [
    {
        "flow": data_pipeline,
        "deployment_name": "data-pipeline",
        "source_kwargs": {
            "source": "./prefect_automation/",
            "entrypoint": "data_pipeline.py:data_pipeline"
        },
        "deploy_kwargs": {
            "name": "data-pipeline",
            "work_pool_name": "default-agent-pool",
            "tags": ["ml", "data"],
            "description": "Used Car Data Pipeline Deployment"
        },
    },
    {
        "flow": training_pipeline,
        "deployment_name": "training-pipeline",
        "source_kwargs": {
            "source": "./prefect_automation/",
            "entrypoint": "training_pipeline.py:training_pipeline"
        },
        "deploy_kwargs": {
            "name": "training-pipeline",
            "work_pool_name": "default-agent-pool",
            "tags": ["ml", "training"],
            "description": "Model Training Pipeline Deployment"
        },
    },
    {
        "flow": deployment_pipeline,
        "deployment_name": "deployment-pipeline",
        "source_kwargs": {
            "source": "./prefect_automation/",
            "entrypoint": "deployment_pipeline.py:deployment_pipeline"
        },
        "deploy_kwargs": {
            "name": "deployment-pipeline",
            "work_pool_name": "default-agent-pool",
            "tags": ["ml", "deployment"],
            "description": "Model Deployment Pipeline Deployment"
        },
    },
    {
        "flow": monitoring_pipeline,
        "deployment_name": "monitoring-pipeline",
        "source_kwargs": {
            "source": "./prefect_automation/",
            "entrypoint": "monitoring_pipeline.py:monitoring_pipeline"
        },
        "deploy_kwargs": {
            "name": "monitoring-pipeline",
            "work_pool_name": "default-agent-pool",
            "tags": ["ml", "monitoring"],
            "description": "Model Monitoring Pipeline Deployment"
        },
    }
]


def deploy_flows():
    # Deploy flows
    for pipeline in PIPELINES:
        print(
            f"Deploying {pipeline['deployment_name']}..."
        )
        pipeline["flow"].from_source(
            **pipeline["source_kwargs"]
        ).deploy(**pipeline["deploy_kwargs"])

    # Deploy db_change_trigger flow
    print("Deploying db_change_trigger (interval = 60 days)...")
    db_change_trigger_pipeline.from_source(
        source="./prefect_automation/",
        entrypoint="db_change_trigger.py:db_change_trigger_pipeline"
    ).deploy(  # type: ignore
        name="db-change-trigger-pipeline",
        work_pool_name="default-agent-pool",
        interval=60 * 24 * 60 * 60,
        tags=["db-monitoring", "auto-trigger"],
        parameters={"force_trigger": False}
    )


async def create_automations():
    # Fetch flow IDs for automations
    async with get_client() as client:
        flows = await client.read_flows()
        print("\nCurrently registered flows:")
        for flow in flows:
            print(f" - {flow.name}")

        flow_map = {flow.name: flow.id for flow in flows}

        # For deployment Id in Actions
        deployments = await client.read_deployments()
        deploy_map = {d.name: d.id for d in deployments}

        # Automations
        automation_training = Automation(
            name="trigger-training-after-data",
            description="Run training after data pipeline",
            enabled=True,
            tags=["ml", "training"],
            trigger=EventTrigger(
                type="event",
                match=ResourceSpecification.model_validate(
                    {"prefect.resource.id": "prefect.flow-run.*"}
                ),
                match_related=[
                    ResourceSpecification.model_validate(
                        {"prefect.resource.role": "flow"}
                    ),
                    ResourceSpecification.model_validate(
                        {"prefect.resource.id": [
                            f"prefect.flow.{
                                flow_map['data-pipeline']
                            }"
                        ]}
                    ),
                ],
                expect={"prefect.flow-run.Completed"},
                for_each={"prefect.resource.id"},
                posture=Posture.Reactive,
                threshold=1,
                within=timedelta(seconds=0)
            ),
            actions=[RunDeployment(
                type="run-deployment",
                source="selected",
                deployment_id=deploy_map["training-pipeline"],
                parameters={},
                job_variables={}
            )]
        )
        created_training = await automation_training.acreate()
        print(
            f"Training Automation: {
                created_training.name} 'ID': {
                    created_training.id}"
        )

        automation_deployment = Automation(
            name="trigger-deployment-after-training",
            description="Run deployment after training pipeline",
            enabled=True,
            tags=["ml", "deployment"],
            trigger=EventTrigger(
                type="event",
                match=ResourceSpecification.model_validate(
                    {"prefect.resource.id": "prefect.flow-run.*"}
                ),
                match_related=[
                    ResourceSpecification.model_validate(
                        {"prefect.resource.role": "flow"}
                    ),
                    ResourceSpecification.model_validate(
                        {"prefect.resource.id": [
                            f"prefect.flow.{
                                flow_map['training-pipeline']
                            }"
                        ]}
                    ),
                ],
                expect={"prefect.flow-run.Completed"},
                posture=Posture.Reactive,
                for_each={"prefect.resource.id"},
                threshold=1,
                within=timedelta(seconds=0)
            ),
            actions=[RunDeployment(
                type="run-deployment",
                source="selected",
                deployment_id=deploy_map["deployment-pipeline"],
                parameters={},
                job_variables={}
            )]
        )
        created_deployment = await automation_deployment.acreate()
        print(
            f"Deployment Automation created: {
                created_deployment.name} 'ID': {
                    created_deployment.id}"
        )

        automation_monitoring = Automation(
            name="trigger-monitoring-after-deployment",
            description="Run monitoring after deployment pipeline",
            enabled=True,
            tags=["ml", "monitoring"],
            trigger=EventTrigger(
                type="event",
                match=ResourceSpecification.model_validate(
                    {"prefect.resource.id": "prefect.flow-run.*"}
                ),
                match_related=[
                    ResourceSpecification.model_validate(
                        {"prefect.resource.role": "flow"}
                    ),
                    ResourceSpecification.model_validate(
                        {"prefect.resource.id": [
                            f"prefect.flow.{
                                flow_map['deployment-pipeline']
                            }"
                        ]}
                    ),
                ],
                expect={"prefect.flow-run.Completed"},
                posture=Posture.Reactive,
                for_each={"prefect.resource.id"},
                threshold=1,
                within=timedelta(seconds=0)
            ),
            actions=[RunDeployment(
                type="run-deployment",
                source="selected",
                deployment_id=deploy_map["monitoring-pipeline"],
                parameters={},
                job_variables={}
            )]
        )
        created_automation = await automation_monitoring.acreate()
        print(
            f"Monitoring Automation created: {
                created_automation.name} 'ID': {
                    created_automation.id}"
        )


# Trigger first data_pipeline run


async def run_data_pipeline():
    print("\nTriggering first run of the data pipeline...")
    async with get_client() as client:
        deployments = await client.read_deployments()
        deployment_map = {d.name: d.id for d in deployments}
        data_deploy_id = deployment_map["data-pipeline"]

        await client.create_flow_run_from_deployment(
            deployment_id=data_deploy_id
        )


async def delete_flows_and_automations():
    async with get_client() as client:
        # Delete automations
        automations = await client.read_automations()
        for a in automations:
            if any(tag in a.name.lower() for tag in [
                    "data", "training", "deployment", "monitoring"]):
                print(f"🗑️ Deleting automation: {a.name}")
                await client.delete_automation(a.id)

        # Delete deployments
        deployments = await client.read_deployments()
        for d in deployments:
            if any(name in d.name for name in [
                    "data-pipeline", "training-pipeline", "deployment-pipeline", "monitoring-pipeline", "db-change-trigger-pipeline"]):
                print(f"🗑️ Deleting deployment: {d.name}")
                await client.delete_deployment(d.id)

        # Delete flows
        flows = await client.read_flows()
        for f in flows:
            if any(name in f.name for name in [
                    "data-pipeline", "training-pipeline", "deployment-pipeline", "monitoring-pipeline", "db-change-trigger-pipeline"
            ]):
                print(f"🗑️ Deleting flow: {f.name}")
                await client.delete_flow(f.id)

        # Delete variables
        variables = await client.read_variables()
        for v in variables:
            print(f"🗑️ Deleting variable: {v.name}")
            await client.delete_variable_by_name(v.name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Controller")
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Deploy all flows"
    )
    parser.add_argument(
        "--automate",
        action="store_true",
        help="Create automations"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Trigger a run of data-pipeline"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all registered deployments"
    )

    args = parser.parse_args()

    if args.deploy:
        asyncio.run(asyncio.to_thread(deploy_flows))
        print("\n✅ All flows deployed")

    if args.automate:
        asyncio.run(create_automations())
        print("\nCreated automations successfully ")

    if args.run:
        asyncio.run(run_data_pipeline())
        print("\nData pipeline triggered successfully!")

    if args.delete:
        asyncio.run(delete_flows_and_automations())
        print("\n🧹 All flows and automations deleted successfully!")
