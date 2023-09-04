import asyncio
import json
import os
import subprocess
from typing import List

from google.cloud import run_v2

from cloud_manager.targets.abc import CloudTargetBase
from cloud_manager.models import ServiceStatus


GCP_REGION = os.getenv("GCP_REGION")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def get_services_client():
    """
    Get a Google Cloud Run client for services.
    """
    return run_v2.ServicesAsyncClient()


def determine_service_status(data: str) -> ServiceStatus:
    """
    Determine the overall status of a Google Cloud Run service from the given string.

    "{'lastTransitionTime': '2023-09-03T12:29:51.937597Z', 'status': 'True', 'type': 'Ready'};{'lastTransitionTime': '2023-09-03T12:29:36.968239Z', 'status': 'True', 'type': 'ConfigurationsReady'};{'lastTransitionTime': '2023-09-03T12:29:52.019117Z', 'status': 'True', 'type': 'RoutesReady'}"
    """
    # Split the data string into individual JSON strings.
    json_strings = data.split(";")
    # Convert each JSON string into a dictionary.
    conditions = [json.loads(j.replace("'", '"')) for j in json_strings]

    # Get the status of the "Ready" type.
    ready_status = next(
        (
            condition["status"]
            for condition in conditions
            if condition["type"] == "Ready"
        ),
        None,
    )

    # Determine the overall status.
    if ready_status == "True":
        return ServiceStatus.RUNNING
    elif ready_status == "False":
        return ServiceStatus.STOPPED
    else:
        return ServiceStatus.FAILED


async def run_command(command: List[str]):
    """
    Run a command in a subprocess.
    """
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return stdout.decode(), stderr.decode()


# TODO: Run this only once not in a module-level.
def run_command_sync(command: List[str]):
    """
    Run a command in a subprocess (sync version).
    """
    process = subprocess.run(command, capture_output=True)
    return process.stdout.decode(), process.stderr.decode()


# Authenticate with Google Cloud only once in a module level.
run_command_sync(
    [
        "gcloud",
        "auth",
        "activate-service-account",
        "--key-file",
        GOOGLE_APPLICATION_CREDENTIALS,
    ]
)
print("Google Cloud authentication activated")


class CloudRun(CloudTargetBase):
    """
    Google Cloud Run API wrapper for a specific project and region.

    NOTE: It is too hard to integrate Google Cloud Python SDK due to lack of
    documentation, so for now, we will use the CLI instead.
    """

    def __init__(self, user_id: str, project_id: str):
        self.gcp_region = GCP_REGION
        self.gcp_project_id = GCP_PROJECT_ID

        if not self.gcp_region or not self.gcp_project_id:
            raise ValueError("GCP_REGION and GCP_PROJECT_ID must be set")

        self.parent = f"projects/{self.gcp_project_id}/locations/{self.gcp_region}"

        super().__init__(user_id, project_id)

    async def start_service(self, deploy_yaml) -> dict[str, str]:
        """
        Create a service to deploy a container to Google Cloud Run.
        """
        # Deploy service.
        print(f"Deploying service {deploy_yaml.deploy.service_name}...")
        stdout, stderr = await run_command(
            [
                "gcloud",
                "run",
                "deploy",
                "--region",
                self.gcp_region,
                deploy_yaml.deploy.service_name,
                "--project",
                self.gcp_project_id,
                "--image",
                deploy_yaml.build.image_uri,
                "--allow-unauthenticated",
                "--quiet",
            ]
        )
        print(f"{stdout=}")
        print(f"{stderr=}")

        # # Get the service URL (why is it delivered via stderr?).
        # match = re.search(r"Service URL: (https://\S+)", stderr)
        # if match:
        #     service_url = match.group(1)
        # service_url = match.group(1) if match else None

        # Allow the service to be accessible from public.
        print(f"Allowing public access to service {deploy_yaml.deploy.service_name}...")
        stdout, stderr = await run_command(
            [
                "gcloud",
                "run",
                "services",
                "add-iam-policy-binding",
                deploy_yaml.deploy.service_name,
                "--region",
                self.gcp_region,
                "--project",
                self.gcp_project_id,
                "--member",
                "allUsers",
                "--role",
                "roles/run.invoker",
            ]
        )
        print(f"{stdout=}")
        print(f"{stderr=}")

    async def stop_service(self, service_name: str):
        """
        Delete a service from Google Cloud Run.
        """
        stdout, stderr = await run_command(
            [
                "gcloud",
                "run",
                "services",
                "delete",
                service_name,
                "--region",
                self.gcp_region,
                "--project",
                self.gcp_project_id,
                "--quiet",
            ]
        )
        print(f"{stdout=}")
        print(f"{stderr=}")

    async def get_service_status(self, service_name: str):
        """
        Get a Google Cloud Run service details.
        """
        print(f"Getting status of service {service_name}...")
        stdout, stderr = await run_command(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                service_name,
                "--region",
                self.gcp_region,
                "--project",
                self.gcp_project_id,
                "--quiet",
                "--format",
                "get(status.conditions)",
            ]
        )
        print(f"{stdout=}")
        print(f"{stderr=}")

        return {
            "status": determine_service_status(stdout),
        }
