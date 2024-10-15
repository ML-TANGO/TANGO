import os
import time
import logging
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError

from cloud_manager.targets.abc import CloudTargetBase
from cloud_manager.models import ServiceStatus

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AWSConfig:
    REGION = os.getenv("AWS_REGION")
    ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    @classmethod
    def validate(cls):
        if not all([cls.REGION, cls.ACCESS_KEY_ID, cls.SECRET_ACCESS_KEY]):
            logger.error("AWS credentials not set")
            raise ValueError("AWS credentials not set")


class AWSClients:
    @staticmethod
    def get_client(service_name: str):
        AWSConfig.validate()
        logger.debug(f"Initializing AWS {service_name} client")
        return boto3.client(
            service_name,
            region_name=AWSConfig.REGION,
            aws_access_key_id=AWSConfig.ACCESS_KEY_ID,
            aws_secret_access_key=AWSConfig.SECRET_ACCESS_KEY,
        )


class AWSECS(CloudTargetBase):
    """
    AWS ECR and ECS API wrapper for a specific project and region.
    """

    def __init__(self, user_id: str, project_id: str):
        logger.info(f"Initializing AWSECS for user {user_id} and project {project_id}")
        AWSConfig.validate()
        super().__init__(user_id, project_id)

        self.ecr_client = AWSClients.get_client("ecr")
        self.ecs_client = AWSClients.get_client("ecs")
        self.ec2_client = AWSClients.get_client("ec2")
        self.elb_client = AWSClients.get_client("elbv2")

        super().__init__(user_id, project_id)

    async def start_service(self, deploy_yaml) -> Dict[str, str]:
        logger.info(f"Starting service for {deploy_yaml.deploy.service_name}")
        try:
            self._register_task_definition(deploy_yaml)
            self._ensure_cluster_exists(deploy_yaml.deploy.service_name)
            return self._create_ecs_service(deploy_yaml)
        except ClientError as e:
            logger.error(f"Error starting service: {str(e)}")
            return {"error": str(e)}

    async def stop_service(self, service_name: str) -> Dict[str, str]:
        logger.info(f"Stopping service {service_name}")
        try:
            self._stop_and_delete_service(service_name)
            self._deregister_task_definitions(service_name)
            self._delete_cluster(service_name)
            logger.info(f"Service {service_name} stopped and resources cleaned up")
            return {
                "message": f"ECS Service {service_name} and all task definitions deleted"
            }
        except ClientError as e:
            logger.error(f"Error stopping service: {str(e)}")
            return {"error": str(e)}

    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        logger.info(f"Getting status for service {service_name}")
        try:
            response = self.ecs_client.describe_services(
                cluster=service_name, services=[service_name]
            )
            if not response["services"]:
                logger.warning(f"No services found for {service_name}")
                return {"status": ServiceStatus.FAILED}

            service = response["services"][0]
            for deploy in service["deployments"]:
                if deploy["rolloutState"] == "FAILED":
                    logger.error(f"Deployment failed for service {service_name}")
                    return {"status": ServiceStatus.FAILED}
                elif deploy["rolloutState"] == "COMPLETED":
                    logger.info(f"Deployment completed for service {service_name}")
                    return self._get_running_service_details(service, service_name)
                else:
                    logger.info(f"Deployment in progress for service {service_name}")
                    return {"status": ServiceStatus.PREPARING}
        except ClientError as e:
            logger.error(f"Error getting service status: {str(e)}")
            return {"error": str(e)}

    def _register_task_definition(self, deploy_yaml) -> None:
        logger.info(
            f"Registering task definition for {deploy_yaml.deploy.service_name}"
        )
        task_definition = self._create_task_definition(deploy_yaml)
        self.ecs_client.register_task_definition(**task_definition)
        logger.info("Task definition registered successfully")

    def _create_task_definition(self, deploy_yaml) -> Dict[str, Any]:
        logger.info("Creating task definition")
        return {
            "family": deploy_yaml.deploy.service_name,
            "containerDefinitions": [
                {
                    "name": deploy_yaml.deploy.service_name,
                    "image": deploy_yaml.build.image_uri,
                    "cpu": deploy_yaml.deploy.resources.cpu * 1024,
                    "memory": deploy_yaml.deploy.resources.memory,
                    # "resourceRequirements": [
                    #     {"type": "GPU", "value": deploy_yaml.deploy.resources.gpu}
                    # ],
                    # command: ["sleep", "360"],
                    "essential": True,
                    "portMappings": [
                        {
                            "containerPort": deploy_yaml.deploy.network.service_container_port,
                            "hostPort": deploy_yaml.deploy.network.service_host_port,
                            "protocol": "tcp",
                        }
                    ],
                }
            ],
            "requiresCompatibilities": ["FARGATE"],
            "networkMode": "awsvpc",  # awsvpc, bridge
            "cpu": str(deploy_yaml.deploy.resources.cpu * 1024),
            "memory": str(deploy_yaml.deploy.resources.memory),
        }

    def _ensure_cluster_exists(self, cluster_name: str) -> None:
        logger.info(f"Ensuring cluster {cluster_name} exists")
        response = self.ecs_client.describe_clusters(clusters=[cluster_name])
        if not response["clusters"] or response["clusters"][0]["status"] == "INACTIVE":
            logger.info(f"Creating cluster {cluster_name}")
            self.ecs_client.create_cluster(clusterName=cluster_name)
        else:
            logger.info(f"Cluster {cluster_name} already exists")

    def _create_ecs_service(self, deploy_yaml) -> Dict[str, str]:
        logger.info(f"Creating ECS service for {deploy_yaml.deploy.service_name}")
        self.ecs_client.create_service(
            cluster=deploy_yaml.deploy.service_name,
            serviceName=deploy_yaml.deploy.service_name,
            taskDefinition=deploy_yaml.deploy.service_name,
            desiredCount=1,
            # TODO: Harded-coded network configs for FARGATE type
            networkConfiguration={
                "awsvpcConfiguration": {
                    "assignPublicIp": "ENABLED",
                    "subnets": ["subnet-0e1c240913e7c480c"],
                    "securityGroups": ["sg-0882f75dd97251038"],
                }
            },
            launchType="FARGATE",
        )
        logger.info(
            f"ECS service {deploy_yaml.deploy.service_name} created successfully"
        )
        return {
            "message": f"Service {deploy_yaml.deploy.service_name} started successfully"
        }

    def _stop_and_delete_service(self, service_name: str) -> None:
        logger.info(f"Stopping and deleting service {service_name}")
        self.ecs_client.update_service(
            cluster=service_name, service=service_name, desiredCount=0
        )
        waiter = self.ecs_client.get_waiter("services_stable")
        waiter.wait(cluster=service_name, services=[service_name])

        self.ecs_client.delete_service(
            cluster=service_name, service=service_name, force=True
        )

        while True:
            try:
                response = self.ecs_client.describe_services(
                    cluster=service_name, services=[service_name]
                )
                if (
                    not response["services"]
                    or response["services"][0]["status"] == "INACTIVE"
                ):
                    break
            except self.ecs_client.exceptions.ServiceNotFoundException:
                break
            time.sleep(10)

        self._stop_running_tasks(service_name)
        logger.info(f"Service {service_name} stopped and deleted")

    def _stop_running_tasks(self, cluster_name: str) -> None:
        logger.info(f"Stopping running tasks in cluster {cluster_name}")
        running_tasks = self.ecs_client.list_tasks(cluster=cluster_name)
        for task_arn in running_tasks["taskArns"]:
            logger.info(f"Stopping task {task_arn}")
            self.ecs_client.stop_task(cluster=cluster_name, task=task_arn)
        logger.info("All running tasks stopped")

    def _deregister_task_definitions(self, service_name: str) -> None:
        logger.info(f"Deregistering task definitions for {service_name}")
        response = self.ecs_client.list_task_definitions(
            familyPrefix=service_name, status="ACTIVE"
        )
        for task_definition_arn in response["taskDefinitionArns"]:
            logger.info(f"Deregistering task definition {task_definition_arn}")
            self.ecs_client.deregister_task_definition(
                taskDefinition=task_definition_arn
            )
        logger.info("All task definitions deregistered")

    def _delete_cluster(self, cluster_name: str) -> None:
        logger.info(f"Deleting cluster {cluster_name}")
        self.ecs_client.delete_cluster(cluster=cluster_name)
        logger.info(f"Cluster {cluster_name} deleted")

    def _get_running_service_details(
        self, service: Dict[str, Any], service_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Getting running service details for {service_name}")
        load_balancers = service.get("loadBalancers", [])
        if load_balancers:
            logger.info("Service has load balancers, getting DNS name")
            return self._get_load_balancer_dns(load_balancers[0]["loadBalancerArn"])
        else:
            logger.info(
                "Service doesn't have load balancers, getting task public IP and port"
            )
            return self._get_task_public_ip_and_port(service, service_name)

    def _get_load_balancer_dns(self, lb_arn: str) -> Dict[str, Any]:
        logger.info(f"Getting DNS name for load balancer {lb_arn}")
        lb_response = self.elb_client.describe_load_balancers(LoadBalancerArns=[lb_arn])
        dns_name = lb_response["LoadBalancers"][0]["DNSName"]
        logger.info(f"Load balancer DNS name: {dns_name}")
        return {"status": ServiceStatus.RUNNING}

    def _get_task_public_ip_and_port(
        self, service: Dict[str, Any], service_name: str
    ) -> Dict[str, Any]:
        logger.info(f"Getting public IP and port for task in service {service_name}")
        network_config = service.get("networkConfiguration", {}).get(
            "awsvpcConfiguration", {}
        )
        if network_config.get("assignPublicIp") == "ENABLED":
            tasks = self.ecs_client.list_tasks(
                cluster=service_name, serviceName=service_name
            )
            if tasks["taskArns"]:
                task_arn = tasks["taskArns"][0]
                task_details = self.ecs_client.describe_tasks(
                    cluster=service_name, tasks=[task_arn]
                )
                task = task_details["tasks"][0]

                # Get the task definition to retrieve the container port
                task_def_arn = task["taskDefinitionArn"]
                task_def = self.ecs_client.describe_task_definition(
                    taskDefinition=task_def_arn
                )["taskDefinition"]

                # Get the first container definition (assuming there's only one)
                container_def = task_def["containerDefinitions"][0]
                host_port = container_def["portMappings"][0]["hostPort"]

                # Get the public IP
                eni_id = task["attachments"][0]["details"][1]["value"]
                eni_details = self.ec2_client.describe_network_interfaces(
                    NetworkInterfaceIds=[eni_id]
                )
                public_ip = (
                    eni_details["NetworkInterfaces"][0]
                    .get("Association", {})
                    .get("PublicIp")
                )

                if public_ip:
                    endpoint = f"{public_ip}:{host_port}"
                    logger.info(f"Public IP and port for task: {endpoint}")
                    return {"status": ServiceStatus.RUNNING}
                else:
                    logger.warning("No public IP found for task")
            else:
                logger.warning("No tasks found for service")
        else:
            logger.info("Public IP assignment is not enabled for this service")
        return {"status": ServiceStatus.RUNNING}
