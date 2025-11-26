"""
Backend.AI Compute Session deployment target.

This module implements the CloudTargetBase interface for deploying AI/ML inference
services to Backend.AI compute sessions with support for heterogeneous accelerators.
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import httpx

from cloud_manager.targets.abc import CloudTargetBase
from cloud_manager.targets.accelerators import AcceleratorSpec
from cloud_manager.models import ServiceStatus

logger = logging.getLogger(__name__)


# Environment variables for Backend.AI configuration
CS_ENDPOINT = os.getenv("CS_ENDPOINT", "https://api.backend.ai")
CS_ACCESS_KEY = os.getenv("CS_ACCESS_KEY")
CS_SECRET_KEY = os.getenv("CS_SECRET_KEY")
CS_DEFAULT_DOMAIN = os.getenv("CS_DEFAULT_DOMAIN", "default")
CS_DEFAULT_GROUP = os.getenv("CS_DEFAULT_GROUP", "default")


def generate_signature(
    method: str,
    path: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    hostname: str,
    content_type: str = "application/json",
    api_version: str = "v8.20240915",
    hash_type: str = "sha256",
) -> Dict[str, str]:
    """
    Generate Backend.AI authentication signature headers.

    Args:
        method: HTTP method (GET, POST, DELETE, etc.)
        path: API endpoint path
        body: Request body as bytes
        access_key: Backend.AI access key
        secret_key: Backend.AI secret key
        hostname: API endpoint hostname
        content_type: Content-Type header value
        api_version: Backend.AI API version
        hash_type: Hash algorithm (default: sha256)

    Returns:
        Dictionary of authentication headers
    """
    from dateutil.tz import tzutc

    now = datetime.now(tzutc())
    date_str = now.strftime('%Y%m%d')

    # Calculate body hash (always empty bytes per Backend.AI spec)
    body_hash = hashlib.new(hash_type, b"").hexdigest()

    # Create canonical request (sign string)
    sign_str = (
        f"{method.upper()}\n{path}\n{now.isoformat()}\n"
        f"host:{hostname}\ncontent-type:{content_type.lower()}\n"
        f"x-backendai-version:{api_version}\n{body_hash}"
    )

    # Calculate signature with chained HMAC
    key = secret_key.encode()
    key = hmac.new(key, date_str.encode(), hash_type).digest()
    key = hmac.new(key, hostname.encode(), hash_type).digest()
    signature = hmac.new(key, sign_str.encode(), hash_type).hexdigest()

    # Return headers
    auth = (
        f"BackendAI signMethod=HMAC-{hash_type.upper()}, "
        f"credential={access_key}:{signature}"
    )

    return {
        "Authorization": auth,
        "Date": now.isoformat(),
        "Content-Type": content_type,
        "X-BackendAI-Version": api_version,
    }


def determine_service_status(status_str: str) -> ServiceStatus:
    """
    Convert Backend.AI service status to TANGO ServiceStatus enum.

    Args:
        status_str: Backend.AI status string

    Returns:
        ServiceStatus enum value
    """
    status_mapping = {
        "READY": ServiceStatus.STOPPED,
        "PROVISIONING": ServiceStatus.STARTING,
        "HEALTHY": ServiceStatus.RUNNING,
        "DEGRADED": ServiceStatus.FAILED,
        "DESTROYING": ServiceStatus.STOPPING,
        "DESTROYED": ServiceStatus.STOPPED,
    }
    return status_mapping.get(status_str, ServiceStatus.UNKNOWN)


class ComputeSession(CloudTargetBase):
    """
    Backend.AI Compute Session deployment target.

    This class provides integration with Backend.AI for deploying inference services
    with support for various accelerators (GPU, NPU, TPU, CPU).
    """

    def __init__(self, user_id: str, project_id: str):
        """
        Initialize ComputeSession target.

        Args:
            user_id: User identifier
            project_id: Project identifier
        """
        super().__init__(user_id, project_id)

        self.endpoint = CS_ENDPOINT
        self.access_key = CS_ACCESS_KEY
        self.secret_key = CS_SECRET_KEY
        self.domain = CS_DEFAULT_DOMAIN
        self.group = CS_DEFAULT_GROUP

        if not self.access_key or not self.secret_key:
            raise ValueError("CS_ACCESS_KEY and CS_SECRET_KEY must be set")

        self.client = httpx.AsyncClient(
            base_url=self.endpoint, timeout=httpx.Timeout(30.0)
        )

    async def _make_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        Make authenticated request to Backend.AI API.

        Args:
            method: HTTP method
            path: API endpoint path
            body: Request body dictionary
            params: Query parameters

        Returns:
            HTTP response
        """
        from yarl import URL

        body_bytes = json.dumps(body).encode() if body else b""

        # Add query parameters to path if present
        if params:
            path = f"{path}?{urlencode(params)}"

        # Extract hostname from endpoint
        endpoint_url = URL(self.endpoint)
        hostname = endpoint_url.authority

        headers = generate_signature(
            method=method,
            path=path,
            body=body_bytes,
            access_key=self.access_key,
            secret_key=self.secret_key,
            hostname=hostname,
        )

        logger.debug(f"Request URL: {self.client.base_url}{path}")
        logger.debug(f"Method: {method}")
        logger.debug(f"Body length: {len(body_bytes)} bytes")

        response = await self.client.request(
            method=method,
            url=path,
            headers=headers,
            content=body_bytes if body else None,
        )

        logger.debug(f"Response Status: {response.status_code}")
        logger.debug(f"Response Body: {response.text[:500]}")

        response.raise_for_status()
        return response

    async def start_service(self, deploy_yaml) -> Dict[str, str]:
        """
        Create and start a Backend.AI model service.

        Args:
            deploy_yaml: Deployment configuration

        Returns:
            Dictionary containing service information
        """
        # Extract basic configuration
        service_name = deploy_yaml.deploy.service_name
        image_uri = deploy_yaml.build.image_uri

        # Get accelerator specification
        accelerator_spec = AcceleratorSpec()
        if hasattr(deploy_yaml.build, "accelerator") and deploy_yaml.build.accelerator:
            if isinstance(deploy_yaml.build.accelerator, str):
                accelerator_spec = AcceleratorSpec.from_string(
                    deploy_yaml.build.accelerator
                )
            else:
                accelerator_spec = deploy_yaml.build.accelerator

        # Build resource specification (Backend.AI format)
        resources = {}

        # Add accelerator resources first (cuda.shares, atom.device, etc.)
        # These are added from build.accelerator if count/shares are specified
        accelerator_resources = accelerator_spec.to_backend_ai_resource()
        resources.update(accelerator_resources)

        if deploy_yaml.deploy.resources:
            # CPU (integer)
            if deploy_yaml.deploy.resources.cpu:
                resources["cpu"] = int(deploy_yaml.deploy.resources.cpu)

            # Memory (convert MB to g)
            if deploy_yaml.deploy.resources.memory:
                memory_mb = int(deploy_yaml.deploy.resources.memory)
                memory_gb = memory_mb / 1024
                resources["mem"] = f"{memory_gb:.1f}g"

            # Add any additional resource fields directly (e.g., atom-max.device)
            # These override accelerator resources if both are specified
            resource_dict = deploy_yaml.deploy.resources.dict()
            for key, value in resource_dict.items():
                if key not in ["cpu", "memory", "gpu"] and value is not None:
                    resources[key] = value

        # Build model service configuration
        model_service_config = deploy_yaml.deploy.model_service

        # Extract config values (handles both Pydantic model and None)
        if model_service_config:
            model_name = model_service_config.model or service_name
            runtime_variant = model_service_config.runtime_variant
            model_mount = model_service_config.model_mount_destination
            environ = model_service_config.environ or {}
            scaling_group = model_service_config.scaling_group
            model_definition_path = model_service_config.model_definition_path
            resource_opts = model_service_config.resource_opts
        else:
            # Use defaults if model_service is not specified
            model_name = service_name
            runtime_variant = "custom"
            model_mount = "/models"
            environ = {}
            scaling_group = "default"
            model_definition_path = None
            resource_opts = None

        model_config = {
            "model": model_name,
            "runtime_variant": runtime_variant,
            "model_mount_destination": model_mount,
            "environ": environ,
            "scaling_group": scaling_group,
            "resources": resources,
        }

        # Add optional fields if present
        if model_definition_path:
            model_config["model_definition_path"] = model_definition_path

        if resource_opts:
            model_config["resource_opts"] = resource_opts
        elif resources.get("mem"):
            # Default shmem to 1/4 of memory if not specified
            mem_value = float(resources["mem"].rstrip("g"))
            default_shmem = max(1, int(mem_value / 4))
            model_config["resource_opts"] = {"shmem": f"{default_shmem}g"}

        # Build request body
        request_body = {
            "name": service_name,
            "replicas": deploy_yaml.deploy.replicas,
            "image": image_uri,
            "group": self.group,
            "domain": self.domain,
            "cluster_size": deploy_yaml.deploy.cluster_size,
            "cluster_mode": deploy_yaml.deploy.cluster_mode,
            "open_to_public": deploy_yaml.deploy.open_to_public,
            "config": model_config,
        }

        logger.info(f"Creating Backend.AI service '{service_name}'...")
        logger.debug(f"Request body: {json.dumps(request_body, indent=2)}")

        # Create service
        response = await self._make_request(
            method="POST",
            path="/services",
            body=request_body,
        )

        result = response.json()
        logger.info(f"Service created: {json.dumps(result, indent=2)}")

        endpoint_id = result.get("endpoint_id")

        # Query endpoint info to get URL and status
        endpoint_info = await self.get_endpoint_info(endpoint_id)

        endpoint_url = None
        endpoint_status = "PROVISIONING"

        if endpoint_info:
            endpoint_url = endpoint_info.get("url")
            endpoint_status = endpoint_info.get("status", "PROVISIONING")
            logger.info(f"Endpoint URL: {endpoint_url or 'Not available yet (provisioning)'}")
            logger.info(f"Endpoint Status: {endpoint_status}")

        return {
            "service_id": endpoint_id,
            "service_name": service_name,
            "status": endpoint_status,
            "url": endpoint_url,
        }

    async def stop_service(self, service_name: str):
        """
        Stop and delete a Backend.AI model service.

        Args:
            service_name: Name of the service to stop
        """
        # Query to find endpoint by name
        query = """
        query($limit: Int!, $offset: Int!, $filter: String) {
            endpoint_list(limit: $limit, offset: $offset, filter: $filter) {
                items {
                    endpoint_id
                    name
                    status
                }
            }
        }
        """

        # Use filter to search by name
        filter_expr = f'name == "{service_name}"'

        request_body = {
            "query": query,
            "variables": {
                "limit": 100,
                "offset": 0,
                "filter": filter_expr,
            },
        }

        try:
            response = await self._make_request(
                method="POST",
                path="/admin/gql",
                body=request_body,
            )

            result = response.json()
            data = result.get("data", {})
            endpoint_list = data.get("endpoint_list", {})
            items = endpoint_list.get("items", [])

            # Find endpoint with matching name
            endpoint = None
            for item in items:
                if item.get("name") == service_name:
                    endpoint = item
                    break

            if not endpoint:
                logger.warning(f"Service '{service_name}' not found")
                return

            endpoint_id = endpoint.get("endpoint_id")

            logger.info(f"Deleting Backend.AI service '{service_name}' (ID: {endpoint_id})...")

            # Delete service
            delete_response = await self._make_request(
                method="DELETE",
                path=f"/services/{endpoint_id}",
            )

            logger.info(f"Service deletion initiated for '{service_name}'")
            logger.debug(f"Delete response: {delete_response.text}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Error deleting service: {e}")
            logger.debug(f"Response: {e.response.text if hasattr(e, 'response') else 'N/A'}")

    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get the status of a Backend.AI service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary containing service status information
        """
        service_info = await self.get_service_info(service_name)

        if not service_info:
            return {
                "status": ServiceStatus.UNKNOWN,
                "message": "Service not found",
            }

        status_str = service_info.get("status", "UNKNOWN")
        status = determine_service_status(status_str)

        return {
            "status": status,
            "raw_status": status_str,
            "service_id": service_info.get("id"),
            "endpoint": service_info.get("endpoint"),
        }

    async def get_endpoint_info(self, endpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed endpoint (model service) information using GraphQL query.

        Args:
            endpoint_id: UUID of the endpoint

        Returns:
            Endpoint information dictionary or None if not found
        """
        # GraphQL query to get endpoint by ID
        query = """
        query($endpoint_id: UUID!) {
            endpoint(endpoint_id: $endpoint_id) {
                endpoint_id
                name
                replicas
                status
                url
                open_to_public
                created_at
                runtime_variant { name }
                routings {
                    routing_id
                    session
                    status
                    traffic_ratio
                }
            }
        }
        """

        request_body = {
            "query": query,
            "variables": {"endpoint_id": endpoint_id},
        }

        try:
            response = await self._make_request(
                method="POST",
                path="/admin/gql",
                body=request_body,
            )

            result = response.json()
            data = result.get("data", {})
            endpoint_info = data.get("endpoint")

            return endpoint_info

        except httpx.HTTPStatusError as e:
            logger.error(f"Error querying endpoint: {e}")
            return None

    async def get_service_info(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed service information using GraphQL query.

        Args:
            service_name: Name of the service

        Returns:
            Service information dictionary or None if not found
        """
        # GraphQL query to get service by name
        query = """
        query($name: String!) {
            compute_session(name: $name) {
                id
                name
                status
                image
                created_at
                service_endpoint
            }
        }
        """

        request_body = {
            "query": query,
            "variables": {"name": service_name},
        }

        try:
            response = await self._make_request(
                method="POST",
                path="/admin/gql",
                body=request_body,
            )

            result = response.json()
            data = result.get("data", {})
            service_info = data.get("compute_session")

            return service_info

        except httpx.HTTPStatusError as e:
            logger.error(f"Error querying service: {e}")
            return None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - close HTTP client."""
        await self.client.aclose()
