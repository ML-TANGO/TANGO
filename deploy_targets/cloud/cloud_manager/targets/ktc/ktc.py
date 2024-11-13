import aiohttp
import yarl
import yaml
import json
from pathlib import Path
import os
from typing import List
from cloud_manager.targets.abc import CloudTargetBase
from cloud_manager.models import ServiceStatus

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class KTCloud(CloudTargetBase):
    def __init__(self, user_id: str, project_id: str):
        self.ktc_address = None
        super().__init__(user_id, project_id)

    async def start_service(self, deploy_yaml) -> dict[str, str]:
        url = yarl.URL(deploy_yaml.deploy.address) / 'start'
        output = None
        payload = {
            "service_name": deploy_yaml.deploy.service_name,
            "port": deploy_yaml.deploy.network.service_container_port,
            "cpu": deploy_yaml.deploy.resources.cpu,
            "memory": deploy_yaml.deploy.resources.memory
        }
        gpu_val = getattr(deploy_yaml.deploy.resources, 'gpu', None)

        if gpu_val:
            payload["gpu"] = gpu_val

        logging.info(f"url: {url} payload: {payload}")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                output = await response.text()
                logging.info(f"start service output: {output}")

    async def stop_service(self, service_name: str):
        deploy_yaml_path = Path(
            f"/shared/common/{self.user_id}/{self.project_id}/deployment.yaml"
        )
        ktc_address = self._get_deploy_address(deploy_yaml_path)
        url = yarl.URL(ktc_address) / 'stop'
        data = {
            "service_name": service_name
        }
        output = None

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                output = await response.text()
                logging.info(f"stop service output: {output}")

    async def get_service_status(self, service_name: str):
        deploy_yaml_path = Path(
            f"/shared/common/{self.user_id}/{self.project_id}/deployment.yaml"
        )
        ktc_address = self._get_deploy_address(deploy_yaml_path)
        url = yarl.URL(ktc_address) / 'status'
        data = {
            "service_name": service_name
        }
        output = None

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                output = await response.text()
                logging.info(f"stop service output: {output}")
        output_object = json.loads(output)
        if output_object["status"] == "running":
            return {"status": ServiceStatus.RUNNING}
        elif output_object["status"] == "stopped":
            return {"status": ServiceStatus.STOPPED}
        else:
            retutn_obj = {"status": ServiceStatus.FAILED}
            error_msg = getattr(output_object, 'error', None)
            if error_msg:
                output_object["error"] = error_msg
            return retutn_obj

    def _get_deploy_address(self, yaml_path):
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

            if 'deploy' in data and 'address' in data['deploy']:
                return data['deploy']['address']
            else:
                logging.error("'deploy.address' not found in the YAML file")
                return None
        except FileNotFoundError:
            logging.error(f"File not found: {yaml_path}")
            return None
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            return None