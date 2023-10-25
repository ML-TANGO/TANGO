from yarl import URL
from cloud_manager.targets.abc import CloudTargetBase
from cloud_manager.targets.defs import DeployYaml
from fastapi import Response, HTTPException
from aiodocker import Docker
from aiodocker.exceptions import DockerError
import os
import aiohttp
import json


IMAGE_BUILDER_URL = os.getenv("IMAGE_BUILDER_URL", "http://127.0.0.1:7077")


class LocalDocker(CloudTargetBase):
    def __init__(self, user_id: str, project_id: str):
        self.user_id = user_id
        self.project_id = project_id

    async def build_image(self, deploy_yaml: DeployYaml):
        try:
            image_builder_url = IMAGE_BUILDER_URL
            data = {
                "user": {
                    "user_id": self.user_id,
                    "project_id": self.project_id,
                },
                "build": {
                    "image_arch": deploy_yaml.build.architecture,  # Source
                    "image_os": deploy_yaml.build.os,              # Source
                    "image_uri": deploy_yaml.build.image_uri,      # Target
                    "image_accelerator": deploy_yaml.build.accelerator,
                    "image_ml_engine": deploy_yaml.build.components.engine,
                    "image_python_base": deploy_yaml.build.components.libs,
                    "image_custom_packages": deploy_yaml.build.components.custom_packages,
                },
                "deploy": {
                    "deploy_type": deploy_yaml.deploy.type,
                    "deploy_service_name": deploy_yaml.deploy.name,
                    "deploy_work_dir": deploy_yaml.deploy.work_dir,
                    "deploy_pre_exec": deploy_yaml.deploy.pre_exec,
                    "deploy_entrypoint": deploy_yaml.deploy.entrypoint,
                    "deploy_cloud_service_host_ip": deploy_yaml.deploy.network.service_host_ip,      # for cloud
                    "deploy_cloud_service_host_port": deploy_yaml.deploy.network.service_host_port,  # for cloud
                    "deploy_k8s_nfsip": deploy_yaml.deploy.k8s.nfsip,                                # for k8s
                    "deploy_k8s_nfspath": deploy_yaml.deploy.k8s.nfspath,                            # for k8s
                },
            }
            await self._build_image(image_builder_url, data)
            return Response(
                content="starting",
                status_code=200,
                media_type="text/plain"
            )
        except Exception as e:
            print(e)
            return HTTPException(
                status_code=400,
                detail="failed",
            )

    async def _build_image(url: URL, data: dict):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url + "/build/submit/preset/",
                    data=json.dumps(data),
                    ssl=False,
                ) as response:
                    return await response.text()
        except aiohttp.ClientResponseError as e:
            print(e)
        except aiohttp.ClientError as e:
            print(e)

    async def start_service(self):
        # start deploy container
        pass

    async def stop_service(self):
        # stop deploy container
        pass

    async def get_service_status(self):
        # container status
        pass

    async def push_image(self, deploy_yaml: DeployYaml, is_public: bool = True):
        docker = Docker()
        try:
            target_uri = deploy_yaml.build.image_uri
            if is_public:
                docker.images.push(target_uri)
            else:
                print("Private registry is not supported yet.")
                raise
        except DockerError as e:
            raise e
        finally:
            docker.close()
