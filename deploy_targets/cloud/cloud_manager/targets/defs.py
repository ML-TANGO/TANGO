from typing import List, Union, Optional
from pydantic import BaseModel

# from cloud_manager.targets.local.docker import LocalDocker
from cloud_manager.targets.gcp.cloudrun import CloudRun
from cloud_manager.targets.aws.ecs import AWSECS
from cloud_manager.targets.ktc.ktc import KTCloud

# Mapping between deployment target strings and their respective classes.
TARGET_CLASS_MAP = {
    # "docker": LocalDocker,
    "gcp-cloudrun": CloudRun,
    "aws-ecs": AWSECS,
    "ktc": KTCloud,
}


class Build(BaseModel):
    class Components(BaseModel):
        class CustomPackages(BaseModel):
            apt: Optional[List[str]]
            pypi: Optional[List[str]]

        engine: str = "pytorch"
        libs: Optional[List[str]]
        custom_packages: Optional[CustomPackages]

    architecture: str = "x86"
    accelerator: Optional[str] = "cpu"
    os: Optional[str] = "ubuntu20.04"
    image_uri: str
    components: Components


class Deploy(BaseModel):
    class Resources(BaseModel):
        cpu: Optional[str | int] = None
        memory: Optional[str | int] = None
        gpu: Optional[str | int] = None

    class Network(BaseModel):
        service_host_ip: str
        service_host_port: int
        service_container_port: Optional[int] = None

    type: str = "gcp-cloudrun"
    work_dir: Optional[str] = "/workspace"
    pre_exec: Optional[List[List[Union[str, int]]]] = None
    entrypoint: List[str]
    resources: Optional[Resources] = None
    network: Network

    class Config:
        extra = "allow"


class DeployYaml(BaseModel):
    """
    Represents the overall configuration delivered by deployment.yaml.
    """

    build: Build
    deploy: Deploy

    class Config:
        extra = "allow"
