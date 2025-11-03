from typing import List, Union, Optional
from pydantic import BaseModel

# from cloud_manager.targets.local.docker import LocalDocker
from cloud_manager.targets.gcp.cloudrun import CloudRun
from cloud_manager.targets.aws.ecs import AWSECS
from cloud_manager.targets.ktc.ktc import KTCloud
from cloud_manager.targets.compute_session.compute_session import ComputeSession
from cloud_manager.targets.accelerators import AcceleratorSpec

# Mapping between deployment target strings and their respective classes.
TARGET_CLASS_MAP = {
    # "docker": LocalDocker,
    "gcp-cloudrun": CloudRun,
    "aws-ecs": AWSECS,
    "ktc": KTCloud,
    "compute-session": ComputeSession,
}


class Build(BaseModel):
    class Components(BaseModel):
        class CustomPackages(BaseModel):
            apt: Optional[List[str]] = None
            pypi: Optional[List[str]] = None

        engine: str = "pytorch"
        libs: Optional[List[str]] = None
        custom_packages: Optional[CustomPackages] = None

    architecture: str = "x86"
    accelerator: Optional[Union[str, AcceleratorSpec]] = "cpu"
    os: Optional[str] = "ubuntu20.04"
    image_uri: str
    components: Optional[Components] = None

    def get_accelerator_spec(self) -> AcceleratorSpec:
        """
        Get accelerator specification, converting from string if necessary.

        Returns:
            AcceleratorSpec instance
        """
        if isinstance(self.accelerator, str):
            return AcceleratorSpec.from_string(self.accelerator)
        return self.accelerator or AcceleratorSpec()


class ModelServiceConfig(BaseModel):
    """
    Configuration for model service deployment.
    """

    model: Optional[str] = None  # Model VFolder name
    runtime_variant: str = "custom"  # custom, cmd, vllm
    model_mount_destination: str = "/models"
    model_definition_path: Optional[str] = None  # Path to model-definition.yml
    environ: Optional[dict] = None  # Environment variables
    scaling_group: str = "default"
    resource_opts: Optional[dict] = None  # Additional resource options like shmem


class Deploy(BaseModel):
    class Resources(BaseModel):
        cpu: Optional[str | int] = None
        memory: Optional[str | int] = None
        gpu: Optional[str | int] = None

        class Config:
            extra = "allow"  # Allow additional resource fields (e.g., atom-max.device)

    class Network(BaseModel):
        service_host_ip: str
        service_host_port: int
        service_container_port: Optional[int] = None

    type: str = "gcp-cloudrun"
    work_dir: Optional[str] = "/workspace"
    pre_exec: Optional[List[List[Union[str, int]]]] = None
    entrypoint: Optional[List[str]] = None
    resources: Optional[Resources] = None
    network: Optional[Network] = None

    # Model service configuration
    model_service: Optional[ModelServiceConfig] = None
    cluster_mode: str = "single-node"  # single-node, multi-node
    cluster_size: int = 1
    replicas: int = 1
    open_to_public: bool = True

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
