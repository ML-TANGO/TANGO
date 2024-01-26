import enum
import json
from typing import Optional

from sqlalchemy import Column
from sqlmodel import Column, Field, JSON, SQLModel

# from cloud_manager.targets.defs import DeployYaml


class ServiceStatus(enum.Enum):
    STARTED = "started"
    BUILDING = "building"
    PREPARING = "preparing"
    RUNNING = "running"
    STOPPED = "stopped"  # normally stopped
    FAILED = "failed"  # launch failed
    COMPLETED = "completed"  # for batch-job case?


RunningServiceStatuses = [
    ServiceStatus.STARTED,
    ServiceStatus.BUILDING,
    ServiceStatus.PREPARING,
    ServiceStatus.RUNNING,
]


# NOTE: Currently, it is not possible to use a custom type with SQLModel.
# https://github.com/tiangolo/sqlmodel/pull/18
# class DeployYamlColumn(TypeDecorator):
#     impl = JSON

#     def process_bind_param(self, value, dialect):
#         if value is None:
#             return None
#         return value.dict()

#     def process_result_value(self, value, dialect):
#         if value is None:
#             return None
#         return DeployYaml(**value)


class ServiceBase(SQLModel):
    user_id: str = Field(index=True)
    project_id: str = Field(index=True)


class Service(ServiceBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    status: ServiceStatus = Field(default=ServiceStatus.STARTED, index=True)
    #: The TANGO YAML spec delivered by deployment.yaml file.
    deploy_yaml: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # deploy_yaml: Optional[DeployYaml] = Field(default=None, sa_column=DeployYamlColumn)
    #: Additional informaiton specific to the target.
    target_info: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class ServiceRead(ServiceBase):
    status: ServiceStatus


class ServiceRequestParams(ServiceBase):
    pass
