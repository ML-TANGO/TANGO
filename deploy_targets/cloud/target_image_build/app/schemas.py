"""
Pydantic의 model과 SQLAlchemy의 model이 헷갈리지 않게 하기위해
Pydantic의 model을 사용하는 파일은 schemas.py로 만들고
SQLAlchemy의 model을 사용하는 파일은 models.py로 구성
SQLAlchemy에서는 정의할 때 '='
Pydantic에서는 정의할 때 ':'
"""
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, EmailStr


class CommandBody(BaseModel):
    environments: Optional[List] = []
    custom_pkg: Dict = {
        "apt": [],
        "pip": [],
        "conda": [],
    }

    class config:
        orm_mode = True  # dict 형태가 아니고 ORM model이더라도 읽게해줌.


class DeploymentTaskBody(BaseModel):
    build: Optional[Dict] = {
        "architecture": "",
        "accelerator": "cpu",
        "os": "ubuntu",
        "components": {
            "custom_packages": {"apt": [], "pypi": []},
        },
        "workdir": ""
    }
    deploy: Optional[Dict] = {
        "entrypoint": [],
        "mount": [
            {
                "src": "",
                "dst": ""
            },
        ],
        "network": {
            "service_host_ip": "",
            "service_host_port": 8000,
            "service_container_port": 8000,
        },
    }
    run_container: bool = False

    class config:
        orm_mode = True


class TaskBody(BaseModel):
    src: str
    target: str
    labels: Dict = {
        "base_distro": "ubuntu",
        "service_ports": [],  # {"name": "jupyter", "protocol": "http", "ports": [8081]},
        "accelerators": ["cuda"],
        "min_cpu": 1,
        "min_mem": "64m",
    }
    runtime_type: str
    runtime_path: str
    docker_commands: Optional[CommandBody] = None
    auto_push: bool = False
    allow_root: bool = False  # If user want to use sudo, Change to 'True'

    class config:
        orm_mode = True  # dict 형태가 아니고 ORM model이더라도 읽게해줌.
        # arbitrary_types_allowed = True


class TaskStatus(str, Enum):
    PENDING: str = "pending"
    BUILDING: str = "running"
    COMPLETE: str = "complete"
    ERROR: str = "error"


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    # phone_number: str = None
    # name: str = None


class TokenPayload(BaseModel):
    sub: Optional[str] = None


class CommitConfiguration(BaseModel):
    Hostname: Optional[str]
    Domainname: Optional[str]
    User: Optional[str]
    AttachStdin: Optional[bool] = False
    AttachStdout: Optional[bool] = True
    AttachStderr: Optional[bool] = True
    ExposedPorts: Optional[Dict] = {"80/tcp": {}, "443/tcp": {}}
    Tty: Optional[bool] = False
    OpenStdin: Optional[bool] = False
    StdinOnce: Optional[bool] = False
    Env: Optional[List] = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ]
    Cmd: Optional[List] = ["/bin/sh"]
    Healthcheck: Optional[Dict] = {
        "Test": ["string"],
        "Interval": 0,
        "Timeout": 0,
        "Retries": 0,
        "StartPeriod": 0,
    }
    ArgsEscaped: Optional[bool] = False
    Image: Optional[str] = "example-image:1.0"
    Volumes: Optional[Dict] = {"property1": {}, "property2": {}}
    WorkingDir: Optional[str] = "/public/"
    Entrypoint: Optional[List]
    NetworkDisabled: Optional[bool] = True
    MacAddress: Optional[str]
    OnBuild: Optional[List]
    Labels: Optional[Dict] = {
        "com.example.some-label": "some-value",
        "com.example.some-other-label": "some-other-value",
    }
    StopSignal: Optional[str] = "SIGTERM"
    StopTimeout: Optional[List[int]] = 10
    Shell: Optional[List] = ["/bin/sh", "-c"]


class CommitParamDict(BaseModel):
    container_id: str
    repo: Optional[str] = None
    tag: Optional[str] = None
    message: Optional[str] = None
    author: Optional[str] = None
    changes: Optional[Union[str, List]] = None
    config: Optional[CommitConfiguration] = None


class CommitParams(BaseModel):
    pause: bool = True
    param: Optional[CommitParamDict]

    class config:
        orm_mode = True


"""
class TaskBodyForFileUpload(BaseModel):
    labels: Dict = {
        "base_dirstro": "ubuntu16.04",
        "service_ports": [  # front에서 이 형식으로 날아와야 함
            {"name": "jupyter", "protocol": "http", "ports": [8081]},
            {"name": "jupyterlab", "protocol": "http", "ports": [8090]},
            {"name": "vscode", "protocol": "http", "ports": [8180]},
            {"name": "tensorboard", "protocol": "http", "ports": [6006]},
        ],
        "accelerators": ["cuda"],
        "min_cpu": 1,
        "min_mem": "1g",
    }
    app: Optional[List] = ['vscode', 'jupyter', 'jupyterlab', 'tensorboard']
    app_use: bool
    target_name: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
"""


class TaskBase(str):
    task_id: str


class TaskCreate(TaskBase):
    task_id: str


class TaskTuple(str):
    image: str


class TaskList(list):
    task_id: str
    status: str
    created_at: str
    pulling_at: str
    building_at: str
    pushing_at: str
    finished_at: str

    class config:
        orm_mode = True  # dict 형태가 아니고 ORM model이더라도 읽게해줌.
