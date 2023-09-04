from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware


from cloud_manager.database import create_db_and_tables
from cloud_manager.models import RunningServiceStatuses, ServiceStatus, Service
from cloud_manager.targets.defs import TARGET_CLASS_MAP
from cloud_manager.service import (
    get_service,
    launch_service,
    read_and_validate_deploy_yaml,
    save_service,
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def status():
    return Response(content="", status_code=200, media_type="text/plain")


@app.get("/start")
# NOTE: FastAPI's dependency injection does not seem to work with SQLite DB.
#       Using the injection raises the following error:
#         SQLite objects created in a thread can only be used in that same
#         thread. The object was created in thread id 140737461385024 and this
#         is thread id 140737390343744.
#       So, we are manually creating a DB session whenever needed.
# async def start_service(user_id: str, project_id: str, db: DBSessionDepends):
async def start_service(user_id: str, project_id: str, bg_tasks: BackgroundTasks):
    """
    Start a service in a container for the given user and project ID.
    """
    service = await get_service(user_id, project_id)
    if service and service.status in RunningServiceStatuses:
        raise HTTPException(
            status_code=400, detail=f"Service already exists ({service.status.value})"
        )

    service = Service(user_id=user_id, project_id=project_id)

    # TODO: Current TANGO implementation assumes there is a shared mount folder
    #       for every service container. This folder contains the deployment YAML
    #       and other files needed to deploy services. This is not a good design
    #       for cloud deployment, where the folder cannot be shared or mounted
    #       across the public network.
    deploy_yaml_path = Path(
        f"/shared/common/{service.user_id}/{service.project_id}/deployment.yaml"
    )
    deploy_yaml = await read_and_validate_deploy_yaml(deploy_yaml_path)

    service.status = ServiceStatus.STARTED
    service.deploy_yaml = deploy_yaml
    await save_service(service)
    await launch_service(user_id, project_id, deploy_yaml, bg_tasks)

    return Response(content="started", status_code=200, media_type="text/plain")


@app.get("/stop")
async def stop_service(user_id: str, project_id: str):
    """
    Stop a service for the given user and project ID.
    """
    service = await get_service(user_id, project_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    target_class = TARGET_CLASS_MAP[service.deploy_yaml.deploy.type]
    target = target_class(service.user_id, service.project_id)
    await target.stop_service(service.deploy_yaml.deploy.service_name)
    service.status = ServiceStatus.STOPPED
    await save_service(service)
    return Response(content="finished", status_code=200, media_type="text/plain")


@app.get("/status_request")
async def status_request(user_id: str, project_id: str):
    # async def status_request(user_id: str, project_id: str):
    """
    Get the status of a service for the given user and project ID.
    """
    service = await get_service(user_id, project_id)
    if not service:
        raise HTTPException(status_code=404, detail="Service not found")
    target_class = TARGET_CLASS_MAP[service.deploy_yaml.deploy.type]
    target = target_class(service.user_id, service.project_id)
    resp = await target.get_service_status(service.deploy_yaml.deploy.service_name)
    # if service.target_info.get("service_url"):
    #     # TANGO manager does not receive JSON response, so just print it here.
    #     print(f"Service URL: {service.target_info['service_url']}")
    return Response(
        content=resp["status"].value, status_code=200, media_type="text/plain"
    )
