import sqlmodel
import yaml
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, HTTPException

from cloud_manager.database import get_db_session
from cloud_manager.models import ServiceStatus, Service
from cloud_manager.targets.defs import DeployYaml, TARGET_CLASS_MAP


async def get_service(user_id: str, project_id: str) -> Optional[Service]:
    with get_db_session() as db:
        service = db.exec(
            sqlmodel.select(Service).where(
                Service.user_id == user_id,
                Service.project_id == project_id,
            )
        ).first()

    if not service:
        return None

    if not service.deploy_yaml:
        service.deploy_yaml = None
    elif isinstance(service.deploy_yaml, dict):
        service.deploy_yaml = DeployYaml(**service.deploy_yaml)

    return service


async def save_service(service: Service):
    if service.deploy_yaml and isinstance(service.deploy_yaml, DeployYaml):
        service.deploy_yaml = service.deploy_yaml.dict()
    with get_db_session() as db:
        db.add(service)
        db.commit()


async def initialize_service(user_id: str, project_id: str):
    return Service(user_id=user_id, project_id=project_id)


async def read_and_validate_deploy_yaml(deploy_yaml_path: Path) -> DeployYaml:
    if not deploy_yaml_path.exists():
        raise HTTPException(status_code=400, detail="Deployment YAML not found")
    try:
        with open(deploy_yaml_path, "r") as f:
            deploy_yaml_raw = yaml.load(f, Loader=yaml.SafeLoader)
            return DeployYaml(**deploy_yaml_raw)
    except yaml.YAMLError:
        raise HTTPException(status_code=400, detail="Invalid deployment YAML spec")


async def launch_service(
    user_id: str, project_id: str, deploy_yaml: DeployYaml, bg_tasks: BackgroundTasks
):
    target_class = TARGET_CLASS_MAP[deploy_yaml.deploy.type]
    target = target_class(user_id, project_id)

    async def _launch():
        try:
            service = await get_service(user_id, project_id)
            if not service:
                raise HTTPException(status_code=404, detail="Service not found")
            # service.status = ServiceStatus.BUILDING
            # await save_service(service)
            # TODO: Generalize the build process to actually build the image per spec.
            # await target.build_image(deploy_yaml.build)
            # service = await get_service(user_id, project_id)
            # with get_db_session() as db:
            #     db.refresh(service)
            #     service.status = ServiceStatus.PREPARING
            #     db.add(service)
            #     db.commit()

            await target.start_service(deploy_yaml)
            service = await get_service(user_id, project_id)
            service.status = ServiceStatus.RUNNING
            await save_service(service)
        except Exception:
            service = await get_service(user_id, project_id)
            if not service:
                raise HTTPException(status_code=404, detail="Service not found")
            service.status = ServiceStatus.FAILED
            await save_service(service)
            raise

    bg_tasks.add_task(_launch)
