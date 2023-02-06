from aiodocker import Docker, docker
from sqlalchemy.orm import Session
import crud


class ManipulateContainer:

    def __init__(self):
        self._docker = Docker()

    async def run_container(self, db, data):
        try:
            config = {
                "Cmd": data["deploy"]["entrypoint"],
                "Image": f'{data["build"]["target_name"]}',
                "AttachStdin": False,
                "AttachStdout": False,
                "AttachStderr": False,
                "Tty": False,
                "OpenStdin": False,
                "ExposedPorts": {
                    f'{data["deploy"]["network"]["service_container_port"]}/tcp': {
                    }
                },
                "HostConfig": {
                    "PortBindings": {
                        f'{data["deploy"]["network"]["service_container_port"]}/tcp': [
                            {
                                "HostPort": f'{data["deploy"]["network"]["service_host_port"]}',
                            }
                        ],
                    },
                },
            }
            container_name = crud.get_container_name(db, data["user"])[0]
            await self._docker.containers.run(config=config, name=container_name)
            await _save_container_id(db, container_name, data["user"])
            crud.modify_tasks_status(db, user_data=data["user"], status="completed")
            await self._docker.close()
        except Exception as e:
            crud.modify_tasks_status(db, user_data=data["user"], status="failed")
            await self._docker.close()
            raise e

    async def stop_container(self, container_id, user_data: dict, db: Session):
        try:
            await self._docker.containers.container(container_id=container_id).stop()
            crud.modify_tasks_status(db, user_data=user_data, status="stopped")
            await self._docker.close()
        except Exception as e:
            await self._docker.close()
            raise e

    async def get_container(self, container_id):
        try:
            status = await self._docker.containers.container(container_id=container_id).show()
            await self._docker.close()
            return status["State"]["Status"]
        except Exception as e:
            await self._docker.close()
            raise e


class CustomDockerContainers(docker.DockerContainers):
    async def list(self, **kwargs):
        data = await self.docker._query_json(
            "containers/json", method="GET", params=kwargs
        )
        return data


async def _save_container_id(db, container_name, user_input):
    try:
        docker_instance = Docker()
        _docker = CustomDockerContainers(docker_instance)
        list = await _docker.list()
        index_num: int = 0
        for i, v in enumerate(list):
            if v["Names"] == container_name:
                index_num = i
                break
        container_id = list[index_num]["Id"]
        crud.modify_container_id(db, project_id=user_input["project_id"], container_id=container_id)
    except Exception as e:
        raise(e)
