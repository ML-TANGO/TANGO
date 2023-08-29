import asyncio
import json
# import os
import tarfile
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import IO, BinaryIO, Dict, List

import aiohttp
import aioredis.client
import jinja2
from aiodocker import Docker, DockerError, docker
from sqlalchemy.orm import Session
from yarl import URL

from . import crud
from .config import read_from_file
from .dockerfile_templates import (DOCKERFILE_TEMPLATE_CUSTOM_LABELS,
                                   DOCKERFILE_TEMPLATE_DEFAULT,)
from .exceptions import AutoPushError
from .setting import settings
from .utils import clear_color_char, create_logger

log = create_logger("image.builder.bg_svc")
MESSAGE_STREAM_DELAY = 0.1  # second
MESSAGE_STREAM_RETRY_TIMEOUT = 15000  # milisecond
Message = [bytes, bytes, Dict[bytes, bytes]]


class Forklift:
    def __init__(self, task_pending_queue, task_running_queue):
        self.task_pending_queue = task_pending_queue
        self.task_running_queue = task_running_queue
        self._docker = Docker()

    async def aclose(self):
        await self._docker.close()

    async def make_dockerfile(self, info: Dict):
        dockerfile_template = DOCKERFILE_TEMPLATE_DEFAULT
        tpl = jinja2.Template(dockerfile_template)
        try:
            for k, v in info["build"]["components"]["custom_packages"].items():
                info["build"]["components"]["custom_packages"][k] = [
                    package.strip() for package in v
                ]
            dockerfile_content = tpl.render(
                {
                    "src": info["build"]["os"],
                    # "runtime_type": info["runtime_type"],
                    # "runtime_path": Path(info["runtime_path"]),
                    # "base_distro": info["labels"]["base_distro"],
                    # "service_ports": info["labels"]["service_ports"],
                    # "min_cpu": info["labels"]["min_cpu"],
                    # "min_mem": info["labels"]["min_mem"],
                    # "accelerators": info["labels"]["accelerators"],
                    # "allow_root": info["allow_root"],
                    "architecture": info["build"]["architecture"],
                    # "envs": yaml_contents["build"]["components"]["environments"],
                    "packages": info["build"]["components"]["custom_packages"],
                    "copy_path": "nn_model",
                    "workdir": info["build"]["workdir"],
                }
            )
            log.info("==Dockerfile created!==")
            log.debug(dockerfile_content)
            return dockerfile_content
            # else:
            #     for k, v in info["docker_commands"]["custom_pkg"].items():
            #         info["docker_commands"]["custom_pkg"][k] = [
            #             package.strip() for package in v
            #         ]
            #     dockerfile_content = tpl.render(
            #         {
            #             "src": info["src"],
            #             "runtime_type": info["runtime_type"],
            #             "runtime_path": Path(info["runtime_path"]),
            #             "base_distro": info["labels"]["base_distro"],
            #             "service_ports": info["labels"]["service_ports"],
            #             "min_cpu": info["labels"]["min_cpu"],
            #             "min_mem": info["labels"]["min_mem"],
            #             "accelerators": info["labels"]["accelerators"],
            #             "envs": info["docker_commands"]["environments"],
            #             "packages": info["docker_commands"]["custom_pkg"],
            #             "allow_root": info["allow_root"],
            #         }
            #     )
            #     log.info("==Dockerfile created!==")
            #     return dockerfile_content
        except AttributeError as e:
            log.exception(e)
            raise
        except Exception as e:
            log.exception(e)
            raise

    async def build(self, db: Session, task: Dict, user_info: Dict):
        logs = []
        self.task_pending_queue.append(task)
        f = BytesIO(bytes(task["requested_dockerfile_contents"], encoding="utf-8"))
        tar_obj = self.mktar_from_dockerfile(f, user_info)
        log.info(f"==Build image...== {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        try:
            pool = await get_redis_pool()
            building_at: datetime = None
            finished_at: datetime = None
            crud.modify_tasks_status(db, task["task_id"], status="running")
            building_at = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            resp = self._docker.images.build(
                remote=None,
                fileobj=tar_obj,
                encoding="identity",
                path_dockerfile="Dockerfile",
                tag=task["requested_info"]["build"]["target_name"],
                quiet=False,
                stream=True,
            )
            # tar_obj.close()
            log.debug("pop from pending queue and append into running queue")
            self.task_running_queue.append(self.task_pending_queue.popleft())
            log.debug(f"pending queue : {len(self.task_pending_queue)}")
            log.debug(f"running queue :  {len(self.task_running_queue)}")
            async for item in resp:
                logs.append(item)
                log.debug(item)
                if "aux" in item.keys():
                    item["aux"] = json.dumps(item["aux"])
                if "errorDetail" in item.keys():
                    item["errorDetail"] = json.dumps(item["errorDetail"])
                if "progressDetail" in item.keys():
                    item["progressDetail"] = json.dumps(item["progressDetail"])
                await pool.xadd(
                    name=task["task_id"],
                    fields=item,
                    maxlen=settings.STREAM_MAX_LEN,
                )
            finished_at = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            ac_log, status = clean_up_build_log(logs)
            crud.modify_tasks_result(db, task["task_id"], ac_log)
            if status == "complete":
                await self.auto_push(task)
            self.task_running_queue.popleft()
            log.debug(f"running queue :  {len(self.task_running_queue)}")
            crud.modify_task(
                db,
                task["task_id"],
                building_at,
                finished_at,
                status=status,
                logs=ac_log,
            )
            if task["requested_info"]["build"] and status == "complete":
                log.debug(
                    f'Send running for {task["requested_target_img"]} container message.'
                )
                await self._handle_preset_result(
                    settings.DEPLOY_SERVER_URL, task["requested_info"]
                )
            # await self.aclose()
        except DockerError as docker_error:
            await self._handling_execption(db, task, docker_error, building_at, logs)
            raise
        except asyncio.exceptions.CancelledError as c:
            await self._handling_execption(db, task, c, building_at, logs)
            raise
        except AutoPushError as ape:
            await self._handling_execption(db, task, ape, building_at, logs)
            raise
        except TypeError as e:
            log.exception(e)
            raise
        except Exception as e:
            await self._handling_execption(db, task, e, building_at, logs)
            raise

    async def _handling_execption(
        self,
        db: Session,
        task: Dict,
        error_message,
        building_at: datetime,
        logs: List,
    ):
        log.exception(error_message)
        self.task_running_queue.popleft()
        log.debug(f"running queue :  {len(self.task_running_queue)}")
        finished_at = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        crud.modify_task(db, task["task_id"], building_at, finished_at, "error", logs)
        # await self.aclose()

    async def auto_push(self, task: Dict):
        if task["requested_auto_push"] is True:
            config, _ = read_from_file(None, daemon_name="forklift")
            auth_info = {
                "username": config["docker"]["auth"]["username"],
                "password": config["docker"]["auth"]["password"],
            }
            push_resp = self._docker.images.push(
                task["requested_info"]["target"], auth=auth_info, stream=True
            )
            async for item in push_resp:
                log.info(item)
                if "errorDetail" in item:
                    raise AutoPushError(item)
            log.info(
                f'Push to {task["requested_info"]["src"]} was succesfully completed'
            )

    async def fileupload_build_image(self, file: str, info):
        if info.app_use:
            dockerfile_template = (
                file.decode("utf-8") + DOCKERFILE_TEMPLATE_CUSTOM_LABELS
            )
        else:
            dockerfile_template = file.decode("utf-8")
        tpl = jinja2.Template(dockerfile_template)
        log.info(dockerfile_template)
        target_image = info.target_name
        log.info(target_image)
        dockerfile_content = tpl.render(
            {
                "runtime_type": "python",
                "runtime_path": Path("/usr/bin/python3"),
                "base_dirstro": info.labels["base_dirstro"],
                "service_ports": info.labels["service_ports"],
                "min_cpu": info.labels["min_cpu"],
                "min_mem": info.labels["min_mem"],
                "accelerators": info.labels["accelerators"],
                "app_use": info.app_use,
                "work_dir": "/Users/kangmin/lablup/backend.ai-forklift/app",
            }
        )
        log.info(dockerfile_content)
        f = BytesIO(bytes(dockerfile_content, encoding="utf-8"))
        tar_obj = self.mktar_from_dockerfile(f)
        time = []
        log.info("==Build image...==", datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        try:
            log.info(target_image)
            building_at = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            resp = self._docker.images.build(
                remote=None,
                fileobj=tar_obj,
                encoding="identity",
                path_dockerfile="Dockerfile",
                tag=target_image,
                quiet=False,
                stream=True,
            )
            tar_obj.close()
            logs = []
            async for item in resp:
                logs.append(item)
                # step이 있으면 스텝 핑을 True로 바꾸고 숫자를 증가시키고 스텝핑을 다시 False로 변경
                if "stream" in item:
                    if "Step" in item["stream"]:
                        global STEP_PING
                        STEP_PING = item["stream"]
                log.debug(item)
            finished_at = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            log.info(
                "Successfully complete to build image",
                datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
            )
            time.append(building_at)
            time.append(finished_at)
            logs.append(time)
            result = clean_up_build_log(logs)
            return json.dumps(result)
        except Exception as e:
            log.exception(e)
            raise

    async def show_image_list(self) -> list:
        try:
            log.info("== Images ==")
            img_list = []
            for image in await self._docker.images.list():
                tags = image["RepoTags"][0] if image["RepoTags"] else ""
                img_list.append(tags)
            # await self.aclose()
            return img_list
        except Exception as e:
            log.exception(e)
            raise

    async def show_image_info(self, target_image: str) -> dict:
        try:
            inspect = await self._docker.images.inspect(name=target_image)
            # await self.aclose()
            return inspect
        except Exception as e:
            log.exception(e)
            raise

    async def delete_image(self, target_image: str, force=False, noprune=False) -> list:
        try:
            result = await self._docker.images.delete(
                name=target_image, force=force, noprune=noprune
            )
            # await self.aclose()
            return result
        except Exception as e:
            log.exception(e)
            return e

    async def _handle_preset_result(self, url: URL, data: dict):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url + "/containers/",
                    data=json.dumps(data),
                ) as response:
                    return await response.text()
        except aiohttp.ClientResponseError as e:
            log.exception(e)
        except aiohttp.ClientError as e:
            log.exception(e)

    def mktar_from_dockerfile(self, dockerfile: BinaryIO, user_info) -> IO:
        """
        Create a zipped tar archive from a Dockerfile
        **Remember to close the file object**
        Args:
            fileobj: a Dockerfile
        Returns:
            a NamedTemporaryFile() object
        """

        f = tempfile.NamedTemporaryFile()
        with tarfile.open(mode="w:gz", fileobj=f) as t:
            # current_path = Path(os.getcwd())  # TODO Path could be change
            path = f"/TANGO/shared/common/{user_info['user_id']}/nn_model"
            if isinstance(dockerfile, BytesIO):
                dfinfo = tarfile.TarInfo("Dockerfile")
                dfinfo.size = len(dockerfile.getvalue())
                dockerfile.seek(0)
            else:
                dfinfo = t.gettarinfo(fileobj=dockerfile, arcname="Dockerfile")
            t.addfile(dfinfo, dockerfile)
            t.add(
                name=f"{path}", arcname="nn_model"
            )
            # COPY를 할때마다 여기서 t.add로 추가해줘야함
            # TODO: COPY 명령어가 추가되었을 때, 그걸 받아서 for문으로 처리해주는게 바람직함.
            log.debug(t.getnames())
        f.seek(0)
        return f


async def docker_commit(params: Dict):
    docker_instance = Docker()
    container = docker.DockerContainers(docker_instance)
    running_container = container.container(params["param"]["container_id"])
    await running_container.commit(
        repository=params["param"]["repo"],  # commit이 될 repository 이름
        tag=params["param"]["tag"],  # 태그
        message=params["param"]["message"],  # 커밋 메세지
        author=params["param"]["author"],  # 작성자
        changes=params["param"]["changes"],  # 새롭게 붙일 Dockerfile 명령어
        config=params["param"]["config"],  # ?
        pause=params["pause"],  # 실행중인 container를 commit하기 위해 일시정지
    )
    await docker_instance.close()


class CustomDockerContainers(docker.DockerContainers):
    async def list(self, **kwargs):
        data = await self.docker._query_json(
            "containers/json", method="GET", params=kwargs
        )
        return data


async def docker_container_list():
    docker_instance = Docker()
    container = CustomDockerContainers(docker_instance)
    container_list = await container.list()
    return container_list


async def get_redis_pool():
    try:
        pool = await aioredis.client.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
        )
        return pool
    except ConnectionRefusedError:
        log.exception(
            (f"cannot connect to redis on: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        )
        return None


async def read_from_stream(
    redis: aioredis.client,
    task_id: str,
    first_read: bool,
    # latest_id: str = None,
    # past_ms: int = None,
    # last_n: int = None,
):
    timeout_ms = 60 * 1000
    # Blocking read for every message added after latest_id, using XREAD
    # if latest_id is not None:
    #     return await redis.xread([stream], latest_ids=[latest_id], timeout=timeout_ms)

    # # Blocking read for every message added after current timestamp minus past_ms, using XREAD
    # if past_ms is not None:
    #     server_time_s = await redis.time()
    #     latest_id = str(round(server_time_s * 1000 - past_ms))
    #     return await redis.xread([stream], latest_ids=[latest_id], timeout=timeout_ms)

    # # Non-blocking read for last_n messages, using XREVRANGE
    # if last_n is not None:
    #     messages = await redis.xrevrange(stream, count=last_n)
    #     # redis.xgroup_create(
    #     #     name='build_event',
    #     #     groupname='',
    #     # )
    #     return list(reversed([(stream.encode("utf-8"), *m) for m in messages]))
    last_id = b"$"
    if first_read:
        return await redis.xread({task_id: 0}, block=timeout_ms)
    else:
        return await redis.xread({task_id: last_id}, block=timeout_ms)
    # Default case, blocking read for all messages added after calling XREAD
    # return await redis.xread({task_id: last_id}, block=timeout_ms)


async def generate_event_source(task_id: str):
    # Create redis connection with aioredis.create_redis
    redis = await get_redis_pool()
    stream = task_id
    # Loop for as long as client is connected and our reads don't time out, sending messages to client over websocket
    first_read: bool = True
    while True:
        messages: List[Message]
        try:
            messages = await read_from_stream(redis, task_id, first_read)
            first_read = False
        except Exception as e:
            log.info(f"read timed out for stream {stream}, {e}")
            return

        # If we have no new messages, note that read timed out and return
        if len(messages) == 0:
            log.info(f"no new messages, read timed out for stream {stream}")
            return
        prepared_messages = []
        for msg in messages:
            latest_id = msg[1][0][0].decode("utf-8")
            payload = {
                k.decode("utf-8"): v.decode("utf-8") for k, v in msg[1][0][1].items()
            }
            prepared_messages.append({"message_id": latest_id, "payload": payload})
        await asyncio.sleep(0.00000001)
        yield {
            "event": "update",
            "retry": MESSAGE_STREAM_RETRY_TIMEOUT,
            "data": json.dumps(prepared_messages),
        }


def clean_up_build_log(log_list: List):
    temp_log_dict: dict = {}
    result: list = []
    status: str = None
    for log_data in log_list:
        if isinstance(log_data, dict):  # In log_list, There is 'time' list.
            if repr(list(log_data.values())[0]) == "'\\n'":
                continue
            elif "error" in log_data or "errorDetail" in log_data:
                status = "error"
                temp_log_dict["error"] = log_data.pop("error")
                temp_log_dict["errorDetail"] = log_data.pop("errorDetail")
                result.append(temp_log_dict.copy())
            elif "aux" in log_data:
                temp_log_dict["stream"] = log_data.pop("aux")
            elif "stream" in log_data:
                temp_log_dict["stream"] = log_data.pop("stream")  # change key name
                temp_log_dict["stream"] = clear_color_char(
                    "".join(temp_log_dict["stream"].splitlines())
                )
                if temp_log_dict["stream"]:
                    result.append(temp_log_dict.copy())
                status = "complete"
            elif "status" in log_data:
                temp_log_dict["stream"] = log_data.pop("status")
            else:
                log.info(f"log : {log_data}")
                continue
    return result, status
