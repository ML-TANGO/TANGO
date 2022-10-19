from aiodocker import Docker


class ManipulateContainer:

    def __init__(self):
        self._docker = Docker()

    async def run_container(self, data):
        try:
            config = {
                "Cmd": data["deploy"]["entrypoint"],
                "Image": "yolov3:1.0v",
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
                    # "Binds": [
                    #     "/tmp:/tmp",  # host-src:container-dest to bind-mount a host path into the container. Both host-src, and container-dest must be an absolute path.
                    # ],
                    "PortBindings": {
                        f'{data["deploy"]["network"]["service_container_port"]}/tcp': [
                            {
                                "HostPort": f'{data["deploy"]["network"]["service_host_port"]}',
                            }
                        ],
                    },
                },
            }
            await self._docker.containers.run(config=config)
            await self._docker.close()
        except Exception as e:
            raise e

    async def stop_container(self, container_id):
        try:
            await self._docker.containers.container(container_id="5f3e23a63d0d").stop()
            await self._docker.close()
        except Exception as e:
            raise e

    async def get_container(self, container_id):
        try:
            status = await self._docker.containers.container(container_id="5f3e23a63d0d").show()
            await self._docker.close()
            return status["State"]["Status"]
        except Exception as e:
            raise e
