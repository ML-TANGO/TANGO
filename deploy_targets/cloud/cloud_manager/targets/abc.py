import os
from abc import ABC, abstractmethod

from yarl import URL


IMAGE_BUILDER_URL = os.getenv("IMAGE_BUILDER_URL", "http://127.0.0.1:7077")


class CloudTargetBase(ABC):
    def __init__(self, user_id: str, project_id: str):
        self.user_id = user_id
        self.project_id = project_id

    @abstractmethod
    async def start_service(self):
        pass

    @abstractmethod
    async def stop_service(self):
        pass

    @abstractmethod
    async def get_service_status(self):
        pass

    async def build_image(self, build_option):
        # TODO: Generalize the build process to actually build the image per spec.
        # crud.create_task(project_data)
        project_data = {"user_id": self.user_id, "project_id": self.project_id}
        return await self._build_image(IMAGE_BUILDER_URL, project_data)

    async def _build_image(url: URL, data: dict):
        pass

    async def push_image(self):
        pass
