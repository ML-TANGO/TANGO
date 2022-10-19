import aiohttp
import json
import os
import yaml
import uvicorn

from deploy import ManipulateContainer
from fastapi import FastAPI, Response, HTTPException
from yarl import URL

app = FastAPI(title='Deploy_Server')


@app.get("/start/")
async def run_container(
    user_id: str,
    project_id: str,
):
    try:
        # IMAGE_BUILD_URL = "http://0.0.0.0:7007"
        user_input_data: dict = {}
        path = f"{os.getcwd()}/shared/common/{user_id}/{project_id}/deployment.yml"  # TODO path could be changed when 공유폴더 is decided
        with open(path) as f:
            deployment_dict = yaml.load(f, Loader=yaml.FullLoader)
        user_input_data = deployment_dict
        user_input_data["user"] = {"user_id": user_id, "project_id": project_id}
        # result = await _build_image(IMAGE_BUILD_URL, user_input_data)
        # await _build_image(
        #     IMAGE_BUILD_URL, user_input_data
        # )

        # Container start TODO: This part and requesting to image_builder part should be run separately
        container = ManipulateContainer()
        await container.run_container(data=user_input_data)
        return Response(
            content="starting",
            status_code=200,
            media_type="text/plain"
        )
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=400,
            detail="failed",
        )


@app.get("/stop/")
async def stpp_container(
    user_id: str,
    project_id: str,
    # container_id: str,
):
    container = ManipulateContainer()
    try:
        await container.stop_container(container_id="")  # TODO Need to know 'container_id'
        return Response(
            content="finished",
            status_code=200,
            media_type="text/plain",
        )
    except Exception as e:
        print(e)


@app.get("/status_request/")
async def get_container(
    user_id: str,
    project_id: str,
    # container_id: str,
):
    container = ManipulateContainer()
    try:
        status = await container.get_container(container_id="")  # TODO Need to know 'container_id'
        return Response(
            content=status,
            status_code=200,
            media_type="text/plain"
        )
    except Exception as e:
        print(e)
        return HTTPException(
            status_code=400,
            detail=e
        )


async def _build_image(url: URL, data: dict):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url + "/build/submit/preset/",
                data=json.dumps(data),
                ssl=False,
            ) as response:
                return await response.text()
    except aiohttp.ClientResponseError as e:
        print(e)
    except aiohttp.ClientError as e:
        print(e)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8890,
        reload=True,
    )
