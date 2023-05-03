import aiohttp
import ast
import json
# import os
import yaml
import uvicorn

from deploy import ManipulateContainer
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from yarl import URL
from database import SessionLocal, engine
from typing import Union
import models
import crud

app = FastAPI(title='Deploy_Server')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response


@app.get("/start/")
async def image_build(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
):
    if user_id and project_id:
        return Response(
            content="started",
            status_code=200,
            media_type="text/plain"
        )
    try:
        IMAGE_BUILD_URL = "http://0.0.0.0:8088"
        user_input_data: dict = {}
        path = f"/TANGO/shared/common/{user_id}/{project_id}/deployment.yml"  # TODO path could be changed when 공유폴더 is decided
        with open(path) as f:
            deployment_dict = yaml.load(f, Loader=yaml.FullLoader)
        user_input_data = deployment_dict
        user_input_data["user"] = {"user_id": user_id, "project_id": project_id}
        crud.create_task(db, user_input_data["user"])
        await _build_image(IMAGE_BUILD_URL, user_input_data, db)
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


@app.post("/containers/")
async def run_container(
    request: Request,
    db: Session = Depends(get_db)
):
    container = ManipulateContainer()
    user_input_data = ast.literal_eval(jsonable_encoder(await request.body()))
    await container.run_container(db, data=user_input_data)
    return Response(
        content="deploy container is started!!",
        status_code=200,
        media_type="text/plain"
    )


@app.get("/stop/")
async def stpp_container(
    user_id: str,
    project_id: str,
    db: Session = Depends(get_db),
):
    container = ManipulateContainer()
    try:
        user_input_data = {"user_id": user_id, "project_id": project_id}
        container_id = crud.get_container_id(db, user_input_data)[0]
        await container.stop_container(container_id=container_id, user_data=user_input_data, db=db)
        return Response(
            content="finished",
            status_code=200,
            media_type="text/plain",
        )
    except Exception as e:
        print(e)


@app.get("/status_request/")
async def get_container(
    user_id: Union[str, None] = None,
    project_id: Union[str, None] = None,
    db: Session = Depends(get_db),
):
    if user_id and project_id:
        return Response(
            content="ready",
            status_code=200,
            media_type="text/plain"
        )
    # container = ManipulateContainer()
    try:
        # user_input_data = {"user_id": user_id, "project_id": project_id}
        # container_id = crud.get_container_id(db, user_input_data)[0]
        # status = await container.get_container(container_id=container_id)
        user_data = {"user_id": user_id, "project_id": project_id}
        status = crud.get_user_specific_tasks_status(db, user_data)
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


async def _build_image(url: URL, data: dict, db: Session):
    try:
        crud.modify_tasks_status(db, user_data=data["user"], status="running")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url + "/build/submit/preset/",
                data=json.dumps(data),
                ssl=False,
            ) as response:
                return await response.text()
    except aiohttp.ClientResponseError as e:
        crud.modify_tasks_status(db, user_data=data["user"], status="failed")
        print(e)
    except aiohttp.ClientError as e:
        crud.modify_tasks_status(db, user_data=data["user"], status="failed")
        print(e)

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8890,
    )
