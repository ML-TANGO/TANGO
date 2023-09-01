import asyncio
import ast
import json
from collections import deque
from datetime import timedelta
from io import BytesIO
from typing import Any

import bcrypt
from fastapi import FastAPI  # BackgroundTasks
from fastapi import (Depends, HTTPException, Request, Response,
                     status)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from pydantic import ValidationError
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from starlette.responses import JSONResponse

from . import crud, models, schemas
from .auth import AuthConfig, create_access_token
from .bg_svc import (Forklift, docker_commit, docker_container_list,
                     generate_event_source)
from .config import read_from_file
from .database import SessionLocal, engine
from .exceptions import BuildLogEmptyError
from .utils import create_logger

models.Base.metadata.create_all(bind=engine)
config, _ = read_from_file(None, daemon_name="forklift")

app = FastAPI(title="forklift")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login/access-token/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

DOCKERFILE_CONTENTS: Any = None
# TASK_DATA = None
config, _ = read_from_file(None, daemon_name="forklift")
task_pending_queue = deque()
task_running_queue = deque(maxlen=config["maxImageRequest"]["max_num"])

log = create_logger("image.builder.server")
builder = Forklift(task_pending_queue, task_running_queue)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    response = Response("Internal server error", status_code=500)
    try:
        request.state.db = SessionLocal()
        response = await call_next(request)
    finally:
        request.state.db.close()
    return response


#  Dependency
def get_db(request: Request):
    """
    request마다 독립된 데이터베이스 session/connection을 가져야하며,
    모든 요청은 한 세션 안에서 이루어지고난 뒤 요청이 끝나면 close 해야한다.
    그리고 다음 요청에는 새로운 요청이 생성된다.
    """
    return request.state.db


def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
) -> models.Users:
    try:
        payload = jwt.decode(
            token, AuthConfig.SECRET_KEY, algorithms=[AuthConfig.ALGORITHM]
        )
        token_data = schemas.TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = crud.get_user_info(db, email=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/build/submit/custom/", name="build custom image")
async def general_submit(
    user_input: schemas.TaskBody,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        if len(builder.task_running_queue) == config["maxImageRequest"]["max_num"]:
            raise HTTPException(
                status_code=400,
                detail="Image is already in building status. Please retry after complete the current image build.",
            )
        prepared_data = await handle_request(db, current_user, user_input)
        data = jsonable_encoder(prepared_data)
        asyncio.create_task(builder.build(db, data))
        return {
            "status_code": 200,
            "detail": "Image build request. If you want to know process status, please check out Tasks tab and see LOGS",
        }
    except HTTPException as http_exception:
        log.exception(http_exception)
        return HTTPException(
            status_code=400,
            detail="Image is already in building status. Please retry after complete the current image build.",
        )
    except Exception as e:
        log.exception(e)
        return e


@app.post("/build/submit/preset/", name="build preset image")
async def build_preset_image(
    user_input: Request,
    db: Session = Depends(get_db),
):
    try:
        if len(builder.task_running_queue) == config["maxImageRequest"]["max_num"]:
            raise HTTPException(
                status_code=400,
                detail="Image is already in building status. Please retry after complete the current image build.",
            )
        user_input_data = ast.literal_eval(jsonable_encoder(await user_input.body()))
        current_user = user_input_data["user"]
        prepared_data = await handle_request(
            db, current_user, user_input_data
        )
        data = jsonable_encoder(prepared_data)
        asyncio.create_task(builder.build(db, data, current_user))
        return {
            "detail": "Image build request. If you want to know process status, please check out Tasks tab and see LOGS",
        }
    except HTTPException as http_exception:
        log.exception(http_exception)
        return HTTPException(
            status_code=400,
            detail="Image is already in building status. Please retry after complete the current image build.",
        )
    except Exception as e:
        log.exception(e)
        return e


@app.get("/build/stream_log/{task_id}/", name="real_time_log")
async def real_time_log_streaming(
    request: Request,
    task_id: str,
    db: Session = Depends(get_db),
    current_user: models.Users = Depends(get_current_user),
):
    stream = generate_event_source(task_id)
    return EventSourceResponse(stream)


@app.get("/user/my_task/{status}/", name="all of my task status")
async def get_task_status(
    status: schemas.TaskStatus,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    #  select status, building_at, finished_at from tasks where user_id=1 and status='complete';
    current_user_data = jsonable_encoder(current_user)
    user_tasks = crud.get_task_data_by_status(db, current_user_data["user_id"], status)
    return user_tasks


@app.get(
    "/user/my_task/requested_dockerfile/{task_id}/", name="specific task's dockerfile"
)
async def get_requested_dockerfile_contents(
    task_id: str,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user_data = jsonable_encoder(current_user)
    user_tasks = crud.get_specific_dockerfile_contents(
        db, current_user_data["user_id"], task_id
    )
    return {
        "requested_dockerfile_contents": user_tasks["requested_dockerfile_contents"]
    }


@app.get("/user/my_task/status/{task_id}/", name="specific task status")
async def get_task_status_by_task_id(
    task_id: str,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user_data = jsonable_encoder(current_user)
    user_tasks = crud.get_user_specific_tasks_status(
        db, current_user_data["user_id"], task_id
    )
    if user_tasks["status"] == "running":
        return {
            "status_code": 200,
            "detail": "Image build process still working.",
            "requested_info": user_tasks["requested_info"],
        }
    elif user_tasks["status"] == "complete" or user_tasks["status"] == "error":
        return {
            "status_code": 200,
            "building_status": user_tasks["status"],
            "requested_info": user_tasks["requested_info"],
        }
    elif user_tasks["status"] == "pending":
        return {
            "status_code": 200,
            "detail": "Please wait until other request is over.",
            "requested_info": user_tasks["requested_info"],
        }
    else:
        return {
            "status_code": 400,
            "detail": "Please build a image first.",
            "requested_info": user_tasks["requested_info"],
        }


@app.get("/user/my_task/log_result/{task_id}/", name="specific task build result")
async def get_task_result(
    task_id: str,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user_data = jsonable_encoder(current_user)
    result = crud.get_task_result(db, current_user_data["user_id"], task_id)
    try:
        return {
            "status_code": 200,
            "detail": result,
        }
    except BuildLogEmptyError as e:
        log.exception(e)
        return {
            "status_code": 400,
            "detail": repr(e),
        }


@app.get("/user/my_task/all/tasks/", name="all of my tasks")
async def get_current_user_task(
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user_data = jsonable_encoder(current_user)
    user_tasks = crud.get_current_user_task(db, current_user_data["user_id"])
    return user_tasks


@app.get("/user/my_task/id/task/", name="request to get all of my tasks")
async def get_current_user_task_id(
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    current_user_data = jsonable_encoder(current_user)
    user_task_id = crud.get_current_user_task_id(db, current_user_data["user_id"])
    return user_task_id


@app.post("/build/dockerfile/preview/", name="show dockerfile")
async def complete_dockerfile(
    user_input: schemas.TaskBody, token: str = Depends(oauth2_scheme)
):
    try:
        dockerfile_contents = await _get_completed_dockerfile(user_input)
        global DOCKERFILE_CONTENTS
        DOCKERFILE_CONTENTS = dockerfile_contents
        return dockerfile_contents
    except Exception as e:
        log.exception(e)
        raise


@app.get("/build/dockerfile/download/", name="download dockerfile")
async def download_dockerfile(token: str = Depends(oauth2_scheme)):
    try:
        dockerfile_contents = DOCKERFILE_CONTENTS
        f = BytesIO(bytes(dockerfile_contents, encoding="utf-8"))
        response = StreamingResponse(f, media_type="text/json")
        response.headers["Content-Disposition"] = "attachment; filename=Dockerfile"
        return response
    except Exception as e:
        log.exception(e)
        raise


@app.get("/download_log_file/{task_id}/", name="download build log file")
async def download_log_file(
    task_id: str,
    current_user: models.Users = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        current_user_data = jsonable_encoder(current_user)
        result = json.dumps(
            jsonable_encoder(
                crud.get_task_result(db, current_user_data["user_id"], task_id)
            )
        )
        f = BytesIO(bytes(result, encoding="utf-8"))
        response = StreamingResponse(f, media_type="text/json")
        response.headers["Content-Disposition"] = "attachment; filename=log.json"
        return response
    except Exception as e:
        log.exception(e)
        raise


@app.post("/register/", name="Sign up")
async def user_register(reg_info: schemas.UserRegister, db: Session = Depends(get_db)):
    email_existance = await _is_email_exist(reg_info.email, db)
    if not reg_info.email or not reg_info.password:
        return JSONResponse(
            status_code=400, content=dict(msg="Email and PW must be provided'")
        )
    if email_existance:
        # email is exist, no need to register
        return JSONResponse(status_code=400, content=dict(msg="EMAIL_EXISTS"))
    # email is not exist, need to register
    hash_pw = bcrypt.hashpw(reg_info.password.encode("utf-8"), bcrypt.gensalt())
    new_user = crud.create_user(db, user_pw=hash_pw, user_email=reg_info.email)
    access_token_expires = timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": new_user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/login/access-token/", name="Sign in")
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = crud.get_user_info_for_authenticate(
        db=db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/build/image/list/", name="get host image list")
async def get_image_list(request: Request, token: str = Depends(oauth2_scheme)):
    try:
        img_list = await builder.show_image_list()
        return img_list
    except Exception as e:
        log.exception(e)
        raise


@app.get("/build/image/info/", name="get specific image infomation")
async def image_info(target_image_name: str, token: str = Depends(oauth2_scheme)):
    try:
        info: dict
        info = await builder.show_image_info(target_image_name)
        return info
    except Exception as e:
        log.exception(e)
        raise


@app.delete("/image/", name="Delete image")
async def clear_image(
    target_image_name, force=False, noprune=False, token: str = Depends(oauth2_scheme)
):
    try:
        deleted_img = await builder.delete_image(target_image_name, force, noprune)
        return deleted_img
    except Exception as e:
        log.exception(e)
        raise


@app.post("/container/commit/", name="Commit container")
async def commit_container(
    params: schemas.CommitParams, token: str = Depends(oauth2_scheme)
):
    params = jsonable_encoder(params)
    print(params)
    try:
        await docker_commit(params)
        return {"status_code": 200, "detail": "Commit process is completed."}
    except Exception as e:
        log.exception(e)
        return {"status_code": 400, "detail": e}


@app.get("/container/list/", name="Show host running container list")
async def container_list(token: str = Depends(oauth2_scheme)):
    try:
        container_list = await docker_container_list()
        # print(container_list)
        return {"status_code": 200, "detail": container_list}
    except Exception as e:
        log.exception(e)
        return {"status_code": 400, "detail": e}


async def _get_completed_dockerfile(user_input):
    dockerfile_contents = await builder.make_dockerfile(user_input)
    return dockerfile_contents


async def _is_email_exist(email: str, db: Session = Depends(get_db)):
    chk = crud.get_user_email(db, email)
    if chk:
        return True
    return False


async def handle_request(db, current_user, user_input):
    current_user_data = jsonable_encoder(current_user)
    user_input_data = jsonable_encoder(user_input)
    dockerfile_contents = await _get_completed_dockerfile(user_input_data)
    global DOCKERFILE_CONTENTS
    DOCKERFILE_CONTENTS = dockerfile_contents
    task_data = crud.create_task(
        db, user_input_data, dockerfile_contents, current_user_data["user_id"]
    )
    log.debug(f"pending queue : {len(builder.task_pending_queue)}")
    log.debug(f"running queue : {len(builder.task_running_queue)}")
    return task_data
