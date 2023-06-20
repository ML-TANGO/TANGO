import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from . import auth, models
from .utils import create_logger

log = create_logger("image.builder.crud")


def get_user_email(db: Session, email):
    chk_email = email
    try:
        result = db.query(models.Users).filter(models.Users.email == chk_email).first()
        if not result:
            return False
        if chk_email == result.email:
            return True
    except Exception as e:
        log.exception(e)
        raise


def get_user_info_for_authenticate(db: Session, email, password):
    try:
        result = db.query(models.Users).filter(models.Users.email == email).first()
        if not result:
            return None
        if not auth.verify_password(password, result.pw):
            return None
        return result
    except Exception as e:
        log.exception(e)
        raise


def get_user_info(db: Session, email):
    try:
        return db.query(models.Users).filter(models.Users.email == email).first()
    except Exception as e:
        log.exception(e)
        raise


def get_current_user_task(db: Session, current_user_id):
    try:
        return (
            db.query(models.Task).filter(models.Task.user_id == current_user_id).all()
        )
    except Exception as e:
        log.exception(e)
        raise


def get_current_user_task_id(db: Session, current_user_id):
    try:
        return (
            db.query(models.Task.task_id, models.Task.created_at)
            .filter(models.Task.user_id == current_user_id)
            .order_by(models.Task.created_at.desc())
            .all()
        )
    except Exception as e:
        log.exception(e)
        raise


def get_specific_dockerfile_contents(db: Session, user_id, task_id):
    try:
        return (
            db.query(models.Task.requested_dockerfile_contents)
            .filter(models.Task.user_id == user_id, models.Task.task_id == task_id)
            .first()
        )
    except Exception as e:
        log.exception(e)
        raise


def get_user_specific_tasks_status(db: Session, user_id, task_id):
    try:
        return (
            db.query(models.Task.status, models.Task.requested_info)
            .filter(models.Task.user_id == user_id, models.Task.task_id == task_id)
            .first()
        )
    except Exception as e:
        log.exception(e)
        raise


def get_task_data_by_status(db: Session, user_id, status):
    try:
        return (
            db.query(
                models.Task.task_id,
                models.Task.requested_image,
                models.Task.requested_target_img,
                models.Task.created_at,
            )
            .filter(
                models.Task.user_id == user_id,
                models.Task.status == status,
            )
            .all()
        )
    except Exception as e:
        log.exception(e)
        raise


def get_task_result(db: Session, user_id, task_id):
    try:
        return (
            db.query(
                models.Task.logs,
            )
            .filter(models.Task.user_id == user_id, models.Task.task_id == task_id)
            .first()
        )
    except Exception as e:
        log.exception(e)
        raise


def create_task(db: Session, user_input, dockerfile_contents, current_user_id):
    new_task_id = uuid.uuid4()
    try:
        db_task = models.Task(
            task_id=str(new_task_id),
            status="pending",
            created_at=datetime.now().strftime(
                "%Y-%m-%d-%H:%M:%S"
            ),  # 유저가 request를 날려 DB에 Task가 생성된 시간
            building_at=None,  # 실제 빌드에 돌입한 시간
            pushing_at=None,  # 레지스트리에 Push 한 시간
            finished_at=None,  # 빌드가 끝난 시간
            requested_info=user_input,
            requested_image=user_input["build"]["os"],
            requested_target_img=user_input["build"]["target_name"],
            requested_labels=None,
            # requested_env_commands=user_input["deployment"]["build"]["components"]["custom_packages"]["environments"],
            requested_custom_pkg_commands=user_input["build"]["components"]["custom_packages"],
            requested_dockerfile_contents=dockerfile_contents,
            requested_auto_push=False,
            user_id=current_user_id,
            logs=None,
        )
        db.add(db_task)  # db_task를 database session에다가 추가
        db.commit()  # 추가한(변경된) db_task를 저장
        db.refresh(db_task)  # 인스턴스 새로 고침
        return db_task
    except Exception as e:
        log.exception(e)
        raise


def create_preset_task(db: Session, user_input, dockerfile_contents, current_user_id):
    new_task_id = uuid.uuid4()
    try:
        db_task = models.Task(
            task_id=str(new_task_id),
            status="pending",
            created_at=datetime.now().strftime(
                "%Y-%m-%d-%H:%M:%S"
            ),  # 유저가 request를 날려 DB에 Task가 생성된 시간
            building_at=None,  # 실제 빌드에 돌입한 시간
            pushing_at=None,  # 레지스트리에 Push 한 시간
            finished_at=None,  # 빌드가 끝난 시간
            requested_info=user_input,
            requested_image=user_input[
                "preset"
            ],  # TODO unify parameter user_input['src']
            requested_target_img=user_input["target"],
            requested_labels=None,
            requested_env_commands=None,
            requested_custom_pkg_commands=None,
            requested_dockerfile_contents=dockerfile_contents,
            requested_auto_push=user_input["auto_push"],
            user_id=current_user_id,
            logs=None,
        )
        db.add(db_task)  # db_task를 database session에다가 추가
        db.commit()  # 추가한(변경된) db_task를 저장
        db.refresh(db_task)  # 인스턴스 새로 고침
        return db_task
    except Exception as e:
        log.exception(e)
        raise


def create_user(db: Session, user_pw, user_email):
    try:
        db_task = models.Users(
            email=user_email,
            pw=user_pw,
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except Exception as e:
        log.exception(e)
        raise


def modify_task(db: Session, task_id, build_time, finished_time, status, logs):
    try:
        model = db.query(models.Task).filter(models.Task.task_id == task_id).first()
        model.building_at = build_time
        model.finished_at = finished_time
        model.status = status
        model.logs = logs
        db.commit()  # 추가한(변경된) db_task를 저장
    except Exception as e:
        log.exception(e)
        raise


def modify_fileupload_task(db: Session, user_input, task_id, time, status):
    try:
        model = db.query(models.Task).filter(models.Task.task_id == task_id).first()
        model.building_at = time[0]
        model.finished_at = time[1]
        model.status = status
        model.requested_image = user_input.target_name
        model.requested_labels = user_input.labels
        db.commit()
    except Exception as e:
        log.exception(e)
        raise


def modify_tasks_status(
    db: Session, task_id: str, status: str
):  # TODO status 받아서 각 상황에 맞는 status로 변경
    try:
        model = db.query(models.Task).filter(models.Task.task_id == task_id).first()
        model.status = status
        db.commit()
    except Exception as e:
        log.exception(e)
        raise


def modify_tasks_result(db: Session, task_id: str, log_result):
    try:
        model = db.query(models.Task).filter(models.Task.task_id == task_id).first()
        model.logs = log_result
    except Exception as e:
        log.exception(e)
        raise
