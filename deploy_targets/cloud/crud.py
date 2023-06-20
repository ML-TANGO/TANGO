from sqlalchemy.orm import Session

import models


def get_container_name(db: Session, user_input):
    try:
        return (
            db.query(models.Task.container_name)
            .filter(models.Task.user_id == user_input["user_id"], models.Task.project_id == user_input["project_id"])
            .first()
        )
    except Exception as e:
        print(e)
        raise


def get_container_id(db: Session, user_input):
    try:
        return (
            db.query(models.Task.container_id)
            .filter(models.Task.user_id == user_input["user_id"], models.Task.project_id == user_input["project_id"])
            .first()
        )
    except Exception as e:
        print(e)
        raise


def modify_container_id(db: Session, project_id, container_id):
    try:
        model = db.query(models.Task).filter(models.Task.project_id == project_id).first()
        model.container_id = container_id
        db.commit()
    except Exception as e:
        print(e)
        raise


def modify_tasks_status(
    db: Session, user_data: dict, status: str
):
    try:
        model = db.query(models.Task).filter(models.Task.user_id == user_data["user_id"], models.Task.project_id == user_data["project_id"]).first()
        model.status = status
        db.commit()
    except Exception as e:
        print(e)
        raise


def get_user_specific_tasks_status(db: Session, user_data: dict):
    try:
        return (
            db.query(models.Task.status)
            .filter(models.Task.user_id == user_data["user_id"], models.Task.project_id == user_data["project_id"])
            .first()
        )
    except Exception as e:
        print(e)
        raise


def create_task(db: Session, user_input):
    try:
        db_task = models.Task(
            user_id=user_input["user_id"],
            project_id=user_input["project_id"],
            container_name=str(f'{user_input["user_id"]}-{user_input["project_id"]}'),
            container_id=None,
            status="started",
        )
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except Exception as e:
        print(e)
        raise
