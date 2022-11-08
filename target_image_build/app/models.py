"""
Base 객체를 상속받아서 SQLAlchemy models을 생성할 것이다.
SQLAlchemy에서 model 이라는 용어를 사용하는데 이는
데이터베이스와 상호작용하는 클래스 및 인스턴스를 의미한다.
Pydantic 또한 model이라는 용어를 사용하는데,
data validation, conversion, documentaion 클래스 혹은 인스턴스를 의미한다.
"""
from sqlalchemy import Boolean, Column, Enum, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import TextPickleType
from .database import Base


class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True, index=True)
    status = Column(String, index=True, default="pending")
    created_at = Column(String, index=True)
    building_at = Column(String, index=True)
    pushing_at = Column(String, index=True)
    finished_at = Column(String, index=True)
    requested_info = Column(TextPickleType(), nullable=True)
    requested_image = Column(String, nullable=True)
    requested_target_img = Column(String, nullable=True)
    requested_labels = Column(TextPickleType(), nullable=True)
    requested_env_commands = Column(TextPickleType(), nullable=True)
    requested_custom_pkg_commands = Column(TextPickleType(), nullable=True)
    requested_dockerfile_contents = Column(String, index=True)
    requested_auto_push = Column(Boolean, nullable=False)
    logs = Column(TextPickleType(), nullable=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)


class Users(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(length=255), nullable=True)
    pw = Column(String(length=2000), nullable=True)
    status = Column(Enum("active", "deleted", "blocked"), default="active")
    name = Column(String(length=255), nullable=True)
    phone_number = Column(String(length=20), nullable=True, unique=True)
    keys = relationship("ApiKeys", back_populates="users")


class ApiKeys(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    access_key = Column(String(length=64), nullable=False, index=True)
    secret_key = Column(String(length=64), nullable=False)
    user_memo = Column(String(length=40), nullable=True)
    status = Column(Enum("active", "stopped", "deleted"), default="active")
    is_whitelisted = Column(Boolean, default=False)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    users = relationship("Users", back_populates="keys")
