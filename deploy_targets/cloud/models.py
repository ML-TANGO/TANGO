from sqlalchemy import Column, String

from database import Base


class Task(Base):
    __tablename__ = "tasks"

    user_id = Column(String(length=255), nullable=True)
    project_id = Column(String(length=255), nullable=True, primary_key=True)
    container_name = Column(String(length=255), nullable=True)
    container_id = Column(String(length=255), nullable=True)
    status = Column(String(length=255), nullable=False)
