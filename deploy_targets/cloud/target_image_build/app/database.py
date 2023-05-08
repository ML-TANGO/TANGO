from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQL_ALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"  # SQLite DB의 위치 ./sql_app.db

# SQLAlchemy 엔진 생성
# connect_args={"check_same_thread": False}는 오직 SQLite 사용할때만 필요
# SQLite는 기본적으로 싱글 스레드만으로 통신한다.
engine = create_engine(
    SQL_ALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 이 클래스를 상속하여서 데이터베이스 모델 혹은 클래스들을 각각 생성할 것이다.
Base = declarative_base()
