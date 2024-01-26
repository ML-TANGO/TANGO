from contextlib import contextmanager

import sqlmodel
from sqlmodel import SQLModel, create_engine  # noqa to load SQLModel


sqlite_file_name = "/source/cmgr_sql.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


# #  Dependencies
# def get_db_session():
#     with sqlmodel.Session(engine) as db_session:
#         yield db_session
# DBSessionDepends = Annotated[sqlmodel.Session, Depends(get_db_session)]


@contextmanager
def get_db_session():
    with sqlmodel.Session(engine) as db:
        yield db
