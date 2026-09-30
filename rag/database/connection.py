from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import sessionmaker


DATABASE_DIRECTORY = Path(
    "data/database"
)

DATABASE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = (
    DATABASE_DIRECTORY / "chat.db"
)

DATABASE_URL = (
    f"sqlite:///{DATABASE_PATH.as_posix()}"
)


class Base(DeclarativeBase):
    pass


engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,
    },
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def create_database() -> None:
    """
    Create all registered database tables.

    Importing models here guarantees that SQLAlchemy
    registers them before Base.metadata.create_all().
    """

    from rag.database import models  # noqa: F401

    Base.metadata.create_all(
        bind=engine,
    )


def get_database_session():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        