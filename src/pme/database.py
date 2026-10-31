import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .config import DATA_DIR, get_database_url
from .models import Base


@event.listens_for(Engine, "connect")
def enable_sqlite_foreign_keys(connection, _):
    if isinstance(connection, sqlite3.Connection):
        connection.execute("PRAGMA foreign_keys=ON")


DATABASE_URL = get_database_url()

connect_args = (
    {"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {}
)

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_pre_ping=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()