# Phase: database connection | Input: configured database URL | Output: SQLAlchemy engine and sessions | Command: ``uv run python -m src.database.run_database --help``.
import sqlite3
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from .config import PROJECT_ROOT, get_database_url
from .models import Base
from .migrations import allow_missing_roll_numbers


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
    (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
    allow_missing_roll_numbers(engine)
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()