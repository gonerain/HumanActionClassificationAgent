from __future__ import annotations

"""SQLite storage for cameras and dwell events."""

import os
from typing import Iterator

from contextlib import contextmanager

from sqlalchemy import Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


DB_URL = os.getenv("SP_DB_URL", "sqlite:///dwell_events.db")

engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class DwellEvent(Base):
    """Single worker dwell event with optional video evidence."""

    __tablename__ = "dwell_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_id: Mapped[str] = mapped_column(String)
    start_ts: Mapped[float] = mapped_column(Float, nullable=False)
    end_ts: Mapped[float] = mapped_column(Float, nullable=False)
    video_path: Mapped[str] = mapped_column(String)


class Camera(Base):
    """Camera configuration with source and ROI polygon.

    ``region_json`` stores a JSON-serialized list of points [[x,y], ...].
    """

    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    region_json: Mapped[str] = mapped_column(String, default="[]")


def init_db(url: str | None = None) -> None:
    """Initialize database and create tables.

    Args:
        url: Optional database URL to override ``DB_URL``.
    """

    global engine, SessionLocal
    if url is not None:
        engine = create_engine(url, future=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
