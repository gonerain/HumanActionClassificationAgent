from __future__ import annotations

"""Lightweight SQLite storage for dwell events."""

import os
from typing import Iterator

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


def get_session() -> Iterator:
    """Yield a SQLAlchemy session (context manager)."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

