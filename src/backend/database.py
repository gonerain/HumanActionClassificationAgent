from __future__ import annotations

"""PostgreSQL 14 + TimescaleDB storage for cameras and dwell events.

仅支持 PostgreSQL；通过环境变量 `SP_DB_URL` 指定连接串：
  postgresql+psycopg2://user:pass@host:5432/dbname
如果 TimescaleDB 可用，会在 `dwell_events.start_ts` 上创建 hypertable（幂等）。
"""

import os
from typing import Iterator

from contextlib import contextmanager

from sqlalchemy import Float, Integer, String, create_engine, DateTime, text, inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session, validates
from datetime import datetime, timezone
import threading
import queue


# PostgreSQL connection URL
DB_URL = os.getenv("SP_DB_URL", "postgresql+psycopg2://postgres:000815@localhost:5432/sglz")

engine = create_engine(DB_URL, future=True, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class DwellEvent(Base):
    """Single worker dwell event with optional video evidence."""

    __tablename__ = "dwell_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_id: Mapped[str] = mapped_column(String)
    camera_id: Mapped[int] = mapped_column(Integer, nullable=True)
    # Use timestamptz for Timescale hypertable when on PostgreSQL
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    video_path: Mapped[str] = mapped_column(String)

    # Accept flexible inputs for timestamps; normalize to tz-aware UTC datetimes
    @staticmethod
    def _to_dt(value: object) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            # best-effort ISO 8601 parse
            try:
                from datetime import datetime as _dt

                dt = _dt.fromisoformat(value)  # Python 3.11+ supports Z/offset
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
        raise TypeError(f"Unsupported timestamp type for DwellEvent: {type(value)!r}")

    @validates("start_ts", "end_ts")
    def _validate_ts(self, key: str, value: object) -> datetime:  # type: ignore[override]
        return self._to_dt(value)


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
        engine = create_engine(url, future=True, pool_pre_ping=True)
        SessionLocal.configure(bind=engine)

    # Set UTC timezone on the session/connection (best-effort)
    with engine.begin() as conn:
        try:
            conn.execute(text("SET TIME ZONE 'UTC'"))
        except Exception:
            pass

    # DDL
    Base.metadata.create_all(engine)

    # Lightweight, idempotent migrations for existing installs
    try:
        _ensure_schema(engine)
    except Exception:
        # schema check is best-effort; continue startup
        pass

    # Enable TimescaleDB and create hypertable (idempotent)
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.execute(text("SELECT create_hypertable('dwell_events','start_ts', if_not_exists => TRUE)"))
        except Exception:
            # Allow running without Timescale extension (plain PostgreSQL)
            pass

    _start_db_worker()


def _ensure_schema(engine) -> None:
    """Ensure backward-compatible columns/indexes exist (idempotent)."""
    insp = inspect(engine)
    tables = insp.get_table_names()
    if "dwell_events" not in tables:
        return
    cols = {c["name"] for c in insp.get_columns("dwell_events")}
    stmts: list[str] = []
    if "camera_id" not in cols:
        # nullable to keep compatibility; apps can backfill
        stmts.append('ALTER TABLE "dwell_events" ADD COLUMN "camera_id" INTEGER NULL')
    # helpful composite index for common queries
    stmts.append('CREATE INDEX IF NOT EXISTS "idx_dwell_camera_start" ON "dwell_events" ("camera_id", "start_ts")')
    if stmts:
        with engine.begin() as conn:
            for sql in stmts:
                try:
                    conn.execute(text(sql))
                except Exception:
                    # ignore if not supported by dialect
                    pass

@contextmanager
def get_session() -> Iterator[Session]:
    """Provide a transactional scope with rollback on failure."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Async DB writer to avoid blocking critical loops

_db_queue: "queue.Queue[tuple[str, dict]]" | None = None
_db_thread: threading.Thread | None = None
_stop_flag = threading.Event()


def _db_loop() -> None:  # pragma: no cover - background worker
    assert _db_queue is not None
    while not _stop_flag.is_set():
        try:
            task, payload = _db_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if task == "insert_dwell_event":
                with get_session() as sess:
                    evt = DwellEvent(**payload)
                    sess.add(evt)
            # add more task types as needed
        except Exception:
            # swallow errors to keep worker alive; logging can be added
            pass
        finally:
            try:
                _db_queue.task_done()
            except Exception:
                pass


def _start_db_worker() -> None:
    global _db_queue, _db_thread
    if _db_thread is not None and _db_thread.is_alive():
        return
    _db_queue = queue.Queue()
    _stop_flag.clear()
    t = threading.Thread(target=_db_loop, name="db-writer", daemon=True)
    t.start()
    _db_thread = t


def enqueue_dwell_event(
    *,
    object_id: str,
    camera_id: int | None,
    start_ts: datetime,
    end_ts: datetime,
    video_path: str,
) -> None:
    """Queue a dwell event insert to avoid blocking the caller."""

    if _db_queue is None:
        _start_db_worker()
    assert _db_queue is not None
    payload = dict(
        object_id=object_id,
        camera_id=camera_id,
        start_ts=start_ts,
        end_ts=end_ts,
        video_path=video_path,
    )
    try:
        _db_queue.put_nowait(("insert_dwell_event", payload))
    except Exception:
        # best-effort; if queue full or other issue, drop to avoid blocking
        pass


def stop_db_worker(timeout: float = 2.0) -> None:
    """Signal DB worker to stop and wait briefly."""
    global _db_thread, _db_queue
    try:
        _stop_flag.set()
        if _db_thread is not None:
            _db_thread.join(timeout=timeout)
    finally:
        _db_thread = None
        _db_queue = None
