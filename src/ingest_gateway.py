"""Ingest-Gateway for video stream access.

This module implements a minimal version of the `Ingest-Gateway` described in
``agents.md``. It registers RTSP (or file) sources, spawns background workers
that read frames using ``cv2.CAP_FFMPEG`` and emits frames to a placeholder
publisher. Each worker exposes health statistics such as estimated FPS and last
frame timestamp which are available through an HTTP API.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:  # pragma: no cover - optional dependency at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


# ---------------------------------------------------------------------------
# Camera registration model


class CameraConfig(BaseModel):
    """Configuration for a camera stream."""

    id: str
    scene_id: str
    rtsp_url: str
    fps_target: int = 10


# ---------------------------------------------------------------------------
# Stream worker


class StreamWorker(threading.Thread):
    """Grab frames from a video source with reconnect and down-sampling."""

    def __init__(
        self,
        camera_id: str,
        source: str,
        fps_target: int,
        publisher: Optional[Callable[[str, float, np.ndarray], None]] = None,
        reconnect_interval: float = 2.0,
    ) -> None:
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.fps_target = fps_target
        self.publisher = publisher or (lambda _cid, _ts, _frm: None)
        self.reconnect_interval = reconnect_interval
        self._running = True
        self._cap = None
        self.last_ts: Optional[float] = None
        self.fps_est: float = 0.0

    # -- helper -------------------------------------------------------------
    def _open(self):
        if cv2 is None:
            raise RuntimeError("cv2 is required for video capture")
        self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    # -- main loop ----------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - background loop
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                try:
                    self._open()
                except Exception:
                    time.sleep(self.reconnect_interval)
                    continue
                if not self._cap.isOpened():
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self._cap.read()
            ts = time.time()
            if ret:
                self.publisher(self.camera_id, ts, frame)
                if self.last_ts is not None:
                    dt = ts - self.last_ts
                    if dt > 0:
                        self.fps_est = 1.0 / dt
                self.last_ts = ts
                if self.fps_target > 0:
                    delay = max(0.0, (1.0 / self.fps_target) - (time.time() - ts))
                    time.sleep(delay)
            else:
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_interval)

        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# FastAPI application


app = FastAPI(title="IngestGateway")

cameras: Dict[str, CameraConfig] = {}
workers: Dict[str, StreamWorker] = {}


def publish_frame(camera_id: str, ts: float, frame: np.ndarray) -> None:  # pragma: no cover - placeholder
    """Placeholder publisher for frames.

    In production this would push the frame to a message broker topic named
    ``video.frames.{camera_id}``.
    """


@app.post("/cameras")
def register_camera(cfg: CameraConfig) -> Dict[str, str]:
    """Register or update a camera stream."""

    # stop existing worker if present
    if cfg.id in workers:
        workers[cfg.id].stop()
        workers[cfg.id].join(timeout=1)

    cameras[cfg.id] = cfg
    worker = StreamWorker(cfg.id, cfg.rtsp_url, cfg.fps_target, publisher=publish_frame)
    worker.start()
    workers[cfg.id] = worker
    return {"status": "ok"}


@app.get("/cameras/{camera_id}/health")
def camera_health(camera_id: str) -> Dict[str, object]:
    """Return health statistics of a camera."""

    worker = workers.get(camera_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return {
        "id": camera_id,
        "running": worker.is_running(),
        "fps_est": worker.fps_est,
        "last_ts": worker.last_ts,
    }
