"""Minimal backend service for scene presence monitoring."""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import time

from scene_presence import ScenePresenceManager

from .config import CONFIG_FILE, load_config, save_config

try:  # pragma: no cover - optional runtime dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


# ---------------------------------------------------------------------------
# Placeholder for secondary-system integration

def get_current_roll_status() -> str:
    """Return current steel roll status.

    TODO: Replace with real implementation (MES / PLC / HTTP).
    """

    return "ready"  # possible values: "empty", "ready", "working", "done"


# ---------------------------------------------------------------------------
# Video capture worker (frame drop strategy)


class VideoCaptureWorker(threading.Thread):
    """Continuously grab frames from a video source."""

    def __init__(self, source: int | str = 0) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open {source}")
        self.latest_frame: np.ndarray | None = None
        self._running = True

    def run(self) -> None:  # pragma: no cover - background loop
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                break
        self.cap.release()

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Frame processor using ScenePresenceManager


class FrameProcessor:
    """Apply detection and update presence state."""

    def __init__(
        self, model_name: str = "yolo11s", conf: float = 0.5, region: List[Tuple[int, int]] | None = None
    ) -> None:
        self.manager = ScenePresenceManager(region=region)
        self.conf = conf
        self.detector = YOLO(model_name) if YOLO is not None else None
        self._clock = lambda: time.time() * 1000.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run detection and draw status on ``frame``."""

        detections: List[Tuple[int, Tuple[int, int, int, int]]] = []
        if self.detector is not None:
            results = self.detector.track(frame, conf=self.conf, persist=True)
            boxes = results[0].boxes
            ids = boxes.id if hasattr(boxes, "id") else None
            classes = boxes.cls if hasattr(boxes, "cls") else None
            if ids is not None and classes is not None:
                names = self.detector.names  # dict: {class_id: "name"}
                for box, obj_id, cls_id in zip(boxes.xyxy, ids, classes):
                    class_name = names.get(int(cls_id), str(cls_id))
                    if class_name != "person":
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    detections.append((int(obj_id), (x1, y1, x2, y2)))
        now_ms = self._clock()

        self.manager.update(detections, now_ms)
        self.manager.draw(frame)
        return frame

    def state(self) -> Dict[str, object]:
        """Return current scene state."""

        active_ids = [oid for oid, st in self.manager.workers.items() if st.status == "active"]
        return {
            "active_ids": active_ids,
            "scene_active": self.manager.is_scene_active(),
            "timestamp_ms": self._clock(),
        }


# ---------------------------------------------------------------------------
# FastAPI application


config = load_config(CONFIG_FILE)

capture_worker: VideoCaptureWorker | None = None


def set_source(source: int | str | None) -> None:
    """(Re)start the video capture worker with ``source``.

    When ``source`` is ``None`` or the stream fails to open, ``capture_worker``
    becomes ``None`` and the backend continues running without frames.
    """

    global capture_worker
    if capture_worker is not None:
        capture_worker.stop()
        capture_worker = None
    if source is not None:
        try:
            worker = VideoCaptureWorker(source)
        except Exception:
            capture_worker = None
        else:
            worker.start()
            capture_worker = worker
    config["source"] = source


set_source(config.get("source"))
processor = FrameProcessor(region=config.get("region"))
app = FastAPI()


@app.get("/inference/report")
def inference_report() -> Dict[str, object]:
    """Return scene presence report."""

    frame = capture_worker.latest_frame if capture_worker else None
    if frame is None:
        return {"detail": "no frame"}
    processor.process(frame.copy())
    data = processor.state()
    data["roll_status"] = get_current_roll_status()
    return data


@app.get("/")
def index() -> HTMLResponse:
    """Serve a minimal frontend to view camera feed and results."""

    root = Path(__file__).resolve().parents[2]
    html_path = root / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/snapshot")
def snapshot() -> Dict[str, object]:
    """Return one processed frame and its recognition results."""

    frame = capture_worker.latest_frame if capture_worker else None
    if frame is None:
        return {"detail": "no frame"}
    processed = processor.process(frame.copy())
    ret, buf = cv2.imencode(".jpg", processed)
    if not ret:
        return {"detail": "encode_failed"}
    b64 = base64.b64encode(buf).decode("ascii")
    payload = processor.state()
    payload["roll_status"] = get_current_roll_status()
    payload["frame"] = b64
    return payload


@app.post("/config")
def update_config(payload: Dict[str, Any], save: bool = False) -> Dict[str, str]:
    """Update runtime configuration.

    ``payload`` may contain ``source`` and ``region``. When ``save`` is ``True``
    the updated configuration is persisted to disk. By default, the config file
    is untouched to avoid pollution.
    """

    if "region" in payload and payload["region"] is not None:
        region = [tuple(map(int, pt)) for pt in payload["region"]]
        processor.manager.set_region(region)
        config["region"] = region
    if "source" in payload:
        set_source(payload["source"])
    if save:
        save_config(config, CONFIG_FILE)
    return {"detail": "updated"}


@app.websocket("/status")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Stream processed frames and status to clients."""

    await ws.accept()
    try:
        while True:
            frame = capture_worker.latest_frame if capture_worker else None
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            processed = processor.process(frame.copy())
            ret, buf = cv2.imencode(".jpg", processed)
            if not ret:
                await asyncio.sleep(0.1)
                continue
            b64 = base64.b64encode(buf).decode("ascii")
            payload = processor.state()
            payload["roll_status"] = get_current_roll_status()
            payload["frame"] = b64
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# The module exposes ``app`` for ASGI servers like uvicorn.
