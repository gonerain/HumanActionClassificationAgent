"""Minimal backend service for scene presence monitoring."""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from scene_presence import ScenePresenceManager

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

    def __init__(self, model_name: str = "yolo11s", conf: float = 0.5) -> None:
        self.manager = ScenePresenceManager()
        self.conf = conf
        self.detector = YOLO(model_name) if YOLO is not None else None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run detection and draw status on ``frame``."""

        detections: List[Tuple[int, Tuple[int, int, int, int]]] = []
        if self.detector is not None:
            results = self.detector.track(frame, conf=self.conf, persist=True)
            boxes = results[0].boxes
            ids = boxes.id if hasattr(boxes, "id") else None
            if ids is not None:
                for box, obj_id in zip(boxes.xyxy, ids):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    detections.append((int(obj_id), (x1, y1, x2, y2)))
        self.manager.update(detections)
        self.manager.draw(frame)
        return frame

    def state(self) -> Dict[str, object]:
        """Return current scene state."""

        active_ids = [oid for oid, st in self.manager.workers.items() if st.status == "active"]
        return {
            "active_ids": active_ids,
            "scene_active": self.manager.is_scene_active(),
        }


# ---------------------------------------------------------------------------
# FastAPI application


capture_worker = VideoCaptureWorker()
capture_worker.start()
processor = FrameProcessor()
app = FastAPI()


@app.get("/inference/report")
def inference_report() -> Dict[str, object]:
    """Return scene presence report."""

    frame = capture_worker.latest_frame
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

    frame = capture_worker.latest_frame
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


@app.websocket("/status")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Stream processed frames and status to clients."""

    await ws.accept()
    try:
        while True:
            frame = capture_worker.latest_frame
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
