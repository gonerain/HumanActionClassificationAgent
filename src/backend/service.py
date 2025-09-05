"""Multi-camera backend service for scene presence monitoring.

This service manages multiple camera streams whose configuration (source, ROI)
is stored in the database. It exposes per-camera REST and WebSocket endpoints
for snapshots and status streaming.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Path as ApiPath
from fastapi.responses import HTMLResponse

import time

from scene_presence import ScenePresenceManager

from .database import DwellEvent, get_session, init_db, Camera

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
        self._last_ts = time.time() * 1000.0
        self.video_dir = Path("../recordings")
        self.video_dir.mkdir(exist_ok=True)
        self._recorders: Dict[int, Tuple[cv2.VideoWriter, float, Path]] = {}
        self.camera_label: Optional[str] = None

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

        # ScenePresenceManager expects current clock in ms; it keeps elapsed internally
        self.manager.update(detections, now_ms)

        # record videos for active workers
        active_ids = [oid for oid, st in self.manager.workers.items() if st.status == "active"]
        now_s = time.time()
        for oid in active_ids:
            writer = self._recorders.get(oid)
            if writer is None:
                prefix = f"{self.camera_label}_" if self.camera_label else ""
                path = self.video_dir / f"{prefix}{int(now_s*1000)}_{oid}.avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                vw = cv2.VideoWriter(str(path), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                self._recorders[oid] = (vw, now_s, path)
            self._recorders[oid][0].write(frame)

        for oid in list(self._recorders.keys()):
            if oid not in active_ids:
                vw, start_ts, path = self._recorders.pop(oid)
                vw.release()
                with get_session() as sess:
                    sess.add(
                        DwellEvent(
                            object_id=str(oid),
                            start_ts=start_ts,
                            end_ts=now_s,
                            video_path=str(path),
                        )
                    )
                    sess.commit()

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

    # ------------------------------------------------------------------
    def _clock(self) -> float:
        return time.time() * 1000.0


# ---------------------------------------------------------------------------
# FastAPI application


init_db()


class CameraRuntime:
    """Holds runtime objects for a camera stream."""

    def __init__(self, cam_id: int, source: str | int, region: Optional[List[Tuple[int, int]]], label: str):
        self.worker = VideoCaptureWorker(source)
        self.processor = FrameProcessor(region=region)
        self.processor.camera_label = label
        self.worker.start()

    def stop(self) -> None:
        self.worker.stop()


class CameraManager:
    """Manage multiple camera runtimes based on DB records."""

    def __init__(self) -> None:
        self.cameras: Dict[int, CameraRuntime] = {}

    def _parse_region(self, region_json: str) -> Optional[List[Tuple[int, int]]]:
        try:
            import json

            pts = json.loads(region_json or "[]")
            if not pts:
                return None
            return [tuple(map(int, p)) for p in pts]
        except Exception:
            return None

    @staticmethod
    def _source_to_vc_arg(source: str) -> str | int:
        # allow integer camera index or url/path
        try:
            return int(source)
        except ValueError:
            return source

    def ensure_running(self, cam: Camera) -> None:
        """Start or restart runtime for camera ``cam``."""
        region = self._parse_region(cam.region_json)
        vc_source = self._source_to_vc_arg(cam.source)

        # stop existing
        if cam.id in self.cameras:
            self.cameras[cam.id].stop()
            self.cameras.pop(cam.id, None)
        # start new
        try:
            runtime = CameraRuntime(cam.id, vc_source, region, label=f"cam{cam.id}")
        except Exception as e:
            # Failed to open; keep it absent but don't crash the app
            return
        self.cameras[cam.id] = runtime

    def stop(self, cam_id: int) -> None:
        rt = self.cameras.pop(cam_id, None)
        if rt is not None:
            rt.stop()

    def get_runtime(self, cam_id: int) -> Optional[CameraRuntime]:
        return self.cameras.get(cam_id)


manager = CameraManager()


def _load_all_cameras() -> None:
    with get_session() as sess:
        cams = sess.query(Camera).all()
        for cam in cams:
            manager.ensure_running(cam)


app = FastAPI()
_load_all_cameras()


@app.get("/cameras/{camera_id}/inference/report")
def inference_report(camera_id: int = ApiPath(..., ge=1)) -> Dict[str, object]:
    """Return scene presence report for a single camera."""

    rt = manager.get_runtime(camera_id)
    if rt is None:
        raise HTTPException(status_code=404, detail="camera not running or not found")
    frame = rt.worker.latest_frame
    if frame is None:
        return {"detail": "no frame"}
    rt.processor.process(frame.copy())
    data = rt.processor.state()
    data["roll_status"] = get_current_roll_status()
    return data


@app.get("/")
def index() -> HTMLResponse:
    """Serve a minimal frontend to view camera feed and results."""

    root = Path(__file__).resolve().parents[2]
    html_path = root / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/cameras/{camera_id}/snapshot")
def snapshot(camera_id: int = ApiPath(..., ge=1)) -> Dict[str, object]:
    """Return one processed frame and its recognition results for a camera."""

    rt = manager.get_runtime(camera_id)
    if rt is None:
        raise HTTPException(status_code=404, detail="camera not running or not found")
    frame = rt.worker.latest_frame
    if frame is None:
        return {"detail": "no frame"}
    processed = rt.processor.process(frame.copy())
    ret, buf = cv2.imencode(".jpg", processed)
    if not ret:
        return {"detail": "encode_failed"}
    b64 = base64.b64encode(buf).decode("ascii")
    payload = rt.processor.state()
    payload["roll_status"] = get_current_roll_status()
    payload["frame"] = b64
    return payload


@app.get("/dwell_events")
def dwell_events() -> List[Dict[str, object]]:
    """Return all recorded dwell events."""

    with get_session() as sess:
        events = sess.query(DwellEvent).order_by(DwellEvent.start_ts).all()
        return [
            {
                "id": e.id,
                "object_id": e.object_id,
                "start_ts": e.start_ts,
                "end_ts": e.end_ts,
                "video_path": e.video_path,
            }
            for e in events
        ]


@app.get("/cameras")
def list_cameras() -> List[Dict[str, Any]]:
    with get_session() as sess:
        cams = sess.query(Camera).all()
        out: List[Dict[str, Any]] = []
        for cam in cams:
            rt = manager.get_runtime(cam.id)
            out.append(
                {
                    "id": cam.id,
                    "name": cam.name,
                    "source": cam.source,
                    "region": cam.region_json,
                    "running": rt is not None,
                }
            )
        return out


@app.post("/cameras")
def create_camera(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name") or "camera")
    source = str(payload.get("source") or "0")
    region = payload.get("region")  # expected list of [x,y]
    import json

    region_json = json.dumps(region or [])
    with get_session() as sess:
        cam = Camera(name=name, source=source, region_json=region_json)
        sess.add(cam)
        sess.commit()
        sess.refresh(cam)
        manager.ensure_running(cam)
        return {"detail": "created", "id": cam.id}


@app.put("/cameras/{camera_id}")
def update_camera(camera_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    with get_session() as sess:
        cam = sess.query(Camera).get(camera_id)
        if cam is None:
            raise HTTPException(status_code=404, detail="camera not found")
        updated = False
        if "name" in payload and payload["name"] is not None:
            cam.name = str(payload["name"])
            updated = True
        if "source" in payload and payload["source"] is not None:
            cam.source = str(payload["source"])
            updated = True
        if "region" in payload:
            import json

            cam.region_json = json.dumps(payload["region"] or [])
            updated = True
        if updated:
            sess.add(cam)
            sess.commit()
            sess.refresh(cam)
            manager.ensure_running(cam)
        return {"detail": "updated"}


@app.delete("/cameras/{camera_id}")
def delete_camera(camera_id: int) -> Dict[str, Any]:
    with get_session() as sess:
        cam = sess.query(Camera).get(camera_id)
        if cam is None:
            return {"detail": "ok"}
        manager.stop(camera_id)
        sess.delete(cam)
        sess.commit()
        return {"detail": "deleted"}


@app.websocket("/cameras/{camera_id}/status")
async def websocket_endpoint(ws: WebSocket, camera_id: int) -> None:
    """Stream processed frames and status to clients for a camera."""

    await ws.accept()
    try:
        while True:
            rt = manager.get_runtime(camera_id)
            if rt is None:
                await asyncio.sleep(0.2)
                continue
            frame = rt.worker.latest_frame
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            processed = rt.processor.process(frame.copy())
            ret, buf = cv2.imencode(".jpg", processed)
            if not ret:
                await asyncio.sleep(0.1)
                continue
            b64 = base64.b64encode(buf).decode("ascii")
            payload = rt.processor.state()
            payload["roll_status"] = get_current_roll_status()
            payload["frame"] = b64
            await ws.send_json(payload)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# The module exposes ``app`` for ASGI servers like uvicorn.
