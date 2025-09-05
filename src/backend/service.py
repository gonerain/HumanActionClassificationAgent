"""Multi-camera backend service for scene presence monitoring.

This service manages multiple camera streams whose configuration (source, ROI)
is stored in the database. It exposes per-camera REST and WebSocket endpoints
for snapshots and status streaming.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Path as ApiPath
from starlette.websockets import WebSocketState  # type: ignore
from fastapi.responses import HTMLResponse

import time

from scene_presence import ScenePresenceManager

from .database import DwellEvent, get_session, init_db, Camera, stop_db_worker
from .workflows import Workflow, ROIWorkflow

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


@dataclass
class HealthStatus:
    status: str = "init"  # init, ok, degraded, error
    last_open_ts: float = 0.0
    last_read_ts: float = 0.0
    last_frame_ts: float = 0.0
    reconnect_attempts: int = 0
    last_error: str | None = None
    stuck_seconds: float = 0.0
    alarm: bool = False
    alarm_message: str | None = None


class VideoCaptureWorker(threading.Thread):
    """Continuously grab frames with reconnection and health checks."""

    def __init__(self, source: int | str = 0) -> None:
        super().__init__(daemon=True)
        self._source_arg = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: np.ndarray | None = None
        self._running = True
        self._prev_small: Optional[np.ndarray] = None
        self.health = HealthStatus()
        self._lock = threading.Lock()

    def _open(self) -> bool:
        # Release any existing cap first
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self._source_arg)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        ok = self.cap.isOpened()
        now = time.time()
        self.health.last_open_ts = now
        if not ok:
            self.health.status = "error"
            self.health.last_error = f"Cannot open {self._source_arg}"
        return ok

    def _small_gray(self, frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sm = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA)
        return sm

    def run(self) -> None:  # pragma: no cover - background loop
        backoff = [1, 2, 5, 10, 30]
        bo_idx = 0
        STUCK_SEC = 5.0
        NOFRAME_SEC = 5.0
        while self._running:
            if self.cap is None or not self.cap.isOpened():
                if not self._open():
                    self.health.reconnect_attempts += 1
                    wait = backoff[min(bo_idx, len(backoff) - 1)]
                    bo_idx = min(bo_idx + 1, len(backoff) - 1)
                    self.health.alarm = True
                    self.health.alarm_message = self.health.last_error or "unknown open error"
                    time.sleep(wait)
                    continue
                # opened successfully
                bo_idx = 0
                self.health.status = "degraded"  # until first frame
                self.health.last_error = None
                self.health.alarm = False
                self.health.alarm_message = None

            ret, frame = self.cap.read()
            now = time.time()
            self.health.last_read_ts = now
            if not ret or frame is None:
                # treat as error -> force reopen
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
                self.health.status = "error"
                self.health.last_error = "read_failed"
                continue

            small = self._small_gray(frame)
            stuck = False
            if self._prev_small is not None:
                diff = cv2.absdiff(small, self._prev_small)
                mean_diff = float(np.mean(diff))
                if mean_diff < 0.5:  # almost identical
                    self.health.stuck_seconds += (now - (self.health.last_frame_ts or now))
                else:
                    self.health.stuck_seconds = 0.0
            self._prev_small = small

            with self._lock:
                self.latest_frame = frame
                self.health.last_frame_ts = now

            # update status
            if self.health.stuck_seconds >= STUCK_SEC:
                self.health.status = "degraded"
                self.health.alarm = True
                self.health.alarm_message = f"stream_stuck_{self.health.stuck_seconds:.1f}s"
            else:
                self.health.status = "ok"
                # clear alarm if good frames continue
                if now - self.health.last_frame_ts < 1.0:
                    self.health.alarm = False
                    self.health.alarm_message = None

            # no frame for too long
            if now - self.health.last_frame_ts > NOFRAME_SEC:
                self.health.status = "error"
                self.health.alarm = True
                self.health.alarm_message = "no_frame_timeout"

        # on exit, release
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        # Attempt to break any blocking read by releasing the capture
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass

    def get_health(self) -> Dict[str, Any]:
        return asdict(self.health)


# ---------------------------------------------------------------------------
# Frame processor using ScenePresenceManager


class FrameProcessor:  # Backward-compatible alias over default workflow
    def __init__(self, model_name: str = "yolo11s", conf: float = 0.5, region: List[Tuple[int, int]] | None = None) -> None:
        self._wf: Workflow = ROIWorkflow(model_name=model_name, conf=conf, region=region)

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self._wf.process(frame)

    def state(self) -> Dict[str, object]:
        return self._wf.state()

    def stop(self) -> None:
        try:
            self._wf.stop()
        except Exception:
            pass

    @property
    def camera_label(self) -> Optional[str]:
        return getattr(self._wf, "camera_label", None)

    @camera_label.setter
    def camera_label(self, value: Optional[str]) -> None:
        if hasattr(self._wf, "set_camera_label"):
            self._wf.set_camera_label(value)

    @property
    def camera_id(self) -> Optional[int]:
        return getattr(self._wf, "camera_id", None)

    @camera_id.setter
    def camera_id(self, value: Optional[int]) -> None:
        if hasattr(self._wf, "set_camera_id"):
            self._wf.set_camera_id(value)


# ---------------------------------------------------------------------------
# FastAPI application


init_db()


# Reduce OpenCV CPU thread usage (helps stability on multi-cam)
try:  # pragma: no cover
    cv2.setNumThreads(1)
except Exception:
    pass


class InferenceWorker(threading.Thread):
    """Per-camera inference worker producing latest jpeg + state at target FPS."""

    def __init__(self, cap_worker: VideoCaptureWorker, processor: FrameProcessor, target_fps: float = 8.0) -> None:
        super().__init__(daemon=True)
        self.cap_worker = cap_worker
        self.processor = processor
        self.target_fps = max(0.5, float(target_fps))
        self.period = 1.0 / self.target_fps
        self._running = True
        self.latest_jpeg: Optional[bytes] = None
        self.latest_payload: Dict[str, Any] = {}

    def run(self) -> None:  # pragma: no cover - background loop
        last = 0.0
        while self._running:
            now = time.time()
            if now - last < self.period:
                time.sleep(min(0.01, self.period))
                continue
            last = now
            frame = self.cap_worker.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                processed = self.processor.process(frame.copy())
                ok, buf = cv2.imencode(".jpg", processed, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    self.latest_jpeg = buf.tobytes()
                self.latest_payload = self.processor.state()
            except Exception:
                time.sleep(0.05)
                continue

    def stop(self) -> None:
        self._running = False


class CameraRuntime:
    """Holds runtime objects for a camera stream."""

    def __init__(self, cam_id: int, source: str | int, region: Optional[List[Tuple[int, int]]], label: str):
        self.worker = VideoCaptureWorker(source)
        self.processor = FrameProcessor(region=region)
        self.processor.camera_label = label
        self.processor.camera_id = cam_id
        self.worker.start()
        self.infer = InferenceWorker(self.worker, self.processor, target_fps=8.0)
        self.infer.start()

    def stop(self) -> None:
        self.worker.stop()
        try:
            self.worker.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.processor.stop()
        except Exception:
            pass
        try:
            self.infer.stop()
            self.infer.join(timeout=2.0)
        except Exception:
            pass


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

    def stop_all(self) -> None:
        for cam_id, rt in list(self.cameras.items()):
            try:
                rt.stop()
            finally:
                self.cameras.pop(cam_id, None)


manager = CameraManager()


def _load_all_cameras() -> None:
    with get_session() as sess:
        cams = sess.query(Camera).all()
        for cam in cams:
            manager.ensure_running(cam)


app = FastAPI()
app.state.shutting_down = False
_load_all_cameras()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    # Gracefully stop camera workers and DB background worker
    try:
        app.state.shutting_down = True
    except Exception:
        pass
    try:
        manager.stop_all()
    except Exception:
        pass
    try:
        stop_db_worker()
    except Exception:
        pass


@app.get("/cameras/{camera_id}/inference/report")
def inference_report(camera_id: int = ApiPath(..., ge=1)) -> Dict[str, object]:
    """Return scene presence report for a single camera."""

    rt = manager.get_runtime(camera_id)
    if rt is None:
        raise HTTPException(status_code=404, detail="camera not running or not found")
    if hasattr(rt, "infer") and rt.infer.latest_payload:
        data = dict(rt.infer.latest_payload)
    else:
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
    if hasattr(rt, "infer") and rt.infer.latest_jpeg is not None:
        b64 = base64.b64encode(rt.infer.latest_jpeg).decode("ascii")
        payload = dict(rt.infer.latest_payload)
    else:
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
                "camera_id": e.camera_id,
                "start_ts": e.start_ts,
                "end_ts": e.end_ts,
                "video_path": e.video_path,
            }
            for e in events
        ]


@app.get("/cameras/{camera_id}/dwell_events")
def dwell_events_by_camera(camera_id: int) -> List[Dict[str, object]]:
    with get_session() as sess:
        events = (
            sess.query(DwellEvent)
            .filter(DwellEvent.camera_id == camera_id)
            .order_by(DwellEvent.start_ts)
            .all()
        )
        return [
            {
                "id": e.id,
                "object_id": e.object_id,
                "camera_id": e.camera_id,
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
                    "health": rt.worker.get_health() if rt else None,
                }
            )
        return out


@app.get("/cameras/{camera_id}/health")
def camera_health(camera_id: int) -> Dict[str, Any]:
    rt = manager.get_runtime(camera_id)
    if rt is None:
        raise HTTPException(status_code=404, detail="camera not running or not found")
    return rt.worker.get_health()


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
        # commit handled by get_session context
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
            # commit handled by get_session context
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
        # commit handled by get_session context
        return {"detail": "deleted"}


@app.websocket("/cameras/{camera_id}/status")
async def websocket_endpoint(ws: WebSocket, camera_id: int) -> None:
    """Stream processed frames and status to clients for a camera."""

    await ws.accept()
    try:
        while True:
            # Break fast on shutdown or client disconnect
            if getattr(app.state, "shutting_down", False):
                break
            try:
                if ws.client_state is not None and ws.client_state != WebSocketState.CONNECTED:
                    break
            except Exception:
                # If we cannot query state, continue and rely on send exceptions
                pass

            try:
                rt = manager.get_runtime(camera_id)
                if rt is None:
                    # only send when connected
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({
                            "alarm": True,
                            "alarm_message": "camera_not_running",
                            "health": None,
                        })
                    await asyncio.sleep(0.2)
                    continue

                # Use precomputed inference; if missing send heartbeat
                if not hasattr(rt, "infer") or rt.infer.latest_jpeg is None:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({
                            "health": rt.worker.get_health(),
                            "alarm": True,
                            "alarm_message": "no_frame",
                        })
                    await asyncio.sleep(0.2)
                    continue

                b64 = base64.b64encode(rt.infer.latest_jpeg).decode("ascii")
                payload = dict(rt.infer.latest_payload)
                payload["roll_status"] = get_current_roll_status()
                payload["frame"] = b64
                health = rt.worker.get_health()
                payload["health"] = health
                payload["alarm"] = bool(health.get("alarm"))
                payload["alarm_message"] = health.get("alarm_message")
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_json(payload)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Propagate to allow server to exit
                raise
            except RuntimeError:
                # Likely "Cannot call send once close sent" => exit loop
                break
            except Exception as e:
                # Try to notify once if still connected; otherwise just back off
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.send_json({
                            "alarm": True,
                            "alarm_message": f"server_error:{type(e).__name__}",
                        })
                except Exception:
                    pass
                await asyncio.sleep(0.1)
                continue
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The module exposes ``app`` for ASGI servers like uvicorn.
