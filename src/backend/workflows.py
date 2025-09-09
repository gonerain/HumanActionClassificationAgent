from __future__ import annotations

"""Workflow abstractions for video processing.

Default workflow implements ROI-based presence detection using YOLO and
ScenePresenceManager, including optional video evidence recording and
persisting dwell events to the database.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import cv2
import numpy as np
from datetime import datetime, timezone
import logging

from scene_presence import ScenePresenceManager
from .database import enqueue_dwell_event

try:  # pragma: no cover - optional runtime dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


class Workflow:
    """Abstract processing workflow interface."""

    def process(self, frame: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def state(self) -> Dict[str, object]:  # pragma: no cover - interface
        raise NotImplementedError

    def set_camera_label(self, label: Optional[str]) -> None:
        pass

    def stop(self) -> None:
        pass

    def set_camera_id(self, camera_id: Optional[int]) -> None:
        pass


class ROIWorkflow(Workflow):
    """ROI-based presence tracking workflow using YOLO."""

    def __init__(
        self,
        model_name: str = "yolo11s",
        conf: float = 0.5,
        region: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.manager = ScenePresenceManager(region=region)
        self.conf = conf
        self.detector = YOLO(model_name) if YOLO is not None else None
        # recordings directory at repo root
        self.video_dir = Path(__file__).resolve().parents[2] / "recordings"
        self.video_dir.mkdir(exist_ok=True)
        self._recorders: Dict[int, Tuple[cv2.VideoWriter, float, Path]] = {}
        self.camera_label: Optional[str] = None
        self.camera_id: Optional[int] = None
        self._stopped: bool = False

        # reduce ultralytics verbosity if present
        try:
            from ultralytics.utils import LOGGER  # type: ignore

            LOGGER.setLevel(logging.ERROR)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _clock(self) -> float:
        return time.time() * 1000.0

    def set_camera_label(self, label: Optional[str]) -> None:
        self.camera_label = label

    def set_camera_id(self, camera_id: Optional[int]) -> None:
        self.camera_id = camera_id

    def stop(self) -> None:
        # release all writers
        for oid, (vw, _, _) in list(self._recorders.items()):
            try:
                vw.release()
            except Exception:
                pass
        self._recorders.clear()
        self._stopped = True
        # try to release detector resources
        try:
            import gc

            self.detector = None
            gc.collect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        if self._stopped:
            return frame
        detections: List[Tuple[int, Tuple[int, int, int, int]]] = []
        if self.detector is not None:
            results = self.detector.track(frame, conf=self.conf, persist=True, verbose=False)
            boxes = results[0].boxes
            ids = boxes.id if hasattr(boxes, "id") else None
            classes = boxes.cls if hasattr(boxes, "cls") else None
            if ids is not None and classes is not None:
                names = self.detector.names
                for box, obj_id, cls_id in zip(boxes.xyxy, ids, classes):
                    class_name = names.get(int(cls_id), str(cls_id))
                    if class_name != "person":
                        continue
                    x1, y1, x2, y2 = map(int, box.tolist())
                    detections.append((int(obj_id), (x1, y1, x2, y2)))

        now_ms = self._clock()
        self.manager.update(detections, now_ms)

        # record videos for active workers
        states = {oid: st.status for oid, st in self.manager.workers.items()}
        active_ids = [oid for oid, status in states.items() if status == "active"]
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

        # finalize events only when worker is truly done
        for oid in list(self._recorders.keys()):
            status = states.get(oid)
            should_close = status in {"inactive", "finished"} or status is None
            if should_close:
                vw, start_ts, path = self._recorders.pop(oid)
                vw.release()
                enqueue_dwell_event(
                    object_id=str(oid),
                    camera_id=self.camera_id,
                    start_ts=datetime.fromtimestamp(start_ts, tz=timezone.utc),
                    end_ts=datetime.fromtimestamp(now_s, tz=timezone.utc),
                    video_path=str(path),
                )

        self.manager.draw(frame)
        return frame

    def state(self) -> Dict[str, object]:
        active_ids = [oid for oid, st in self.manager.workers.items() if st.status == "active"]
        return {
            "active_ids": active_ids,
            "scene_active": self.manager.is_scene_active(),
            "timestamp_ms": self._clock(),
        }


# ---------------------------------------------------------------------------
# Registry and factory for future extensibility

WORKFLOW_REGISTRY = {
    "roi": ROIWorkflow,
}


def create_workflow(name: str = "roi", **kwargs) -> Workflow:
    cls = WORKFLOW_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown workflow: {name}")
    return cls(**kwargs)
