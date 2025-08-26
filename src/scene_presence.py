from __future__ import annotations

"""Scene-level presence detection with adjustable region and visualization."""

from dataclasses import dataclass

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import json
import time

import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency
    YOLO = None



CONFIG_FILE = Path("scene_presence_config.json")


def load_config(path: Path, default: Dict) -> Dict:
    """Load configuration from JSON file or return ``default``."""

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            default.update(data)
            region = default.get("region")
            if region:
                default["region"] = [tuple(map(int, pt)) for pt in region]
        except Exception:
            pass
    return default


def save_config(path: Path, config: Dict) -> None:
    """Persist configuration to JSON file."""

    to_save = config.copy()
    region = to_save.get("region")
    if region:
        to_save["region"] = [list(pt) for pt in region]
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_save, fh)


def select_polygon(window: str, frame: np.ndarray) -> List[Tuple[int, int]]:
    """Interactively select polygon points on ``frame``."""

    points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        draw = frame.copy()
        if points:
            cv2.polylines(draw, [np.array(points, np.int32)], False, (0, 255, 0), 2)
            for pt in points:
                cv2.circle(draw, pt, 3, (0, 255, 0), -1)
        cv2.imshow(window, draw)
        key = cv2.waitKey(1) & 0xFF
        if key in (13, ord(" ")) and len(points) >= 3:  # Enter/Space to finish
            break
        if key == 27:  # Esc cancels
            points = []
            break

    cv2.setMouseCallback(window, lambda *args: None)
    return points

@dataclass
class WorkerState:
    """State machine for a single worker."""

    status: str = "inactive"
    inside_ms: float = 0.0
    outside_ms: float = 0.0
    total_ms: float = 0.0
    bbox: Tuple[int, int, int, int] | None = None


class ScenePresenceManager:
    """Manage presence state of multiple IDs inside a region."""

    def __init__(
        self,
        region: List[Tuple[int, int]] | None = None,
        *,
        enter_ms: float | None = None,
        leave_ms: float | None = None,
        finish_ms: float | None = None,
        enter_s: float | None = None,
        leave_s: float | None = None,
        finish_s: float | None = None,
    ) -> None:
        """Initialize manager with time-based thresholds.

        Args:
            region: Polygon region as list of points. ``None`` means full frame.
            enter_ms: Milliseconds required to switch from ``pending`` to ``active``.
            leave_ms: Milliseconds tolerated outside the region before ``inactive``.
            finish_ms: Optional max milliseconds in ``active`` before ``finished``.
            enter_s: Same as ``enter_ms`` but specified in seconds.
            leave_s: Same as ``leave_ms`` but specified in seconds.
            finish_s: Same as ``finish_ms`` but specified in seconds.
        """
        if enter_ms is None and enter_s is not None:
            enter_ms = enter_s * 1000.0
        if leave_ms is None and leave_s is not None:
            leave_ms = leave_s * 1000.0
        if finish_ms is None and finish_s is not None:
            finish_ms = finish_s * 1000.0

        self.region = region
        self.enter_ms = float(enter_ms if enter_ms is not None else 500.0)
        self.leave_ms = float(leave_ms if leave_ms is not None else 1000.0)
        self.finish_ms = float(finish_ms) if finish_ms is not None else None
        self.workers: Dict[int, WorkerState] = {}

    # ------------------------------------------------------------------
    def set_region(self, polygon: List[Tuple[int, int]]) -> None:
        """Update detection region."""

        self.region = polygon

    # ------------------------------------------------------------------
    def _inside(self, bbox: Tuple[int, int, int, int]) -> bool:
        if self.region is None:
            return True
        x1, y1, x2, y2 = bbox
        foot_x, foot_y = (x1 + x2) // 2, y2  # 底边中心点
        pts = np.array(self.region, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (foot_x, foot_y), False) >= 0


    # ------------------------------------------------------------------
    def update(
        self,
        detections: Iterable[Tuple[int, Tuple[int, int, int, int]]],
        elapsed_ms: float,
    ) -> None:
        """Update state machine with current frame detections."""

        seen: set[int] = set()
        for oid, bbox in detections:
            seen.add(oid)
            inside = self._inside(bbox)
            state = self.workers.setdefault(oid, WorkerState())
            state.bbox = bbox  # type: ignore[attr-defined]
            if inside:
                state.inside_ms += elapsed_ms
                state.outside_ms = 0.0
                if state.status in {"inactive", "paused"}:
                    state.status = "pending"
                if state.status == "pending" and state.inside_ms >= self.enter_ms:
                    state.status = "active"
                if state.status == "active":
                    state.total_ms += elapsed_ms
                    if self.finish_ms and state.total_ms >= self.finish_ms:
                        state.status = "finished"
            else:
                state.inside_ms = 0.0
                state.outside_ms += elapsed_ms
                if state.outside_ms > self.leave_ms:
                    state.status = "inactive"
                elif state.status == "active":
                    state.status = "paused"

        # cleanup missing workers
        for oid in list(self.workers.keys()):
            if oid not in seen:
                state = self.workers[oid]
                state.outside_ms += elapsed_ms
                if state.outside_ms > self.leave_ms:
                    self.workers.pop(oid)

    # ------------------------------------------------------------------
    def is_scene_active(self) -> bool:
        """Return True if any worker is active."""

        return any(st.status == "active" for st in self.workers.values())

    # ------------------------------------------------------------------
    def draw(self, frame: np.ndarray) -> None:
        """Visualize region and worker states on frame."""

        if self.region is not None:
            pts = np.array(self.region, dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color=(255, 0, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        for oid, state in self.workers.items():
            color_map = {
                "pending": (0, 255, 255),
                "active": (0, 255, 0),
                "paused": (0, 0, 255),
                "finished": (255, 0, 255),
            }
            bbox = getattr(state, "bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = color_map.get(state.status, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"ID{oid}:{state.status}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # ``cv2.putText`` only supports ASCII characters. Using emojis or
        # non-ASCII text can trigger a segmentation fault on some builds of
        # OpenCV, so we keep the overlay simple and ASCII-only.
        text = "ACTIVE" if self.is_scene_active() else "INACTIVE"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if self.is_scene_active() else (0, 0, 255),
            2,
        )


# ---------------------------------------------------------------------------


def run_demo(
    video_source: str | int = 0,
    model_name: str | None = None,
    conf: float | None = None,
    visualize: bool = True,
    config_path: Path = CONFIG_FILE,
    enter_ms: float | None = None,
    leave_ms: float | None = None,
    finish_ms: float | None = None,
    enter_s: float | None = None,
    leave_s: float | None = None,
    finish_s: float | None = None,
    classes: List[str] | None = None,
    min_area: int | None = None,
) -> None:

    """Run presence detection demo with adjustable region.

    ``video_source`` may be a camera index, a local file path or an RTSP URL.
    """

    if YOLO is None:
        raise ImportError("ultralytics is required for presence detection")

    is_rtsp = isinstance(video_source, str) and video_source.startswith("rtsp://")
    # ``cv2.CAP_FFMPEG`` is more robust for network streams like RTSP.
    if is_rtsp:
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_source}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read from video source")

    h, w = frame.shape[:2]
    default_cfg: Dict = {
        "region": [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)],
        "model_name": "yolo11s",
        "conf": 0.5,
        "timing": {"enter_s": 0.5, "leave_s": 1.0, "finish_s": None},
        # Filter detections: only keep specified classes and discard small boxes.
        # ``classes`` uses YOLO class names, e.g. ["person"].
        "classes": ["person"],
        # Minimum bounding-box area in pixels to keep a detection.
        "min_area": 4000,
    }
    config = load_config(config_path, default_cfg)

    # Override config with CLI-specified values when provided
    if model_name is not None:
        config["model_name"] = model_name
    if conf is not None:
        config["conf"] = conf
    timing = config.setdefault("timing", {})
    if enter_ms is not None:
        timing["enter_s"] = enter_ms / 1000.0
    if leave_ms is not None:
        timing["leave_s"] = leave_ms / 1000.0
    if finish_ms is not None:
        timing["finish_s"] = finish_ms / 1000.0
    if enter_s is not None:
        timing["enter_s"] = enter_s
    if leave_s is not None:
        timing["leave_s"] = leave_s
    if finish_s is not None:
        timing["finish_s"] = finish_s
    if classes is not None:
        config["classes"] = classes
    if min_area is not None:
        config["min_area"] = min_area

    save_config(config_path, config)  # ensure config file exists

    detector = YOLO(config["model_name"])
    conf = float(config["conf"])
    timing = config.get("timing", {})
    manager = ScenePresenceManager(
        region=config["region"],
        enter_s=float(timing.get("enter_s", 0.5)),
        leave_s=float(timing.get("leave_s", 1.0)),
        finish_s=timing.get("finish_s"),
    )
    if not is_rtsp:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    last_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
    if last_ts <= 0:
        last_ts = time.time() * 1000.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if timestamp <= 0:
            timestamp = time.time() * 1000.0
        elapsed_ms = timestamp - last_ts
        if elapsed_ms < 0:
            elapsed_ms = 0.0
        last_ts = timestamp

        results = detector.track(frame, conf=conf, persist=True)
        boxes = results[0].boxes
        ids = boxes.id if hasattr(boxes, "id") else None

        detections: List[Tuple[int, Tuple[int, int, int, int]]] = []

        if ids is not None:
            names = getattr(detector, "names", {})
            allowed = set(config.get("classes", []))
            min_area = int(config.get("min_area", 0))
            for box, obj_id, cls_id in zip(boxes.xyxy, ids, boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                area = (x2 - x1) * (y2 - y1)
                class_name = names.get(int(cls_id), str(cls_id)) if names else str(int(cls_id))
                if allowed and class_name not in allowed:
                    continue
                if area < min_area:
                    continue
                detections.append((int(obj_id), (x1, y1, x2, y2)))

        manager.update(detections, elapsed_ms)


        if visualize:
            manager.draw(frame)
            cv2.imshow("scene_presence", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                poly = select_polygon("scene_presence", frame)
                if poly:
                    manager.set_region(poly)
                    config["region"] = poly
                    save_config(config_path, config)
        else:
            # When visualization is disabled, simply continue processing frames.
            pass

    cap.release()
    if visualize:
        cv2.destroyAllWindows()



if __name__ == "__main__":  # pragma: no cover - demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Scene-level presence detection demo")
    parser.add_argument(
        "--video",
        default=0,
        help="Video source (int, file path, or RTSP URL)",
    )
    parser.add_argument("--model", help="YOLO model name")
    parser.add_argument("--conf", type=float, help="Detection confidence")
    parser.add_argument("--enter-ms", type=float, help="Milliseconds required to activate")
    parser.add_argument("--leave-ms", type=float, help="Milliseconds tolerated outside region")
    parser.add_argument("--finish-ms", type=float, help="Max milliseconds in active state")
    parser.add_argument("--enter-s", type=float, help="Seconds required to activate")
    parser.add_argument("--leave-s", type=float, help="Seconds tolerated outside region")
    parser.add_argument("--finish-s", type=float, help="Max seconds in active state")
    parser.add_argument("--classes", nargs="*", help="Allowed detection classes")
    parser.add_argument("--min-area", type=int, help="Minimum bbox area to keep")

    parser.add_argument(
        "--config", default=str(CONFIG_FILE), help="Path to configuration JSON"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable visualization windows"
    )
    args = parser.parse_args()

    src = int(args.video) if str(args.video).isdigit() else args.video
    run_demo(
        src,
        model_name=args.model,
        conf=args.conf,
        visualize=not args.no_display,
        config_path=Path(args.config),
        enter_ms=args.enter_ms,
        leave_ms=args.leave_ms,
        finish_ms=args.finish_ms,
        enter_s=args.enter_s,
        leave_s=args.leave_s,
        finish_s=args.finish_s,
        classes=args.classes,
        min_area=args.min_area,
    )

