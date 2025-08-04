from __future__ import annotations

"""Scene-level presence detection with adjustable region and visualization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import json
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - runtime dependency
    YOLO = None


CONFIG_FILE = Path("scene_presence_config.json")


def load_region(path: Path, default: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Load polygon region from JSON file or return default."""

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            region = data.get("region")
            if region:
                return [tuple(map(int, pt)) for pt in region]
        except Exception:
            pass
    return default


def save_region(path: Path, region: List[Tuple[int, int]]) -> None:
    """Persist polygon region to JSON file."""

    with path.open("w", encoding="utf-8") as fh:
        json.dump({"region": region}, fh)


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
    inside_frames: int = 0
    outside_frames: int = 0
    total_frames: int = 0
    bbox: Tuple[int, int, int, int] | None = None


class ScenePresenceManager:
    """Manage presence state of multiple IDs inside a region."""

    def __init__(
        self,
        region: List[Tuple[int, int]] | None = None,
        enter_frames: int = 15,
        leave_frames: int = 30,
        finish_frames: int | None = None,
    ) -> None:
        """Initialize manager.

        Args:
            region: Polygon region as list of points. None means full frame.
            enter_frames: Frames required to switch from ``pending`` to ``active``.
            leave_frames: Allowed missing frames before ``inactive``.
            finish_frames: Optional max frames in ``active`` before ``finished``.
        """
        self.region = region
        self.enter_frames = enter_frames
        self.leave_frames = leave_frames
        self.finish_frames = finish_frames
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
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pts = np.array(self.region, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (cx, cy), False) >= 0

    # ------------------------------------------------------------------
    def update(self, detections: Iterable[Tuple[int, Tuple[int, int, int, int]]]) -> None:
        """Update state machine with current frame detections."""

        seen: set[int] = set()
        for oid, bbox in detections:
            seen.add(oid)
            inside = self._inside(bbox)
            state = self.workers.setdefault(oid, WorkerState())
            state.bbox = bbox  # type: ignore[attr-defined]
            if inside:
                state.inside_frames += 1
                state.outside_frames = 0
                if state.status in {"inactive", "paused"}:
                    state.status = "pending"
                if state.status == "pending" and state.inside_frames >= self.enter_frames:
                    state.status = "active"
                if state.status == "active":
                    state.total_frames += 1
                    if self.finish_frames and state.total_frames >= self.finish_frames:
                        state.status = "finished"
            else:
                state.inside_frames = 0
                state.outside_frames += 1
                if state.outside_frames > self.leave_frames:
                    state.status = "inactive"
                elif state.status == "active":
                    state.status = "paused"

        # cleanup missing workers
        for oid in list(self.workers.keys()):
            if oid not in seen:
                state = self.workers[oid]
                state.outside_frames += 1
                if state.outside_frames > self.leave_frames:
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
    model_name: str = "yolo11s",
    conf: float = 0.5,
    visualize: bool = True,
    config_path: Path = CONFIG_FILE,
) -> None:
    """Run presence detection demo with adjustable region."""

    if YOLO is None:
        raise ImportError("ultralytics is required for presence detection")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_source}")

    detector = YOLO(model_name)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise IOError("Cannot read from video source")

    h, w = frame.shape[:2]
    default_region = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
    region = load_region(config_path, default_region)
    save_region(config_path, region)  # ensure config file exists
    manager = ScenePresenceManager(region=region)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.track(frame, conf=conf, persist=True)
        boxes = results[0].boxes
        ids = boxes.id if hasattr(boxes, "id") else None
        detections: List[Tuple[int, Tuple[int, int, int, int]]] = []
        if ids is not None:
            for box, obj_id in zip(boxes.xyxy, ids):
                x1, y1, x2, y2 = map(int, box.tolist())
                detections.append((int(obj_id), (x1, y1, x2, y2)))

        manager.update(detections)

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
                    save_region(config_path, poly)
        else:
            # When visualization is disabled, simply continue processing frames.
            pass

    cap.release()
    if visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":  # pragma: no cover - demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Scene-level presence detection demo")
    parser.add_argument("--video", default=0, help="Video source (int or file path)")
    parser.add_argument("--model", default="yolo11s", help="YOLO model name")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    parser.add_argument(
        "--config", default=str(CONFIG_FILE), help="Path to region config JSON"
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
    )
