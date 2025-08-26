"""Person detector based on Ultralytics YOLO models."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


class PersonDetector:
    """Wrap a YOLO model to detect persons in images."""

    def __init__(self, model_name: str = "yolov8n", conf: float = 0.5) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO detection")
        self.model = YOLO(model_name)
        self.conf = conf

    def detect(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """Detect persons in a frame.

        Returns a tuple of bounding boxes ``[x1, y1, x2, y2]`` and their scores.
        Only class ``person`` (index 0) is kept.
        """
        results = self.model(frame, conf=self.conf, classes=[0])[0]
        boxes, scores = [], []
        for box, score in zip(results.boxes.xyxy, results.boxes.conf):
            boxes.append(box.tolist())
            scores.append(float(score))
        return boxes, scores
