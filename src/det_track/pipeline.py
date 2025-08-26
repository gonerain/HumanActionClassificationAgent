"""Combine person detection and simple tracking."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .detector import PersonDetector
from .tracker import SimpleTracker


class DetTrackPipeline:
    """Run person detection and multi-object tracking on frames."""

    def __init__(
        self,
        detector: Optional[PersonDetector] = None,
        tracker: Optional[SimpleTracker] = None,
        *,
        model_name: str = "yolov8n",
        det_conf: float = 0.5,
        nms_iou: float = 0.7,
        max_age: int = 30,
        min_hits: int = 3,
        min_area: int = 0,
        min_area_ratio: float = 0.0,
    ) -> None:
        self.detector = detector or PersonDetector(model_name=model_name, conf=det_conf)
        self.tracker = tracker or SimpleTracker(max_age=max_age, min_hits=min_hits)
        self.nms_iou = nms_iou  # kept for future extension
        self.min_area = min_area
        self.min_area_ratio = min_area_ratio

    def step(self, frame: np.ndarray) -> List[Dict[str, object]]:
        """Process a single frame and return track results."""
        boxes, scores = self.detector.detect(frame)
        h, w = frame.shape[:2]
        area_th = self.min_area if self.min_area > 0 else self.min_area_ratio * w * h

        filtered_boxes = []
        filtered_scores = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area >= area_th:
                filtered_boxes.append(box)
                filtered_scores.append(score)

        tracked = self.tracker.update(filtered_boxes, filtered_scores)
        output = [
            {"track_id": tid, "bbox": bbox, "score": score}
            for tid, bbox, score in tracked
        ]
        return output
