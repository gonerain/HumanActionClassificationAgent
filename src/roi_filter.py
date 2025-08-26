from __future__ import annotations

"""Region-of-interest filter based on footpoint and foot-strip overlap.

This module implements ROI hit detection using a hysteresis strategy. For each
track we look at the bottom centre ("foot point") of its bounding box as the
primary anchor and additionally compute the overlap between a small foot strip
(the bottom ``foot_ratio`` portion of the box) and the ROI polygon. Two
thresholds control state transitions to avoid jitter: ``enter_th`` is required
for a track to switch from outside to inside; ``leave_th`` is the minimum ratio
required to remain inside.
"""

from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]


class ROIFilter:
    """ROI hit test with footpoint + foot-strip overlap and hysteresis."""

    def __init__(
        self,
        polygon: List[Tuple[int, int]] | None,
        *,
        enter_th: float = 0.5,
        leave_th: float = 0.3,
        foot_ratio: float = 0.2,
    ) -> None:
        self.polygon = polygon
        self.enter_th = enter_th
        self.leave_th = leave_th
        self.foot_ratio = foot_ratio
        self._inside: Dict[int, bool] = {}

    # ------------------------------------------------------------------
    def set_polygon(self, polygon: List[Tuple[int, int]]) -> None:
        """Update ROI polygon."""

        self.polygon = polygon

    # ------------------------------------------------------------------
    def update(self, tracks: Iterable[Tuple[int, BBox]]) -> Dict[int, bool]:
        """Return inside state for each track ID."""

        results: Dict[int, bool] = {}
        for tid, bbox in tracks:
            results[tid] = self._check(tid, bbox)
        return results

    # ------------------------------------------------------------------
    def _check(self, tid: int, bbox: BBox) -> bool:
        if self.polygon is None:
            inside = True
        else:
            foot = self._foot_point(bbox)
            pts = np.array(self.polygon, dtype=np.int32)
            pt_inside = cv2.pointPolygonTest(pts, foot, False) >= 0
            ratio = self._overlap_ratio(bbox)
            prev = self._inside.get(tid, False)
            if prev:
                inside = pt_inside and ratio >= self.leave_th
            else:
                inside = pt_inside and ratio >= self.enter_th
        self._inside[tid] = inside
        return inside

    # ------------------------------------------------------------------
    @staticmethod
    def _foot_point(bbox: BBox) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int(y2))

    # ------------------------------------------------------------------
    def _overlap_ratio(self, bbox: BBox) -> float:
        """Return overlap ratio between foot strip and ROI polygon."""

        if self.polygon is None:
            return 1.0
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        strip_h = max(1, int(h * self.foot_ratio))
        strip = np.array(
            [[x1, y2 - strip_h], [x2, y2 - strip_h], [x2, y2], [x1, y2]],
            dtype=np.int32,
        )
        pts = np.array(self.polygon, dtype=np.int32)

        xs = np.concatenate([pts[:, 0], strip[:, 0]])
        ys = np.concatenate([pts[:, 1], strip[:, 1]])
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        w = int(x_max - x_min + 1)
        h = int(y_max - y_min + 1)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        strip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts - [x_min, y_min]], 1)
        cv2.fillPoly(strip_mask, [strip - [x_min, y_min]], 1)
        inter = cv2.bitwise_and(roi_mask, strip_mask)
        strip_area = int(strip_mask.sum())
        if strip_area == 0:
            return 0.0
        return float(inter.sum()) / strip_area
