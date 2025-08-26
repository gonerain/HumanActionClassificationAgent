"""A very simple centroid-based multi-object tracker."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class SimpleTracker:
    """Track objects across frames using centroid distance.

    Parameters
    ----------
    max_age: int
        Frames to keep a track alive without matching detections.
    min_hits: int
        Minimum consecutive matches before reporting a track.
    dist_threshold: float
        Maximum centroid distance to match detections to existing tracks.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, dist_threshold: float = 50.0) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    @staticmethod
    def _centroid(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, boxes: List[List[float]], scores: List[float]) -> List[Tuple[int, List[float], float]]:
        """Update tracker with new detections.

        Returns list of ``(track_id, bbox, score)`` for active tracks.
        """
        assigned_tracks = {}
        centroids = [self._centroid(b) for b in boxes]

        # match detections to existing tracks
        for det_idx, det_cent in enumerate(centroids):
            best_id = None
            best_dist = self.dist_threshold
            for tid, trk in self.tracks.items():
                dist = np.linalg.norm(np.array(det_cent) - np.array(trk["centroid"]))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                trk = self.tracks[best_id]
                trk["bbox"] = boxes[det_idx]
                trk["centroid"] = det_cent
                trk["age"] = 0
                trk["hits"] += 1
                trk["score"] = scores[det_idx]
                assigned_tracks[det_idx] = best_id

        # create new tracks for unmatched detections
        for idx, box in enumerate(boxes):
            if idx in assigned_tracks:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": box,
                "centroid": centroids[idx],
                "age": 0,
                "hits": 1,
                "score": scores[idx],
            }
            assigned_tracks[idx] = tid

        # age and remove stale tracks
        stale = []
        for tid, trk in self.tracks.items():
            if tid not in assigned_tracks.values():
                trk["age"] += 1
                if trk["age"] > self.max_age:
                    stale.append(tid)
        for tid in stale:
            self.tracks.pop(tid, None)

        # prepare output
        outputs: List[Tuple[int, List[float], float]] = []
        for det_idx, tid in assigned_tracks.items():
            trk = self.tracks[tid]
            if trk["hits"] >= self.min_hits:
                outputs.append((tid, trk["bbox"], trk.get("score", 0.0)))
        return outputs
