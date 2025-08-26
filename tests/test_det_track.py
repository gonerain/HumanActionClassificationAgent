import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from det_track.pipeline import DetTrackPipeline


class DummyDetector:
    """Deterministic detector for testing."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        # two slightly moved boxes across calls
        if self.calls == 0:
            self.calls += 1
            return [[0, 0, 10, 10]], [0.9]
        else:
            self.calls += 1
            return [[1, 1, 11, 11]], [0.8]


def test_pipeline_tracks_consistently():
    pipeline = DetTrackPipeline(detector=DummyDetector(), min_hits=1)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    t1 = pipeline.step(frame)
    t2 = pipeline.step(frame)
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0]["track_id"] == t2[0]["track_id"]
