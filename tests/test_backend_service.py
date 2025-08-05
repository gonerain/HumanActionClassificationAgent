import importlib
import sys
import types
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_inference_report(monkeypatch):
    """Smoke test the /inference/report endpoint."""

    class DummyCap:
        def __init__(self, *args, **kwargs):
            self.frame = np.zeros((4, 4, 3), dtype=np.uint8)

        def isOpened(self):
            return True

        def read(self):
            return True, self.frame

        def release(self):
            pass

    dummy_cv2 = types.SimpleNamespace()
    dummy_cv2.VideoCapture = lambda *a, **k: DummyCap()
    dummy_cv2.imencode = lambda *a, **k: (True, b"")
    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)

    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    backend_service = importlib.import_module("backend_service")

    # stop background thread and stub processor/state for deterministic test
    backend_service.capture_worker.stop()
    backend_service.capture_worker.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    class DummyProcessor:
        @staticmethod
        def process(frame):
            return frame

        @staticmethod
        def state():
            return {"active_ids": [], "scene_active": False}

    backend_service.processor = DummyProcessor()

    client = TestClient(backend_service.app)
    resp = client.get("/inference/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["roll_status"] == "ready"
    assert "scene_active" in data
