import sys
import time
import types
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_camera_registration_and_health(monkeypatch):
    """Ensure /cameras and /cameras/{id}/health work."""

    class DummyCap:
        def __init__(self, *args, **kwargs):
            self.frame = np.zeros((2, 2, 3), dtype=np.uint8)

        def isOpened(self):
            return True

        def read(self):
            return True, self.frame

        def release(self):
            pass

    dummy_cv2 = types.SimpleNamespace()
    dummy_cv2.VideoCapture = lambda *a, **k: DummyCap()
    dummy_cv2.CAP_FFMPEG = 0
    monkeypatch.setitem(sys.modules, "cv2", dummy_cv2)

    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    ingest_gateway = __import__("ingest_gateway")

    client = TestClient(ingest_gateway.app)

    payload = {"id": "cam1", "scene_id": "scene1", "rtsp_url": "rtsp://example", "fps_target": 100}
    resp = client.post("/cameras", json=payload)
    assert resp.status_code == 200

    # allow worker to grab at least one frame
    time.sleep(0.05)

    resp2 = client.get("/cameras/cam1/health")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["running"] is True
    assert "fps_est" in data

    ingest_gateway.workers["cam1"].stop()
    ingest_gateway.workers["cam1"].join(timeout=1)
