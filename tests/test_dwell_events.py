import importlib
import sys
import types
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def test_dwell_events_api(monkeypatch, tmp_path):
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

    db_url = f"sqlite:///{tmp_path/'test.db'}"
    monkeypatch.setenv("SP_DB_URL", db_url)

    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
    backend = importlib.import_module("backend.service")

    from backend.database import SessionLocal, DwellEvent

    with SessionLocal() as sess:
        sess.add(DwellEvent(object_id="1", start_ts=1.0, end_ts=2.0, video_path="v.mp4"))
        sess.commit()

    client = TestClient(backend.app)
    resp = client.get("/dwell_events")
    assert resp.status_code == 200
    data = resp.json()
    assert data[0]["video_path"] == "v.mp4"
