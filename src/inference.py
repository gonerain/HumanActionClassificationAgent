import os
from typing import Dict, List
import cv2
import numpy as np
import torch
from torch import nn

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import mediapipe as mp
except Exception:
    mp = None

from .model_trainer import LSTMClassifier


class ActionInference:
    """Run trained action classifier on video stream."""

    def __init__(self, model_path: str, model_name: str = "yolov8n", conf: float = 0.5, window_size: int = 30):
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO detection")
        if mp is None:
            raise ImportError("mediapipe is required for pose extraction")
        self.detector = YOLO(model_name)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)
        self.conf = conf
        self.window_size = window_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dummy input to infer size
        dummy = np.zeros((window_size, 33, 3), dtype=np.float32)
        input_size = dummy.shape[-1] * dummy.shape[-2]
        self.model = LSTMClassifier(input_size=input_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def _extract_pose(self, frame: np.ndarray) -> np.ndarray | None:
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark], dtype=np.float32)

    def run(self, video_source: str | int = 0, debug: bool = True):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_source}")

        sequences: Dict[int, List[np.ndarray]] = {}
        missing: Dict[int, int] = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.track(frame, conf=self.conf, persist=True)
            boxes = results[0].boxes
            ids = boxes.id if hasattr(boxes, "id") else None
            current_ids = []

            if ids is not None:
                for box, obj_id in zip(boxes.xyxy, ids):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    crop = frame[y1:y2, x1:x2]
                    pose = self._extract_pose(crop)
                    if pose is None:
                        continue
                    oid = int(obj_id)
                    current_ids.append(oid)
                    seq = sequences.setdefault(oid, [])
                    seq.append(pose)
                    missing[oid] = 0

                    if len(seq) >= self.window_size:
                        window = np.stack(seq[-self.window_size :], axis=0)
                        tensor = torch.from_numpy(window).view(1, self.window_size, -1).to(self.device)
                        with torch.no_grad():
                            pred = self.model(tensor).argmax(dim=1).item()
                        label_text = str(pred)
                        if debug:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID{oid}:{label_text}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if debug:
                    for box, obj_id in zip(boxes.xyxy, ids):
                        if int(obj_id) not in current_ids:
                            x1, y1, x2, y2 = map(int, box.tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, str(int(obj_id)), (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # update missing counters
            for oid in list(sequences.keys()):
                if oid not in current_ids:
                    missing[oid] = missing.get(oid, 0) + 1
                    if missing[oid] > self.window_size:
                        sequences.pop(oid, None)
                        missing.pop(oid, None)

            if debug:
                cv2.imshow("inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if debug:
            cv2.destroyAllWindows()
