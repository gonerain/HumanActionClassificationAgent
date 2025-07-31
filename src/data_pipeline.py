import os
from typing import List, Dict
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import mediapipe as mp
except Exception:
    mp = None


class DataPipeline:
    """Pipeline for extracting skeleton sequences from videos.

    修复多人场景下 pose 与 id 不匹配的问题，并在对象丢失时清理缓存。
    """

    def __init__(self, model_name: str = "yolov8n", conf: float = 0.5):
        if YOLO is None:
            raise ImportError("ultralytics is required for YOLO detection")
        self.detector = YOLO(model_name)
        self.conf = conf
        if mp is None:
            raise ImportError("mediapipe is required for pose extraction")
        self.pose = mp.solutions.pose.Pose(static_image_mode=False)

    def _extract_pose(self, frame: np.ndarray) -> np.ndarray | None:
        """Extract pose landmarks from a frame."""
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        landmarks = []
        for lm in result.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks, dtype=np.float32)

    def process_video(
        self,
        video_path: str,
        label: int,
        output_dir: str,
        window_size: int = 30,
        debug: bool = False,
    ) -> List[str]:
        """Extract pose sequences from video and save them.

        Args:
            video_path: path to the video file.
            label: action label for all sequences in this video.
            output_dir: directory to save npz sequences.
            window_size: number of frames per sequence.
        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {video_path}")

        sequences: Dict[int, List[np.ndarray]] = {}
        boxes_seq: Dict[int, List[np.ndarray]] = {}
        missing: Dict[int, int] = {}
        saved_files = []
        frame_idx = 0
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
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    crop = frame[y1:y2, x1:x2]
                    pose = self._extract_pose(crop)
                    if pose is None:
                        continue
                    current_ids.append(int(obj_id))
                    seq = sequences.setdefault(int(obj_id), [])
                    seq.append(pose)
                    box_seq = boxes_seq.setdefault(int(obj_id), [])
                    box_seq.append(np.array([x1, y1, x2, y2], dtype=np.int32))
                    missing[int(obj_id)] = 0
                    if len(seq) >= window_size:
                        start_frame = frame_idx - window_size + 1
                        array = np.stack(seq[:window_size], axis=0)
                        box_array = np.stack(box_seq[:window_size], axis=0)
                        file_path = os.path.join(
                            output_dir,
                            f"{os.path.basename(video_path)}_{obj_id}_{start_frame}.npz",
                        )
                        np.savez_compressed(file_path, data=array, boxes=box_array, label=label)
                        saved_files.append(file_path)
                        sequences[int(obj_id)] = []
                        boxes_seq[int(obj_id)] = []

            if debug:
                vis = frame.copy()
                if ids is not None:
                    for box, obj_id in zip(boxes.xyxy, ids):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            vis,
                            str(int(obj_id)),
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                cv2.imshow("debug", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # update missing counters for ids not detected this frame
            for obj_id in list(sequences.keys()):
                if obj_id not in current_ids:
                    missing[obj_id] = missing.get(obj_id, 0) + 1
                    if missing[obj_id] > window_size:
                        # save partial sequence if long enough
                        seq = sequences[obj_id]
                        bseq = boxes_seq.get(obj_id, [])
                        if len(seq) >= window_size:
                            start_frame = frame_idx - len(seq)
                            array = np.stack(seq[:window_size], axis=0)
                            box_array = np.stack(bseq[:window_size], axis=0) if bseq else None
                            file_path = os.path.join(
                                output_dir,
                                f"{os.path.basename(video_path)}_{obj_id}_{start_frame}.npz",
                            )
                            if box_array is not None:
                                np.savez_compressed(file_path, data=array, boxes=box_array, label=label)
                            else:
                                np.savez_compressed(file_path, data=array, label=label)
                            saved_files.append(file_path)
                        sequences.pop(obj_id, None)
                        boxes_seq.pop(obj_id, None)
                        missing.pop(obj_id, None)

            frame_idx += 1
        cap.release()
        if debug:
            cv2.destroyAllWindows()
        return saved_files
