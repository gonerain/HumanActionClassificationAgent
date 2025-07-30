import os
from glob import glob
import cv2
import numpy as np
import mediapipe as mp
import argparse

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

CONNECTIONS = mp_pose.POSE_CONNECTIONS


def draw_landmarks(frame, landmarks, bbox=None):
    h, w, _ = frame.shape
    points = []
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        for lm in landmarks:
            px = int(lm[0] * bw) + x1
            py = int(lm[1] * bh) + y1
            points.append((px, py))
    else:
        for lm in landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            points.append((px, py))

    for c in CONNECTIONS:
        p1, p2 = points[c[0]], points[c[1]]
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for p in points:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    return frame


def play_sequence(video_path, start, data, boxes=None, label=0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(len(data)):
        ret, frame = cap.read()
        if not ret:
            break
        box = boxes[i] if boxes is not None and i < len(boxes) else None
        frame = draw_landmarks(frame, data[i], box)
        cv2.putText(frame, f'label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('sequence', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()


def main(dataset_dir, video_dir):
    files = sorted(glob(os.path.join(dataset_dir, '**', '*.npz'), recursive=True))
    idx = 0
    while True:
        file = files[idx]
        data = np.load(file)
        seq = data['data']
        base = os.path.basename(file)
        vid_name, id_label, start = base.rsplit('_', 2)
        obj_id, label_in_name = id_label.split('.') if '.' in id_label else (id_label, '0')
        label = int(data.get('label', int(label_in_name)))
        boxes = data['boxes'] if 'boxes' in data else None
        start = int(os.path.splitext(start)[0])
        video_path = os.path.join(video_dir, vid_name)
        play_sequence(video_path, start, seq, boxes, label)
        print(f'当前文件: {base} 标签:{label}')
        print('n:下一条  p:上一条  0-9:设置标签  q:退出')
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                idx = (idx + 1) % len(files)
                break
            elif key == ord('p'):
                idx = (idx - 1) % len(files)
                break
            elif ord('0') <= key <= ord('9'):
                label = key - ord('0')
                new_base = f"{vid_name}_{obj_id}.{label}_{start}.npz"
                new_file = os.path.join(os.path.dirname(file), new_base)
                np.savez_compressed(new_file, data=seq, boxes=boxes, label=label)
                if new_file != file:
                    os.remove(file)
                    files[idx] = new_file
                    file = new_file
                    base = new_base
                print(f'已保存标签 {label} -> {base}')
            elif key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Skeleton dataset annotator')
    parser.add_argument('dataset_dir', help='目录，包含 npz 序列文件')
    parser.add_argument('video_dir', help='原始视频所在目录')
    args = parser.parse_args()
    main(args.dataset_dir, args.video_dir)
