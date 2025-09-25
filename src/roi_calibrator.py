from __future__ import annotations

"""Simple ROI calibration tool that captures a frame and saves polygon region."""

import argparse
from pathlib import Path

import cv2

from scene_presence import CONFIG_FILE, load_config, save_config, select_polygon


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ROI calibration")
    parser.add_argument(
        "--video", default=0, help="Video source index or file path"
    )
    parser.add_argument(
        "--config", default=str(CONFIG_FILE), help="Output configuration file"
    )
    parser.add_argument(
        "--frame", type=int, default=0, help="Frame index to capture (default=0)"
    )
    parser.add_argument(
        "--skip", type=int, default=0, help="Number of frames to skip before capture"
    )
    parser.add_argument(
        "--count", type=int, default=1, help="Number of frames to step through (choose interactively)"
    )
    args = parser.parse_args()

    src: str | int
    src = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    # 跳过指定帧
    frame_to_capture = args.frame + args.skip
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)

    ret, frame = cap.read()
    if not ret:
        raise SystemExit("Failed to read from video source")

    # 如果 count > 1，就让用户用空格翻下一帧，回车选择
    if args.count > 1:
        i = 0
        while ret and i < args.count:
            cv2.imshow("preview", frame)
            key = cv2.waitKey(0)
            if key in (13, 32):  # 回车 / 空格
                break
            i += 1
            ret, frame = cap.read()

    cap.release()

    poly = select_polygon("roi_calibration", frame)
    cv2.destroyAllWindows()
    if not poly:
        print("No region selected; nothing saved")
        return

    cfg_path = Path(args.config)
    config = load_config(cfg_path, {})
    config["region"] = poly
    save_config(cfg_path, config)
    print(f"Saved region with {len(poly)} points to {cfg_path}")


if __name__ == "__main__":  # pragma: no cover - CLI usage
    main()
