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
    args = parser.parse_args()

    src: str | int
    src = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Failed to read from video source")

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
