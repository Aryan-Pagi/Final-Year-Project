"""Headless webcam continuity diagnostic.

This captures a small sequence of frames from the webcam and reports whether
they are changing over time, which helps verify that the camera feed is still
continuous even when the lens is covered.
"""

from __future__ import annotations

import argparse
import os
from statistics import mean
import hashlib

import cv2
import numpy as np


def _open_camera(device_index: int = 0):
    if os.name == "nt":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(device_index)


def _frame_delta(prev_gray: np.ndarray, current_gray: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(prev_gray, current_gray)))


def run_diagnostic(device_index: int = 0, frames: int = 60, frozen_threshold: float = 1.5):
    cap = _open_camera(device_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {device_index}.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Checking {frames} frames from camera {device_index}...")

    deltas = []
    identical_pairs = 0
    frame_hashes = []
    prev_gray = None
    captured = 0

    try:
        while captured < frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Frame {captured + 1}: read failed")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hashes.append(hashlib.md5(gray.tobytes()).hexdigest())
            if prev_gray is not None:
                delta = _frame_delta(prev_gray, gray)
                deltas.append(delta)
                if delta <= frozen_threshold:
                    identical_pairs += 1
                print(f"Frame {captured + 1}: delta={delta:.3f}")
            else:
                print(f"Frame {captured + 1}: captured")

            prev_gray = gray
            captured += 1

    finally:
        cap.release()

    if captured < 2:
        print("RESULT: Not enough frames to judge continuity.")
        return 2

    avg_delta = mean(deltas) if deltas else 0.0
    max_delta = max(deltas) if deltas else 0.0
    min_delta = min(deltas) if deltas else 0.0

    print("--- Summary ---")
    print(f"Captured frames: {captured}")
    print(f"Average delta: {avg_delta:.3f}")
    print(f"Min delta: {min_delta:.3f}")
    print(f"Max delta: {max_delta:.3f}")
    print(f"Near-identical consecutive pairs: {identical_pairs}/{len(deltas)}")
    print(f"Unique frame hashes: {len(set(frame_hashes))}/{len(frame_hashes)}")

    if len(set(frame_hashes)) == 1:
        print("RESULT: Camera feed appears frozen; every sampled frame was identical.")
        return 2

    if avg_delta <= frozen_threshold:
        print("RESULT: Camera frames are continuous, but the scene is mostly static (expected when covered).")
    else:
        print("RESULT: Camera frames are clearly changing, so the feed is continuous.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check whether webcam frames are continuous")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to sample")
    parser.add_argument("--threshold", type=float, default=1.5, help="Mean pixel-delta threshold for frozen frames")
    args = parser.parse_args()
    raise SystemExit(run_diagnostic(args.device, args.frames, args.threshold))


if __name__ == "__main__":
    main()