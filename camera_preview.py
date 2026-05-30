"""Standalone webcam gesture preview for local testing.

This keeps the camera open in one continuous loop and runs the gesture
recognizer directly so you can verify live prediction in VS Code before
reconnecting the Flask dashboard.
"""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.realtime_predict import load_model, predict_realtime


def run_camera_preview(device_index: int = 0, model_path: str = 'models/gesture_model.pkl'):
    """Run live gesture recognition until the user quits."""
    model, label_encoder, _, _, model_meta = load_model(model_path)
    if model is None or label_encoder is None:
        return 1

    resolved_model = model_path
    if model_meta.get('model_type') == 'BiLSTM':
        resolved_model = 'models/bilstm_model.keras'
    elif model_meta.get('model_type') == 'RandomForest_Static':
        resolved_model = 'models/static_classifier.pkl'

    print(f"Using model: {resolved_model}")
    print("Live gesture preview running. Press 'q' in the preview window to quit.")
    predict_realtime(model_path=resolved_model, device_index=device_index, display=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Standalone live gesture preview")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--model", type=str, default='models/gesture_model.pkl', help="Model path")
    args = parser.parse_args()
    raise SystemExit(run_camera_preview(args.device, args.model))


if __name__ == "__main__":
    main()