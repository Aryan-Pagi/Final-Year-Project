"""
Quick script to inspect the saved model's reported metrics.
"""

import argparse
import os
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "alphabet": os.path.join(script_dir, "models", "alphabet_model.pkl"),
    "word": os.path.join(script_dir, "models", "word_model.pkl"),
}


def _print_model_info(model_path):
    """Print the stored metrics for one saved model bundle."""
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    classes = model_data.get('index_to_class') or list(model_data.get('label_encoder').classes_)
    confusion_matrix_data = model_data.get('confusion_matrix')

    print("\n" + "=" * 60)
    print("SAVED MODEL INFORMATION")
    print("=" * 60)
    print(f"\nModel Path: {os.path.basename(model_path)}")
    print(f"Model Type: {model_data.get('model_type', 'Unknown')}")
    print(f"Gesture Mode: {model_data.get('gesture_mode', 'Unknown')}")
    print(f"Number of Classes: {len(classes)}")
    print(f"Classes: {', '.join(classes)}")
    print(f"\n{'=' * 60}")
    print("ACCURACY METRICS")
    print("=" * 60)

    val_accuracy = model_data.get('val_accuracy')
    test_accuracy = model_data.get('test_accuracy')
    if val_accuracy is not None:
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    else:
        print("Validation Accuracy: unavailable")

    if test_accuracy is not None:
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    else:
        print("Test Accuracy: unavailable")

    if 'cv_scores' in model_data:
        cv = model_data['cv_scores']
        print(f"Cross-Validation: {cv.mean() * 100:.2f}% +/- {cv.std() * 100:.2f}%")

    print(f"\n{'=' * 60}")
    print("CONFUSION MATRIX")
    print("=" * 60)
    if confusion_matrix_data is not None:
        header = "{:>14}".format("") + "".join(f"{name[:12]:>14}" for name in classes)
        print(header)
        for label, row in zip(classes, confusion_matrix_data):
            row_text = "".join(f"{int(value):>14}" for value in row)
            print(f"{label[:12]:>14}{row_text}")
    else:
        print("Confusion matrix not available in this saved model.")

    print("=" * 60 + "\n")

parser = argparse.ArgumentParser(description="Inspect saved ISL model metrics")
parser.add_argument(
    "--mode",
    choices=["alphabet", "word", "all"],
    default="alphabet",
    help="Which saved model bundle to inspect",
)
args = parser.parse_args()

if args.mode == "all":
    for key in ("alphabet", "word"):
        _print_model_info(MODEL_PATHS[key])
else:
    _print_model_info(MODEL_PATHS[args.mode])
