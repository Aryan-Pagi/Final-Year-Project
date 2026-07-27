"""
Quick script to check the saved model's accuracy
"""
import pickle
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "alphabet": os.path.join(script_dir, "models", "gesture_model.pkl"),
    "word": os.path.join(script_dir, "models", "word_model.pkl"),
}


def _print_model_info(model_path):
    """Print the stored metrics for one saved model bundle."""
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        return

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    print("\n" + "=" * 60)
    print(f"MODEL: {os.path.basename(model_path)}")
    print("=" * 60)
    print(f"Model Type: {type(model_data['model']).__name__}")
    print(f"Number of Classes: {len(model_data['label_encoder'].classes_)}")
    print(f"Classes: {', '.join(model_data['label_encoder'].classes_)}")

    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)
    print(f"Training Accuracy: {model_data['train_accuracy'] * 100:.2f}%")
    print(f"Test Accuracy: {model_data['test_accuracy'] * 100:.2f}%")

    if "cv_scores" in model_data:
        cv = model_data["cv_scores"]
        print(f"Cross-Validation: {cv.mean() * 100:.2f}% +/- {cv.std() * 100:.2f}%")

    if "model_type" in model_data:
        print(f"Best Model Type: {model_data['model_type']}")

    print("=" * 60 + "\n")


for name, path in MODEL_PATHS.items():
    print(f"\nChecking {name} model...")
    _print_model_info(path)