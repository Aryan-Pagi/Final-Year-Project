"""
Quick script to check the saved model's accuracy
"""
import pickle
import os

model_path = "models/gesture_model.pkl"

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print("\n" + "="*60)
    print("SAVED MODEL INFORMATION")
    print("="*60)
    print(f"\nModel Type: {type(model_data['model']).__name__}")
    print(f"Number of Classes: {len(model_data['label_encoder'].classes_)}")
    print(f"Classes: {', '.join(model_data['label_encoder'].classes_)}")
    print(f"\n{'='*60}")
    print("ACCURACY METRICS")
    print("="*60)
    print(f"Training Accuracy: {model_data['train_accuracy']*100:.2f}%")
    print(f"Test Accuracy: {model_data['test_accuracy']*100:.2f}%")
    if 'cv_scores' in model_data:
        cv = model_data['cv_scores']
        print(f"Cross-Validation: {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
    if 'model_type' in model_data:
        print(f"Best Model Type: {model_data['model_type']}")
    print("="*60 + "\n")
else:
    print(f"\nError: Model file not found at {model_path}")
    print("Please train the model first using: python scripts/train_model.py\n")
