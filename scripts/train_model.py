"""
Model Training Script for ISL Gesture Recognition
This script trains models for gesture recognition:
- Static classifier (RandomForest) for all discovered folder labels
- Dynamic classifier (BiLSTM) for motion-based word gestures
"""

import os
import sys
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

tf = None
Sequential = None
Bidirectional = None
LSTM = None
Dense = None
Dropout = None
Masking = None
EarlyStopping = None
ReduceLROnPlateau = None
ModelCheckpoint = None
Adam = None

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_SEQ_FRAMES = 30


def train_model(sequences_npz='dataset/sequences.npz',
                model_output='models/gesture_model.pkl',
                keras_model_output='models/bilstm_model.keras',
                test_size=0.2, random_state=42,
                epochs=100, batch_size=20):
    """
    Train a Bidirectional LSTM classifier on the landmark sequences dataset.

    Args:
        sequences_npz (str): Path to the sequences .npz file produced by extract_landmarks.py
        model_output (str): Path to save the metadata pickle
        keras_model_output (str): Path to save the Keras model
        test_size (float): Proportion of dataset to use as test set
        random_state (int): Random seed for reproducibility
        epochs (int): Number of training epochs
        batch_size (int): Size of training batches
    """
    global tf, Sequential, Bidirectional, LSTM, Dense, Dropout, Masking
    global EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Adam

    if tf is None:
        try:
            import tensorflow as tf  # type: ignore
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
            from tensorflow.keras.optimizers import Adam
        except Exception as exc:
            print("Error: TensorFlow could not be loaded.")
            print("Dynamic BiLSTM training is unavailable until the TensorFlow runtime works.")
            print(f"Details: {exc}")
            return

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_npz_path = os.path.join(script_dir, sequences_npz)
    full_model_path = os.path.join(script_dir, model_output)
    full_keras_path = os.path.join(script_dir, keras_model_output)

    if not os.path.exists(full_npz_path):
        print(f"Sequences file not found: {full_npz_path}")
        print("Trying to extract dynamic sequences from dataset/raw_clips/...")
        try:
            from scripts.extract_landmarks import extract_sequences_from_dataset
            extract_sequences_from_dataset(
                dataset_path='dataset/raw_clips',
                output_npz=sequences_npz,
                use_normalized=True,
            )
        except Exception as exc:
            print(f"Error: could not auto-extract sequences: {exc}")
            print("Please create dataset/raw_clips/<label>/clip_* folders first.")
            return

    if not os.path.exists(full_npz_path):
        print(f"Error: Sequences file not found: {full_npz_path}")
        print("Please run extract_landmarks.py first or add dynamic clip folders.")
        return

    print(f"\n{'='*60}")
    print("Training ISL Gesture Recognition - Bidirectional LSTM")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────
    print(f"Loading sequences from: {full_npz_path}")
    data = np.load(full_npz_path, allow_pickle=True)
    X = data['X'].astype(np.float32)          # (N, SEQ_LEN, FEAT_SIZE)
    raw_labels = data['labels'].astype(str)

    print(f"✓ Dataset loaded: {X.shape[0]} samples, "
          f"sequence length {X.shape[1]}, {X.shape[2]} features per frame")

    label_counts = Counter(raw_labels)
    print(f"\nLabel distribution:")
    for lbl in sorted(label_counts.keys(), key=str):
        print(f"  {lbl:20s}: {label_counts[lbl]}")
    print()

    counts = list(label_counts.values())
    min_samples = min(counts)
    max_samples = max(counts)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0

    print(f"Class Balance Check:")
    print(f"  - Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print(f"  ⚠ Warning: High imbalance detected. Model may be biased towards {max(label_counts, key=label_counts.get)}")

    if min_samples < 10:
        print(f"{'='*60}")
        print("⚠ DATA QUALITY WARNING")
        print(f"{'='*60}")
        print(f"  Minimum samples per class: {min_samples}")
        print(f"  Classes with <10 samples: "
              f"{[k for k,v in label_counts.items() if v < 10]}")
        print(f"  Collect at least 50-100 samples per class for good results.")
        print(f"{'='*60}\n")
    elif min_samples < 50:
        print(f"⚠ Warning: Some classes have fewer than 50 samples.\n")

    # ── Encode labels ─────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)

    print(f"{'='*60}")
    print("Splitting dataset...")
    print(f"{'='*60}")

    min_class = int(np.bincount(y).min())
    strat = y if min_class >= 2 else None
    if min_class < 2:
        print(f"⚠ Warning: Some classes have <2 samples; stratification disabled.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    print(f"✓ Training set : {len(X_train)} samples")
    print(f"✓ Test set     : {len(X_test)} samples\n")

    # ── Build Bidirectional LSTM ───────────────────────────────────
    seq_len, feat_size = X_train.shape[1], X_train.shape[2]

    print(f"{'='*60}")
    print("Building Bidirectional LSTM Model")
    print(f"  Input shape  : ({seq_len}, {feat_size})")
    print(f"  Num classes  : {num_classes}")
    print(f"{'='*60}\n")

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(seq_len, feat_size)),
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ], name='BiLSTM_ISL')

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()
    print()

    # ── Callbacks ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(full_keras_path), exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            full_keras_path, monitor='val_accuracy', save_best_only=True, verbose=0
        ),
    ]

    # ── Training ──────────────────────────────────────────────────
    print(f"{'='*60}")
    print("Training...")
    print(f"{'='*60}\n")

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_acc = max(history.history.get('val_accuracy', [0]))
    print(f"\n✓ Best validation accuracy: {best_val_acc:.2%}")

    # ── Evaluation ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}\n")

    # Load best checkpoint weights saved by ModelCheckpoint
    if os.path.exists(full_keras_path):
        model = tf.keras.models.load_model(full_keras_path)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print("=" * 60)
    unique_test = np.unique(y_test)
    target_names = label_encoder.classes_[unique_test]
    print(classification_report(y_test, y_pred, labels=unique_test, target_names=target_names))

    print("\nConfusion Matrix (Test Set):")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    if cm_off.max() > 0:
        print("\nMost Confused Gesture Pairs:")
        print("-" * 40)
        flat_indices = np.argsort(cm_off.ravel())[::-1]
        shown = 0
        for flat_idx in flat_indices:
            i, j = divmod(flat_idx, cm_off.shape[1])
            if cm_off[i, j] > 0 and shown < 5:
                li = label_encoder.classes_[unique_test[i]] if i < len(unique_test) else '?'
                lj = label_encoder.classes_[unique_test[j]] if j < len(unique_test) else '?'
                print(f"  {li} misclassified as {lj}: {cm_off[i, j]} times")
                shown += 1
    print()

    # ── Save metadata ─────────────────────────────────────────────
    print(f"{'='*60}")
    print("Saving Model...")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(full_model_path), exist_ok=True)

    model_data = {
        'model_type': 'BiLSTM',
        'keras_model_path': 'models/bilstm_model.keras',
        'label_encoder': label_encoder,
        'max_seq_frames': int(seq_len),
        'feature_size': int(feat_size),
        'num_features': int(feat_size),    # kept for backward-compat detection
        'uses_engineered_features': False,
        'test_accuracy': test_accuracy,
        'val_accuracy': best_val_acc,
    }

    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Keras model saved to : {full_keras_path}")
    print(f"✓ Metadata saved to    : {full_model_path}")
    print(f"✓ Model includes:")
    print(f"  - Bidirectional LSTM (2 layers)")
    print(f"  - Label encoder ({num_classes} classes)")
    print(f"  - Input shape: ({seq_len}, {feat_size})")
    print(f"  - Validation accuracy: {best_val_acc:.2%}")
    print(f"  - Test accuracy: {test_accuracy:.2%}")
    print()

    print(f"{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


def _train_static_random_forest(landmarks_csv='dataset/landmarks.csv',
                                model_output='models/static_classifier.pkl',
                                test_size=0.2, random_state=42,
                                n_estimators=300, min_samples_split=3,
                                training_title='Training Unified Static Classifier - RandomForest',
                                save_label='Unified Static Classifier'):
    """Train a data-driven RandomForest classifier on every discovered static label."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_csv_path = os.path.join(script_dir, landmarks_csv)
    full_model_path = os.path.join(script_dir, model_output)

    if not os.path.exists(full_csv_path):
        print(f"Landmarks CSV not found: {full_csv_path}")
        print("Trying to auto-extract static landmarks from dataset/raw_images/...")
        try:
            from scripts.extract_landmarks import extract_landmarks_from_dataset
            extract_landmarks_from_dataset(
                dataset_path='dataset/raw_images',
                output_csv=landmarks_csv,
                use_normalized=True,
            )
        except Exception as exc:
            print(f"Error: could not auto-extract static landmarks: {exc}")
            print("Please create dataset/raw_images/<label>/ folders first.")
            return

    if not os.path.exists(full_csv_path):
        print(f"Error: Landmarks CSV not found: {full_csv_path}")
        return

    print(f"\n{'='*60}")
    print(training_title)
    print(f"{'='*60}\n")

    try:
        import pandas as pd
        df = pd.read_csv(full_csv_path)
        print(f"✓ Loaded {len(df)} samples from {landmarks_csv}")
    except ImportError:
        print("Error: Pandas required for CSV loading. Install with: pip install pandas")
        return
    except Exception as e:
        print(f"Error loading landmarks CSV: {e}")
        return

    if 'label' not in df.columns:
        print("Error: CSV must have a 'label' column")
        return

    X = df.drop('label', axis=1).values.astype(np.float32)
    raw_labels = df['label'].values.astype(str)

    if len(raw_labels) == 0:
        print("Error: No labeled samples found in the dataset.")
        return

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Loaded {len(np.unique(raw_labels))} discovered classes\n")

    label_counts = Counter(raw_labels)
    print("Label distribution:")
    for lbl in sorted(label_counts.keys(), key=str):
        print(f"  {lbl:20s}: {label_counts[lbl]:4d} samples")
    print()

    counts = list(label_counts.values())
    min_samples = min(counts)
    max_samples = max(counts)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0

    print("Class Balance Check:")
    print(f"  - Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print("  ⚠ Warning: High imbalance detected.")
        print(f"    Consider collecting more samples for: {[k for k, v in label_counts.items() if v < max_samples // 2]}")
    else:
        print("  ✓ Classes are reasonably balanced")

    if min_samples < 10:
        print(f"{'='*60}")
        print("⚠ DATA QUALITY WARNING")
        print(f"{'='*60}")
        print(f"  Minimum samples per class: {min_samples}")
        print(f"  Classes with <10 samples: {[k for k, v in label_counts.items() if v < 10]}")
        print("  Collect at least 50-100 samples per class for reliable training.")
        print(f"{'='*60}\n")
    elif min_samples < 50:
        print("⚠ Note: Some classes have fewer than 50 samples. More data may improve accuracy.\n")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)

    print(f"{'='*60}")
    print("Splitting dataset...")
    print(f"{'='*60}\n")

    min_class = int(np.bincount(y).min())
    strat = y if min_class >= 2 else None
    if min_class < 2:
        print("⚠ Warning: Some classes have <2 samples; stratification disabled.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    print(f"✓ Training set : {len(X_train)} samples")
    print(f"✓ Test set     : {len(X_test)} samples\n")

    print(f"{'='*60}")
    print("Training RandomForest Classifier")
    print(f"  n_estimators     : {n_estimators} trees")
    print(f"  min_samples_split: {min_samples_split}")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  Classes          : {num_classes}")
    print(f"{'='*60}\n")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=min_samples_split,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )

    model.fit(X_train, y_train)
    print("✓ Training complete\n")

    print(f"{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}\n")

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print("=" * 60)
    unique_test = np.unique(y_test)
    target_names = label_encoder.classes_[unique_test]
    print(classification_report(y_test, y_pred, labels=unique_test, target_names=target_names))

    print("\nConfusion Matrix (Test Set):")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    if cm_off.max() > 0:
        print("\nMost Confused Gesture Pairs:")
        print("-" * 40)
        flat_indices = np.argsort(cm_off.ravel())[::-1]
        shown = 0
        for flat_idx in flat_indices:
            i, j = divmod(flat_idx, cm_off.shape[1])
            if cm_off[i, j] > 0 and shown < 5:
                gi = label_encoder.classes_[unique_test[i]] if i < len(unique_test) else '?'
                gj = label_encoder.classes_[unique_test[j]] if j < len(unique_test) else '?'
                print(f"  {gi} misclassified as {gj}: {cm_off[i, j]} times")
                shown += 1
    print()

    print(f"{'='*60}")
    print("Saving Model...")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(full_model_path), exist_ok=True)

    model_data = {
        'model_type': 'RandomForest_Static',
        'model': model,
        'label_encoder': label_encoder,
        'feature_size': X_train.shape[1],
        'num_features': X_train.shape[1],
        'uses_engineered_features': False,
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
    }

    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Model saved to: {full_model_path}")
    print("✓ Model includes:")
    print(f"  - RandomForest ({n_estimators} trees)")
    print(f"  - Label encoder ({num_classes} discovered classes)")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"{'='*60}")
    print(f"{save_label} Training Complete!")
    print(f"{'='*60}\n")
def train_static_model(landmarks_csv='dataset/landmarks.csv',
                       model_output='models/static_classifier.pkl',
                       test_size=0.2, random_state=42):
    """Train the unified static RandomForest classifier on every discovered label."""
    _train_static_random_forest(
        landmarks_csv=landmarks_csv,
        model_output=model_output,
        test_size=test_size,
        random_state=random_state,
        n_estimators=300,
        min_samples_split=3,
        training_title='Training Unified Static Classifier - RandomForest',
        save_label='Unified Static Classifier',
    )


def train_combined_static_model(landmarks_csv='dataset/landmarks.csv',
                               model_output='models/static_classifier.pkl',
                               test_size=0.2, random_state=42):
    """Backward-compatible alias for the unified static classifier."""
    train_static_model(
        landmarks_csv=landmarks_csv,
        model_output=model_output,
        test_size=test_size,
        random_state=random_state,
    )


def main():
    """Run the model training script with unified static or dynamic training."""
    print("\n" + "=" * 60)
    print("ISL Gesture Recognition - Model Training")
    print("=" * 60)

    print("\nChoose which model to train:")
    print("  1. Static Gestures — train on all discovered labels in dataset/raw_images/")
    print("  2. Dynamic Words  — train on motion clips in dataset/raw_clips/")

    mode = input("\nTraining mode (1/2, default: 1): ").strip() or "1"

    test_size_input = input("\nEnter test set size (0-1, default: 0.2): ").strip()
    try:
        test_size = float(test_size_input) if test_size_input else 0.2
        if not (0 < test_size < 1):
            print("Warning: Test size must be between 0 and 1. Using default (0.2)")
            test_size = 0.2
    except ValueError:
        print("Warning: Invalid input. Using default test size (0.2)")
        test_size = 0.2

    if mode == "2":
        print("\nTraining Dynamic Gesture Classifier (BiLSTM)...")
        train_model(test_size=test_size)
    else:
        print("\nTraining Unified Static Gesture Classifier...")
        train_static_model(test_size=test_size)


if __name__ == "__main__":
    main()
