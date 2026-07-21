"""
Model Training Script for ISL Gesture Recognition.

This script now supports two independent training modes:
- alphabet: static gestures (A-Z and 0-9)
- word: dynamic word gestures
"""

import argparse
import os
import sys
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Add parent directory to path AT THE FRONT (index 0) to avoid 'utils' naming conflicts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_utils.logger import logger

MAX_SEQ_FRAMES = 30
STATIC_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
LEGACY_SEQUENCES = 'dataset/sequences.npz'
MODE_CONFIGS = {
    'alphabet': {
        'dataset': 'dataset/alphabet_sequences.npz',
        'pkl': 'models/alphabet_model.pkl',
        'keras': 'models/alphabet_model.keras',
        'title': 'Static Gesture Recognition',
    },
    'word': {
        'dataset': 'dataset/word_sequences.npz',
        'pkl': 'models/word_model.pkl',
        'keras': 'models/word_model.keras',
        'title': 'Dynamic Word Recognition',
    },
}


def _normalize_mode(mode):
    """Return a supported mode name and fail fast on invalid input."""
    normalized = (mode or 'alphabet').strip().lower()
    if normalized not in MODE_CONFIGS:
        raise ValueError(f"Unsupported mode: {mode}. Use 'alphabet' or 'word'.")
    return normalized


def _is_static_label(label):
    """Return True when the label belongs to the static A-Z / 0-9 task."""
    label = str(label).strip().upper()
    return len(label) == 1 and label in STATIC_LABELS


def _mode_config(mode):
    """Resolve paths and human-readable labels for the selected mode."""
    normalized = _normalize_mode(mode)
    return normalized, MODE_CONFIGS[normalized]


def _is_mode_label(label, mode):
    """Return True when a label belongs to the requested training task."""
    return _is_static_label(label) if mode == 'alphabet' else not _is_static_label(label)


def _load_sequences_for_mode(script_dir, mode, preferred_rel_path):
    """
    Load the dataset for the requested mode.

    If the split file is missing, fall back to the legacy mixed sequences file,
    filter it down to the requested mode, and materialize the split file so the
    new layout is created automatically.
    """
    preferred_full_path = os.path.join(script_dir, preferred_rel_path)
    if os.path.exists(preferred_full_path):
        return preferred_full_path, np.load(preferred_full_path, allow_pickle=True)

    legacy_full_path = os.path.join(script_dir, LEGACY_SEQUENCES)
    if not os.path.exists(legacy_full_path):
        return preferred_full_path, None

    legacy_data = np.load(legacy_full_path, allow_pickle=True)
    raw_labels = legacy_data['labels'].astype(str)
    keep_mask = np.array([_is_mode_label(label, mode) for label in raw_labels], dtype=bool)

    if not keep_mask.any():
        return preferred_full_path, None

    filtered_X = legacy_data['X'][keep_mask].astype(np.float32)
    filtered_labels = raw_labels[keep_mask].astype(str)

    os.makedirs(os.path.dirname(preferred_full_path), exist_ok=True)
    np.savez_compressed(preferred_full_path, X=filtered_X, labels=filtered_labels)
    print(f"Warning: {os.path.basename(preferred_full_path)} was missing, so it was built from the legacy mixed dataset.")
    print(f"  Saved filtered subset to: {preferred_full_path}")
    return preferred_full_path, np.load(preferred_full_path, allow_pickle=True)


def train_model(mode='alphabet',
                sequences_npz=None,
                model_output=None,
                keras_model_output=None,
                test_size=0.2, random_state=42):
    """
    Train a Bidirectional LSTM classifier on the landmark sequences dataset.

    Args:
        mode (str): 'alphabet' for static gestures or 'word' for dynamic words
        sequences_npz (str): Optional path to the sequences .npz file
        model_output (str): Optional path to save the metadata pickle
        keras_model_output (str): Optional path to save the Keras model
        test_size (float): Proportion of dataset to use as test set
        random_state (int): Random seed for reproducibility
    """
    mode, config = _mode_config(mode)

    sequences_npz = sequences_npz or config['dataset']
    model_output = model_output or config['pkl']
    keras_model_output = keras_model_output or config['keras']

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_model_path = os.path.join(script_dir, model_output)
    full_keras_path = os.path.join(script_dir, keras_model_output)

    full_npz_path, data = _load_sequences_for_mode(script_dir, mode, sequences_npz)
    if data is None:
        print(f"Error: Sequences file not found: {full_npz_path}")
        if mode == 'alphabet':
            print("Please run extract_landmarks.py with the static/alphabet mode first.")
        else:
            print("Please run extract_landmarks.py with the dynamic/word mode first.")
        return

    print(f"\n{'='*60}")
    print(f"Training ISL {config['title']} - Bidirectional LSTM")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────
    print(f"Loading sequences from: {full_npz_path}")
    X = data['X'].astype(np.float32)          # (N, SEQ_LEN, FEAT_SIZE)
    raw_labels = data['labels'].astype(str)

    invalid_labels = sorted({label for label in np.unique(raw_labels) if not _is_mode_label(label, mode)})

    if invalid_labels:
        print("Error: The selected dataset contains labels from the wrong task.")
        print(f"  Mode   : {mode}")
        print(f"  Invalid: {', '.join(invalid_labels)}")
        print("Please regenerate the dataset with the matching extraction mode.")
        return

    print(f"✓ Dataset loaded: {X.shape[0]} samples, "
          f"sequence length {X.shape[1]}, {X.shape[2]} features per frame")

    label_counts = Counter(raw_labels)
    print(f"\nLabel distribution:")
    for lbl in sorted(label_counts.keys(), key=str):
        print(f"  {lbl:20s}: {label_counts[lbl]}")
    print()

    min_samples = min(label_counts.values())
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
        epochs=100,
        batch_size=32,
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
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes))
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

    # Create explicit class index mappings to ensure consistent ordering
    index_to_class = list(label_encoder.classes_)
    class_to_index = {c: i for i, c in enumerate(index_to_class)}

    model_data = {
        'model_type': 'BiLSTM',
        'gesture_mode': mode,
        'keras_model_path': keras_model_output,
        'label_encoder': label_encoder,
        'index_to_class': index_to_class,
        'class_to_index': class_to_index,
        'max_seq_frames': int(seq_len),
        'feature_size': int(feat_size),
        'num_features': int(feat_size),    # kept for backward-compat detection
        'uses_engineered_features': True,
        'test_accuracy': test_accuracy,
        'val_accuracy': best_val_acc,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': index_to_class,
    }

    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"✓ Keras model saved to : {full_keras_path}")
    logger.info(f"✓ Metadata saved to    : {full_model_path}")
    print(f"✓ Model includes:")
    print(f"  - Bidirectional LSTM (2 layers)")
    print(f"  - Label encoder ({num_classes} classes)")
    print(f"  - Input shape: ({seq_len}, {feat_size})")
    print(f"  - Mode: {mode}")
    print(f"  - Validation accuracy: {best_val_acc:.2%}")
    print(f"  - Test accuracy: {test_accuracy:.2%}")
    print()

    print(f"{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


def main():
    """
    Main function to run the model training script.
    """
    parser = argparse.ArgumentParser(description="Train ISL gesture models")
    parser.add_argument(
        '--mode',
        choices=sorted(MODE_CONFIGS.keys()),
        default=None,
        help="Training mode: alphabet for static gestures or word for dynamic words",
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=None,
        help="Optional test set size. If omitted, the script asks interactively.",
    )
    args = parser.parse_args()

    if args.mode is None:
        print("\nChoose training mode:")
        print("  1. Static gestures  - alphabet_model")
        print("  2. Dynamic words    - word_model")
        mode_choice = input("Training mode (1/2, default: 2): ").strip()
        args.mode = 'alphabet' if mode_choice == '1' else 'word'

    print("\n" + "=" * 60)
    print(f"ISL Gesture Recognition - {MODE_CONFIGS[args.mode]['title']}")
    print("=" * 60)

    if args.test_size is None:
        test_size_input = input("\nEnter test set size (0-1, default: 0.2): ").strip()
        try:
            test_size = float(test_size_input) if test_size_input else 0.2
            if not (0 < test_size < 1):
                print("Warning: Test size must be between 0 and 1. Using default (0.2)")
                test_size = 0.2
        except ValueError:
            print("Warning: Invalid input. Using default test size (0.2)")
            test_size = 0.2
    else:
        test_size = args.test_size

    train_model(mode=args.mode, test_size=test_size)


if __name__ == "__main__":
    main()
