"""
Model Training Script for ISL Gesture Recognition
This script trains a Bidirectional LSTM on the extracted landmark sequences.
"""

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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

MAX_SEQ_FRAMES = 30


def train_model(sequences_npz='dataset/sequences.npz',
                model_output='models/gesture_model.pkl',
                keras_model_output='models/bilstm_model.keras',
                test_size=0.2, random_state=42):
    """
    Train a Bidirectional LSTM classifier on the landmark sequences dataset.

    Args:
        sequences_npz (str): Path to the sequences .npz file produced by extract_landmarks.py
        model_output (str): Path to save the metadata pickle
        keras_model_output (str): Path to save the Keras model
        test_size (float): Proportion of dataset to use as test set
        random_state (int): Random seed for reproducibility
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_npz_path = os.path.join(script_dir, sequences_npz)
    full_model_path = os.path.join(script_dir, model_output)
    full_keras_path = os.path.join(script_dir, keras_model_output)

    if not os.path.exists(full_npz_path):
        print(f"Error: Sequences file not found: {full_npz_path}")
        print("Please run extract_landmarks.py first and choose sequence mode.")
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

    # Create explicit class index mappings to ensure consistent ordering
    index_to_class = list(label_encoder.classes_)
    class_to_index = {c: i for i, c in enumerate(index_to_class)}

    model_data = {
        'model_type': 'BiLSTM',
        'keras_model_path': 'models/bilstm_model.keras',
        'label_encoder': label_encoder,
        'index_to_class': index_to_class,
        'class_to_index': class_to_index,
        'max_seq_frames': int(seq_len),
        'feature_size': int(feat_size),
        'num_features': int(feat_size),    # kept for backward-compat detection
        'uses_engineered_features': True,
        'test_accuracy': test_accuracy,
        'val_accuracy': best_val_acc,
    }

    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"✓ Keras model saved to : {full_keras_path}")
    logger.info(f"✓ Metadata saved to    : {full_model_path}")
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


def main():
    """
    Main function to run the model training script.
    """
    print("\n" + "=" * 60)
    print("ISL Gesture Recognition - Bidirectional LSTM Training")
    print("=" * 60)

    test_size_input = input("\nEnter test set size (0-1, default: 0.2): ").strip()
    try:
        test_size = float(test_size_input) if test_size_input else 0.2
        if not (0 < test_size < 1):
            print("Warning: Test size must be between 0 and 1. Using default (0.2)")
            test_size = 0.2
    except ValueError:
        print("Warning: Invalid input. Using default test size (0.2)")
        test_size = 0.2

    train_model(test_size=test_size)


if __name__ == "__main__":
    main()
