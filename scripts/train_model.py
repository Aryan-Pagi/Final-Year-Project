"""
Model Training Script for ISL Gesture Recognition
This script trains models for gesture recognition:
- Static classifiers (RandomForest) for digits/letters (single-frame)
- Dynamic classifiers (BiLSTM) for motion-based word gestures
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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import get_engineered_feature_names

MAX_SEQ_FRAMES = 30
_BASE_FEATURES = len(get_engineered_feature_names())


def train_model(sequences_npz='dataset/sequences.npz',
                model_output='models/gesture_model.pkl',
                keras_model_output='models/bilstm_model.keras',
                test_size=0.2, random_state=42,
                epochs=10, batch_size=12):
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
        'uses_engineered_features': True,
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


def train_static_model(landmarks_csv='dataset/landmarks.csv',
                       model_output='models/static_classifier.pkl',
                       test_size=0.2, random_state=42):
    """
    Train a lightweight RandomForest classifier on static gesture landmarks.
    
    This classifier is optimized for single-frame static gestures (digits 0-9,
    letters A-Z, or still-pose words). It uses unified landmark features
    (mean + std across frames).
    
    Args:
        landmarks_csv (str): Path to the landmarks CSV produced by extract_landmarks.py
        model_output (str): Path to save the trained model and metadata
        test_size (float): Proportion of dataset to use as test set
        random_state (int): Random seed for reproducibility
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_csv_path = os.path.join(script_dir, landmarks_csv)
    full_model_path = os.path.join(script_dir, model_output)
    
    if not os.path.exists(full_csv_path):
        print(f"Error: Landmarks CSV not found: {full_csv_path}")
        print("Please run extract_landmarks.py first.")
        return
    
    print(f"\n{'='*60}")
    print("Training Static Gesture Classifier - RandomForest")
    print(f"{'='*60}\n")
    
    # ── Load dataset ──────────────────────────────────────────────
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
    
    # Extract features and labels
    # Expected CSV format: columns are features (x0, y0, z0, ...) and a 'label' column
    if 'label' not in df.columns:
        print("Error: CSV must have a 'label' column")
        return
    
    X = df.drop('label', axis=1).values.astype(np.float32)
    raw_labels = df['label'].values.astype(str)
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Loaded {len(np.unique(raw_labels))} gesture classes\n")
    
    label_counts = Counter(raw_labels)
    print(f"Label distribution:")
    for lbl in sorted(label_counts.keys(), key=str):
        print(f"  {lbl:20s}: {label_counts[lbl]:4d} samples")
    print()
    
    counts = list(label_counts.values())
    min_samples = min(counts)
    max_samples = max(counts)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0
    
    print(f"Class Balance Check:")
    print(f"  - Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print(f"  ⚠ Warning: High imbalance detected. "
              f"Recommend collecting more samples for: {[k for k, v in label_counts.items() if v < max_samples // 2]}")
    else:
        print(f"  ✓ Classes are well-balanced")
    
    if min_samples < 10:
        print(f"{'='*60}")
        print("⚠ DATA QUALITY WARNING")
        print(f"{'='*60}")
        print(f"  Minimum samples per class: {min_samples}")
        print(f"  Classes with <10 samples: "
              f"{[k for k, v in label_counts.items() if v < 10]}")
        print(f"  Collect at least 50-100 samples per gesture for reliable training.")
        print(f"{'='*60}\n")
    elif min_samples < 50:
        print(f"⚠ Note: Some gestures have fewer than 50 samples. More data may improve accuracy.\n")
    
    # ── Encode labels ─────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"{'='*60}")
    print("Splitting dataset...")
    print(f"{'='*60}\n")
    
    # Stratified split
    min_class = int(np.bincount(y).min())
    strat = y if min_class >= 2 else None
    if min_class < 2:
        print(f"⚠ Warning: Some classes have <2 samples; stratification disabled.\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    print(f"✓ Training set : {len(X_train)} samples")
    print(f"✓ Test set     : {len(X_test)} samples\n")
    
    # ── Build and train RandomForest ───────────────────────────────
    print(f"{'='*60}")
    print("Training RandomForest Classifier")
    print(f"  n_estimators  : 200 trees")
    print(f"  max_depth     : None (unlimited)")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Classes       : {num_classes}")
    print(f"{'='*60}\n")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Training complete\n")
    
    # ── Evaluation ────────────────────────────────────────────────
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
    
    # ── Save model and metadata ───────────────────────────────────
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
        'uses_engineered_features': True,
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
    }
    
    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Model saved to: {full_model_path}")
    print(f"✓ Model includes:")
    print(f"  - RandomForest (200 trees)")
    print(f"  - Label encoder ({num_classes} classes)")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"{'='*60}")
    print("Static Classifier Training Complete!")
    print(f"{'='*60}\n")


def train_combined_static_model(landmarks_csv='dataset/landmarks.csv',
                               model_output='models/static_classifier_full.pkl',
                               test_size=0.2, random_state=42):
    """
    Train a RandomForest classifier on combined digits (0-9) + letters (A-Z).
    
    This classifier combines Phase 1 (digits) and Phase 2 (letters) into a single
    36-class model. It requires data for both digits and letters to be collected.
    
    Args:
        landmarks_csv (str): Path to the landmarks CSV produced by extract_landmarks.py
        model_output (str): Path to save the trained combined model
        test_size (float): Proportion of dataset to use as test set
        random_state (int): Random seed for reproducibility
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_csv_path = os.path.join(script_dir, landmarks_csv)
    full_model_path = os.path.join(script_dir, model_output)
    
    if not os.path.exists(full_csv_path):
        print(f"Error: Landmarks CSV not found: {full_csv_path}")
        print("Please run extract_landmarks.py first.")
        return
    
    print(f"\n{'='*60}")
    print("Training Combined Static Classifier - Phase 2 (0-9 + A-Z)")
    print(f"{'='*60}\n")
    
    # ── Load dataset ──────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(full_csv_path)
        print(f"✓ Loaded {len(df)} total samples from {landmarks_csv}")
    except ImportError:
        print("Error: Pandas required for CSV loading. Install with: pip install pandas")
        return
    except Exception as e:
        print(f"Error loading landmarks CSV: {e}")
        return
    
    # Extract features and labels
    if 'label' not in df.columns:
        print("Error: CSV must have a 'label' column")
        return
    
    # Filter to only include digits (0-9) and letters (A-Z)
    valid_labels = (
        [str(i) for i in range(10)] +  # Digits 0-9
        [chr(i) for i in range(ord('A'), ord('Z')+1)]  # Letters A-Z
    )
    
    df_filtered = df[df['label'].isin(valid_labels)].copy()
    
    if len(df_filtered) == 0:
        print("Error: No valid digit or letter samples found in landmarks.csv")
        print(f"Expected labels: {', '.join(valid_labels[:15])}... (36 total)")
        return
    
    removed = len(df) - len(df_filtered)
    if removed > 0:
        print(f"⚠ Removed {removed} non-standard labels (keeping only 0-9, A-Z)")
    
    X = df_filtered.drop('label', axis=1).values.astype(np.float32)
    raw_labels = df_filtered['label'].values.astype(str)
    
    print(f"✓ Filtered feature matrix shape: {X.shape}")
    print(f"✓ Loaded {len(np.unique(raw_labels))} gesture classes\n")
    
    label_counts = Counter(raw_labels)
    print(f"Label distribution (showing first 20):")
    for lbl in sorted(label_counts.keys(), key=lambda x: (len(x), x)):
        print(f"  {lbl:20s}: {label_counts[lbl]:4d} samples")
    print()
    
    # Check that we have both digits and letters
    has_digits = any(lbl in label_counts for lbl in [str(i) for i in range(10)])
    has_letters = any(lbl in label_counts for lbl in [chr(i) for i in range(ord('A'), ord('Z')+1)])
    
    if not has_digits or not has_letters:
        print("⚠ Warning: Not all phases present!")
        if not has_digits:
            print("  - Missing digits (0-9). Collect digit data first for Phase 1.")
        if not has_letters:
            print("  - Missing letters (A-Z). Collect letter data to complete Phase 2.")
        proceed = input("\nContinue with partial data? (y/n, default: n): ").strip().lower()
        if proceed != 'y':
            return
    
    counts = list(label_counts.values())
    min_samples = min(counts)
    max_samples = max(counts)
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else 0
    
    print(f"Class Balance Check:")
    print(f"  - Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
    if imbalance_ratio > 2.0:
        print(f"  ⚠ Warning: Imbalanced data detected (ratio: {imbalance_ratio:.2f})")
        underrep = [k for k, v in label_counts.items() if v < max_samples // 2]
        print(f"     Under-represented: {', '.join(sorted(underrep))}")
    else:
        print(f"  ✓ Classes are well-balanced")
    
    if min_samples < 20:
        print(f"\n{'='*60}")
        print("⚠ DATA QUALITY WARNING")
        print(f"{'='*60}")
        print(f"  Minimum samples per class: {min_samples}")
        print(f"  Classes with <20 samples: "
              f"{[k for k, v in label_counts.items() if v < 20]}")
        print(f"  Recommend collecting 100+ samples per gesture for Phase 2.")
        print(f"{'='*60}\n")
    elif min_samples < 50:
        print(f"⚠ Note: Some gestures have <50 samples. More data may improve accuracy.\n")
    
    # ── Encode labels ─────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"{'='*60}")
    print(f"Phase 2 Summary: {num_classes} classes")
    print(f"  Digits: {sum(1 for lbl in label_encoder.classes_ if lbl in [str(i) for i in range(10)])}")
    print(f"  Letters: {sum(1 for lbl in label_encoder.classes_ if lbl in [chr(i) for i in range(ord('A'), ord('Z')+1)])}")
    print(f"{'='*60}\n")
    
    print(f"{'='*60}")
    print("Splitting dataset...")
    print(f"{'='*60}\n")
    
    # Stratified split
    min_class = int(np.bincount(y).min())
    strat = y if min_class >= 2 else None
    if min_class < 2:
        print(f"⚠ Warning: Some classes have <2 samples; stratification disabled.\n")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    print(f"✓ Training set : {len(X_train)} samples")
    print(f"✓ Test set     : {len(X_test)} samples\n")
    
    # ── Build and train RandomForest ───────────────────────────────
    print(f"{'='*60}")
    print("Training RandomForest Classifier (Phase 2 Parameters)")
    print(f"  n_estimators  : 300 trees (increased for 36 classes)")
    print(f"  min_samples_split: 3 (increased to reduce overfitting)")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Classes       : {num_classes}")
    print(f"{'='*60}\n")
    
    model = RandomForestClassifier(
        n_estimators=300,  # Increased from 200 for more classes
        max_depth=None,
        min_samples_split=3,  # Increased from 2 to reduce overfitting with more classes
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    print(f"✓ Training complete\n")
    
    # ── Evaluation ────────────────────────────────────────────────
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
    
    # ── Save model and metadata ───────────────────────────────────
    print(f"{'='*60}")
    print("Saving Combined Model...")
    print(f"{'='*60}\n")
    
    os.makedirs(os.path.dirname(full_model_path), exist_ok=True)
    
    model_data = {
        'model_type': 'RandomForest_Static',
        'model': model,
        'label_encoder': label_encoder,
        'feature_size': X_train.shape[1],
        'num_features': X_train.shape[1],
        'uses_engineered_features': True,
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
        'phase': 2,  # Phase 2: combined model
    }
    
    with open(full_model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Combined model saved to: {full_model_path}")
    print(f"✓ Model includes:")
    print(f"  - RandomForest (300 trees, Phase 2 parameters)")
    print(f"  - Label encoder ({num_classes} classes: 0-9, A-Z)")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Test accuracy: {test_accuracy:.2%}")
    print()
    
    print(f"{'='*60}")
    print("Phase 2 Combined Classifier Training Complete!")
    print(f"{'='*60}\n")
    print(f"Next: Use this model in real-time recognition!")
    print(f"  python scripts/realtime_predict.py")
    print()


def main():
    """
    Main function to run the model training script.
    """
    print("\n" + "=" * 60)
    print("ISL Gesture Recognition - Model Training")
    print("=" * 60)
    
    print("\nChoose which model to train:")
    print("  1a. Static Classifier (Digits 0-9 ONLY) — Phase 1 baseline")
    print("  1b. Static Classifier (Digits 0-9 + Letters A-Z) — Phase 2 combined")
    print("  2.  Dynamic Classifier (BiLSTM) — Motion-based word gestures")
    
    mode = input("\nTraining mode (1a/1b/2, default: 1b): ").strip().lower() or "1b"
    
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
    elif mode == "1a":
        print("\nTraining Static Gesture Classifier - Phase 1 (Digits 0-9 only)...")
        train_static_model(test_size=test_size)
    else:  # 1b or default
        print("\nTraining Static Gesture Classifier - Phase 2 (Digits + Letters Combined)...")
        train_combined_static_model(test_size=test_size)


if __name__ == "__main__":
    main()
