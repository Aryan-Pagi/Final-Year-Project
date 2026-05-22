"""
Landmark Extraction Script for ISL Gesture Recognition
This script reads images from the dataset and extracts hand landmarks using MediaPipe.
The landmarks are saved to a CSV file for training.
"""

import cv2
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import HandDetector, compute_engineered_features, get_engineered_feature_names


# Number of base engineered features per single frame
_BASE_FEATURES = len(get_engineered_feature_names())
# Unified feature vector: mean + std across frames (works for 1-frame static images too)
UNIFIED_FEATURES = _BASE_FEATURES * 2

# Maximum sequence length for BiLSTM training (frames per clip)
MAX_SEQ_FRAMES = 30


def _frame_feature(detector, image, use_normalized):
    """Extract engineered features from a single image, returns None on failure."""
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=50)
    _, results = detector.find_hands(enhanced, draw=False)
    landmarks = (
        detector.extract_landmarks_normalized(results, enhanced.shape)
        if use_normalized else
        detector.extract_landmarks(results)
    )
    if landmarks is None:
        _, results = detector.find_hands(image, draw=False)
        landmarks = (
            detector.extract_landmarks_normalized(results, image.shape)
            if use_normalized else
            detector.extract_landmarks(results)
        )
    if landmarks is not None:
        return compute_engineered_features(landmarks)
    return None


def _clip_to_feature(detector, clip_dir, use_normalized):
    """
    Process a clip folder (clip_N/frame_M.jpg) and return a unified feature
    vector of length UNIFIED_FEATURES (mean + std of per-frame features).
    Returns None if no frames yielded landmarks.
    """
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.lower().endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0
    )
    per_frame = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is None:
            continue
        feat = _frame_feature(detector, img, use_normalized)
        if feat is not None:
            per_frame.append(feat)

    if not per_frame:
        return None
    arr = np.array(per_frame)          # shape (T, _BASE_FEATURES)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])


def _static_to_feature(detector, image, use_normalized):
    """
    Process a single static image and return a unified feature vector of
    length UNIFIED_FEATURES (feature values + zeros for the std half).
    """
    feat = _frame_feature(detector, image, use_normalized)
    if feat is None:
        return None
    # Pad with zeros for the std half so vector length matches clips
    return np.concatenate([feat, np.zeros(_BASE_FEATURES)])


# ── Sequence extraction helpers (for BiLSTM) ─────────────────────────────────

def _clip_to_sequence(detector, clip_dir, use_normalized):
    """
    Process a clip folder and return a zero-padded sequence array of shape
    (MAX_SEQ_FRAMES, _BASE_FEATURES).  Returns None if no frames yielded landmarks.
    """
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.lower().endswith('.jpg')],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0
    )
    per_frame = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is None:
            continue
        feat = _frame_feature(detector, img, use_normalized)
        if feat is not None:
            per_frame.append(feat)

    if not per_frame:
        return None

    arr = np.array(per_frame, dtype=np.float32)   # (T, _BASE_FEATURES)
    # Truncate to MAX_SEQ_FRAMES
    if len(arr) > MAX_SEQ_FRAMES:
        arr = arr[:MAX_SEQ_FRAMES]
    # Zero-pad if shorter
    if len(arr) < MAX_SEQ_FRAMES:
        pad = np.zeros((MAX_SEQ_FRAMES - len(arr), _BASE_FEATURES), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return arr  # (MAX_SEQ_FRAMES, _BASE_FEATURES)


def _static_to_sequence(detector, image, use_normalized):
    """
    Process a single static image and replicate it to shape
    (MAX_SEQ_FRAMES, _BASE_FEATURES).  Returns None if hand not detected.
    """
    feat = _frame_feature(detector, image, use_normalized)
    if feat is None:
        return None
    return np.tile(feat.astype(np.float32), (MAX_SEQ_FRAMES, 1))  # (MAX_SEQ_FRAMES, _BASE_FEATURES)


def extract_landmarks_from_dataset(dataset_path='dataset/raw_images', 
                                   output_csv='dataset/landmarks.csv',
                                   use_normalized=True,
                                   target_labels=None,
                                   max_samples_per_label=None):
    """
    Extract hand landmarks from all images in the dataset.
    
    Args:
        dataset_path (str): Path to the raw images directory
        output_csv (str): Path to save the landmarks CSV file
        use_normalized (bool): Whether to use normalized landmarks
        target_labels (list): Optional list of labels to include
        max_samples_per_label (int): Optional cap on samples per gesture
    """
    # Get absolute paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_csv)
    
    if not os.path.exists(full_dataset_path):
        print(f"Error: Dataset path does not exist: {full_dataset_path}")
        return
    
    # Initialize hand detector
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    # Prepare data storage
    all_landmarks = []
    all_labels = []

    # Get all label directories
    label_dirs = [d for d in os.listdir(full_dataset_path)
                  if os.path.isdir(os.path.join(full_dataset_path, d))]

    if target_labels:
        label_dirs = [d for d in label_dirs if d in target_labels]

    if not label_dirs:
        print(f"Error: No label directories found in {full_dataset_path}")
        print("Please collect data first using collect_data.py")
        return

    print(f"\n{'='*60}")
    print("Extracting Landmarks from Dataset")
    print(f"{'='*60}")
    print(f"Dataset path: {full_dataset_path}")
    print(f"Found {len(label_dirs)} label(s): {', '.join(sorted(label_dirs))}")
    print(f"Feature vector size: {UNIFIED_FEATURES} (mean+std of {_BASE_FEATURES} features)")
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    failed_images = []
    label_sample_counts = {}

    # Process each label directory
    for label in sorted(label_dirs):
        label_path = os.path.join(full_dataset_path, label)

        # Detect whether this label uses clip sub-folders or flat images
        clip_dirs = sorted(
            [d for d in os.listdir(label_path)
             if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')]
        )
        image_files = [f for f in os.listdir(label_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_sample_counts[label] = 0

        if clip_dirs:
            # ── Video-sequence label ──────────────────────────────────
            print(f"Processing label '{label}': {len(clip_dirs)} video clips")
            for clip_name in tqdm(clip_dirs, desc=f"  {label}", unit="clip"):
                clip_path = os.path.join(label_path, clip_name)
                feat = _clip_to_feature(detector, clip_path, use_normalized)
                if feat is not None:
                    all_landmarks.append(feat)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break # Stop processing clips for this label
                else:
                    total_skipped += 1
                    failed_images.append(f"{label}/{clip_name}")

        elif image_files:
            # ── Static-image label ────────────────────────────────────
            print(f"Processing label '{label}': {len(image_files)} images")
            for img_file in tqdm(image_files, desc=f"  {label}", unit="img"):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"  Warning: Could not read image: {img_file}")
                    total_skipped += 1
                    continue
                feat = _static_to_feature(detector, image, use_normalized)
                if feat is not None:
                    all_landmarks.append(feat)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break # Stop processing images for this label
                else:
                    total_skipped += 1
                    failed_images.append(f"{label}/{img_file}")
        else:
            print(f"Warning: No images or clips found for label '{label}' — skipping")
    
    # Close detector
    detector.close()
    
    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped (no hand detected): {total_skipped}")
    print(f"Success rate: {(total_processed/(total_processed+total_skipped)*100):.1f}%")
    print(f"{'='*60}\n")
    
    if total_skipped > 0:
        print(f"⚠ Warning: {total_skipped} images failed hand detection.")
        if total_skipped <= 20:
            print("Failed images:")
            for failed in failed_images[:20]:
                print(f"  - {failed}")
        else:
            print(f"Showing first 20 failed images:")
            for failed in failed_images[:20]:
                print(f"  - {failed}")
            print(f"  ... and {total_skipped - 20} more")
        print("\nPossible reasons:")
        print("  - Hand not clearly visible in frame")
        print("  - Poor lighting or image quality")
        print("  - Hand too small or too large in frame")
        print("  - Non-hand gestures in image")
        print(f"{'='*60}\n")
    
    if total_processed == 0:
        print("Error: No landmarks extracted. Please check your dataset.")
        return
    
    # Create DataFrame with unified feature columns
    columns = ([f"{n}_mean" for n in get_engineered_feature_names()]
               + [f"{n}_std" for n in get_engineered_feature_names()])
    columns.append('label')
    
    # Combine landmarks and labels
    data = np.column_stack((np.array(all_landmarks), np.array(all_labels)))
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns to float
    for col in columns[:-1]:  # All except 'label'
        df[col] = df[col].astype(float)
    
    # Ensure labels are strings
    df['label'] = df['label'].astype(str)
    
    # Save to CSV
    df.to_csv(full_output_path, index=False)
    
    print(f"✓ Landmarks saved to: {full_output_path}")
    print(f"✓ Dataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    # Sort labels naturally
    label_counts = df['label'].value_counts()
    label_counts = label_counts.reindex(sorted(label_counts.index, key=str))
    print(label_counts)
    print()


def extract_sequences_from_dataset(dataset_path='dataset/raw_images',
                                   output_npz='dataset/sequences.npz',
                                   use_normalized=True,
                                   target_labels=None,
                                   max_samples_per_label=None):
    """
    Extract per-frame landmark sequences from all clips/images and save as a
    compressed .npz file for Bidirectional LSTM training.

    The .npz contains:
        X      : float32 array of shape (N, MAX_SEQ_FRAMES, _BASE_FEATURES)
        labels : str array of shape (N,)
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_npz)

    if not os.path.exists(full_dataset_path):
        print(f"Error: Dataset path does not exist: {full_dataset_path}")
        return

    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    all_sequences = []
    all_labels = []

    label_dirs = [d for d in os.listdir(full_dataset_path)
                  if os.path.isdir(os.path.join(full_dataset_path, d))]

    if target_labels:
        label_dirs = [d for d in label_dirs if d in target_labels]

    if not label_dirs:
        print(f"Error: No label directories found in {full_dataset_path}")
        return

    print(f"\n{'='*60}")
    print("Extracting Landmark Sequences (for BiLSTM)")
    print(f"{'='*60}")
    print(f"Dataset path  : {full_dataset_path}")
    print(f"Sequence shape: ({MAX_SEQ_FRAMES}, {_BASE_FEATURES})")
    print(f"Labels found  : {', '.join(sorted(label_dirs))}")
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    label_sample_counts = {}

    for label in sorted(label_dirs):
        label_path = os.path.join(full_dataset_path, label)

        clip_dirs = sorted(
            [d for d in os.listdir(label_path)
             if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')]
        )
        image_files = [f for f in os.listdir(label_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_sample_counts[label] = 0

        if clip_dirs:
            print(f"Processing '{label}': {len(clip_dirs)} video clips")
            for clip_name in tqdm(clip_dirs, desc=f"  {label}", unit="clip"):
                clip_path = os.path.join(label_path, clip_name)
                seq = _clip_to_sequence(detector, clip_path, use_normalized)
                if seq is not None:
                    all_sequences.append(seq)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break
                else:
                    total_skipped += 1

        elif image_files:
            print(f"Processing '{label}': {len(image_files)} images")
            for img_file in tqdm(image_files, desc=f"  {label}", unit="img"):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    total_skipped += 1
                    continue
                seq = _static_to_sequence(detector, image, use_normalized)
                if seq is not None:
                    all_sequences.append(seq)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break # Stop processing clips for this label
                else:
                    total_skipped += 1
        else:
            print(f"Warning: No images or clips found for label '{label}' — skipping")

    detector.close()

    print(f"\n{'='*60}")
    print("Sequence Extraction Summary")
    print(f"{'='*60}")
    print(f"Sequences extracted : {total_processed}")
    print(f"Samples skipped     : {total_skipped}")
    if total_processed + total_skipped > 0:
        print(f"Success rate        : {total_processed/(total_processed+total_skipped)*100:.1f}%")
    print(f"{'='*60}\n")

    if total_processed == 0:
        print("Error: No sequences extracted. Please check your dataset.")
        return

    X_array = np.array(all_sequences, dtype=np.float32)   # (N, MAX_SEQ_FRAMES, _BASE_FEATURES)
    labels_array = np.array(all_labels)

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    np.savez_compressed(full_output_path, X=X_array, labels=labels_array)

    print(f"✓ Sequences saved to : {full_output_path}")
    print(f"✓ Array shape        : {X_array.shape}  (samples, frames, features)")
    print(f"✓ Labels             : {len(np.unique(labels_array))} classes")
    print()


def main():
    """
    Main function to run the landmark extraction script.
    """
    print("\n" + "="*60)
    print("ISL Gesture Recognition - Landmark Extraction")
    print("="*60)

    print("\nExtraction mode:")
    print("  1) Sequence (.npz) — for Bidirectional LSTM  [default]")
    print("  2) Flat CSV        — for Random Forest / legacy models")
    mode_choice = input("Choose mode (1/2, default 1): ").strip()

    print("\nLabel Scope:")
    print("  1) All labels in dataset")
    print("  2) Digits only (0-9)")
    scope_choice = input("Choose scope (1/2, default 2): ").strip() or "2"
    target_labels = [str(i) for i in range(10)] if scope_choice == "2" else None

    max_samples_input = input("Cap samples per label (e.g., 100, or leave blank for no cap): ").strip()
    max_samples_per_label = None
    if max_samples_input:
        try:
            max_samples_per_label = int(max_samples_input)
            if max_samples_per_label <= 0:
                print("Warning: Max samples must be positive. No cap applied.")
                max_samples_per_label = None
        except ValueError:
            print("Warning: Invalid input for max samples. No cap applied.")

    normalize_choice = input("Use normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'

    if use_normalized:
        print("Using normalized landmarks (recommended)")
    else:
        print("Using raw landmarks")

    if mode_choice == '2':
        extract_landmarks_from_dataset(use_normalized=use_normalized, target_labels=target_labels, max_samples_per_label=max_samples_per_label)
    else:
        extract_sequences_from_dataset(use_normalized=use_normalized, target_labels=target_labels, max_samples_per_label=max_samples_per_label)


if __name__ == "__main__":
    main()
