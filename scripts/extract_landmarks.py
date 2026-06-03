"""
Landmark Extraction Script for ISL Gesture Recognition
This script standardizes inputs to a square, extracts wrist-relative landmarks,
and saves either per-image features or fixed-length sequences.
"""

import cv2
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import HandDetector


MAX_SEQ_FRAMES = 30
TARGET_PREPROCESS_SIZE = 640
HAND_LANDMARKS = 21
HAND_FEATURES = HAND_LANDMARKS * 3
FEATURE_SIZE = HAND_FEATURES * 2


def standardize_image(image, target_size=TARGET_PREPROCESS_SIZE):
    """Letterbox an image into a fixed square without distorting aspect ratio."""
    if image is None:
        return None
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def extract_wrist_relative_features(results):
    """Return wrist-relative landmark coordinates for up to two hands."""
    if not results or not results.multi_hand_landmarks:
        return None
    features = np.zeros(FEATURE_SIZE, dtype=np.float32)
    ordered_hands = sorted(
        results.multi_hand_landmarks[:2],
        key=lambda hand_landmarks: hand_landmarks.landmark[0].x
    )
    for i, hand_landmarks in enumerate(ordered_hands):
        wrist = hand_landmarks.landmark[0]
        base_idx = i * HAND_FEATURES
        for j, lm in enumerate(hand_landmarks.landmark):
            idx = base_idx + (j * 3)
            features[idx] = lm.x - wrist.x
            features[idx + 1] = lm.y - wrist.y
            features[idx + 2] = lm.z - wrist.z
    return features


def _frame_feature(detector, image):
    """Standardize and extract wrist-relative features from one image frame."""
    standardized = standardize_image(image)
    if standardized is None:
        return None
    _, results = detector.find_hands(standardized, draw=False)
    return extract_wrist_relative_features(results)


def _clip_to_feature(detector, clip_dir):
    """Aggregate a clip into a single feature vector by averaging valid frames."""
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0
    )
    per_frame = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is None:
            continue
        feat = _frame_feature(detector, img)
        if feat is not None:
            per_frame.append(feat)
    if not per_frame:
        return None
    return np.mean(np.stack(per_frame, axis=0), axis=0)


def _static_to_feature(detector, image):
    """Extract a single wrist-relative feature vector from a static image."""
    return _frame_feature(detector, image)


def _clip_to_sequence(detector, clip_dir):
    """Build a fixed-length sequence of wrist-relative features."""
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(x.split('_')[1].split('.')[0]) if '_' in x else 0
    )
    per_frame = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(clip_dir, fname))
        if img is None:
            continue
        feat = _frame_feature(detector, img)
        if feat is not None:
            per_frame.append(feat)
    if not per_frame:
        return None
    arr = np.array(per_frame, dtype=np.float32)
    if len(arr) > MAX_SEQ_FRAMES:
        arr = arr[:MAX_SEQ_FRAMES]
    if len(arr) < MAX_SEQ_FRAMES:
        pad = np.zeros((MAX_SEQ_FRAMES - len(arr), FEATURE_SIZE), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return arr


def _static_to_sequence(detector, image):
    """Repeat a static frame to match sequence length."""
    feat = _frame_feature(detector, image)
    if feat is None:
        return None
    return np.tile(feat.astype(np.float32), (MAX_SEQ_FRAMES, 1))


def extract_landmarks_from_dataset(dataset_path='dataset/raw_images', 
                                   output_csv='dataset/landmarks.csv',
                                   use_normalized=True,
                                   target_labels=None,
                                   max_samples_per_label=None):
    """Extract hand landmarks from all structural directory targets across the flat layout."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_csv)
    
    if not os.path.exists(full_dataset_path):
        print(f"Error: Dataset path does not exist: {full_dataset_path}")
        return
    
    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=1
    )

    all_landmarks = []
    all_labels = []

    label_dirs = [d for d in os.listdir(full_dataset_path)
                  if os.path.isdir(os.path.join(full_dataset_path, d))]

    if target_labels:
        label_dirs = [d for d in label_dirs if d in target_labels]

    if not label_dirs:
        print(f"Error: No valid target directories found in {full_dataset_path}")
        return

    print(f"\n{'='*60}\nExtracting Landmarks from Dataset\n{'='*60}")

    total_processed = 0
    total_skipped = 0
    failed_images = []
    label_sample_counts = {}

    for label in sorted(label_dirs):
        label_path = os.path.join(full_dataset_path, label)
        clip_dirs = sorted([d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')])
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_sample_counts[label] = 0

        if clip_dirs:
            print(f"Processing label '{label}': {len(clip_dirs)} video clips")
            for clip_name in tqdm(clip_dirs, desc=f"  {label}", unit="clip"):
                clip_path = os.path.join(label_path, clip_name)
                feat = _clip_to_feature(detector, clip_path)
                if feat is not None:
                    all_landmarks.append(feat)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break
                else:
                    total_skipped += 1
                    failed_images.append(f"{label}/{clip_name}")

        elif image_files:
            print(f"Processing label '{label}': {len(image_files)} images")
            for img_file in tqdm(image_files, desc=f"  {label}", unit="img"):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    total_skipped += 1
                    continue
                feat = _static_to_feature(detector, image)
                if feat is not None:
                    all_landmarks.append(feat)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    total_processed += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break
                else:
                    total_skipped += 1
                    failed_images.append(f"{label}/{img_file}")

    detector.close()
    
    if total_processed == 0:
        print("Extraction failed. Verify image configurations.")
        return
    
    # Save processed items out to standardized target system tables
    columns = [f"feat_{i}" for i in range(FEATURE_SIZE)] + ['label']
    data = np.column_stack((np.array(all_landmarks), np.array(all_labels)))
    df = pd.DataFrame(data, columns=columns)
    
    for col in columns[:-1]:
        df[col] = df[col].astype(float)
    df['label'] = df['label'].astype(str)
    
    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    df.to_csv(full_output_path, index=False)
    print(f"✓ Output saved successfully to: {full_output_path}\nShape: {df.shape}")


def extract_sequences_from_dataset(dataset_path='dataset/raw_clips',
                                   output_npz='dataset/sequences.npz',
                                   use_normalized=True,
                                   target_labels=None,
                                   max_samples_per_label=None):
    """Extract per-frame landmark arrays for sequence classification models."""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_npz)

    if not os.path.exists(full_dataset_path):
        print(f"Error: Dataset path does not exist: {full_dataset_path}")
        return

    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        model_complexity=1
    )

    all_sequences = []
    all_labels = []
    label_dirs = [d for d in os.listdir(full_dataset_path) if os.path.isdir(os.path.join(full_dataset_path, d))]

    if target_labels:
        label_dirs = [d for d in label_dirs if d in target_labels]

    for label in sorted(label_dirs):
        label_path = os.path.join(full_dataset_path, label)
        clip_dirs = sorted([d for d in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')])
        image_files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        label_sample_counts = {label: 0}

        if clip_dirs:
            print(f"Processing '{label}': {len(clip_dirs)} sequences")
            for clip_name in tqdm(clip_dirs, desc=f"  {label}"):
                clip_path = os.path.join(label_path, clip_name)
                seq = _clip_to_sequence(detector, clip_path)
                if seq is not None:
                    all_sequences.append(seq)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break

        elif image_files:
            print(f"Processing '{label}': {len(image_files)} static targets")
            for img_file in tqdm(image_files, desc=f"  {label}"):
                img_path = os.path.join(label_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                seq = _static_to_sequence(detector, image)
                if seq is not None:
                    all_sequences.append(seq)
                    all_labels.append(label)
                    label_sample_counts[label] += 1
                    if max_samples_per_label and label_sample_counts[label] >= max_samples_per_label:
                        break

    detector.close()
    
    if not all_sequences:
        print("Zero sequences compiled.")
        return

    X_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels)

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    np.savez_compressed(full_output_path, X=X_array, labels=labels_array)
    print(f"✓ Sequence file saved successfully to: {full_output_path}\nMatrix Shape: {X_array.shape}")


def main():
    print("\n" + "="*60 + "\nISL Gesture Preprocessing & Landmark Extraction\n" + "="*60)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_images_path = os.path.join(script_dir, 'dataset', 'raw_images')
    raw_clips_path = os.path.join(script_dir, 'dataset', 'raw_clips')

    ran_anything = False

    if os.path.exists(raw_images_path):
        extract_landmarks_from_dataset(
            dataset_path='dataset/raw_images',
            output_csv='dataset/landmarks.csv',
            use_normalized=True,
        )
        ran_anything = True

    if os.path.exists(raw_clips_path):
        extract_sequences_from_dataset(
            dataset_path='dataset/raw_clips',
            output_npz='dataset/sequences.npz',
            use_normalized=True,
        )
        ran_anything = True

    if not ran_anything:
        print("Missing dataset paths: raw_images or raw_clips directories are not found.")


if __name__ == "__main__":
    main()