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

from utils.mediapipe_utils import (
    HandDetector,
    aspect_aware_padding,
    compute_engineered_features,
    compute_hand_bounding_box,
    get_engineered_feature_names,
)


# Number of base engineered features per single frame
_BASE_FEATURES = len(get_engineered_feature_names())
# Unified feature vector: mean + std across frames (works for 1-frame static images too)
UNIFIED_FEATURES = _BASE_FEATURES * 2

# Maximum sequence length for BiLSTM training (frames per clip)
MAX_SEQ_FRAMES = 30

MIN_IMAGE_DIMENSION = 96
MIN_LAPLACIAN_VARIANCE = 18.0
MIN_BRIGHTNESS = 28.0
MAX_BRIGHTNESS = 235.0
MIN_CONTRAST = 12.0
MIN_HAND_AREA_RATIO = 0.008
MAX_HAND_AREA_RATIO = 0.75
MIN_COVERAGE_RATIO = 0.90
MIN_HAND_CONFIDENCE = 0.45
MAX_RETRY_CANDIDATES = 6


def _two_hand_feature_vector(detector, image, use_normalized):
    """Extract up to two hands from one image and concatenate engineered features."""
    _, results = detector.find_hands(image, draw=False)
    if use_normalized:
        hand_landmarks = detector.extract_landmarks_normalized(results, image.shape, hand_index=None)
    else:
        hand_landmarks = detector.extract_landmarks(results, hand_index=None)

    if not hand_landmarks:
        return None

    if not isinstance(hand_landmarks, list):
        hand_landmarks = [hand_landmarks]

    hand_features = [compute_engineered_features(np.asarray(hand, dtype=np.float32))
                     for hand in hand_landmarks[:2]]
    if not hand_features:
        return None

    while len(hand_features) < 2:
        hand_features.append(np.zeros(_BASE_FEATURES, dtype=np.float32))

    return np.concatenate(hand_features)


def _image_quality_metrics(image):
    """Return inexpensive quality signals used to guide preprocessing retries."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return {
        'height': height,
        'width': width,
        'min_dim': min(height, width),
        'brightness': float(gray.mean()),
        'contrast': float(gray.std()),
        'blur': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _enhance_image(image):
    """Build a small set of recoverable preprocessing variants."""
    variants = []

    variants.append(('original', image))
    variants.append(('letterbox', aspect_aware_padding(image)[0]))
    variants.append(('bright_contrast', cv2.convertScaleAbs(image, alpha=1.25, beta=25)))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    clahe_image = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    variants.append(('clahe', clahe_image))

    denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
    variants.append(('denoised_sharpened', sharpened))

    upscaled = cv2.resize(image, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
    variants.append(('upscaled', upscaled))

    return variants[:MAX_RETRY_CANDIDATES]


def _hand_detection_stats(results, raw_landmarks):
    """Compute confidence and coverage checks for a detected hand."""
    if results.multi_handedness and len(results.multi_handedness) > 0:
        confidence = float(results.multi_handedness[0].classification[0].score)
    else:
        confidence = 0.0

    coverage_ratio = 0.0
    bbox = None
    if raw_landmarks is not None:
        bbox = compute_hand_bounding_box(raw_landmarks)
        if bbox is not None:
            points = raw_landmarks.reshape(21, 3)
            inside_x = np.logical_and(points[:, 0] >= 0.0, points[:, 0] <= 1.0)
            inside_y = np.logical_and(points[:, 1] >= 0.0, points[:, 1] <= 1.0)
            coverage_ratio = float(np.mean(np.logical_and(inside_x, inside_y)))

    return confidence, coverage_ratio, bbox


def _crop_around_hand(image, raw_landmarks, padding=0.30):
    """Crop around the detected hand while keeping a safety margin."""
    if raw_landmarks is None:
        return image

    bbox = compute_hand_bounding_box(raw_landmarks)
    if bbox is None:
        return image

    height, width = image.shape[:2]
    x_min = max(0, int((bbox['x_min'] - padding) * width))
    y_min = max(0, int((bbox['y_min'] - padding) * height))
    x_max = min(width, int((bbox['x_max'] + padding) * width))
    y_max = min(height, int((bbox['y_max'] + padding) * height))

    if x_max <= x_min or y_max <= y_min:
        return image

    return image[y_min:y_max, x_min:x_max]


def _extract_from_image(detector, image, use_normalized):
    """Extract landmarks from one image variant and validate quality."""
    _, results = detector.find_hands(image, draw=False)
    raw_landmarks = detector.extract_landmarks(results)
    if raw_landmarks is None:
        return None, None, None

    confidence, coverage_ratio, bbox = _hand_detection_stats(results, raw_landmarks)
    if confidence < MIN_HAND_CONFIDENCE:
        return None, None, None
    if coverage_ratio < MIN_COVERAGE_RATIO:
        return None, None, None

    if bbox is not None:
        area_ratio = bbox['area']
        if area_ratio < MIN_HAND_AREA_RATIO or area_ratio > MAX_HAND_AREA_RATIO:
            return None, None, None
        if bbox['x_min'] < -0.03 or bbox['y_min'] < -0.03 or bbox['x_max'] > 1.03 or bbox['y_max'] > 1.03:
            return None, None, None

    landmarks = (
        detector.extract_landmarks_normalized(results, image.shape)
        if use_normalized else raw_landmarks
    )
    if landmarks is None:
        return None, None, None

    return landmarks, raw_landmarks, results


def _frame_feature(detector, image, use_normalized, combine_two_hands=False):
    """Extract engineered features from a single image, with recovery retries."""
    if image is None:
        return None

    quality = _image_quality_metrics(image)
    if quality['min_dim'] < MIN_IMAGE_DIMENSION:
        image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        quality = _image_quality_metrics(image)

    candidates = _enhance_image(image)
    if quality['brightness'] < MIN_BRIGHTNESS or quality['brightness'] > MAX_BRIGHTNESS:
        candidates = candidates[1:] + candidates[:1]
    elif quality['contrast'] < MIN_CONTRAST or quality['blur'] < MIN_LAPLACIAN_VARIANCE:
        candidates = candidates[1:] + candidates[:1]

    if combine_two_hands:
        for _, candidate in candidates:
            feat = _two_hand_feature_vector(detector, candidate, use_normalized)
            if feat is not None:
                return feat
        return None

    for _, candidate in candidates:
        landmarks, raw_landmarks, _ = _extract_from_image(detector, candidate, use_normalized)
        if landmarks is not None:
            return compute_engineered_features(landmarks)

        _, results = detector.find_hands(candidate, draw=False)
        raw_candidate = detector.extract_landmarks(results)
        if raw_candidate is None:
            continue

        cropped = _crop_around_hand(candidate, raw_candidate)
        if cropped is candidate or cropped.size == 0:
            continue

        landmarks, _, _ = _extract_from_image(detector, cropped, use_normalized)
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
        [f for f in os.listdir(clip_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
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
    Process a single static image and return a two-hand feature vector of
    length UNIFIED_FEATURES (one engineered vector per hand).
    """
    feat = _frame_feature(detector, image, use_normalized, combine_two_hands=True)
    if feat is None:
        return None
    return feat


# ── Sequence extraction helpers (for BiLSTM) ─────────────────────────────────

def _clip_to_sequence(detector, clip_dir, use_normalized):
    """
    Process a clip folder and return a zero-padded sequence array of shape
    (MAX_SEQ_FRAMES, _BASE_FEATURES).  Returns None if no frames yielded landmarks.
    """
    frame_files = sorted(
        [f for f in os.listdir(clip_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
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


def extract_sequences_from_dataset(dataset_path='dataset/raw_clips',
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

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_images_path = os.path.join(script_dir, 'dataset', 'raw_images')
    raw_clips_path = os.path.join(script_dir, 'dataset', 'raw_clips')

    ran_anything = False

    if os.path.exists(raw_images_path):
        print("\nStatic dataset detected. Extracting landmarks from dataset/raw_images/...")
        extract_landmarks_from_dataset(
            dataset_path='dataset/raw_images',
            output_csv='dataset/landmarks.csv',
            use_normalized=True,
        )
        ran_anything = True

    if os.path.exists(raw_clips_path):
        print("\nDynamic dataset detected. Extracting sequences from dataset/raw_clips/...")
        extract_sequences_from_dataset(
            dataset_path='dataset/raw_clips',
            output_npz='dataset/sequences.npz',
            use_normalized=True,
        )
        ran_anything = True

    if not ran_anything:
        print("Error: No dataset folders found. Create dataset/raw_images/ or dataset/raw_clips/ first.")


if __name__ == "__main__":
    main()
