"""
Landmark Extraction Script for ISL Gesture Recognition.

This script now supports separate extraction outputs for:
- alphabet: static gestures (A-Z and 0-9)
- word: dynamic word gestures
"""

import argparse
import cv2
import json
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_utils.mediapipe_utils import HandDetector, compute_engineered_features, get_engineered_feature_names


# Number of base engineered features per single frame
_BASE_FEATURES = len(get_engineered_feature_names())
# Unified feature vector: mean + std across frames (works for 1-frame static images too)
UNIFIED_FEATURES = _BASE_FEATURES * 2

# Maximum sequence length for BiLSTM training (frames per clip)
MAX_SEQ_FRAMES = 30
STATIC_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
MODE_OUTPUTS = {
    'alphabet': 'dataset/alphabet_sequences.npz',
    'word': 'dataset/word_sequences.npz',
}


def _numeric_sort_key(name):
    """Sort names using a numeric suffix when available."""
    base = os.path.splitext(name)[0]
    if '_' in base:
        suffix = base.rsplit('_', 1)[-1]
        if suffix.isdigit():
            return (0, int(suffix), base.lower())
    return (1, base.lower())


def _sample_key(label, item_name):
    """Create a stable relative key for a raw dataset sample."""
    return f"{label}/{item_name}"


def _normalize_mode(mode):
    """Return a supported extraction mode."""
    normalized = (mode or 'alphabet').strip().lower()
    if normalized not in MODE_OUTPUTS:
        raise ValueError(f"Unsupported mode: {mode}. Use 'alphabet' or 'word'.")
    return normalized


def _is_static_label(label):
    """Return True when a label belongs to the static gesture set."""
    label = str(label).strip().upper()
    return len(label) == 1 and label in STATIC_LABELS


def _label_matches_mode(label, mode):
    """Decide whether a label belongs to the requested training task."""
    is_static = _is_static_label(label)
    if mode == 'alphabet':
        return is_static
    return not is_static


def _filter_samples_by_mode(samples_by_label, mode):
    """Keep only the labels that belong to the selected task."""
    filtered = {}
    for label, samples in samples_by_label.items():
        if _label_matches_mode(label, mode):
            filtered[label] = samples
    return filtered


def _manifest_path(output_path):
    """Return the sidecar manifest path for an output artifact."""
    root, _ = os.path.splitext(output_path)
    return f"{root}.processed.json"


def _load_manifest(manifest_path):
    """Load previously processed sample keys from a sidecar manifest."""
    if not os.path.exists(manifest_path):
        return []

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('entries', [])
        return [str(entry) for entry in entries]
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return []


def _save_manifest(manifest_path, entries):
    """Persist processed sample keys for the next incremental run."""
    payload = {
        'version': 1,
        'entries': list(entries),
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _scan_dataset(full_dataset_path):
    """Return dataset samples grouped by label in a deterministic order."""
    label_dirs = [d for d in os.listdir(full_dataset_path)
                  if os.path.isdir(os.path.join(full_dataset_path, d))]

    samples_by_label = {}
    for label in sorted(label_dirs):
        label_path = os.path.join(full_dataset_path, label)

        clip_dirs = sorted(
            [d for d in os.listdir(label_path)
             if os.path.isdir(os.path.join(label_path, d)) and d.startswith('clip_')],
            key=_numeric_sort_key,
        )
        image_files = sorted(
            [f for f in os.listdir(label_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=_numeric_sort_key,
        )

        samples = []
        if clip_dirs:
            for clip_name in clip_dirs:
                samples.append({
                    'key': _sample_key(label, clip_name),
                    'label': label,
                    'path': os.path.join(label_path, clip_name),
                    'kind': 'clip',
                    'name': clip_name,
                })
        elif image_files:
            for img_name in image_files:
                samples.append({
                    'key': _sample_key(label, img_name),
                    'label': label,
                    'path': os.path.join(label_path, img_name),
                    'kind': 'image',
                    'name': img_name,
                })

        if samples:
            samples_by_label[label] = samples

    return samples_by_label


def _load_existing_counts(full_output_path):
    """Infer how many samples per label already exist in an output file."""
    if not os.path.exists(full_output_path):
        return {}

    try:
        if full_output_path.lower().endswith('.csv'):
            df = pd.read_csv(full_output_path)
            if 'label' not in df.columns:
                return {}
            return df['label'].astype(str).value_counts().to_dict()

        if full_output_path.lower().endswith('.npz'):
            data = np.load(full_output_path, allow_pickle=True)
            labels = data['labels'].astype(str)
            return pd.Series(labels).value_counts().to_dict()
    except Exception:
        return {}

    return {}


def _bootstrap_processed_entries(samples_by_label, existing_counts):
    """Rebuild a processed-entry list from an existing output artifact."""
    processed_entries = []
    for label, samples in samples_by_label.items():
        processed_count = int(existing_counts.get(label, 0))
        processed_entries.extend(sample['key'] for sample in samples[:processed_count])
    return processed_entries


def _frame_feature(detector, image, use_normalized):
    """Extract engineered features from a single image with debugging."""

    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=50)

    _, results = detector.find_hands(enhanced, draw=False)

    landmarks = (
        detector.extract_landmarks_normalized(results, enhanced.shape)
        if use_normalized
        else detector.extract_landmarks(results)
    )

    if landmarks is None:
        _, results = detector.find_hands(image, draw=False)

        landmarks = (
            detector.extract_landmarks_normalized(results, image.shape)
            if use_normalized
            else detector.extract_landmarks(results)
        )

    if landmarks is None:
        os.makedirs("failed_debug", exist_ok=True)
        import time
        filename = os.path.join("failed_debug", f"{time.time_ns()}.jpg")
        cv2.imwrite(filename, image)
        print(f"❌ Saved failed image -> {filename}")
        return None

    features = compute_engineered_features(landmarks)

    if features is None:
        print("❌ Feature engineering failed")
        return None

    return features


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
                                   use_normalized=True):
    """
    Extract hand landmarks from all images in the dataset.
    
    Args:
        dataset_path (str): Path to the raw images directory
        output_csv (str): Path to save the landmarks CSV file
        use_normalized (bool): Whether to use normalized landmarks
    """
    # Get absolute paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_csv)
    manifest_path = _manifest_path(full_output_path)
    
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

    samples_by_label = _scan_dataset(full_dataset_path)
    if not samples_by_label:
        print(f"Error: No label directories found in {full_dataset_path}")
        print("Please collect data first using collect_data.py")
        return

    existing_counts = {}
    processed_entries = _load_manifest(manifest_path)
    if processed_entries:
        processed_set = set(processed_entries)
    else:
        existing_counts = _load_existing_counts(full_output_path)
        processed_entries = _bootstrap_processed_entries(samples_by_label, existing_counts)
        processed_set = set(processed_entries)

    # Prepare data storage
    all_landmarks = []
    all_labels = []

    print(f"\n{'='*60}")
    print("Extracting Landmarks from Dataset")
    print(f"{'='*60}")
    print(f"Dataset path: {full_dataset_path}")
    print(f"Found {len(samples_by_label)} label(s): {', '.join(sorted(samples_by_label))}")
    print(f"Feature vector size: {UNIFIED_FEATURES} (mean+std of {_BASE_FEATURES} features)")
    if processed_entries:
        print(f"Already indexed samples: {len(processed_entries)}")
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    total_existing = 0
    failed_images = []

    # Process each label directory
    for label in sorted(samples_by_label):
        samples = samples_by_label[label]
        new_samples = [sample for sample in samples if sample['key'] not in processed_set]
        if not new_samples:
            total_existing += len(samples)
            continue

        total_existing += len(samples) - len(new_samples)
        sample_kind = new_samples[0]['kind']
        print(f"Processing label '{label}': {len(new_samples)} new {sample_kind}(s)")

        for sample in tqdm(new_samples, desc=f"  {label}", unit=sample_kind):
            if sample['kind'] == 'clip':
                feat = _clip_to_feature(detector, sample['path'], use_normalized)
            else:
                image = cv2.imread(sample['path'])
                if image is None:
                    print(f"  Warning: Could not read image: {sample['name']}")
                    total_skipped += 1
                    continue
                feat = _static_to_feature(detector, image, use_normalized)

            if feat is not None:
                processed_entries.append(sample['key'])
                all_landmarks.append(feat)
                all_labels.append(label)
                total_processed += 1
            else:
                total_skipped += 1
                failed_images.append(sample['key'])
    
    # Close detector
    detector.close()

    if total_processed == 0:
        if processed_entries:
            _save_manifest(manifest_path, processed_entries)
            print("No new landmarks found. Existing output is already up to date.")
            return
        print("Error: No landmarks extracted. Please check your dataset.")
        return
    
    print(f"\n{'='*60}")
    print("Extraction Summary")
    print(f"{'='*60}")
    if total_existing > 0:
        print(f"Total samples already indexed: {total_existing}")
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

    if os.path.exists(full_output_path):
        try:
            existing_df = pd.read_csv(full_output_path)
            if not existing_df.empty:
                df = pd.concat([existing_df, df], ignore_index=True)
        except Exception:
            pass
    
    # Save to CSV
    df.to_csv(full_output_path, index=False)
    _save_manifest(manifest_path, processed_entries)
    
    print(f"✓ Landmarks saved to: {full_output_path}")
    print(f"✓ Dataset shape: {df.shape}")
    print(f"\nLabel distribution:")
    # Sort labels naturally
    label_counts = df['label'].value_counts()
    label_counts = label_counts.reindex(sorted(label_counts.index, key=str))
    print(label_counts)
    print()


def extract_sequences_from_dataset(dataset_path='dataset/raw_images',
                                   output_npz=None,
                                   use_normalized=True,
                                   mode='alphabet'):
    """
    Extract per-frame landmark sequences from all clips/images and save as a
    compressed .npz file for Bidirectional LSTM training.

    The .npz contains:
        X      : float32 array of shape (N, MAX_SEQ_FRAMES, _BASE_FEATURES)
        labels : str array of shape (N,)
    """
    mode = _normalize_mode(mode)
    output_npz = output_npz or MODE_OUTPUTS[mode]

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_dataset_path = os.path.join(script_dir, dataset_path)
    full_output_path = os.path.join(script_dir, output_npz)
    manifest_path = _manifest_path(full_output_path)

    if not os.path.exists(full_dataset_path):
        print(f"Error: Dataset path does not exist: {full_dataset_path}")
        return

    detector = HandDetector(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1
    )

    samples_by_label = _filter_samples_by_mode(_scan_dataset(full_dataset_path), mode)
    if not samples_by_label:
        print(f"Error: No label directories found for mode '{mode}' in {full_dataset_path}")
        return

    processed_entries = _load_manifest(manifest_path)
    if processed_entries:
        processed_set = set(processed_entries)
    else:
        existing_counts = _load_existing_counts(full_output_path)
        processed_entries = _bootstrap_processed_entries(samples_by_label, existing_counts)
        processed_set = set(processed_entries)

    all_sequences = []
    all_labels = []

    print(f"\n{'='*60}")
    print(f"Extracting Landmark Sequences ({mode})")
    print(f"{'='*60}")
    print(f"Dataset path  : {full_dataset_path}")
    print(f"Sequence shape: ({MAX_SEQ_FRAMES}, {_BASE_FEATURES})")
    print(f"Mode          : {mode}")
    print(f"Labels found  : {', '.join(sorted(samples_by_label))}")
    if processed_entries:
        print(f"Already indexed samples: {len(processed_entries)}")
    print(f"{'='*60}\n")

    total_processed = 0
    total_skipped = 0
    total_existing = 0

    for label in sorted(samples_by_label):
        samples = samples_by_label[label]
        new_samples = [sample for sample in samples if sample['key'] not in processed_set]
        if not new_samples:
            total_existing += len(samples)
            continue

        total_existing += len(samples) - len(new_samples)
        sample_kind = new_samples[0]['kind']
        print(f"Processing '{label}': {len(new_samples)} new {sample_kind}(s)")

        for sample in tqdm(new_samples, desc=f"  {label}", unit=sample_kind):
            if sample['kind'] == 'clip':
                seq = _clip_to_sequence(detector, sample['path'], use_normalized)
            else:
                image = cv2.imread(sample['path'])
                if image is None:
                    total_skipped += 1
                    continue
                seq = _static_to_sequence(detector, image, use_normalized)

            if seq is not None:
                processed_entries.append(sample['key'])
                all_sequences.append(seq)
                all_labels.append(label)
                total_processed += 1
            else:
                total_skipped += 1

    detector.close()

    if total_processed == 0:
        if processed_entries:
            _save_manifest(manifest_path, processed_entries)
            print("No new sequences found. Existing output is already up to date.")
            return
        print("Error: No sequences extracted. Please check your dataset.")
        return

    print(f"\n{'='*60}")
    print("Sequence Extraction Summary")
    print(f"{'='*60}")
    if total_existing > 0:
        print(f"Total samples already indexed: {total_existing}")
    print(f"Sequences extracted : {total_processed}")
    print(f"Samples skipped     : {total_skipped}")
    if total_processed + total_skipped > 0:
        print(f"Success rate        : {total_processed/(total_processed+total_skipped)*100:.1f}%")
    print(f"{'='*60}\n")

    X_array = np.array(all_sequences, dtype=np.float32)   # (N, MAX_SEQ_FRAMES, _BASE_FEATURES)
    labels_array = np.array(all_labels)

    if os.path.exists(full_output_path):
        try:
            existing = np.load(full_output_path, allow_pickle=True)
            existing_X = existing['X']
            existing_labels = existing['labels']
            if existing_X.size > 0:
                X_array = np.concatenate([existing_X, X_array], axis=0)
                labels_array = np.concatenate([existing_labels.astype(str), labels_array.astype(str)], axis=0)
        except Exception:
            pass

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    np.savez_compressed(full_output_path, X=X_array, labels=labels_array)
    _save_manifest(manifest_path, processed_entries)

    print(f"✓ Sequences saved to : {full_output_path}")
    print(f"✓ Array shape        : {X_array.shape}  (samples, frames, features)")
    print(f"✓ Labels             : {len(np.unique(labels_array))} classes")
    print()


def main():
    """
    Main function to run the landmark extraction script.
    """
    parser = argparse.ArgumentParser(description="Extract ISL landmarks")
    parser.add_argument(
        '--mode',
        choices=sorted(list(MODE_OUTPUTS.keys()) + ['csv']),
        default=None,
        help="Extraction mode: alphabet, word, or csv for the legacy flat dataset",
    )
    parser.add_argument(
        '--normalized',
        action='store_true',
        help="Use normalized landmarks without prompting",
    )
    parser.add_argument(
        '--raw',
        action='store_true',
        help="Use raw landmarks without prompting",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ISL Gesture Recognition - Landmark Extraction")
    print("="*60)

    if args.mode is None:
        print("\nExtraction mode:")
        print("  1) Static gestures (alphabet)")
        print("  2) Dynamic words")
        print("  3) Legacy flat CSV")
        mode_choice = input("Choose mode (1/2/3, default 1): ").strip()
        mode = 'csv' if mode_choice == '3' else ('word' if mode_choice == '2' else 'alphabet')
    else:
        mode = args.mode.lower()
        if mode not in MODE_OUTPUTS and mode != 'csv':
            raise ValueError(f"Unsupported mode: {args.mode}")

    if args.normalized and args.raw:
        print("Warning: --normalized and --raw were both set. Using normalized landmarks.")
        use_normalized = True
    elif args.normalized:
        use_normalized = True
    elif args.raw:
        use_normalized = False
    else:
        normalize_choice = input("Use normalized landmarks? (Y/n): ").strip().lower()
        use_normalized = normalize_choice != 'n'

    if use_normalized:
        print("Using normalized landmarks (recommended)")
    else:
        print("Using raw landmarks")

    if mode == 'csv':
        extract_landmarks_from_dataset(use_normalized=use_normalized)
    else:
        extract_sequences_from_dataset(use_normalized=use_normalized, mode=mode)


if __name__ == "__main__":
    main()
