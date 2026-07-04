"""
Real-time Gesture Prediction Script for ISL Gesture Recognition
This script uses the trained model to predict hand gestures in real-time from webcam feed.
"""

import cv2
import os
import sys
import pickle
import numpy as np
import time
import re
from collections import deque

# Force UTF-8 console output on Windows to prevent Unicode print errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import HandDetector, display_text, get_fps, compute_engineered_features
from utils.word_builder import WordBuilder, is_word_label
from utils.logger import logger

STATIC_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ALPHABET_MODEL_PATH = 'models/alphabet_model.pkl'
WORD_MODEL_PATH = 'models/word_model.pkl'
LEGACY_MODEL_PATH = 'models/gesture_model.pkl'


def _dedupe_consecutive_words(text):
    """Remove consecutive duplicate words: 'YOU YOU HELP' -> 'YOU HELP'."""
    words = text.split()
    if not words:
        return ""
    deduped = [words[0]]
    for word in words[1:]:
        if word.lower() != deduped[-1].lower():
            deduped.append(word)
    return " ".join(deduped)


def _rewrite_common_phrases(text):
    """Apply lightweight grammar rewrites for common sign-to-text patterns."""
    words = text.split()
    if not words:
        return ""

    # Longer patterns first to avoid partial replacement conflicts.
    rules = [
        (("thank", "you", "you", "welcome"), ["thank", "you", "you", "are", "welcome"]),
        (("hello", "how", "you"), ["hello", "how", "are", "you"]),
        (("what", "your", "name"), ["what", "is", "your", "name"]),
        (("what", "you", "name"), ["what", "is", "your", "name"]),
        (("my", "name"), ["my", "name", "is"]),
        (("how", "you", "doing"), ["how", "are", "you"]),
        (("you", "welcome"), ["you", "are", "welcome"]),
        (("please", "help"), ["please", "help", "me"]),
        (("where", "bathroom"), ["where", "is", "the", "bathroom"]),
        (("what", "time"), ["what", "time", "is", "it"]),
        (("i", "am", "help"), ["i", "need", "help"]),
        (("i", "help"), ["i", "need", "help"]),
        (("how", "you"), ["how", "are", "you"]),
        (("i", "sorry"), ["i", "am", "sorry"]),
        (("i", "thank", "you"), ["thank", "you"]),
    ]

    i = 0
    out = []
    lowered = [w.lower() for w in words]
    while i < len(words):
        matched = False
        for pattern, replacement in rules:
            n = len(pattern)
            if i + n <= len(words) and tuple(lowered[i:i + n]) == pattern:
                out.extend(replacement)
                i += n
                matched = True
                break
        if not matched:
            out.append(words[i])
            i += 1

    return " ".join(out)


def _normalize_pronoun_i(text):
    """Ensure standalone pronoun 'i' is capitalized."""
    return re.sub(r"\bi\b", "I", text)


def normalize_sentence_text(text, add_terminal_punctuation=False):
    """Light cleanup for recognized text before display/printing."""
    if not text:
        return ""

    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = _dedupe_consecutive_words(cleaned)
    cleaned = _rewrite_common_phrases(cleaned)
    cleaned = _dedupe_consecutive_words(cleaned)
    cleaned = _normalize_pronoun_i(cleaned)

    lower_cleaned = cleaned.lower()
    formatted_sentences = {
        "hello how are you": "Hello, how are you?",
        "how are you": "How are you?",
        "what is your name": "What is your name?",
        "my name is": "My name is.",
        "i need help": "I need help.",
        "please help me": "Please help me.",
        "thank you": "Thank you.",
        "you are welcome": "You are welcome.",
        "i am sorry": "I am sorry.",
        "where is the bathroom": "Where is the bathroom?",
        "what time is it": "What time is it?",
        "stop": "Stop.",
        "wait": "Wait.",
    }
    if lower_cleaned in formatted_sentences:
        return formatted_sentences[lower_cleaned]

    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    if add_terminal_punctuation and cleaned and cleaned[-1] not in ".!?":
        cleaned += "."

    return cleaned


def load_model(model_path=LEGACY_MODEL_PATH):
    """
    Load the trained model and label encoder.

    Returns:
        tuple: (model, label_encoder, uses_engineered, num_features, model_meta)
               model_meta is a dict with at least 'model_type', 'max_seq_frames',
               'feature_size' keys (only populated for BiLSTM models).
    """
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_model_path = model_path
    candidate_paths = [resolved_model_path]

    # Letter-based modes should keep working even when the split alphabet bundle
    # has not been created yet.
    if os.path.basename(resolved_model_path) == os.path.basename(ALPHABET_MODEL_PATH):
        candidate_paths.append(LEGACY_MODEL_PATH)

    full_model_path = None
    for candidate_path in candidate_paths:
        candidate_full_path = os.path.join(script_dir, candidate_path)
        if os.path.exists(candidate_full_path):
            resolved_model_path = candidate_path
            full_model_path = candidate_full_path
            break

    if full_model_path is None:
        full_model_path = os.path.join(script_dir, resolved_model_path)

    if not os.path.exists(full_model_path):
        print(f"Error: Model file not found: {full_model_path}")
        print("Please train the model first using train_model.py")
        return None, None, False, None, {}

    try:
        with open(full_model_path, 'rb') as f:
            model_data = pickle.load(f)

        label_encoder = model_data['label_encoder']
        uses_engineered = model_data.get('uses_engineered_features', False)
        num_features = model_data.get('num_features', None)
        model_type = model_data.get('model_type', 'sklearn')

        index_to_class = model_data.get('index_to_class', list(getattr(label_encoder, 'classes_', [])))
        default_keras_rel = os.path.splitext(resolved_model_path)[0] + '.keras'

        model_meta = {
            'model_type': model_type,
            'max_seq_frames': model_data.get('max_seq_frames', 30),
            'feature_size': model_data.get('feature_size', 93),
            'index_to_class': index_to_class,
            'model_path': resolved_model_path,
        }

        if model_type == 'BiLSTM':
            import tensorflow as tf
            keras_rel_path = model_data.get('keras_model_path', default_keras_rel)
            keras_full_path = os.path.join(script_dir, keras_rel_path)
            if not os.path.exists(keras_full_path):
                legacy_keras_path = os.path.join(script_dir, default_keras_rel)
                if os.path.exists(legacy_keras_path):
                    keras_full_path = legacy_keras_path
                else:
                    logger.error(f"Error: Keras model file not found: {keras_full_path}")
                    return None, None, False, None, {}
            model = tf.keras.models.load_model(keras_full_path)
            logger.info("✓ BiLSTM model loaded successfully")
        else:
            model = model_data['model']
            logger.info(f"✓ Model loaded successfully ({model_type})")

        logger.info(f"  - Classes: {', '.join(index_to_class)}")
        if isinstance(model_data.get('test_accuracy'), float):
            logger.info(f"  - Test Accuracy: {model_data['test_accuracy']:.2%}")

        return model, label_encoder, uses_engineered, num_features, model_meta

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, False, None, {}


def _predict_sequence(model, seq_buffer, feat_size, max_seq_frames):
    """Convert the rolling frame buffer into model input and return probabilities."""
    seq = np.zeros((max_seq_frames, feat_size), dtype=np.float32)
    recent = list(seq_buffer)
    if recent:
        seq[max_seq_frames - len(recent):] = np.array(recent, dtype=np.float32)
    proba = model.predict(seq[np.newaxis], verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    return pred_idx, confidence


def _detect_static_motion(motion_values, movement_threshold=0.015, static_ratio_threshold=0.7):
    """Classify the current hand stream as STATIC or DYNAMIC using landmark motion."""
    if not motion_values:
        return 'STATIC'

    static_votes = sum(1 for value in motion_values if value <= movement_threshold)
    static_ratio = static_votes / max(len(motion_values), 1)
    return 'STATIC' if static_ratio >= static_ratio_threshold else 'DYNAMIC'


def _movement_score(previous_landmarks, current_landmarks):
    """Compute mean landmark movement between consecutive frames."""
    if previous_landmarks is None or current_landmarks is None:
        return None
    previous = np.asarray(previous_landmarks, dtype=np.float32)
    current = np.asarray(current_landmarks, dtype=np.float32)
    if previous.shape != current.shape:
        return None
    return float(np.mean(np.abs(current - previous)))


def _update_motion_state(current_motion, motion_state, static_streak, dynamic_hold_frames=8):
    """Keep dynamic gestures on the word model for a few static-looking frames."""
    if current_motion == 'DYNAMIC':
        return 'DYNAMIC', 0

    if motion_state == 'DYNAMIC':
        static_streak += 1
        if static_streak < dynamic_hold_frames:
            return 'DYNAMIC', static_streak

    return 'STATIC', 0


def predict_realtime(alphabet_model_path=ALPHABET_MODEL_PATH,
                    word_model_path=WORD_MODEL_PATH,
                    use_normalized=True,
                    confidence_threshold=0.7):
    """
    Run real-time gesture prediction using webcam.
    
    Args:
        alphabet_model_path (str): Path to the static gesture model bundle
        word_model_path (str): Path to the dynamic word model bundle
        use_normalized (bool): Whether to use normalized landmarks
        confidence_threshold (float): Minimum confidence for displaying prediction
    """
    # Load both independent models up front so the runtime can switch tasks.
    print(f"\n{'='*60}")
    print("Loading Models...")
    print(f"{'='*60}\n")
    
    alphabet_model, alphabet_label_encoder, alphabet_uses_engineered, alphabet_num_features, alphabet_meta = load_model(alphabet_model_path)
    word_model, word_label_encoder, word_uses_engineered, word_num_features, word_meta = load_model(word_model_path)

    if (alphabet_model is None or alphabet_label_encoder is None
            or word_model is None or word_label_encoder is None):
        print("Error: Both alphabet and word models must be available before realtime prediction.")
        return

    max_seq_frames = alphabet_meta.get('max_seq_frames', 30)
    feat_size = alphabet_meta.get('feature_size', 93)
    if word_meta.get('max_seq_frames', max_seq_frames) != max_seq_frames:
        print("Warning: The alphabet and word models use different sequence lengths. Using the alphabet model length.")
    if word_meta.get('feature_size', feat_size) != feat_size:
        print("Warning: The alphabet and word models use different feature sizes. Using the alphabet model size.")

    # Motion gate thresholds are intentionally simple and conservative.
    movement_threshold = 0.015 if use_normalized else 0.04
    static_ratio_threshold = 0.7
    use_engineered_features = alphabet_uses_engineered or word_uses_engineered
    seq_buffer = deque(maxlen=max_seq_frames)
    motion_values = deque(maxlen=max_seq_frames - 1)
    previous_raw_landmarks = None
    motion_state = 'STATIC'
    static_streak = 0

    print(f"\n{'='*60}")
    print("Starting Real-time Prediction")
    print(f"{'='*60}")
    print("Instructions:")
    print("  - Show your hand(s) to the camera")
    print("  - Up to 2 hands can be detected")
    print("  - The system will auto-select the static or word model")
    print("  - Press 'q' to quit")
    print("  - Press 'f' to toggle FPS display")
    print(f"{'='*60}\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize hand detector with 2 hands support
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,  # Support up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # For FPS calculation
    prev_time = time.time()
    show_fps = True
    
    # For prediction smoothing
    prediction_history = []
    history_size = 7
    last_detected_type = None
    
    print("Webcam started. Showing predictions...\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        frame, results = detector.find_hands(frame, draw=True)
        
        # Get number of hands detected
        num_hands = detector.get_hand_count(results)
        
        # Display hand count
        if num_hands > 0:
            hand_count_text = f"Hands detected: {num_hands}"
            frame = display_text(frame, hand_count_text, 
                               position=(frame.shape[1] - 250, 30), font_scale=0.6, 
                               color=(255, 255, 0), thickness=2)
        
        # Extract landmarks from first hand for prediction
        hand_index = 0
        if use_normalized:
            landmarks = detector.extract_landmarks_normalized(results, frame.shape, hand_index=hand_index)
        else:
            landmarks = detector.extract_landmarks(results, hand_index=hand_index)

        # Get handedness label if available
        handedness = detector.get_handedness(results, hand_index=hand_index)
        if handedness is None:
            handedness = "Unknown"
        
        # Predict gesture if hand is detected
        if landmarks is not None:
            motion_score = _movement_score(previous_raw_landmarks, landmarks)
            if motion_score is not None:
                motion_values.append(motion_score)
            previous_raw_landmarks = np.asarray(landmarks, dtype=np.float32)

            # Preserve the engineered feature representation used by both models.
            features = compute_engineered_features(landmarks) if use_engineered_features else landmarks
            seq_buffer.append(np.asarray(features, dtype=np.float32))

            current_motion = _detect_static_motion(motion_values, movement_threshold, static_ratio_threshold)
            motion_state, static_streak = _update_motion_state(current_motion, motion_state, static_streak)
            detected_type = motion_state
            if detected_type != last_detected_type:
                prediction_history.clear()
                last_detected_type = detected_type

            active_model = alphabet_model if detected_type == 'STATIC' else word_model
            active_label_encoder = alphabet_label_encoder if detected_type == 'STATIC' else word_label_encoder
            active_meta = alphabet_meta if detected_type == 'STATIC' else word_meta

            pred_idx, confidence = _predict_sequence(active_model, seq_buffer, feat_size, max_seq_frames)
            index_to_class = active_meta.get('index_to_class') or list(active_label_encoder.classes_)
            predicted_label = index_to_class[pred_idx]

            # Add to prediction history for smoothing
            prediction_history.append(predicted_label)
            if len(prediction_history) > history_size:
                prediction_history.pop(0)
            
            # Get most common prediction in history
            if prediction_history:
                smoothed_prediction = max(set(prediction_history), 
                                         key=prediction_history.count)
            else:
                smoothed_prediction = predicted_label
            
            # Display prediction
            if confidence >= confidence_threshold:
                type_text = f"Detected Type: {detected_type}"
                prediction_text = f"Prediction: {smoothed_prediction}"
                confidence_text = f"Confidence: {confidence:.2%}"

                # Display with high confidence color
                frame = display_text(frame, type_text,
                                   position=(10, 30), font_scale=1.0,
                                   color=(0, 255, 0), thickness=2)
                frame = display_text(frame, prediction_text,
                                   position=(10, 70), font_scale=1.2,
                                   color=(0, 255, 0), thickness=2)
                frame = display_text(frame, confidence_text,
                                   position=(10, 110), font_scale=0.7,
                                   color=(0, 255, 0), thickness=2)
            else:
                # Low confidence
                type_text = f"Detected Type: {detected_type}"
                prediction_text = f"Prediction: {smoothed_prediction} (?)"
                confidence_text = f"Confidence: {confidence:.2%} (Low)"

                frame = display_text(frame, type_text,
                                   position=(10, 30), font_scale=1.0,
                                   color=(0, 165, 255), thickness=2)
                frame = display_text(frame, prediction_text,
                                   position=(10, 70), font_scale=1.2,
                                   color=(0, 165, 255), thickness=2)
                frame = display_text(frame, confidence_text,
                                   position=(10, 110), font_scale=0.7,
                                   color=(0, 165, 255), thickness=2)
        else:
            # No hand detected
            frame = display_text(frame, "No hand detected", 
                               position=(10, 30), font_scale=1, 
                               color=(0, 0, 255), thickness=2)
            prediction_history.clear()
            seq_buffer.clear()
            motion_values.clear()
            previous_raw_landmarks = None
            motion_state = 'STATIC'
            static_streak = 0
            last_detected_type = None
        curr_time = time.time()
        fps = get_fps(prev_time, curr_time)
        prev_time = curr_time
        
        if show_fps:
            frame = display_text(frame, f"FPS: {int(fps)}", 
                               position=(10, 450), font_scale=0.6, 
                               color=(255, 255, 0), thickness=1)
        
        # Display instructions
        frame = display_text(frame, "Press 'q' to quit | 'f' to toggle FPS", 
                           position=(10, frame.shape[0] - 10), font_scale=0.5, 
                           color=(255, 255, 255), thickness=1)
        
        # Show the frame
        cv2.imshow('ISL Gesture Recognition', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nStopping real-time prediction...")
            break
        elif key == ord('f'):
            show_fps = not show_fps
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("✓ Real-time prediction stopped\n")


def predict_words(model_path=WORD_MODEL_PATH,
                  use_normalized=True,
                  confidence_threshold=0.7,
                  hold_duration=1.0):
    """
    Run real-time word formation from individual letter gestures.

    Hold a letter gesture steadily to add it to the current word.
    Use keyboard keys for spacing, backspace, and clearing.

    Args:
        model_path (str): Path to the trained model
        use_normalized (bool): Whether to use normalized landmarks
        confidence_threshold (float): Minimum confidence for accepting a prediction
        hold_duration (float): Seconds to hold a letter before it is confirmed
    """
    # Load the model
    print(f"\n{'='*60}")
    print("Loading Model...")
    print(f"{'='*60}\n")

    model, label_encoder, uses_engineered, num_features, model_meta = load_model(model_path)

    if model is None or label_encoder is None:
        return

    is_bilstm = model_meta.get('model_type') == 'BiLSTM'
    max_seq_frames = model_meta.get('max_seq_frames', 30)
    feat_size = model_meta.get('feature_size', 93)

    from utils.mediapipe_utils import get_engineered_feature_names
    base_feat_count = len(get_engineered_feature_names())
    unified_mode = (not is_bilstm
                    and num_features is not None
                    and num_features > base_feat_count)
    TIME_WINDOW = 1.5  # seconds — used in unified_mode only
    feat_window = deque()     # (timestamp, array) pairs — unified_mode only
    seq_buffer = deque(maxlen=max_seq_frames)  # feature arrays — BiLSTM only

    print(f"\n{'='*60}")
    print("Starting Word Recognition Mode")
    print(f"{'='*60}")
    print("Instructions:")
    print("  - Hold a gesture steadily to confirm it")
    print("  - Letters (A-Z, 0-9) are appended to the current word")
    print("  - Word gestures (HELLO, etc.) are added as whole words")
    print("  - A progress bar shows how close you are to confirming")
    print("  - Press SPACE to finish current word and add a space")
    print("  - Press BACKSPACE to delete the last character")
    print("  - Press 'c' to clear all text")
    print("  - Press 'q' to quit")
    print(f"{'='*60}\n")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize hand detector
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    word_builder = WordBuilder(hold_duration=hold_duration)
    prev_time = time.time()

    print("Webcam started. Form words by holding gestures...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame, draw=True)
        h, w = frame.shape[:2]

        # Extract landmarks
        if use_normalized:
            landmarks = detector.extract_landmarks_normalized(results, frame.shape, hand_index=0)
        else:
            landmarks = detector.extract_landmarks(results, hand_index=0)

        predicted_label = None
        confidence = 0.0

        if landmarks is not None:
            if uses_engineered:
                landmarks = compute_engineered_features(landmarks)

            if is_bilstm:
                seq_buffer.append(landmarks.astype(np.float32))
                seq = np.zeros((max_seq_frames, feat_size), dtype=np.float32)
                recent = list(seq_buffer)
                seq[max_seq_frames - len(recent):] = np.array(recent)
                proba = model.predict(seq[np.newaxis], verbose=0)[0]
                pred_idx = int(np.argmax(proba))
                confidence = float(proba[pred_idx])
                predicted_label = label_encoder.inverse_transform([pred_idx])[0]
            elif unified_mode:
                _now = time.time()
                feat_window.append((_now, landmarks))
                while feat_window and (_now - feat_window[0][0]) > TIME_WINDOW:
                    feat_window.popleft()
                arr = np.array([f for _, f in feat_window])
                lm = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])
                prediction = model.predict(lm.reshape(1, -1))[0]
                prediction_proba = model.predict_proba(lm.reshape(1, -1))[0]
                confidence = prediction_proba[prediction]
                predicted_label = label_encoder.inverse_transform([prediction])[0]
            else:
                landmarks_reshaped = landmarks.reshape(1, -1)
                prediction = model.predict(landmarks_reshaped)[0]
                prediction_proba = model.predict_proba(landmarks_reshaped)[0]
                confidence = prediction_proba[prediction]
                predicted_label = label_encoder.inverse_transform([prediction])[0]

        # Feed prediction to word builder (only if confident enough)
        confirmed = None
        if predicted_label is not None and confidence >= confidence_threshold:
            confirmed = word_builder.update(predicted_label)
        else:
            # No confident prediction — reset hold tracking
            word_builder._reset_tracking()
            if is_bilstm:
                seq_buffer.clear()
            else:
                feat_window.clear()

        # ── Draw UI ──────────────────────────────────────────

        # Current detected gesture + confidence
        if predicted_label and confidence >= confidence_threshold:
            gesture_type = "Word" if is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{gesture_type}: {predicted_label}",
                               position=(10, 30), font_scale=1.0,
                               color=(0, 255, 0), thickness=2)
            frame = display_text(frame, f"Confidence: {confidence:.0%}",
                               position=(10, 65), font_scale=0.6,
                               color=(0, 255, 0), thickness=1)
        elif predicted_label:
            gesture_type = "Word" if is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{gesture_type}: {predicted_label} (low)",
                               position=(10, 30), font_scale=1.0,
                               color=(0, 165, 255), thickness=2)
        else:
            frame = display_text(frame, "No hand detected",
                               position=(10, 30), font_scale=0.8,
                               color=(0, 0, 255), thickness=2)

        # Hold progress bar
        progress = word_builder.get_hold_progress()
        pending = word_builder.get_pending_label()
        if pending and progress > 0:
            bar_x, bar_y, bar_w, bar_h = 10, 90, 200, 20
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            fill_w = int(bar_w * progress)
            bar_color = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
            frame = display_text(frame, f"Hold '{pending}'",
                               position=(bar_x + bar_w + 10, bar_y + bar_h - 3),
                               font_scale=0.6, color=(255, 255, 255), thickness=1)

        # Confirmed flash
        if confirmed:
            frame = display_text(frame, f"+ {confirmed}",
                               position=(10, 140), font_scale=1.0,
                               color=(0, 255, 255), thickness=2)

        # Word / sentence display (bottom area)
        display_text_str = word_builder.get_display_text()
        if display_text_str:
            # Background bar for text
            cv2.rectangle(frame, (0, h - 80), (w, h - 40), (40, 40, 40), -1)
            # Truncate if too long for the frame
            max_chars = w // 14
            shown = display_text_str[-max_chars:] if len(display_text_str) > max_chars else display_text_str
            frame = display_text(frame, shown,
                               position=(10, h - 45), font_scale=0.9,
                               color=(255, 255, 255), thickness=2)

        # Current word indicator
        if word_builder.current_word:
            frame = display_text(frame, f"Current word: {word_builder.current_word}",
                               position=(10, h - 95), font_scale=0.6,
                               color=(200, 200, 200), thickness=1)

        # Instructions bar
        frame = display_text(frame, "SPACE=space  BKSP=delete  C=clear  Q=quit",
                           position=(10, h - 10), font_scale=0.45,
                           color=(180, 180, 180), thickness=1)

        # FPS
        curr_time = time.time()
        fps = get_fps(prev_time, curr_time)
        prev_time = curr_time
        frame = display_text(frame, f"FPS: {int(fps)}",
                           position=(w - 110, 30), font_scale=0.6,
                           color=(255, 255, 0), thickness=1)

        cv2.imshow('ISL Word Formation', frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            word_builder.add_space()
        elif key == 8:  # Backspace
            word_builder.backspace()
        elif key == ord('c'):
            word_builder.clear()

    # Print final text
    final_text = word_builder.get_display_text()
    if final_text:
        print(f"\nFormed text: {final_text}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("✓ Word recognition mode stopped\n")


def predict_sentence(alphabet_model_path=ALPHABET_MODEL_PATH,
                     word_model_path=WORD_MODEL_PATH,
                     use_normalized=True,
                     confidence_threshold=0.7,
                     hold_duration=1.0,
                     auto_space_after=1.0):
    """
    Real-time sentence formation.

    - Hold a gesture steadily to confirm it.
    - Letters spell a word; remove hand for auto_space_after seconds to auto-commit the word.
    - Word gestures (HELLO, THANKS, etc.) are added as whole words automatically.
    - Press ENTER  : finalise the sentence and start a new one.
    - Press SPACE  : manually add a space (commit current word).
    - Press BACKSPACE: delete last character.
    - Press 'c'    : clear current sentence.
    - Press 'q'    : quit.
    """
    print(f"\n{'='*60}")
    print("Loading Model...")
    print(f"{'='*60}\n")

    alphabet_model, alphabet_label_encoder, alphabet_uses_engineered, alphabet_num_features, alphabet_meta = load_model(alphabet_model_path)
    word_model, word_label_encoder, word_uses_engineered, word_num_features, word_meta = load_model(word_model_path)
    if (alphabet_model is None or alphabet_label_encoder is None
            or word_model is None or word_label_encoder is None):
        return

    max_seq_frames = alphabet_meta.get('max_seq_frames', 30)
    feat_size = alphabet_meta.get('feature_size', 93)
    if word_meta.get('max_seq_frames', max_seq_frames) != max_seq_frames:
        print("Warning: The alphabet and word models use different sequence lengths. Using the alphabet model length.")
    if word_meta.get('feature_size', feat_size) != feat_size:
        print("Warning: The alphabet and word models use different feature sizes. Using the alphabet model size.")

    movement_threshold = 0.015 if use_normalized else 0.04
    static_ratio_threshold = 0.7
    use_engineered_features = alphabet_uses_engineered or word_uses_engineered
    seq_buffer = deque(maxlen=max_seq_frames)
    motion_values = deque(maxlen=max_seq_frames - 1)
    previous_raw_landmarks = None
    motion_state = 'STATIC'
    static_streak = 0

    print(f"\n{'='*60}")
    print("Starting Sentence Formation Mode")
    print(f"{'='*60}")
    print("  - Hold a gesture steadily to confirm it")
    print(f"  - Remove hand for {auto_space_after}s to auto-commit current word")
    print("  - Word gestures are added as full words automatically")
    print("  - ENTER=finish sentence  |  SPACE=manual space")
    print("  - BACKSPACE=delete  |  C=clear  |  Q=quit")
    print(f"{'='*60}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    word_builder = WordBuilder(hold_duration=hold_duration)
    sentence_history = []   # list of finalised sentences
    no_hand_since = None    # timestamp when hand last disappeared
    prev_time = time.time()

    print("Webcam started. Begin signing...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame, draw=True)
        h, w = frame.shape[:2]
        now = time.time()

        if use_normalized:
            landmarks = detector.extract_landmarks_normalized(results, frame.shape, hand_index=0)
        else:
            landmarks = detector.extract_landmarks(results, hand_index=0)

        predicted_label = None
        confidence = 0.0

        if landmarks is not None:
            no_hand_since = None  # hand is present — reset absence timer
            motion_score = _movement_score(previous_raw_landmarks, landmarks)
            if motion_score is not None:
                motion_values.append(motion_score)
            previous_raw_landmarks = np.asarray(landmarks, dtype=np.float32)

            if use_engineered_features:
                landmarks = compute_engineered_features(landmarks)

            current_motion = _detect_static_motion(motion_values, movement_threshold, static_ratio_threshold)
            motion_state, static_streak = _update_motion_state(current_motion, motion_state, static_streak)
            detected_type = motion_state
            active_model = alphabet_model if detected_type == 'STATIC' else word_model
            active_label_encoder = alphabet_label_encoder if detected_type == 'STATIC' else word_label_encoder
            active_meta = alphabet_meta if detected_type == 'STATIC' else word_meta

            seq_buffer.append(landmarks.astype(np.float32))
            seq = np.zeros((max_seq_frames, feat_size), dtype=np.float32)
            recent = list(seq_buffer)
            seq[max_seq_frames - len(recent):] = np.array(recent)
            proba = active_model.predict(seq[np.newaxis], verbose=0)[0]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])
            predicted_label = (active_meta.get('index_to_class') or list(active_label_encoder.classes_))[pred_idx]
        else:
            # Hand absent
            seq_buffer.clear()
            motion_values.clear()
            previous_raw_landmarks = None
            motion_state = 'STATIC'
            static_streak = 0
            word_builder._reset_tracking()
            if word_builder.current_word:
                if no_hand_since is None:
                    no_hand_since = now
                elif now - no_hand_since >= auto_space_after:
                    word_builder.add_space()
                    no_hand_since = None

        # Feed confident predictions to word builder
        confirmed = None
        if predicted_label is not None and confidence >= confidence_threshold:
            confirmed = word_builder.update(predicted_label)
        elif predicted_label is not None:
            word_builder._reset_tracking()

        # ── Draw UI ──────────────────────────────────────────────────

        # Current gesture (top left)
        if predicted_label and confidence >= confidence_threshold:
            g_type = "Word" if detected_type == 'DYNAMIC' or is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{g_type}: {predicted_label}",
                                 (10, 30), font_scale=1.0, color=(0, 255, 0), thickness=2)
            frame = display_text(frame, f"{confidence:.0%}",
                                 (10, 65), font_scale=0.6, color=(0, 255, 0), thickness=1)
        elif predicted_label:
            g_type = "Word" if detected_type == 'DYNAMIC' or is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{g_type}: {predicted_label} (low)",
                                 (10, 30), font_scale=0.9, color=(0, 165, 255), thickness=2)
        else:
            frame = display_text(frame, "No hand detected",
                                 (10, 30), font_scale=0.8, color=(0, 0, 255), thickness=2)
            # Auto-space countdown
            if no_hand_since is not None and word_builder.current_word:
                remaining = max(0.0, auto_space_after - (now - no_hand_since))
                frame = display_text(frame, f"Auto-space in {remaining:.1f}s",
                                     (10, 65), font_scale=0.6, color=(255, 200, 0), thickness=1)

        # Hold progress bar
        progress = word_builder.get_hold_progress()
        pending = word_builder.get_pending_label()
        if pending and progress > 0:
            bx, by, bw, bh = 10, 90, 200, 18
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
            fw = int(bw * progress)
            bcol = (0, 255, 0) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (bx, by), (bx + fw, by + bh), bcol, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            frame = display_text(frame, f"Hold '{pending}'",
                                 (bx + bw + 10, by + bh - 2),
                                 font_scale=0.55, color=(255, 255, 255), thickness=1)

        # Confirmed flash
        if confirmed:
            frame = display_text(frame, f"+ '{confirmed}'",
                                 (10, 130), font_scale=0.9, color=(0, 255, 255), thickness=2)

        # ── Word collection status (top-right area) ──────────────────────────
        # Show confirmed words count
        confirmed_words = word_builder.sentence.split() if word_builder.sentence else []
        word_count = len(confirmed_words)
        cv2.putText(frame, f"Words: {word_count}", (w - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 150), 2, cv2.LINE_AA)

        # Current word being typed (middle-right)
        if word_builder.current_word:
            cv2.rectangle(frame, (w - 220, 60), (w - 10, 100), (50, 100, 50), -1)
            cv2.rectangle(frame, (w - 220, 60), (w - 10, 100), (100, 255, 100), 2)
            cv2.putText(frame, "Current:", (w - 210, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA)
            cv2.putText(frame, word_builder.current_word, (w - 210, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 100), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Waiting for gesture...", (w - 240, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Sentence history (grey, above the main sentence panel)
        for i, hist in enumerate(reversed(sentence_history[-2:])):
            max_c = (w - 30) // 11
            trunc = hist[-max_c:] if len(hist) > max_c else hist
            frame = display_text(frame, f"> {trunc}",
                                 (10, h - 180 - i * 32),
                                 font_scale=0.55, color=(100, 200, 100), thickness=1)

        # ── Main sentence display panel (LARGE, prominent) ────────────────────
        cv2.rectangle(frame, (0, h - 110), (w, h - 45), (20, 40, 80), -1)
        cv2.line(frame, (0, h - 110), (w, h - 110), (0, 255, 0), 3)
        
        # Label with word count badge
        cv2.putText(frame, "SENTENCE:", (10, h - 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2, cv2.LINE_AA)
        
        # Display confirmed sentence words
        if word_builder.sentence:
            sentence_text = normalize_sentence_text(word_builder.sentence)
            max_chars = w // 13
            shown = sentence_text[-max_chars:] if len(sentence_text) > max_chars else sentence_text
            cv2.putText(frame, shown, (115, h - 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 150), 3, cv2.LINE_AA)
            # Show indicator that words are being collected
            cv2.circle(frame, (w - 20, h - 70), 6, (0, 255, 0), -1)
        elif word_builder.current_word:
            # Show current word in progress in the sentence panel
            cv2.putText(frame, word_builder.current_word, (115, h - 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 200, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Sign to begin...",
                        (115, h - 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (140, 140, 140), 2, cv2.LINE_AA)

        # Instructions bar
        frame = display_text(frame,
                             "ENTER=finish sentence  SPACE=space  BKSP=del  C=clear  Q=quit",
                             (10, h - 10), font_scale=0.42,
                             color=(160, 160, 160), thickness=1)

        # FPS
        curr_time = time.time()
        fps = get_fps(prev_time, curr_time)
        prev_time = curr_time
        frame = display_text(frame, f"FPS:{int(fps)}",
                             (w - 90, 30), font_scale=0.55, color=(255, 255, 0), thickness=1)

        cv2.imshow('ISL Sentence Formation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 13:  # Enter — finalise current sentence
            word_builder.add_space()  # commit any in-progress word
            sentence = word_builder.sentence.strip()
            if sentence:
                sentence = normalize_sentence_text(sentence, add_terminal_punctuation=True)
                sentence_history.append(sentence)
                print(f"Sentence: {sentence}")
            word_builder.clear()
            feat_window.clear()
        elif key == ord(' '):
            word_builder.add_space()
        elif key == 8:  # Backspace
            word_builder.backspace()
        elif key == ord('c'):
            word_builder.clear()
            feat_window.clear()

    # Commit any unsaved text on exit
    final_text = word_builder.get_display_text().strip()
    if final_text:
        sentence_history.append(normalize_sentence_text(final_text, add_terminal_punctuation=True))

    if sentence_history:
        print("\n" + "="*50)
        print("Sentences formed this session:")
        for i, s in enumerate(sentence_history, 1):
            print(f"  {i}. {s}")
        print("="*50)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\u2713 Sentence formation mode stopped\n")


def predict_stable_sentence(model_path=ALPHABET_MODEL_PATH,
                             use_normalized=True,
                             confidence_threshold=0.5,
                             buffer_size=10,
                             stability_threshold=6,
                             use_tts=True):
    """
    Real-time sentence builder using a prediction buffer for stability.

    Logic:
      - Every frame prediction is added to a rolling buffer (deque, maxlen=buffer_size).
      - A gesture is confirmed only when the same label appears >= stability_threshold
        times inside the buffer. This prevents flicker / repeated words.
      - Confirmed gestures are appended to the sentence only when they differ from
        the last added word, so holding a sign does not repeat it.
      - Optional text-to-speech speaks each newly added word via pyttsx3.

    Keys:
      C = clear sentence
      Q = quit
    """
    # ── TTS setup (optional) ─────────────────────────────────────────────────
    tts_engine = None
    if use_tts:
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty('rate', 150)
            print("\u2713 Text-to-speech enabled")
        except ImportError:
            print("[TTS] pyttsx3 not installed — run: pip install pyttsx3")
        except Exception as e:
            print(f"[TTS] Could not initialise: {e}")

    # ── Model loading ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Loading Model...")
    print(f"{'='*60}\n")

    model, label_encoder, uses_engineered, num_features = load_model(model_path)
    if model is None or label_encoder is None:
        return

    from utils.mediapipe_utils import get_engineered_feature_names
    base_feat_count = len(get_engineered_feature_names())
    unified_mode = (num_features is not None and num_features > base_feat_count)
    TIME_WINDOW = 1.5  # seconds — matches training clip duration
    feat_window = deque()  # stores (timestamp, feature_array) pairs

    print(f"\n{'='*60}")
    print("Sentence Builder")
    print(f"{'='*60}")
    print(f"  Buffer size          : {buffer_size} frames")
    print(f"  Stability threshold  : {stability_threshold}/{buffer_size} same predictions")
    print(f"  Confidence threshold : {confidence_threshold:.0%}")
    print(f"  TTS                  : {'on' if tts_engine else 'off'}")
    print("  C = clear sentence  |  Q = quit")
    print(f"{'='*60}\n")

    # ── State ────────────────────────────────────────────────────────────────
    # Rolling buffer of the last `buffer_size` high-confidence predictions
    pred_buffer = deque(maxlen=buffer_size)
    # The sentence: list of confirmed words/gestures
    sentence = []
    # Flash message shown briefly when a word is added
    flash_word = None
    flash_until = 0.0

    # ── Webcam ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    prev_time = time.time()
    print("Webcam started. Begin signing...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame, draw=True)
        h, w = frame.shape[:2]
        now = time.time()

        # ── Step 1: extract landmarks ─────────────────────────────────────────
        if use_normalized:
            landmarks = detector.extract_landmarks_normalized(results, frame.shape, hand_index=0)
        else:
            landmarks = detector.extract_landmarks(results, hand_index=0)

        predicted_label = None
        confidence = 0.0

        if landmarks is not None:
            # ── Step 2: feature engineering + unified mode ────────────────────
            if uses_engineered:
                landmarks = compute_engineered_features(landmarks)
            if unified_mode:
                feat_window.append((now, landmarks))
                while feat_window and (now - feat_window[0][0]) > TIME_WINDOW:
                    feat_window.popleft()
                arr = np.array([f for _, f in feat_window])
                landmarks = np.concatenate([arr.mean(axis=0), arr.std(axis=0)])

            # ── Step 3: model prediction ──────────────────────────────────────
            try:
                pred_enc = model.predict(landmarks.reshape(1, -1))[0]
                prob = model.predict_proba(landmarks.reshape(1, -1))[0]
                confidence = prob[pred_enc]
                predicted_label = label_encoder.inverse_transform([pred_enc])[0]
            except Exception:
                predicted_label = None

            # ── Step 4: update prediction buffer (only if confident) ──────────
            if predicted_label is not None and confidence >= confidence_threshold:
                pred_buffer.append(predicted_label)
            # No hand = don't touch the buffer (handled below)
        else:
            # No hand detected — clear feature window but keep pred buffer
            feat_window.clear()

        # ── Step 5: check buffer for stability ───────────────────────────────
        #  Confirmed only when buffer is full and one label dominates
        if len(pred_buffer) == buffer_size:
            from collections import Counter
            most_common_label, count = Counter(pred_buffer).most_common(1)[0]
            if count >= stability_threshold:
                # ── Step 6: add to sentence if new word ───────────────────────
                if not sentence or sentence[-1] != most_common_label:
                    sentence.append(most_common_label)
                    flash_word = most_common_label
                    flash_until = now + 1.0
                    live_sentence = normalize_sentence_text(" ".join(sentence))
                    print(f"  + '{most_common_label}'  ->  {live_sentence}")
                    # ── Step 7: text-to-speech ─────────────────────────────────
                    if tts_engine:
                        try:
                            tts_engine.say(most_common_label)
                            tts_engine.runAndWait()
                        except Exception:
                            pass
                # Clear buffer after a confirmed word so next word can register
                pred_buffer.clear()

        # ── Draw UI ───────────────────────────────────────────────────────────

        # Current gesture label (top-left)
        if predicted_label and confidence >= confidence_threshold:
            g_type = "Word" if is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{g_type}: {predicted_label}",
                                 (10, 30), font_scale=1.0, color=(0, 255, 0), thickness=2)
            frame = display_text(frame, f"{confidence:.0%}",
                                 (10, 65), font_scale=0.6, color=(0, 255, 0), thickness=1)
        elif predicted_label:
            g_type = "Word" if is_word_label(predicted_label) else "Letter"
            frame = display_text(frame, f"{g_type}: {predicted_label} (low conf)",
                                 (10, 30), font_scale=0.9, color=(0, 165, 255), thickness=2)
        else:
            frame = display_text(frame, "No hand detected",
                                 (10, 30), font_scale=0.8, color=(0, 0, 255), thickness=2)

        # Buffer fill bar
        if len(pred_buffer) > 0:
            bx, by, bw, bh = 10, 80, 180, 16
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
            fill = int(bw * len(pred_buffer) / buffer_size)
            # colour: green if stable, yellow otherwise
            from collections import Counter as _C
            top_count = _C(pred_buffer).most_common(1)[0][1] if pred_buffer else 0
            bar_col = (0, 220, 0) if top_count >= stability_threshold else (0, 200, 220)
            cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), bar_col, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
            frame = display_text(frame, f"Buffer {len(pred_buffer)}/{buffer_size}",
                                 (bx + bw + 8, by + bh - 2),
                                 font_scale=0.48, color=(200, 200, 200), thickness=1)

        # Flash confirmed word
        if flash_word and now < flash_until:
            frame = display_text(frame, f"+ '{flash_word}'",
                                 (10, 115), font_scale=1.0, color=(0, 255, 255), thickness=2)

        # ── Step 8: sentence panel ─────────────────────────────────────────
        # Draw a clearly visible panel at the bottom of the frame
        panel_top = h - 90
        # Bright teal background bar
        cv2.rectangle(frame, (0, panel_top), (w, h), (20, 60, 20), -1)
        cv2.line(frame, (0, panel_top), (w, panel_top), (0, 200, 0), 2)

        # Label
        cv2.putText(frame, "SENTENCE:", (10, panel_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 1, cv2.LINE_AA)

        # Sentence text — big, bright white
        sentence_text = normalize_sentence_text(" ".join(sentence)) if sentence else "(waiting for signs...)"
        max_chars = (w - 130) // 16
        shown = sentence_text[-max_chars:] if len(sentence_text) > max_chars else sentence_text
        txt_color = (255, 255, 255) if sentence else (160, 160, 160)
        cv2.putText(frame, shown, (120, panel_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, txt_color, 2, cv2.LINE_AA)

        # Buffer progress dots row
        dot_y = panel_top + 50
        cv2.putText(frame, "Buffer:", (10, dot_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        from collections import Counter as _Ct
        top_lbl = _Ct(pred_buffer).most_common(1)[0][0] if pred_buffer else ""
        top_cnt = _Ct(pred_buffer).most_common(1)[0][1] if pred_buffer else 0
        for i in range(buffer_size):
            cx = 75 + i * 22
            if i < len(pred_buffer):
                dot_col = (0, 230, 0) if top_cnt >= stability_threshold else (0, 200, 230)
            else:
                dot_col = (60, 60, 60)
            cv2.circle(frame, (cx, dot_y - 6), 8, dot_col, -1)
        if pred_buffer:
            cv2.putText(frame, f" {top_lbl} ({top_cnt}/{buffer_size})",
                        (80 + buffer_size * 22, dot_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 0), 1, cv2.LINE_AA)

        # Instructions
        cv2.putText(frame, "C=clear  Q=quit",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (120, 180, 120), 1, cv2.LINE_AA)

        # FPS (top-right, above the panel)
        curr_time = time.time()
        fps = get_fps(prev_time, curr_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS:{int(fps)}", (w - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('ISL Sentence Builder', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # ── Step 9: reset sentence ────────────────────────────────────────
            sentence = []
            pred_buffer.clear()
            print("  Sentence cleared.")

    # Final output
    if sentence:
        final = normalize_sentence_text(" ".join(sentence), add_terminal_punctuation=True)
        print(f"\nFinal sentence: {final}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\u2713 Sentence builder stopped\n")


def main():
    """
    Main function to run real-time modes directly from this script.
    """
    print("\n" + "="*60)
    print("ISL Gesture Recognition - Real-time Modes")
    print("="*60)

    print("\nChoose mode:")
    print("  1. Letters")
    print("  2. Words")
    print("  3. Sentence builder")
    mode = input("Enter mode (1-3, default: 3): ").strip() or "3"
    
    # Option to use normalized landmarks
    normalize_choice = input("\nUse normalized landmarks? (Y/n): ").strip().lower()
    use_normalized = normalize_choice != 'n'

    default_threshold = 0.6 if mode == "1" else 0.5
    threshold_input = input(f"Enter confidence threshold (0-1, default: {default_threshold}): ").strip()
    
    try:
        if threshold_input:
            confidence_threshold = float(threshold_input)
            if confidence_threshold < 0 or confidence_threshold > 1:
                print(f"Warning: Threshold must be between 0 and 1. Using default ({default_threshold})")
                confidence_threshold = default_threshold
        else:
            confidence_threshold = default_threshold
    except ValueError:
        print(f"Warning: Invalid input. Using default threshold ({default_threshold})")
        confidence_threshold = default_threshold

    if mode == "1":
        predict_realtime(use_normalized=use_normalized,
                         confidence_threshold=confidence_threshold)
    elif mode == "2":
        predict_words(use_normalized=use_normalized,
                      confidence_threshold=confidence_threshold)
    elif mode == "3":
        hold_input = input("Enter hold duration in seconds (default: 1.0): ").strip()
        try:
            hold_duration = float(hold_input) if hold_input else 1.0
        except ValueError:
            hold_duration = 1.0
        auto_space_input = input("Auto-space delay when hand removed (default: 1.0): ").strip()
        try:
            auto_space_after = float(auto_space_input) if auto_space_input else 1.0
            if auto_space_after <= 0:
                auto_space_after = 1.0
        except ValueError:
            auto_space_after = 1.0
        predict_sentence(use_normalized=use_normalized,
                         confidence_threshold=confidence_threshold,
                         hold_duration=hold_duration,
                         auto_space_after=auto_space_after)
    else:
        print("Invalid mode. Starting Sentence builder by default.")
        predict_sentence(use_normalized=use_normalized,
                         confidence_threshold=confidence_threshold,
                         hold_duration=1.0,
                         auto_space_after=1.0)


if __name__ == "__main__":
    main()
