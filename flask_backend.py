import cv2
import numpy as np
import time
from collections import deque
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.realtime_predict import (
    load_model, _movement_score, _detect_static_motion,
    _update_motion_state, normalize_sentence_text,
    ALPHABET_MODEL_PATH, WORD_MODEL_PATH
)
from core_utils.mediapipe_utils import HandDetector, compute_engineered_features, display_text
from core_utils.word_builder import WordBuilder, is_word_label

class FlaskISLRuntime:
    def __init__(self):
        self.alphabet_model = None
        self.word_model = None
        self.detector = None
        self.is_running = False
        self.cap = None
        
        self.use_normalized = True
        self.confidence_threshold = 0.7
        self.hold_duration = 1.0
        self.auto_space_after = 1.0
        self.movement_threshold = 0.015
        self.static_ratio_threshold = 0.7
        
        self.word_builder = WordBuilder(hold_duration=self.hold_duration)
        self.current_prediction_text = ""
        self.status_msg = "Ready. Press START INFERENCE."
        self.initialized = False

    def initialize_models(self):
        if self.initialized: return
        print("Loading AI Models & MediaPipe for Flask...")
        self.alphabet_model, self.alphabet_label_encoder, self.alphabet_uses_engineered, self.alphabet_num_features, self.alphabet_meta = load_model(ALPHABET_MODEL_PATH)
        self.word_model, self.word_label_encoder, self.word_uses_engineered, self.word_num_features, self.word_meta = load_model(WORD_MODEL_PATH)
        
        self.max_seq_frames = self.alphabet_meta.get('max_seq_frames', 30)
        self.feat_size = self.alphabet_meta.get('feature_size', 93)
        self.use_engineered_features = self.alphabet_uses_engineered or self.word_uses_engineered
        
        # DISCREPANCY FIXED: Initialize MediaPipe exactly once here, NOT in the video loop.
        self.detector = HandDetector(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.seq_buffer = deque(maxlen=self.max_seq_frames)
        self.motion_values = deque(maxlen=self.max_seq_frames - 1)
        self.previous_raw_landmarks = None
        self.motion_state = 'STATIC'
        self.static_streak = 0
        self.no_hand_since = None
        self.initialized = True
        print("Backend Successfully Initialized.")

    def start_inference(self):
        if not self.initialized:
            self.initialize_models()
        self.is_running = True
        self.status_msg = "🟢 RECOGNIZING"

    def stop_inference(self):
        self.is_running = False
        self.status_msg = "🔴 STOPPED"

    def clear_text(self):
        self.word_builder.clear()
        self.current_prediction_text = ""

    def generate_frames(self):
        # Prevent accessing the camera before initialization is complete
        if not self.initialized:
            self.initialize_models()
            
        self.cap = cv2.VideoCapture(0)
        
        while True:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue
                
            frame = cv2.flip(frame, 1)
            now = time.time()
            
            if self.is_running:
                try:
                    frame, results = self.detector.find_hands(frame, draw=True)
                    if self.use_normalized:
                        landmarks = self.detector.extract_landmarks_normalized(results, frame.shape, hand_index=0)
                    else:
                        landmarks = self.detector.extract_landmarks(results, hand_index=0)
                        
                    predicted_label = None
                    confidence = 0.0
                    
                    if landmarks is not None:
                        self.no_hand_since = None
                        motion_score = _movement_score(self.previous_raw_landmarks, landmarks)
                        if motion_score is not None:
                            self.motion_values.append(motion_score)
                        self.previous_raw_landmarks = np.asarray(landmarks, dtype=np.float32)
                        
                        if self.use_engineered_features:
                            landmarks = compute_engineered_features(landmarks)
                            
                        current_motion = _detect_static_motion(self.motion_values, self.movement_threshold, self.static_ratio_threshold)
                        self.motion_state, self.static_streak = _update_motion_state(current_motion, self.motion_state, self.static_streak)
                        
                        active_model = self.alphabet_model if self.motion_state == 'STATIC' else self.word_model
                        active_label_encoder = self.alphabet_label_encoder if self.motion_state == 'STATIC' else self.word_label_encoder
                        active_meta = self.alphabet_meta if self.motion_state == 'STATIC' else self.word_meta
                        
                        self.seq_buffer.append(landmarks.astype(np.float32))
                        seq = np.zeros((self.max_seq_frames, self.feat_size), dtype=np.float32)
                        recent = list(self.seq_buffer)
                        seq[self.max_seq_frames - len(recent):] = np.array(recent)
                        
                        proba = active_model.predict(seq[np.newaxis], verbose=0)[0]
                        pred_idx = int(np.argmax(proba))
                        confidence = float(proba[pred_idx])
                        index_to_class = active_meta.get('index_to_class') or list(active_label_encoder.classes_)
                        predicted_label = index_to_class[pred_idx]
                        
                    else:
                        self.seq_buffer.clear()
                        self.motion_values.clear()
                        self.previous_raw_landmarks = None
                        self.motion_state = 'STATIC'
                        self.static_streak = 0
                        self.word_builder._reset_tracking()
                        
                        if self.word_builder.current_word:
                            if self.no_hand_since is None:
                                self.no_hand_since = now
                            elif now - self.no_hand_since >= self.auto_space_after:
                                self.word_builder.add_space()
                                self.no_hand_since = None
                                
                    confirmed = None
                    if predicted_label is not None and confidence >= self.confidence_threshold:
                        confirmed = self.word_builder.update(predicted_label)
                    elif predicted_label is not None:
                        self.word_builder._reset_tracking()
                        
                    if predicted_label and confidence >= self.confidence_threshold:
                        g_type = "Word" if self.motion_state == 'DYNAMIC' or is_word_label(predicted_label) else "Letter"
                        frame = display_text(frame, f"{g_type}: {predicted_label} ({confidence:.0%})", (10, 30), font_scale=1.0, color=(0, 255, 0), thickness=2)
                    elif predicted_label:
                        g_type = "Word" if self.motion_state == 'DYNAMIC' or is_word_label(predicted_label) else "Letter"
                        frame = display_text(frame, f"{g_type}: {predicted_label} (low)", (10, 30), font_scale=0.9, color=(0, 165, 255), thickness=2)
                    
                    if confirmed:
                        frame = display_text(frame, f"+ '{confirmed}'", (10, 130), font_scale=0.9, color=(0, 255, 255), thickness=2)
                        
                    self.current_prediction_text = normalize_sentence_text(self.word_builder.sentence + " " + self.word_builder.current_word)
                
                except Exception as e:
                    print(f"Prediction Error: {e}")
            else:
                frame = display_text(frame, "System Idle - Press START INFERENCE", (10, 30), font_scale=0.8, color=(0, 0, 255), thickness=2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        self.cap.release()