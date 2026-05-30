"""
MediaPipe Utilities for Hand Landmark Detection
This module provides helper functions for detecting and extracting hand landmarks
using MediaPipe Hands solution.
"""

import os
import cv2
import numpy as np
import sys
import types

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def _install_tensorflow_doc_stub():
    """Install a minimal tensorflow.tools.docs.doc_controls stub for MediaPipe."""
    tensorflow_module = types.ModuleType('tensorflow')
    tools_module = types.ModuleType('tensorflow.tools')
    docs_module = types.ModuleType('tensorflow.tools.docs')
    doc_controls_module = types.ModuleType('tensorflow.tools.docs.doc_controls')

    def _identity(value=None, *args, **kwargs):
        return value

    def _decorator_factory(*args, **kwargs):
        def _decorator(obj):
            return obj

        return _decorator

    doc_controls_module.do_not_generate_docs = _identity
    doc_controls_module.do_not_doc_inheritable = _identity
    doc_controls_module.set_deprecated = _identity
    doc_controls_module.inheritable_header = _decorator_factory
    doc_controls_module.header = doc_controls_module.inheritable_header
    doc_controls_module.get_header = lambda obj: None
    doc_controls_module.get_inheritable_header = lambda obj: None

    tensorflow_module.tools = tools_module
    tools_module.docs = docs_module
    docs_module.doc_controls = doc_controls_module

    sys.modules['tensorflow'] = tensorflow_module
    sys.modules['tensorflow.tools'] = tools_module
    sys.modules['tensorflow.tools.docs'] = docs_module
    sys.modules['tensorflow.tools.docs.doc_controls'] = doc_controls_module


class HandDetector:
    """
    A class to handle hand detection and landmark extraction using MediaPipe.
    """
    
    def __init__(self, static_image_mode=False, max_num_hands=1, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the HandDetector with MediaPipe Hands.
        
        Args:
            static_image_mode (bool): Whether to treat input as static images
            max_num_hands (int): Maximum number of hands to detect
            min_detection_confidence (float): Minimum confidence for hand detection
            min_tracking_confidence (float): Minimum confidence for hand tracking
        """
        try:
            import mediapipe as mp
        except ImportError as exc:
            if 'tensorflow' not in str(exc).lower():
                raise
            _install_tensorflow_doc_stub()
            import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def find_hands(self, frame, draw=True):
        """
        Detect hands in the given frame and optionally draw landmarks.
        
        Args:
            frame (numpy.ndarray): Input image frame (BGR format)
            draw (bool): Whether to draw landmarks on the frame
        
        Returns:
            tuple: (processed_frame, results) where results contain hand landmarks
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks if detected and draw is True
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return frame, results
    
    def extract_landmarks(self, results, hand_index=0):
        """
        Extract hand landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe results object containing hand landmarks
            hand_index: Index of the hand to extract (0 for first hand, 1 for second)
        
        Returns:
            numpy.ndarray or None: Flattened array of landmarks (21 landmarks * 3 coords = 63 values)
                                    or None if no hand detected
        """
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
            # Get the specified hand's landmarks
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            
            # Extract x, y, z coordinates for all 21 landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def extract_all_landmarks(self, results):
        """
        Extract landmarks from all detected hands.
        
        Args:
            results: MediaPipe results object containing hand landmarks
        
        Returns:
            list: List of numpy arrays, one for each detected hand
        """
        all_hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                all_hands.append(np.array(landmarks))
        
        return all_hands if all_hands else None
    
    def get_hand_count(self, results):
        """
        Get the number of hands detected.
        
        Args:
            results: MediaPipe results object
        
        Returns:
            int: Number of hands detected
        """
        if results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0
    
    def extract_landmarks_normalized(self, results, frame_shape, hand_index=0):
        """
        Extract and normalize hand landmarks relative to the hand's bounding box.
        This provides better generalization for different hand sizes and positions.
        
        Args:
            results: MediaPipe results object containing hand landmarks
            frame_shape: Shape of the input frame (height, width, channels)
            hand_index: Index of the hand to extract (0 for first hand, 1 for second)
        
        Returns:
            numpy.ndarray or None: Normalized flattened array of landmarks
        """
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            
            # Extract all coordinates
            landmarks = []
            x_coords = []
            y_coords = []
            
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            
            # Find min and max to normalize relative to hand bounding box
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Normalize coordinates relative to bounding box
            for landmark in hand_landmarks.landmark:
                # Normalize x and y to be relative to hand bounding box
                if x_max - x_min > 0:
                    norm_x = (landmark.x - x_min) / (x_max - x_min)
                else:
                    norm_x = 0.5
                
                if y_max - y_min > 0:
                    norm_y = (landmark.y - y_min) / (y_max - y_min)
                else:
                    norm_y = 0.5
                
                landmarks.extend([norm_x, norm_y, landmark.z])
            
            return np.array(landmarks)
        
        return None
    
    def extract_all_landmarks_normalized(self, results, frame_shape):
        """
        Extract and normalize landmarks from all detected hands.
        
        Args:
            results: MediaPipe results object containing hand landmarks
            frame_shape: Shape of the input frame (height, width, channels)
        
        Returns:
            list: List of normalized numpy arrays, one for each detected hand
        """
        all_hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                x_coords = []
                y_coords = []
                
                for landmark in hand_landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                
                # Find min and max to normalize relative to hand bounding box
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Normalize coordinates relative to bounding box
                for landmark in hand_landmarks.landmark:
                    if x_max - x_min > 0:
                        norm_x = (landmark.x - x_min) / (x_max - x_min)
                    else:
                        norm_x = 0.5
                    
                    if y_max - y_min > 0:
                        norm_y = (landmark.y - y_min) / (y_max - y_min)
                    else:
                        norm_y = 0.5
                    
                    landmarks.extend([norm_x, norm_y, landmark.z])
                
                all_hands.append(np.array(landmarks))
        
        return all_hands if all_hands else None
    
    def close(self):
        """
        Release MediaPipe resources.
        """
        self.hands.close()


def compute_engineered_features(landmarks):
    """
    Compute additional engineered features from a 63-value landmark array.
    Adds pairwise distances, joint angles, and finger extension indicators.

    Args:
        landmarks (numpy.ndarray): Flattened array of 21 landmarks * 3 coords = 63 values

    Returns:
        numpy.ndarray: Extended feature array (63 base + 15 distances + 10 angles + 5 extensions = 93)
    """
    points = landmarks.reshape(21, 3)

    # Pairwise distances between key landmarks
    distance_pairs = [
        (4, 8), (4, 12), (4, 16), (4, 20),
        (8, 12), (8, 16), (8, 20),
        (12, 16), (12, 20),
        (16, 20),
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    ]
    distances = [np.linalg.norm(points[i] - points[j]) for i, j in distance_pairs]

    # Angles at finger joints (measures finger curl)
    angle_triplets = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    angles = []
    for a, b, c in angle_triplets:
        v1 = points[a] - points[b]
        v2 = points[c] - points[b]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Finger extension indicators
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [2, 5, 9, 13, 17]
    wrist = points[0]
    extensions = []
    for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
        tip_dist = np.linalg.norm(points[tip_idx] - wrist)
        mcp_dist = np.linalg.norm(points[mcp_idx] - wrist)
        extensions.append(1.0 if tip_dist > mcp_dist else 0.0)

    return np.concatenate([landmarks, distances, angles, extensions])


def get_engineered_feature_names():
    """
    Get column names for the full engineered feature set.

    Returns:
        list: List of feature name strings (93 total)
    """
    columns = []
    for i in range(21):
        columns.extend([f'x{i}', f'y{i}', f'z{i}'])

    distance_pairs = [
        (4, 8), (4, 12), (4, 16), (4, 20),
        (8, 12), (8, 16), (8, 20),
        (12, 16), (12, 20),
        (16, 20),
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    ]
    for i, j in distance_pairs:
        columns.append(f'dist_{i}_{j}')

    angle_triplets = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]
    for a, b, c in angle_triplets:
        columns.append(f'angle_{a}_{b}_{c}')

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for name in finger_names:
        columns.append(f'ext_{name}')

    return columns


def display_text(frame, text, position=(10, 30), font_scale=1, 
                 color=(0, 255, 0), thickness=2):
    """
    Display text on the frame with a background for better visibility.
    
    Args:
        frame (numpy.ndarray): Input image frame
        text (str): Text to display
        position (tuple): (x, y) position for the text
        font_scale (float): Font scale factor
        color (tuple): Text color in BGR format
        thickness (int): Text thickness
    
    Returns:
        numpy.ndarray: Frame with text overlaid
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - text_height - 5),
        (position[0] + text_width + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        position,
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )
    
    return frame


def get_fps(prev_time, curr_time):
    """
    Calculate frames per second.
    
    Args:
        prev_time (float): Previous frame timestamp
        curr_time (float): Current frame timestamp
    
    Returns:
        float: FPS value
    """
    if curr_time - prev_time > 0:
        fps = 1.0 / (curr_time - prev_time)
    else:
        fps = 0
    
    return fps


# ──────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS FOR STATIC/DYNAMIC SPLIT
# ──────────────────────────────────────────────────────────────────────────

def aspect_aware_padding(frame, target_size=(640, 480)):
    """
    Apply aspect-aware letterbox padding to ensure consistent preprocessing
    without distorting the hand.
    
    Resizes frame to target_size while maintaining aspect ratio and padding
    any empty space with a neutral color.
    
    Args:
        frame (numpy.ndarray): Input image (BGR format)
        target_size (tuple): Target (width, height) for the padded output
    
    Returns:
        tuple: (padded_frame, scale, top_left_offset)
            - padded_frame: Resized and padded frame
            - scale: Scaling factor applied
            - top_left_offset: (x, y) offset of original frame in padded frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to fit within target while maintaining aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded frame with neutral gray background
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 128
    
    # Calculate offset to center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized frame in center
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return padded, scale, (x_offset, y_offset)


def normalize_landmarks_to_unit_square(landmarks):
    """
    Normalize landmarks to a unit square [0, 1] based on their bounding box,
    ensuring consistent scale and position invariance.
    
    This ensures static digits maintain consistent proportions regardless of
    hand size or distance from camera.
    
    Args:
        landmarks (numpy.ndarray): Flattened array of 21 landmarks × 3 coords (63 values)
    
    Returns:
        numpy.ndarray: Normalized landmarks, or original if normalization fails
    """
    if landmarks is None or len(landmarks) < 63:
        return landmarks
    
    try:
        points = landmarks.reshape(21, 3)
        
        # Extract x and y coordinates (ignore z for bounding box)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Find bounding box
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Normalize each landmark
        normalized = points.copy()
        normalized[:, 0] = (points[:, 0] - x_min) / x_range
        normalized[:, 1] = (points[:, 1] - y_min) / y_range
        # Keep z-coordinate as-is (already normalized by MediaPipe)
        
        return normalized.flatten()
    except Exception:
        # On any error, return original landmarks
        return landmarks


def compute_hand_bounding_box(landmarks):
    """
    Compute bounding box dimensions for a hand from landmarks.
    
    Useful for checking hand size and position consistency.
    
    Args:
        landmarks (numpy.ndarray): Flattened array of 21 landmarks × 3 coords (63 values)
    
    Returns:
        dict: {
            'x_min', 'x_max', 'y_min', 'y_max': Bounding box coordinates
            'width', 'height': Bounding box dimensions
            'center_x', 'center_y': Center of bounding box
            'area': Approximate area (width * height)
        }
        or None if landmarks are invalid
    """
    if landmarks is None or len(landmarks) < 63:
        return None
    
    try:
        points = landmarks.reshape(21, 3)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        
        width = x_max - x_min
        height = y_max - y_min
        
        return {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': width,
            'height': height,
            'center_x': (x_min + x_max) / 2,
            'center_y': (y_min + y_max) / 2,
            'area': width * height
        }
    except Exception:
        return None
