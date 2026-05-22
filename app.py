"""
Flask Web Server for ISL Gesture Recognition Admin Dashboard
Connects the web interface to the existing Python pipeline.
"""

import os
import sys
import json
import threading
import queue
import shutil
import cv2
import time
from pathlib import Path
from datetime import datetime
import logging

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask import Response
from flask_cors import CORS

from scripts.collect_data import collect_data, collect_video_sequence
from scripts.extract_landmarks import extract_landmarks_from_dataset
from scripts.train_model import train_model
from scripts.realtime_predict import predict_realtime, predict_words, predict_sentence, predict_stable_sentence

# ─── APP SETUP ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configure logging to capture output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── GLOBAL STATE ──────────────────────────────────────────────────────────
class SystemState:
    """Thread-safe system state management."""
    def __init__(self):
        self.state = "IDLE"  # IDLE, CAPTURING, TRAINING, EXTRACTING, RECOGNIZING
        self.progress = 0  # 0-100
        self.message = ""
        self.gesture_count = 0
        self.logs = queue.Queue(maxsize=1000)
        # Recognition results
        self.last_recognized_gesture = None
        self.last_confidence = None
        self.recognition_text = ""  # For multi-word or sentence output
        self._lock = threading.Lock()
    
    def update(self, state=None, progress=None, message=None):
        """Update system state."""
        with self._lock:
            if state:
                self.state = state
            if progress is not None:
                self.progress = progress
            if message:
                self.message = message
                self.log(message)
    
    def update_recognition(self, gesture=None, confidence=None, text=None):
        """Update recognition results."""
        with self._lock:
            if gesture is not None:
                self.last_recognized_gesture = gesture
            if confidence is not None:
                self.last_confidence = confidence
            if text is not None:
                self.recognition_text = text
    
    def log(self, message):
        """Add a message to the log queue."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        try:
            self.logs.put_nowait(log_entry)
        except queue.Full:
            self.logs.get_nowait()
            self.logs.put_nowait(log_entry)
    
    def get_status(self):
        """Get current status as dict."""
        with self._lock:
            return {
                "state": self.state,
                "progress": self.progress,
                "message": self.message,
                "gesture_count": self.gesture_count,
                "last_recognized_gesture": self.last_recognized_gesture,
                "last_confidence": self.last_confidence,
                "recognition_text": self.recognition_text
            }
    
    def get_logs(self, limit=50):
        """Get recent logs."""
        logs = []
        temp = []
        while not self.logs.empty():
            try:
                logs.append(self.logs.get_nowait())
            except queue.Empty:
                break
        
        # Put logs back and keep last N
        for log in logs[-limit:]:
            temp.append(log)
        for log in temp:
            try:
                self.logs.put_nowait(log)
            except queue.Full:
                pass
        
        return temp

state = SystemState()

# ─── PATHS ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'raw_images')
MODELS_PATH = os.path.join(BASE_DIR, 'models')

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

state.log("App initialized")

# ─── MJPEG STREAM STATE ─────────────────────────────────────────────────-
latest_frame = None
latest_frame_lock = threading.Lock()


def _update_latest_frame(frame):
    """Store the latest frame as JPEG bytes for the MJPEG stream."""
    if frame is None:
        return
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return
    with latest_frame_lock:
        global latest_frame
        latest_frame = buffer.tobytes()


def _update_prediction_result(gesture_label, confidence):
    """Callback to receive prediction results from the prediction functions."""
    state.update_recognition(gesture=gesture_label, confidence=confidence, text=gesture_label)
    state.log(f"Recognized: {gesture_label} (confidence: {confidence:.2%})")


def _mjpeg_generator():
    """Yield JPEG frames for the browser stream."""
    while True:
        with latest_frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# ─── PRE-FLIGHT CHECKS ───────────────────────────────────────────────────

def _model_metadata_path():
    """Return the expected model metadata path."""
    return os.path.join(MODELS_PATH, 'gesture_model.pkl')


def _check_model_available():
    """Ensure the model metadata exists before starting prediction."""
    model_path = _model_metadata_path()
    if not os.path.exists(model_path):
        state.update(state="IDLE", message="Model not found. Train the model first.")
        state.log(f"Model metadata missing: {model_path}")
        return False
    return True


def _check_camera_available(device_index=0):
    """Verify the webcam can be opened and read a frame."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        state.update(state="IDLE", message="Camera not available. Close other apps and retry.")
        state.log("Camera not available (VideoCapture failed)")
        return False

    ret, _ = cap.read()
    cap.release()
    if not ret:
        state.update(state="IDLE", message="Camera opened but no frames read.")
        state.log("Camera opened but failed to read a frame")
        return False

    return True

# ─── ROUTES ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/video_feed')
def video_feed():
    """MJPEG stream for the browser camera preview."""
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ─── API: Status ─────────────────────────────────────────────────────────

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status."""
    status = state.get_status()
    status['gestures'] = get_gesture_list()
    return jsonify(status)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs (SSE endpoint for real-time updates)."""
    logs = state.get_logs(limit=100)
    return jsonify({"logs": logs})

# ─── API: Gesture Management ─────────────────────────────────────────────

def get_gesture_list():
    """Get list of available gestures."""
    gestures = []
    if os.path.exists(DATASET_PATH):
        for item in os.listdir(DATASET_PATH):
            item_path = os.path.join(DATASET_PATH, item)
            if os.path.isdir(item_path):
                # Count samples (either images or video clips)
                file_count = len(os.listdir(item_path))
                gestures.append({
                    "name": item,
                    "samples": file_count,
                    "path": item_path
                })
    return sorted(gestures, key=lambda x: x['name'])

@app.route('/api/gestures', methods=['GET'])
def list_gestures():
    """List all available gestures."""
    gestures = get_gesture_list()
    return jsonify({
        "count": len(gestures),
        "gestures": gestures
    })

@app.route('/api/gesture/<gesture_name>', methods=['DELETE'])
def delete_gesture(gesture_name):
    """Delete a gesture and its training data."""
    try:
        gesture_path = os.path.join(DATASET_PATH, gesture_name)
        
        if not os.path.exists(gesture_path):
            return jsonify({"error": f"Gesture '{gesture_name}' not found"}), 404
        
        # Remove the gesture folder and all its contents
        shutil.rmtree(gesture_path)
        state.log(f"Deleted gesture: {gesture_name}")
        
        return jsonify({
            "success": True,
            "message": f"Gesture '{gesture_name}' deleted successfully",
            "gestures": get_gesture_list()
        })
    except Exception as e:
        state.log(f"Error deleting gesture: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── API: Data Collection ─────────────────────────────────────────────────

def _collect_data_worker(gesture_name, num_samples, use_video, num_clips, clip_frames):
    """Background worker for data collection."""
    try:
        state.update(state="CAPTURING", progress=0, message=f"Starting capture for '{gesture_name}'...")
        
        if use_video:
            state.log(f"Recording {num_clips} video clips for '{gesture_name}'...")
            collect_video_sequence(gesture_name, num_clips=num_clips, clip_frames=clip_frames)
        else:
            state.log(f"Capturing {num_samples} images for '{gesture_name}'...")
            # Note: The existing collect_data uses interactive mode; we'll need a non-interactive version
            collect_data(gesture_name, num_samples=num_samples)
        
        state.update(state="IDLE", progress=100, message=f"Capture complete for '{gesture_name}'")
        state.log(f"Successfully captured {num_samples} samples for '{gesture_name}'")
    except Exception as e:
        state.update(state="IDLE", progress=0, message=f"Error during capture: {str(e)}")
        state.log(f"Capture error: {str(e)}")

@app.route('/api/capture/start', methods=['POST'])
def start_capture():
    """Start data collection for a gesture."""
    try:
        data = request.get_json()
        gesture_name = data.get('gesture_name', '').strip().upper()
        num_samples = int(data.get('num_samples', 50))
        use_video = data.get('use_video', False)
        num_clips = int(data.get('num_clips', 50))
        clip_frames = int(data.get('clip_frames', 30))
        
        if not gesture_name:
            return jsonify({"error": "Gesture name is required"}), 400
        
        if state.state != "IDLE":
            return jsonify({"error": f"System is currently {state.state}. Wait for it to finish."}), 409
        
        # Start collection in background thread
        thread = threading.Thread(
            target=_collect_data_worker,
            args=(gesture_name, num_samples, use_video, num_clips, clip_frames),
            daemon=True
        )
        thread.start()
        
        state.update(state="CAPTURING", message=f"Starting capture for '{gesture_name}'...")
        
        return jsonify({
            "success": True,
            "message": f"Started capture for '{gesture_name}'",
            "gesture_name": gesture_name
        })
    except Exception as e:
        state.log(f"Error starting capture: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── API: Landmark Extraction ─────────────────────────────────────────────

def _extract_landmarks_worker(use_normalized):
    """Background worker for landmark extraction."""
    try:
        state.update(state="EXTRACTING", progress=0, message="Starting landmark extraction...")
        state.log("Extracting landmarks from collected images...")
        
        extract_landmarks_from_dataset(use_normalized=use_normalized)
        
        state.update(state="IDLE", progress=100, message="Landmark extraction complete")
        state.log("Landmark extraction completed successfully")
    except Exception as e:
        state.update(state="IDLE", progress=0, message=f"Extraction error: {str(e)}")
        state.log(f"Extraction error: {str(e)}")

@app.route('/api/landmarks/extract', methods=['POST'])
def extract_landmarks():
    """Extract landmarks from collected gesture images."""
    try:
        data = request.get_json()
        use_normalized = data.get('use_normalized', True)
        
        if state.state != "IDLE":
            return jsonify({"error": f"System is currently {state.state}. Wait for it to finish."}), 409
        
        thread = threading.Thread(
            target=_extract_landmarks_worker,
            args=(use_normalized,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Started landmark extraction"
        })
    except Exception as e:
        state.log(f"Error starting extraction: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── API: Model Training ─────────────────────────────────────────────────

def _train_model_worker(test_size):
    """Background worker for model training."""
    try:
        state.update(state="TRAINING", progress=0, message="Starting model training...")
        state.log("Training gesture recognition model...")
        
        train_model(test_size=test_size)
        
        state.update(state="IDLE", progress=100, message="Model training complete")
        state.log("Model training completed successfully")
    except Exception as e:
        state.update(state="IDLE", progress=0, message=f"Training error: {str(e)}")
        state.log(f"Training error: {str(e)}")

@app.route('/api/model/train', methods=['POST'])
def train():
    """Train the gesture recognition model."""
    try:
        data = request.get_json()
        test_size = float(data.get('test_size', 0.2))
        
        if state.state != "IDLE":
            return jsonify({"error": f"System is currently {state.state}. Wait for it to finish."}), 409
        
        thread = threading.Thread(
            target=_train_model_worker,
            args=(test_size,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Started model training"
        })
    except Exception as e:
        state.log(f"Error starting training: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── API: Real-time Prediction ──────────────────────────────────────────

stream_active = False
stream_thread = None
stream_stop_event = None

def _stream_worker(mode, confidence_threshold, stop_event):
    """Background worker for real-time gesture prediction."""
    global stream_active
    try:
        state.update(state="RECOGNIZING", message=f"Starting {mode} mode...")
        state.log(f"Started real-time prediction in {mode} mode")
        
        if mode == "letter":
            predict_realtime(
                use_normalized=True,
                confidence_threshold=confidence_threshold,
                stop_event=stop_event,
                display=False,
                frame_callback=_update_latest_frame,
                results_callback=_update_prediction_result
            )
        elif mode == "word":
            predict_words(
                use_normalized=True,
                confidence_threshold=confidence_threshold,
                stop_event=stop_event,
                display=False,
                frame_callback=_update_latest_frame,
                results_callback=_update_prediction_result
            )
        elif mode == "sentence":
            predict_sentence(
                use_normalized=True,
                confidence_threshold=confidence_threshold,
                stop_event=stop_event,
                display=False,
                frame_callback=_update_latest_frame,
                results_callback=_update_prediction_result
            )
        elif mode == "stable":
            predict_stable_sentence(
                use_normalized=True,
                confidence_threshold=confidence_threshold,
                stop_event=stop_event,
                display=False,
                frame_callback=_update_latest_frame,
                results_callback=_update_prediction_result
            )
        
        state.update(state="IDLE", message="Recognition stopped")
        state.log("Prediction stopped")
    except Exception as e:
        state.update(state="IDLE", message=f"Recognition error: {str(e)}")
        state.log(f"Prediction error: {str(e)}")
    finally:
        stream_active = False

@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start real-time gesture prediction."""
    global stream_active, stream_thread, stream_stop_event
    try:
        if stream_active:
            return jsonify({"error": "Stream is already running"}), 409

        if state.state != "IDLE":
            return jsonify({"error": f"System is currently {state.state}. Wait for it to finish."}), 409
        
        data = request.get_json()
        mode = data.get('mode', 'letter')  # letter, word, sentence, stable
        confidence_threshold = float(data.get('confidence_threshold', 0.7))

        if not _check_model_available():
            return jsonify({"error": "Model not found. Run extraction + training first."}), 409

        if not _check_camera_available():
            return jsonify({"error": "Camera not available. Close other apps and retry."}), 409
        
        stream_active = True
        stream_stop_event = threading.Event()
        stream_thread = threading.Thread(
            target=_stream_worker,
            args=(mode, confidence_threshold, stream_stop_event),
            daemon=True
        )
        stream_thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Started {mode} recognition mode",
            "mode": mode
        })
    except Exception as e:
        stream_active = False
        state.log(f"Error starting stream: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop real-time gesture prediction."""
    global stream_active, stream_stop_event
    try:
        if stream_stop_event is not None:
            stream_stop_event.set()
        stream_active = False
        with latest_frame_lock:
            global latest_frame
            latest_frame = None
        state.update(state="IDLE", message="Stream stopped")
        state.log("User stopped real-time prediction")
        
        return jsonify({
            "success": True,
            "message": "Stream stopped"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── ERROR HANDLERS ─────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    state.log(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# ─── MAIN ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*70)
    print(" "*15 + "ISL GESTURE RECOGNITION - WEB DASHBOARD")
    print("="*70)
    print(f"\n✓ Server starting on http://localhost:5000")
    print(f"✓ Dataset path: {DATASET_PATH}")
    print(f"✓ Models path: {MODELS_PATH}")
    print("\nPress Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(debug=True, host='localhost', port=5000, use_reloader=False)
