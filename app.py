import os
import sys
from flask import Flask, render_template, Response, jsonify, request

# ==========================================
# TERMINAL LOG INTERCEPTOR
# ==========================================
class AppLogger:
    def __init__(self):
        self.logs = []
        self.terminal_out = sys.__stdout__
        self.terminal_err = sys.__stderr__

    def write(self, message):
        # Always print to the real terminal
        self.terminal_out.write(message)
        self.terminal_out.flush()
        
        msg = message.strip()
        if msg:
            # Filter out the noisy repeating Flask background requests
            if "HTTP/1.1" in msg and any(ep in msg for ep in ["/status", "/video_feed", "/api/dataset_info", "/static"]):
                return
            self.logs.append(msg)

    def flush(self):
        self.terminal_out.flush()
        self.terminal_err.flush()

# Override stdout and stderr to capture print() and tracebacks
app_logger = AppLogger()
sys.stdout = app_logger
sys.stderr = app_logger

# ==========================================

from flask_backend import FlaskISLRuntime

# Import original scripts for admin panel
from scripts.collect_data import collect_data, collect_video_sequence
from scripts.extract_landmarks import extract_landmarks_from_dataset, extract_sequences_from_dataset
from scripts.train_model import train_model

app = Flask(__name__)
runtime = FlaskISLRuntime()
runtime.initialize_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(runtime.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    runtime.start_inference()
    return jsonify({"status": "success", "message": runtime.status_msg})

@app.route('/stop', methods=['POST'])
def stop():
    runtime.stop_inference()
    return jsonify({"status": "success", "message": runtime.status_msg})

@app.route('/clear', methods=['POST'])
def clear():
    runtime.clear_text()
    return jsonify({"status": "success"})

@app.route('/status', methods=['GET'])
def status():
    # Grab the terminal logs since the last poll, then clear the queue
    logs_to_send = list(app_logger.logs)
    app_logger.logs.clear()
    
    return jsonify({
        "status_msg": runtime.status_msg,
        "prediction_text": runtime.current_prediction_text,
        "is_running": runtime.is_running,
        "new_logs": logs_to_send
    })

# --- DATASET FOLDER ROUTE ---
@app.route('/api/dataset_info', methods=['GET'])
def dataset_info():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(base_dir, 'dataset', 'raw_images')
    dynamic_dir = os.path.join(base_dir, 'dataset', 'video_clips')
    
    static_gestures = []
    dynamic_gestures = []
    
    if os.path.exists(static_dir):
        static_gestures = [d for d in os.listdir(static_dir) if os.path.isdir(os.path.join(static_dir, d))]
    if os.path.exists(dynamic_dir):
        dynamic_gestures = [d for d in os.listdir(dynamic_dir) if os.path.isdir(os.path.join(dynamic_dir, d))]
        
    return jsonify({
        "static": sorted(static_gestures),
        "dynamic": sorted(dynamic_gestures)
    })

# --- ADMIN ROUTES ---
@app.route('/api/collect', methods=['POST'])
def api_collect():
    data = request.json
    try:
        if data.get('captureType') == "video":
            collect_video_sequence(data.get('gestureName').upper(), num_clips=int(data.get('numSamples')), clip_frames=30)
        else:
            collect_data(data.get('gestureName').upper(), int(data.get('numSamples')))
        return jsonify({"status": "success", "message": f"Finished collection process for {data.get('gestureName')}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/extract', methods=['POST'])
def api_extract():
    try:
        extract_landmarks_from_dataset(use_normalized=request.json.get('useNormalized'))
        extract_sequences_from_dataset(use_normalized=request.json.get('useNormalized'), mode='word')
        return jsonify({"status": "success", "message": "Landmarks extracted successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        train_model(mode='word', test_size=float(request.json.get('testSize')))
        return jsonify({"status": "success", "message": "Model trained successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json
    try:
        if 'mode' in data:
            runtime.inference_mode = data['mode']
            return jsonify({"status": "success", "message": f"Inference mode forced to: {data['mode']}"})
        if 'reload' in data and data['reload']:
            msg = runtime.reload_models()
            return jsonify({"status": "success", "message": msg})
        return jsonify({"status": "error", "message": "Invalid request"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def launch():
    print("\n" + "="*50)
    print("🚀 FLASK APP IS RUNNING ON PORT 5000 🚀")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    launch()