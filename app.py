import os
from flask import Flask, render_template, Response, jsonify, request
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
    return jsonify({
        "status_msg": runtime.status_msg,
        "prediction_text": runtime.current_prediction_text,
        "is_running": runtime.is_running
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
        return jsonify({"status": "success", "message": f"Collected data for {data.get('gestureName')}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/extract', methods=['POST'])
def api_extract():
    try:
        extract_landmarks_from_dataset(use_normalized=request.json.get('useNormalized'))
        extract_sequences_from_dataset(use_normalized=request.json.get('useNormalized'), mode='word')
        return jsonify({"status": "success", "message": "Landmarks extracted."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        train_model(mode='word', test_size=float(request.json.get('testSize')))
        return jsonify({"status": "success", "message": "Model trained."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def launch():
    print("\n" + "="*50)
    print("🚀 FLASK APP IS RUNNING ON PORT 5000 🚀")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    launch()