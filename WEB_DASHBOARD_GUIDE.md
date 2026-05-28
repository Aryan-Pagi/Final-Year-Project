# ISL Gesture Recognition - Web Dashboard

## 📦 Quick Start

### Prerequisites
- Python 3.9+
- Virtual environment activated
- Dependencies installed

### Installation

1. **Install Flask and dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Flask server:**
```bash
cd Final-Year-Project
python app.py
```

3. **Access the dashboard:**
Open your browser and go to:
```
http://localhost:5000
```

---

## � Recommended Workflow: Digit-First Phase

**Start with digits 0-9 to build a stable classifier, then extend to letters and words.**

### Phase 1: Collect & Train Digits 0-9

1. **Go to Admin View → Data Collection tab**
2. **For each digit (0-9):**
   - Enter gesture name: **0** (then 1, 2, ..., 9)
   - Select capture type: **Static Images**
   - Set sample count: **100** (aim for balanced counts)
   - Click **Start Capture**
   - Show your hand making the digit gesture (centered, well-lit, plain background)
   - Press SPACE to capture each image
   - Wait for completion

3. **Extract landmarks:**
   - Go to **Gesture Library** tab
   - Click **Extract Landmarks**
   - Select: **Use normalized landmarks** (recommended)
   - Monitor progress in console

4. **Train the model:**
   - Go to **Model Training** tab
   - Test size: **0.2** (80% train, 20% test)
   - Click **Train Model**
   - Monitor progress and console output
   - Check test accuracy in logs (should be >90%)

5. **Test recognition:**
   - Switch to **User View**
   - Click **Start Prediction**
   - Show digits 0-9 to camera
   - Verify predictions are correct

### Phase 2: Extend to A-Z (After Digit Base is Stable)

- Repeat Phase 1 steps, but collect letters **A-Z** instead
- Re-extract landmarks and re-train

### Phase 3: Add Dynamic Words (Advanced)

- Collect **video clips** for words like HELLO, THANK YOU, etc.
- The system will automatically train a separate BiLSTM for dynamic gestures
- Predictions will be routed to the correct model (static or dynamic)

---

## 🎮 Dashboard Features

### User View (Prediction)
- **Live Recognition**: Real-time gesture recognition from webcam
- **Translation Display**: Large, high-contrast text showing recognized gestures and confidence
- **Status Light**: Color-coded system state indicator
  - 🔵 Blue (IDLE) - System ready
  - 🟢 Green (RECOGNIZING) - Currently predicting
  - 🟡 Yellow (CAPTURING) - Collecting data
  - 🟠 Orange (TRAINING/EXTRACTING) - Processing

### Admin View (Control Panel)

#### 📹 Data Collection Tab
- **Gesture Name Input**: Enter custom gesture labels (digits 0-9, letters A-Z, or words)
- **Capture Type Toggle**:
  - **Static Images**: For single-pose gestures (digits, letters, still-pose words)
    - Best for: Clear, centered, front-facing hand poses
    - Samples recommended: 50-100 per gesture
  - **Video Clips**: For dynamic/motion gestures (HELLO, THANK YOU, GOOD, BAD, etc.)
    - Best for: Gestures with hand movement or shape change over time
    - Clips recommended: 50 clips × 30 frames each per gesture
- **Sample Configuration**: 
  - Static: Number of images
  - Video: Number of clips and frames per clip
- **Real-time Progress**: Visual feedback and console output during capture

#### 📚 Gesture Library Tab
- **Gesture List**: View all saved gestures with sample counts
- **Delete Gestures**: Remove unwanted training data
- **Landmark Extraction**:
  - Choose: **Normalized vs Raw landmarks**
  - Recommended: **Normalized** for better generalization
  - Processes all images in `dataset/raw_images/`
  - Reports detection success rate and extracted features

#### 🧠 Model Training Tab
- **Model Type Selection**: 
  - For static digits/letters: Uses RandomForest (lightweight, fast)
  - For dynamic clips: Uses compact BiLSTM (handles temporal sequences)
  - System auto-detects based on collected data
- **Test Size Configuration**: Adjust train/test split ratio (default: 0.2 = 80/20)
- **Progress Bar**: Visual training progress indicator
- **Training Information**: 
  - Class balance check (warns if imbalanced)
  - Model architecture details
  - Validation and test accuracy
  - Training time and model size

#### 💻 Console Tab
- **Real-time Logs**: See all system operations and status updates
- **Error Reporting**: Catch and display any errors during capture/training
- **Auto-scroll**: Keep console scrolled to latest message
- **Clear Button**: Reset console history
- **Sample output**:
  ```
  ✓ Dataset loaded: 1000 samples, 10 classes
  Class Balance Check:
    - Imbalance Ratio (Max/Min): 1.05
    - All classes well-balanced ✓
  ✓ Model trained successfully
  - Test Accuracy: 96.50%
  ```

---

## 🏗️ Architecture

### Backend (Flask)
- **app.py**: Main Flask server with API routes
  - Endpoints for capture, landmark extraction, training, prediction
  - Background threading for long-running tasks
  - Status polling system
  - Logging system with queue-based message handling

### Frontend (HTML/CSS/JS)
- **index.html**: Main dashboard interface
  - Dual-view system (User View & Admin View)
  - Tab-based admin panel
  - Responsive design for various screen sizes
  
- **static/styles.css**: Modern styling
  - CSS variables for easy customization
  - High-contrast accessibility
  - Responsive layout
  - Status color system (Blue/Green/Yellow/Orange)
  
- **static/main.js**: Interactive logic
  - API communication (fetch)
  - Real-time status polling
  - Form handling and validation
  - Visual feedback
  - Console logging with auto-scroll

---

## 📡 API Routes

All routes return JSON responses.

### Status & Monitoring
- `GET /api/status` - Get current system state
- `GET /api/logs` - Retrieve recent system logs

### Gesture Management
- `GET /api/gestures` - List all saved gestures
- `DELETE /api/gesture/<name>` - Delete a gesture

### Data Processing
- `POST /api/capture/start` - Start data collection
  - Parameters: `gesture_name`, `num_samples` (static) or `num_clips`, `clip_frames` (video)
- `POST /api/landmarks/extract` - Extract landmarks from images
  - Parameters: `use_normalized` (true/false)
- `POST /api/model/train` - Train the gesture model
  - Parameters: `test_size` (float)

### Real-time Prediction
- `POST /api/stream/start` - Start live gesture recognition
- `POST /api/stream/stop` - Stop live gesture recognition

---

## 🎯 Best Practices

### Data Collection
- **Lighting**: Use consistent, well-lit environments (avoid shadows)
- **Background**: Plain, neutral backgrounds (desk, wall, curtain)
- **Positioning**: Keep hand centered and at similar distance from camera
- **Consistency**: Maintain similar hand orientation across samples
- **Balance**: Collect equal samples for each gesture (100 per digit recommended)
- **Quality**: Clean, clear images are better than large quantities of blurry data

### Model Training
- **Phase approach**: Start with digits 0-9, verify >90% accuracy, then expand
- **Class balance**: Dashboard warns if classes are imbalanced
- **Test size**: Use 0.2 (20% test) for datasets with <1000 samples
- **Monitoring**: Check console output for accuracy, confusion matrix, and warnings

### Real-time Prediction
- **Confidence threshold**: Adjust based on your needs (default 0.7 = 70%)
- **FPS**: Monitor FPS to ensure smooth real-time performance
- **Prediction smoothing**: 7-frame history prevents jittery predictions

---

## 🎨 Customization

### Theme Colors
Edit the CSS variables in `static/styles.css` (top of file):

```css
:root {
    --color-primary: #0066cc;      /* Primary blue */
    --color-success: #00aa00;      /* Success green */
    --color-danger: #dd0000;       /* Error red */
    --color-warning: #ff9900;      /* Warning orange */
    /* ... more colors ... */
}
```

### Status Polling Frequency
In `static/main.js`:
```javascript
const STATUS_POLL_INTERVAL = 1000; // 1 second
const LOGS_POLL_INTERVAL = 500;    // 500ms
```

---

## ⚙️ Technical Details

### Threading Model
- Long-running tasks (capture, training, extraction) run in background threads
- Main Flask thread remains responsive
- Status updates via polling mechanism

### State Management
System states:
- `IDLE` - No operation in progress
- `CAPTURING` - Collecting gesture data
- `EXTRACTING` - Processing landmarks
- `TRAINING` - Training the model
- `RECOGNIZING` - Running real-time prediction

### Model Routing
- **Static gestures** (digits, letters): RandomForest classifier on 93 engineered features
- **Dynamic gestures** (words with motion): Compact BiLSTM on 30-frame sequences
- Automatic detection based on dataset layout and training data

### Progress Tracking
- Progress bar updates via status polling
- Real-time log updates via separate polling mechanism
- No blocking operations on main thread

---

## 🚀 Workflow Summary

### Recommended: Digit-First Approach

1. **Collect Digits 0-9** (100 images each, balanced, well-lit, centered)
2. **Extract Landmarks** (normalized, aim for >90% detection)
3. **Train Model** (RandomForest on static digits, check test accuracy >90%)
4. **Test Recognition** (verify predictions for 0-9)
5. **Expand to A-Z** (after digit base is stable)
6. **Add Dynamic Words** (HELLO, THANK YOU, etc. as video clips)

---


## 🎯 Accessibility Features

- **High Contrast**: All colors meet WCAG AA standards
- **Large Text**: Readable fonts and sizes
- **Visual Feedback**: Every action triggers visible feedback
- **Keyboard Navigation**: Full keyboard support
- **Status Indicators**: Color + text for redundancy
- **Reduced Motion**: Respects system preferences
- **Auto-scaling**: Responsive on all screen sizes

---

## 🐛 Troubleshooting

### Port Already in Use
If port 5000 is taken, modify `app.py`:
```python
app.run(debug=True, host='localhost', port=5001)  # Change 5000 to 5001
```

### Module Not Found Errors
Ensure virtual environment is activated:
```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

### Webcam Not Working
- Ensure camera is connected and not in use by other apps
- Check system camera permissions
- Verify OpenCV/MediaPipe installation

### Training Hangs
- Check console logs for errors
- Ensure you have enough RAM (BiLSTM needs ~4GB)
- Verify dataset has sufficient samples (>10 per class)

---

## 📋 Glossary

- **Landmark**: Hand position (21 points per hand detected by MediaPipe)
- **Gesture**: A hand pose or motion (letters A-Z, words like HELLO)
- **BiLSTM**: Bidirectional LSTM neural network for sequence classification
- **Confidence Threshold**: Minimum certainty required to accept a prediction
- **Normalized Landmarks**: Features scaled to be rotation/scale invariant

---

## 📝 Notes

- The webcam feed in the browser is a placeholder; actual video is processed locally by Python
- Training times vary based on:
  - Number of gestures
  - Samples per gesture
  - Hardware (GPU highly recommended)
  - Model complexity
- All data is stored locally; no cloud uploads
- Model persists in `models/bilstm_model.keras`

---

**Version**: 1.0  
**Last Updated**: May 2026  
**Designed for**: Deaf & Mute users | Accessibility first
