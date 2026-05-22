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

## 🎮 Dashboard Features

### User View (Prediction)
- **Live Recognition**: Real-time gesture recognition from webcam
- **Translation Display**: Large, high-contrast text showing recognized gestures
- **Status Light**: Color-coded system state indicator
  - 🔵 Blue (IDLE) - System ready
  - 🟢 Green (RECOGNIZING) - Currently predicting
  - 🟡 Yellow (CAPTURING) - Collecting data
  - 🟠 Orange (TRAINING/EXTRACTING) - Processing

### Admin View (Control Panel)

#### 📹 Data Collection Tab
- **Gesture Name Input**: Enter custom gesture labels (A-Z, 0-9, words)
- **Capture Type Toggle**:
  - Static Images: For single-pose gestures (letters, digits)
  - Video Clips: For dynamic/motion gestures (HELLO, THANK YOU)
- **Sample Configuration**: Set number of samples to collect
- **Real-time Progress**: Visual feedback during capture

#### 📚 Gesture Library Tab
- **Gesture List**: View all saved gestures with sample counts
- **Delete Gestures**: Remove unwanted training data
- **Landmark Extraction**: Extract hand landmarks from images before training
- **Normalized vs Raw**: Toggle between normalized and raw feature extraction

#### 🧠 Model Training Tab
- **Train Button**: Start BiLSTM model training
- **Test Size Configuration**: Adjust train/test split ratio
- **Progress Bar**: Visual training progress indicator
- **Training Information**: Details about the training process

#### 💻 Console Tab
- **Real-time Logs**: See all system operations and status updates
- **Error Reporting**: Catch and display any errors
- **Auto-scroll**: Keep console scrolled to latest message
- **Clear Button**: Reset console history

---

## 🏗️ Architecture

### Backend (Flask)
- **app.py**: Main Flask server with API routes
- Endpoints for all operations (capture, train, predict)
- Background threading for long-running tasks
- Status polling system
- Logging system

### Frontend (HTML/CSS/JS)
- **index.html**: Main dashboard interface
  - Two-view system (User & Admin)
  - Tab-based admin panel
  - Responsive design
  
- **static/styles.css**: Modern styling with:
  - CSS variables for easy customization
  - High-contrast accessibility
  - Responsive layout
  - Status color system
  
- **static/main.js**: Interactive logic
  - API communication (fetch)
  - Real-time status polling
  - Form handling
  - Visual feedback
  - Console logging

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
- `POST /api/landmarks/extract` - Extract landmarks from images
- `POST /api/model/train` - Train the gesture model

### Real-time Prediction
- `POST /api/stream/start` - Start live gesture recognition
- `POST /api/stream/stop` - Stop live gesture recognition

---

## 🎨 Customization

### Theme Colors
Edit the CSS variables in `static/styles.css` (top of file):

```css
:root {
    --color-primary: #0066cc;      /* Primary blue */
    --color-success: #00aa00;      /* Success green */
    --color-danger: #dd0000;       /* Error red */
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

### Progress Tracking
- Progress bar updates via status polling
- Real-time log updates via separate polling mechanism
- No blocking operations on main thread

---

## 🚀 Workflow

### 1. Collect Training Data
1. Switch to Admin View
2. Go to "Data Collection" tab
3. Enter gesture name and set sample count
4. Click "Start Capture"
5. Repeat for all desired gestures

### 2. Extract Landmarks
1. Go to "Gesture Library" tab
2. Click "Extract Landmarks"
3. Wait for completion (shows in console)

### 3. Train Model
1. Go to "Model Training" tab
2. Adjust test size if needed
3. Click "Train Model"
4. Monitor progress and console
5. Training completes and saves model

### 4. Test Recognition
1. Switch to User View
2. Select prediction mode (Letter/Word/Sentence)
3. Adjust confidence threshold
4. Click "Start Prediction"
5. Perform gestures in front of camera
6. See real-time results in Translation Display

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
