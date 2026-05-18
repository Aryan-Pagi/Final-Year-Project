# ISL Gesture Recognition System

Note: README updated to reflect BiLSTM training pipeline and current filenames.

A real-time **Indian Sign Language (ISL)** gesture recognition system using computer vision and machine learning. This system can recognize hand gestures from a webcam and classify them into characters (A–Z and 0–9), with support for training custom gestures.

## 🎯 Features

- **Real-time hand gesture recognition** using webcam
- **MediaPipe-based hand landmark detection** for accurate tracking
- **Machine learning classification** using Random Forest Classifier
- **Extensible design** - easily add new gesture classes
- **User-friendly menu interface** for all operations
- **Normalized landmark features** for better generalization
- **Confidence scoring** for predictions
- **FPS monitoring** for performance tracking

## 🛠️ Technologies Used

- **Python 3.9+**
- **OpenCV** - Computer vision and webcam handling
- **MediaPipe** - Hand landmark detection (21 landmarks per hand)
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning (RandomForestClassifier)

## 📁 Project Structure

```
isl_sign_recognition/
│
├── dataset/
│   ├── raw_images/          # Collected gesture images
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
│   └── landmarks.csv        # Extracted landmark features
│
├── models/
│   └── gesture_model.pkl    # Trained ML model
│
├── scripts/
│   ├── __init__.py
│   ├── collect_data.py      # Data collection from webcam
│   ├── extract_landmarks.py # Landmark extraction from images
│   ├── train_model.py       # Model training
│   └── realtime_predict.py  # Real-time gesture prediction
│
├── utils/
│   ├── __init__.py
│   └── mediapipe_utils.py   # MediaPipe helper functions
│
├── main.py                  # Main menu interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- Webcam
- Windows/Linux/macOS

### Setup Steps

1. **Clone or download the project**

2. **Navigate to the project directory**

   ```bash
   cd "C:\Users\aryan\OneDrive\Documents\Final Year Project\isl_sign_recognition"
   ```

3. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   ```

4. **Activate the virtual environment**

   **Windows:**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **Linux/macOS:**

   ```bash
   source .venv/bin/activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Method 1: Using the Main Menu (Recommended)

Run the main application:

```bash
python main.py
```

The menu provides access to all features:

1. **Collect Data** - Capture gesture images
2. **Extract Landmarks** - Process images and extract features
3. **Train Model** - Train the machine learning model
4. **Real-time Prediction** - Recognize gestures in real-time
5. **Quick Setup** - First-time user guide
6. **Exit** - Quit the application

### Method 2: Running Scripts Individually

#### 1. Collect Data

```bash
python scripts/collect_data.py
```

- Enter the gesture label (e.g., A, B, 1, 2)
- Enter the number of samples to collect
- Show your hand gesture to the webcam
- Press **SPACE** to capture images
- Press **q** to quit

#### 2. Extract Landmarks

```bash
python scripts/extract_landmarks.py
```

- Processes all images in `dataset/raw_images/`
- Extracts 21 hand landmarks (x, y, z coordinates)
- Saves to `dataset/landmarks.csv`

#### 3. Train Model

```bash
python scripts/train_model.py
```

- Loads landmarks from CSV
- Trains a RandomForestClassifier
- Displays accuracy and classification report
- Saves model to `models/gesture_model.pkl`

#### 4. Real-time Prediction

```bash
python scripts/realtime_predict.py
```

- Opens webcam for real-time gesture recognition
- Displays predicted gesture and confidence score
- Press **q** to quit
- Press **f** to toggle FPS display

## 📝 Step-by-Step Guide for First-Time Users

### Step 1: Collect Training Data

For each gesture (A-Z, 0-9), collect 100-200 samples:

```bash
python scripts/collect_data.py
```

Example workflow:

- Enter label: **A**
- Enter samples: **100**
- Show your hand making the 'A' gesture
- Press **SPACE** 100 times to capture images
- Repeat for other labels (B, C, D, ..., 1, 2, 3, ...)

**Tips:**

- Use good lighting
- Vary hand positions slightly for better generalization
- Ensure the hand is clearly visible
- Collect data with different backgrounds

### Step 2: Extract Landmarks

After collecting data for all desired gestures:

```bash
python scripts/extract_landmarks.py
```

This will:

- Process all collected images
- Extract hand landmarks using MediaPipe
- Save features to `dataset/landmarks.csv`

### Step 3: Train the Model

```bash
python scripts/train_model.py
```

The model will:

- Load the landmark data
- Split into training and test sets
- Train a RandomForestClassifier
- Display accuracy metrics
- Save the trained model

**Expected Output:**

- Training accuracy: ~95-99%
- Test accuracy: ~90-95% (depending on data quality)

### Step 4: Test Real-time Recognition

```bash
python scripts/realtime_predict.py
```

Show your hand gestures to the webcam and see real-time predictions!

## 🎨 Customization

### Adding Custom Gestures (Words)

You can easily add custom gestures beyond A-Z and 0-9:

1. **Collect data with a custom label:**

   ```bash
   python scripts/collect_data.py
   # Enter label: HELLO
   ```

2. **Re-extract landmarks:**

   ```bash
   python scripts/extract_landmarks.py
   ```

3. **Retrain the model:**

   ```bash
   python scripts/train_model.py
   ```

4. **Test the new gesture:**
   ```bash
   python scripts/realtime_predict.py
   ```

### Modifying Model Parameters

To adjust the RandomForest model, edit `scripts/train_model.py`:

```python
model = RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42,
    n_jobs=-1
)
```

### Adjusting Detection Sensitivity

Modify confidence thresholds in `scripts/realtime_predict.py`:

```python
detector = HandDetector(
    min_detection_confidence=0.7,  # Hand detection threshold
    min_tracking_confidence=0.7    # Hand tracking threshold
)
```

## 🔧 Troubleshooting

### Common Issues

1. **Webcam not opening**
   - Check if another application is using the webcam
   - Try changing camera index in code: `cv2.VideoCapture(1)`

2. **Low accuracy**
   - Collect more training samples (200+ per gesture)
   - Ensure consistent hand positioning during data collection
   - Use good lighting and plain background
   - Use normalized landmarks (default)

3. **"No hand detected" error**
   - Ensure your hand is clearly visible
   - Adjust lighting conditions
   - Lower the detection confidence threshold

4. **Import errors**
   - Make sure all dependencies are installed
   - Activate the virtual environment
   - Try reinstalling requirements: `pip install -r requirements.txt --force-reinstall`

## 📊 Performance Tips

- **CPU Usage**: MediaPipe is optimized for CPU. For better performance on GPU, consider switching to TensorFlow-based solutions.
- **FPS**: Expected 20-30 FPS on average laptops
- **Lighting**: Good lighting significantly improves detection accuracy
- **Background**: Use a plain background for better hand detection

## 🔮 Future Enhancements

- [ ] Word and sentence recognition (sequence of gestures)
- [ ] Support for two-handed gestures
- [ ] Deep learning models (CNN + LSTM for sequences)
- [ ] Mobile app deployment
- [ ] Multi-language support
- [ ] Gesture recording and playback
- [ ] Integration with text-to-speech
- [ ] Integration with text-to-speech

## 📄 License

This project is created for educational purposes as part of a Final Year Project.

## 👨‍💻 Author

**Aryan**  
Final Year Project - Indian Sign Language Recognition System

## 🙏 Acknowledgments

- **Google MediaPipe** for excellent hand tracking
- **OpenCV** community for computer vision tools
- **Scikit-learn** for machine learning utilities

## 📞 Support

For issues or questions:

1. Check the troubleshooting section
2. Review the code comments for implementation details
3. Ensure all dependencies are correctly installed

---

**Happy Gesture Recognition! 🖐️✨**
