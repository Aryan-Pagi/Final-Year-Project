# ISL Gesture Recognition System

A real-time **Indian Sign Language (ISL)** gesture recognition system using computer vision and machine learning. This system recognizes hand gestures with a digit-first strategy using **shared MediaPipe landmark extraction** but **separate lightweight classifiers** for static vs. dynamic gestures to maximize accuracy without overcomplication.

## 🎯 Features

- **Real-time hand gesture recognition** using webcam
- **MediaPipe-based hand landmark detection** for accurate tracking
- **Dual classifier architecture**:
  - **Static classifier** (sklearn RandomForest) for digits 0-9, letters A-Z with single-frame landmark recognition
  - **Dynamic classifier** (compact BiLSTM) for motion-based word gestures (HELLO, THANK YOU, etc.)
  - **User-defined dynamic words** - flexible system to add any custom gesture word with video clips
- **Shared preprocessing pipeline** - consistent landmark normalization and aspect-aware padding
- **Digit-first training strategy** - stable 0-9 base set before extending to A-Z or custom words
- **Class balance prioritization** - balanced samples and clean capture conditions over raw volume
- **User-friendly menu interface** and web dashboard for all operations
- **Normalized landmark features** for better generalization
- **Confidence scoring** and prediction smoothing
- **FPS monitoring** for performance tracking

## 🛠️ Technologies Used

- **Python 3.9+**
- **OpenCV** - Computer vision and webcam handling
- **MediaPipe** - Hand landmark detection (21 landmarks per hand, 93 engineered features)
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning (RandomForestClassifier for static digits)
- **TensorFlow/Keras** - Deep learning (compact BiLSTM for dynamic gestures)

## 📁 Project Structure

```
Final-Year-Project/
│
├── dataset/
│   ├── raw_images/          # Collected static gesture images (digits 0-9, letters A-Z)
│   │   ├── 0/, 1/, ..., 9/  # Digit samples (static, single-frame)
│   │   ├── A/, B/, ..., Z/  # Letter samples (static, single-frame) [Phase 2]
│   │   └── [custom labels]/ # Any custom gesture labels
│   ├── raw_clips/           # Dynamic word gestures (HELLO, THANK YOU, SORRY, etc.)
│   │   ├── HELLO/
│   │   │   ├── clip_0/frame_0.jpg, frame_1.jpg, ...
│   │   │   ├── clip_1/frame_0.jpg, frame_1.jpg, ...
│   │   │   └── ... (50+ clips recommended per word)
│   │   ├── THANK_YOU/
│   │   │   └── [similar structure]
│   │   └── [user-defined words]/  # Add any custom dynamic gestures
│   ├── sequences.npz        # Extracted landmark sequences for BiLSTM
│   └── landmarks.csv        # Extracted landmark features (unified format)
│
├── models/
│   ├── bilstm_model.keras   # Trained BiLSTM for dynamic gestures
│   ├── static_classifier.pkl  # Phase 1: RandomForest for digits 0-9 only
│   ├── static_classifier_full.pkl  # Phase 2: RandomForest for digits 0-9 + letters A-Z
│   └── gesture_model.pkl    # Model metadata and routing info
│
├── scripts/
│   ├── __init__.py
│   ├── collect_data.py      # Data collection (static images & video clips)
│   ├── extract_landmarks.py # Landmark extraction (shared pipeline)
│   ├── train_model.py       # Model training (split static/dynamic)
│   └── realtime_predict.py  # Real-time gesture prediction (routing + inference)
│
├── utils/
│   ├── __init__.py
│   ├── mediapipe_utils.py   # MediaPipe & preprocessing helpers
│   └── word_builder.py      # Word/sentence assembly from predictions
│
├── static/ & templates/     # Web dashboard (Flask UI)
├── app.py                   # Flask web server
├── main.py                  # CLI menu interface
├── requirements.txt         # Python dependencies
├── launch_dashboard.bat     # Windows launcher
├── launch_dashboard.ps1     # PowerShell launcher
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

### 🎯 Phase 1: Digit Recognition (0-9) — START HERE

This is the **recommended starting point**. Build a stable digit classifier before extending to letters or custom words.

#### Step 1: Collect Static Digit Images (0-9)

Collect clean, centered static poses for each digit:

```bash
python scripts/collect_data.py
```

**For each digit (0-9):**

- Enter label: **0** (then 1, 2, ..., 9)
- Choose mode: **1** (Static images)
- Enter samples: **100** (aim for balanced counts)
- Show your hand making the digit gesture (centered, front-facing)
- Press **SPACE** to capture each image
- Repeat until 100 samples collected

**Tips for consistent capture:**

- Use **consistent lighting** (well-lit, no harsh shadows)
- Keep hand **centered** in frame and at similar distance
- Use **plain, neutral background** (desk or wall)
- Maintain **similar hand orientation** across samples
- Avoid extreme angles or partial hand views
- Take breaks to minimize repetitive strain
- Verify samples visually before/after collection

**Quality over quantity:** 50 clean, well-lit images beats 200 blurry ones.

#### Step 2: Extract Landmarks

After collecting all 10 digits, extract the shared landmark features:

```bash
python scripts/extract_landmarks.py
```

This will:

- Process all images in `dataset/raw_images/0/` through `dataset/raw_images/9/`
- Extract 21 hand landmarks per image (63 raw coordinates)
- Compute 93 engineered features (coordinates + distances + angles + finger extensions)
- Save unified feature vectors for training
- Report any failed detections (shows if hand wasn't properly captured)

**Output:** You'll see a progress report and success rate. Aim for >90% detection rate.

#### Step 3: Train the Static Digit Classifier

```bash
python scripts/train_model.py
```

The trainer will:

- Load extracted digit landmarks
- Check class balance and warn if imbalanced
- Split into training (80%) and test (20%) sets with stratification
- Train a lightweight RandomForest classifier on the 93 engineered features
- Report test accuracy and confusion matrix
- Save `models/static_classifier.pkl` and metadata

**Expected results:**
- Test accuracy: **90–98%** (depending on data quality and balance)
- Training typically completes in seconds

**Quality checks:**
- If any digit shows <50% accuracy or high confusion, re-collect cleaner samples for that digit
- If overall accuracy is <85%, review capture conditions (lighting, centering, background)

#### Step 4: Test Real-time Recognition

```bash
python scripts/realtime_predict.py
```

Show your hand gestures for digits 0-9 to the webcam and verify real-time predictions!

- Display shows predicted digit and confidence
- Press **q** to quit
- Press **f** to toggle FPS display

---

### � Phase 2: Letters A-Z + Dynamic Words (After Digit Base is Stable)

Once digits 0-9 are working reliably (>90% accuracy), expand to the full alphabet and custom dynamic gestures.

#### Step 5: Collect A-Z Letter Images

Use the same collection process as digits, but for letters:

```bash
python scripts/collect_data.py
```

**For each letter (A-Z):**
- Enter label: **A** (then B, C, ... Z)
- Choose mode: **1** (Static images)
- Enter samples: **100** (aim for consistent counts with digits)
- Collect with same quality standards as Phase 1 digits

**Tips:**
- Match the **same lighting, background, and distance** as your digit data
- Keep hand position **consistent** to the Phase 1 digit images
- Aim for **uniform number of samples per letter** (e.g., 100 each like digits)

#### Step 6: Train Combined Static Classifier (0-9 + A-Z)

```bash
python scripts/train_model.py
```

Select **Training Mode: 1** (Static Classifier)

The trainer will:
- Combine all digits (0-9) with all letters (A-Z) into a single 36-class classifier
- Save merged model as `models/static_classifier_full.pkl`
- Report per-class accuracy for all 36 labels

**Expected results:**
- Combined accuracy: **85-95%** (slightly lower than digits alone due to more classes, but still solid)
- If any letter shows <70% accuracy, recollect samples for that letter

#### Step 7: Add Dynamic Word Gestures (Optional)

For motion-based gestures like HELLO, THANK YOU, etc., collect video clips:

```bash
python scripts/collect_data.py
```

**For each custom word:**
- Enter label: **HELLO** (or any custom word)
- Choose mode: **2** (Video clips)
- Enter num_clips: **50** (or more for stability)
- Enter frames_per_clip: **30** (or adjust for gesture speed)
- Perform the gesture 50 times, each capture ~1 second

**Dynamic gesture tips:**
- Perform **consistent motion** (same speed, same trajectory)
- Start from **neutral/resting position** each time
- Avoid extreme angles; keep hands **centered and visible**
- Record in **same lighting** as static data
- Aim for **balanced clips across variations** (left/right hand, different speeds)

#### Step 8: Train BiLSTM for Dynamic Gestures

```bash
python scripts/train_model.py
```

Select **Training Mode: 2** (Dynamic/BiLSTM Classifier)

The trainer will:
- Auto-detect all video clips in `dataset/raw_clips/`
- Extract 30-frame sequences from each clip
- Train a compact BiLSTM on temporal patterns
- Save model as `models/bilstm_model.keras`
- Include all custom words automatically (no manual enum needed)

**Expected results:**
- Accuracy varies by gesture complexity; aim for >80%
- Dynamic gestures typically need more data than static (50+ clips per word recommended)

---

### 🎨 Advanced Customization

#### After Phase 2 — Adding New Custom Gestures

The system supports unlimited custom static or dynamic gestures:

1. **New static gesture** (e.g., custom letter variant or symbol):
   - Create `dataset/raw_images/{LABEL}/` and collect ~100 images
   - Re-run extraction and re-train the static classifier

2. **New dynamic gesture** (e.g., HELLO, SORRY, HELP, etc.):
   - Create `dataset/raw_clips/{WORD}/clip_0/, clip_1/, ...` with video frames
   - Re-run training (BiLSTM auto-detects and includes)
   - Flexible: add as many custom words as desired

3. **Retrain existing models** after adding new data:
   - Run `extract_landmarks.py` to refresh features
   - Run `train_model.py` to retrain with expanded classes

---

### Modifying Model Parameters

#### Static Classifier (RandomForest)

Edit `scripts/train_model.py` in the static training section:

```python
model = RandomForestClassifier(
    n_estimators=200,      # Number of trees
    max_depth=None,        # Max depth (None = unlimited)
    min_samples_split=2,   # Minimum samples to split node
    min_samples_leaf=1,    # Minimum samples per leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
```

**For Phase 2 (36 classes: 0-9, A-Z):**
- Increase `n_estimators=300` for better accuracy with more classes
- Adjust `min_samples_split=3` to reduce overfitting with expanded data

#### Dynamic Classifier (BiLSTM)

For video gestures, the BiLSTM uses:

```python
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2)),
    Dropout(0.3),
    Bidirectional(LSTM(32, dropout=0.2)),
    Dropout(0.3),
    Dense(num_classes, activation='softmax'),
])
```

#### Detection Sensitivity

Modify confidence thresholds in `scripts/collect_data.py` and `realtime_predict.py`:

```python
detector = HandDetector(
    min_detection_confidence=0.7,  # Hand detection threshold (0.1-1.0)
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
