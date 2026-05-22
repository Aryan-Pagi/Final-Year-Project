/**
 * ISL Gesture Recognition Dashboard - Interactive JavaScript
 * Handles all UI interactions, API calls, and real-time updates
 */

// ─── CONSTANTS ────────────────────────────────────────────────────────────
const API_BASE = '/api';
const STATUS_POLL_INTERVAL = 1000; // Poll every 1 second
const LOGS_POLL_INTERVAL = 500;    // Poll logs every 500ms

// Status colors
const STATUS_COLORS = {
    'IDLE': '#0088ff',
    'CAPTURING': '#ffcc00',
    'EXTRACTING': '#ff6600',
    'TRAINING': '#ff6600',
    'RECOGNIZING': '#00cc00'
};

const STATUS_LIGHT_CLASSES = {
    'IDLE': '',
    'CAPTURING': 'capturing',
    'EXTRACTING': 'capturing',
    'TRAINING': 'training',
    'RECOGNIZING': 'recognizing'
};

// ─── DOM ELEMENTS ────────────────────────────────────────────────────────
const elements = {
    // Status indicators
    statusLight: document.getElementById('statusLight'),
    statusText: document.getElementById('statusText'),
    cameraCaption: document.getElementById('cameraCaption'),
    recognizingIndicator: document.getElementById('recognizingIndicator'),
    cameraStream: document.getElementById('cameraStream'),
    cameraPlaceholder: document.querySelector('.camera-placeholder'),

    // View containers
    userView: document.getElementById('userView'),
    adminView: document.getElementById('adminView'),
    adminBackdrop: document.getElementById('adminBackdrop'),
    toggleAdminBtn: document.getElementById('toggleAdminBtn'),
    toggleConsoleBtn: document.getElementById('toggleConsoleBtn'),
    toggleSettingsBtn: document.getElementById('toggleSettingsBtn'),
    closeAdminBtn: document.getElementById('closeAdminBtn'),

    // Translation display
    translationOutput: document.getElementById('translationOutput'),
    clearTranslationBtn: document.getElementById('clearTranslationBtn'),
    muteBtn: document.getElementById('muteBtn'),

    // Data collection
    gestureName: document.getElementById('gestureName'),
    captureType: document.getElementsByName('captureType'),
    numSamples: document.getElementById('numSamples'),
    numClips: document.getElementById('numClips'),
    clipFrames: document.getElementById('clipFrames'),
    startCaptureBtn: document.getElementById('startCaptureBtn'),
    captureProgress: document.getElementById('captureProgress'),
    captureProgressBar: document.getElementById('captureProgressBar'),
    captureProgressText: document.getElementById('captureProgressText'),

    // Gesture library
    gestureList: document.getElementById('gestureList'),
    refreshGesturesBtn: document.getElementById('refreshGesturesBtn'),
    useNormalizedCheckbox: document.getElementById('useNormalizedCheckbox'),
    extractLandmarksBtn: document.getElementById('extractLandmarksBtn'),
    extractProgress: document.getElementById('extractProgress'),
    extractProgressBar: document.getElementById('extractProgressBar'),
    extractProgressText: document.getElementById('extractProgressText'),

    // Training
    testSize: document.getElementById('testSize'),
    testSizeValue: document.getElementById('testSizeValue'),
    trainModelBtn: document.getElementById('trainModelBtn'),
    trainingProgress: document.getElementById('trainingProgress'),
    trainingProgressBar: document.getElementById('trainingProgressBar'),
    trainingProgressText: document.getElementById('trainingProgressText'),
    trainingDetails: document.getElementById('trainingDetails'),

    // Prediction
    predictionMode: document.getElementById('predictionMode'),
    confidenceThreshold: document.getElementById('confidenceThreshold'),
    thresholdValue: document.getElementById('thresholdValue'),
    startStreamBtn: document.getElementById('startStreamBtn'),
    stopStreamBtn: document.getElementById('stopStreamBtn'),

    // Console
    consoleOutput: document.getElementById('consoleOutput'),
    clearLogsBtn: document.getElementById('clearLogsBtn'),
    autoScrollCheckbox: document.getElementById('autoScrollCheckbox'),

    // Tabs
    tabButtons: document.querySelectorAll('.tab-button'),
    tabContents: document.querySelectorAll('.tab-content')
};

// ─── STATE MANAGEMENT ────────────────────────────────────────────────────
let appState = {
    systemState: 'IDLE',
    isStreaming: false,
    statusPollTimer: null,
    logsPollTimer: null,
    lastTranslation: ''
};

// ─── INITIALIZATION ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized');
    initializeEventListeners();
    if (elements.stopStreamBtn) {
        elements.stopStreamBtn.disabled = true;
    }
    startStatusPolling();
    startLogsPolling();
    loadGestures();
});

// ─── EVENT LISTENERS ────────────────────────────────────────────────────
function initializeEventListeners() {
    // Admin drawer controls
    if (elements.toggleAdminBtn) {
        elements.toggleAdminBtn.addEventListener('click', () => openAdminPanel());
    }
    if (elements.closeAdminBtn) {
        elements.closeAdminBtn.addEventListener('click', closeAdminPanel);
    }
    if (elements.adminBackdrop) {
        elements.adminBackdrop.addEventListener('click', closeAdminPanel);
    }
    if (elements.toggleConsoleBtn) {
        elements.toggleConsoleBtn.addEventListener('click', () => openAdminPanel('console'));
    }
    if (elements.toggleSettingsBtn) {
        elements.toggleSettingsBtn.addEventListener('click', () => openAdminPanel('training'));
    }

    // Tab navigation
    elements.tabButtons.forEach(btn => {
        btn.addEventListener('click', (e) => handleTabClick(e));
    });

    // Data collection
    if (elements.startCaptureBtn) {
        elements.startCaptureBtn.addEventListener('click', startCapture);
    }
    if (elements.captureType) {
        elements.captureType.forEach(radio => {
            radio.addEventListener('change', handleCaptureTypeChange);
        });
    }

    // Gesture library
    if (elements.refreshGesturesBtn) {
        elements.refreshGesturesBtn.addEventListener('click', loadGestures);
    }
    if (elements.extractLandmarksBtn) {
        elements.extractLandmarksBtn.addEventListener('click', extractLandmarks);
    }

    // Training
    if (elements.testSize) {
        elements.testSize.addEventListener('input', (e) => {
            const percent = Math.round(parseFloat(e.target.value) * 100);
            elements.testSizeValue.textContent = `${e.target.value} (${percent}%)`;
        });
    }
    if (elements.trainModelBtn) {
        elements.trainModelBtn.addEventListener('click', trainModel);
    }

    // Prediction
    if (elements.confidenceThreshold) {
        elements.confidenceThreshold.addEventListener('input', (e) => {
            elements.thresholdValue.textContent = e.target.value;
        });
    }
    if (elements.startStreamBtn) {
        elements.startStreamBtn.addEventListener('click', startStream);
    }
    if (elements.stopStreamBtn) {
        elements.stopStreamBtn.addEventListener('click', stopStream);
    }
    if (elements.clearTranslationBtn) {
        elements.clearTranslationBtn.addEventListener('click', clearTranslation);
    }

    // Console
    if (elements.clearLogsBtn) {
        elements.clearLogsBtn.addEventListener('click', clearLogs);
    }
}

// ─── ADMIN DRAWER MANAGEMENT ───────────────────────────────────────────
function openAdminPanel(tabName = null) {
    document.body.classList.add('admin-open');
    if (tabName) {
        setActiveTab(tabName);
    }
}

function closeAdminPanel() {
    document.body.classList.remove('admin-open');
}

// ─── TAB MANAGEMENT ────────────────────────────────────────────────────
function handleTabClick(e) {
    const tabName = e.target.dataset.tab;
    setActiveTab(tabName);
}

function setActiveTab(tabName) {
    if (!tabName) return;
    elements.tabButtons.forEach(btn => btn.classList.remove('active'));
    elements.tabContents.forEach(content => content.classList.remove('active'));

    const targetButton = Array.from(elements.tabButtons).find(
        (btn) => btn.dataset.tab === tabName
    );
    const targetContent = document.getElementById(`tab-${tabName}`);

    if (targetButton) targetButton.classList.add('active');
    if (targetContent) targetContent.classList.add('active');
}

// ─── CAPTURE TYPE HANDLING ────────────────────────────────────────────────
function handleCaptureTypeChange() {
    const isVideo = document.querySelector('input[name="captureType"]:checked').value === 'video';

    if (elements.numSamples && elements.numSamples.parentElement) {
        elements.numSamples.parentElement.classList.toggle('hidden', isVideo);
    }
    if (elements.numClips && elements.numClips.parentElement) {
        elements.numClips.parentElement.classList.toggle('hidden', !isVideo);
    }
    if (elements.clipFrames && elements.clipFrames.parentElement) {
        elements.clipFrames.parentElement.classList.toggle('hidden', !isVideo);
    }
}

// ─── DATA COLLECTION ────────────────────────────────────────────────────
async function startCapture() {
    const gestureName = elements.gestureName.value.trim().toUpperCase();
    if (!gestureName) {
        alert('Please enter a gesture name');
        return;
    }

    const isVideo = document.querySelector('input[name="captureType"]:checked').value === 'video';
    const numSamples = parseInt(elements.numSamples.value) || 50;
    const numClips = parseInt(elements.numClips.value) || 50;
    const clipFrames = parseInt(elements.clipFrames.value) || 30;

    try {
        showProgress(elements.captureProgress, 'Initializing capture...');

        const response = await fetch(`${API_BASE}/capture/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gesture_name: gestureName,
                num_samples: numSamples,
                use_video: isVideo,
                num_clips: numClips,
                clip_frames: clipFrames
            })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${data.error}`);
            hideProgress(elements.captureProgress);
            return;
        }

        visualFeedback(elements.startCaptureBtn, 'Capture started!');
        elements.gestureName.value = '';
        addLog(`✓ Started capturing data for '${gestureName}'`, 'success');

    } catch (error) {
        console.error('Error starting capture:', error);
        alert('Failed to start capture: ' + error.message);
        hideProgress(elements.captureProgress);
    }
}

// ─── GESTURE MANAGEMENT ────────────────────────────────────────────────
async function loadGestures() {
    try {
        const response = await fetch(`${API_BASE}/gestures`);
        const data = await response.json();

        elements.gestureList.innerHTML = '';

        if (!data.gestures || data.gestures.length === 0) {
            elements.gestureList.innerHTML = '<p class="empty-state">No gestures found. Start by capturing some data.</p>';
            return;
        }

        data.gestures.forEach(gesture => {
            const card = document.createElement('div');
            card.className = 'gesture-card';
            card.innerHTML = `
                <div class="gesture-info">
                    <div class="gesture-name">${gesture.name}</div>
                    <div class="gesture-samples">${gesture.samples} samples</div>
                </div>
                <button class="gesture-delete" title="Delete '${gesture.name}'" data-gesture="${gesture.name}">
                    🗑️ Delete
                </button>
            `;
            
            card.querySelector('.gesture-delete').addEventListener('click', (e) => {
                deleteGesture(gesture.name);
            });

            elements.gestureList.appendChild(card);
        });

    } catch (error) {
        console.error('Error loading gestures:', error);
        addLog('Error loading gestures: ' + error.message, 'error');
    }
}

async function deleteGesture(gestureName) {
    if (!confirm(`Delete gesture '${gestureName}' and all its data? This cannot be undone.`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/gesture/${gestureName}`, {
            method: 'DELETE'
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${data.error}`);
            return;
        }

        visualFeedback(elements.refreshGesturesBtn, 'Gesture deleted');
        addLog(`✓ Deleted gesture '${gestureName}'`, 'success');
        loadGestures();

    } catch (error) {
        console.error('Error deleting gesture:', error);
        alert('Failed to delete gesture: ' + error.message);
    }
}

// ─── LANDMARK EXTRACTION ────────────────────────────────────────────────
async function extractLandmarks() {
    const useNormalized = elements.useNormalizedCheckbox.checked;

    try {
        showProgress(elements.extractProgress, 'Starting landmark extraction...');

        const response = await fetch(`${API_BASE}/landmarks/extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ use_normalized: useNormalized })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${data.error}`);
            hideProgress(elements.extractProgress);
            return;
        }

        visualFeedback(elements.extractLandmarksBtn, 'Extraction started!');
        addLog('✓ Started landmark extraction', 'success');

    } catch (error) {
        console.error('Error extracting landmarks:', error);
        alert('Failed to start extraction: ' + error.message);
        hideProgress(elements.extractProgress);
    }
}

// ─── MODEL TRAINING ────────────────────────────────────────────────────
async function trainModel() {
    const testSize = parseFloat(elements.testSize.value) || 0.2;

    try {
        showProgress(elements.trainingProgress, 'Starting model training...');

        const response = await fetch(`${API_BASE}/model/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ test_size: testSize })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${data.error}`);
            hideProgress(elements.trainingProgress);
            return;
        }

        visualFeedback(elements.trainModelBtn, 'Training started!');
        addLog('✓ Started model training. This may take several minutes...', 'success');

    } catch (error) {
        console.error('Error training model:', error);
        alert('Failed to start training: ' + error.message);
        hideProgress(elements.trainingProgress);
    }
}

// ─── REAL-TIME PREDICTION ────────────────────────────────────────────────
async function startStream() {
    const mode = elements.predictionMode ? (elements.predictionMode.value || 'letter') : 'letter';
    const confidenceThreshold = elements.confidenceThreshold
        ? (parseFloat(elements.confidenceThreshold.value) || 0.7)
        : 0.7;

    try {
        const response = await fetch(`${API_BASE}/stream/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: mode,
                confidence_threshold: confidenceThreshold
            })
        });

        const data = await response.json();

        if (!response.ok) {
            alert(`Error: ${data.error}`);
            return;
        }

        appState.isStreaming = true;
        visualFeedback(elements.startStreamBtn, 'Stream started!');
        addLog(`✓ Started ${mode} recognition mode`, 'success');

        // Disable start button, enable stop button
        if (elements.startStreamBtn) elements.startStreamBtn.disabled = true;
        if (elements.stopStreamBtn) elements.stopStreamBtn.disabled = false;

        if (elements.cameraStream) {
            elements.cameraStream.src = `/video_feed?ts=${Date.now()}`;
            elements.cameraStream.classList.remove('hidden');
        }
        if (elements.cameraPlaceholder) {
            elements.cameraPlaceholder.classList.add('hidden');
        }

    } catch (error) {
        console.error('Error starting stream:', error);
        alert('Failed to start stream: ' + error.message);
    }
}

async function stopStream() {
    try {
        const response = await fetch(`${API_BASE}/stream/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        appState.isStreaming = false;
        visualFeedback(elements.stopStreamBtn, 'Stream stopped');
        addLog('✓ Stopped real-time prediction', 'success');

        // Enable start button, disable stop button
        if (elements.startStreamBtn) elements.startStreamBtn.disabled = false;
        if (elements.stopStreamBtn) elements.stopStreamBtn.disabled = true;

        if (elements.cameraStream) {
            elements.cameraStream.src = '';
            elements.cameraStream.classList.add('hidden');
        }
        if (elements.cameraPlaceholder) {
            elements.cameraPlaceholder.classList.remove('hidden');
        }

    } catch (error) {
        console.error('Error stopping stream:', error);
        addLog('Error stopping stream: ' + error.message, 'error');
    }
}

// ─── STATUS POLLING ────────────────────────────────────────────────────
function startStatusPolling() {
    appState.statusPollTimer = setInterval(updateStatus, STATUS_POLL_INTERVAL);
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();

        // Update system state
        appState.systemState = data.state;
        updateStatusIndicator(data.state);
        updateProgressBars(data);

        // Persist and display the latest recognized text whenever it changes
        const recognizedText = data.recognition_text || data.last_recognized_gesture;
        if (recognizedText) {
            appState.lastTranslation = recognizedText;
            updateTranslation(recognizedText, data.last_confidence);
        } else if (appState.lastTranslation && elements.translationOutput) {
            updateTranslation(appState.lastTranslation, data.last_confidence);
        }

    } catch (error) {
        console.error('Error polling status:', error);
    }
}

function updateStatusIndicator(state) {
    if (elements.statusText) {
        const label = state === 'RECOGNIZING' ? 'ACTIVE' : state;
        elements.statusText.textContent = label;
    }

    if (elements.statusLight) {
        elements.statusLight.style.backgroundColor = STATUS_COLORS[state] || STATUS_COLORS['IDLE'];
    }

    if (elements.recognizingIndicator) {
        elements.recognizingIndicator.classList.toggle('active', state === 'RECOGNIZING');
    }

    if (elements.cameraCaption) {
        elements.cameraCaption.textContent = state === 'RECOGNIZING' ? 'RECOGNIZING' : 'READY';
    }
}

function updateProgressBars(data) {
    const progress = data.progress || 0;
    
    // Update capture progress
    if (data.state === 'CAPTURING') {
        updateProgressBar(elements.captureProgressBar, progress);
        elements.captureProgressText.textContent = data.message || 'Capturing...';
    } else if (elements.captureProgress.classList.contains('hidden') === false) {
        if (progress === 100) {
            hideProgress(elements.captureProgress);
        }
    }

    // Update extraction progress
    if (data.state === 'EXTRACTING') {
        updateProgressBar(elements.extractProgressBar, progress);
        elements.extractProgressText.textContent = data.message || 'Extracting...';
    } else if (elements.extractProgress.classList.contains('hidden') === false) {
        if (progress === 100) {
            hideProgress(elements.extractProgress);
        }
    }

    // Update training progress
    if (data.state === 'TRAINING') {
        updateProgressBar(elements.trainingProgressBar, progress);
        elements.trainingProgressText.textContent = data.message || 'Training...';
    } else if (elements.trainingProgress.classList.contains('hidden') === false) {
        if (progress === 100) {
            hideProgress(elements.trainingProgress);
        }
    }
}

function updateTranslation(text, confidence = null) {
    if (!elements.translationOutput) return;
    if (!text || text.length === 0) return;

    const confidenceText = confidence !== null && confidence !== undefined
        ? `<span class="translation-confidence">${Math.round(confidence * 100)}%</span>`
        : '';

    elements.translationOutput.innerHTML = `
        <div class="translation-value">${text}</div>
        ${confidenceText}
    `;
}

function clearTranslation() {
    if (!elements.translationOutput) return;
    appState.lastTranslation = '';
    elements.translationOutput.innerHTML = '<span class="idle-text">Waiting for hand gesture...</span>';
    if (elements.cameraCaption) {
        elements.cameraCaption.textContent = 'READY';
    }
}

// ─── LOGS POLLING ────────────────────────────────────────────────────────
function startLogsPolling() {
    appState.logsPollTimer = setInterval(updateLogs, LOGS_POLL_INTERVAL);
}

async function updateLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs`);
        const data = await response.json();

        if (!elements.consoleOutput) return;
        data.logs.forEach(log => {
            // Avoid duplicates
            const exists = Array.from(elements.consoleOutput.children).some(child => 
                child.textContent === log
            );
            if (!exists) {
                addLog(log);
            }
        });

    } catch (error) {
        // Silent fail - logs are optional
    }
}

function addLog(message, type = 'info') {
    if (!elements.consoleOutput) return;
    const line = document.createElement('div');
    line.className = `console-line${type ? ' ' + type : ''}`;
    line.textContent = message;
    
    elements.consoleOutput.appendChild(line);

    // Auto scroll if enabled
    if (elements.autoScrollCheckbox.checked) {
        elements.consoleOutput.scrollTop = elements.consoleOutput.scrollHeight;
    }

    // Limit console size
    while (elements.consoleOutput.children.length > 500) {
        elements.consoleOutput.removeChild(elements.consoleOutput.firstChild);
    }
}

function clearLogs() {
    elements.consoleOutput.innerHTML = '<div class="console-line">[System] Console cleared</div>';
}

// ─── PROGRESS BAR UTILITIES ────────────────────────────────────────────
function updateProgressBar(progressBarElement, percentage) {
    progressBarElement.style.setProperty('--progress', `${Math.min(percentage, 100)}%`);
    progressBarElement.style.width = `${Math.min(percentage, 100)}%`;
}

function showProgress(progressContainer, message) {
    progressContainer.classList.remove('hidden');
    const progressText = progressContainer.querySelector('.progress-text');
    if (progressText) {
        progressText.textContent = message;
    }
}

function hideProgress(progressContainer) {
    progressContainer.classList.add('hidden');
}

// ─── VISUAL FEEDBACK ──────────────────────────────────────────────────
function visualFeedback(element, message = '') {
    // Temporarily highlight the element
    const originalBg = window.getComputedStyle(element).backgroundColor;
    const originalTransform = window.getComputedStyle(element).transform;
    
    element.style.transform = 'scale(0.95)';
    element.style.boxShadow = '0 0 20px rgba(0, 102, 204, 0.5)';
    
    setTimeout(() => {
        element.style.transform = originalTransform;
        element.style.boxShadow = '';
    }, 200);

    // Show message
    if (message) {
        const originalText = element.textContent;
        element.textContent = message;
        setTimeout(() => {
            element.textContent = originalText;
        }, 1500);
    }
}

// ─── UTILITY FUNCTIONS ────────────────────────────────────────────────
function formatTime(date) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
}

// ─── CLEANUP ──────────────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
    if (appState.statusPollTimer) clearInterval(appState.statusPollTimer);
    if (appState.logsPollTimer) clearInterval(appState.logsPollTimer);
});
