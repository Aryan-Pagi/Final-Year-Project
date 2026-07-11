document.addEventListener("DOMContentLoaded", () => {
    
    // 1. Core Inference Controls
    document.getElementById("btn-start").addEventListener("click", () => fetch('/start', { method: 'POST' }));
    document.getElementById("btn-stop").addEventListener("click", () => fetch('/stop', { method: 'POST' }));
    document.getElementById("btn-clear").addEventListener("click", () => fetch('/clear', { method: 'POST' }));

    // 2. Status Polling Loop
    const statusIndicator = document.getElementById("status-indicator");
    const predictionText = document.getElementById("prediction-text");

    setInterval(() => {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                statusIndicator.innerText = data.status_msg;
                predictionText.value = data.prediction_text;
                statusIndicator.style.color = data.is_running ? "#38a169" : "#e53e3e";
            })
            .catch(err => console.error("Polling error:", err));
    }, 500);

    // 3. Unified Dataset Info Fetcher
    async function loadDatasetInfo() {
        try {
            const res = await fetch('/api/dataset_info');
            const data = await res.json();
            const unifiedList = document.getElementById('unified-dataset-list');
            
            // Combine both static and dynamic lists into a single consolidated view
            const combined = [...data.static, ...data.dynamic];
            
            unifiedList.innerHTML = combined.length ? 
                combined.map(g => `<li>${g}</li>`).join('') : '<li class="empty">No collected gesture folders found</li>';
        } catch(e) {
            console.error("Failed to load dataset info");
        }
    }

    // 4. Admin Drawer Toggles
    const adminDrawer = document.getElementById('adminView');
    const adminBackdrop = document.getElementById('adminBackdrop');
    
    document.getElementById('toggleAdminBtn').addEventListener('click', () => {
        adminDrawer.classList.add('open');
        adminBackdrop.classList.add('open');
        loadDatasetInfo();
    });

    const closeAdmin = () => {
        adminDrawer.classList.remove('open');
        adminBackdrop.classList.remove('open');
    };
    document.getElementById('closeAdminBtn').addEventListener('click', closeAdmin);
    adminBackdrop.addEventListener('click', closeAdmin);

    // 5. Tab Switching Logic
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            button.classList.add('active');
            document.getElementById('tab-' + button.dataset.tab).classList.add('active');
        });
    });

    // 6. Admin API Calls & Logging
    const consoleOutput = document.getElementById('consoleOutput');
    function logToConsole(msg) {
        const line = document.createElement('div');
        line.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    document.getElementById('clearLogsBtn').addEventListener('click', () => {
        consoleOutput.innerHTML = '<div>[System] Logs cleared.</div>';
    });

    document.getElementById('startCaptureBtn').addEventListener('click', async () => {
        const gestureName = document.getElementById('gestureName').value;
        const captureType = document.querySelector('input[name="captureType"]:checked').value;
        const numSamples = document.getElementById('numSamples').value;
        
        logToConsole(`Initializing collection for '${gestureName}'...`);
        const res = await fetch('/api/collect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gestureName, captureType, numSamples })
        });
        const data = await res.json();
        logToConsole(data.message);
        
        if (data.status === 'success') loadDatasetInfo();
    });

    document.getElementById('extractLandmarksBtn').addEventListener('click', async () => {
        const useNormalized = document.getElementById('useNormalizedCheckbox').checked;
        logToConsole("Extracting landmarks... This may take a moment.");
        const res = await fetch('/api/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ useNormalized })
        });
        const data = await res.json();
        logToConsole(data.message);
    });

    document.getElementById('trainModelBtn').addEventListener('click', async () => {
        const testSize = document.getElementById('testSize').value;
        logToConsole(`Training model (Test split: ${testSize})...`);
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ testSize })
        });
        const data = await res.json();
        logToConsole(data.message);
    });
});