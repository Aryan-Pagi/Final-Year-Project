document.addEventListener("DOMContentLoaded", () => {
    
    // 1. Core Inference Controls
    document.getElementById("btn-start").addEventListener("click", () => fetch('/start', { method: 'POST' }));
    document.getElementById("btn-stop").addEventListener("click", () => fetch('/stop', { method: 'POST' }));
    document.getElementById("btn-clear").addEventListener("click", () => fetch('/clear', { method: 'POST' }));

    // 2. Status Polling Loop & Live Logs
    const statusIndicator = document.getElementById("status-indicator");
    const predictionText = document.getElementById("prediction-text");

    setInterval(() => {
        fetch('/status')
            .then(res => res.json())
            .then(data => {
                statusIndicator.innerText = data.status_msg;
                predictionText.value = data.prediction_text;
                statusIndicator.style.color = data.is_running ? "#38a169" : "#e53e3e";
                
                // Process Live Terminal Logs
                if (data.new_logs && data.new_logs.length > 0) {
                    data.new_logs.forEach(log => {
                        logToConsole(log, true); // True flags it as a Terminal/Backend log
                    });
                }
            })
            .catch(err => console.error("Polling error:", err));
    }, 500);

    // 3. Unified Dataset Info Fetcher
    async function loadDatasetInfo() {
        try {
            const res = await fetch('/api/dataset_info');
            const data = await res.json();
            const unifiedList = document.getElementById('unified-dataset-list');
            
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

    // 6. Formatted UI Logger
    const consoleOutput = document.getElementById('consoleOutput');
    function logToConsole(msg, isBackend = false) {
        const line = document.createElement('div');
        const time = new Date().toLocaleTimeString();
        
        // Color-code the source: Orange/Gray for Terminal prints, Blue for Frontend requests
        const source = isBackend ? '<span style="color:#a0aec0; font-weight:bold;">[Terminal]</span>' : '<span style="color:#3182ce; font-weight:bold;">[Frontend]</span>';
        
        // Safe escape to prevent raw data from breaking the HTML wrapper
        const safeMsg = msg.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        line.innerHTML = `<span style="color:#718096">[${time}]</span> ${source} ${safeMsg}`;
        consoleOutput.appendChild(line);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    document.getElementById('clearLogsBtn').addEventListener('click', () => {
        consoleOutput.innerHTML = '<div><span style="color:#718096">[System]</span> Logs cleared.</div>';
    });

    // 7. API Event Listeners
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
        logToConsole("Requesting landmark extraction. Monitor terminal stream below...");
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
        logToConsole(`Requesting model training (Test split: ${testSize})...`);
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ testSize })
        });
        const data = await res.json();
        logToConsole(data.message);
    });

    // 8. Settings & Model Switching
    document.getElementById('modelModeSelect').addEventListener('change', async (e) => {
        const mode = e.target.value;
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });
        const data = await res.json();
        logToConsole(data.message);
    });

    document.getElementById('reloadModelsBtn').addEventListener('click', async () => {
        logToConsole("Requesting hot-reload of models...");
        const res = await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ reload: true })
        });
        const data = await res.json();
        logToConsole(data.message);
    });
});