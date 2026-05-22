# ISL Gesture Recognition - Web Dashboard Launcher (PowerShell)
# Run this script to start the Flask dashboard

param(
    [switch]$port = $false
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "     ISL GESTURE RECOGNITION - WEB DASHBOARD LAUNCHER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
$pythonCheck = & {
    try {
        python --version 2>&1
        return $true
    } catch {
        return $false
    }
}

if (-not $pythonCheck) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "app.py")) {
    Write-Host "ERROR: app.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the Final-Year-Project directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check for virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host ""
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Check if Flask is installed
$flaskCheck = & {
    try {
        python -c "import flask" 2>&1
        return $true
    } catch {
        return $false
    }
}

if (-not $flaskCheck) {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Packages installed successfully!" -ForegroundColor Green
    Write-Host ""
}

# Launch Flask app
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Starting Flask Web Server..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard URL: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

python app.py

Read-Host "Press Enter to exit"
