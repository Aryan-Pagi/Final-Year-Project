#!/usr/bin/env python3
"""
ISL Web Dashboard - Verification Script
Checks all components are in place and working
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description:50} {path if not exists else ''}")
    return exists

def check_import(module_name, description):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description:50} Available")
        return True
    except ImportError:
        print(f"✗ {description:50} NOT INSTALLED")
        return False

def main():
    print("\n" + "="*80)
    print(" "*20 + "ISL WEB DASHBOARD - VERIFICATION")
    print("="*80 + "\n")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check Flask app files
    print("📁 Flask Application Files:")
    print("-" * 80)
    all_files_ok = True
    all_files_ok &= check_file(os.path.join(base_dir, 'app.py'), 'Flask backend (app.py)')
    all_files_ok &= check_file(os.path.join(base_dir, 'templates', 'index.html'), 'HTML template')
    all_files_ok &= check_file(os.path.join(base_dir, 'static', 'styles.css'), 'CSS stylesheet')
    all_files_ok &= check_file(os.path.join(base_dir, 'static', 'main.js'), 'JavaScript code')
    
    print()
    print("📋 Documentation Files:")
    print("-" * 80)
    all_files_ok &= check_file(os.path.join(base_dir, 'WEB_DASHBOARD_GUIDE.md'), 'User guide')
    all_files_ok &= check_file(os.path.join(base_dir, 'launch_dashboard.bat'), 'Windows launcher')
    all_files_ok &= check_file(os.path.join(base_dir, 'launch_dashboard.ps1'), 'PowerShell launcher')

    print()
    print("🐍 Python Dependencies:")
    print("-" * 80)
    all_deps_ok = True
    all_deps_ok &= check_import('flask', 'Flask web framework')
    all_deps_ok &= check_import('flask_cors', 'Flask CORS support')
    all_deps_ok &= check_import('cv2', 'OpenCV (for capture)')
    all_deps_ok &= check_import('mediapipe', 'MediaPipe (for landmarks)')
    all_deps_ok &= check_import('tensorflow', 'TensorFlow (for BiLSTM)')
    all_deps_ok &= check_import('sklearn', 'Scikit-learn')

    print()
    print("📂 Data Directories:")
    print("-" * 80)
    dataset_path = os.path.join(base_dir, 'dataset', 'raw_images')
    models_path = os.path.join(base_dir, 'models')
    
    dataset_ok = os.path.exists(dataset_path)
    models_ok = os.path.exists(models_path)
    
    status = "✓" if dataset_ok else "✗"
    print(f"{status} Dataset directory:              {dataset_path}")
    status = "✓" if models_ok else "✗"
    print(f"{status} Models directory:               {models_path}")

    print()
    print("="*80)
    print("VERIFICATION RESULTS:")
    print("="*80)
    
    if all_files_ok and all_deps_ok and dataset_ok and models_ok:
        print("\n✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        print("\nYou can now launch the dashboard:")
        print("  Windows:  launch_dashboard.bat")
        print("  Windows:  launch_dashboard.ps1")
        print("  Other:    python app.py")
        print("\nThen open: http://localhost:5000")
        return 0
    else:
        print("\n⚠ Some checks failed!")
        if not all_files_ok:
            print("  - Missing Flask application files")
        if not all_deps_ok:
            print("  - Missing Python dependencies (run: pip install -r requirements.txt)")
        if not dataset_ok or not models_ok:
            print("  - Missing data directories (will be created on first use)")
        return 1

if __name__ == '__main__':
    sys.exit(main())
