"""
Gradio Backend Wrapper
Stage 1
"""

from threading import Thread
import threading
import time
import webbrowser

from scripts.realtime_predict import predict_realtime

_backend_thread = None
_running = False
_lock = threading.Lock()


def _run():
    global _running

    try:
        predict_realtime()
    finally:
        with _lock:
            _running = False


def start_backend():

    global _backend_thread
    global _running

    with _lock:

        if _running:
            return "🟢 Online"

        _running = True

        _backend_thread = Thread(
            target=_run,
            daemon=True
        )

        _backend_thread.start()

    return "🟢 Online"


def stop_backend():

    """
    Stage 1 placeholder.

    Real shutdown will be added
    after realtime_predict.py
    becomes frame-based.
    """

    return "🔴 Offline"


def backend_status():

    return "🟢 Online" if _running else "🔴 Offline"


def clear_prediction():

    return "Waiting...", "0%"


def open_browser():

    time.sleep(1)

    webbrowser.open(
        "http://127.0.0.1:5000"
    )