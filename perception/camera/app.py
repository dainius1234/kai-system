from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException


app = FastAPI(title="Camera Service", version="0.2.0")

CAMERA_DEVICE = os.getenv("CAMERA_DEVICE", "/dev/video0")
VIRTUAL_DEVICE = os.getenv("VIRTUAL_DEVICE", "/dev/video10")
CAPTURE_DIR = Path(os.getenv("CAPTURE_DIR", "/tmp/screen-captures"))
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Optional OpenCV — graceful degradation without it
_cv2_available = False
try:
    import cv2  # noqa: F401
    _cv2_available = True
except ImportError:
    pass

# Optional numpy — needed for image analysis
_numpy_available = False
try:
    import numpy as np  # noqa: F401
    _numpy_available = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════
# SCREEN / FRAME ANALYSIS
#  Captures frames from webcam or screen and analyses:
#    - Brightness level (mean pixel intensity)
#    - Motion detection (frame difference)
#    - Screen activity (text density via edge detection)
#  All offline, no cloud, CPU-only.
#
#  Source: simular-ai Agent-S multi-modal sensory patterns
# ═══════════════════════════════════════════════════════════════════════

_last_frame: Optional[Any] = None  # For motion detection
_analysis_history: List[Dict[str, Any]] = []
_MAX_HISTORY = 50


def _analyse_frame(frame) -> Dict[str, Any]:
    """Analyse a single frame for brightness, motion, and edge density."""
    global _last_frame
    result: Dict[str, Any] = {"timestamp": time.time()}

    if not _numpy_available:
        result["error"] = "numpy not available"
        return result

    # Convert to grayscale
    if _cv2_available and len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) == 2:
        gray = frame
    else:
        gray = frame[:, :, 0] if len(frame.shape) == 3 else frame

    # Brightness (mean pixel intensity 0-255)
    brightness = float(np.mean(gray))
    result["brightness"] = round(brightness, 1)
    if brightness < 50:
        result["brightness_label"] = "very_dark"
        result["nudge"] = "Screen is very dark — lights off? Consider taking a break"
    elif brightness < 100:
        result["brightness_label"] = "dim"
    elif brightness > 220:
        result["brightness_label"] = "very_bright"
    else:
        result["brightness_label"] = "normal"

    # Edge density (proxy for text/activity on screen)
    if _cv2_available:
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / max(edges.size, 1)
        result["edge_density"] = round(edge_ratio, 4)
        result["screen_activity"] = (
            "high" if edge_ratio > 0.15
            else "medium" if edge_ratio > 0.05
            else "low"
        )

    # Motion detection (frame difference)
    if _last_frame is not None and _last_frame.shape == gray.shape:
        diff = cv2.absdiff(_last_frame, gray) if _cv2_available else np.abs(
            gray.astype(np.int16) - _last_frame.astype(np.int16)
        ).astype(np.uint8)
        motion = float(np.mean(diff))
        result["motion_level"] = round(motion, 1)
        result["motion_detected"] = motion > 10.0
    else:
        result["motion_level"] = 0.0
        result["motion_detected"] = False

    _last_frame = gray.copy()

    # Dimensions
    result["height"] = gray.shape[0]
    result["width"] = gray.shape[1]

    return result


def _capture_camera_frame():
    """Capture a single frame from the camera device."""
    if not _cv2_available:
        raise HTTPException(
            status_code=503,
            detail="OpenCV not available — install opencv-python-headless",
        )
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=503,
            detail=f"Cannot open camera device: {CAMERA_DEVICE}",
        )
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame")
    return frame


def _capture_screen():
    """Capture the screen (if available) or return a dummy frame."""
    # Try mss (cross-platform screen capture) first
    try:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            img = sct.grab(monitor)
            if _numpy_available:
                frame = np.array(img)[:, :, :3]  # Drop alpha channel
                return frame
    except (ImportError, Exception):
        pass
    # Fallback: dummy frame for testing
    if _numpy_available:
        return np.zeros((480, 640), dtype=np.uint8)
    raise HTTPException(status_code=503, detail="No screen capture backend available")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "camera_device": CAMERA_DEVICE,
        "virtual_device": VIRTUAL_DEVICE,
        "opencv_available": str(_cv2_available),
        "numpy_available": str(_numpy_available),
    }


@app.post("/capture/camera")
async def capture_camera() -> Dict[str, Any]:
    """Capture and analyse a frame from the camera device."""
    frame = _capture_camera_frame()
    analysis = _analyse_frame(frame)
    _analysis_history.append(analysis)
    if len(_analysis_history) > _MAX_HISTORY:
        _analysis_history.pop(0)
    return {"status": "ok", "analysis": analysis}


@app.post("/capture/screen")
async def capture_screen() -> Dict[str, Any]:
    """Capture and analyse the current screen."""
    frame = _capture_screen()
    analysis = _analyse_frame(frame)
    analysis["source"] = "screen"
    _analysis_history.append(analysis)
    if len(_analysis_history) > _MAX_HISTORY:
        _analysis_history.pop(0)
    return {"status": "ok", "analysis": analysis}


@app.post("/process")
async def process_frame() -> Dict[str, Any]:
    """Legacy endpoint — captures screen and analyses."""
    frame = _capture_screen()
    analysis = _analyse_frame(frame)
    _analysis_history.append(analysis)
    if len(_analysis_history) > _MAX_HISTORY:
        _analysis_history.pop(0)
    return {"status": "ok", "analysis": analysis}


@app.get("/analysis/history")
async def analysis_history(limit: int = 10) -> Dict[str, Any]:
    """Return recent frame analysis history."""
    limit = min(max(limit, 1), _MAX_HISTORY)
    return {"count": len(_analysis_history), "history": _analysis_history[-limit:]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8020")))
