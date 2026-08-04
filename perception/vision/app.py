"""Vision Service — webcam frame analysis for Kai.

Accepts JPEG/PNG blobs from the browser (getUserMedia → canvas → blob → POST).

Endpoints:
  POST /analyze/frame    → {faces, presence, face_count, dominant_emotion, emotions, backend}
  POST /analyze/presence → {present, confidence, face_count, backend}
  GET  /health
  GET  /metrics
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("vision-service", os.getenv("LOG_PATH", "/tmp/vision-service.json.log"))
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vision-service")

    class ErrorBudget:  # type: ignore[no-redef]
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}

try:
    import cv2
    import numpy as np
    _CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)
    _OPENCV_OK = True
    logger.info("OpenCV face detection ready")
except ImportError:
    _OPENCV_OK = False
    logger.info("OpenCV not available — vision in stub mode")
except Exception as exc:
    # A *partial* OpenCV is not the same as an absent one, and only the
    # second was handled. `opencv-python-headless` and several slim wheels
    # import fine and then have no `CascadeClassifier` or no `cv2.data`,
    # which raises AttributeError here — outside the guard, so the whole
    # service died at import rather than degrading. The message says
    # "stub mode"; the code could not deliver it.
    #
    # Found by CI, which installs such a build. It cannot be found on a
    # box where OpenCV is simply absent, because there the ImportError
    # branch is correct.
    _OPENCV_OK = False
    logger.warning(
        "OpenCV present but unusable (%s: %s) — vision in stub mode",
        type(exc).__name__, exc,
    )

try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
    logger.info("DeepFace emotion detection ready")
except (ImportError, Exception):
    _DEEPFACE_OK = False
    logger.info("DeepFace not available — emotion detection disabled")

PORT = int(os.getenv("PORT", "8023"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "30"))

app = FastAPI(title="vision-service")
budget = ErrorBudget(window_seconds=300)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    budget.record(response.status_code >= 500)
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "opencv": _OPENCV_OK, "deepface": _DEEPFACE_OK}


@app.get("/metrics")
async def metrics():
    return budget.snapshot()


def _decode_image(data: bytes) -> Optional[Any]:
    if not _OPENCV_OK:
        return None
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _detect_faces(img) -> List[Dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = _FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(detections) == 0:
        return []
    return [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in detections]


def _detect_emotion(img) -> Optional[Dict]:
    if not _DEEPFACE_OK:
        return None
    try:
        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0]
        return {
            "dominant": result.get("dominant_emotion", "neutral"),
            "scores": {k: round(float(v), 3) for k, v in result.get("emotion", {}).items()},
        }
    except Exception as exc:
        logger.debug("emotion analysis error: %s", exc)
        return None


@app.post("/analyze/frame")
async def analyze_frame(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty frame")

    if not _OPENCV_OK:
        return {"faces": [], "presence": False, "face_count": 0,
                "dominant_emotion": None, "emotions": None, "backend": "stub"}

    img = _decode_image(data)
    if img is None:
        raise HTTPException(400, "could not decode image — send JPEG or PNG")

    faces = _detect_faces(img)
    present = len(faces) > 0
    emotion = _detect_emotion(img) if present else None

    return {
        "faces": faces,
        "presence": present,
        "face_count": len(faces),
        "dominant_emotion": emotion["dominant"] if emotion else None,
        "emotions": emotion["scores"] if emotion else None,
        "backend": "opencv" + ("+deepface" if _DEEPFACE_OK else ""),
    }


@app.post("/analyze/presence")
async def analyze_presence(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty frame")

    if not _OPENCV_OK:
        return {"present": False, "confidence": 0.0, "face_count": 0, "backend": "stub"}

    img = _decode_image(data)
    if img is None:
        raise HTTPException(400, "could not decode image")

    faces = _detect_faces(img)
    confidence = min(1.0, len(faces) * 0.85) if faces else 0.0

    return {
        "present": len(faces) > 0,
        "confidence": round(confidence, 2),
        "face_count": len(faces),
        "backend": "opencv",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
