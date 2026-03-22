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


# ═══════════════════════════════════════════════════════════════════════
# J4: PROACTIVE LOW-LATENCY VOICE
#  Combines audio (emotion, energy) + video (brightness, motion) signals
#  to decide whether Kai should speak unprompted.
#
#  speak_or_not gate:
#    - Input: audio emotion scores + video frame analysis
#    - Output: should_speak (bool) + reason + suggested_message
#    - Cooldown: won't re-trigger within PROACTIVE_COOLDOWN_SECONDS
#
#  All offline, no cloud, CPU-only.
# ═══════════════════════════════════════════════════════════════════════

import re  # noqa: E402

PROACTIVE_COOLDOWN = int(os.getenv("PROACTIVE_COOLDOWN_SECONDS", "120"))
TTS_URL = os.getenv("TTS_URL", "http://tts:8022")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
_last_proactive_ts: float = 0.0


def _speak_or_not(
    audio_signals: Dict[str, Any],
    video_signals: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether Kai should speak based on combined sensory signals.

    Returns dict with: should_speak, reason, urgency (0-1), suggested_message.
    """
    global _last_proactive_ts
    now = time.time()

    # Cooldown check
    if now - _last_proactive_ts < PROACTIVE_COOLDOWN:
        return {
            "should_speak": False,
            "reason": "cooldown_active",
            "urgency": 0.0,
            "suggested_message": None,
            "cooldown_remaining": round(PROACTIVE_COOLDOWN - (now - _last_proactive_ts)),
        }

    urgency = 0.0
    reasons = []
    messages = []

    # Audio signals
    emotion_scores = audio_signals.get("scores", {})
    dominant = audio_signals.get("dominant", "neutral")
    dominant_score = audio_signals.get("dominant_score", 0.0)
    rms = audio_signals.get("rms_level")

    if dominant == "stress" and dominant_score >= 0.4:
        urgency += 0.4
        reasons.append("high_stress_detected")
        messages.append("You sound stressed. Want to take a step back for a minute?")
    if dominant == "fatigue" and dominant_score >= 0.3:
        urgency += 0.3
        reasons.append("fatigue_detected")
        messages.append("Energy seems low — maybe time for a break?")
    if rms is not None and rms < 300:
        urgency += 0.1
        reasons.append("very_low_voice_energy")
    if rms is not None and rms > 6000:
        urgency += 0.2
        reasons.append("shouting_detected")
        messages.append("Things sound heated. Need to talk it through?")

    # Video signals
    brightness = video_signals.get("brightness", 128)
    motion = video_signals.get("motion_level", 0)
    motion_detected = video_signals.get("motion_detected", False)

    if brightness < 30:
        urgency += 0.2
        reasons.append("very_dark_environment")
        messages.append("It's quite dark — everything okay?")
    if not motion_detected and motion < 2.0:
        # User hasn't moved in a while (based on last frame)
        urgency += 0.1
        reasons.append("no_motion_detected")
    if motion > 50:
        urgency += 0.15
        reasons.append("excessive_motion")
        messages.append("Lots of movement — need anything?")

    # Decision gate
    should_speak = urgency >= 0.3
    suggested = messages[0] if messages else None

    if should_speak:
        _last_proactive_ts = now

    return {
        "should_speak": should_speak,
        "reason": "|".join(reasons) if reasons else "normal",
        "urgency": round(min(urgency, 1.0), 2),
        "suggested_message": suggested,
        "audio_dominant": dominant,
        "video_brightness": brightness,
    }


# ── P2: Multi-modal LLM fusion ──────────────────────────────────────
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8000")


async def interpret_multi(
    audio_signals: Dict[str, Any],
    video_signals: Dict[str, Any],
) -> Dict[str, Any]:
    """Send combined audio + video signals to LLM for richer interpretation.

    Falls back to the heuristic _speak_or_not() if the LLM is unavailable.
    """
    heuristic = _speak_or_not(audio_signals, video_signals)
    summary = (
        f"Audio: dominant={audio_signals.get('dominant', 'unknown')}, "
        f"score={audio_signals.get('dominant_score', 0)}, "
        f"rms={audio_signals.get('rms_level', 'n/a')}. "
        f"Video: brightness={video_signals.get('brightness', 'n/a')}, "
        f"motion={video_signals.get('motion_level', 0)}."
    )
    try:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{LANGGRAPH_URL}/run",
                json={"message": f"[SENSOR FUSION] {summary} Should I speak? Why?"},
            )
            if resp.status_code == 200:
                data = resp.json()
                llm_text = data.get("response", "")
                return {
                    **heuristic,
                    "llm_interpretation": llm_text[:500],
                    "fusion_mode": "llm",
                }
    except Exception:
        pass
    return {**heuristic, "llm_interpretation": None, "fusion_mode": "heuristic"}


class ProactiveRequest:
    """Simple container for proactive voice request data."""
    pass


@app.post("/proactive/evaluate")
async def proactive_evaluate(
    audio_signals: Dict[str, Any] = {},
    video_signals: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Evaluate whether Kai should speak proactively based on sensor fusion.

    Accepts audio emotion signals and video frame analysis signals.
    Returns speak_or_not decision with urgency score.
    """
    decision = _speak_or_not(audio_signals, video_signals)
    return {"status": "ok", **decision}


@app.post("/proactive/interpret")
async def proactive_interpret(
    audio_signals: Dict[str, Any] = {},
    video_signals: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """LLM-enhanced multi-modal fusion interpretation."""
    result = await interpret_multi(audio_signals, video_signals)
    return {"status": "ok", **result}


@app.post("/proactive/auto")
async def proactive_auto() -> Dict[str, Any]:
    """Auto-capture from both sensors and decide whether to speak.

    Captures fresh screen frame + uses the latest audio emotion from
    the analysis history, then runs the speak-or-not gate.
    """
    # Get latest video signal
    try:
        frame = _capture_screen()
        video_signals = _analyse_frame(frame)
    except Exception:
        video_signals = {"brightness": 128, "motion_level": 0, "motion_detected": False}

    # Get latest audio signal from audio service
    audio_signals: Dict[str, Any] = {"scores": {}, "dominant": "neutral", "dominant_score": 0.0}
    try:
        import httpx
        audio_url = os.getenv("AUDIO_URL", "http://perception-audio:8021")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{audio_url}/transcripts", params={"limit": 1})
            if resp.status_code == 200:
                data = resp.json()
                transcripts = data.get("transcripts", [])
                if transcripts and "emotion" in transcripts[-1]:
                    audio_signals = transcripts[-1]["emotion"]
    except Exception:
        pass  # Use defaults

    decision = _speak_or_not(audio_signals, video_signals)

    # If should speak, optionally trigger TTS
    if decision["should_speak"] and decision.get("suggested_message"):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(f"{TTS_URL}/synthesize", json={
                    "text": decision["suggested_message"],
                    "voice": "kai-default",
                })
        except Exception:
            pass  # TTS is best-effort

    return {"status": "ok", **decision}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8020")))
