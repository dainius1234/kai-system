"""Audio service — captures audio, transcribes via Whisper, and memorizes.

Core pipeline:
  1. Capture audio from microphone (sounddevice) or accept uploaded WAV
  2. Detect hotword ("ara") to trigger active listening
  3. Transcribe via Whisper (faster-whisper when GPU available, stub for now)
  4. Injection detection — block prompt injection patterns
  5. Auto-memorize transcripts to memu-core

Designed to work without GPU: capture and injection detection work on CPU.
Transcription uses a stub until RTX 5080 arrives, then flip WHISPER_BACKEND=local.
"""
from __future__ import annotations

import io
import os
import re
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("perception-audio", os.getenv("LOG_PATH", "/tmp/perception-audio.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Audio Service", version="0.5.0")
HOTWORD = os.getenv("PORCUPINE_KEYWORD", "ara")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "local")  # "stub", "local", "api"
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "/tmp/audio-captures"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
RECORD_SECONDS = int(os.getenv("AUDIO_RECORD_SECONDS", "10"))

INJECTION_RE = re.compile(
    r"\b(ignore\s+(all\s+)?previous|system\s+prompt|override\s+instructions"
    r"|you\s+are\s+now|act\s+as\s+if|disregard\s+(all|previous))\b",
    re.IGNORECASE,
)
INJECTION_LOG = Path(os.getenv("INJECTION_LOG", "/tmp/injection_events.log"))
budget = ErrorBudget(window_seconds=300)

# optional dependencies
_sounddevice_available = False
_whisper_available = False

try:
    import sounddevice  # noqa: F401
    _sounddevice_available = True
except ImportError:
    logger.info("sounddevice not available — mic capture disabled, file upload only")

try:
    from faster_whisper import WhisperModel  # noqa: F401
    _whisper_available = True
except ImportError:
    logger.info("faster-whisper not available — transcription in stub mode")

# Cached Whisper model — loaded once on first transcription request
_whisper_model = None


def _get_whisper_model():
    """Lazy-load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None and _whisper_available:
        logger.info(
            "Loading Whisper model '%s' (first-time download may be slow)...",
            WHISPER_MODEL,
        )
        _whisper_model = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8",
        )
        logger.info("Whisper model '%s' loaded successfully.", WHISPER_MODEL)
    return _whisper_model


# transcript buffer for recent captures (ring buffer, max 50)
_transcript_buffer: List[Dict[str, Any]] = []
_MAX_BUFFER = 50


class TranscriptRequest(BaseModel):
    text: str
    session_id: str = "unknown"


class AudioCaptureResult(BaseModel):
    status: str
    transcript: str
    source: str
    duration_seconds: float
    timestamp: float
    whisper_backend: str
    injection_detected: bool


def _check_injection(text: str, session_id: str = "unknown") -> bool:
    """Return True if prompt injection detected."""
    if INJECTION_RE.search(text):
        INJECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(INJECTION_LOG, "a", encoding="utf-8") as f:
            f.write(f"blocked session={sanitize_string(session_id)} text={sanitize_string(text[:200])}\n")
        logger.warning("prompt injection blocked in audio transcript")
        return True
    return False


def _transcribe_audio(audio_bytes: bytes, ext: str = ".wav") -> str:
    """Transcribe audio bytes via configured backend."""
    if WHISPER_BACKEND == "local" and _whisper_available:
        model = _get_whisper_model()
        if model is None:
            return "[transcript: Whisper model failed to load]"
        # write to temp file (faster-whisper needs a file path)
        tmp = AUDIO_DIR / f"_tmp_{int(time.time())}{ext}"
        try:
            tmp.write_bytes(audio_bytes)
            segments, info = model.transcribe(
                str(tmp), language="en", beam_size=1,
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.info(
                "transcribed %d bytes → %d chars (lang=%s prob=%.2f)",
                len(audio_bytes), len(text),
                info.language, info.language_probability,
            )
            return text if text else "[transcript: no speech detected]"
        except Exception as e:
            logger.error("Whisper transcription error: %s", e)
            return f"[transcript: error — {str(e)[:100]}]"
        finally:
            tmp.unlink(missing_ok=True)

    if WHISPER_BACKEND == "api":
        logger.info("Whisper API backend not yet implemented")
        return "[transcript: Whisper API backend pending]"

    # stub mode
    return "[transcript: stub mode — set WHISPER_BACKEND=local for real STT]"


def _record_microphone(seconds: int = RECORD_SECONDS) -> bytes:
    """Record from microphone and return WAV bytes."""
    if not _sounddevice_available:
        raise HTTPException(status_code=503, detail="sounddevice not available — cannot capture from mic")

    import sounddevice as sd
    import numpy as np

    logger.info("recording %d seconds from microphone at %d Hz", seconds, SAMPLE_RATE)
    audio_data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    # convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())
    return buf.getvalue()


async def _auto_memorize(transcript: str, source: str) -> None:
    """Send transcript to memu-core for memorization."""
    if not transcript.strip() or transcript.startswith("[transcript:"):
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{MEMU_URL}/memory/memorize", json={
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_type": "audio_transcript",
                "result_raw": sanitize_string(transcript[:2000]),
                "user_id": "perception-audio",
                "category": "daily-logs",
            })
    except Exception as e:
        logger.warning("auto-memorize failed: %s", e)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "hotword": HOTWORD,
        "whisper_model": WHISPER_MODEL,
        "whisper_backend": WHISPER_BACKEND,
        "device": DEVICE,
        "mic_available": str(_sounddevice_available),
        "whisper_available": str(_whisper_available),
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return {
        **budget.snapshot(),
        "transcript_buffer_size": len(_transcript_buffer),
    }


@app.post("/listen")
async def listen(request: TranscriptRequest) -> Dict[str, str]:
    """Accept a pre-transcribed text (for testing or external transcription)."""
    clean = sanitize_string(request.text)
    if _check_injection(clean, request.session_id):
        raise HTTPException(status_code=400, detail="prompt injection pattern blocked")

    # buffer it
    entry = {"text": clean, "session_id": request.session_id, "ts": time.time(), "source": "manual"}
    _transcript_buffer.append(entry)
    if len(_transcript_buffer) > _MAX_BUFFER:
        _transcript_buffer.pop(0)

    # auto-memorize if enabled
    if os.getenv("AUTO_MEMORIZE_AUDIO", "true").lower() == "true":
        await _auto_memorize(clean, "listen")

    return {"status": "ok", "message": "accepted", "text": clean}


@app.post("/capture/mic", response_model=AudioCaptureResult)
async def capture_mic(seconds: int = RECORD_SECONDS) -> AudioCaptureResult:
    """Record from microphone, transcribe, and return result."""
    seconds = min(max(seconds, 1), 60)  # clamp 1-60s
    audio_bytes = _record_microphone(seconds)

    # save to disk
    ts = int(time.time())
    out_path = AUDIO_DIR / f"mic_{ts}.wav"
    out_path.write_bytes(audio_bytes)

    transcript = _transcribe_audio(audio_bytes, ext=".wav")
    injection = _check_injection(transcript)

    if not injection and os.getenv("AUTO_MEMORIZE_AUDIO", "true").lower() == "true":
        await _auto_memorize(transcript, f"mic:{out_path.name}")

    _transcript_buffer.append({"text": transcript, "source": "mic", "ts": time.time()})
    if len(_transcript_buffer) > _MAX_BUFFER:
        _transcript_buffer.pop(0)

    return AudioCaptureResult(
        status="ok" if not injection else "injection_blocked",
        transcript=sanitize_string(transcript[:5000]),
        source=f"mic:{out_path.name}",
        duration_seconds=float(seconds),
        timestamp=time.time(),
        whisper_backend=WHISPER_BACKEND,
        injection_detected=injection,
    )


@app.post("/capture/file", response_model=AudioCaptureResult)
async def capture_file(file: UploadFile = File(...)) -> AudioCaptureResult:
    """Upload an audio file, transcribe it, and return result."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    audio_bytes = await file.read()
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    # save to disk
    ts = int(time.time())
    out_path = AUDIO_DIR / f"upload_{ts}_{sanitize_string(file.filename)}"
    out_path.write_bytes(audio_bytes)

    # detect format from filename extension
    _ext = ".wav"
    if file.filename:
        _parts = file.filename.rsplit(".", 1)
        if len(_parts) > 1:
            _ext = f".{_parts[1].lower()}"

    transcript = _transcribe_audio(audio_bytes, ext=_ext)
    injection = _check_injection(transcript)

    if not injection and os.getenv("AUTO_MEMORIZE_AUDIO", "true").lower() == "true":
        await _auto_memorize(transcript, f"file:{file.filename}")

    return AudioCaptureResult(
        status="ok" if not injection else "injection_blocked",
        transcript=sanitize_string(transcript[:5000]),
        source=f"file:{file.filename}",
        duration_seconds=0.0,  # unknown from file
        timestamp=time.time(),
        whisper_backend=WHISPER_BACKEND,
        injection_detected=injection,
    )


@app.get("/transcripts")
async def get_transcripts(limit: int = 10) -> Dict[str, Any]:
    """Return recent transcripts from the ring buffer."""
    limit = min(max(limit, 1), _MAX_BUFFER)
    return {
        "count": len(_transcript_buffer),
        "transcripts": _transcript_buffer[-limit:],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8021")))
