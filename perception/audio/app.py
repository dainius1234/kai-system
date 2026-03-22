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
    emotion: Optional[Dict[str, Any]] = None


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


# ═══════════════════════════════════════════════════════════════════════
# VOICE EMOTION ANALYSIS
#  Analyses transcript text for emotional tone indicators using keyword
#  heuristics.  No ML model needed — fast, offline, low-CPU.
#  When audio energy analysis is available (numpy), also checks RMS level.
#
#  Signals: stress/frustration, calm, excitement, fatigue, uncertainty
#  Action: "voice low, take break" nudge type
#
#  Source: simular-ai Agent-S multi-modal sensory patterns
# ═══════════════════════════════════════════════════════════════════════

_EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "stress": ["frustrated", "annoyed", "angry", "damn", "ugh",
               "terrible", "stressed", "overwhelm", "deadline", "panic"],
    "fatigue": ["tired", "exhausted", "sleepy", "yawn", "drained",
                "can't focus", "need break", "so tired", "burned out"],
    "excitement": ["amazing", "awesome", "brilliant", "fantastic",
                   "great news", "love it", "excited", "perfect"],
    "uncertainty": ["maybe", "not sure", "i think", "hmm", "i guess",
                    "don't know", "confused", "unclear", "doubt"],
    "calm": ["okay", "fine", "good", "alright", "no worries",
             "sounds good", "sure", "understood"],
}

_EMOTION_NUDGES: Dict[str, str] = {
    "stress": "Voice sounds tense — consider a 5-minute break",
    "fatigue": "Energy seems low — maybe take a break or stretch",
    "uncertainty": "Sounds uncertain — want to break this problem down?",
}


def _analyse_voice_emotion(
    transcript: str,
    audio_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Analyse voice emotion from transcript keywords and optional audio energy.

    Returns emotion scores (0.0-1.0 per emotion) and a nudge suggestion
    if stress/fatigue is detected.
    """
    text_lower = transcript.lower()
    scores: Dict[str, float] = {}

    for emotion, keywords in _EMOTION_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        scores[emotion] = min(hits / max(len(keywords) * 0.3, 1.0), 1.0)

    # Audio energy analysis (RMS) if numpy and raw audio available
    rms_level = None
    if audio_bytes:
        try:
            import numpy as np
            # Parse WAV: skip 44-byte header, read int16 samples
            if len(audio_bytes) > 44:
                samples = np.frombuffer(audio_bytes[44:], dtype=np.int16)
                if len(samples) > 0:
                    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
                    rms_level = round(rms, 1)
                    # Low RMS (<500) may indicate low energy / whisper
                    if rms < 500:
                        scores["fatigue"] = min(scores.get("fatigue", 0) + 0.3, 1.0)
                    # High RMS (>5000) may indicate shouting / stress
                    elif rms > 5000:
                        scores["stress"] = min(scores.get("stress", 0) + 0.3, 1.0)
        except (ImportError, Exception):
            pass  # numpy not required

    # Determine dominant emotion
    dominant = max(scores, key=scores.get) if scores else "neutral"
    dominant_score = scores.get(dominant, 0.0)

    # Generate nudge if stress/fatigue high
    nudge = None
    if dominant_score >= 0.3 and dominant in _EMOTION_NUDGES:
        nudge = _EMOTION_NUDGES[dominant]

    return {
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "dominant": dominant,
        "dominant_score": round(dominant_score, 2),
        "rms_level": rms_level,
        "nudge": nudge,
    }


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
    emotion = _analyse_voice_emotion(transcript, audio_bytes)

    if not injection and os.getenv("AUTO_MEMORIZE_AUDIO", "true").lower() == "true":
        await _auto_memorize(transcript, f"mic:{out_path.name}")

    _transcript_buffer.append({"text": transcript, "source": "mic", "ts": time.time(), "emotion": emotion})
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
        emotion=emotion,
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


@app.post("/analyse/emotion")
async def analyse_emotion(request: TranscriptRequest) -> Dict[str, Any]:
    """Analyse voice emotion from a transcript text."""
    clean = sanitize_string(request.text)
    emotion = _analyse_voice_emotion(clean)
    return {"status": "ok", **emotion}


# ═══════════════════════════════════════════════════════════════════════
# J2: WAKE-WORD "KAI" + INTENT JUDGE
#  Keyword-spot for "Kai" in whisper transcript stream.
#  Tiny LLM (qwen2:0.5b) classifies intent: command | mention | echo.
#  Wired into nudge engine via memu-core /memory/proactive.
# ═══════════════════════════════════════════════════════════════════════

WAKE_WORD = os.getenv("WAKE_WORD", "kai").lower()
WAKE_WORD_RE = re.compile(
    r"\b" + re.escape(WAKE_WORD) + r"\b",
    re.IGNORECASE,
)

# Intent classification categories
_INTENT_COMMAND = "command"     # "Kai, what's the weather?"
_INTENT_MENTION = "mention"    # "I told Kai about it"
_INTENT_ECHO = "echo"          # "...kai..." in background noise
_INTENT_UNKNOWN = "unknown"

_INTENT_PROMPT = (
    "Classify the user's intent toward the AI assistant named Kai.\n"
    "Given this transcript, determine if the speaker is:\n"
    "- 'command': directly asking Kai to do something or answering Kai\n"
    "- 'mention': talking about Kai to someone else\n"
    "- 'echo': Kai's name appears incidentally or in background noise\n\n"
    "Respond with ONLY one word: command, mention, or echo.\n\n"
    "Transcript: {text}"
)

# Recent wake-word detections (ring buffer)
_wake_detections: List[Dict[str, Any]] = []
_MAX_WAKE_BUFFER = 100


def detect_wake_word(transcript: str) -> bool:
    """Return True if wake word "Kai" found in transcript."""
    return bool(WAKE_WORD_RE.search(transcript))


async def classify_intent(transcript: str) -> str:
    """Use tiny LLM to classify whether this is a command, mention, or echo."""
    try:
        from common.llm import query_specialist
        prompt = _INTENT_PROMPT.format(text=sanitize_string(transcript[:500]))
        resp = await query_specialist(
            "Ollama", prompt,
            system="You are an intent classifier. Respond with exactly one word.",
            temperature=0.1,
            max_tokens=10,
        )
        raw = resp.text.strip().lower()
        # Extract the intent from the response
        for intent in (_INTENT_COMMAND, _INTENT_MENTION, _INTENT_ECHO):
            if intent in raw:
                return intent
        return _INTENT_UNKNOWN
    except Exception as e:
        logger.warning("intent classification failed: %s", e)
        # Fallback: heuristic — if "Kai" is at start or after comma, likely command
        text_lower = transcript.lower().strip()
        if text_lower.startswith(WAKE_WORD) or f", {WAKE_WORD}" in text_lower:
            return _INTENT_COMMAND
        return _INTENT_UNKNOWN


async def _send_wake_nudge(transcript: str, intent: str) -> None:
    """Forward wake-word activation to memu-core proactive nudge engine."""
    if intent not in (_INTENT_COMMAND,):
        return  # only nudge for direct commands
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{MEMU_URL}/memory/memorize", json={
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event_type": "wake_word_activation",
                "result_raw": sanitize_string(transcript[:500]),
                "user_id": "perception-audio",
                "category": "daily-logs",
            })
    except Exception as e:
        logger.warning("wake-word nudge failed: %s", e)


@app.post("/wake-word/detect")
async def wake_word_detect(request: TranscriptRequest) -> Dict[str, Any]:
    """Detect wake-word and classify intent in a transcript."""
    clean = sanitize_string(request.text)
    detected = detect_wake_word(clean)

    result: Dict[str, Any] = {
        "wake_word": WAKE_WORD,
        "detected": detected,
        "intent": None,
        "transcript": clean[:200],
    }

    if detected:
        intent = await classify_intent(clean)
        result["intent"] = intent

        entry = {
            "text": clean[:200],
            "intent": intent,
            "ts": time.time(),
            "session_id": request.session_id,
        }
        _wake_detections.append(entry)
        if len(_wake_detections) > _MAX_WAKE_BUFFER:
            _wake_detections.pop(0)

        # Forward command intents to nudge engine
        await _send_wake_nudge(clean, intent)

    return result


@app.get("/wake-word/history")
async def wake_word_history(limit: int = 20) -> Dict[str, Any]:
    """Return recent wake-word detections."""
    limit = min(max(limit, 1), _MAX_WAKE_BUFFER)
    return {
        "wake_word": WAKE_WORD,
        "count": len(_wake_detections),
        "detections": _wake_detections[-limit:],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8021")))
