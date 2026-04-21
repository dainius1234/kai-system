from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from common.llm import _validate_llm_response, query_specialist
from common.prompt_templates import wake_intent_prompt
from common.runtime import sanitize_string, setup_json_logger

logger = setup_json_logger("perception-wake", os.getenv("LOG_PATH", "/tmp/perception-wake.json.log"))
app = FastAPI(title="Wake Intent Service", version="0.1.0")

WAKE_WORDS = [w.strip().lower() for w in os.getenv("WAKE_WORDS", "kai").split(",") if w.strip()]
if not WAKE_WORDS:
    WAKE_WORDS = ["kai"]
WAKE_COOLDOWN_SECONDS = float(os.getenv("WAKE_COOLDOWN_SECONDS", "2"))
WAKE_INTENT_MODEL = os.getenv("WAKE_INTENT_MODEL", "qwen2:0.5b")
WAKE_CONFIDENCE_THRESHOLD = float(os.getenv("WAKE_CONFIDENCE_THRESHOLD", "0.6"))
WAKE_WHISPER_MODEL = os.getenv("WAKE_WHISPER_MODEL", "tiny")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

_ALLOWED_INTENTS = {"chat", "task", "question", "command", "emotional", "unknown"}
_last_wake_ts: float = 0.0

_whisper_available = False
try:
    from faster_whisper import WhisperModel  # type: ignore

    _whisper_available = True
except Exception:
    WhisperModel = None  # type: ignore

_whisper_model = None


class WakeDetectRequest(BaseModel):
    text: Optional[str] = None
    audio_b64: Optional[str] = None

    @model_validator(mode="after")
    def _validate_payload(self):
        if not self.text and not self.audio_b64:
            raise ValueError("Either text or audio_b64 is required")
        return self


class WakeIntentRequest(BaseModel):
    text: str


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _compile_wake_pattern(wake_word: str) -> re.Pattern[str]:
    escaped = re.escape(wake_word).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def detect_wake_word(text: str) -> Tuple[bool, float, Optional[str]]:
    cleaned = _normalize(text)
    if not cleaned:
        return False, 0.0, None

    best_word: Optional[str] = None
    best_conf = 0.0
    for wake_word in WAKE_WORDS:
        pattern = _compile_wake_pattern(wake_word)
        if pattern.search(cleaned):
            conf = 0.95 if " " in wake_word else 0.85
            if conf > best_conf:
                best_word = wake_word
                best_conf = conf

    if not best_word:
        return False, 0.0, None

    global _last_wake_ts
    now = time.time()
    if now - _last_wake_ts < WAKE_COOLDOWN_SECONDS:
        return False, min(best_conf, 0.49), best_word

    _last_wake_ts = now
    return True, best_conf, best_word


def _heuristic_intent(text: str) -> Dict[str, Any]:
    t = _normalize(text)
    if re.search(r"\b(stop|status|sleep|shutdown|restart|pause|mute)\b", t):
        return {"intent": "command", "confidence": 0.78, "reasoning": "Matched operator command keywords"}
    if "?" in text or re.search(r"^(what|why|how|when|where|who|is|are|can|could|do|does)\b", t):
        return {"intent": "question", "confidence": 0.74, "reasoning": "Detected question form"}
    if re.search(r"\b(remind|schedule|set|create|open|run|send|book|write|add)\b", t):
        return {"intent": "task", "confidence": 0.72, "reasoning": "Detected actionable request keywords"}
    if re.search(r"\b(sad|anxious|stressed|overwhelmed|lonely|upset|vent|depressed|hurt)\b", t):
        return {"intent": "emotional", "confidence": 0.76, "reasoning": "Detected emotional support language"}
    if re.search(r"\b(hi|hello|hey|thanks|thank you|morning|evening)\b", t):
        return {"intent": "chat", "confidence": 0.66, "reasoning": "Detected general conversation cues"}
    return {"intent": "unknown", "confidence": 0.3, "reasoning": "No strong intent signal detected"}


def _validate_intent_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    intent = str(parsed.get("intent", "")).strip().lower()
    confidence = parsed.get("confidence")
    reasoning = str(parsed.get("reasoning", "")).strip()
    if intent not in _ALLOWED_INTENTS:
        return None
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        return None
    if not reasoning:
        return None
    return {"intent": intent, "confidence": round(float(confidence), 3), "reasoning": reasoning[:300]}


async def classify_intent(text: str) -> Dict[str, Any]:
    clean = sanitize_string(text[:1000])
    try:
        response = await query_specialist(
            "Ollama",
            wake_intent_prompt(clean),
            system=f"You are a strict JSON intent classifier. Prefer model: {WAKE_INTENT_MODEL}",
            temperature=0.0,
            max_tokens=120,
        )
        validated_text = _validate_llm_response(response.text)
        payload = _validate_intent_payload(validated_text)
        if payload is not None:
            return payload
    except Exception as exc:
        logger.warning("wake-intent llm classification failed: %s", exc)
    return _heuristic_intent(clean)


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None and _whisper_available and WhisperModel is not None:
        _whisper_model = WhisperModel(WAKE_WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper_model


def _transcribe_audio(audio_bytes: bytes) -> str:
    if not _whisper_available:
        return ""
    model = _get_whisper_model()
    if model is None:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, _ = model.transcribe(tmp.name, language="en", beam_size=1)
        return " ".join(seg.text for seg in segments).strip()


def _decode_audio_b64(audio_b64: str) -> bytes:
    encoded = audio_b64.strip()
    if encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio_b64 payload: {exc}") from exc


@app.post("/wake/detect")
async def wake_detect(req: WakeDetectRequest) -> Dict[str, Any]:
    start = time.monotonic()
    text = sanitize_string(req.text or "")
    if not text and req.audio_b64:
        audio_bytes = _decode_audio_b64(req.audio_b64)
        text = sanitize_string(_transcribe_audio(audio_bytes))

    detected, confidence, wake_word = detect_wake_word(text)
    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "detected": detected and confidence >= WAKE_CONFIDENCE_THRESHOLD,
        "confidence": round(confidence, 3),
        "wake_word": wake_word,
        "latency_ms": latency_ms,
    }


@app.post("/wake/intent")
async def wake_intent(req: WakeIntentRequest) -> Dict[str, Any]:
    return await classify_intent(req.text)


@app.post("/wake/process")
async def wake_process(req: WakeDetectRequest) -> Dict[str, Any]:
    wake = await wake_detect(req)
    text = sanitize_string(req.text or "")
    if wake.get("detected") and text:
        intent = await classify_intent(text)
    else:
        intent = {"intent": "unknown", "confidence": 0.0, "reasoning": "Wake word not detected"}
    return {"wake": wake, "intent": intent}


@app.get("/health")
async def health() -> Dict[str, Any]:
    llm_ok = True
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            llm_ok = resp.status_code < 500
    except Exception:
        llm_ok = False

    status = "ok" if llm_ok else "degraded"
    return {
        "status": status,
        "wake_words": WAKE_WORDS,
        "cooldown_seconds": WAKE_COOLDOWN_SECONDS,
        "wake_intent_model": WAKE_INTENT_MODEL,
        "wake_threshold": WAKE_CONFIDENCE_THRESHOLD,
        "whisper_available": _whisper_available,
        "llm_available": llm_ok,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8022")))
