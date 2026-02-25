"""TTS Service — Kai's voice.

Backends:
  - edge-tts (default): Microsoft Edge TTS. Great quality, many voices, async.
    Requires internet. Will be replaced by Piper for air-gapped local mode.
  - piper (planned): Offline TTS via onnxruntime + voice models. Zero internet.

Endpoints:
  GET  /health      — service status + config
  GET  /voices      — list available voice presets
  POST /synthesize  — text → audio/mpeg stream (real audio bytes)
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.runtime import detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger(
    "tts-service",
    os.getenv("LOG_PATH", "/tmp/tts-service.json.log"),
)
DEVICE = detect_device()

app = FastAPI(title="TTS Service — Kai's Voice", version="1.0.0")

# ── config ──────────────────────────────────────────────────────────
TTS_BACKEND = os.getenv("TTS_BACKEND", "edge")  # "edge" | "piper" (future)
DEFAULT_VOICE = os.getenv("TTS_VOICE", "en-GB-RyanNeural")
DEFAULT_RATE = os.getenv("TTS_RATE", "+0%")
DEFAULT_VOLUME = os.getenv("TTS_VOLUME", "+0%")

# Curated voice presets — British-first for Kai
VOICE_MAP: Dict[str, str] = {
    "kai-default": "en-GB-RyanNeural",        # British male (Kai's main voice)
    "kai-warm": "en-GB-ThomasNeural",         # Warm British narrator
    "kai-female": "en-GB-SoniaNeural",        # British female
    "kai-us": "en-US-GuyNeural",              # American male
    "kai-us-female": "en-US-JennyNeural",     # American female
}

_edge_available = False
try:
    import edge_tts  # noqa: F401
    _edge_available = True
    logger.info("edge-tts loaded — voice synthesis enabled")
except ImportError:
    logger.warning("edge-tts not installed — TTS disabled")

_synth_count = 0
_synth_chars = 0


# ── models ──────────────────────────────────────────────────────────
class SynthesisRequest(BaseModel):
    text: str
    voice: str = "kai-default"
    rate: str = "+0%"
    volume: str = "+0%"


# ── endpoints ───────────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _edge_available else "degraded",
        "backend": TTS_BACKEND,
        "default_voice": DEFAULT_VOICE,
        "edge_tts_available": _edge_available,
        "synth_count": _synth_count,
        "synth_chars": _synth_chars,
        "device": DEVICE,
    }


@app.get("/voices")
async def list_voices() -> Dict[str, Any]:
    """List available voice presets."""
    return {"presets": VOICE_MAP, "backend": TTS_BACKEND}


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    """Synthesize speech from text. Returns audio/mpeg bytes."""
    global _synth_count, _synth_chars

    text = sanitize_string(request.text).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars)")

    # resolve voice preset → edge-tts voice name
    voice = VOICE_MAP.get(request.voice, request.voice)
    if not voice:
        voice = DEFAULT_VOICE

    if TTS_BACKEND == "edge" and _edge_available:
        audio = await _synthesize_edge(text, voice, request.rate, request.volume)
        _synth_count += 1
        _synth_chars += len(text)
        return StreamingResponse(
            io.BytesIO(audio),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=kai_speech.mp3",
                "X-Voice": voice,
                "X-Chars": str(len(text)),
            },
        )

    raise HTTPException(
        status_code=503,
        detail=f"TTS backend '{TTS_BACKEND}' not available",
    )


# ── edge-tts backend ───────────────────────────────────────────────
async def _synthesize_edge(
    text: str, voice: str, rate: str, volume: str,
) -> bytes:
    """Collect audio bytes from edge-tts stream."""
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio = buf.getvalue()
        if not audio:
            raise HTTPException(status_code=500, detail="TTS returned empty audio")
        logger.info(
            "synthesized %d chars → %d bytes (voice=%s)",
            len(text), len(audio), voice,
        )
        return audio
    except HTTPException:
        raise
    except Exception as e:
        logger.error("edge-tts failed: %s", e)
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {e}")


# ── entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8030")))
