from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from common.runtime import sanitize_string

app = FastAPI(title="TTS Service", version="0.1.0")

MODEL = os.getenv("MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
LANGUAGE = os.getenv("LANGUAGE", "en")


class SynthesisRequest(BaseModel):
    text: str
    voice: str = "keeper"
    emotion: str = "neutral"


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "model": MODEL,
        "sample_rate": str(SAMPLE_RATE),
        "language": LANGUAGE,
    }


@app.post("/synthesize")
async def synthesize(request: SynthesisRequest) -> Dict[str, str]:
    return {
        "status": "ok",
        "voice": sanitize_string(request.voice),
        "emotion": sanitize_string(request.emotion),
        "audio_path": f"/audio/{sanitize_string(request.voice)}_{sanitize_string(request.emotion)}.wav",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8030")))
