from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Avatar Service", version="0.1.0")

TTS_URL = os.getenv("TTS_URL", "http://tts-service:8030")
WEBRTC_PORT = os.getenv("WEBRTC_PORT", "8081")


class SpeakRequest(BaseModel):
    text: str
    voice: str = "keeper"
    emotion: str = "neutral"


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "tts_url": TTS_URL, "webrtc_port": WEBRTC_PORT}


@app.post("/speak")
async def speak(request: SpeakRequest) -> Dict[str, str]:
    return {
        "status": "ok",
        "tts_url": TTS_URL,
        "voice": request.voice,
        "emotion": request.emotion,
        "message": "avatar request queued",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8081")))
