from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from common.runtime import sanitize_string


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
        "voice": sanitize_string(request.voice),
        "emotion": sanitize_string(request.emotion),
        "message": "avatar request queued",
        "text": sanitize_string(request.text),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8081")))
