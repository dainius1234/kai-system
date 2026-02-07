from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="Audio Service", version="0.1.0")

HOTWORD = os.getenv("PORCUPINE_KEYWORD", "ara")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "hotword": HOTWORD,
        "whisper_model": WHISPER_MODEL,
    }


@app.post("/listen")
async def listen() -> Dict[str, str]:
    return {"status": "ok", "message": "listening"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8021")))
