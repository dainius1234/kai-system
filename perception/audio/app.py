from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("perception-audio", os.getenv("LOG_PATH", "/tmp/perception-audio.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Audio Service", version="0.4.0")
HOTWORD = os.getenv("PORCUPINE_KEYWORD", "ara")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
INJECTION_LOG = Path(os.getenv("INJECTION_LOG", "/tmp/injection_events.log"))
budget = ErrorBudget(window_seconds=300)


class TranscriptRequest(BaseModel):
    text: str
    session_id: str = "unknown"


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
    return {"status": "ok", "hotword": HOTWORD, "whisper_model": WHISPER_MODEL, "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/listen")
async def listen(request: TranscriptRequest) -> Dict[str, str]:
    clean = sanitize_string(request.text)
    if INJECTION_RE.search(clean):
        INJECTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        INJECTION_LOG.write_text(f"blocked session={sanitize_string(request.session_id)} text={clean}\n", encoding="utf-8")
        logger.warning("prompt injection blocked")
        raise HTTPException(status_code=400, detail="prompt injection pattern blocked")
    return {"status": "ok", "message": "accepted", "text": clean}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8021")))
