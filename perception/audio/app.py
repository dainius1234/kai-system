from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

LOG_PATH = os.getenv("LOG_PATH", "/tmp/perception-audio.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("perception-audio")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Audio Service", version="0.3.0")

HOTWORD = os.getenv("PORCUPINE_KEYWORD", "ara")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
INJECTION_LOG = Path(os.getenv("INJECTION_LOG", "/tmp/injection_events.log"))
ERROR_WINDOW_SECONDS = 300
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "audio-service")
_metrics: Deque[Tuple[float, int]] = deque()


class TranscriptRequest(BaseModel):
    text: str
    session_id: str = "unknown"


def sanitize_string(value: str) -> str:
    sanitized = re.sub(r"[;|&]", "", value)
    return sanitized[:1024]


def _prune_metrics(now: float) -> None:
    while _metrics and now - _metrics[0][0] > ERROR_WINDOW_SECONDS:
        _metrics.popleft()


def _record_status(code: int) -> None:
    now = time.time()
    _metrics.append((now, code))
    _prune_metrics(now)


def _error_budget() -> Dict[str, float]:
    now = time.time()
    _prune_metrics(now)
    total = len(_metrics)
    if total == 0:
        return {"error_ratio": 0.0, "total": 0}
    errors = sum(1 for _, code in _metrics if code in {429, 500, 408})
    return {"error_ratio": errors / total, "total": total}


def _maybe_restart() -> None:
    budget = _error_budget()
    if budget["total"] < 10:
        return
    if budget["error_ratio"] > 0.03 and shutil.which("docker"):
        subprocess.run(["docker", "restart", SERVICE_CONTAINER], check=False)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        _record_status(response.status_code)
        _maybe_restart()
        return response
    except Exception:
        _record_status(500)
        _maybe_restart()
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "hotword": HOTWORD,
        "whisper_model": WHISPER_MODEL,
        "device": DEVICE,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return _error_budget()


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
