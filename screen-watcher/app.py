"""Screen-watcher service — periodic screenshot diff + change alerting.

Captures screenshots from screen-capture service on an interval, computes
a perceptual hash diff, and fires notify/TTS alerts when change exceeds threshold.

Endpoints:
  GET  /health          → {status, watching, uptime_seconds}
  GET  /metrics         → error budget
  GET  /status          → {watching, interval, last_capture_ts, last_change_ts, diff_score}
  GET  /snapshot        → latest screenshot bytes (image/png) from cache
  POST /watch/start     {interval_seconds?, threshold?}  → start watching
  POST /watch/stop      → stop watching
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
from contextlib import asynccontextmanager, suppress
from typing import Optional

import httpx

from common.http_hygiene import pooled_client
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("screen-watcher", os.getenv("LOG_PATH", "/tmp/screen-watcher.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("screen-watcher")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

PORT = int(os.getenv("PORT", "8036"))
SCREEN_CAPTURE_URL = os.getenv("SCREEN_CAPTURE_URL", "http://screen-capture:8020")
NOTIFY_URL = os.getenv("NOTIFY_SERVICE_URL", "http://notify-service:8031")
TTS_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8030")
DEFAULT_INTERVAL = int(os.getenv("WATCH_INTERVAL_SECONDS", "10"))
DEFAULT_THRESHOLD = float(os.getenv("CHANGE_THRESHOLD", "0.05"))  # 5% pixel hash difference

_start = time.time()
_watching = False
_interval = DEFAULT_INTERVAL
_threshold = DEFAULT_THRESHOLD
_last_capture_ts: float = 0.0
_last_change_ts: float = 0.0
_last_diff: float = 0.0
_last_screenshot: Optional[bytes] = None
_prev_hash: Optional[str] = None
_watch_task: Optional[asyncio.Task] = None


def _image_hash(data: bytes) -> str:
    """Simple block-sample hash for change detection — no PIL dependency."""
    # Sample every Nth byte for a rough perceptual fingerprint
    step = max(1, len(data) // 1024)
    sample = data[::step][:1024]
    return hashlib.md5(sample).hexdigest()


def _diff_score(h1: str, h2: str) -> float:
    """Fraction of hex chars that differ — crude but fast."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 1.0
    diffs = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return diffs / len(h1)


async def _capture_screen() -> Optional[bytes]:
    try:
        async with pooled_client(timeout=10) as client:
            resp = await client.post(f"{SCREEN_CAPTURE_URL}/screenshot")
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.warning("screen capture error: %s", exc)
        return None


async def _send_notify(title: str, body: str):
    with suppress(Exception):
        async with pooled_client(timeout=5) as client:
            await client.post(f"{NOTIFY_URL}/notify", json={"title": title, "body": body})


async def _send_tts(text: str):
    with suppress(Exception):
        async with pooled_client(timeout=10) as client:
            await client.post(f"{TTS_URL}/synthesize", json={"text": text})


async def _watch_loop():
    global _last_capture_ts, _last_change_ts, _last_diff, _last_screenshot, _prev_hash
    while _watching:
        data = await _capture_screen()
        if data:
            _last_screenshot = data
            _last_capture_ts = time.time()
            h = _image_hash(data)
            if _prev_hash is not None:
                diff = _diff_score(_prev_hash, h)
                _last_diff = diff
                if diff >= _threshold:
                    _last_change_ts = time.time()
                    logger.info("screen change detected: diff=%.3f (threshold=%.3f)", diff, _threshold)
                    asyncio.create_task(_send_notify("Screen changed", f"Change score: {diff:.2%}"))
            _prev_hash = h
        await asyncio.sleep(_interval)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    yield


app = FastAPI(title="screen-watcher", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "watching": _watching, "uptime_seconds": round(time.time() - _start, 1)}


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/status")
def status():
    return {
        "watching": _watching,
        "interval_seconds": _interval,
        "threshold": _threshold,
        "last_capture_ts": _last_capture_ts,
        "last_change_ts": _last_change_ts,
        "last_diff_score": _last_diff,
    }


@app.get("/snapshot")
def snapshot():
    if not _last_screenshot:
        raise HTTPException(404, "no screenshot available yet")
    return Response(content=_last_screenshot, media_type="image/png")


class WatchStartRequest(BaseModel):
    interval_seconds: Optional[int] = None
    threshold: Optional[float] = None


@app.post("/watch/start")
async def watch_start(req: WatchStartRequest = WatchStartRequest()):
    global _watching, _interval, _threshold, _watch_task, _prev_hash
    if req.interval_seconds is not None:
        _interval = max(2, req.interval_seconds)
    if req.threshold is not None:
        _threshold = max(0.0, min(1.0, req.threshold))
    if _watching:
        return {"ok": True, "already_watching": True}
    _watching = True
    _prev_hash = None
    _watch_task = asyncio.create_task(_watch_loop())
    logger.info("screen watcher started: interval=%ds threshold=%.3f", _interval, _threshold)
    return {"ok": True, "interval_seconds": _interval, "threshold": _threshold}


@app.post("/watch/stop")
async def watch_stop():
    global _watching, _watch_task
    _watching = False
    if _watch_task:
        _watch_task.cancel()
        _watch_task = None
    logger.info("screen watcher stopped")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
