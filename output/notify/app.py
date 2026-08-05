"""Notify Service — desktop + in-app notification dispatch for Kai.

Primary: notify-send (Linux libnotify) for OS-level popups.
Fallback: in-memory pending queue that the dashboard polls.

Endpoints:
  POST /notify        {title, body, urgency?, timeout_ms?}  → {ok, id, channel}
  GET  /pending       →                                       [{id, title, body, timestamp}]
  DELETE /pending/:id →                                       {cleared}
  GET  /health
  GET  /metrics
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, HTTPException, Request

import sys as _sys, os as _os
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'common')):
    _repo = _os.path.dirname(_repo)
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
from common.service_auth import require_service_auth

from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("notify-service", os.getenv("LOG_PATH", "/tmp/notify-service.json.log"))
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("notify-service")

    class ErrorBudget:  # type: ignore[no-redef]
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}

PORT = int(os.getenv("PORT", "8031"))
MAX_PENDING = int(os.getenv("NOTIFY_MAX_PENDING", "100"))
NOTIFY_SEND_TIMEOUT = int(os.getenv("NOTIFY_SEND_TIMEOUT_MS", "5000"))

_pending: Deque[Dict] = deque(maxlen=MAX_PENDING)
_counter = 0

URGENCY_MAP = {"low": "low", "normal": "normal", "critical": "critical"}


def _try_notify_send(title: str, body: str, urgency: str, timeout_ms: int) -> bool:
    """Attempt OS-level notification via notify-send. Returns True on success."""
    try:
        result = subprocess.run(
            ["notify-send", "--urgency", urgency, "--expire-time", str(timeout_ms), title, body],
            timeout=3, capture_output=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


app = FastAPI(title="notify-service")
budget = ErrorBudget(window_seconds=300)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    budget.record(response.status_code >= 500)
    return response


@app.get("/health")
async def health():
    # Check if notify-send is available
    try:
        subprocess.run(["notify-send", "--version"], capture_output=True, timeout=2)
        notify_send = True
    except (FileNotFoundError, OSError):
        notify_send = False
    return {"status": "ok", "notify_send": notify_send, "pending": len(_pending)}


@app.get("/metrics")
async def metrics():
    return budget.snapshot()


class NotifyRequest(BaseModel):
    title: str
    body: str
    urgency: Optional[str] = "normal"
    timeout_ms: Optional[int] = 5000


@app.post("/notify",
          dependencies=[Depends(require_service_auth("desktop_notify"))])
async def notify(req: NotifyRequest):
    global _counter
    if not req.title.strip():
        raise HTTPException(400, "title is required")
    urgency = URGENCY_MAP.get(req.urgency or "normal", "normal")
    timeout_ms = max(1000, min(req.timeout_ms or 5000, 30000))

    sent_via = "queue"
    ok = await asyncio.get_event_loop().run_in_executor(
        None, _try_notify_send, req.title, req.body, urgency, timeout_ms
    )
    if ok:
        sent_via = "notify-send"
        logger.info("notify-send: %s", req.title)
    else:
        _counter += 1
        entry = {
            "id": _counter,
            "title": req.title,
            "body": req.body,
            "urgency": urgency,
            "timestamp": time.time(),
            "read": False,
        }
        _pending.append(entry)
        logger.info("notify queued id=%d: %s", _counter, req.title)

    return {"ok": True, "id": _counter if sent_via == "queue" else None, "channel": sent_via}


@app.get("/pending")
async def pending(unread_only: bool = True):
    entries = [e for e in _pending if not e["read"]] if unread_only else list(_pending)
    return {"notifications": list(reversed(entries)), "total": len(entries)}


@app.delete("/pending/{notification_id}",
            dependencies=[Depends(require_service_auth("notify_dismiss_one"))])
async def dismiss(notification_id: int):
    for entry in _pending:
        if entry["id"] == notification_id:
            entry["read"] = True
            return {"cleared": True}
    raise HTTPException(404, f"notification {notification_id} not found")


@app.delete("/pending",
            dependencies=[Depends(require_service_auth("notify_dismiss_all"))])
async def dismiss_all():
    for entry in _pending:
        entry["read"] = True
    return {"cleared": len(_pending)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
