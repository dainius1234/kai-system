"""Clipboard Service — browser-relayed clipboard history for Kai.

The browser frontend intercepts copy events and POSTs content here.
Kai's agentic layer reads /latest to enrich context when user says "this" or
"what I just copied".

Endpoints:
  POST /push          {content, source?}   → {ok, id}
  GET  /latest        →                      {content, source, timestamp, id}
  GET  /history       →                      [{content, source, timestamp, id}]
  DELETE /history     →                      {cleared}
  GET  /health
  GET  /metrics
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("clipboard-service", os.getenv("LOG_PATH", "/tmp/clipboard-service.json.log"))
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("clipboard-service")

    class ErrorBudget:  # type: ignore[no-redef]
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}

PORT = int(os.getenv("PORT", "8024"))
MAX_HISTORY = int(os.getenv("CLIPBOARD_MAX_HISTORY", "50"))
MAX_CONTENT_BYTES = int(os.getenv("CLIPBOARD_MAX_BYTES", str(256 * 1024)))  # 256 KB

_history: Deque[Dict] = deque(maxlen=MAX_HISTORY)
_counter = 0

app = FastAPI(title="clipboard-service")
budget = ErrorBudget(window_seconds=300)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    budget.record(response.status_code >= 500)
    return response


@app.get("/health")
async def health():
    return {"status": "ok", "entries": len(_history)}


@app.get("/metrics")
async def metrics():
    return budget.snapshot()


class PushRequest(BaseModel):
    content: str
    source: Optional[str] = "browser"


@app.post("/push")
async def push(req: PushRequest):
    global _counter
    if len(req.content.encode()) > MAX_CONTENT_BYTES:
        raise HTTPException(413, "clipboard content exceeds size limit")
    if not req.content.strip():
        return {"ok": True, "id": None, "note": "empty content ignored"}
    # Deduplicate consecutive identical pushes
    if _history and _history[-1]["content"] == req.content:
        return {"ok": True, "id": _history[-1]["id"], "note": "duplicate"}
    _counter += 1
    entry = {"id": _counter, "content": req.content, "source": req.source or "browser",
             "timestamp": time.time()}
    _history.append(entry)
    logger.debug("clipboard push id=%d source=%s len=%d", _counter, entry["source"], len(req.content))
    return {"ok": True, "id": _counter}


@app.get("/latest")
async def latest():
    if not _history:
        raise HTTPException(404, "clipboard is empty")
    return _history[-1]


@app.get("/history")
async def history(limit: int = 20):
    limit = min(limit, MAX_HISTORY)
    entries = list(_history)[-limit:]
    return {"entries": list(reversed(entries)), "total": len(_history)}


@app.delete("/history")
async def clear_history():
    _history.clear()
    return {"cleared": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
