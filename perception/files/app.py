"""File Watcher Service — inotify-based directory monitoring for Kai.

Watches configured directories for create/modify/delete/move events and keeps
a rolling event log. Agentic can poll /events to know what the user is working on.

Endpoints:
  GET  /events        ?limit=N            → [{path, event, timestamp}]
  POST /watch         {directory}         → {ok, watching}
  DELETE /watch       {directory}         → {ok, watching}
  GET  /watching      →                   [directories]
  GET  /health
  GET  /metrics

Configure watched dirs via WATCH_DIRS env (colon-separated paths).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("files-service", os.getenv("LOG_PATH", "/tmp/files-service.json.log"))
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("files-service")

    class ErrorBudget:  # type: ignore[no-redef]
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    _WATCHDOG_OK = True
except ImportError:
    _WATCHDOG_OK = False
    logger.info("watchdog not available — file-watcher in stub mode")

PORT = int(os.getenv("PORT", "8025"))
MAX_EVENTS = int(os.getenv("FILES_MAX_EVENTS", "200"))
WATCH_DIRS_ENV = os.getenv("WATCH_DIRS", "")

_events: Deque[Dict] = deque(maxlen=MAX_EVENTS)
_watching: List[str] = []
_observer: Optional["Observer"] = None


class _Handler(FileSystemEventHandler):  # type: ignore[misc]
    def on_any_event(self, event: "FileSystemEvent"):
        if event.is_directory:
            return
        _events.append({
            "path": str(event.src_path),
            "event": event.event_type,
            "timestamp": time.time(),
        })


def _start_watching(directory: str) -> bool:
    global _observer
    if not _WATCHDOG_OK:
        return False
    p = Path(directory)
    if not p.exists():
        logger.warning("watch dir does not exist: %s", directory)
        return False
    if directory in _watching:
        return True
    if _observer is None:
        _observer = Observer()
        _observer.start()
    _observer.schedule(_Handler(), str(p), recursive=True)
    _watching.append(directory)
    logger.info("watching: %s", directory)
    return True


def _stop_watching(directory: str) -> bool:
    if directory in _watching:
        _watching.remove(directory)
        return True
    return False


# Start watching configured dirs at import time
for _d in [d.strip() for d in WATCH_DIRS_ENV.split(":") if d.strip()]:
    _start_watching(_d)

app = FastAPI(title="files-service")
budget = ErrorBudget(window_seconds=300)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    budget.record(response.status_code >= 500)
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "watchdog": _WATCHDOG_OK,
        "watching": _watching,
        "events_buffered": len(_events),
    }


@app.get("/metrics")
async def metrics():
    return budget.snapshot()


@app.get("/events")
async def events(limit: int = 50, event_type: Optional[str] = None):
    limit = min(limit, MAX_EVENTS)
    result = list(_events)
    if event_type:
        result = [e for e in result if e["event"] == event_type]
    return {"events": list(reversed(result))[-limit:], "total": len(_events)}


@app.get("/watching")
async def watching():
    return {"directories": _watching, "watchdog": _WATCHDOG_OK}


class WatchRequest(BaseModel):
    directory: str


@app.post("/watch")
async def add_watch(req: WatchRequest):
    if not _WATCHDOG_OK:
        raise HTTPException(503, "watchdog not available")
    ok = _start_watching(req.directory)
    if not ok:
        raise HTTPException(400, f"directory not found or already errored: {req.directory}")
    return {"ok": True, "watching": _watching}


@app.delete("/watch")
async def remove_watch(req: WatchRequest):
    removed = _stop_watching(req.directory)
    return {"ok": removed, "watching": _watching}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
