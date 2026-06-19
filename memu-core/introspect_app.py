"""memU Introspection Service.

Owns the cold-path memory-maintenance endpoints that used to live inside
memu-core/app.py: store compaction/cleanup, spaced-repetition decay,
self-reflection insight generation, active-context compression, revert,
and quarantine management. None of these are on agentic's per-turn
/chat hot path — they're triggered periodically by memory-compressor or
manually by an operator — so they run as a separate process. A bug or
hang here (e.g. a slow focus-compress clustering pass) can no longer
stall or take down live retrieve/memorize traffic.

Every endpoint here is re-registered straight from app.py's own handler
functions (not duplicated), and all of them operate only on the shared
VectorStore (Postgres/Redis-backed) — never on the P17-P22 in-process
dicts (_formed_values, _autobiography, _echo_history, etc.) that the
eleven asyncio.Lock()s in app.py protect. That's a deliberate scope
boundary: those dicts only exist in app.py's process memory and are
flushed to Redis on a 5-minute lag, so anything that reads them live
(MARS-consolidate's conscience filter, self-reflect's feedback/emotion
signals, and the entire P17-P22 personality surface) stays in app.py's
process rather than risking a stale or empty read here. See DECISIONS.md
D21 for the full reasoning — this mirrors the two scope traps D9 found
when splitting agentic, except there are many more of them here.

The weekly store-compaction sweep also moves here in full (it only ever
touched store.compress(), never any locked state) as its own periodic
loop, instead of being checked on every hot-path request via app.py's
request middleware.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict

from fastapi import FastAPI

from common.runtime import AuditStream, detect_device, setup_json_logger

from app import (  # noqa: E402 — reuses app.py's store + handlers, see module docstring
    apply_spaced_repetition_decay,
    clear_quarantine,
    focus_compress,
    list_quarantined,
    memory_categories,
    memory_cleanup,
    memory_compress,
    memory_diagnostics,
    memory_revert,
    memory_state,
    memory_stats,
    quarantine_record,
    reflect,
    search_by_category,
    store,
)

logger = setup_json_logger("memu-core-introspect", os.getenv("LOG_PATH", "/tmp/memu_introspect.json.log"))
DEVICE = detect_device()

app = FastAPI(title="memU Introspection Service", version="0.1.0")
audit = AuditStream("memu-core-introspect", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")

WEEKLY_COMPRESS_INTERVAL_SECONDS = 7 * 24 * 3600


@app.middleware("http")
async def audit_middleware(request, call_next):
    try:
        response = await call_next(request)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "device": DEVICE}


# ── re-registered store-maintenance endpoints (handlers live in app.py) ──
app.post("/memory/compress")(memory_compress)
app.post("/memory/focus-compress")(focus_compress)
app.post("/memory/reflect")(reflect)
app.post("/memory/decay")(apply_spaced_repetition_decay)
app.post("/memory/cleanup")(memory_cleanup)
app.get("/memory/diagnostics")(memory_diagnostics)
app.post("/memory/revert")(memory_revert)
app.post("/revert")(memory_revert)
app.post("/memory/quarantine")(quarantine_record)
app.post("/memory/quarantine/clear")(clear_quarantine)
app.get("/memory/quarantine/list")(list_quarantined)
app.get("/memory/state")(memory_state)
app.get("/memory/categories")(memory_categories)
app.get("/memory/search-by-category")(search_by_category)
app.get("/memory/stats")(memory_stats)


async def _weekly_compress_loop() -> None:
    """Independent periodic store compaction — no longer gated on hot-path traffic."""
    while True:
        await asyncio.sleep(WEEKLY_COMPRESS_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(store.compress)
        except Exception:
            logger.warning("Weekly compress failed", exc_info=True)


@app.on_event("startup")
async def _start_background_loops() -> None:
    asyncio.create_task(_weekly_compress_loop())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8009")))
