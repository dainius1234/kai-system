"""memory-compressor — background memory maintenance worker.

Runs scheduled jobs to keep the memory store healthy:
  1. Compression — archives old, low-relevance memories via memu-core
  2. Reflection — calls memu-core's reflect endpoint (sleep-cycle insights)
  3. Integrity check — verifies memory stats and flags anomalies

Can run on a cron schedule, or be triggered manually via its HTTP API.
Not latency-sensitive — designed to run during quiet periods (e.g. nightly).
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

from common.runtime import AuditStream, ErrorBudget, setup_json_logger

logger = setup_json_logger("memory-compressor", os.getenv("LOG_PATH", "/tmp/memory-compressor.json.log"))

app = FastAPI(title="Memory Compressor", version="0.5.0")

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
SCHEDULE_INTERVAL_HOURS = int(os.getenv("COMPRESSOR_INTERVAL_HOURS", "24"))
MAX_RETRIES = int(os.getenv("COMPRESSOR_MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("COMPRESSOR_RETRY_BACKOFF", "5.0"))

budget = ErrorBudget(window_seconds=600)
audit = AuditStream("memory-compressor", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")

# ── run history ───────────────────────────────────────────────────────
_run_history: List[Dict[str, Any]] = []
_scheduler_task: asyncio.Task | None = None


async def _call_memu(path: str, method: str = "POST", timeout: float = 60.0) -> Dict[str, Any]:
    """Call a memu-core endpoint with retries and exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "POST":
                    resp = await client.post(f"{MEMU_URL}{path}")
                else:
                    resp = await client.get(f"{MEMU_URL}{path}")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_err = exc
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning(
                "memu-core %s attempt %d/%d failed: %s — retrying in %.1fs",
                path, attempt, MAX_RETRIES, exc, wait,
            )
            await asyncio.sleep(wait)
    logger.error("memu-core %s failed after %d attempts", path, MAX_RETRIES)
    raise last_err  # type: ignore[misc]


async def run_compression_cycle() -> Dict[str, Any]:
    """Execute a full compression + reflection cycle.

    Steps:
      1. GET /memory/stats — record baseline
      2. POST /memory/compress — archive old records
      3. POST /memory/reflect — generate insight summaries
      4. GET /memory/stats — record post-cycle state
    """
    started = time.time()
    result: Dict[str, Any] = {"started_at": datetime.utcnow().isoformat(), "steps": {}}

    try:
        # 1. Pre-cycle stats
        pre_stats = await _call_memu("/memory/stats", method="GET")
        result["steps"]["pre_stats"] = pre_stats
        result["pre_record_count"] = pre_stats.get("records", 0)
        logger.info("Pre-compression: %d records", pre_stats.get("records", 0))

        # 2. Compress — archive stale memories
        compress_result = await _call_memu("/memory/compress")
        result["steps"]["compress"] = compress_result
        logger.info(
            "Compression: archived=%s, bytes_saved=%s",
            compress_result.get("archived", 0),
            compress_result.get("bytes_saved", 0),
        )

        # 3. Reflect — consolidate recent memories into insights
        reflect_result = await _call_memu("/memory/reflect")
        result["steps"]["reflect"] = reflect_result
        insights_count = len(reflect_result.get("insights", []))
        logger.info("Reflection: %d insights generated", insights_count)

        # 4. Post-cycle stats
        post_stats = await _call_memu("/memory/stats", method="GET")
        result["steps"]["post_stats"] = post_stats
        result["post_record_count"] = post_stats.get("records", 0)

        result["status"] = "completed"
        result["duration_ms"] = int((time.time() - started) * 1000)
        result["archived"] = compress_result.get("archived", 0)
        result["bytes_saved"] = compress_result.get("bytes_saved", 0)
        result["insights_generated"] = insights_count

        audit.log(
            "info",
            f"Compression cycle complete: archived="
            f"{compress_result.get('archived', 0)}, insights={insights_count}",
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["duration_ms"] = int((time.time() - started) * 1000)
        audit.log("error", f"Compression cycle failed: {exc}")
        logger.error("Compression cycle failed: %s", exc)

    _run_history.append(result)
    # keep only last 50 runs
    if len(_run_history) > 50:
        _run_history.pop(0)

    return result


async def _scheduled_loop() -> None:
    """Background loop that runs compression cycles on a schedule."""
    logger.info("Scheduled compressor started (interval=%dh)", SCHEDULE_INTERVAL_HOURS)
    while True:
        await asyncio.sleep(SCHEDULE_INTERVAL_HOURS * 3600)
        try:
            await run_compression_cycle()
        except Exception as exc:
            logger.error("Scheduled compression failed: %s", exc)


# ── HTTP endpoints ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global _scheduler_task
    if os.getenv("COMPRESSOR_SCHEDULE_ENABLED", "true").lower() == "true":
        _scheduler_task = asyncio.create_task(_scheduled_loop())
        logger.info("Background scheduler enabled")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "memory-compressor",
        "scheduler": "active" if _scheduler_task and not _scheduler_task.done() else "inactive",
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return {
        **budget.snapshot(),
        "total_runs": len(_run_history),
        "last_run": _run_history[-1] if _run_history else None,
    }


@app.post("/compress/run")
async def trigger_compression() -> Dict[str, Any]:
    """Manually trigger a compression + reflection cycle."""
    result = await run_compression_cycle()
    if result["status"] == "failed":
        raise HTTPException(status_code=502, detail=result)
    return result


@app.get("/compress/history")
async def compression_history(limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent compression run history."""
    return _run_history[-limit:]


@app.get("/compress/status")
async def compression_status() -> Dict[str, Any]:
    """Current compressor status summary."""
    last = _run_history[-1] if _run_history else None
    return {
        "status": "ok",
        "total_runs": len(_run_history),
        "scheduler_active": _scheduler_task is not None and not _scheduler_task.done(),
        "interval_hours": SCHEDULE_INTERVAL_HOURS,
        "last_run": {
            "status": last.get("status"),
            "started_at": last.get("started_at"),
            "duration_ms": last.get("duration_ms"),
            "archived": last.get("archived", 0),
            "insights_generated": last.get("insights_generated", 0),
        } if last else None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8057")))
