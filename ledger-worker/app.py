"""ledger-worker — audit-trail maintenance and integrity monitor.

Background worker that monitors and maintains the tool-gate's
immutable hash-chain ledger.  Responsibilities:

  1. Integrity verification — periodic chain validation
  2. Statistics — ledger growth, approval rates, tool usage
  3. Archival — exports snapshots for offline backup
  4. Nonce cleanup — prunes expired replay-prevention nonces
  5. Alerting — flags integrity failures to heartbeat

Runs alongside tool-gate in the sovereign stack.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

from common.runtime import AuditStream, ErrorBudget, setup_json_logger

logger = setup_json_logger("ledger-worker", os.getenv("LOG_PATH", "/tmp/ledger-worker.json.log"))

app = FastAPI(title="Ledger Worker", version="0.5.0")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")
ARCHIVE_DIR = Path(os.getenv("LEDGER_ARCHIVE_DIR", "/tmp/ledger-archives"))
VERIFY_INTERVAL_MINUTES = int(os.getenv("LEDGER_VERIFY_INTERVAL", "60"))
MAX_RETRIES = int(os.getenv("LEDGER_WORKER_MAX_RETRIES", "3"))
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "")
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

budget = ErrorBudget(window_seconds=600)
audit = AuditStream("ledger-worker", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")

# ── state ─────────────────────────────────────────────────────────────
_verification_history: List[Dict[str, Any]] = []
_last_stats: Dict[str, Any] = {}
_scheduler_task: asyncio.Task | None = None


def _auth_headers() -> Dict[str, str]:
    """Build auth headers for tool-gate requests."""
    if BEARER_TOKEN:
        return {"Authorization": f"Bearer {BEARER_TOKEN}"}
    return {}


async def _call_tool_gate(path: str, method: str = "GET", timeout: float = 30.0) -> Dict[str, Any]:
    """Call a tool-gate endpoint with retries."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "POST":
                    resp = await client.post(f"{TOOL_GATE_URL}{path}", headers=_auth_headers())
                else:
                    resp = await client.get(f"{TOOL_GATE_URL}{path}", headers=_auth_headers())
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


async def _notify_heartbeat(event: str, details: Dict[str, Any]) -> None:
    """Send alert to heartbeat service."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(f"{HEARTBEAT_URL}/event", json={"status": event, **details})
    except Exception:
        logger.warning("Heartbeat notification failed for event=%s", event)


async def verify_chain() -> Dict[str, Any]:
    """Verify the entire ledger hash chain via tool-gate's /ledger/verify."""
    started = time.time()
    result: Dict[str, Any] = {"timestamp": datetime.utcnow().isoformat()}

    try:
        verify = await _call_tool_gate("/ledger/verify")
        result["valid"] = verify.get("valid", False)
        result["count"] = verify.get("count", 0)
        result["status"] = "ok" if verify.get("valid") else "integrity_failure"

        if not verify.get("valid"):
            result["failed_request_id"] = verify.get("failed_request_id")
            logger.error("LEDGER INTEGRITY FAILURE at request_id=%s",
                         verify.get("failed_request_id", "unknown"))
            await _notify_heartbeat("ledger_integrity_failure", {
                "reason": "Hash chain verification failed",
                "failed_request_id": verify.get("failed_request_id"),
            })
            audit.log("error", f"Ledger integrity failure: {verify}")

        # also get merkle root
        try:
            merkle = await _call_tool_gate("/ledger/merkle")
            result["merkle_root"] = merkle.get("merkle_root")
        except Exception:
            result["merkle_root"] = None

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["valid"] = None
        logger.error("Chain verification failed: %s", exc)

    result["duration_ms"] = int((time.time() - started) * 1000)
    _verification_history.append(result)
    if len(_verification_history) > 100:
        _verification_history.pop(0)

    return result


async def collect_stats() -> Dict[str, Any]:
    """Gather ledger statistics from tool-gate."""
    global _last_stats
    try:
        stats = await _call_tool_gate("/ledger/stats")
        tail = await _call_tool_gate("/ledger/tail?limit=100")

        # analyze recent entries
        approved = sum(1 for e in tail if e.get("approved"))
        denied = len(tail) - approved
        tools_used: Dict[str, int] = {}
        for entry in tail:
            tool = entry.get("payload", {}).get("tool")
            if tool:
                tools_used[tool] = tools_used.get(tool, 0) + 1

        _last_stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_entries": stats.get("count", 0),
            "recent_sample": len(tail),
            "approved": approved,
            "denied": denied,
            "approval_rate": round(approved / max(len(tail), 1), 3),
            "tools_used": tools_used,
        }
        return _last_stats
    except Exception as exc:
        logger.error("Stats collection failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def archive_snapshot() -> Dict[str, Any]:
    """Export the current ledger tail to an archive file."""
    try:
        tail = await _call_tool_gate("/ledger/tail?limit=10000")
        merkle = await _call_tool_gate("/ledger/merkle")

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = ARCHIVE_DIR / f"ledger_snapshot_{ts}.jsonl"

        with archive_path.open("w", encoding="utf-8") as fh:
            # header with metadata
            fh.write(json.dumps({
                "_type": "ledger_archive_header",
                "timestamp": ts,
                "count": len(tail),
                "merkle_root": merkle.get("merkle_root"),
                "valid": merkle.get("valid"),
            }, sort_keys=True) + "\n")
            for entry in tail:
                fh.write(json.dumps(entry, sort_keys=True) + "\n")

        size = archive_path.stat().st_size
        logger.info("Ledger archived: %s (%d entries, %d bytes)", archive_path.name, len(tail), size)
        audit.log("info", f"Ledger archived: {archive_path.name} ({len(tail)} entries)")

        # cleanup old archives (keep last 30)
        archives = sorted(ARCHIVE_DIR.glob("ledger_snapshot_*.jsonl"), reverse=True)
        for old in archives[30:]:
            old.unlink()
            logger.info("Removed old archive: %s", old.name)

        return {
            "status": "ok",
            "path": str(archive_path),
            "entries": len(tail),
            "bytes": size,
            "merkle_root": merkle.get("merkle_root"),
        }
    except Exception as exc:
        logger.error("Archive failed: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _scheduled_loop() -> None:
    """Background loop: verify chain + collect stats on schedule."""
    logger.info("Ledger worker scheduler started (interval=%dm)", VERIFY_INTERVAL_MINUTES)
    while True:
        await asyncio.sleep(VERIFY_INTERVAL_MINUTES * 60)
        try:
            await verify_chain()
            await collect_stats()
        except Exception as exc:
            logger.error("Scheduled verification failed: %s", exc)


# ── HTTP endpoints ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global _scheduler_task
    if os.getenv("LEDGER_WORKER_SCHEDULE_ENABLED", "true").lower() == "true":
        _scheduler_task = asyncio.create_task(_scheduled_loop())
        logger.info("Background scheduler enabled")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "ledger-worker",
        "scheduler": "active" if _scheduler_task and not _scheduler_task.done() else "inactive",
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return {
        **budget.snapshot(),
        "verifications": len(_verification_history),
        "last_verification": _verification_history[-1] if _verification_history else None,
        "last_stats": _last_stats,
    }


@app.post("/verify")
async def trigger_verify() -> Dict[str, Any]:
    """Manually trigger ledger chain verification."""
    return await verify_chain()


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get current ledger statistics (if cached) or collect fresh."""
    if not _last_stats:
        return await collect_stats()
    return _last_stats


@app.post("/stats/refresh")
async def refresh_stats() -> Dict[str, Any]:
    """Force-refresh ledger statistics."""
    return await collect_stats()


@app.post("/archive")
async def trigger_archive() -> Dict[str, Any]:
    """Manually trigger a ledger snapshot archive."""
    result = await archive_snapshot()
    if result.get("status") == "error":
        raise HTTPException(status_code=502, detail=result)
    return result


@app.get("/archive/list")
async def list_archives(offset: int = 0, limit: int = 30) -> Dict[str, Any]:
    """List available archived snapshots with pagination."""
    archives = sorted(ARCHIVE_DIR.glob("ledger_snapshot_*.jsonl"), reverse=True)
    total = len(archives)
    page = archives[offset:offset + limit]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "archives": [
            {
                "name": a.name,
                "bytes": a.stat().st_size,
                "created": datetime.fromtimestamp(a.stat().st_mtime).isoformat(),
            }
            for a in page
        ],
    }


@app.get("/history")
async def verification_history(
    limit: int = 20, offset: int = 0,
) -> Dict[str, Any]:
    """Return recent verification results with pagination."""
    total = len(_verification_history)
    # slice from the end (most recent first)
    end = total - offset
    start = max(end - limit, 0)
    page = _verification_history[start:end]
    page.reverse()  # newest first
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "history": page,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8056")))
