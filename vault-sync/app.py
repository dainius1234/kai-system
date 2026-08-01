"""D91: Vault-sync FastAPI service.

Bidirectional Obsidian vault ↔ memu-core knowledge graph sync.
Port 8047 / 172.20.0.36
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException

import sys as _sys, os as _os
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'common')):
    _repo = _os.path.dirname(_repo)
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
from common.service_auth import require_service_auth

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mapper import VaultMapper
from parser import NoteData, parse_note
from watcher import FileWatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vault-sync")

app = FastAPI(title="vault-sync", version="1.0.0")

# ── Config ────────────────────────────────────────────────────────────────────

VAULT_PATH = os.getenv("VAULT_PATH", "/vault")
MEMU_CORE_URL = os.getenv("MEMU_CORE_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
FF_VAULT_SYNC = os.getenv("FF_VAULT_SYNC", "true").lower() == "true"

# Gate conviction threshold for autonomous writes
VAULT_WRITE_CONVICTION_THRESHOLD = float(os.getenv("VAULT_WRITE_CONVICTION_THRESHOLD", "9.0"))

_mapper: Optional[VaultMapper] = None
_watcher: Optional[FileWatcher] = None
_ingest_queue: asyncio.Queue = asyncio.Queue()
_delete_queue: asyncio.Queue = asyncio.Queue()


# ── Pydantic models ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    filepath: str
    force: bool = False


class ExportRequest(BaseModel):
    filepath: str            # relative to vault root, e.g. "KAI/lesson-2026-07-24.md"
    content: str
    conviction: float = 0.0  # must be ≥ VAULT_WRITE_CONVICTION_THRESHOLD
    requester: str = "kai"


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    folder_filter: Optional[str] = None


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global _mapper, _watcher

    if not FF_VAULT_SYNC:
        logger.info("FF_VAULT_SYNC=false — vault-sync service inactive")
        return

    _mapper = VaultMapper(VAULT_PATH)
    logger.info("VaultMapper loaded — %d entries", len(_mapper))

    _watcher = FileWatcher(
        vault_path=VAULT_PATH,
        on_change=_enqueue_change,
        on_delete=_enqueue_delete,
    )
    _watcher.start()

    asyncio.create_task(_process_ingest_queue())
    asyncio.create_task(_process_delete_queue())

    logger.info("vault-sync startup complete")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _watcher:
        _watcher.stop()


# ── Queue callbacks (called from watchdog thread) ─────────────────────────────

def _enqueue_change(filepath: str) -> None:
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(_ingest_queue.put_nowait, filepath)
    except Exception as exc:
        logger.error("Failed to enqueue change for %s: %s", filepath, exc)


def _enqueue_delete(filepath: str) -> None:
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(_delete_queue.put_nowait, filepath)
    except Exception as exc:
        logger.error("Failed to enqueue delete for %s: %s", filepath, exc)


# ── Background queue workers ──────────────────────────────────────────────────

async def _process_ingest_queue() -> None:
    while True:
        filepath = await _ingest_queue.get()
        try:
            await _ingest_note(filepath)
        except Exception as exc:
            logger.error("Ingest queue error for %s: %s", filepath, exc)
        finally:
            _ingest_queue.task_done()


async def _process_delete_queue() -> None:
    while True:
        filepath = await _delete_queue.get()
        try:
            await _handle_delete(filepath)
        except Exception as exc:
            logger.error("Delete queue error for %s: %s", filepath, exc)
        finally:
            _delete_queue.task_done()


# ── Core ingest logic ─────────────────────────────────────────────────────────

async def _ingest_note(filepath: str) -> None:
    """Parse a note and sync it to memu-core. Skip if checksum unchanged."""
    note = parse_note(filepath)
    if note is None:
        logger.warning("Could not parse note: %s", filepath)
        return

    entry = _mapper.get_by_filepath(filepath) if _mapper else None
    if entry and entry.get("last_synced_checksum") == note.checksum:
        return  # no change

    result = await _push_to_memu_core(note)
    if result is None:
        return

    _mapper.upsert(
        filepath=filepath,
        note_node_id=result.get("note_node_id", ""),
        concept_ids=result.get("concept_ids", []),
        checksum=note.checksum,
        synced_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info("Ingested %s → node %s", filepath, result.get("note_node_id"))


async def _handle_delete(filepath: str) -> None:
    """Remove a deleted note from the knowledge graph."""
    if _mapper is None:
        return
    entry = _mapper.get_by_filepath(filepath)
    if not entry:
        return

    note_node_id = entry.get("note_node_id", "")
    if note_node_id:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.delete(f"{MEMU_CORE_URL}/memory/vault/{note_node_id}")
        except Exception as exc:
            logger.error("Failed to delete node %s from memu-core: %s", note_node_id, exc)

    _mapper.remove(filepath)
    logger.info("Deleted note mapping: %s", filepath)


async def _push_to_memu_core(note: NoteData) -> Optional[Dict]:
    """POST note to memu-core vault ingest endpoint."""
    payload = {
        "filepath": note.filepath,
        "title": note.title,
        "content": note.content,
        "frontmatter": note.frontmatter,
        "wikilinks": [{"target": t, "alias": a} for t, a in note.wikilinks],
        "tags": note.tags,
        "checksum": note.checksum,
        "modified_at": note.modified_at,
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{MEMU_CORE_URL}/memory/vault/ingest", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error("memu-core ingest failed for %s: %s", note.filepath, exc)
        return None


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict:
    return {
        "status": "ok",
        "ff_vault_sync": FF_VAULT_SYNC,
        "watcher_running": _watcher.running if _watcher else False,
        "mapped_notes": len(_mapper) if _mapper else 0,
        "vault_path": VAULT_PATH,
        "queue_depth": _ingest_queue.qsize(),
    }


@app.post("/ingest")
async def ingest(req: IngestRequest) -> Dict:
    """Manually trigger ingest for a specific filepath."""
    if not FF_VAULT_SYNC:
        raise HTTPException(503, "FF_VAULT_SYNC is disabled")
    if _mapper is None:
        raise HTTPException(503, "Service not ready")

    note = parse_note(req.filepath)
    if note is None:
        raise HTTPException(404, f"Cannot read note: {req.filepath}")

    entry = _mapper.get_by_filepath(req.filepath)
    if not req.force and entry and entry.get("last_synced_checksum") == note.checksum:
        return {"status": "skipped", "reason": "checksum_unchanged", "filepath": req.filepath}

    result = await _push_to_memu_core(note)
    if result is None:
        raise HTTPException(502, "memu-core ingest failed")

    _mapper.upsert(
        filepath=req.filepath,
        note_node_id=result.get("note_node_id", ""),
        concept_ids=result.get("concept_ids", []),
        checksum=note.checksum,
        synced_at=datetime.now(timezone.utc).isoformat(),
    )
    return {"status": "ok", "filepath": req.filepath, "note_node_id": result.get("note_node_id")}


@app.post("/export",
          dependencies=[Depends(require_service_auth("vault_export"))])
async def export(req: ExportRequest) -> Dict:
    """Write a note from Kai into the vault (gate-checked, conviction ≥ threshold)."""
    if not FF_VAULT_SYNC:
        raise HTTPException(503, "FF_VAULT_SYNC is disabled")

    if req.conviction < VAULT_WRITE_CONVICTION_THRESHOLD:
        raise HTTPException(
            403,
            f"Conviction {req.conviction:.1f} below threshold {VAULT_WRITE_CONVICTION_THRESHOLD}. "
            "Vault write denied.",
        )

    import pathlib
    target = pathlib.Path(VAULT_PATH) / req.filepath
    # Safety: must stay within vault root
    try:
        target.resolve().relative_to(pathlib.Path(VAULT_PATH).resolve())
    except ValueError:
        raise HTTPException(400, "filepath escapes vault root")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    logger.info("Exported note to %s (conviction=%.1f, requester=%s)", target, req.conviction, req.requester)

    # Immediately ingest so the graph reflects the new note
    await _ingest_note(str(target))

    return {"status": "ok", "filepath": req.filepath, "absolute_path": str(target)}


@app.get("/search")
async def search(query: str, limit: int = 10, folder_filter: Optional[str] = None) -> Dict:
    """Proxy hybrid search to memu-core vault search."""
    if not FF_VAULT_SYNC:
        raise HTTPException(503, "FF_VAULT_SYNC is disabled")

    params: Dict = {"query": query, "limit": limit}
    if folder_filter:
        params["folder_filter"] = folder_filter

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{MEMU_CORE_URL}/memory/vault/search", params=params)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        raise HTTPException(502, f"memu-core search failed: {exc}")


@app.get("/mapping")
async def mapping() -> Dict:
    """Return all known filepath→node mappings (diagnostic)."""
    if _mapper is None:
        return {"entries": {}}
    return {"entries": _mapper.all_entries()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8047, reload=False)
