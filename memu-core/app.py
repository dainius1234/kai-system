from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, init_audit_or_exit, sanitize_string, setup_json_logger

logger = setup_json_logger("memu-core", os.getenv("LOG_PATH", "/tmp/memu-core.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)
audit = init_audit_or_exit("memu-core", logger)

app = FastAPI(title="Sovereign Memory Core", version="0.5.0")
budget = ErrorBudget(window_seconds=300)
last_compress_run = 0.0


class MemoryRequest(BaseModel):
    query: str
    session_id: str
    timestamp: str


class RoutingResponse(BaseModel):
    specialist: str
    context_payload: Dict[str, Any]


class MemoryUpdate(BaseModel):
    timestamp: str
    event_type: str
    task_id: Optional[str] = None
    result_raw: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    state_delta: Optional[Dict[str, Any]] = None
    relevance: float = 1.0
    user_id: str = "keeper"


class MemoryRecord(BaseModel):
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    embedding: List[float]
    relevance: float = 1.0


class VersionRef(BaseModel):
    version_id: int
    commit_hash: str
    message: str
    timestamp: str


class RevertRequest(BaseModel):
    version_id: int


SPECIALISTS = ["DeepSeek-V4", "Kimi-2.5", "Qwen-VL"]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []
        self._state: Dict[str, Any] = {}
        self._versions: List[Dict[str, Any]] = []
        self._last_commit_hash = "GENESIS"

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "records": [r.model_dump() for r in self._records],
            "state": dict(self._state),
        }

    def _commit(self, message: str) -> VersionRef:
        snapshot = self._snapshot()
        material = json.dumps(snapshot, sort_keys=True)
        commit_hash = hashlib.sha256(f"{self._last_commit_hash}|{material}|{message}".encode("utf-8")).hexdigest()
        version = {
            "version_id": len(self._versions),
            "commit_hash": commit_hash,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "snapshot": snapshot,
        }
        self._versions.append(version)
        self._last_commit_hash = commit_hash
        return VersionRef(version_id=version["version_id"], commit_hash=commit_hash, message=message, timestamp=version["timestamp"])

    def insert(self, record: MemoryRecord, user_id: str) -> VersionRef:
        self._records.append(record)
        msg = f"update: user_id={sanitize_string(user_id)}, timestamp={record.timestamp}, record_id={record.id}"
        return self._commit(msg)

    def search(self, top_k: int) -> List[MemoryRecord]:
        return list(reversed(self._records))[:top_k]

    def count(self) -> int:
        return len(self._records)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> None:
        self._state.update(delta)

    def revert(self, version_id: int) -> VersionRef:
        if version_id < 0 or version_id >= len(self._versions):
            raise ValueError("invalid version_id")
        snapshot = self._versions[version_id]["snapshot"]
        self._records = [MemoryRecord(**r) for r in snapshot["records"]]
        self._state = dict(snapshot["state"])
        return self._commit(f"revert: version={version_id}")

    def versions(self, limit: int = 20) -> List[VersionRef]:
        payload = self._versions[-limit:]
        return [VersionRef(version_id=v["version_id"], commit_hash=v["commit_hash"], message=v["message"], timestamp=v["timestamp"]) for v in payload]

    def compress(self) -> Dict[str, Any]:
        threshold = datetime.utcnow() - timedelta(days=90)
        before_bytes = sum(len(r.model_dump_json()) for r in self._records)
        kept: List[MemoryRecord] = []
        for record in self._records:
            ts = datetime.fromisoformat(record.timestamp) if "T" in record.timestamp else datetime.utcnow()
            if ts > threshold or record.relevance >= 0.2:
                kept.append(record)
        before = len(self._records)
        self._records = kept
        after_bytes = sum(len(r.model_dump_json()) for r in self._records)
        saved = max(before_bytes - after_bytes, 0)
        self._commit("compress: prune records older than 90 days")
        logger.info("weekly compression complete, bytes_saved=%s", saved)
        audit.write("INFO", f"weekly compression complete bytes_saved={saved}")
        return {"before": before, "after": len(self._records), "bytes_saved": saved, "retention_days": 90}


store = InMemoryVectorStore()


def generate_embedding(text: str) -> List[float]:
    seed = sum(bytearray(text.encode("utf-8"))) % 100
    return [seed / 100.0 for _ in range(8)]


def select_specialist(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["image", "vision", "camera", "diagram"]):
        return "Qwen-VL"
    if any(k in q for k in ["plan", "reason", "policy", "risk"]):
        return "DeepSeek-V4"
    return "Kimi-2.5"


def _weekly_compress_if_due() -> None:
    global last_compress_run
    now = time.time()
    if now - last_compress_run >= 7 * 24 * 3600:
        store.compress()
        last_compress_run = now


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        _weekly_compress_if_due()
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "storage": os.getenv("VECTOR_STORE", "memory"), "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/route", response_model=RoutingResponse)
async def route_request(request: MemoryRequest) -> RoutingResponse:
    query = sanitize_string(request.query)
    session_id = sanitize_string(request.session_id)
    similar = store.search(top_k=50)
    return RoutingResponse(
        specialist=select_specialist(query),
        context_payload={
            "query": query,
            "memory_vectors": [record.embedding for record in similar],
            "metadata": {"time": datetime.utcnow().isoformat(), "session_id": session_id, "specialists": SPECIALISTS},
            "device": DEVICE,
        },
    )


@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate) -> Dict[str, str]:
    update = update.model_copy(update={"event_type": sanitize_string(update.event_type), "result_raw": sanitize_string(update.result_raw) if update.result_raw else None, "user_id": sanitize_string(update.user_id)})
    if update.state_delta:
        existing = store.get_state()
        for key in update.state_delta:
            if key in existing:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
        store.apply_state_delta(update.state_delta)

    record = MemoryRecord(id=str(uuid.uuid4()), timestamp=update.timestamp, event_type=update.event_type, content={"result": update.result_raw, "metrics": update.metrics or {}, "state_changes": update.state_delta or {}}, embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"), relevance=update.relevance)
    version = store.insert(record, user_id=update.user_id)
    audit.write("INFO", f"memory append id={record.id} version={version.version_id} hash={version.commit_hash}")
    return {"status": "appended", "id": record.id, "version": str(version.version_id), "hash": version.commit_hash}


@app.post("/memory/revert")
async def revert_memory(payload: RevertRequest) -> Dict[str, str]:
    try:
        version = store.revert(payload.version_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit.write("WARN", f"memory reverted target={payload.version_id} new_hash={version.commit_hash}")
    return {"status": "reverted", "target_version": str(payload.version_id), "new_version": str(version.version_id), "hash": version.commit_hash}


@app.get("/memory/versions")
async def memory_versions(limit: int = 20) -> List[VersionRef]:
    return store.versions(limit=limit)


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    _ = sanitize_string(query)
    _ = sanitize_string(user_id)
    return store.search(top_k=top_k)


@app.get("/memory/state")
async def memory_state() -> Dict[str, Any]:
    return {"status": "ok", "state": store.get_state()}


@app.get("/memory/stats")
async def memory_stats() -> Dict[str, Any]:
    return {"status": "ok", "records": store.count(), "versions": len(store.versions(limit=100000))}


@app.post("/memory/compress")
async def compress_now() -> Dict[str, Any]:
    return store.compress()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
