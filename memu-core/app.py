from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from common.runtime import AuditStream, ErrorBudget, detect_device, sanitize_string, setup_json_logger
from lakefs_client import LakeFSClient, VersionCommit

logger = setup_json_logger("memu-core", os.getenv("LOG_PATH", "/tmp/memu-core.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Memory Core", version="0.5.0")
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("memu-core", required=os.getenv("AUDIT_REQUIRED", "false").lower()=="true")
last_compress_run = 0.0
MAX_MEMORY_RECORDS = int(os.getenv("MAX_MEMORY_RECORDS", "5000"))
MAX_STATE_KEY_SIZE = int(os.getenv("MAX_STATE_KEY_SIZE", "128"))
MAX_STATE_VALUE_SIZE = int(os.getenv("MAX_STATE_VALUE_SIZE", "4096"))


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
    pin: bool = False


class MemoryRecord(BaseModel):
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    embedding: List[float]
    relevance: float = 1.0
    pinned: bool = False


SPECIALISTS = ["DeepSeek-V4", "Kimi-2.5", "Qwen-VL"]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []
        self._state: Dict[str, Any] = {}
        self._compressed_archive: List[bytes] = []
        self.vc = LakeFSClient()

    def insert(self, record: MemoryRecord) -> VersionCommit:
        branch = self.vc.create_branch("main", f"update-keeper-{int(time.time())}")
        next_records = [*self._records, record]
        if MAX_MEMORY_RECORDS > 0 and len(next_records) > MAX_MEMORY_RECORDS:
            next_records = next_records[-MAX_MEMORY_RECORDS:]
        commit = self.vc.put_branch_state(branch, [r.model_dump() for r in next_records], self._state, f"update: user_id=keeper, ts={int(time.time())}")
        self._records = next_records
        return commit

    def search(self, top_k: int) -> List[MemoryRecord]:
        return list(reversed(self._records))[:top_k]

    def count(self) -> int:
        return len(self._records)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> VersionCommit:
        next_state = {**self._state, **delta}
        branch = self.vc.create_branch("main", f"update-state-{int(time.time())}")
        commit = self.vc.put_branch_state(branch, [r.model_dump() for r in self._records], next_state, "update: user_id=keeper, state delta")
        self._state = next_state
        return commit

    def compress(self) -> Dict[str, Any]:
        threshold = datetime.utcnow() - timedelta(days=90)
        before_bytes = sum(len(r.model_dump_json()) for r in self._records)
        kept: List[MemoryRecord] = []
        archived = 0
        try:
            import zstandard as zstd

            compressor = zstd.ZstdCompressor(level=10)
            use_zstd = True
        except Exception:
            compressor = None
            use_zstd = False

        for record in self._records:
            ts = datetime.fromisoformat(record.timestamp) if "T" in record.timestamp else datetime.utcnow()
            if record.pinned or ts > threshold or record.relevance >= 0.2:
                kept.append(record)
            else:
                blob = record.model_dump_json().encode("utf-8")
                packed = compressor.compress(blob) if use_zstd else blob
                self._compressed_archive.append(packed)
                archived += 1
        before = len(self._records)
        self._records = kept
        after_bytes = sum(len(r.model_dump_json()) for r in self._records)
        target_bytes = int(before_bytes * 0.1)
        saved = max(before_bytes - max(after_bytes, target_bytes), 0)
        logger.info("weekly compression complete, bytes_saved=%s archived=%s", saved, archived)
        return {"before": before, "after": len(self._records), "bytes_saved": saved, "archived": archived}

    def revert(self, commit_id: str) -> None:
        self.vc.revert(commit_id)
        main = self.vc.latest_main()
        self._records = [MemoryRecord.model_validate(r) for r in main["records"]]
        self._state = dict(main["state"])


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




def _similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def retrieve_ranked(query: str, user_id: str, top_k: int) -> List[MemoryRecord]:
    q_emb = generate_embedding(query)
    ranked: List[tuple[float, MemoryRecord]] = []
    for record in store.search(top_k=10_000):
        rid = str(record.content.get("user_id", ""))
        if user_id and rid and user_id != rid:
            continue
        score = _similarity(q_emb, record.embedding) + float(record.relevance)
        if record.pinned:
            score += 0.5
        ranked.append((score, record))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:max(1, min(top_k, 100))]]

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
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
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


def _validate_state_delta_size(delta: Dict[str, Any]) -> None:
    for key, value in delta.items():
        key_len = len(str(key))
        value_len = len(json.dumps(value, ensure_ascii=False))
        if key_len > MAX_STATE_KEY_SIZE:
            raise HTTPException(status_code=400, detail=f"state key too large: {key}")
        if value_len > MAX_STATE_VALUE_SIZE:
            raise HTTPException(status_code=400, detail=f"state value too large for key: {key}")


@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate) -> Dict[str, str]:
    update = update.model_copy(update={"event_type": sanitize_string(update.event_type), "result_raw": sanitize_string(update.result_raw) if update.result_raw else None})
    commit = None
    if update.state_delta:
        existing = store.get_state()
        for key in update.state_delta:
            if key in existing:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
        _validate_state_delta_size(update.state_delta)
        commit = store.apply_state_delta(update.state_delta)

    user_id = sanitize_string(update.user_id)
    pin_default = os.getenv("PIN_KEEPER_DEFAULT", "false").lower() == "true"
    keeper_pin = user_id == "keeper" and (update.pin or pin_default)
    relevance = 1.0 if keeper_pin else update.relevance
    record = MemoryRecord(id=str(uuid.uuid4()), timestamp=update.timestamp, event_type=update.event_type, content={"result": update.result_raw, "metrics": update.metrics or {}, "state_changes": update.state_delta or {}, "user_id": user_id, "pin": keeper_pin}, embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"), relevance=relevance, pinned=keeper_pin)
    record_commit = store.insert(record)
    return {"status": "appended", "id": record.id, "commit": record_commit.commit_id, "state_commit": commit.commit_id if commit else "none"}


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    q = sanitize_string(query)
    uid = sanitize_string(user_id)
    return retrieve_ranked(q, uid, top_k=top_k)


@app.get("/memory/state")
async def memory_state() -> Dict[str, Any]:
    return {"status": "ok", "state": store.get_state()}


@app.get("/memory/stats")
async def memory_stats() -> Dict[str, Any]:
    counts = Counter(record.event_type for record in store.search(top_k=10_000))
    return {"status": "ok", "records": store.count(), "event_types": dict(counts), "commits": [c.__dict__ for c in store.vc.list_commits()[:20]]}


@app.get("/memory/diagnostics")
async def memory_diagnostics() -> Dict[str, Any]:
    counts = Counter(record.event_type for record in store.search(top_k=10_000))
    return {
        "status": "ok",
        "records": store.count(),
        "max_memory_records": MAX_MEMORY_RECORDS,
        "state_limits": {"max_key_size": MAX_STATE_KEY_SIZE, "max_value_size": MAX_STATE_VALUE_SIZE},
        "event_type_counts": dict(counts),
    }


@app.post("/memory/compress")
async def memory_compress() -> Dict[str, Any]:
    return {"status": "ok", **store.compress()}


@app.post("/memory/revert")
@app.post("/revert")
async def memory_revert(version: str = Query(..., description="Commit hash/id")) -> Dict[str, Any]:
    try:
        store.revert(sanitize_string(version))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    chain = hashlib.sha256(json.dumps([c.__dict__ for c in store.vc.list_commits()], sort_keys=True).encode("utf-8")).hexdigest()
    return {"status": "ok", "reverted_to": version, "sha256_chain": chain, "records": store.count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
