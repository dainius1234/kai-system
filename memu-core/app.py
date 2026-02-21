from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("memu-core", os.getenv("LOG_PATH", "/tmp/memu-core.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Memory Core", version="0.4.0")
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


class MemoryRecord(BaseModel):
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    embedding: List[float]
    relevance: float = 1.0


SPECIALISTS = ["DeepSeek-V4", "Kimi-2.5", "Qwen-VL"]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []
        self._state: Dict[str, Any] = {}

    def insert(self, record: MemoryRecord) -> None:
        self._records.append(record)

    def search(self, top_k: int) -> List[MemoryRecord]:
        return list(reversed(self._records))[:top_k]

    def count(self) -> int:
        return len(self._records)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> None:
        self._state.update(delta)

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
        logger.info("weekly compression complete, bytes_saved=%s", saved)
        return {"before": before, "after": len(self._records), "bytes_saved": saved}


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
    update = update.model_copy(update={"event_type": sanitize_string(update.event_type), "result_raw": sanitize_string(update.result_raw) if update.result_raw else None})
    if update.state_delta:
        existing = store.get_state()
        for key in update.state_delta:
            if key in existing:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
        store.apply_state_delta(update.state_delta)

    record = MemoryRecord(id=str(uuid.uuid4()), timestamp=update.timestamp, event_type=update.event_type, content={"result": update.result_raw, "metrics": update.metrics or {}, "state_changes": update.state_delta or {}}, embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"), relevance=update.relevance)
    store.insert(record)
    return {"status": "appended", "id": record.id}


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
    counts = Counter(record.event_type for record in store.search(top_k=10_000))
    return {"status": "ok", "records": store.count(), "event_types": dict(counts)}


@app.post("/memory/compress")
async def memory_compress() -> Dict[str, Any]:
    return {"status": "ok", **store.compress()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
