from __future__ import annotations

import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Sovereign Memory Core", version="0.2.0")

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("No GPU — running on CPU only")

MAX_MEMORY_RECORDS = int(os.getenv("MAX_MEMORY_RECORDS", "5000"))
MAX_STATE_KEY_LEN = int(os.getenv("MAX_STATE_KEY_LEN", "128"))
MAX_STATE_VALUE_LEN = int(os.getenv("MAX_STATE_VALUE_LEN", "2048"))


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


class MemoryRecord(BaseModel):
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, Any]
    embedding: List[float]


SPECIALISTS = ["DeepSeek-V4", "Kimi-2.5", "Qwen-VL"]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []
        self._state: Dict[str, Any] = {}

    def insert(self, record: MemoryRecord) -> None:
        self._records.append(record)
        if len(self._records) > MAX_MEMORY_RECORDS:
            self._records = self._records[-MAX_MEMORY_RECORDS:]

    def search(self, query: str, top_k: int) -> List[MemoryRecord]:
        return list(reversed(self._records))[:top_k]

    def count(self) -> int:
        return len(self._records)

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> None:
        self._state.update(delta)

    def diagnostics(self) -> Dict[str, Any]:
        by_event = Counter([item.event_type for item in self._records])
        return {
            "records": len(self._records),
            "state_keys": len(self._state),
            "by_event_type": dict(by_event),
        }


store = InMemoryVectorStore()


def generate_embedding(text: str) -> List[float]:
    seed = sum(bytearray(text.encode("utf-8"))) % 100
    return [seed / 100.0 for _ in range(8)]


def extract_tags(query: str) -> List[str]:
    return [token.strip(".,!?;:") for token in query.lower().split()[:5]]


def select_specialist(query: str) -> str:
    if any(keyword in query.lower() for keyword in ["image", "vision", "camera", "diagram"]):
        return "Qwen-VL"
    if any(keyword in query.lower() for keyword in ["plan", "reason", "policy", "risk"]):
        return "DeepSeek-V4"
    return "Kimi-2.5"


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "storage": os.getenv("VECTOR_STORE", "memory"),
        "device": DEVICE,
    }


@app.post("/route", response_model=RoutingResponse)
async def route_request(request: MemoryRequest) -> RoutingResponse:
    similar = store.search(request.query, top_k=50)
    metadata = {
        "time": datetime.utcnow().isoformat(),
        "session_id": request.session_id,
        "tags": extract_tags(request.query),
        "specialists": SPECIALISTS,
        "device": DEVICE,
    }
    specialist = select_specialist(request.query)
    return RoutingResponse(
        specialist=specialist,
        context_payload={
            "query": request.query,
            "memory_vectors": [record.embedding for record in similar],
            "metadata": metadata,
            "device": DEVICE,
        },
    )


@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate) -> Dict[str, str]:
    if update.state_delta:
        existing = store.get_state()
        for key, value in update.state_delta.items():
            if key in existing:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
            if len(str(key)) > MAX_STATE_KEY_LEN:
                raise HTTPException(status_code=400, detail=f"state key too long: {key}")
            if len(str(value)) > MAX_STATE_VALUE_LEN:
                raise HTTPException(status_code=400, detail=f"state value too long for key: {key}")
        store.apply_state_delta(update.state_delta)

    record = MemoryRecord(
        id=str(uuid.uuid4()),
        timestamp=update.timestamp,
        event_type=update.event_type,
        content={
            "result": update.result_raw,
            "metrics": update.metrics or {},
            "state_changes": update.state_delta or {},
        },
        embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"),
    )
    store.insert(record)
    return {"status": "appended", "id": record.id}


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    return store.search(query, top_k=top_k)


@app.get("/memory/state")
async def memory_state() -> Dict[str, Any]:
    return {"status": "ok", "state": store.get_state()}


@app.get("/memory/stats")
async def memory_stats() -> Dict[str, Any]:
    return {
        "status": "ok",
        "records": store.count(),
        "max_records": MAX_MEMORY_RECORDS,
    }


@app.get("/memory/diagnostics")
async def memory_diagnostics() -> Dict[str, Any]:
    return {"status": "ok", **store.diagnostics()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
