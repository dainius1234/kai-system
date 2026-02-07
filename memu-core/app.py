from __future__ import annotations

import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Sovereign Memory Core", version="0.1.0")


class MemoryRequest(BaseModel):
    query: str
    session_id: str
    timestamp: Optional[str] = None
    timestamp: str


class RoutingResponse(BaseModel):
    specialist: str
    context_payload: Dict[str, object]
    context_payload: Dict[str, Any]


class MemoryUpdate(BaseModel):
    timestamp: str
    event_type: str
    task_id: Optional[str] = None
    result_raw: Optional[str] = None
    metrics: Optional[Dict[str, object]] = None
    state_delta: Optional[Dict[str, object]] = None


class MemoryEntry(BaseModel):
    id: str
    timestamp: str
    event_type: str
    content: Dict[str, object]
    session_id: Optional[str] = None
    query: Optional[str] = None


class MemoryRetrieve(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    top_k: int = Field(default=20, ge=1, le=200)


@dataclass
class MemoryStore:
    max_entries: int
    entries: Deque[MemoryEntry]

    def append(self, entry: MemoryEntry) -> None:
        self.entries.append(entry)

    def retrieve(self, query: str, top_k: int) -> List[MemoryEntry]:
        if not query:
            return list(self.entries)[-top_k:]
        matches = [entry for entry in self.entries if entry.query and query.lower() in entry.query.lower()]
        if not matches:
            return list(self.entries)[-top_k:]
        return matches[-top_k:]


store = MemoryStore(max_entries=2000, entries=deque(maxlen=2000))

SPECIALIST_ROUTING = {
    "vision": "Qwen-VL",
    "image": "Qwen-VL",
    "chart": "DeepSeek-V4",
    "analysis": "DeepSeek-V4",
    "research": "Kimi-2.5",
    "strategy": "Kimi-2.5",
}

DEFAULT_SPECIALIST = os.getenv("DEFAULT_SPECIALIST", "DeepSeek-V4")
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

    def search(self, query: str, top_k: int) -> List[MemoryRecord]:
        return list(reversed(self._records))[:top_k]

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> None:
        self._state.update(delta)


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
        "entries": str(len(store.entries)),
        "time": str(time.time()),
    }
    return {"status": "ok", "storage": os.getenv("VECTOR_STORE", "memory")}


@app.post("/route", response_model=RoutingResponse)
async def route_request(request: MemoryRequest) -> RoutingResponse:
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required.")
    lowered = request.query.lower()
    specialist = DEFAULT_SPECIALIST
    for keyword, target in SPECIALIST_ROUTING.items():
        if keyword in lowered:
            specialist = target
            break

    context_payload = {
        "query": request.query,
        "session_id": request.session_id,
        "recent": [entry.model_dump() for entry in store.retrieve(request.query, top_k=5)],
        "routed_at": time.time(),
    }
    return RoutingResponse(specialist=specialist, context_payload=context_payload)
    similar = store.search(request.query, top_k=50)
    metadata = {
        "time": datetime.utcnow().isoformat(),
        "session_id": request.session_id,
        "tags": extract_tags(request.query),
        "specialists": SPECIALISTS,
    }
    specialist = select_specialist(request.query)
    return RoutingResponse(
        specialist=specialist,
        context_payload={
            "query": request.query,
            "memory_vectors": [record.embedding for record in similar],
            "metadata": metadata,
        },
    )


@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate) -> Dict[str, str]:
    if update.state_delta:
        seen = set()
        for key in update.state_delta:
            if key in seen:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
            seen.add(key)

    entry = MemoryEntry(
        existing = store.get_state()
        for key in update.state_delta:
            if key in existing:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
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
    )
    store.append(entry)
    return {"status": "appended", "id": entry.id}


@app.post("/memory/retrieve")
async def retrieve_context(request: MemoryRetrieve) -> List[Dict[str, object]]:
    results = store.retrieve(request.query, top_k=request.top_k)
    return [entry.model_dump() for entry in results]
        embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"),
    )
    store.insert(record)
    return {"status": "appended", "id": record.id}


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    return store.search(query, top_k=top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
