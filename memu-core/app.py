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


class RoutingResponse(BaseModel):
    specialist: str
    context_payload: Dict[str, object]


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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "entries": str(len(store.entries)),
        "time": str(time.time()),
    }


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


@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate) -> Dict[str, str]:
    if update.state_delta:
        seen = set()
        for key in update.state_delta:
            if key in seen:
                raise HTTPException(status_code=400, detail=f"Duplicate key in state_delta: {key}")
            seen.add(key)

    entry = MemoryEntry(
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
