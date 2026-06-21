"""memu-graph — standalone graph-memory layer (Phase A).

Wraps Cognee (picked over Graphiti in DECISIONS.md D29: Cognee owns its own
Kuzu fork, `ladybug`, avoiding the upstream-Kuzu-deprecation risk found in
Graphiti, and has no Python-version gate on its embedded no-extra-server
option) behind a small HTTP API so memu-core never imports Cognee directly.

This is Phase A only (see kai-pm/MEMORY_GRAPH_DESIGN.md): a standalone
service, verifiable on its own via curl. memu-core does NOT call this
service yet — that wiring is Phase B/C/D, not done here.

Single-user system: multi-tenant access control and session caching (both
on by default in Cognee) are disabled at startup, not left as surprise
defaults.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.runtime import setup_json_logger

logger = setup_json_logger("memu-graph", os.getenv("LOG_PATH", "/tmp/memu-graph.json.log"))

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
os.environ.setdefault("CACHING", "false")
os.environ.setdefault("TELEMETRY_DISABLED", "true")

DATASET_NAME = os.getenv("MEMU_GRAPH_DATASET", "memu")
DEFAULT_TOP_K = int(os.getenv("MEMU_GRAPH_DEFAULT_TOP_K", "10"))

# id -> {"data_id": str, "dataset_id": str} so /graph/forget has something
# to delete against. In-memory only for Phase A — lost on restart. Phase D
# (wiring MARS's delete path) should reconsider whether this needs to be
# durable; tracked as an open question in MEMORY_GRAPH_DESIGN.md.
_source_index: Dict[str, Dict[str, str]] = {}


class IngestRequest(BaseModel):
    text: str
    source_id: str
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ForgetRequest(BaseModel):
    source_id: str


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("memu-graph starting up, dataset=%s", DATASET_NAME)
    yield


app = FastAPI(title="memu-graph", version="0.1.0", lifespan=lifespan)


def _cognee():
    import cognee

    return cognee


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "memu-graph",
        "dataset": DATASET_NAME,
        "indexed_sources": len(_source_index),
    }


@app.post("/graph/ingest")
async def graph_ingest(req: IngestRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    cognee = _cognee()
    node_set = [req.source_id]
    if req.category:
        node_set.append(req.category)

    try:
        add_result = await cognee.add(
            req.text,
            dataset_name=DATASET_NAME,
            node_set=node_set,
        )
        await cognee.cognify(datasets=[DATASET_NAME])
    except Exception as exc:  # noqa: BLE001 — best-effort ingest, caller decides how to treat failure
        logger.warning("graph_ingest failed for source_id=%s: %s", req.source_id, exc)
        raise HTTPException(status_code=502, detail=f"graph ingest failed: {exc}") from exc

    data_id = None
    dataset_id = None
    if isinstance(add_result, list) and add_result:
        first = add_result[0]
        data_id = str(getattr(first, "id", "") or getattr(first, "data_id", "") or "")
        dataset_id = str(getattr(first, "dataset_id", "") or "")
    if data_id:
        _source_index[req.source_id] = {"data_id": data_id, "dataset_id": dataset_id or ""}

    return {"status": "ingested", "source_id": req.source_id, "data_id": data_id}


@app.get("/graph/query")
async def graph_query(q: str, top_k: int = DEFAULT_TOP_K, query_type: str = "GRAPH_COMPLETION") -> Dict[str, Any]:
    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    cognee = _cognee()
    from cognee.modules.search.types import SearchType

    try:
        search_type = SearchType[query_type]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"unknown query_type: {query_type}") from exc

    try:
        results = await cognee.search(
            query_text=q,
            query_type=search_type,
            datasets=[DATASET_NAME],
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001 — caller (memu-core proxy) treats this as "graph unavailable"
        logger.warning("graph_query failed for q=%r: %s", q, exc)
        raise HTTPException(status_code=502, detail=f"graph query failed: {exc}") from exc

    return {"query": q, "query_type": query_type, "results": results}


@app.post("/graph/forget")
async def graph_forget(req: ForgetRequest) -> Dict[str, Any]:
    entry = _source_index.pop(req.source_id, None)
    if entry is None or not entry.get("data_id"):
        return {"status": "not_found", "source_id": req.source_id}

    cognee = _cognee()
    import uuid as _uuid

    try:
        await cognee.delete(
            data_id=_uuid.UUID(entry["data_id"]),
            dataset_id=_uuid.UUID(entry["dataset_id"]) if entry.get("dataset_id") else _uuid.UUID(int=0),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, MARS pruning should not fail on this
        logger.warning("graph_forget failed for source_id=%s: %s", req.source_id, exc)
        return {"status": "error", "source_id": req.source_id, "detail": str(exc)}

    return {"status": "forgotten", "source_id": req.source_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8061")))
