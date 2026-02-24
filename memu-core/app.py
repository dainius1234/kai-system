"""memU core — file-based agent memory engine.

The cognitive heart of the sovereign AI system.  Turns logs, sessions,
and interactions into persistent, queryable memory with vector search.
Replaces OpenClaw's executor-centric approach with long-term memory
that remembers everything, builds context, and stays proactive.

This service IS the orchestrator: it routes decisions to the right
LLM specialist, manages memory retrieval, and handles state.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from common.runtime import AuditStream, ErrorBudget, detect_device, sanitize_string, setup_json_logger

# lakefs client is optional; provide a simple in-memory stub if unavailable
try:
    from lakefs_client import LakeFSClient, VersionCommit
except Exception:  # pragma: no cover - stub if package missing
    class VersionCommit:
        def __init__(self, commit_id: str = ""):
            self.commit_id = commit_id

    class LakeFSClient:
        def __init__(self) -> None:
            self._commits: list[dict] = []

        def create_branch(self, src: str, name: str) -> str:
            return name

        def put_branch_state(self, branch: str, records: list, state: dict, msg: str) -> VersionCommit:
            cid = f"commit-{len(self._commits)}"
            self._commits.append({"branch": branch, "records": records, "state": state, "msg": msg, "id": cid})
            return VersionCommit(commit_id=cid)

        def list_commits(self) -> list:
            return self._commits

        def revert(self, commit_id: str) -> None:
            # no-op stub
            pass

        def latest_main(self) -> dict:
            if self._commits:
                latest = self._commits[-1]
                return {"records": latest.get("records", []), "state": latest.get("state", {})}
            return {"records": [], "state": {}}

logger = setup_json_logger("memu-core", os.getenv("LOG_PATH", "/tmp/memu-core.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="memU — core memory engine", version="0.6.0")
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
    category: Optional[str] = None  # UK construction domain category (auto-classified if omitted)
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
    category: str = "general"  # domain category
    content: Dict[str, Any]
    embedding: List[float]
    relevance: float = 1.0
    pinned: bool = False


# ── UK construction domain categories ───────────────────────────────
# memU auto-classifies incoming records when no explicit category is
# provided.  This turns free-form logs into structured, queryable data
# so you can ask "last week's setting-out on Grid B" and it pulls
# everything relevant.

CONSTRUCTION_CATEGORIES: Dict[str, List[str]] = {
    "setting-out":   ["setting out", "set out", "peg", "grid", "baseline", "offset", "coord", "station", "benchmark", "control point"],
    "survey-data":   ["survey", "level", "total station", "gps", "rtk", "theodolite", "traverse", "elevation", "datum", "topographic"],
    "rams":          ["rams", "risk assessment", "method statement", "hazard", "coshh", "ppe", "permit to work", "safe system"],
    "itp":           ["itp", "inspection", "test plan", "hold point", "witness point", "quality check", "ncr", "snag", "defect"],
    "drawings":      ["drawing", "cad", "autocad", "lisp", "dwg", "revision", "rfi", "design", "detail", "section", "elevation drawing"],
    "hs-briefings":  ["briefing", "toolbox talk", "safety", "h&s", "health and safety", "induction", "near miss", "accident", "incident"],
    "client-ncrs":   ["ncr", "non-conformance", "non conformance", "client complaint", "deficiency", "remedial", "corrective action"],
    "daily-logs":    ["daily log", "daily report", "site diary", "progress", "weather", "labour", "plant", "material delivery"],
}
DEFAULT_CATEGORY = "general"


def classify_category(text: str) -> str:
    """Auto-classify text into a construction domain category.

    Scans the combined event_type + content for domain keywords.
    Returns the best-matching category or 'general' if nothing fits.
    """
    lower = text.lower()
    best_cat = DEFAULT_CATEGORY
    best_hits = 0
    for cat, keywords in CONSTRUCTION_CATEGORIES.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits > best_hits:
            best_hits = hits
            best_cat = cat
    return best_cat


# LLM specialists available for routing.  PUB mode defaults to Dolphin
# (uncensored).  WORK mode routes to DeepSeek-V4 or Kimi-2.5 based on
# task type.  Memu picks the right one; executor only runs what memu says.
SPECIALISTS = ["DeepSeek-V4", "Kimi-2.5", "Dolphin"]


@runtime_checkable
class VectorStore(Protocol):
    """Contract that every memory backend must satisfy.

    Adding a new backend (FAISS, SQLite, etc.)? Implement this Protocol
    and the linter will tell you if you missed anything.
    """

    def insert(self, record: MemoryRecord) -> VersionCommit: ...
    def search(self, top_k: int, query: Optional[str] = None) -> List[MemoryRecord]: ...
    def count(self) -> int: ...
    def get_state(self) -> Dict[str, Any]: ...
    def apply_state_delta(self, delta: Dict[str, Any]) -> VersionCommit: ...
    def compress(self) -> Dict[str, Any]: ...
    def revert(self, commit_id: str) -> None: ...


# persistent vector store using PostgreSQL + pgvector
class PGVectorStore:
    def __init__(self) -> None:
        import psycopg2
        from psycopg2.extras import Json as _Json

        self._Json = _Json
        self.conn = psycopg2.connect(os.getenv("PG_URI", "postgresql://keeper:localdev@postgres:5432/sovereign"))
        self._init_schema()
        self.vc = LakeFSClient()
        self._state: Dict[str, Any] = {}

    def _init_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id text PRIMARY KEY,
                    timestamp text,
                    event_type text,
                    content jsonb,
                    embedding vector,
                    relevance float,
                    pinned bool
                );
                """
            )
        self.conn.commit()

    def insert(self, record: MemoryRecord) -> VersionCommit:
        # store record in Postgres and enforce retention
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (id, timestamp, event_type, content, embedding, relevance, pinned)\
                 VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING",
                (
                    record.id,
                    record.timestamp,
                    record.event_type,
                    self._Json(record.content),
                    record.embedding,
                    record.relevance,
                    record.pinned,
                ),
            )
        self.conn.commit()
        # trim oldest entries if over limit
        if MAX_MEMORY_RECORDS > 0:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id FROM memories ORDER BY timestamp ASC")
                ids = [r[0] for r in cur.fetchall()]
            if len(ids) > MAX_MEMORY_RECORDS:
                to_delete = ids[: len(ids) - MAX_MEMORY_RECORDS]
                with self.conn.cursor() as cur:
                    cur.execute("DELETE FROM memories WHERE id = ANY(%s)", (to_delete,))
                self.conn.commit()
        # commit to version control for state awareness, using same pattern as memory store
        branch = self.vc.create_branch("main", f"update-keeper-{int(time.time())}")
        commit = self.vc.put_branch_state(branch, [], self._state, f"update: user_id=keeper, ts={int(time.time())}")
        return commit

    def search(self, top_k: int, query: Optional[str] = None) -> List[MemoryRecord]:
        # if query is provided we perform embedding similarity, otherwise return
        # the most recent records (by timestamp) up to top_k.
        if query is None:
            with self.conn.cursor() as cur:
                cur.execute("SELECT id, timestamp, event_type, content, embedding, relevance, pinned FROM memories ORDER BY timestamp DESC LIMIT %s", (top_k,))
                rows = cur.fetchall()
        else:
            emb = generate_embedding(query)
            with self.conn.cursor() as cur:
                # using pgvector similarity operator <=>
                cur.execute(
                    "SELECT id, timestamp, event_type, content, embedding, relevance, pinned "
                    "FROM memories ORDER BY embedding <=> %s LIMIT %s",
                    (emb, top_k),
                )
                rows = cur.fetchall()
        result: List[MemoryRecord] = []
        for r in rows:
            content = r[3] if isinstance(r[3], dict) else json.loads(r[3])
            result.append(
                MemoryRecord(
                    id=r[0],
                    timestamp=r[1],
                    event_type=r[2],
                    content=content,
                    embedding=list(r[4]),
                    relevance=r[5],
                    pinned=r[6],
                )
            )
        return result

    def count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            return cur.fetchone()[0]

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def apply_state_delta(self, delta: Dict[str, Any]) -> VersionCommit:
        next_state = {**self._state, **delta}
        branch = self.vc.create_branch("main", f"update-state-{int(time.time())}")
        commit = self.vc.put_branch_state(branch, [], next_state, "update: user_id=keeper, state delta")
        self._state = next_state
        return commit

    def compress(self) -> Dict[str, Any]:
        # compression logic can remain in memory or be no-op
        return {"before": 0, "after": 0, "bytes_saved": 0, "archived": 0}

    def revert(self, commit_id: str) -> None:
        self.vc.revert(commit_id)
        main = self.vc.latest_main()
        self._state = dict(main["state"])


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

    def search(self, top_k: int, query: Optional[str] = None) -> List[MemoryRecord]:
        # In-memory: query param is accepted for interface compatibility but
        # similarity search is not meaningful with fake embeddings.  Returns
        # most-recent records, matching PGVectorStore's fallback behaviour.
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


# choose store implementation based on configuration
store: VectorStore
if os.getenv("VECTOR_STORE", "memory") == "postgres":
    logger.info("Using Postgres-backed vector store")
    store = PGVectorStore()
else:
    store = InMemoryVectorStore()


def generate_embedding(text: str) -> List[float]:
    """Generate a semantic embedding for *text*.

    Uses ``sentence-transformers`` when available (real 384-dim vectors
    with ``all-MiniLM-L6-v2`` by default).  Falls back to a deterministic
    hash-based pseudo-embedding (8-dim) when the library is not installed
    — keeps CI and lightweight tests running without the heavy dependency.
    """
    return _embedding_backend(text)


# ── embedding backend (loaded once at import time) ──────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

try:
    from sentence_transformers import SentenceTransformer as _ST

    _st_model = _ST(EMBEDDING_MODEL_NAME)
    logger.info("sentence-transformers loaded — model=%s  dim=%d", EMBEDDING_MODEL_NAME, _st_model.get_sentence_embedding_dimension())

    def _embedding_backend(text: str) -> List[float]:
        vec = _st_model.encode(text, show_progress_bar=False)
        return vec.tolist()

except Exception:  # pragma: no cover
    logger.warning("sentence-transformers not available — using hash-based fake embeddings")

    def _embedding_backend(text: str) -> List[float]:
        # deterministic pseudo-embedding: SHA-256 → 8 floats in [0,1)
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in h[:8]]


def select_specialist(query: str, mode: str = "WORK") -> str:
    """Pick the right LLM based on mode and query content.

    PUB mode (chill/uncensored):  Always routes to Dolphin.
    WORK mode (gated execution):  Routes to DeepSeek-V4 for reasoning/code
                                  tasks, Kimi-2.5 for general or multimodal.
    """
    if mode == "PUB":
        return "Dolphin"

    q = query.lower()
    if any(k in q for k in ["code", "plan", "reason", "policy", "risk", "debug", "build"]):
        return "DeepSeek-V4"
    return "Kimi-2.5"




def _similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def retrieve_ranked(query: str, user_id: str, top_k: int) -> List[MemoryRecord]:
    q_emb = generate_embedding(query)
    ranked: List[tuple[float, MemoryRecord]] = []
    candidates = store.search(top_k=10_000, query=query)
    for record in candidates:
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
    # auto-classify into construction domain category if not provided
    text_for_classify = f"{update.event_type} {update.result_raw or ''}"
    category = update.category or classify_category(text_for_classify)
    record = MemoryRecord(
        id=str(uuid.uuid4()),
        timestamp=update.timestamp,
        event_type=update.event_type,
        category=category,
        content={
            "result": update.result_raw,
            "metrics": update.metrics or {},
            "state_changes": update.state_delta or {},
            "user_id": user_id,
            "pin": keeper_pin,
        },
        embedding=generate_embedding(f"{update.event_type}: {update.result_raw}"),
        relevance=relevance,
        pinned=keeper_pin,
    )
    record_commit = store.insert(record)
    return {"status": "appended", "id": record.id, "category": category, "commit": record_commit.commit_id, "state_commit": commit.commit_id if commit else "none"}


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    q = sanitize_string(query)
    uid = sanitize_string(user_id)
    return retrieve_ranked(q, uid, top_k=top_k)


@app.get("/memory/state")
async def memory_state() -> Dict[str, Any]:
    return {"status": "ok", "state": store.get_state()}


@app.get("/memory/categories")
async def memory_categories() -> Dict[str, Any]:
    """List known construction domain categories and per-category record counts."""
    all_records = store.search(top_k=10_000)
    cat_counts: Dict[str, int] = {}
    for rec in all_records:
        cat = getattr(rec, "category", "general")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    return {
        "status": "ok",
        "available_categories": list(CONSTRUCTION_CATEGORIES.keys()) + [DEFAULT_CATEGORY],
        "record_counts": cat_counts,
    }


@app.get("/memory/search-by-category")
async def search_by_category(
    category: str,
    query: Optional[str] = None,
    top_k: int = Query(default=20, ge=1, le=200),
) -> List[MemoryRecord]:
    """Retrieve memory records filtered by construction domain category."""
    cat = sanitize_string(category).lower()
    candidates = store.search(top_k=10_000, query=query)
    filtered = [r for r in candidates if getattr(r, "category", "general") == cat]
    if query:
        q_emb = generate_embedding(query)
        scored = [((_similarity(q_emb, r.embedding) + r.relevance), r) for r in filtered]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
    return filtered[:top_k]


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
