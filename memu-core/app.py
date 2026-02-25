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
    importance: Optional[float] = None  # 0-1 importance score (auto-scored if omitted)
    user_id: str = "keeper"
    pin: bool = False


class NoteRequest(BaseModel):
    """Quick free-text note — the 'jot it down' endpoint."""
    text: str
    category: Optional[str] = None
    user_id: str = "keeper"
    pin: bool = False


class SessionMessage(BaseModel):
    """A single turn in the working-memory session buffer."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = 0.0


class MemoryRecord(BaseModel):
    id: str
    timestamp: str
    event_type: str
    category: str = "general"  # domain category
    content: Dict[str, Any]
    embedding: List[float]
    relevance: float = 1.0
    importance: float = 0.5  # 0-1 importance score (novelty + specificity + keeper)
    access_count: int = 0  # how many times this memory was retrieved (spaced repetition)
    last_accessed: Optional[str] = None  # ISO timestamp of last retrieval
    pinned: bool = False
    # v7 evidence pack fields (populated on retrieval, not stored)
    rank_score: Optional[float] = None  # composite retrieval score
    trust_tier: str = "unverified"  # unverified | PASS | REPAIR | FAIL_CLOSED
    source_id: Optional[str] = None  # originating service or actor
    # quarantine fields
    poisoned: bool = False
    quarantine_reason: Optional[str] = None


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


# ── importance scoring ──────────────────────────────────────────────
# Models how the human hippocampus prioritizes encoding:  novel,
# specific, and emotionally-tagged events are remembered better.

_IMPORTANCE_BOOST_WORDS = {
    "critical", "urgent", "important", "danger", "safety", "ncr",
    "incident", "accident", "deadline", "milestone", "client",
    "defect", "failure", "approval", "sign-off", "handover",
}


def score_importance(text: str, is_keeper: bool = False, is_pinned: bool = False) -> float:
    """Compute importance score 0.0-1.0 for a memory.

    Factors:
      - keyword salience (domain-critical words)
      - specificity (length / detail)
      - keeper flag (keeper's own notes are boosted)
      - pinned flag (explicit pin = max importance)
    """
    if is_pinned:
        return 1.0

    score = 0.3  # baseline

    lower = text.lower()
    hits = sum(1 for w in _IMPORTANCE_BOOST_WORDS if w in lower)
    score += min(hits * 0.1, 0.3)  # up to +0.3 for keywords

    # specificity: longer, more detailed text is more important
    word_count = len(text.split())
    if word_count > 50:
        score += 0.15
    elif word_count > 20:
        score += 0.1
    elif word_count > 10:
        score += 0.05

    # keeper boost
    if is_keeper:
        score += 0.1

    return round(min(score, 1.0), 3)


# ── recency weighting ──────────────────────────────────────────────
# Ebbinghaus forgetting curve: memories decay exponentially with time
# but are rescued by access frequency (spaced repetition).

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "14"))


def _recency_weight(record_timestamp: str, access_count: int = 0) -> float:
    """Compute a 0.0-1.0 recency weight using exponential decay.

    Half-life is configurable (default 14 days).  Each access extends
    the effective age by 10% — spaced repetition effect.
    """
    try:
        if "T" in record_timestamp:
            record_dt = datetime.fromisoformat(record_timestamp.replace("Z", "+00:00"))
        else:
            record_dt = datetime.fromisoformat(record_timestamp)
        age_seconds = max((datetime.now(record_dt.tzinfo) - record_dt).total_seconds(), 0)
    except Exception:
        age_seconds = 0.0

    age_days = age_seconds / 86400.0

    # spaced repetition: each access effectively makes the memory 10% "younger"
    effective_age = age_days / (1.0 + 0.1 * access_count)

    import math
    decay = math.exp(-0.693 * effective_age / max(RECENCY_HALF_LIFE_DAYS, 0.1))
    return round(decay, 4)




def _similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def retrieve_ranked(query: str, user_id: str, top_k: int) -> List[MemoryRecord]:
    """Rank memories using a multi-signal scoring model.

    Signals combined (weighted sum):
      1. Embedding similarity     — semantic match to the query
      2. Relevance score          — original relevance at write time
      3. Importance               — novelty / specificity / keeper
      4. Recency (Ebbinghaus)     — exponential time-decay with spaced-rep
      5. Pin bonus                — pinned memories always surface
      6. Access frequency bonus   — frequently-recalled memories resist decay

    Every record returned has its access_count incremented — the act of
    remembering strengthens the memory, just like in a human brain.
    """
    q_emb = generate_embedding(query)
    ranked: List[tuple[float, MemoryRecord]] = []
    candidates = store.search(top_k=10_000, query=query)
    now_iso = datetime.utcnow().isoformat()

    for record in candidates:
        # skip quarantined records — they never surface in retrieval
        if getattr(record, "poisoned", False):
            continue
        rid = str(record.content.get("user_id", ""))
        if user_id and rid and user_id != rid:
            continue

        sim = _similarity(q_emb, record.embedding)
        recency = _recency_weight(record.timestamp, getattr(record, "access_count", 0))
        importance = getattr(record, "importance", 0.5)

        # weighted combination — tuned so recent + relevant + important = top
        score = (
            sim * 0.35
            + float(record.relevance) * 0.20
            + importance * 0.20
            + recency * 0.20
            + (0.05 if record.pinned else 0.0)
        )

        ranked.append((score, record))

    ranked.sort(key=lambda x: x[0], reverse=True)
    results = [r for _, r in ranked[:max(1, min(top_k, 100))]]

    # attach rank_score to each returned record + bump access count
    for i, (score, _) in enumerate(ranked[:max(1, min(top_k, 100))]):
        results[i].rank_score = round(score, 4)

    # bump access count on retrieved records (spaced repetition)
    for record in results:
        record.access_count = getattr(record, "access_count", 0) + 1
        record.last_accessed = now_iso

    return results

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
    # auto-score importance if not explicitly provided
    importance = update.importance if update.importance is not None else score_importance(
        text_for_classify, is_keeper=(user_id == "keeper"), is_pinned=keeper_pin
    )
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
        importance=importance,
        access_count=0,
        last_accessed=None,
        pinned=keeper_pin,
    )
    record_commit = store.insert(record)
    return {"status": "appended", "id": record.id, "category": category, "commit": record_commit.commit_id, "state_commit": commit.commit_id if commit else "none"}


@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20) -> List[MemoryRecord]:
    q = sanitize_string(query)
    uid = sanitize_string(user_id)
    return retrieve_ranked(q, uid, top_k=top_k)


@app.get("/memory/evidence-pack")
async def evidence_pack(query: str, user_id: str = "keeper",
                        top_k: int = 10) -> Dict[str, Any]:
    """Return a scored evidence pack for the verifier.

    Each record includes its rank_score, trust_tier, and a summary of
    why it was selected. The verifier uses this to decide PASS/REPAIR/FAIL_CLOSED.
    """
    q = sanitize_string(query)
    uid = sanitize_string(user_id)
    records = retrieve_ranked(q, uid, top_k=top_k)

    pack = []
    for rec in records:
        pack.append({
            "id": rec.id,
            "rank_score": rec.rank_score,
            "trust_tier": rec.trust_tier,
            "source_id": rec.source_id or rec.content.get("user_id", "unknown"),
            "category": rec.category,
            "relevance": rec.relevance,
            "importance": rec.importance,
            "pinned": rec.pinned,
            "content": rec.content,
            "timestamp": rec.timestamp,
        })

    return {
        "query": q,
        "pack_size": len(pack),
        "evidence": pack,
    }


# ── Quarantine endpoints ────────────────────────────────────────────


class QuarantineRequest(BaseModel):
    record_id: str
    reason: str = "manual quarantine"


@app.post("/memory/quarantine")
async def quarantine_record(req: QuarantineRequest) -> Dict[str, str]:
    """Mark a memory record as poisoned — excluded from all future retrieval."""
    all_records = store.search(top_k=10_000)
    for rec in all_records:
        if rec.id == req.record_id:
            rec.poisoned = True
            rec.quarantine_reason = req.reason
            logger.info("quarantined record=%s reason=%s", req.record_id, req.reason)
            return {"status": "quarantined", "id": req.record_id, "reason": req.reason}
    raise HTTPException(status_code=404, detail=f"record {req.record_id} not found")


@app.post("/memory/quarantine/clear")
async def clear_quarantine(req: QuarantineRequest) -> Dict[str, str]:
    """Remove quarantine flag from a record, restoring it to retrieval."""
    all_records = store.search(top_k=10_000)
    for rec in all_records:
        if rec.id == req.record_id:
            rec.poisoned = False
            rec.quarantine_reason = None
            logger.info("cleared quarantine record=%s", req.record_id)
            return {"status": "cleared", "id": req.record_id}
    raise HTTPException(status_code=404, detail=f"record {req.record_id} not found")


@app.get("/memory/quarantine/list")
async def list_quarantined() -> Dict[str, Any]:
    """List all quarantined (poisoned) records."""
    all_records = store.search(top_k=10_000)
    quarantined = [
        {"id": r.id, "reason": r.quarantine_reason, "timestamp": r.timestamp,
         "event_type": r.event_type, "category": r.category}
        for r in all_records if getattr(r, "poisoned", False)
    ]
    return {"count": len(quarantined), "quarantined": quarantined}


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


# ═══════════════════════════════════════════════════════════════════════
#  SESSION BUFFER — Working Memory (prefrontal cortex analogy)
#
#  A per-session sliding window of recent conversation turns stored in
#  Redis (or in-memory fallback).  This is the "what were we just
#  talking about?" context that makes the AI feel continuous.
#
#  - Each session has a message list capped at SESSION_MAX_TURNS
#  - Sessions expire after SESSION_TTL_SECONDS of inactivity
#  - The buffer is injected into every LLM prompt so the model sees
#    the full conversation history within the current session
# ═══════════════════════════════════════════════════════════════════════

SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "40"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

# session store: Redis when available, dict fallback
_session_store: Dict[str, List[Dict[str, Any]]] = {}
_session_timestamps: Dict[str, float] = {}
_redis_client = None

try:
    import redis as _redis_module
    _redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    _redis_client = _redis_module.from_url(_redis_url, decode_responses=True)
    _redis_client.ping()
    logger.info("Session buffer using Redis at %s", _redis_url)
except Exception:
    _redis_client = None
    logger.info("Session buffer using in-memory fallback (Redis unavailable)")


def _session_key(session_id: str) -> str:
    return f"session:{session_id}:messages"


def _get_session_messages(session_id: str) -> List[Dict[str, Any]]:
    """Read the session buffer — last N turns for this session."""
    if _redis_client:
        try:
            raw = _redis_client.lrange(_session_key(session_id), 0, SESSION_MAX_TURNS - 1)
            return [json.loads(r) for r in raw]
        except Exception:
            pass
    # fallback: in-memory
    _cleanup_expired_sessions()
    return list(_session_store.get(session_id, []))[-SESSION_MAX_TURNS:]


def _append_session_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session buffer."""
    msg = {"role": role, "content": content, "timestamp": time.time()}
    if _redis_client:
        try:
            key = _session_key(session_id)
            _redis_client.rpush(key, json.dumps(msg))
            _redis_client.ltrim(key, -SESSION_MAX_TURNS, -1)
            _redis_client.expire(key, SESSION_TTL_SECONDS)
            return
        except Exception:
            pass
    # fallback: in-memory
    buf = _session_store.setdefault(session_id, [])
    buf.append(msg)
    if len(buf) > SESSION_MAX_TURNS:
        _session_store[session_id] = buf[-SESSION_MAX_TURNS:]
    _session_timestamps[session_id] = time.time()


def _cleanup_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, ts in _session_timestamps.items() if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        _session_store.pop(sid, None)
        _session_timestamps.pop(sid, None)


@app.get("/session/{session_id}")
async def get_session(session_id: str) -> Dict[str, Any]:
    """Retrieve the working memory buffer for a session."""
    sid = sanitize_string(session_id)
    messages = _get_session_messages(sid)
    return {"status": "ok", "session_id": sid, "turns": len(messages), "messages": messages}


@app.post("/session/{session_id}/append")
async def append_to_session(session_id: str, msg: SessionMessage) -> Dict[str, Any]:
    """Append a user or assistant turn to the session buffer."""
    sid = sanitize_string(session_id)
    role = sanitize_string(msg.role)
    if role not in {"user", "assistant", "system"}:
        raise HTTPException(status_code=400, detail="role must be user, assistant, or system")
    _append_session_message(sid, role, msg.content)
    messages = _get_session_messages(sid)
    return {"status": "ok", "session_id": sid, "turns": len(messages)}


@app.delete("/session/{session_id}")
async def clear_session(session_id: str) -> Dict[str, Any]:
    """Clear the working memory for a session (start fresh)."""
    sid = sanitize_string(session_id)
    if _redis_client:
        try:
            _redis_client.delete(_session_key(sid))
        except Exception:
            pass
    _session_store.pop(sid, None)
    _session_timestamps.pop(sid, None)
    return {"status": "ok", "session_id": sid, "cleared": True}


@app.get("/session/{session_id}/context")
async def session_context(session_id: str, query: str = "", top_k: int = 5) -> Dict[str, Any]:
    """Build a complete context payload for an LLM call.

    Assembles:
      1. System persona
      2. Relevant long-term memories (vector-ranked)
      3. Session buffer (recent conversation turns)
      4. Current user query

    This is what gets injected into the LLM prompt — the full picture
    from working memory + long-term memory combined.
    """
    sid = sanitize_string(session_id)
    q = sanitize_string(query)

    # long-term memories
    long_term = retrieve_ranked(q, "keeper", top_k=top_k) if q else []

    # session buffer (working memory)
    session_msgs = _get_session_messages(sid)

    # format long-term memories for prompt injection
    ltm_context = []
    for rec in long_term:
        age_label = rec.timestamp[:10] if rec.timestamp else "unknown"
        content_text = rec.content.get("result", str(rec.content))
        ltm_context.append(f"[{age_label} | {rec.category} | importance={rec.importance}] {content_text}")

    return {
        "status": "ok",
        "long_term_memories": ltm_context,
        "long_term_count": len(ltm_context),
        "session_messages": session_msgs,
        "session_turns": len(session_msgs),
        "query": q,
    }


# ═══════════════════════════════════════════════════════════════════════
#  QUICK NOTES — "Jot it down" endpoint
#
#  For the keeper to quickly save a free-text observation on site
#  without needing structured event_type / metrics fields.  Auto-
#  classifies into construction category automatically.
# ═══════════════════════════════════════════════════════════════════════

@app.post("/memory/note")
async def quick_note(note: NoteRequest) -> Dict[str, str]:
    """Save a free-text note to long-term memory."""
    text = sanitize_string(note.text)
    if not text:
        raise HTTPException(status_code=400, detail="Note text cannot be empty")

    user_id = sanitize_string(note.user_id)
    category = note.category or classify_category(text)
    importance = score_importance(text, is_keeper=(user_id == "keeper"), is_pinned=note.pin)

    record = MemoryRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        event_type="note",
        category=category,
        content={"result": text, "user_id": user_id, "pin": note.pin},
        embedding=generate_embedding(text),
        relevance=0.8,
        importance=importance,
        access_count=0,
        last_accessed=None,
        pinned=note.pin,
    )
    commit = store.insert(record)
    return {"status": "noted", "id": record.id, "category": category, "importance": str(importance), "commit": commit.commit_id}


# ═══════════════════════════════════════════════════════════════════════
#  REFLECTION / CONSOLIDATION — "Sleep" endpoint
#
#  Inspired by how the human brain consolidates memories during sleep:
#  reviews recent memories, detects recurring patterns and themes, and
#  writes high-importance insight summaries back into long-term memory.
#
#  Call this periodically (e.g., nightly cron, or end-of-shift).
# ═══════════════════════════════════════════════════════════════════════

REFLECTION_WINDOW_DAYS = int(os.getenv("REFLECTION_WINDOW_DAYS", "7"))
MIN_MEMORIES_FOR_REFLECTION = int(os.getenv("MIN_MEMORIES_FOR_REFLECTION", "5"))


@app.post("/memory/reflect")
async def reflect() -> Dict[str, Any]:
    """Consolidate recent memories into pattern insights.

    Scans the last REFLECTION_WINDOW_DAYS of memories, finds:
      1. Recurring categories — what topics keep coming up?
      2. Frequently accessed memories — what matters most?
      3. Keyword clusters — emerging themes across notes

    Writes insight summaries back as high-importance pinned memories
    so the system "learns" from its own experience over time.
    """
    threshold = datetime.utcnow() - timedelta(days=REFLECTION_WINDOW_DAYS)
    all_records = store.search(top_k=10_000)

    # filter to recent window
    recent: List[MemoryRecord] = []
    for r in all_records:
        try:
            ts = datetime.fromisoformat(r.timestamp.replace("Z", "+00:00")) if "T" in r.timestamp else datetime.fromisoformat(r.timestamp)
            if ts.replace(tzinfo=None) >= threshold:
                recent.append(r)
        except Exception:
            recent.append(r)  # if we can't parse, include it

    if len(recent) < MIN_MEMORIES_FOR_REFLECTION:
        return {"status": "skipped", "reason": f"only {len(recent)} recent memories (need {MIN_MEMORIES_FOR_REFLECTION})", "insights": []}

    # 1. Category distribution — what are we thinking about most?
    cat_counts = Counter(getattr(r, "category", "general") for r in recent)
    top_categories = cat_counts.most_common(5)

    # 2. Most-accessed memories — what keeps being recalled?
    by_access = sorted(recent, key=lambda r: getattr(r, "access_count", 0), reverse=True)
    most_accessed = by_access[:5]

    # 3. Keyword frequency — emerging themes
    all_text = " ".join(str(r.content.get("result", "")) for r in recent).lower()
    import re as _re
    words = _re.findall(r"\b[a-z]{4,}\b", all_text)
    # filter out common stop words
    stop = {"that", "this", "with", "from", "have", "been", "were", "they", "their", "will",
            "would", "could", "should", "about", "which", "there", "these", "those", "what",
            "when", "where", "some", "than", "then", "into", "also", "just", "more", "very",
            "only", "over", "such", "after", "before", "other", "each", "most", "none", "like"}
    filtered = [w for w in words if w not in stop]
    word_freq = Counter(filtered).most_common(15)

    # build insight summaries
    insights: List[str] = []

    if top_categories:
        cat_summary = ", ".join(f"{cat} ({cnt})" for cat, cnt in top_categories)
        insights.append(f"Focus areas this week: {cat_summary}")

    if most_accessed:
        access_texts = [str(r.content.get("result", ""))[:80] for r in most_accessed[:3] if r.content.get("result")]
        if access_texts:
            insights.append(f"Most revisited topics: {'; '.join(access_texts)}")

    if word_freq:
        theme_words = ", ".join(w for w, _ in word_freq[:10])
        insights.append(f"Emerging keyword themes: {theme_words}")

    # write insights back as high-importance memories
    written_ids = []
    for insight in insights:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            event_type="reflection",
            category="general",
            content={"result": insight, "user_id": "system", "pin": False, "reflection_window_days": REFLECTION_WINDOW_DAYS, "source_count": len(recent)},
            embedding=generate_embedding(insight),
            relevance=0.9,
            importance=0.85,
            access_count=0,
            last_accessed=None,
            pinned=False,
        )
        store.insert(record)
        written_ids.append(record.id)

    return {
        "status": "ok",
        "window_days": REFLECTION_WINDOW_DAYS,
        "memories_analyzed": len(recent),
        "insights": insights,
        "insight_ids": written_ids,
        "top_categories": dict(top_categories),
        "keyword_themes": [w for w, _ in word_freq[:10]],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
