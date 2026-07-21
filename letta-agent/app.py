from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from common.runtime import setup_json_logger

# ── ENV ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LETTA_MODEL     = os.getenv("LETTA_MODEL", "qwen2.5:0.5b")
EMBEDDING_MODEL = os.getenv("LETTA_EMBEDDING_MODEL", "all-minilm")
EMBEDDING_DIM   = int(os.getenv("LETTA_EMBEDDING_DIM", "384"))
CONTEXT_WINDOW  = int(os.getenv("LETTA_CONTEXT_WINDOW", "32768"))
PORT            = int(os.getenv("PORT", "8062"))

# Letta reads LETTA_BASE_PATH at import time to locate its SQLite store.
# Set it before any letta import so the agent's working memory lands in our
# named volume instead of ~/.letta (which doesn't exist for the app user).
_data_dir = os.getenv("LETTA_BASE_PATH", "/data/letta")
os.environ.setdefault("LETTA_BASE_PATH", _data_dir)

logger = setup_json_logger("letta-agent", os.getenv("LOG_PATH", "/tmp/letta-agent.json.log"))

# ── Module-level state (not durable across container restarts) ───────
_letta_client: Any = None
_agent_id: Optional[str] = None


def _client() -> tuple:
    """Lazy init: create the Letta client + agent on first call, then cache."""
    global _letta_client, _agent_id
    if _letta_client is not None:
        return _letta_client, _agent_id

    from letta import create_client  # noqa: PLC0415
    from letta.schemas.embedding_config import EmbeddingConfig  # noqa: PLC0415
    from letta.schemas.llm_config import LLMConfig  # noqa: PLC0415

    os.makedirs(_data_dir, exist_ok=True)

    lc = create_client()

    agent_state = lc.create_agent(
        name="kai-memory-agent",
        llm_config=LLMConfig(
            model=LETTA_MODEL,
            model_endpoint_type="ollama",
            model_endpoint=OLLAMA_BASE_URL,
            context_window=CONTEXT_WINDOW,
        ),
        embedding_config=EmbeddingConfig(
            embedding_endpoint_type="ollama",
            embedding_model=EMBEDDING_MODEL,
            embedding_endpoint=OLLAMA_BASE_URL,
            embedding_dim=EMBEDDING_DIM,
            embedding_chunk_size=300,
        ),
    )

    _letta_client = lc
    _agent_id = agent_state.id
    logger.info("letta-agent initialized agent_id=%s model=%s", _agent_id, LETTA_MODEL)
    return _letta_client, _agent_id


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("letta-agent starting on port %d", PORT)
    yield


app = FastAPI(title="letta-agent", version="0.1.0", lifespan=lifespan)


# ── Request models ───────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "letta-agent",
        "agent_id": _agent_id,
        "model": LETTA_MODEL,
    }


@app.post("/agent/run")
async def agent_run(req: RunRequest) -> Dict[str, Any]:
    """Run the Letta agent on a task, optionally enriched with context."""
    try:
        lc, aid = _client()

        message = req.task
        if req.context:
            ctx_str = "; ".join(f"{k}={v}" for k, v in req.context.items())
            message = f"[context: {ctx_str}]\n{req.task}"

        response = lc.send_message(agent_id=aid, role="user", message=message)

        reply_parts: List[str] = []
        memories_updated = False
        for msg in response.messages:
            # AssistantMessage carries the final text reply
            text = getattr(msg, "assistant_message", None) or getattr(msg, "text", None)
            if text:
                reply_parts.append(str(text))
            # Any archival_memory_* tool call means the agent wrote memories
            fn = getattr(getattr(msg, "function_call", None), "name", "") or ""
            if "archival" in fn or "memory" in fn.lower():
                memories_updated = True

        return {
            "response": "\n".join(reply_parts),
            "memories_updated": memories_updated,
            "agent_id": aid,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"agent/run failed: {exc}")


@app.get("/agent/memory/export")
async def memory_export() -> Dict[str, Any]:
    """Export the agent's archival memory as a flat list of strings."""
    try:
        lc, aid = _client()
        passages = lc.get_archival_memory(agent_id=aid, limit=200)
        memories = [p.text for p in passages if getattr(p, "text", None)]
        return {"memories": memories, "count": len(memories)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"memory/export failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
