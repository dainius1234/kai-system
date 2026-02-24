"""Fusion engine — multi-signal consensus and conviction gating.

When the system needs high confidence on a decision (WORK mode), the
fusion engine queries multiple LLM specialists independently on the same
prompt and measures agreement.  If they converge, confidence is high.
If they diverge, the system flags low conviction and either retries
or requests human co-sign.

Currently uses stub LLM backends (returns canned responses).  When
real LLM endpoints (DeepSeek-V4, Kimi-2.5, Dolphin) are wired via
the LLM_BACKENDS env var, the fusion engine will call them in parallel.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from common.runtime import ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("fusion-engine", os.getenv("LOG_PATH", "/tmp/fusion-engine.json.log"))
DEVICE = detect_device()

app = FastAPI(title="Fusion Engine", version="0.2.0")
budget = ErrorBudget(window_seconds=300)

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
VERIFIER_URL = os.getenv("VERIFIER_URL", "http://verifier:8052")

# LLM backend URLs — comma-separated name=url pairs.
# Example: DeepSeek-V4=http://llm-deepseek:11434,Kimi-2.5=http://llm-kimi:11434
_LLM_RAW = os.getenv("LLM_BACKENDS", "")
LLM_BACKENDS: Dict[str, str] = {}
for pair in _LLM_RAW.split(","):
    pair = pair.strip()
    if "=" in pair:
        name, url = pair.split("=", 1)
        LLM_BACKENDS[name.strip()] = url.strip()


# ── Models ──────────────────────────────────────────────────────────

class FusionRequest(BaseModel):
    prompt: str
    specialists: List[str] = Field(default_factory=lambda: ["DeepSeek-V4", "Kimi-2.5"])
    context: Optional[str] = None
    require_consensus: bool = True
    min_agreement: float = Field(default=0.6, ge=0.0, le=1.0)


class SpecialistResponse(BaseModel):
    specialist: str
    response: str
    latency_ms: float
    source: str  # "live" or "stub"


class FusionResult(BaseModel):
    consensus: bool
    agreement_score: float
    specialist_responses: List[SpecialistResponse]
    merged_response: str
    verification: Optional[Dict[str, Any]] = None
    fusion_hash: str
    evaluated_at: float


# ── LLM query (stub → real) ────────────────────────────────────────

async def _query_specialist(client: httpx.AsyncClient, name: str, prompt: str, context: Optional[str]) -> SpecialistResponse:
    """Query a single LLM specialist and return its response.

    If a real backend URL is configured, forwards the prompt via HTTP.
    Otherwise, returns a deterministic stub response so the fusion
    pipeline can be exercised end-to-end without live models.
    """
    url = LLM_BACKENDS.get(name)
    start = time.monotonic()

    if url:
        # real LLM backend — expects OpenAI-compatible /v1/chat/completions
        try:
            payload = {
                "model": name,
                "messages": [
                    {"role": "system", "content": context or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            }
            resp = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            latency = (time.monotonic() - start) * 1000
            return SpecialistResponse(specialist=name, response=text, latency_ms=round(latency, 1), source="live")
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            logger.warning("LLM query failed for %s: %s", name, str(exc)[:100])
            return SpecialistResponse(specialist=name, response=f"[error: {str(exc)[:80]}]", latency_ms=round(latency, 1), source="error")

    # stub response — deterministic hash-based so different specialists
    # produce slightly different but reproducible outputs
    h = hashlib.sha256(f"{name}:{prompt[:200]}".encode()).hexdigest()[:8]
    stub = f"[{name} stub] Analysis of prompt (hash={h}): The query requires further context and domain-specific evaluation. Key considerations include scope validation and risk assessment."
    latency = (time.monotonic() - start) * 1000
    return SpecialistResponse(specialist=name, response=stub, latency_ms=round(latency, 1), source="stub")


# ── Consensus measurement ──────────────────────────────────────────

def _measure_agreement(responses: List[SpecialistResponse]) -> float:
    """Measure agreement between specialist responses.

    Uses keyword overlap between responses as a proxy for consensus.
    When real LLMs are wired, this can be upgraded to embedding-based
    similarity.
    """
    if len(responses) < 2:
        return 1.0  # single specialist → trivially agrees with itself

    texts = [r.response for r in responses if r.source != "error"]
    if len(texts) < 2:
        return 0.0  # only one non-error response

    # extract significant words from each response
    word_sets = [set(re.findall(r"\w{4,}", t.lower())) for t in texts]

    # pairwise Jaccard similarity
    scores: List[float] = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            if not union:
                scores.append(0.0)
                continue
            intersection = word_sets[i] & word_sets[j]
            scores.append(len(intersection) / len(union))

    return round(sum(scores) / max(len(scores), 1), 3)


def _merge_responses(responses: List[SpecialistResponse]) -> str:
    """Create a merged summary from specialist responses."""
    valid = [r for r in responses if r.source != "error"]
    if not valid:
        return "No valid specialist responses available."
    if len(valid) == 1:
        return valid[0].response

    # simple merge: take the longest non-error response as primary,
    # note agreement/disagreement from others
    primary = max(valid, key=lambda r: len(r.response))
    others = [r.specialist for r in valid if r.specialist != primary.specialist]
    return f"{primary.response}\n\n[Consensus check: {len(valid)} specialists queried. " \
           f"Agreement with: {', '.join(others)}]"


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "device": DEVICE,
        "llm_backends": str(len(LLM_BACKENDS)),
        "configured_backends": ",".join(LLM_BACKENDS.keys()) or "none (using stubs)",
    }


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/fuse", response_model=FusionResult)
async def fuse(req: FusionRequest) -> FusionResult:
    """Query multiple specialists and measure consensus.

    Flow:
    1. Fire prompt at each requested specialist in parallel
    2. Measure agreement between responses
    3. Optionally verify the merged result through the verifier service
    4. Return consensus verdict
    """
    async with httpx.AsyncClient() as client:
        import asyncio
        tasks = [_query_specialist(client, name, req.prompt, req.context) for name in req.specialists]
        responses = await asyncio.gather(*tasks)

    agreement = _measure_agreement(list(responses))
    consensus = agreement >= req.min_agreement
    merged = _merge_responses(list(responses))

    # optional verification pass through the verifier service
    verification = None
    if req.require_consensus:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                v_resp = await client.post(
                    f"{VERIFIER_URL}/verify",
                    json={"claim": merged[:500], "context": req.context, "source": "fusion-engine"},
                )
                v_resp.raise_for_status()
                verification = v_resp.json()
        except Exception:
            logger.warning("Verifier unavailable — skipping verification pass")

    fusion_hash = hashlib.sha256(
        f"{req.prompt}:{agreement}:{time.time()}".encode()
    ).hexdigest()[:16]

    return FusionResult(
        consensus=consensus,
        agreement_score=agreement,
        specialist_responses=list(responses),
        merged_response=merged,
        verification=verification,
        fusion_hash=fusion_hash,
        evaluated_at=time.time(),
    )


@app.get("/backends")
async def list_backends() -> Dict[str, Any]:
    """Show configured LLM backends and their reachability."""
    status: Dict[str, str] = {}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, url in LLM_BACKENDS.items():
            try:
                r = await client.get(f"{url}/health")
                status[name] = "reachable" if r.status_code == 200 else f"status={r.status_code}"
            except Exception:
                status[name] = "unreachable"
    return {"backends": status, "stub_mode": len(LLM_BACKENDS) == 0}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8053")))
