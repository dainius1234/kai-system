"""Verifier — fact-checking and signal cross-validation.

Accepts a claim plus optional context, runs multiple verification
strategies, and returns a verdict with a confidence score.

Verification strategies:
  1. Memory cross-ref:  check the claim against stored records in memu
  2. Multi-signal consensus:  when real LLMs are wired, query 2+ models
     independently and compare — for now uses heuristic scoring
  3. Self-consistency:  checks whether the plan's own steps are coherent

The verifier never executes anything.  It only reads and reports.
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

logger = setup_json_logger("verifier", os.getenv("LOG_PATH", "/tmp/verifier.json.log"))
DEVICE = detect_device()

app = FastAPI(title="Verifier", version="0.2.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
budget = ErrorBudget(window_seconds=300)

# ── Models ──────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    claim: str
    context: Optional[str] = None
    source: str = "unknown"
    plan: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=10, ge=1, le=100)


class Signal(BaseModel):
    strategy: str
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class VerifyResponse(BaseModel):
    verdict: str          # "supported", "contested", "unverifiable"
    confidence: float     # 0.0 – 1.0
    signals: List[Signal]
    claim_hash: str
    evaluated_at: float


# ── Verification strategies ─────────────────────────────────────────


async def _memory_cross_ref(claim: str, top_k: int) -> Signal:
    """Check whether memu holds records that support the claim."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{MEMU_URL}/memory/retrieve",
                params={"query": claim, "user_id": "keeper", "top_k": top_k},
            )
            resp.raise_for_status()
            records = resp.json()
    except Exception:
        return Signal(strategy="memory_cross_ref", score=0.0, detail="memu unreachable — cannot verify against memory")

    if not records:
        return Signal(strategy="memory_cross_ref", score=0.0, detail="no matching memory records found")

    # score based on how many records relate to the claim (keyword overlap)
    claim_words = set(re.findall(r"\w{3,}", claim.lower()))
    hits = 0
    for rec in records:
        content = rec if isinstance(rec, dict) else {}
        text = str(content.get("content", content.get("result", "")))
        rec_words = set(re.findall(r"\w{3,}", text.lower()))
        overlap = claim_words & rec_words
        if len(overlap) >= max(1, len(claim_words) // 3):
            hits += 1

    ratio = min(hits / max(len(records), 1), 1.0)
    return Signal(
        strategy="memory_cross_ref",
        score=round(ratio, 3),
        detail=f"{hits}/{len(records)} records support the claim (keyword overlap)",
    )


def _self_consistency_check(plan: Optional[Dict[str, Any]]) -> Signal:
    """Verify internal consistency of a plan's steps."""
    if not plan:
        return Signal(strategy="self_consistency", score=0.5, detail="no plan provided — neutral score")

    steps = plan.get("steps", [])
    if not steps:
        return Signal(strategy="self_consistency", score=0.3, detail="plan has no steps")

    # check that steps have required fields and flow logically
    issues: List[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"step {i} is not a dict")
            continue
        if "action" not in step:
            issues.append(f"step {i} missing 'action'")

    # check the plan has a summary
    if not plan.get("summary"):
        issues.append("plan missing summary")

    score = max(1.0 - (len(issues) * 0.2), 0.0)
    detail = "plan is internally consistent" if not issues else f"issues: {'; '.join(issues)}"
    return Signal(strategy="self_consistency", score=round(score, 3), detail=detail)


def _keyword_plausibility(claim: str, context: Optional[str]) -> Signal:
    """Basic heuristic: does the claim contain hedging or absolute language?

    This is a placeholder for real multi-LLM consensus.  When we wire
    DeepSeek-V4 and Kimi-2.5 as verification backends, this will be
    replaced with actual cross-model agreement scoring.
    """
    lower = claim.lower()

    # absolute/suspect patterns get penalised
    suspect = ["always", "never", "guaranteed", "100%", "impossible", "certainly"]
    hedged = ["might", "could", "likely", "probably", "suggests", "appears"]

    suspect_hits = sum(1 for w in suspect if w in lower)
    hedge_hits = sum(1 for w in hedged if w in lower)

    if suspect_hits >= 2:
        score = 0.3
        detail = "claim uses multiple absolute terms — suspicious"
    elif suspect_hits == 1 and hedge_hits == 0:
        score = 0.5
        detail = "claim uses absolute language without hedging"
    elif hedge_hits >= 1:
        score = 0.7
        detail = "claim uses appropriately hedged language"
    else:
        score = 0.6
        detail = "neutral plausibility — no strong signals"

    # boost if context was provided (more data = more verifiable)
    if context and len(context.strip()) > 20:
        score = min(score + 0.1, 1.0)
        detail += "; context provided"

    return Signal(strategy="keyword_plausibility", score=round(score, 3), detail=detail)


# ── Verdict aggregation ─────────────────────────────────────────────


def _aggregate(signals: List[Signal]) -> tuple[str, float]:
    """Combine signal scores into a final verdict."""
    if not signals:
        return "unverifiable", 0.0

    avg = sum(s.score for s in signals) / len(signals)

    if avg >= 0.65:
        verdict = "supported"
    elif avg >= 0.35:
        verdict = "contested"
    else:
        verdict = "unverifiable"

    return verdict, round(avg, 3)


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest) -> VerifyResponse:
    claim_hash = hashlib.sha256(req.claim.encode("utf-8")).hexdigest()[:16]
    signals: List[Signal] = []

    # run all independent verification strategies
    mem_signal = await _memory_cross_ref(req.claim, req.top_k)
    signals.append(mem_signal)
    signals.append(_self_consistency_check(req.plan))
    signals.append(_keyword_plausibility(req.claim, req.context))

    verdict, confidence = _aggregate(signals)

    logger.info("verify claim_hash=%s verdict=%s confidence=%.3f", claim_hash, verdict, confidence)

    return VerifyResponse(
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        claim_hash=claim_hash,
        evaluated_at=time.time(),
    )


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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8052")))
