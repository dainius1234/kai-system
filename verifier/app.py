"""Verifier — deterministic fact-checking and verdict gating.

The single authority that decides whether a claim is trustworthy enough
for memory promotion or tool execution.  Returns one of three verdicts:

  PASS        → memory promotion allowed, tool execution approved
  REPAIR      → auto-retry with broadened retrieval, then re-verify
  FAIL_CLOSED → block promotion, quarantine trajectory, log only

Verification strategies:
  1. Memory cross-ref:  check the claim against stored records in memu
  2. Material claim extraction: identify numbers, dates, IDs, instructions
  3. Self-consistency:  check whether the plan's own steps are coherent
  4. Keyword plausibility: heuristic language analysis (placeholder for
     multi-LLM consensus when DeepSeek-V4 + Kimi-2.5 are wired)

The verifier never executes anything.  It only reads and reports.
Thresholds are read from security/policy.yml via common.policy.
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

from common.policy import policy_hash, policy_version, verifier_thresholds
from common.runtime import ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("verifier", os.getenv("LOG_PATH", "/tmp/verifier.json.log"))
DEVICE = detect_device()

app = FastAPI(title="Verifier", version="0.5.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
budget = ErrorBudget(window_seconds=300)

# load thresholds from policy
_thresholds = verifier_thresholds()
PASS_THRESHOLD = float(_thresholds.get("pass_threshold", 0.65))
REPAIR_THRESHOLD = float(_thresholds.get("repair_threshold", 0.35))
MIN_STRONG_CHUNKS = int(_thresholds.get("min_strong_chunks", 2))
STRONG_CHUNK_THRESHOLD = float(_thresholds.get("strong_chunk_threshold", 0.60))
MATERIAL_CLAIM_TYPES = _thresholds.get("material_claim_types", [
    "numbers", "dates", "identifiers", "instructions", "safety",
])

# ── verdict counters (for Prometheus / alerting) ────────────────────
_verdict_counts: Dict[str, int] = {"PASS": 0, "REPAIR": 0, "FAIL_CLOSED": 0}


# ── Models ──────────────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    claim: str
    context: Optional[str] = None
    source: str = "unknown"
    plan: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=10, ge=1, le=100)
    # optional: evidence pack supplied by caller (from memu-core)
    evidence_pack: Optional[List[Dict[str, Any]]] = None


class Signal(BaseModel):
    strategy: str
    score: float = Field(ge=0.0, le=1.0)
    detail: str


class MaterialClaim(BaseModel):
    """A specific, verifiable claim extracted from prose."""
    claim_type: str       # numbers, dates, identifiers, instructions, safety, etc.
    raw_text: str         # the extracted fragment
    confidence: float     # how confident we are this is a material claim


class VerifyResponse(BaseModel):
    verdict: str          # "PASS", "REPAIR", "FAIL_CLOSED"
    confidence: float     # 0.0 – 1.0
    signals: List[Signal]
    material_claims: List[MaterialClaim]
    claim_hash: str
    evaluated_at: float
    policy_version: str
    policy_hash: str
    strong_chunks: int    # how many evidence chunks scored above threshold
    evidence_summary: str  # human-readable evidence assessment


# ── Material claim extraction ───────────────────────────────────────

_CLAIM_PATTERNS: Dict[str, List[re.Pattern]] = {  # type: ignore[type-arg]
    "numbers": [
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:m|mm|km|kg|tonnes?|nr|no\.?|%)\b", re.I),
        re.compile(r"\b(?:grid|ch|chainage)\s*\d+", re.I),
    ],
    "dates": [
        re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
        re.compile(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s*\d{4}\b", re.I),
        re.compile(r"\b(?:week|wk)\s*\d{1,2}\b", re.I),
        re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),  # ISO dates
    ],
    "identifiers": [
        re.compile(r"\b(?:DWG|REV|NCR|RFI|ITP|RAMS|PO|WO)[\s-]*\d+", re.I),
        re.compile(r"\b[A-Z]{2,5}-\d{3,}\b"),  # codes like NCR-001, DWG-1234
        re.compile(r"\bgrid\s+[A-Z]\d*\b", re.I),
    ],
    "instructions": [
        re.compile(r"\b(?:must|shall|do not|ensure|verify|check|confirm|install|remove|replace)\b", re.I),
        re.compile(r"\bstep\s+\d+\b", re.I),
    ],
    "safety": [
        re.compile(r"\b(?:RAMS|COSHH|PPE|permit|H&S|hazard|risk\s+assessment|method\s+statement)\b", re.I),
        re.compile(r"\b(?:RIDDOR|near\s+miss|accident|incident|exclusion\s+zone)\b", re.I),
    ],
    "financial": [
        re.compile(r"£\d+", re.I),
        re.compile(r"\b(?:invoice|payment|cost|budget|variation|VO)\b", re.I),
    ],
    "config": [
        re.compile(r"\b(?:port|auth|token|password|key|encrypt|backup|firewall|TLS|SSL)\b", re.I),
    ],
}


def extract_material_claims(text: str) -> List[MaterialClaim]:
    """Extract material (verifiable, specific) claims from text."""
    claims: List[MaterialClaim] = []
    seen: set = set()

    for claim_type, patterns in _CLAIM_PATTERNS.items():
        if claim_type not in MATERIAL_CLAIM_TYPES:
            continue
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw = match.group().strip()
                if raw not in seen and len(raw) >= 2:
                    seen.add(raw)
                    # confidence based on match specificity
                    conf = 0.9 if claim_type in ("numbers", "dates", "identifiers") else 0.7
                    claims.append(MaterialClaim(
                        claim_type=claim_type,
                        raw_text=raw,
                        confidence=conf,
                    ))
    return claims


# ── Verification strategies ─────────────────────────────────────────


async def _memory_cross_ref(claim: str, top_k: int,
                            evidence_pack: Optional[List[Dict]] = None) -> tuple[Signal, int]:
    """Check whether memu holds records that support the claim.

    Returns the signal AND the count of strong evidence chunks.
    """
    records: List[Dict] = []

    # prefer pre-supplied evidence pack if available
    if evidence_pack:
        records = evidence_pack
    else:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{MEMU_URL}/memory/retrieve",
                    params={"query": claim, "user_id": "keeper", "top_k": top_k},
                )
                resp.raise_for_status()
                records = resp.json()
        except Exception:
            return Signal(
                strategy="memory_cross_ref",
                score=0.0,
                detail="memu unreachable — cannot verify against memory",
            ), 0

    if not records:
        return Signal(
            strategy="memory_cross_ref",
            score=0.0,
            detail="no matching memory records found",
        ), 0

    # score based on keyword overlap + relevance scoring
    claim_words = set(re.findall(r"\w{3,}", claim.lower()))
    hits = 0
    strong_chunks = 0

    for rec in records:
        content = rec if isinstance(rec, dict) else {}
        text = str(content.get("content", content.get("result", "")))
        rec_words = set(re.findall(r"\w{3,}", text.lower()))
        overlap = claim_words & rec_words
        overlap_ratio = len(overlap) / max(len(claim_words), 1)

        if overlap_ratio >= 0.3:
            hits += 1
        # strong chunk: high relevance AND good overlap
        relevance = float(content.get("relevance", 0))
        importance = float(content.get("importance", 0.5))
        chunk_score = overlap_ratio * 0.5 + relevance * 0.3 + importance * 0.2
        if chunk_score >= STRONG_CHUNK_THRESHOLD:
            strong_chunks += 1

    ratio = min(hits / max(len(records), 1), 1.0)
    return Signal(
        strategy="memory_cross_ref",
        score=round(ratio, 3),
        detail=f"{hits}/{len(records)} records support the claim; "
               f"{strong_chunks} strong chunks (threshold={STRONG_CHUNK_THRESHOLD})",
    ), strong_chunks


def _self_consistency_check(plan: Optional[Dict[str, Any]]) -> Signal:
    """Verify internal consistency of a plan's steps."""
    if not plan:
        return Signal(strategy="self_consistency", score=0.5,
                      detail="no plan provided — neutral score")

    steps = plan.get("steps", [])
    if not steps:
        return Signal(strategy="self_consistency", score=0.3,
                      detail="plan has no steps")

    issues: List[str] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"step {i} is not a dict")
            continue
        if "action" not in step:
            issues.append(f"step {i} missing 'action'")

    if not plan.get("summary"):
        issues.append("plan missing summary")

    score = max(1.0 - (len(issues) * 0.2), 0.0)
    detail = ("plan is internally consistent" if not issues
              else f"issues: {'; '.join(issues)}")
    return Signal(strategy="self_consistency", score=round(score, 3),
                  detail=detail)


def _keyword_plausibility(claim: str, context: Optional[str]) -> Signal:
    """Heuristic language analysis — placeholder for multi-LLM consensus.

    When DeepSeek-V4 and Kimi-2.5 are wired as verification backends,
    this will be replaced with actual cross-model agreement scoring.
    """
    lower = claim.lower()

    suspect = ["always", "never", "guaranteed", "100%", "impossible",
               "certainly"]
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

    if context and len(context.strip()) > 20:
        score = min(score + 0.1, 1.0)
        detail += "; context provided"

    return Signal(strategy="keyword_plausibility", score=round(score, 3),
                  detail=detail)


def _material_claim_signal(claims: List[MaterialClaim],
                           strong_chunks: int) -> Signal:
    """Score based on material claims: more claims need more evidence."""
    if not claims:
        return Signal(
            strategy="material_claims",
            score=0.7,
            detail="no material claims detected — low-risk prose",
        )

    n_claims = len(claims)
    # require proportional evidence for material claims
    if strong_chunks >= max(n_claims, MIN_STRONG_CHUNKS):
        score = 0.9
        detail = (f"{n_claims} material claims fully supported by "
                  f"{strong_chunks} strong evidence chunks")
    elif strong_chunks >= 1:
        score = 0.5
        detail = (f"{n_claims} material claims, only {strong_chunks} "
                  f"strong chunks (need {MIN_STRONG_CHUNKS}+)")
    else:
        score = 0.2
        detail = (f"{n_claims} material claims with NO strong evidence "
                  f"chunks — high risk of confabulation")

    return Signal(strategy="material_claims", score=round(score, 3),
                  detail=detail)


# ── Verdict aggregation ─────────────────────────────────────────────


def _aggregate(signals: List[Signal],
               strong_chunks: int) -> tuple[str, float, str]:
    """Combine signal scores into a deterministic verdict.

    Returns (verdict, confidence, evidence_summary).
    """
    if not signals:
        return "FAIL_CLOSED", 0.0, "no verification signals produced"

    avg = sum(s.score for s in signals) / len(signals)

    # build evidence summary
    summaries = [f"{s.strategy}={s.score:.2f}" for s in signals]
    evidence_summary = (
        f"avg={avg:.3f} strong_chunks={strong_chunks} "
        f"signals=[{', '.join(summaries)}]"
    )

    if avg >= PASS_THRESHOLD and strong_chunks >= MIN_STRONG_CHUNKS:
        verdict = "PASS"
    elif avg >= PASS_THRESHOLD and strong_chunks < MIN_STRONG_CHUNKS:
        verdict = "REPAIR"
        evidence_summary += (
            f" — score meets threshold but insufficient strong chunks "
            f"({strong_chunks}<{MIN_STRONG_CHUNKS})"
        )
    elif avg >= REPAIR_THRESHOLD:
        verdict = "REPAIR"
    else:
        verdict = "FAIL_CLOSED"

    return verdict, round(avg, 3), evidence_summary


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "device": DEVICE,
        "policy_version": policy_version,
        "policy_hash": policy_hash,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return {
        **budget.snapshot(),
        "verdicts": dict(_verdict_counts),
        "policy_version": policy_version,
    }


@app.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest) -> VerifyResponse:
    claim_hash = hashlib.sha256(req.claim.encode("utf-8")).hexdigest()[:16]
    signals: List[Signal] = []

    # 1. extract material claims from the prose
    material_claims = extract_material_claims(req.claim)
    if req.context:
        material_claims.extend(extract_material_claims(req.context))

    # 2. memory cross-reference (returns signal + strong chunk count)
    mem_signal, strong_chunks = await _memory_cross_ref(
        req.claim, req.top_k, req.evidence_pack,
    )
    signals.append(mem_signal)

    # 3. self-consistency check
    signals.append(_self_consistency_check(req.plan))

    # 4. keyword plausibility
    signals.append(_keyword_plausibility(req.claim, req.context))

    # 5. material claim signal (needs strong chunk count from step 2)
    signals.append(_material_claim_signal(material_claims, strong_chunks))

    # aggregate all signals into a deterministic verdict
    verdict, confidence, evidence_summary = _aggregate(signals, strong_chunks)

    # track verdict counts for Prometheus alerting
    _verdict_counts[verdict] = _verdict_counts.get(verdict, 0) + 1

    logger.info(
        "verify claim_hash=%s verdict=%s confidence=%.3f "
        "strong_chunks=%d material_claims=%d",
        claim_hash, verdict, confidence, strong_chunks, len(material_claims),
    )

    return VerifyResponse(
        verdict=verdict,
        confidence=confidence,
        signals=signals,
        material_claims=material_claims,
        claim_hash=claim_hash,
        evaluated_at=time.time(),
        policy_version=policy_version,
        policy_hash=policy_hash,
        strong_chunks=strong_chunks,
        evidence_summary=evidence_summary,
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
