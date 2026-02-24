"""Conviction scoring — multi-signal plan confidence gate.

Conviction is a 0–10 score measuring how confident the system should be
before proceeding with plan execution.  It combines five independent
signals:

  1. Context coverage    — how well does stored memory support the plan?
  2. Plan specificity    — does the plan have concrete, actionable steps?
  3. Query clarity       — is the user input well-formed and specific?
  4. Rethink improvement — did reflection loops produce better plans?
  5. Specialist fit      — is the chosen specialist appropriate?

The system refuses to execute below MIN_CONVICTION (default 8.0) unless
the operator has filed a conviction override.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# ── specialist → expected keyword domains ────────────────────────────
_SPECIALIST_DOMAINS: Dict[str, List[str]] = {
    "DeepSeek-V4": ["code", "plan", "reason", "build", "debug", "policy", "risk", "architecture"],
    "Kimi-2.5": ["general", "summarise", "search", "image", "translate", "write", "multimodal"],
    "Dolphin": ["chat", "opinion", "creative", "uncensored", "story", "joke"],
}


def build_plan(user_input: str, specialist: str, context_chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    chunks = context_chunks or []
    return {
        "specialist": specialist,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": [{"action": "analyze", "input": user_input}, {"action": "propose", "output": "draft"}],
        "offline_context_used": len(chunks),
    }


def _context_coverage(chunks: List[Dict[str, Any]], user_input: str) -> float:
    """Score 0–2:  how many retrieved memory chunks are relevant?

    Relevant = at least one meaningful keyword from the user input
    appears in the chunk content.
    """
    if not chunks:
        return 0.0

    query_words = set(re.findall(r"\w{3,}", user_input.lower()))
    if not query_words:
        return 0.0

    relevant = 0
    for chunk in chunks:
        text = str(chunk.get("content", chunk.get("result", "")))
        chunk_words = set(re.findall(r"\w{3,}", text.lower()))
        if query_words & chunk_words:
            relevant += 1

    ratio = relevant / len(chunks)
    # scale: 0 chunks relevant → 0, ≥50 % relevant → up to 2.0
    return round(min(ratio * 2.0, 2.0), 3)


def _plan_specificity(plan: Dict[str, Any]) -> float:
    """Score 0–2:  does the plan have concrete, actionable steps?"""
    score = 0.0
    steps = plan.get("steps", [])

    # must have at least 2 steps (analyze + propose)
    if len(steps) >= 2:
        score += 0.5
    if len(steps) >= 3:
        score += 0.3

    # each step should specify an action
    actions_defined = sum(1 for s in steps if isinstance(s, dict) and s.get("action"))
    if actions_defined == len(steps) and steps:
        score += 0.5

    # plan should have a summary
    if plan.get("summary") and len(str(plan["summary"])) > 10:
        score += 0.4

    # bonus if specialist is named
    if plan.get("specialist"):
        score += 0.3

    return round(min(score, 2.0), 3)


def _query_clarity(user_input: str) -> float:
    """Score 0–2:  is the user input well-formed and specific enough?"""
    words = user_input.split()
    n = len(words)

    if n < 3:
        return 0.3  # too vague
    if n < 8:
        return 0.8
    if n < 20:
        return 1.3

    # long, detailed input
    score = 1.5

    # bonus for containing question marks or keywords that imply specificity
    if "?" in user_input:
        score += 0.2
    specific = ["file", "function", "error", "endpoint", "line", "deploy", "build", "run"]
    if any(w in user_input.lower() for w in specific):
        score += 0.3

    return round(min(score, 2.0), 3)


def _rethink_improvement(rethink_count: int) -> float:
    """Score 0–2:  reward the system for having iterated on the plan.

    First rethink gives a good bump.  Diminishing returns after that.
    """
    if rethink_count <= 0:
        return 0.0
    if rethink_count == 1:
        return 0.8
    if rethink_count == 2:
        return 1.2
    return 1.5  # max rethinks


def _specialist_fit(specialist: str, user_input: str) -> float:
    """Score 0–2:  is the right specialist assigned for this query type?"""
    q = user_input.lower()
    domains = _SPECIALIST_DOMAINS.get(specialist, [])
    if not domains:
        return 0.5  # unknown specialist — neutral

    hits = sum(1 for kw in domains if kw in q)
    if hits >= 3:
        return 2.0
    if hits >= 2:
        return 1.5
    if hits >= 1:
        return 1.0
    return 0.5  # no keyword match — might still be right but unconfirmed


def score_conviction(user_input: str, plan: Dict[str, Any], context_chunks: List[Dict[str, Any]], rethink_count: int) -> float:
    """Combined conviction score on a 0–10 scale.

    Each signal contributes 0–2 points.  The sum maps directly to the
    0–10 range the rest of the system expects.
    """
    signals = [
        _context_coverage(context_chunks, user_input),
        _plan_specificity(plan),
        _query_clarity(user_input),
        _rethink_improvement(rethink_count),
        _specialist_fit(plan.get("specialist", ""), user_input),
    ]
    return round(min(sum(signals), 10.0), 2)


def low_conviction_feedback(score: float, chunks: List[Dict[str, Any]]) -> str:
    """Human-readable explanation of why conviction is low."""
    reasons: List[str] = []
    if len(chunks) < 2:
        reasons.append("very few offline context chunks retrieved")
    if score < 4.0:
        reasons.append("plan lacks specificity — try adding concrete constraints")
    if not reasons:
        reasons.append("overall signal quality is below threshold")
    return f"low conviction ({score}/10): {'; '.join(reasons)}"
