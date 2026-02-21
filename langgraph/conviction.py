from __future__ import annotations

from typing import Any, Dict, List


def build_plan(user_input: str, specialist: str, context_chunks: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    chunks = context_chunks or []
    return {
        "specialist": specialist,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": [{"action": "analyze", "input": user_input}, {"action": "propose", "output": "draft"}],
        "offline_context_used": len(chunks),
    }


def score_conviction(user_input: str, plan: Dict[str, Any], context_chunks: List[Dict[str, Any]], rethink_count: int) -> float:
    score = 7.2
    if len(user_input.strip()) >= 24:
        score += 0.4
    if plan.get("summary"):
        score += 0.4
    if len(context_chunks) >= 1:
        score += 0.4
    if len(context_chunks) >= 3:
        score += 0.4
    if rethink_count > 0:
        score += 0.2
    return round(min(score, 10.0), 2)


def low_conviction_feedback(score: float, chunks: List[Dict[str, Any]]) -> str:
    missing = "offline context" if len(chunks) < 2 else "specific constraints"
    return f"low conviction ({score}/10), missing {missing}"
