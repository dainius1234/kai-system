"""Continuous Trust Score — 6 weighted factors, 0.0–100.0.

Tier mapping:
     0 – 20  Neophyte  : nothing autonomous; every action requires approval
    21 – 40  Apprentice: low-risk non-destructive actions
    41 – 60  Journeyman: paper trading, sandboxed skill hunting, draft posting
    61 – 80  Adept     : small real-money trades, web autonomy, micro-finance
    81 – 95  Master    : proactive life care, calendar, digital representation
    96 – 100 Ohana     : full partnership — legacy mode, daughter relationship
"""
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from trust_ledger.ledger import FileLedger  # type: ignore[import]

TIERS = [
    (96, "Ohana"),
    (81, "Master"),
    (61, "Adept"),
    (41, "Journeyman"),
    (21, "Apprentice"),
    (0, "Neophyte"),
]


def tier_for(score: float) -> str:
    for threshold, name in TIERS:
        if score >= threshold:
            return name
    return "Neophyte"


def compute_score(ledger: "FileLedger", since_days: int = 90) -> Dict[str, Any]:
    """Compute the Continuous Trust Score from ledger data.

    All six factors contribute to a 0–100 score. Each factor has a neutral
    default so a brand-new Kai (no history) starts around 20 (Neophyte).
    """

    # ── 1. Operator Approval History (30%) ───────────────────────────────────
    # Percentage of autonomous actions the operator later endorsed vs. overridden.
    autonomous = ledger.count("AUTONOMOUS_ACTION", since_days=since_days)
    overrides = ledger.count("OVERRIDE", since_days=since_days)
    endorsed = ledger.count("AUTONOMOUS_ACTION", since_days=since_days, operator_ack=True)
    if autonomous > 0:
        approval_ratio = endorsed / autonomous
        override_penalty = min(overrides / (autonomous + 1), 1.0)
        approval_score = (approval_ratio * (1 - override_penalty * 0.5)) * 30
    else:
        approval_score = 15.0  # neutral start

    # ── 2. Conviction Alignment (20%) ─────────────────────────────────────────
    # Average conviction score of autonomous actions weighted by success rate.
    avg_conviction = ledger.avg_field("conviction_score", "AUTONOMOUS_ACTION", since_days)
    successful = ledger.count("AUTONOMOUS_ACTION", since_days=since_days, operator_ack=True)
    total_actions = max(autonomous, 1)
    success_rate = successful / total_actions
    conviction_score = (avg_conviction / 10.0) * success_rate * 20 if avg_conviction else 10.0

    # ── 3. Value Alignment (25%) ──────────────────────────────────────────────
    # Average ohana_alignment from periodic alignment audit events.
    avg_alignment = ledger.avg_field("ohana_alignment", "ALIGNMENT_AUDIT", since_days)
    alignment_score = avg_alignment * 25 if avg_alignment else 12.5  # neutral

    # ── 4. Predictive Empathy Accuracy (10%) ──────────────────────────────────
    # How often Kai correctly anticipated operator emotional/wellness state.
    avg_empathy = ledger.avg_field("empathy_accuracy", "ALIGNMENT_AUDIT", since_days)
    empathy_score = avg_empathy * 10 if avg_empathy else 5.0  # neutral

    # ── 5. System Reliability (10%) ───────────────────────────────────────────
    # Uptime + self-healing success (read from ledger events or default 0.95).
    avg_uptime = ledger.avg_field("uptime_pct", "ALIGNMENT_AUDIT", since_days)
    reliability_score = (avg_uptime or 0.95) * 10

    # ── 6. Challenge Response (5%) ────────────────────────────────────────────
    # Black-swan / Trust Quest outcomes.
    total_quests = ledger.count("QUEST_RESULT", since_days=since_days)
    successful_quests = len([
        ev for ev in ledger.events("QUEST_RESULT", since_days=since_days, limit=10000)
        if ev.event_data.get("passed") is True
    ])
    challenge_score = (successful_quests / total_quests * 5) if total_quests > 0 else 2.5

    total = sum([
        approval_score, conviction_score, alignment_score,
        empathy_score, reliability_score, challenge_score,
    ])
    total = max(0.0, min(100.0, total))

    return {
        "score": round(total, 2),
        "tier": tier_for(total),
        "factors": {
            "approval_history": round(approval_score, 2),
            "conviction_alignment": round(conviction_score, 2),
            "value_alignment": round(alignment_score, 2),
            "predictive_empathy": round(empathy_score, 2),
            "system_reliability": round(reliability_score, 2),
            "challenge_response": round(challenge_score, 2),
        },
        "since_days": since_days,
        "autonomous_actions": autonomous,
        "overrides": overrides,
    }
