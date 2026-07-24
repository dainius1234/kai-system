"""D90: Swarm Assembly — shared context, reputation tracking, and conflict resolution.

SwarmContext accumulates evidence/claims/challenges/verdicts across all pipeline
stages so each stage can see what previous stages found.

Reputation tracks per-teammate quality over time and weights their votes in the
CONVICTION_GATE conflict resolution.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from cognitive_fsm import SwarmConfig

REPUTATION_PATH = Path("/data/teammate_reputation.json")


@dataclass
class SwarmContext:
    query: str
    session_id: str
    swarm_type: str

    # Accumulated cross-stage
    evidence: List[Dict] = field(default_factory=list)
    claims: List[str] = field(default_factory=list)
    challenges: List[str] = field(default_factory=list)
    verdicts: Dict[str, str] = field(default_factory=dict)   # claim → supported|unsupported|uncertain
    causal_chains: List[str] = field(default_factory=list)

    # Reputation-weighted voting: teammate_slug → confidence score
    teammate_votes: Dict[str, float] = field(default_factory=dict)

    stage_log: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)

    def log_stage(self, stage: str, teammate: str, status: str, elapsed_ms: float, confidence: float) -> None:
        self.stage_log.append({
            "stage": stage,
            "teammate": teammate,
            "status": status,
            "elapsed_ms": round(elapsed_ms, 1),
            "confidence": round(confidence, 2),
        })

    def summary(self) -> Dict:
        return {
            "evidence_count": len(self.evidence),
            "claim_count": len(self.claims),
            "challenge_count": len(self.challenges),
            "verdict_count": len(self.verdicts),
            "causal_chain_count": len(self.causal_chains),
            "teammate_votes": dict(self.teammate_votes),
            "stages_completed": len(self.stage_log),
        }


# ── Reputation ───────────────────────────────────────────────────────────────

@dataclass
class TeammateRep:
    slug: str
    total_calls: int = 0
    successful_handoffs: int = 0
    total_confidence: float = 0.0
    error_count: int = 0

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / max(1, self.successful_handoffs)

    @property
    def reliability(self) -> float:
        return self.successful_handoffs / max(1, self.total_calls)

    def weight(self) -> float:
        """Reliability × normalised avg_confidence — used to weight CONVICTION_GATE votes."""
        return self.reliability * (self.avg_confidence / 10.0)


_REPUTATION: Dict[str, TeammateRep] = {}


def load_reputation() -> None:
    global _REPUTATION
    try:
        if REPUTATION_PATH.exists():
            data = json.loads(REPUTATION_PATH.read_text())
            _REPUTATION = {
                slug: TeammateRep(**fields)
                for slug, fields in data.items()
            }
    except Exception:
        _REPUTATION = {}


def save_reputation() -> None:
    try:
        REPUTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            slug: {
                "slug": rep.slug,
                "total_calls": rep.total_calls,
                "successful_handoffs": rep.successful_handoffs,
                "total_confidence": rep.total_confidence,
                "error_count": rep.error_count,
            }
            for slug, rep in _REPUTATION.items()
        }
        REPUTATION_PATH.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def get_rep(slug: str) -> TeammateRep:
    if slug not in _REPUTATION:
        _REPUTATION[slug] = TeammateRep(slug=slug)
    return _REPUTATION[slug]


def record_success(slug: str, confidence: float) -> None:
    rep = get_rep(slug)
    rep.total_calls += 1
    rep.successful_handoffs += 1
    rep.total_confidence += confidence


def record_error(slug: str) -> None:
    rep = get_rep(slug)
    rep.total_calls += 1
    rep.error_count += 1


def list_reputation() -> List[Dict]:
    return [
        {
            "slug": rep.slug,
            "total_calls": rep.total_calls,
            "successful_handoffs": rep.successful_handoffs,
            "avg_confidence": round(rep.avg_confidence, 2),
            "reliability": round(rep.reliability, 2),
            "weight": round(rep.weight(), 3),
            "error_count": rep.error_count,
        }
        for rep in _REPUTATION.values()
    ]


# ── Conflict resolution ───────────────────────────────────────────────────────

def resolve_conflict(
    context: SwarmContext,
    config: SwarmConfig,
    adversary_modifier: float = 0.0,
) -> float:
    """Priority: evidence weight → causal chain quality → verdict fraction →
    reputation-weighted vote → skeptic adversary modifier.

    Returns a final conviction score 0.0–10.0.
    """
    # 1. Evidence weight: more evidence → higher baseline (cap at 10)
    evidence_score = min(10.0, len(context.evidence) * 1.5)

    # 2. Causal chain quality: each verified chain boosts score
    causal_score = min(10.0, len(context.causal_chains) * 2.0)

    # 3. Fact-check verdict fraction: portion of claims that are supported
    if context.verdicts:
        supported = sum(1 for v in context.verdicts.values() if v == "supported")
        verdict_score = (supported / len(context.verdicts)) * 10.0
    else:
        verdict_score = 5.0  # neutral when no verdicts run

    # 4. Reputation-weighted teammate vote
    total_weight = 0.0
    weighted_sum = 0.0
    for slug, vote in context.teammate_votes.items():
        w = _REPUTATION[slug].weight() if slug in _REPUTATION else 0.5
        weighted_sum += vote * w
        total_weight += w
    vote_score = (weighted_sum / total_weight) if total_weight > 0 else 5.0

    # 5. Adversary/skeptic: normalise total_modifier (−3..+1) to 0–10 centred at 5
    skeptic_score = min(10.0, max(0.0, 5.0 + adversary_modifier * 1.67))

    final = (
        evidence_score * 0.30
        + causal_score * 0.25
        + verdict_score * 0.20
        + vote_score * 0.15
        + skeptic_score * 0.10
    )
    return round(min(10.0, max(0.0, final)), 2)
