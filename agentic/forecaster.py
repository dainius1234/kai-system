"""D94: Temporal Projection — fan-of-futures forecasting.

Extends Oracle's causal tracing into explicit multi-branch scenario planning.
From supported claims, TemporalForecaster generates four scenario branches:

  base        — most probable continuation given current evidence
  optimistic  — best-case if positive assumptions hold
  pessimistic — worst-case if negative tail risks materialise
  wild_card   — low-probability, high-impact discontinuity

Each branch gets a probability estimate and key assumptions.
The ForecastFan is attached to SwarmContext for downstream synthesis.

CPU-safe: runs with any LLM. GPU accelerates quality, not correctness.
Feature flag: FF_TEMPORAL_PROJECTION (default True)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("kai.forecaster")

LLMChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]

SCENARIO_LABELS = ("base", "optimistic", "pessimistic", "wild_card")

_FORECAST_SYSTEM = """You are a scenario planner. Given a set of supported claims about a situation,
produce exactly 4 scenarios as a JSON array. Each scenario object must have:
  label        — one of: base | optimistic | pessimistic | wild_card
  narrative    — 1-2 sentence description of how things unfold in this scenario
  probability  — float 0.0-1.0 (all four must sum to ≈1.0)
  key_assumptions — list of 1-3 short strings

Return ONLY the JSON array. No preamble."""


@dataclass
class ScenarioBranch:
    label: str                          # base | optimistic | pessimistic | wild_card
    narrative: str
    probability: float
    key_assumptions: List[str] = field(default_factory=list)
    confidence_modifier: float = 0.0    # applied to final conviction score


@dataclass
class ForecastFan:
    query: str
    base_claim: str
    branches: List[ScenarioBranch] = field(default_factory=list)
    elapsed_ms: float = 0.0
    used_llm: bool = False

    @property
    def consensus_probability(self) -> float:
        base = next((b for b in self.branches if b.label == "base"), None)
        return base.probability if base else 0.5

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "base_claim": self.base_claim,
            "consensus_probability": round(self.consensus_probability, 3),
            "branches": [
                {
                    "label": b.label,
                    "narrative": b.narrative,
                    "probability": round(b.probability, 3),
                    "key_assumptions": b.key_assumptions,
                }
                for b in self.branches
            ],
            "elapsed_ms": self.elapsed_ms,
            "used_llm": self.used_llm,
        }


_FALLBACK_BRANCHES: List[ScenarioBranch] = [
    ScenarioBranch(
        label="base",
        narrative="The most likely trajectory continues along current evidence.",
        probability=0.50,
        key_assumptions=["current trends persist", "no major discontinuities"],
    ),
    ScenarioBranch(
        label="optimistic",
        narrative="Positive factors compound and accelerate the best outcome.",
        probability=0.25,
        key_assumptions=["key uncertainties resolve favourably"],
    ),
    ScenarioBranch(
        label="pessimistic",
        narrative="Tail risks materialise and compound into a worse-than-expected outcome.",
        probability=0.20,
        key_assumptions=["negative assumptions prove correct"],
    ),
    ScenarioBranch(
        label="wild_card",
        narrative="A low-probability, high-impact event reshapes the landscape entirely.",
        probability=0.05,
        key_assumptions=["discontinuity event occurs"],
    ),
]


class TemporalForecaster:
    """Projects supported claims into a fan of four future scenarios."""

    def __init__(self, llm_chat_fn: Optional[LLMChatFn] = None) -> None:
        self._llm = llm_chat_fn

    async def project(
        self,
        query: str,
        supported_claims: List[str],
        causal_chains: Optional[List[str]] = None,
    ) -> ForecastFan:
        t0 = time.monotonic()
        base_claim = supported_claims[0] if supported_claims else query
        branches, used_llm = await self._generate_branches(query, supported_claims, causal_chains or [])
        fan = ForecastFan(
            query=query,
            base_claim=base_claim,
            branches=branches,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            used_llm=used_llm,
        )
        return fan

    # ── Internal ─────────────────────────────────────────────────────

    async def _generate_branches(
        self,
        query: str,
        claims: List[str],
        chains: List[str],
    ) -> tuple[List[ScenarioBranch], bool]:
        if self._llm is None:
            return list(_FALLBACK_BRANCHES), False

        claims_text = "\n".join(f"- {c}" for c in claims[:5]) or "(none)"
        chains_text = "\n".join(f"- {c}" for c in chains[:3]) or "(none)"

        try:
            raw = await self._llm([
                {"role": "system", "content": _FORECAST_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Supported claims:\n{claims_text}\n\n"
                        f"Causal chains:\n{chains_text}"
                    ),
                },
            ])
            branches = _parse_branches(raw)
            if len(branches) == 4:
                return branches, True
        except Exception as exc:
            logger.debug("Forecast LLM call failed, using fallback: %s", exc)

        return list(_FALLBACK_BRANCHES), False


def _parse_branches(raw: str) -> List[ScenarioBranch]:
    try:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        data = json.loads(raw[start:end]) if start >= 0 else []
        branches = []
        for item in data:
            label = item.get("label", "base")
            if label not in SCENARIO_LABELS:
                continue
            branches.append(ScenarioBranch(
                label=label,
                narrative=str(item.get("narrative", "")),
                probability=float(item.get("probability", 0.25)),
                key_assumptions=[str(a) for a in item.get("key_assumptions", [])],
            ))
        return branches
    except Exception:
        return []
