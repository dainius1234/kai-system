"""D99: Synthetic Experience Generator — fictional scenario generation during dream cycles.

During dream phases, Kai generates synthetic experiences: fictional but
internally consistent scenarios that exercise reasoning pathways that would
rarely be stimulated by real interactions alone.

Purpose:
  - Strengthen causal reasoning chains by running them in imagined worlds
  - Build empathy-adjacent models by simulating other perspectives
  - Surface edge-case reasoning gaps before they appear in real situations

GPU-era stub: can_generate() returns False until GPU dream cycles are active.
Feature flag: FF_SYNTHETIC_EXPERIENCE (default False)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("kai.synthetic_experience")

LLMChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]

EXPERIENCE_TYPES = ("counterfactual", "perspective_shift", "edge_case", "stress_test")


@dataclass
class SyntheticScenario:
    premise: str
    narrative: str = ""
    entities: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0          # -1=negative, 0=neutral, +1=positive
    reasoning_pathways_exercised: List[str] = field(default_factory=list)
    experience_type: str = "counterfactual"
    insight: str = ""                        # what was learned
    confidence: float = 0.0


class SyntheticExperienceGenerator:
    """Generates fictional scenarios during dream cycles to exercise reasoning pathways.

    Requires:
      - Active dream cycle (FF_DREAM_ENABLED=True)
      - GPU-class generative capacity for coherent narrative
      - FF_SYNTHETIC_EXPERIENCE=True
    """

    def __init__(self, llm_chat_fn: Optional[LLMChatFn] = None) -> None:
        self._llm = llm_chat_fn

    def can_generate(self) -> bool:
        """False until dream GPU cycles are active."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("SYNTHETIC_EXPERIENCE"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate

    async def generate(
        self,
        seed_concept: str,
        experience_type: str = "counterfactual",
    ) -> SyntheticScenario:
        scenario = SyntheticScenario(
            premise=seed_concept,
            experience_type=experience_type,
        )
        if not self.can_generate():
            logger.debug("D99 synthetic experience pending GPU dream cycles; returning stub")
            scenario.narrative = (
                f"[STUB] Synthetic scenario from '{seed_concept[:60]}' "
                f"(type={experience_type}) — pending GPU dream activation."
            )
            scenario.insight = "Pending Phase 1 GPU provisioning."
            scenario.confidence = 0.0
            return scenario
        # Phase 1: LLM generates coherent fictional narrative, extracts insight
        return scenario

    async def batch_generate(
        self,
        seeds: List[str],
        experience_type: str = "counterfactual",
    ) -> List[SyntheticScenario]:
        return [await self.generate(s, experience_type) for s in seeds[:5]]
