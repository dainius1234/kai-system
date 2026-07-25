"""D97: Concept Blending — novel emergent concept synthesis.

Takes two distant concept nodes from the knowledge graph and generates a
blended concept whose emergent properties are not present in either parent.

Based on Fauconnier & Turner's conceptual blending theory.
Example: "fire" + "flower" → "firework" (neither burns nor grows alone).

GPU-era stub: requires a populated concept graph and a generative model
capable of divergent synthesis. can_blend() returns False until ready.

Feature flag: FF_CONCEPT_BLENDING (default False)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("kai.concept_blend")

LLMChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]


@dataclass
class BlendedConcept:
    concept_a: str
    concept_b: str
    blended_name: str = ""
    emergent_properties: List[str] = field(default_factory=list)
    inherited_from_a: List[str] = field(default_factory=list)
    inherited_from_b: List[str] = field(default_factory=list)
    suppressed_properties: List[str] = field(default_factory=list)
    novelty_score: float = 0.0     # 0=trivial, 10=highly emergent
    confidence: float = 0.0


class ConceptBlender:
    """Synthesises novel concepts by blending two distant knowledge graph nodes.

    Requires:
      - memu-graph with rich concept property annotations
      - Generative LLM capable of divergent creative synthesis (GPU)
      - FF_CONCEPT_BLENDING=True
    """

    def __init__(self, llm_chat_fn: Optional[LLMChatFn] = None) -> None:
        self._llm = llm_chat_fn

    def can_blend(self) -> bool:
        """False until concept graph and GPU are available."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("CONCEPT_BLENDING"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate

    async def blend(self, concept_a: str, concept_b: str) -> BlendedConcept:
        blend = BlendedConcept(concept_a=concept_a, concept_b=concept_b)
        if not self.can_blend():
            logger.debug("D97 concept blending pending graph + GPU; returning stub")
            blend.blended_name = f"[STUB] {concept_a[:20]}×{concept_b[:20]}"
            blend.emergent_properties = [
                "pending GPU generative synthesis",
                "pending populated concept graph",
            ]
            blend.novelty_score = 0.0
            blend.confidence = 0.0
            return blend
        # Phase 1: graph traversal → property extraction → LLM blend generation
        return blend
