"""D92: Socratic Self-Questioning — query decomposition before the GATHER stage.

Before Scout gathers evidence, a SocraticQuestioner generates 3-5 precise
decomposition questions that reframe the original query. These are injected
into SwarmContext.enriched_query so every downstream stage reasons against a
deeper, more explicitly decomposed problem.

The improvement is structural, not quality-dependent: even a tiny LLM benefits
from a well-decomposed problem statement.

Feature flag: FF_SOCRATIC (default True)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("kai.questioner")

# Injected dependency type
LLMChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]

_SOCRATIC_SYSTEM = """You are a Socratic questioner. Given a query, generate exactly 3 to 5 precise
decomposition questions that will make any answer to the original query more rigorous.
Each question should do ONE of:
  - Surface a hidden assumption
  - Identify evidence that would disprove the obvious answer
  - Find the simplest explanation
  - Trace second-order consequences
  - Clarify what is actually being asked beneath the surface

Return ONLY a numbered list of questions. No preamble, no commentary, no blank lines between items."""

FALLBACK_QUESTIONS: List[str] = [
    "What is this question actually asking beneath the surface?",
    "What assumptions are embedded in this query?",
    "What evidence would disprove the most obvious answer?",
    "What are the second-order consequences of any answer here?",
    "What is the simplest explanation that fits all known facts?",
]


@dataclass
class SocraticResult:
    original_query: str
    questions: List[str]
    enriched_query: str
    elapsed_ms: float
    used_llm: bool


class SocraticQuestioner:
    """Decomposes a query into 3-5 Socratic sub-questions before evidence gathering."""

    def __init__(self, llm_chat_fn: Optional[LLMChatFn] = None) -> None:
        self._llm = llm_chat_fn

    # ── Public API ──────────────────────────────────────────────────────

    async def decompose(self, query: str) -> SocraticResult:
        t0 = time.monotonic()
        questions, used_llm = await self._generate_questions(query)
        enriched = _build_enriched_query(query, questions)
        return SocraticResult(
            original_query=query,
            questions=questions,
            enriched_query=enriched,
            elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            used_llm=used_llm,
        )

    def can_question(self) -> bool:
        try:
            from feature_flags import is_enabled
            return is_enabled("SOCRATIC")
        except ImportError:
            return True

    # ── Internal ────────────────────────────────────────────────────────

    async def _generate_questions(self, query: str) -> tuple[List[str], bool]:
        if self._llm is None:
            return FALLBACK_QUESTIONS[:3], False
        try:
            raw = await self._llm([
                {"role": "system", "content": _SOCRATIC_SYSTEM},
                {"role": "user", "content": f"Query: {query}"},
            ])
            questions = _parse_question_list(raw)
            if questions:
                return questions[:5], True
        except Exception as exc:
            logger.debug("Socratic LLM call failed, using fallback: %s", exc)
        return FALLBACK_QUESTIONS[:3], False


# ── Helpers ─────────────────────────────────────────────────────────────

def _parse_question_list(raw: str) -> List[str]:
    """Extract numbered or bulleted question lines from raw LLM output."""
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 10:
            continue
        # strip leading number / bullet: "1. ", "1) ", "- ", "• "
        for prefix in ("1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-", "•", "*"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
                break
        if stripped and stripped.endswith("?"):
            lines.append(stripped)
    return lines


def _build_enriched_query(query: str, questions: List[str]) -> str:
    q_block = "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions))
    return f"{query}\n\nKey questions to address before answering:\n{q_block}"
