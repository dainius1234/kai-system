"""D93: Autonomous Hypothesis Engine — idle-cycle knowledge gap scanning.

During idle cycles, HypothesisEngine:
  1. Scans memories for low-confidence or contradicted beliefs.
  2. Forms a testable hypothesis: "If X is true, Y should follow."
  3. Tests the hypothesis against available memory evidence.
  4. Logs confirmed/refuted/open hypotheses to CURIOSITY.md and memu-core.

CPU-safe: no GPU required. Works with any LLM via the injected llm_chat_fn.
Feature flag: FF_HYPOTHESIS_ENGINE (default True)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("kai.hypothesis")

CURIOSITY_LOG = Path("/data/CURIOSITY.md")

LLMChatFn = Callable[[List[Dict[str, str]]], Awaitable[str]]
MemoriesFn = Callable[[str], Awaitable[List[str]]]

_HYPOTHESIS_SYSTEM = """You are a hypothesis generator. Given a belief that is uncertain or
contradicted, generate ONE precise, falsifiable hypothesis in the form:
"If [condition] is true, then [observable consequence] should be the case."

Return ONLY the hypothesis sentence. No preamble, no explanation."""

_TEST_SYSTEM = """You are a hypothesis tester. Given a hypothesis and supporting evidence,
determine whether the evidence supports or refutes the hypothesis.

Return exactly one of: SUPPORTED | REFUTED | INCONCLUSIVE
followed by a single-sentence rationale. Example:
SUPPORTED — All three memory entries confirm the causal link."""


@dataclass
class Hypothesis:
    statement: str
    basis_memory: str
    test_predicate: str
    result: str = "untested"      # SUPPORTED | REFUTED | INCONCLUSIVE | untested
    rationale: str = ""
    confidence: float = 0.0
    formed_at: float = field(default_factory=time.monotonic)


class HypothesisEngine:
    """Scans low-confidence memories and forms + tests hypotheses."""

    MIN_MEMORIES_TO_SCAN = 3
    MAX_HYPOTHESES_PER_CYCLE = 3

    def __init__(
        self,
        llm_chat_fn: Optional[LLMChatFn] = None,
        memories_fn: Optional[MemoriesFn] = None,
    ) -> None:
        self._llm = llm_chat_fn
        self._memories = memories_fn

    async def run_cycle(
        self,
        seed_topics: List[str],
    ) -> List[Hypothesis]:
        """Run one hypothesis-formation-and-test cycle over seed topics.

        Args:
            seed_topics: Low-confidence topics or open questions to explore.

        Returns:
            List of tested Hypothesis objects.
        """
        if not seed_topics:
            return []

        results: List[Hypothesis] = []
        for topic in seed_topics[: self.MAX_HYPOTHESES_PER_CYCLE]:
            hyp = await self._form_hypothesis(topic)
            if hyp:
                hyp = await self._test_hypothesis(hyp)
                results.append(hyp)
                _append_to_log(hyp)

        return results

    # ── Internal ─────────────────────────────────────────────────────

    async def _form_hypothesis(self, basis_memory: str) -> Optional[Hypothesis]:
        if self._llm is None:
            statement = (
                f"If the pattern described in '{basis_memory[:60]}' holds, "
                "then related memories should show the same trend."
            )
            return Hypothesis(
                statement=statement,
                basis_memory=basis_memory,
                test_predicate="related memories show same trend",
            )
        try:
            raw = await self._llm([
                {"role": "system", "content": _HYPOTHESIS_SYSTEM},
                {"role": "user", "content": f"Uncertain belief: {basis_memory}"},
            ])
            statement = raw.strip()
            if not statement:
                return None
            predicate = statement.split("then", 1)[-1].strip() if "then" in statement.lower() else statement
            return Hypothesis(
                statement=statement,
                basis_memory=basis_memory,
                test_predicate=predicate,
            )
        except Exception as exc:
            logger.debug("Hypothesis formation failed: %s", exc)
            return None

    async def _test_hypothesis(self, hyp: Hypothesis) -> Hypothesis:
        evidence: List[str] = []
        if self._memories is not None:
            try:
                evidence = await self._memories(hyp.test_predicate)
            except Exception as exc:
                logger.debug("Memory fetch for hypothesis test failed: %s", exc)

        if not evidence:
            hyp.result = "INCONCLUSIVE"
            hyp.rationale = "No supporting evidence found in memory store."
            hyp.confidence = 3.0
            return hyp

        if self._llm is None:
            hyp.result = "INCONCLUSIVE"
            hyp.rationale = f"Found {len(evidence)} memory entries; LLM needed to adjudicate."
            hyp.confidence = 5.0
            return hyp

        evidence_text = "\n".join(f"- {e}" for e in evidence[:5])
        try:
            raw = await self._llm([
                {"role": "system", "content": _TEST_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Hypothesis: {hyp.statement}\n\n"
                        f"Evidence:\n{evidence_text}"
                    ),
                },
            ])
            first_line = raw.strip().splitlines()[0] if raw.strip() else ""
            for verdict in ("SUPPORTED", "REFUTED", "INCONCLUSIVE"):
                if verdict in first_line.upper():
                    hyp.result = verdict
                    break
            else:
                hyp.result = "INCONCLUSIVE"
            hyp.rationale = raw.strip()
            hyp.confidence = {"SUPPORTED": 8.0, "REFUTED": 7.0, "INCONCLUSIVE": 4.0}.get(hyp.result, 4.0)
        except Exception as exc:
            logger.debug("Hypothesis test LLM call failed: %s", exc)
            hyp.result = "INCONCLUSIVE"
            hyp.rationale = str(exc)
            hyp.confidence = 3.0

        return hyp


# ── Logging helper ────────────────────────────────────────────────────

def _append_to_log(hyp: Hypothesis) -> None:
    try:
        if not CURIOSITY_LOG.exists():
            CURIOSITY_LOG.parent.mkdir(parents=True, exist_ok=True)
            CURIOSITY_LOG.write_text(
                "# Curiosity Log\n\n_Kai's hypotheses and open questions._\n\n",
                encoding="utf-8",
            )
        entry = (
            f"\n## Hypothesis [{hyp.result}]\n"
            f"**Basis:** {hyp.basis_memory}\n\n"
            f"**Statement:** {hyp.statement}\n\n"
            f"**Verdict:** {hyp.result} (confidence {hyp.confidence}/10)\n\n"
            f"**Rationale:** {hyp.rationale}\n"
        )
        with CURIOSITY_LOG.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.debug("Could not append hypothesis to CURIOSITY.md: %s", exc)
