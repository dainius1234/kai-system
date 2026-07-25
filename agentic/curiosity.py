"""D89/Idea-F: Resource-Aware Curiosity — foundation stub.

When Kai is IDLE and spare GPU cycles are available, he picks a domain
from his knowledge-gap list, researches it, and appends findings to
CURIOSITY.md — a personal side-project log.

This gives Kai an inner life that is not purely task-driven.
Full implementation requires Phase 1 GPU availability.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("kai.curiosity")

CURIOSITY_LOG = Path("/data/CURIOSITY.md")

_CURIOSITY_HEADER = """\
# Curiosity Log

_Kai's side projects and open questions. Written during idle cycles._

_Full entries require Phase 1 GPU. This file is created now so the
schema and location are stable before the hardware arrives._

<!-- entries appended below -->

"""


def _ensure_curiosity_log() -> None:
    try:
        if not CURIOSITY_LOG.exists():
            CURIOSITY_LOG.parent.mkdir(parents=True, exist_ok=True)
            CURIOSITY_LOG.write_text(_CURIOSITY_HEADER, encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not initialise CURIOSITY.md: %s", exc)


def get_open_questions(world_state: Dict[str, Any]) -> List[str]:
    """Return candidate topics for idle exploration.

    Topics are extracted from world state gaps and knowledge map holes.
    Currently returns a static seed list; Phase 1 will mine the memory graph.
    """
    return [
        "What are the latest advances in time-series anomaly detection?",
        "How do causal graphs differ from correlation matrices in practice?",
        "What lightweight LLMs can run inference at <1B parameters?",
        "How does seasonal decomposition (STL) handle irregular time series?",
        "What are the most robust methods for unsupervised behavioral fingerprinting?",
    ]


async def idle_curiosity_tick(
    world_state: Dict[str, Any],
    is_gpu_available: bool = False,
    llm_chat_fn: Optional[Any] = None,
    memories_fn: Optional[Any] = None,
) -> Optional[str]:
    """Called when Kai is IDLE and resource budget allows exploration.

    Args:
        world_state: Current world model snapshot.
        is_gpu_available: True when Phase 1 GPU is provisioned and online.
        llm_chat_fn: Optional LLM callable for D93 hypothesis engine.
        memories_fn: Optional memory retrieval callable for D93 testing.

    Returns:
        The topic explored this tick, or None if no tick occurred.
    """
    _ensure_curiosity_log()

    # D93: run hypothesis engine on CPU regardless of GPU availability
    try:
        from feature_flags import is_enabled
        if is_enabled("HYPOTHESIS_ENGINE"):
            from hypothesis import HypothesisEngine
            engine = HypothesisEngine(llm_chat_fn=llm_chat_fn, memories_fn=memories_fn)
            topics = get_open_questions(world_state)
            hypotheses = await engine.run_cycle(topics[:2])
            if hypotheses:
                tested = hypotheses[0]
                logger.debug(
                    "D93 hypothesis cycle: %d hypotheses, first result=%s",
                    len(hypotheses), tested.result,
                )
                return tested.basis_memory
    except Exception as exc:
        logger.debug("D93 hypothesis tick failed: %s", exc)

    if not is_gpu_available:
        logger.debug("Curiosity tick: hypothesis cycle ran; full GPU research skipped (Phase 0)")
        return None

    # Phase 1+: pick a question, research it, append to CURIOSITY.md
    return None
