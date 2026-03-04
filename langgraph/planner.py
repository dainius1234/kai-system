"""Memory-Driven Planner — history-aware plan construction for kai-system.

Before building a plan or calling the LLM, the planner checks:
  1. Episode history — have we seen a similar request before?
  2. Past outcomes — did it succeed or fail?
  3. Correction memories — any recorded fixes for this pattern?
  4. Proactive nudges — any time-sensitive context?
  5. P10: Sequence prediction — what will the operator likely ask next?

This is what makes Kai learn from experience rather than repeat mistakes.

Usage:
    from planner import memory_driven_plan
    decision = await memory_driven_plan("deploy the server", "session-1")
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx


@dataclass
class PastOutcome:
    """A relevant past interaction."""
    episode_id: str
    input_text: str
    output_text: str
    conviction_score: float
    outcome_score: float
    similarity: float       # 0-1 keyword overlap with current input
    age_days: float


@dataclass
class PlanContext:
    """Enriched context gathered before plan construction."""
    user_input: str
    session_id: str
    memory_chunks: List[Dict[str, Any]] = field(default_factory=list)
    episode_history: List[Dict[str, Any]] = field(default_factory=list)
    past_outcomes: List[PastOutcome] = field(default_factory=list)
    correction_memories: List[Dict[str, Any]] = field(default_factory=list)
    nudges: List[Dict[str, Any]] = field(default_factory=list)
    preferences: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PlanDecision:
    """Result of memory-driven planning."""
    plan: Dict[str, Any]
    conviction_modifier: float    # added to base conviction score
    history_influence: str        # "boosted", "penalised", "neutral", "corrected"
    reuse_episode_id: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    context_summary: str = ""


# ── Similarity scoring ───────────────────────────────────────────────

def _keyword_similarity(text_a: str, text_b: str) -> float:
    """Simple keyword overlap ratio between two texts.

    Returns 0.0-1.0.  Not a vector similarity — just fast keyword Jaccard.
    """
    words_a = set(re.findall(r"\w{3,}", text_a.lower()))
    words_b = set(re.findall(r"\w{3,}", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _find_similar_episodes(
    user_input: str,
    episodes: List[Dict[str, Any]],
    threshold: float = 0.3,
    max_results: int = 5,
) -> List[PastOutcome]:
    """Find episodes with similar inputs to the current query."""
    results: List[PastOutcome] = []
    now = time.time()

    for ep in episodes:
        ep_input = str(ep.get("input", ""))
        if not ep_input:
            continue

        sim = _keyword_similarity(user_input, ep_input)
        if sim < threshold:
            continue

        age_days = (now - float(ep.get("ts", now))) / 86400.0

        results.append(PastOutcome(
            episode_id=str(ep.get("episode_id", "")),
            input_text=ep_input[:200],
            output_text=str(ep.get("output", ""))[:200],
            conviction_score=float(ep.get("final_conviction", ep.get("conviction_score", 0))),
            outcome_score=float(ep.get("outcome_score", 0)),
            similarity=round(sim, 3),
            age_days=round(age_days, 1),
        ))

    # sort by similarity descending, then by recency
    results.sort(key=lambda p: (-p.similarity, p.age_days))
    return results[:max_results]


# ── Context gathering ────────────────────────────────────────────────

async def _fetch_memory_chunks(
    memu_url: str, query: str, user_id: str = "keeper", top_k: int = 5
) -> List[Dict[str, Any]]:
    """Fetch relevant memory chunks from Memu-Core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{memu_url}/memory/retrieve",
                params={"query": query, "user_id": user_id, "top_k": top_k},
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
    except Exception:
        return []


async def _fetch_correction_memories(
    memu_url: str, query: str, top_k: int = 3
) -> List[Dict[str, Any]]:
    """Fetch correction-type memories that match the query."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{memu_url}/memory/search-by-category",
                params={"category": "general", "query": query, "top_k": top_k},
            )
            resp.raise_for_status()
            records = resp.json()
            if isinstance(records, dict):
                records = records.get("results", [])
            # filter to correction-type events
            corrections = []
            for r in records:
                content = r.get("content", {})
                result_text = str(content.get("result", ""))
                if "correction" in result_text.lower() or r.get("event_type") == "correction":
                    corrections.append(r)
            return corrections
    except Exception:
        return []


async def _fetch_nudges(memu_url: str) -> List[Dict[str, Any]]:
    """Fetch proactive nudges that might be relevant."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{memu_url}/memory/proactive")
            resp.raise_for_status()
            data = resp.json()
            return data.get("nudges", [])
    except Exception:
        return []


async def _fetch_preferences(memu_url: str) -> List[Dict[str, Any]]:
    """Fetch operator preferences for plan constraint injection (P5 GEM)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{memu_url}/memory/preferences")
            resp.raise_for_status()
            data = resp.json()
            return data.get("preferences", [])
    except Exception:
        return []


# ── Plan construction ────────────────────────────────────────────────

def _compute_conviction_modifier(past_outcomes: List[PastOutcome], corrections: List[Dict[str, Any]]) -> tuple[float, str, List[str]]:
    """Compute conviction modifier based on history.

    Returns:
        (modifier, influence_type, warnings)
    """
    if not past_outcomes and not corrections:
        return 0.0, "neutral", []

    modifier = 0.0
    influence = "neutral"
    warnings: List[str] = []

    # check for past successes (high outcome + high conviction)
    successes = [p for p in past_outcomes if p.outcome_score >= 0.7 and p.conviction_score >= 7.0]
    failures = [p for p in past_outcomes if p.outcome_score < 0.4]

    if successes:
        best = max(successes, key=lambda p: p.similarity)
        modifier += min(best.similarity * 1.5, 1.0)  # up to +1.0
        influence = "boosted"

    if failures:
        worst = max(failures, key=lambda p: p.similarity)
        penalty = min(worst.similarity * 2.0, 1.5)    # up to -1.5
        modifier -= penalty
        influence = "penalised"
        warnings.append(
            f"Similar request failed before (similarity={worst.similarity:.0%}, "
            f"outcome={worst.outcome_score:.1f}): {worst.input_text[:100]}"
        )

    if corrections:
        modifier -= 0.5  # caution when corrections exist
        if influence == "neutral":
            influence = "corrected"
        warnings.append(
            f"{len(corrections)} past correction(s) found for similar queries"
        )

    return round(modifier, 2), influence, warnings


async def gather_context(
    user_input: str,
    session_id: str,
    episodes: List[Dict[str, Any]],
    memu_url: str = "http://memu-core:8001",
) -> PlanContext:
    """Gather all context needed for memory-driven planning.

    Fetches memory chunks, finds similar episodes, looks for corrections
    and nudges — all in parallel where possible.
    """
    import asyncio

    # parallel fetch
    chunks_task = asyncio.create_task(_fetch_memory_chunks(memu_url, user_input))
    corrections_task = asyncio.create_task(_fetch_correction_memories(memu_url, user_input))
    nudges_task = asyncio.create_task(_fetch_nudges(memu_url))
    prefs_task = asyncio.create_task(_fetch_preferences(memu_url))

    chunks = await chunks_task
    corrections = await corrections_task
    nudges = await nudges_task
    preferences = await prefs_task

    # find similar past episodes (local, no network call)
    similar = _find_similar_episodes(user_input, episodes)

    return PlanContext(
        user_input=user_input,
        session_id=session_id,
        memory_chunks=chunks,
        episode_history=episodes,
        past_outcomes=similar,
        correction_memories=corrections,
        nudges=nudges,
        preferences=preferences,
    )


def build_enriched_plan(context: PlanContext, specialist: str) -> PlanDecision:
    """Build a plan enriched with historical context.

    This is the core planning function. It:
    1. Checks past outcomes for the same pattern
    2. Injects correction context if found
    3. Computes conviction modifier
    4. Adds nudge context if relevant
    """
    modifier, influence, warnings = _compute_conviction_modifier(
        context.past_outcomes, context.correction_memories
    )

    # build the plan dict
    steps = [
        {"action": "analyze", "input": context.user_input},
    ]

    # if reusing a past success, add its output as a reference
    reuse_id = None
    successes = [p for p in context.past_outcomes if p.outcome_score >= 0.7 and p.conviction_score >= 7.0]
    if successes:
        best = max(successes, key=lambda p: p.similarity)
        if best.similarity >= 0.5:  # strong match
            steps.append({"action": "reference_past", "episode_id": best.episode_id, "past_output": best.output_text})
            reuse_id = best.episode_id

    # if corrections exist, inject them as constraints
    if context.correction_memories:
        for corr in context.correction_memories[:2]:
            content = corr.get("content", {})
            correction_text = str(content.get("result", ""))[:300]
            steps.append({"action": "apply_correction", "correction": correction_text})

    # P5 GEM: inject operator preferences as planning constraints
    if context.preferences:
        for pref in context.preferences[:5]:
            pref_text = str(pref.get("content", {}).get("text", ""))[:200]
            if pref_text:
                steps.append({"action": "apply_preference", "preference": pref_text})

    steps.append({"action": "propose", "output": "draft"})

    # context summary for transparency
    summary_parts = []
    if context.past_outcomes:
        summary_parts.append(f"{len(context.past_outcomes)} similar past interaction(s)")
    if context.correction_memories:
        summary_parts.append(f"{len(context.correction_memories)} correction(s)")
    if context.nudges:
        relevant_nudges = [n for n in context.nudges if _keyword_similarity(context.user_input, n.get("content_preview", "")) > 0.1]
        if relevant_nudges:
            summary_parts.append(f"{len(relevant_nudges)} relevant reminder(s)")

    plan = {
        "specialist": specialist,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": steps,
        "offline_context_used": len(context.memory_chunks),
        "history_consulted": len(context.past_outcomes),
        "corrections_applied": len(context.correction_memories),
        "preferences_applied": len(context.preferences),
    }

    return PlanDecision(
        plan=plan,
        conviction_modifier=modifier,
        history_influence=influence,
        reuse_episode_id=reuse_id,
        warnings=warnings,
        context_summary="; ".join(summary_parts) if summary_parts else "no relevant history",
    )


# ── P10: Predictive Pre-Computation ─────────────────────────────────
# Mine operator request sequences to predict what's coming next.
# If the operator always asks A→B→C, and we just saw A→B, predict C
# and pre-fetch relevant context so the response is faster + richer.

SEQUENCE_MIN_SUPPORT = int(__import__("os").getenv("SEQUENCE_MIN_SUPPORT", "2"))
PREDICTION_CONFIDENCE_THRESHOLD = float(__import__("os").getenv("PREDICTION_CONFIDENCE", "0.3"))


@dataclass
class PredictedRequest:
    """A predicted next request based on sequential pattern mining."""
    predicted_topic: str
    confidence: float        # 0-1, fraction of times this followed
    support: int             # how many times the bigram appeared
    pre_fetched_context: List[Dict[str, Any]] = field(default_factory=list)


def _extract_topic_key(text: str) -> str:
    """Extract a normalised topic key from request text."""
    words = sorted(set(re.findall(r"\w{4,}", text.lower())))[:3]
    return "+".join(words) if words else "general"


def mine_request_sequences(
    episodes: List[Dict[str, Any]],
    min_support: int = SEQUENCE_MIN_SUPPORT,
) -> Dict[str, List[Tuple[str, float, int]]]:
    """Build a bigram transition table from episode history.

    Returns {topic_a: [(topic_b, probability, count), ...]} sorted
    by probability descending.  Only includes transitions with
    >= min_support occurrences.
    """
    # sort episodes chronologically
    sorted_eps = sorted(episodes, key=lambda e: float(e.get("ts", 0)))

    # extract topic sequence
    topic_sequence: List[str] = []
    for ep in sorted_eps:
        topic = _extract_topic_key(str(ep.get("input", "")))
        if topic != "general":
            topic_sequence.append(topic)

    if len(topic_sequence) < 2:
        return {}

    # count bigrams
    bigram_counts: Counter = Counter()
    prefix_counts: Counter = Counter()
    for i in range(len(topic_sequence) - 1):
        a, b = topic_sequence[i], topic_sequence[i + 1]
        if a != b:  # skip self-loops (repeated same topic)
            bigram_counts[(a, b)] += 1
            prefix_counts[a] += 1

    # build transition table
    transitions: Dict[str, List[Tuple[str, float, int]]] = {}
    for (a, b), count in bigram_counts.items():
        if count < min_support:
            continue
        prob = count / prefix_counts[a] if prefix_counts[a] > 0 else 0.0
        transitions.setdefault(a, []).append((b, round(prob, 3), count))

    # sort each topic's transitions by probability
    for topic in transitions:
        transitions[topic].sort(key=lambda x: -x[1])

    return transitions


def predict_next_request(
    current_input: str,
    episodes: List[Dict[str, Any]],
    min_support: int = SEQUENCE_MIN_SUPPORT,
    confidence_threshold: float = PREDICTION_CONFIDENCE_THRESHOLD,
) -> List[PredictedRequest]:
    """Predict what the operator will ask next based on historical patterns.

    Mines bigram sequences from episode history and returns predictions
    for the most likely follow-up topics, filtered by confidence threshold.
    """
    current_topic = _extract_topic_key(current_input)
    if current_topic == "general":
        return []

    transitions = mine_request_sequences(episodes, min_support=min_support)
    candidates = transitions.get(current_topic, [])

    predictions: List[PredictedRequest] = []
    for next_topic, prob, count in candidates:
        if prob >= confidence_threshold:
            predictions.append(PredictedRequest(
                predicted_topic=next_topic,
                confidence=prob,
                support=count,
            ))

    return predictions


async def pre_fetch_predicted_context(
    predictions: List[PredictedRequest],
    memu_url: str,
    top_k: int = 3,
) -> List[PredictedRequest]:
    """Pre-fetch memory context for predicted next requests.

    For each prediction with sufficient confidence, query memu-core
    so the context is already warm when the operator asks.
    """
    if not predictions:
        return predictions

    async with httpx.AsyncClient(timeout=3.0) as client:
        for pred in predictions[:3]:  # cap at 3 pre-fetches
            query = pred.predicted_topic.replace("+", " ")
            try:
                resp = await client.get(
                    f"{memu_url}/memory/retrieve",
                    params={"query": query, "user_id": "keeper", "top_k": top_k},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pred.pre_fetched_context = data if isinstance(data, list) else []
            except Exception:
                pass  # pre-fetch is best-effort

    return predictions
