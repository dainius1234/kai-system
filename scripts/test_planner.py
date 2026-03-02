"""Tests for the Memory-Driven Planner — langgraph/planner.py

Validates keyword similarity, episode matching, conviction modifiers,
and enriched plan construction.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

# Load planner.py directly (avoid installed langgraph package)
_mod_path = Path(__file__).resolve().parents[1] / "langgraph" / "planner.py"
_spec = importlib.util.spec_from_file_location("planner", _mod_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["planner"] = _mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec.loader.exec_module(_mod)

_keyword_similarity = _mod._keyword_similarity
_find_similar_episodes = _mod._find_similar_episodes
_compute_conviction_modifier = _mod._compute_conviction_modifier
build_enriched_plan = _mod.build_enriched_plan
PastOutcome = _mod.PastOutcome
PlanContext = _mod.PlanContext
PlanDecision = _mod.PlanDecision

# ── _keyword_similarity tests ────────────────────────────────────────

# identical strings → 1.0
assert _keyword_similarity("buy groceries at the store", "buy groceries at the store") == 1.0

# completely disjoint → 0.0
assert _keyword_similarity("the cat sat", "quick brown fox") == 0.0

# partial overlap
sim = _keyword_similarity("deploy the docker image", "build the docker container")
assert 0.0 < sim < 1.0, f"expected partial overlap, got {sim}"

# empty inputs
assert _keyword_similarity("", "") == 0.0
assert _keyword_similarity("hello", "") == 0.0

print("  _keyword_similarity tests passed")

# ── _find_similar_episodes tests ─────────────────────────────────────
# Episode dicts use "input" key, "ts" for timestamp, "outcome_score",
# "final_conviction", "episode_id"

now = time.time()
episodes = [
    {"input": "how much tax do I owe", "output": "you owe 500",
     "outcome_score": 0.9, "final_conviction": 8.5,
     "ts": now - 86400, "episode_id": "ep-1"},
    {"input": "deploy the new build", "output": "deployed v2.3",
     "outcome_score": 0.2, "final_conviction": 7.0,
     "ts": now - 172800, "episode_id": "ep-2"},
    {"input": "what tax returns are due", "output": "SA100 due 31 Jan",
     "outcome_score": 0.8, "final_conviction": 9.0,
     "ts": now - 259200, "episode_id": "ep-3"},
    {"input": "tell me a joke", "output": "why did the chicken...",
     "outcome_score": 0.5, "final_conviction": 6.0,
     "ts": now - 345600, "episode_id": "ep-4"},
]

# search for tax-related → should find the two tax episodes
results = _find_similar_episodes("how much tax do I owe", episodes, threshold=0.3)
assert len(results) >= 1, f"expected at least 1 match, got {len(results)}"
assert all(isinstance(r, PastOutcome) for r in results)
# best match should be the exact-match input
assert results[0].similarity == 1.0 or results[0].input_text == "how much tax do I owe"

# search for unrelated → no matches above threshold
results = _find_similar_episodes("paint the house purple", episodes, threshold=0.3)
assert len(results) == 0, f"expected 0 matches, got {len(results)}"

# empty episodes list
results = _find_similar_episodes("anything", [], threshold=0.3)
assert len(results) == 0

print("  _find_similar_episodes tests passed")

# ── _compute_conviction_modifier tests ───────────────────────────────
# Returns (modifier: float, influence: str, warnings: list)

# past success (high outcome_score + high conviction_score) → positive boost
p1 = PastOutcome(
    episode_id="ep-1", input_text="tax owed", output_text="500",
    conviction_score=8.5, outcome_score=0.9, similarity=0.9, age_days=1.0,
)
mod, inf, warns = _compute_conviction_modifier([p1], [])
assert mod > 0, f"expected positive modifier for success, got {mod}"
assert inf == "boosted"

# past failure (low outcome_score) → negative penalty
p2 = PastOutcome(
    episode_id="ep-2", input_text="deploy script", output_text="failed",
    conviction_score=7.0, outcome_score=0.2, similarity=0.8, age_days=2.0,
)
mod, inf, warns = _compute_conviction_modifier([p2], [])
assert mod < 0, f"expected negative modifier for failure, got {mod}"
assert inf == "penalised"
assert len(warns) >= 1

# corrections → small negative
corrections = [{"content": {"result": "threshold is 85000 not 1000"}}]
mod, inf, warns = _compute_conviction_modifier([], corrections)
assert mod < 0, f"expected negative modifier for corrections, got {mod}"
assert inf == "corrected"
assert len(warns) >= 1

# mixed: one success + one failure
mod, inf, warns = _compute_conviction_modifier([p1, p2], [])
assert isinstance(mod, (int, float)), f"expected numeric modifier, got {type(mod)}"

# no history → zero
mod, inf, warns = _compute_conviction_modifier([], [])
assert mod == 0.0, f"expected 0.0 for no history, got {mod}"
assert inf == "neutral"
assert warns == []

print("  _compute_conviction_modifier tests passed")

# ── build_enriched_plan tests ────────────────────────────────────────

ctx = PlanContext(
    user_input="how much tax do I owe",
    session_id="test-session",
    memory_chunks=[{"content": "User discussed VAT thresholds previously"}],
    past_outcomes=[p1],
    correction_memories=[{"content": {"result": "threshold changed to 85000"}}],
    nudges=[{"content_preview": "Reminder: check HMRC deadlines"}],
)

plan = build_enriched_plan(ctx, specialist="TAX_ADVISORY")
assert isinstance(plan, PlanDecision), f"expected PlanDecision, got {type(plan)}"
assert isinstance(plan.plan, dict), f"plan should be dict, got {type(plan.plan)}"
assert len(plan.plan["steps"]) > 0, "plan should have at least one step"
assert plan.plan["specialist"] == "TAX_ADVISORY"

# with corrections, there should be at least one warning
assert len(plan.warnings) >= 1, f"expected warnings for corrections, got {plan.warnings}"

# conviction modifier should reflect the past success + correction penalty
assert isinstance(plan.conviction_modifier, (int, float))

# history_influence captures the outcome
assert plan.history_influence in ("boosted", "penalised", "corrected", "neutral")

# empty context → still produces a plan
empty_ctx = PlanContext(
    user_input="hello world",
    session_id="test-session-2",
    memory_chunks=[],
    past_outcomes=[],
    correction_memories=[],
    nudges=[],
)
plan2 = build_enriched_plan(empty_ctx, specialist="GENERAL_CHAT")
assert len(plan2.plan["steps"]) > 0, "should produce steps even with empty context"
assert plan2.conviction_modifier == 0.0, f"empty context should give 0.0 modifier, got {plan2.conviction_modifier}"
assert plan2.history_influence == "neutral"

print("  build_enriched_plan tests passed")

# ── Summary ──────────────────────────────────────────────────────────
print("\nAll planner tests passed")
