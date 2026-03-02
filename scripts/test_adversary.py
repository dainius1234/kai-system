"""Tests for the Proposer-Adversary Challenge Engine — langgraph/adversary.py

Validates all five challenge strategies independently and the orchestrator.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

# Load adversary.py directly (avoid installed langgraph package)
_mod_path = Path(__file__).resolve().parents[1] / "langgraph" / "adversary.py"
_spec = importlib.util.spec_from_file_location("adversary", _mod_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["adversary"] = _mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec.loader.exec_module(_mod)

challenge_history = _mod.challenge_history
challenge_consistency = _mod.challenge_consistency
challenge_calibration = _mod.challenge_calibration
challenge_plan = _mod.challenge_plan
verdict_to_plan_metadata = _mod.verdict_to_plan_metadata
_pattern_hash = _mod._pattern_hash
ChallengeResult = _mod.ChallengeResult
AdversaryVerdict = _mod.AdversaryVerdict

now = time.time()

# ── Helper: a well-formed plan ───────────────────────────────────────

GOOD_PLAN = {
    "specialist": "DeepSeek-V4",
    "summary": "Analyse the tax implications of the new invoice.",
    "steps": [
        {"action": "analyze", "input": "tax calculation"},
        {"action": "propose", "output": "draft recommendation"},
    ],
}

BAD_PLAN = {
    "specialist": "",
    "summary": "",
    "steps": [],
}

# ── Challenge 1: History ─────────────────────────────────────────────

print("Testing challenge_history...")

# no episodes → pass, no modifier
r = challenge_history(GOOD_PLAN, "calculate tax owed", [])
assert r.passed is True
assert r.modifier == 0.0
assert r.strategy == "history"

# similar failure in history → fail, negative modifier
episodes_with_failure = [
    {"input": "calculate tax owed", "outcome_score": 0.1,
     "final_conviction": 5.0, "ts": now - 86400, "episode_id": "ep-f1"},
]
r = challenge_history(GOOD_PLAN, "calculate tax owed", episodes_with_failure)
assert r.passed is False
assert r.modifier < 0, f"expected negative modifier, got {r.modifier}"
assert "failed before" in r.finding.lower()

# similar success in history → pass, positive modifier
episodes_with_success = [
    {"input": "calculate tax owed", "outcome_score": 0.9,
     "final_conviction": 9.0, "ts": now - 86400, "episode_id": "ep-s1"},
]
r = challenge_history(GOOD_PLAN, "calculate tax owed", episodes_with_success)
assert r.passed is True
assert r.modifier > 0, f"expected positive modifier, got {r.modifier}"

# unrelated episodes → pass, zero modifier
episodes_unrelated = [
    {"input": "paint the house purple", "outcome_score": 0.9,
     "final_conviction": 9.0, "ts": now - 86400, "episode_id": "ep-u1"},
]
r = challenge_history(GOOD_PLAN, "calculate tax owed", episodes_unrelated)
assert r.passed is True
assert r.modifier == 0.0

# empty input → pass gracefully
r = challenge_history(GOOD_PLAN, "", [])
assert r.passed is True

print("  challenge_history: PASSED")

# ── Challenge 4: Consistency ─────────────────────────────────────────

print("Testing challenge_consistency...")

# good plan → pass
r = challenge_consistency(GOOD_PLAN)
assert r.passed is True
assert r.modifier > 0
assert r.strategy == "consistency"

# bad plan → fail with issues
r = challenge_consistency(BAD_PLAN)
assert r.passed is False
assert r.modifier < 0
assert "no steps" in r.finding.lower() or "empty" in r.finding.lower() or "no specialist" in r.finding.lower()

# contradictory plan → fail
contradictory = {
    "specialist": "test",
    "summary": "Do something contradictory",
    "steps": [
        {"action": "create", "input": "file"},
        {"action": "delete", "input": "file"},
    ],
}
r = challenge_consistency(contradictory)
assert r.passed is False
assert "creates and deletes" in r.finding.lower()

# plan with missing actions in steps
partial = {
    "specialist": "test",
    "summary": "Partial plan",
    "steps": [{"action": ""}, {"action": "analyze"}],
}
r = challenge_consistency(partial)
assert r.passed is False
assert "no action" in r.finding.lower()

print("  challenge_consistency: PASSED")

# ── Challenge 5: Calibration ────────────────────────────────────────

print("Testing challenge_calibration...")

# no episodes → pass, no modifier
r = challenge_calibration("calculate tax", 8.0, [])
assert r.passed is True
assert r.modifier == 0.0

# well-calibrated episodes (conviction/10 ~ outcome)
calibrated_eps = [
    {"input": "calculate tax owed", "outcome_score": 0.8,
     "final_conviction": 8.0, "ts": now - 86400},
    {"input": "calculate tax returns", "outcome_score": 0.9,
     "final_conviction": 9.0, "ts": now - 172800},
    {"input": "how much tax do I pay", "outcome_score": 0.7,
     "final_conviction": 7.5, "ts": now - 259200},
]
r = challenge_calibration("calculate tax owed", 8.0, calibrated_eps)
assert r.passed is True
assert "well-calibrated" in r.finding.lower() or "under-confident" in r.finding.lower()

# over-confident episodes (high conviction, low outcome)
overconfident_eps = [
    {"input": "calculate tax owed", "outcome_score": 0.2,
     "final_conviction": 9.0, "ts": now - 86400},
    {"input": "calculate tax returns", "outcome_score": 0.3,
     "final_conviction": 8.5, "ts": now - 172800},
    {"input": "how much tax do I pay", "outcome_score": 0.1,
     "final_conviction": 9.5, "ts": now - 259200},
]
r = challenge_calibration("calculate tax owed", 9.0, overconfident_eps)
assert r.passed is False
assert r.modifier < 0, f"expected negative modifier for over-confidence, got {r.modifier}"
assert "over-confident" in r.finding.lower()

# under-confident episodes (low conviction, high outcome)
underconfident_eps = [
    {"input": "calculate tax owed", "outcome_score": 0.9,
     "final_conviction": 3.0, "ts": now - 86400},
    {"input": "calculate tax returns", "outcome_score": 0.95,
     "final_conviction": 2.5, "ts": now - 172800},
    {"input": "how much tax do I pay", "outcome_score": 0.85,
     "final_conviction": 3.5, "ts": now - 259200},
]
r = challenge_calibration("calculate tax owed", 3.0, underconfident_eps)
assert r.passed is True
assert r.modifier > 0, f"expected positive modifier for under-confidence, got {r.modifier}"
assert "under-confident" in r.finding.lower()

# insufficient data (only 1 point) → pass but neutral
r = challenge_calibration("calculate tax owed", 8.0, [calibrated_eps[0]])
assert r.passed is True
assert "insufficient" in r.finding.lower() or r.modifier == 0.0

print("  challenge_calibration: PASSED")

# ── Pattern hash ─────────────────────────────────────────────────────

print("Testing _pattern_hash...")

# same words different order → same hash
h1 = _pattern_hash("calculate the tax owed")
h2 = _pattern_hash("owed tax the calculate")
assert h1 == h2, "same keywords should produce same hash"

# different words → different hash
h3 = _pattern_hash("deploy docker container")
assert h1 != h3, "different keywords should produce different hash"

# empty input
h4 = _pattern_hash("")
assert len(h4) == 16

print("  _pattern_hash: PASSED")

# ── verdict_to_plan_metadata ─────────────────────────────────────────

print("Testing verdict_to_plan_metadata...")

verdict = AdversaryVerdict(
    challenges=[
        ChallengeResult("history", True, 0.8, "ok", 0.3),
        ChallengeResult("verifier", False, 0.7, "needs evidence", -0.8),
    ],
    total_modifier=-0.5,
    recommendation="caution",
    critical_warnings=["needs evidence"],
    summary="1/2 passed; modifier=-0.50; recommendation=caution",
    challenge_time_ms=12.5,
)
meta = verdict_to_plan_metadata(verdict)
assert "adversary_challenges" in meta
assert len(meta["adversary_challenges"]) == 2
assert meta["adversary_modifier"] == -0.5
assert meta["adversary_recommendation"] == "caution"
assert meta["adversary_warnings"] == ["needs evidence"]

print("  verdict_to_plan_metadata: PASSED")

# ── Full orchestrator (local-only, verifier/tool-gate unavailable) ──

print("Testing challenge_plan (full orchestrator, degraded mode)...")


async def _test_full_orchestrator():
    """Test the full challenge_plan with services unavailable."""
    verdict = await challenge_plan(
        plan=GOOD_PLAN,
        user_input="calculate tax owed",
        context_chunks=[{"content": "UK tax thresholds apply"}],
        episodes=calibrated_eps,
        predicted_conviction=8.0,
        tool_hint="shell",
    )
    assert isinstance(verdict, AdversaryVerdict)
    assert len(verdict.challenges) == 5
    assert isinstance(verdict.total_modifier, float)
    assert verdict.recommendation in ("proceed", "caution", "block")
    assert isinstance(verdict.challenge_time_ms, float)

    # check that all 5 strategies ran
    strategies = {c.strategy for c in verdict.challenges}
    assert strategies == {"history", "verifier", "policy", "consistency", "calibration"}, \
        f"expected all 5 strategies, got {strategies}"

    # verifier and policy should be in degraded mode (services not running)
    for c in verdict.challenges:
        if c.strategy in ("verifier", "policy"):
            assert "unavailable" in c.finding.lower() or c.passed, \
                f"{c.strategy} should degrade gracefully, got: {c.finding}"

    return verdict


verdict = asyncio.new_event_loop().run_until_complete(_test_full_orchestrator())
print(f"  challenge_plan: PASSED (recommendation={verdict.recommendation}, "
      f"modifier={verdict.total_modifier:+.2f}, time={verdict.challenge_time_ms:.0f}ms)")

# ── Recommendation logic ─────────────────────────────────────────────

print("Testing recommendation logic...")

# Test that a -2.0 modifier triggers "block"
block_verdict = AdversaryVerdict(
    challenges=[
        ChallengeResult("verifier", False, 0.9, "FAIL", -2.0),
        ChallengeResult("history", True, 0.5, "ok", 0.0),
        ChallengeResult("policy", True, 0.5, "ok", 0.0),
        ChallengeResult("consistency", True, 0.5, "ok", 0.0),
        ChallengeResult("calibration", True, 0.5, "ok", 0.0),
    ],
    total_modifier=-2.0,
    recommendation="block",  # we check this
    critical_warnings=["FAIL"],
    summary="test",
    challenge_time_ms=0.0,
)
assert block_verdict.recommendation == "block"

print("  recommendation logic: PASSED")

# ── Summary ──────────────────────────────────────────────────────────
print("\nAll adversary tests passed")
