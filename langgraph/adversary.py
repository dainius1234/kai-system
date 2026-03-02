"""Proposer-Adversary Challenge Engine — langgraph/adversary.py

The adversary is Kai's self-doubt engine.  Before any plan reaches the
executor, it must survive five independent challenges.  Each challenge
attacks the plan from a different angle:

1. History challenge  — "Did something like this fail before?"
2. Verifier challenge — "Is the core claim factually supported?"
3. Policy challenge   — "Does the gate policy allow this tool/action?"
4. Consistency check  — "Is the plan internally coherent?"
5. Calibration check  — "Were we right when we were this confident before?"

The adversary returns an AdversaryVerdict that includes:
- An aggregated modifier to add to the conviction score
- Per-challenge findings for transparency
- Go/no-go recommendation
- Specific warnings the planner should surface

This is Kai's unfair advantage #5: no other build stress-tests its own
plans before execution.

Usage:
    from adversary import challenge_plan, AdversaryVerdict
    verdict = await challenge_plan(plan, context, episodes)
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

# ── Data structures ──────────────────────────────────────────────────

@dataclass
class ChallengeResult:
    """Result of a single challenge strategy."""
    strategy: str           # e.g. "history", "verifier", "policy"
    passed: bool            # True = plan survived this challenge
    confidence: float       # 0.0–1.0 how confident the challenge is
    finding: str            # human-readable explanation
    modifier: float         # conviction modifier (-2.0 to +1.0)
    evidence: List[str] = field(default_factory=list)


@dataclass
class CalibrationPoint:
    """Tracks predicted vs actual outcome for calibration analysis."""
    pattern_hash: str       # SHA-256 of input keywords
    predicted_conviction: float
    actual_outcome: float   # 0.0–1.0 from episode outcome_score
    timestamp: float


@dataclass
class AdversaryVerdict:
    """Aggregated result of all challenges."""
    challenges: List[ChallengeResult]
    total_modifier: float           # sum of all challenge modifiers
    recommendation: str             # "proceed", "caution", "block"
    critical_warnings: List[str]    # warnings that should block execution
    summary: str                    # human-readable summary
    challenge_time_ms: float        # total time for all challenges


# ── Challenge strategies ─────────────────────────────────────────────

def challenge_history(
    plan: Dict[str, Any],
    user_input: str,
    episodes: List[Dict[str, Any]],
) -> ChallengeResult:
    """Challenge 1: Check if similar plans have failed before.

    Searches episode history for inputs with keyword overlap.
    If past outcomes were failures (outcome_score < 0.4), this
    challenge penalises the plan.  If past outcomes were successes,
    it provides a small boost.

    This uses the same keyword similarity as planner.py but with
    a focus on failure detection.
    """
    if not episodes:
        return ChallengeResult(
            strategy="history",
            passed=True,
            confidence=0.3,
            finding="No episode history available — cannot challenge from history.",
            modifier=0.0,
        )

    input_words = set(re.findall(r"\w{3,}", user_input.lower()))
    if not input_words:
        return ChallengeResult(
            strategy="history",
            passed=True,
            confidence=0.3,
            finding="Input too short for meaningful history comparison.",
            modifier=0.0,
        )

    failures: List[Dict[str, Any]] = []
    successes: List[Dict[str, Any]] = []

    for ep in episodes:
        ep_input = str(ep.get("input", ""))
        ep_words = set(re.findall(r"\w{3,}", ep_input.lower()))
        if not ep_words:
            continue
        overlap = len(input_words & ep_words) / len(input_words | ep_words)
        if overlap < 0.25:
            continue

        outcome = float(ep.get("outcome_score", 0.5))
        ep["_similarity"] = overlap
        if outcome < 0.4:
            failures.append(ep)
        elif outcome >= 0.7:
            successes.append(ep)

    if failures:
        worst = max(failures, key=lambda e: e["_similarity"])
        return ChallengeResult(
            strategy="history",
            passed=False,
            confidence=min(worst["_similarity"] + 0.2, 1.0),
            finding=(
                f"Similar request failed before (similarity={worst['_similarity']:.0%}, "
                f"outcome={worst.get('outcome_score', '?')}): "
                f"{str(worst.get('input', ''))[:120]}"
            ),
            modifier=-1.5 * worst["_similarity"],
            evidence=[str(worst.get("input", ""))[:200]],
        )

    if successes:
        best = max(successes, key=lambda e: e["_similarity"])
        return ChallengeResult(
            strategy="history",
            passed=True,
            confidence=min(best["_similarity"] + 0.1, 1.0),
            finding=(
                f"Similar request succeeded before (similarity={best['_similarity']:.0%}, "
                f"outcome={best.get('outcome_score', '?')})"
            ),
            modifier=0.3 * best["_similarity"],
            evidence=[str(best.get("input", ""))[:200]],
        )

    return ChallengeResult(
        strategy="history",
        passed=True,
        confidence=0.4,
        finding="No closely similar episodes found in history.",
        modifier=0.0,
    )


async def challenge_verifier(
    plan: Dict[str, Any],
    user_input: str,
    context_chunks: List[Dict[str, Any]],
) -> ChallengeResult:
    """Challenge 2: Ask the Verifier to assess the plan's core claims.

    Sends the plan summary + user input to verifier /verify.
    If verdict is FAIL_CLOSED → hard penalty.
    If verdict is REPAIR → moderate penalty.
    If verdict is PASS → small boost.

    Graceful degradation: if verifier is down, passes with low confidence.
    """
    verifier_url = os.getenv("VERIFIER_URL", "http://verifier:8052")
    claim = f"{user_input}\n\nPlan summary: {plan.get('summary', 'no summary')}"
    evidence_pack = [
        {"content": str(c.get("content", c) if isinstance(c, dict) else c)[:500], "rank_score": 0.5}
        for c in context_chunks[:10]
    ]

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                f"{verifier_url}/verify",
                json={
                    "claim": claim[:2000],
                    "context": user_input[:500],
                    "source": "adversary",
                    "plan": plan,
                    "top_k": 10,
                    "evidence_pack": evidence_pack,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return ChallengeResult(
            strategy="verifier",
            passed=True,
            confidence=0.2,
            finding="Verifier unavailable — challenge skipped (degraded mode).",
            modifier=0.0,
        )

    verdict = data.get("verdict", "UNKNOWN")
    ver_confidence = float(data.get("confidence", 0.5))
    strong_chunks = int(data.get("strong_chunks", 0))
    evidence_summary = str(data.get("evidence_summary", ""))

    if verdict == "FAIL_CLOSED":
        return ChallengeResult(
            strategy="verifier",
            passed=False,
            confidence=ver_confidence,
            finding=f"Verifier FAIL_CLOSED (confidence={ver_confidence:.2f}, strong_chunks={strong_chunks}): {evidence_summary[:200]}",
            modifier=-2.0,
            evidence=[evidence_summary[:500]],
        )
    elif verdict == "REPAIR":
        return ChallengeResult(
            strategy="verifier",
            passed=False,
            confidence=ver_confidence,
            finding=f"Verifier REPAIR (confidence={ver_confidence:.2f}, strong_chunks={strong_chunks}): needs more evidence.",
            modifier=-0.8,
            evidence=[evidence_summary[:500]],
        )
    else:  # PASS
        return ChallengeResult(
            strategy="verifier",
            passed=True,
            confidence=ver_confidence,
            finding=f"Verifier PASS (confidence={ver_confidence:.2f}, strong_chunks={strong_chunks}).",
            modifier=0.3,
        )


async def challenge_policy(
    plan: Dict[str, Any],
    tool_hint: Optional[str],
) -> ChallengeResult:
    """Challenge 3: Pre-check whether tool-gate policy would allow this.

    Checks the tool-gate /health to see current mode (PUB/WORK).
    In PUB mode, most tool actions are blocked — flag this early.

    Also checks if the plan's tool is in the known risky category.
    """
    tool_gate_url = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
    mode = "WORK"  # default assumption

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{tool_gate_url}/health")
            resp.raise_for_status()
            health = resp.json()
            mode = health.get("mode", "WORK")
    except Exception:
        return ChallengeResult(
            strategy="policy",
            passed=True,
            confidence=0.2,
            finding="Tool-gate unavailable — cannot pre-check policy (degraded mode).",
            modifier=0.0,
        )

    warnings = []

    # PUB mode = all execute actions blocked
    if mode == "PUB":
        if tool_hint:
            return ChallengeResult(
                strategy="policy",
                passed=False,
                confidence=0.95,
                finding=f"Gate is in PUB mode — tool '{tool_hint}' will be blocked.",
                modifier=-3.0,
                evidence=[f"Current mode: {mode}"],
            )
        # even without explicit tool hint, flag it
        warnings.append("Gate is in PUB mode — execution actions may be restricted.")

    # check for inherently risky tools
    risky_tools = {"shell", "script", "python"}
    if tool_hint and tool_hint.lower() in risky_tools:
        warnings.append(f"Tool '{tool_hint}' is in the high-risk category — may require co-sign.")

    if warnings:
        return ChallengeResult(
            strategy="policy",
            passed=True,
            confidence=0.7,
            finding="; ".join(warnings),
            modifier=-0.3,
            evidence=[f"mode={mode}, tool={tool_hint}"],
        )

    return ChallengeResult(
        strategy="policy",
        passed=True,
        confidence=0.8,
        finding=f"Policy pre-check OK (mode={mode}).",
        modifier=0.1,
    )


def challenge_consistency(plan: Dict[str, Any]) -> ChallengeResult:
    """Challenge 4: Check if the plan is internally consistent.

    Verifies:
    - Plan has steps with actions
    - Plan has a non-empty summary
    - Steps don't contain contradictory actions
    - Specialist is assigned
    """
    issues: List[str] = []

    # must have steps
    steps = plan.get("steps", [])
    if not steps:
        issues.append("Plan has no steps.")

    # must have a summary
    summary = str(plan.get("summary", ""))
    if len(summary) < 5:
        issues.append("Plan summary is empty or too short.")

    # must have a specialist
    specialist = plan.get("specialist", "")
    if not specialist:
        issues.append("No specialist assigned.")

    # steps should have actions
    actions = [s.get("action", "") for s in steps if isinstance(s, dict)]
    empty_actions = [a for a in actions if not a]
    if empty_actions:
        issues.append(f"{len(empty_actions)} step(s) have no action defined.")

    # check for contradictory actions (e.g., create + delete same target)
    action_set = set(actions)
    contradictions = []
    if "create" in action_set and "delete" in action_set:
        contradictions.append("Plan both creates and deletes — verify this is intentional.")
    if "start" in action_set and "stop" in action_set:
        contradictions.append("Plan both starts and stops — verify this is intentional.")
    if contradictions:
        issues.extend(contradictions)

    if issues:
        return ChallengeResult(
            strategy="consistency",
            passed=False,
            confidence=0.9,
            finding=f"Plan consistency issues: {'; '.join(issues)}",
            modifier=-0.5 * len(issues),
            evidence=issues,
        )

    return ChallengeResult(
        strategy="consistency",
        passed=True,
        confidence=0.85,
        finding=f"Plan is consistent ({len(steps)} steps, specialist={specialist}).",
        modifier=0.2,
    )


def challenge_calibration(
    user_input: str,
    predicted_conviction: float,
    episodes: List[Dict[str, Any]],
) -> ChallengeResult:
    """Challenge 5: Check if our confidence predictions match reality.

    Looks at past episodes with similar inputs and compares their
    conviction scores to their actual outcomes.  If the system has
    been consistently over-confident for this pattern, penalise.
    If under-confident, boost.

    This is Kai's meta-learning advantage — it tracks whether its
    own confidence is calibrated.
    """
    if not episodes:
        return ChallengeResult(
            strategy="calibration",
            passed=True,
            confidence=0.2,
            finding="No calibration data available yet.",
            modifier=0.0,
        )

    input_words = set(re.findall(r"\w{3,}", user_input.lower()))
    if not input_words:
        return ChallengeResult(
            strategy="calibration",
            passed=True,
            confidence=0.2,
            finding="Input too short for calibration analysis.",
            modifier=0.0,
        )

    # find episodes with similar inputs
    calibration_points: List[CalibrationPoint] = []
    for ep in episodes:
        ep_input = str(ep.get("input", ""))
        ep_words = set(re.findall(r"\w{3,}", ep_input.lower()))
        if not ep_words:
            continue
        overlap = len(input_words & ep_words) / len(input_words | ep_words)
        if overlap < 0.2:
            continue

        ep_conviction = float(ep.get("final_conviction", ep.get("conviction_score", 0)))
        ep_outcome = float(ep.get("outcome_score", 0.5))

        calibration_points.append(CalibrationPoint(
            pattern_hash=_pattern_hash(ep_input),
            predicted_conviction=ep_conviction,
            actual_outcome=ep_outcome,
            timestamp=float(ep.get("ts", 0)),
        ))

    if len(calibration_points) < 2:
        return ChallengeResult(
            strategy="calibration",
            passed=True,
            confidence=0.3,
            finding=f"Only {len(calibration_points)} calibration point(s) — insufficient for analysis.",
            modifier=0.0,
        )

    # Calculate calibration drift
    # conviction is 0-10, outcome is 0-1 — normalise conviction to 0-1
    drifts = []
    for cp in calibration_points:
        normalised_conviction = cp.predicted_conviction / 10.0
        drift = normalised_conviction - cp.actual_outcome
        drifts.append(drift)

    avg_drift = sum(drifts) / len(drifts)

    # positive drift = over-confident (predicted high, got low outcome)
    # negative drift = under-confident (predicted low, got high outcome)

    if avg_drift > 0.15:
        # system is over-confident for this pattern
        penalty = min(avg_drift * 2.0, 1.5)
        return ChallengeResult(
            strategy="calibration",
            passed=False,
            confidence=min(0.5 + len(calibration_points) * 0.1, 0.9),
            finding=(
                f"Calibration drift detected: system is OVER-CONFIDENT for this "
                f"pattern (avg drift={avg_drift:+.2f} across {len(calibration_points)} "
                f"similar episodes). Past conviction scores overestimated actual outcomes."
            ),
            modifier=-penalty,
            evidence=[f"drift={avg_drift:+.3f}", f"points={len(calibration_points)}"],
        )
    elif avg_drift < -0.15:
        # system is under-confident — small boost
        boost = min(abs(avg_drift), 0.5)
        return ChallengeResult(
            strategy="calibration",
            passed=True,
            confidence=min(0.5 + len(calibration_points) * 0.1, 0.9),
            finding=(
                f"Calibration check: system is UNDER-CONFIDENT for this "
                f"pattern (avg drift={avg_drift:+.2f} across {len(calibration_points)} "
                f"similar episodes). Past outcomes exceeded predictions."
            ),
            modifier=boost,
            evidence=[f"drift={avg_drift:+.3f}", f"points={len(calibration_points)}"],
        )
    else:
        return ChallengeResult(
            strategy="calibration",
            passed=True,
            confidence=min(0.5 + len(calibration_points) * 0.1, 0.9),
            finding=(
                f"Calibration is well-calibrated for this pattern "
                f"(avg drift={avg_drift:+.2f} across {len(calibration_points)} episodes)."
            ),
            modifier=0.1,
        )


# ── Orchestration ────────────────────────────────────────────────────

async def challenge_plan(
    plan: Dict[str, Any],
    user_input: str,
    context_chunks: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
    predicted_conviction: float,
    tool_hint: Optional[str] = None,
) -> AdversaryVerdict:
    """Run all five challenges against the plan.

    Challenges 1, 4, 5 are local (no network), so they're always fast.
    Challenges 2, 3 require network (verifier, tool-gate) but degrade
    gracefully if services are down.

    Returns an AdversaryVerdict with aggregated modifiers and findings.
    """
    import asyncio
    start = time.monotonic()

    # Run network challenges in parallel, local challenges synchronously
    verifier_task = asyncio.create_task(
        challenge_verifier(plan, user_input, context_chunks)
    )
    policy_task = asyncio.create_task(
        challenge_policy(plan, tool_hint)
    )

    # Local challenges (instant)
    history_result = challenge_history(plan, user_input, episodes)
    consistency_result = challenge_consistency(plan)
    calibration_result = challenge_calibration(
        user_input, predicted_conviction, episodes
    )

    # Await network challenges
    verifier_result = await verifier_task
    policy_result = await policy_task

    elapsed_ms = (time.monotonic() - start) * 1000

    challenges = [
        history_result,
        verifier_result,
        policy_result,
        consistency_result,
        calibration_result,
    ]

    # Aggregate
    total_modifier = sum(c.modifier for c in challenges)
    critical_warnings = [
        c.finding for c in challenges
        if not c.passed and c.confidence >= 0.7
    ]
    failed_count = sum(1 for c in challenges if not c.passed)
    passed_count = sum(1 for c in challenges if c.passed)

    # Recommendation logic
    if any(c.modifier <= -2.0 for c in challenges):
        recommendation = "block"
    elif failed_count >= 3:
        recommendation = "block"
    elif failed_count >= 1:
        recommendation = "caution"
    else:
        recommendation = "proceed"

    # Build human-readable summary
    summary_parts = [f"{passed_count}/5 challenges passed"]
    if critical_warnings:
        summary_parts.append(f"{len(critical_warnings)} critical warning(s)")
    summary_parts.append(f"modifier={total_modifier:+.2f}")
    summary_parts.append(f"recommendation={recommendation}")

    return AdversaryVerdict(
        challenges=challenges,
        total_modifier=round(total_modifier, 2),
        recommendation=recommendation,
        critical_warnings=critical_warnings,
        summary="; ".join(summary_parts),
        challenge_time_ms=round(elapsed_ms, 1),
    )


# ── Helpers ──────────────────────────────────────────────────────────

def _pattern_hash(text: str) -> str:
    """Generate a structural hash of the input's keyword pattern.

    This fingerprints the *type* of request, not the exact words.
    Two inputs with the same keywords in different order get the same hash.
    """
    words = sorted(set(re.findall(r"\w{3,}", text.lower())))
    return hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]


def verdict_to_plan_metadata(verdict: AdversaryVerdict) -> Dict[str, Any]:
    """Convert an AdversaryVerdict into metadata to inject into the plan dict.

    This makes adversary findings visible in the plan output, dashboard,
    and episode history — full transparency.
    """
    return {
        "adversary_challenges": [
            {
                "strategy": c.strategy,
                "passed": c.passed,
                "confidence": round(c.confidence, 3),
                "finding": c.finding,
                "modifier": round(c.modifier, 2),
            }
            for c in verdict.challenges
        ],
        "adversary_modifier": verdict.total_modifier,
        "adversary_recommendation": verdict.recommendation,
        "adversary_warnings": verdict.critical_warnings,
        "adversary_summary": verdict.summary,
        "adversary_time_ms": verdict.challenge_time_ms,
    }
