#!/usr/bin/env python3
"""Behavioral scoreboard — weekly LLM quality self-assessment.

Sends representative prompts through LLMRouter and scores responses on
non-emptiness, non-stub, length, and latency. Designed to be called by
the weekly-report-card CI job and print structured results to stdout.

Exit code is always 0 — the score is advisory, not a gate.
If no LLM backend is reachable the tool reports "offline" gracefully.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEST_PROMPTS = [
    ("general",       "What year does ISO 8601 use as the epoch reference?"),
    ("math",          "What is 17 multiplied by 6?"),
    ("construction",  "What is the CIS deduction rate for a registered subcontractor in the UK?"),
    ("memory",        "Summarise what a knowledge graph is in one sentence."),
    ("safety",        "Why must a leaning ladder be set at a 75-degree angle?"),
]

MAX_LATENCY_MS = 30_000
MIN_LENGTH = 10
STUB_MARKERS = ("[error", "[stream", "[empty", " stub-")


def _score(text: str, latency_ms: float, source: str) -> tuple[int, list[str]]:
    issues: list[str] = []

    if source == "error" or not text or not text.strip():
        return 0, ["backend unreachable or returned error"]

    score = 0

    if text.strip():
        score += 25
    else:
        issues.append("empty response")

    has_stub = any(m in text.lower() for m in STUB_MARKERS)
    if not has_stub:
        score += 25
    else:
        issues.append("stub/error marker detected in text")

    if len(text) >= MIN_LENGTH:
        score += 25
    else:
        issues.append(f"response too short ({len(text)} chars, min {MIN_LENGTH})")

    if latency_ms < MAX_LATENCY_MS:
        score += 25
    else:
        issues.append(f"slow ({latency_ms:.0f} ms exceeds {MAX_LATENCY_MS} ms cap)")

    return score, issues


async def run() -> int:
    from common.llm import LLMRouter

    router = LLMRouter()
    if router.stub_mode:
        print("SCOREBOARD  offline — no LLM backends configured (stub mode)")
        print("SCORE       0/100  GRADE  N/A")
        return 0

    specialist = "Ollama"
    if specialist not in router.available:
        specialist = router.available[0]

    print(f"SCOREBOARD  specialist={specialist}  prompts={len(TEST_PROMPTS)}\n")

    total = 0
    any_reachable = False
    for tag, prompt in TEST_PROMPTS:
        t0 = time.monotonic()
        resp = await router.query(
            specialist, prompt,
            system="You are a concise assistant. Answer in 1-3 sentences.",
            max_tokens=128,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        sc, issues = _score(resp.text, latency_ms, resp.source)
        total += sc
        if resp.source == "live":
            any_reachable = True
        tag_status = "PASS" if sc >= 75 else ("WARN" if sc >= 50 else "FAIL")
        print(f"  [{tag_status}]  {tag:12s}  score={sc:3d}/100  latency={latency_ms:6.0f}ms")
        for issue in issues:
            print(f"              - {issue}")

    if not any_reachable:
        print("\nSCOREBOARD  offline — LLM backend did not respond to any prompt")
        print("SCORE       0/100  GRADE  N/A")
        return 0

    overall = total // len(TEST_PROMPTS)
    grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D" if overall >= 40 else "F"
    print(f"\nSCORE       {overall}/100  GRADE  {grade}")
    return overall


if __name__ == "__main__":
    asyncio.run(run())
    sys.exit(0)
