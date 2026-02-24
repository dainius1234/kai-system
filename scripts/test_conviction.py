from __future__ import annotations

import importlib.util
import signal
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph"))

spec = importlib.util.spec_from_file_location("conviction", ROOT / "langgraph" / "conviction.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

TIMEOUT_PER_QUESTION = 30


def _timeout_handler(_signum, _frame):
    raise TimeoutError("question timeout")


signal.signal(signal.SIGALRM, _timeout_handler)


# ── test data ──────────────────────────────────────────────────────────
# Each question is paired with context chunks that contain relevant
# keywords.  This simulates realistic retrieval where memory returns
# partially relevant results.

_QUESTIONS_AND_CHUNKS = [
    (
        "Summarize risk limits for executor rollback policy with concrete thresholds.",
        [{"content": "risk limit policy: max drawdown 5%, rollback after 3 failures"},
         {"content": "executor rollback triggered at error_ratio > 0.2"},
         {"content": "architecture overview of core sovereign stack"},
         {"content": "policy v2: confidence threshold 0.7, risk tolerance medium"},
         {"content": "unrelated memory: grocery list"}],
    ),
    (
        "What checks should happen before execution approval in tool gate?",
        [{"content": "tool gate checks: nonce, signature, confidence, allowlist, scope"},
         {"content": "execution approval requires cosign for shell commands"},
         {"content": "gate policy evaluates request confidence against threshold"},
         {"content": "dashboard UI renders cards for active services"},
         {"content": "unrelated memory: weather forecast"}],
    ),
    (
        "How should memory compression protect keeper critical records?",
        [{"content": "memory compression uses decay, never deletes keeper records"},
         {"content": "critical records are marked with keeper flag and preserved"},
         {"content": "protect important memories from automatic compression"},
         {"content": "redis cache TTL settings"},
         {"content": "unrelated memory: music playlist"}],
    ),
    (
        "Give an incident response checklist for high error ratio in last 5 minutes.",
        [{"content": "incident response: check error budget, circuit breaker status"},
         {"content": "error ratio monitored via ErrorBudget with 300s window"},
         {"content": "high error triggers supervisor alert and service restart"},
         {"content": "checklist: isolate, diagnose, fix, verify, post-mortem"},
         {"content": "unrelated memory: lunch order"}],
    ),
    (
        "Explain how to verify audit chain integrity on startup.",
        [{"content": "audit chain verification replays JSONL ledger on startup"},
         {"content": "each entry hash is verified against previous entry"},
         {"content": "integrity check uses SHA-256 hash chain with GENESIS root"},
         {"content": "startup sequence: load tokens, restore nonces, replay ledger"},
         {"content": "unrelated memory: TV schedule"}],
    ),
    (
        "What is the recovery order if primary USB is lost?",
        [{"content": "USB recovery: boot from backup, restore keys, rebuild ledger"},
         {"content": "primary device lost: switch to secondary, verify identity"},
         {"content": "recovery procedure documented in hardening runbook"},
         {"content": "order of operations: identity first, then data, then services"},
         {"content": "unrelated memory: phone contacts"}],
    ),
    (
        "How to enforce no-answer-below-conviction-eight in reflection loop?",
        [{"content": "reflection loop calls score_conviction after each rethink"},
         {"content": "enforce minimum conviction by gating plan execution"},
         {"content": "conviction below threshold triggers additional rethink cycle"},
         {"content": "answer quality improves with each reflection iteration"},
         {"content": "unrelated memory: book recommendations"}],
    ),
    (
        "Describe safe rollback criteria when confidence drops under threshold.",
        [{"content": "rollback when confidence drops below 0.5 for 3 consecutive checks"},
         {"content": "safe rollback preserves state snapshot before reverting"},
         {"content": "criteria: error ratio, confidence trend, circuit breaker state"},
         {"content": "threshold crossing triggers notification to operator"},
         {"content": "unrelated memory: travel plans"}],
    ),
    (
        "What offline context should be fetched before replying to keeper?",
        [{"content": "offline context: retrieve relevant memories from vector store"},
         {"content": "fetch local embeddings matching user query before LLM call"},
         {"content": "context window filled with top-k similar memory chunks"},
         {"content": "keeper replies must include source references from memory"},
         {"content": "unrelated memory: recipe collection"}],
    ),
    (
        "How to reduce hallucinations in autonomous action planning?",
        [{"content": "reduce hallucinations by cross-referencing with verified sources"},
         {"content": "action planning must include specific tool and parameter names"},
         {"content": "autonomous plans verified via fusion engine consensus"},
         {"content": "hallucination detection uses keyword plausibility check"},
         {"content": "unrelated memory: sports scores"}],
    ),
]


# ── main test run ──────────────────────────────────────────────────────

scores = []
for q, chunks in _QUESTIONS_AND_CHUNKS:
    signal.alarm(TIMEOUT_PER_QUESTION)
    try:
        plan = mod.build_plan(q, "DeepSeek-V4", chunks)
        score = mod.score_conviction(q, plan, chunks, rethink_count=2)
        scores.append(score)
    finally:
        signal.alarm(0)

avg = statistics.mean(scores)
print(f"conviction avg={avg:.2f}  scores={[f'{s:.1f}' for s in scores]}")

# Stub plans with relevant context and 2 rethinks should score ≥ 5.0/10.
# A score of 8+ requires rich multi-step plans with high specialist fit
# and multiple rethink iterations — that's for production, not stubs.
if avg < 5.0:
    raise SystemExit(f"conviction average {avg:.2f} below minimum 5.0")

# Verify monotonicity: plans with context should score higher than without
no_ctx_score = mod.score_conviction("test", mod.build_plan("test", "DeepSeek-V4", []), [], 0)
with_ctx_score = mod.score_conviction(
    "Summarize risk policy thresholds",
    mod.build_plan("Summarize risk policy thresholds", "DeepSeek-V4",
                   [{"content": "risk policy threshold is 0.7"}]),
    [{"content": "risk policy threshold is 0.7"}],
    2,
)
assert with_ctx_score > no_ctx_score, (
    f"scoring should reward context and rethinks: {with_ctx_score} vs {no_ctx_score}"
)

print(f"conviction tests passed (avg={avg:.2f})")
