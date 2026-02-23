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

questions = [
    "Summarize risk limits for executor rollback policy with concrete thresholds.",
    "What checks should happen before execution approval in tool gate?",
    "How should memory compression protect keeper critical records?",
    "Give an incident response checklist for high error ratio in last 5 minutes.",
    "Explain how to verify audit chain integrity on startup.",
    "What is the recovery order if primary USB is lost?",
    "How to enforce no-answer-below-conviction-eight in reflection loop?",
    "Describe safe rollback criteria when confidence drops under threshold.",
    "What offline context should be fetched before replying to keeper?",
    "How to reduce hallucinations in autonomous action planning?",
]

scores = []
for q in questions:
    signal.alarm(TIMEOUT_PER_QUESTION)
    try:
        chunks = [{"id": i} for i in range(5)]
        plan = mod.build_plan(q, "DeepSeek-V4", chunks)
        score = mod.score_conviction(q, plan, chunks, 1)
        scores.append(score)
    finally:
        signal.alarm(0)

avg = statistics.mean(scores)
print(f"conviction avg={avg:.2f}")
if avg < 8.0:
    raise SystemExit("conviction average below 8")
