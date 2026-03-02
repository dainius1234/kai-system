"""Tests for the Specialist Router — langgraph/router.py

Validates that user messages are classified into the correct route
and that the zero-LLM dispatch functions return sensible results.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Load router.py directly to avoid the installed langgraph package
_mod_path = Path(__file__).resolve().parents[1] / "langgraph" / "router.py"
_spec = importlib.util.spec_from_file_location("router", _mod_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["router"] = _mod

# router imports from common.self_emp_advisor — ensure common is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_spec.loader.exec_module(_mod)

classify = _mod.classify
RouteDecision = _mod.RouteDecision

# ── Route classification tests ───────────────────────────────────────

_TEST_CASES = [
    # (input, expected_route, description)
    ("do you remember what I said about the car?", "MEMORY_RECALL", "explicit 'remember'"),
    ("what did I say about taxes last time?", "MEMORY_RECALL", "'what did I' + 'last time'"),
    ("find my notes on the meeting", "MEMORY_RECALL", "'find my notes'"),
    ("search for the invoice from last month", "MEMORY_RECALL", "'search for'"),

    ("how much tax do I owe?", "TAX_ADVISORY", "'tax'"),
    ("what's the VAT threshold?", "TAX_ADVISORY", "'VAT' + 'threshold'"),
    ("can I deduct mileage as an expense?", "TAX_ADVISORY", "'deduct' + 'expense'"),
    ("when is my HMRC self-employment deadline?", "TAX_ADVISORY", "'HMRC' + 'self-employ'"),

    ("is it true that the earth is flat?", "FACT_CHECK", "'is it true'"),
    ("verify this claim about climate change", "FACT_CHECK", "'verify'"),
    ("fact check what they said", "FACT_CHECK", "'fact check'"),

    ("what should I know today?", "PROACTIVE_REVIEW", "'what should I know'"),
    ("any reminders for me?", "PROACTIVE_REVIEW", "'any reminder'"),
    ("give me my morning brief", "PROACTIVE_REVIEW", "'morning brief'"),
    ("anything urgent?", "PROACTIVE_REVIEW", "'anything urgent'"),

    ("summarise my week", "REFLECT", "'summarise my week'"),
    ("what have I been working on?", "REFLECT", "'what have I been working on'"),
    ("end of day summary please", "REFLECT", "'end of day summary'"),

    ("run the deploy script", "EXECUTE_ACTION", "'run' + 'deploy'"),
    ("create a new file for the config", "EXECUTE_ACTION", "'create' + 'file'"),
    ("build the docker image", "EXECUTE_ACTION", "'build'"),

    ("compare the two approaches", "MULTI_SIGNAL", "'compare'"),
    ("get me a second opinion on this", "MULTI_SIGNAL", "'second opinion'"),

    # fallback cases — should go to GENERAL_CHAT
    ("hello", "GENERAL_CHAT", "greeting"),
    ("what do you think about the weather?", "GENERAL_CHAT", "casual chat"),
    ("tell me a joke", "GENERAL_CHAT", "entertainment"),
    ("", "GENERAL_CHAT", "empty input"),
]

passed = 0
failed = 0

for user_input, expected_route, description in _TEST_CASES:
    decision = classify(user_input)
    if decision.route == expected_route:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: '{user_input}' → {decision.route} (expected {expected_route}) — {description}")
        print(f"        reason: {decision.reason}, confidence: {decision.confidence}")

# ── Property tests ───────────────────────────────────────────────────

# confidence is always 0-1
for text in ["hello", "remember my notes", "how much tax", "verify this"]:
    d = classify(text)
    assert 0.0 <= d.confidence <= 1.0, f"confidence out of range: {d.confidence}"

# bypass_llm is correct for each route
zero_llm_routes = {"MEMORY_RECALL", "TAX_ADVISORY", "FACT_CHECK", "PROACTIVE_REVIEW", "REFLECT"}
for text, route, _ in _TEST_CASES:
    d = classify(text)
    if d.route in zero_llm_routes:
        assert d.bypass_llm, f"route {d.route} should have bypass_llm=True"
    elif d.route in {"GENERAL_CHAT", "EXECUTE_ACTION", "MULTI_SIGNAL"}:
        assert not d.bypass_llm, f"route {d.route} should have bypass_llm=False"

# context boost works
d1 = classify("what should I know?")
d2 = classify("what should I know?", session_context={"last_route": "PROACTIVE_REVIEW"})
assert d2.confidence >= d1.confidence, "context boost should not decrease confidence"

# ── Summary ──────────────────────────────────────────────────────────
total = passed + failed
print(f"\nRouter tests: {passed}/{total} passed, {failed} failed")
if failed:
    raise SystemExit(1)
print("router classification tests passed")
