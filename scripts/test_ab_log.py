"""C10 A/B query logger tests.

Verifies that common/ab_log.py writes valid JSONL rows and that the
feature can be disabled via AB_LOG_ENABLED=false.

Run with:
    PYTHONPATH=. python scripts/test_ab_log.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.ab_log import log_ab_entry, _quality_fields
from common.llm import LLMResponse


def _make_resp(
    specialist: str = "DeepSeek-V4",
    model: str = "deepseek-v4",
    source: str = "stub",
    text: str = "The VAT threshold is £90,000 for the 2024–25 tax year.",
    latency_ms: float = 45.2,
    usage: dict | None = None,
) -> LLMResponse:
    return LLMResponse(
        specialist=specialist,
        text=text,
        latency_ms=latency_ms,
        source=source,
        model=model,
        usage=usage or {"prompt_tokens": 10, "completion_tokens": 15},
    )


# ── 1. Basic write ────────────────────────────────────────────────────

def test_writes_jsonl_row(tmp_path: Path) -> None:
    log_path = tmp_path / "ab.jsonl"
    os.environ["AB_LOG_PATH"] = str(log_path)
    os.environ["AB_LOG_ENABLED"] = "true"

    import importlib
    import common.ab_log as ab_mod
    importlib.reload(ab_mod)

    resp = _make_resp()
    ab_mod.log_ab_entry(resp, prompt="What is the VAT threshold?", session_id="s-test-1")

    assert log_path.exists(), "log file should be created"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1, f"expected 1 line, got {len(lines)}"
    row = json.loads(lines[0])

    assert row["specialist"] == "DeepSeek-V4"
    assert row["model"] == "deepseek-v4"
    assert row["source"] == "stub"
    assert row["latency_ms"] == 45.2
    assert row["session_id"] == "s-test-1"
    assert "ts" in row
    assert "prompt_hash" in row and len(row["prompt_hash"]) == 8
    assert "word_count" in row
    assert "lexical_diversity" in row
    assert "net_quality_signal" in row
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 15


# ── 2. Multiple rows accumulate ───────────────────────────────────────

def test_multiple_rows_append(tmp_path: Path) -> None:
    log_path = tmp_path / "multi.jsonl"
    os.environ["AB_LOG_PATH"] = str(log_path)
    os.environ["AB_LOG_ENABLED"] = "true"

    import importlib
    import common.ab_log as ab_mod
    importlib.reload(ab_mod)

    for i in range(5):
        resp = _make_resp(specialist=f"Model-{i}", model=f"model-{i}")
        ab_mod.log_ab_entry(resp, prompt=f"query {i}")

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert row["specialist"] == f"Model-{i}"


# ── 3. Disabled via env ───────────────────────────────────────────────

def test_disabled_writes_nothing(tmp_path: Path) -> None:
    log_path = tmp_path / "disabled.jsonl"
    os.environ["AB_LOG_PATH"] = str(log_path)
    os.environ["AB_LOG_ENABLED"] = "false"

    import importlib
    import common.ab_log as ab_mod
    importlib.reload(ab_mod)

    resp = _make_resp()
    ab_mod.log_ab_entry(resp, prompt="Should not appear")

    assert not log_path.exists(), "log file must not be created when disabled"


# ── 4. Quality field correctness ─────────────────────────────────────

def test_quality_fields_range() -> None:
    texts = [
        "The VAT threshold for self-employed traders is £90,000 per year.",
        "",
        "Maybe perhaps possibly uncertain unclear not sure approximately roughly " * 5,
        " ".join([str(i) for i in range(100)]),
    ]
    for text in texts:
        q = _quality_fields(text)
        assert 0.0 <= q["lexical_diversity"] <= 1.0, f"diversity out of range for: {text[:40]!r}"
        assert 0.0 <= q["uncertainty_penalty"] <= 0.5, f"penalty out of range for: {text[:40]!r}"
        assert isinstance(q["word_count"], int)
        net = round(q["lexical_diversity"] - q["uncertainty_penalty"], 3)
        assert abs(q["net_quality_signal"] - net) < 0.001


# ── 5. No session_id is OK ────────────────────────────────────────────

def test_no_session_id(tmp_path: Path) -> None:
    log_path = tmp_path / "nosession.jsonl"
    os.environ["AB_LOG_PATH"] = str(log_path)
    os.environ["AB_LOG_ENABLED"] = "true"

    import importlib
    import common.ab_log as ab_mod
    importlib.reload(ab_mod)

    ab_mod.log_ab_entry(_make_resp(), prompt="hello")
    row = json.loads(log_path.read_text().strip())
    assert row["session_id"] is None


# ── 6. prompt_hash is stable ──────────────────────────────────────────

def test_prompt_hash_stability() -> None:
    import hashlib
    prompt = "What is the CIS deduction rate?"
    expected = hashlib.sha256(prompt[:200].encode()).hexdigest()[:8]

    import common.ab_log as ab_mod
    # call the function directly through a temp path
    with tempfile.TemporaryDirectory() as td:
        os.environ["AB_LOG_PATH"] = str(Path(td) / "hash.jsonl")
        os.environ["AB_LOG_ENABLED"] = "true"
        import importlib; importlib.reload(ab_mod)
        ab_mod.log_ab_entry(_make_resp(), prompt=prompt)
        row = json.loads(Path(td, "hash.jsonl").read_text().strip())
        assert row["prompt_hash"] == expected, f"expected {expected}, got {row['prompt_hash']}"


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_writes_jsonl_row(tp / "t1")
        print("  ✓ test_writes_jsonl_row")
        test_multiple_rows_append(tp / "t2")
        print("  ✓ test_multiple_rows_append")
        test_disabled_writes_nothing(tp / "t3")
        print("  ✓ test_disabled_writes_nothing")
    test_quality_fields_range()
    print("  ✓ test_quality_fields_range")
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_no_session_id(tp / "t5")
        print("  ✓ test_no_session_id")
        test_prompt_hash_stability()
        print("  ✓ test_prompt_hash_stability")

    print("\nab_log tests: 6/6 passed")
