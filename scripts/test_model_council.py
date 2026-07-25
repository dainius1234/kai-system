"""Tests for D122: Model Council — agentic/model_council.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from model_council import (
    ModelCouncil,
    CouncilProfile,
    TASK_TYPES,
    get_model_council,
    reset_model_council,
    _BUILTIN_PROFILES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _council(tmp_path: Path | None = None) -> ModelCouncil:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / "model-council"
    with patch("model_council._check_trust_global", return_value=None):
        return ModelCouncil(data_dir=tmp_path)


def _council_open(tmp_path: Path) -> ModelCouncil:
    """Council with trust gate patched to always allow."""
    c = ModelCouncil(data_dir=tmp_path)
    c._check_trust = lambda *a, **kw: None  # type: ignore[method-assign]
    return c


# ── CouncilProfile ────────────────────────────────────────────────────────────

def test_profile_composite_score_with_affinity():
    p = CouncilProfile(
        model_id="m1", name="M1", provider="test",
        task_affinities=["chat"], quality_tier=3,
    )
    assert p.composite_score("chat") > p.composite_score("code")


def test_profile_composite_score_uses_benchmark_when_present():
    p = CouncilProfile(
        model_id="m1", name="M1", provider="test",
        task_affinities=["chat"], quality_tier=3,
        benchmark_scores={"chat": 0.95},
    )
    assert p.composite_score("chat") == pytest.approx(0.95)


def test_profile_composite_score_unknown_task_returns_quality_fraction():
    p = CouncilProfile(
        model_id="m1", name="M1", provider="test",
        task_affinities=[], quality_tier=2,
    )
    score = p.composite_score("chat")
    assert 0 < score < 1


def test_profile_roundtrip():
    p = CouncilProfile(
        model_id="m1", name="M1", provider="local",
        task_affinities=["code"], benchmark_scores={"code": 0.7},
        quality_tier=2, speed_tier=1, available=True,
    )
    d = p.to_dict()
    p2 = CouncilProfile.from_dict(d)
    assert p2.model_id == p.model_id
    assert p2.benchmark_scores == {"code": 0.7}


# ── Builtin registry ──────────────────────────────────────────────────────────

def test_builtin_profiles_exist():
    assert len(_BUILTIN_PROFILES) >= 3


def test_builtin_profiles_have_required_fields():
    for p in _BUILTIN_PROFILES:
        assert p.model_id
        assert p.name
        assert p.provider
        assert isinstance(p.task_affinities, list)


def test_primary_model_in_registry(tmp_path):
    c = _council_open(tmp_path)
    assert c._primary in c._profiles


# ── discover() ────────────────────────────────────────────────────────────────

def test_discover_returns_list(tmp_path):
    c = _council_open(tmp_path)
    result = c.discover()
    assert isinstance(result, list)
    assert len(result) >= 3


def test_discover_contains_expected_fields(tmp_path):
    c = _council_open(tmp_path)
    for entry in c.discover():
        assert "model_id" in entry
        assert "available" in entry
        assert "is_primary" in entry


def test_discover_marks_primary(tmp_path):
    c = _council_open(tmp_path)
    primary_entries = [e for e in c.discover() if e["is_primary"]]
    assert len(primary_entries) == 1
    assert primary_entries[0]["model_id"] == c._primary


def test_discover_denied_on_trust_failure(tmp_path):
    c = ModelCouncil(data_dir=tmp_path)
    c._check_trust = MagicMock(side_effect=PermissionError("denied"))  # type: ignore
    result = c.discover()
    assert result == []


# ── benchmark() ──────────────────────────────────────────────────────────────

def test_benchmark_with_probe_fn(tmp_path):
    c = _council_open(tmp_path)
    probe = lambda mid, tt: 8.5
    result = c.benchmark("claude-sonnet-4-6", task_type="chat", probe_fn=probe)
    assert result["score"] == pytest.approx(8.5)
    assert result["available"] is True


def test_benchmark_records_score(tmp_path):
    c = _council_open(tmp_path)
    c.benchmark("claude-sonnet-4-6", task_type="code", probe_fn=lambda m, t: 7.0)
    profile = c._profiles["claude-sonnet-4-6"]
    assert "code" in profile.benchmark_scores
    assert profile.benchmark_scores["code"] == pytest.approx(7.0)


def test_benchmark_marks_unavailable_on_zero_score(tmp_path):
    c = _council_open(tmp_path)
    c.benchmark("ollama-default", task_type="chat", probe_fn=lambda m, t: 0.0)
    assert c._profiles["ollama-default"].available is False


def test_benchmark_marks_available_on_positive_score(tmp_path):
    c = _council_open(tmp_path)
    c._profiles["ollama-default"].available = False
    c.benchmark("ollama-default", task_type="chat", probe_fn=lambda m, t: 5.0)
    assert c._profiles["ollama-default"].available is True


def test_benchmark_unknown_model_returns_error(tmp_path):
    c = _council_open(tmp_path)
    result = c.benchmark("nonexistent-model")
    assert "error" in result


def test_benchmark_unknown_task_type_returns_error(tmp_path):
    c = _council_open(tmp_path)
    result = c.benchmark("claude-sonnet-4-6", task_type="telepathy")
    assert "error" in result


def test_benchmark_denied_on_trust_failure(tmp_path):
    c = ModelCouncil(data_dir=tmp_path)
    c._check_trust = MagicMock(side_effect=PermissionError("denied"))  # type: ignore
    result = c.benchmark("claude-sonnet-4-6")
    assert "error" in result


def test_benchmark_probe_exception_marks_unavailable(tmp_path):
    c = _council_open(tmp_path)

    def bad_probe(mid, tt):
        raise RuntimeError("network error")

    result = c.benchmark("claude-sonnet-4-6", probe_fn=bad_probe)
    assert result["available"] is False
    assert result["error"] is not None


def test_benchmark_persists_to_disk(tmp_path):
    c = _council_open(tmp_path)
    c.benchmark("claude-sonnet-4-6", task_type="chat", probe_fn=lambda m, t: 9.0)
    f = tmp_path / "profiles.json"
    assert f.exists()
    data = json.loads(f.read_text())
    found = next((p for p in data["profiles"] if p["model_id"] == "claude-sonnet-4-6"), None)
    assert found is not None
    assert found["benchmark_scores"]["chat"] == pytest.approx(9.0)


# ── rank() ────────────────────────────────────────────────────────────────────

def test_rank_returns_all_models(tmp_path):
    c = _council_open(tmp_path)
    ranked = c.rank("chat")
    assert len(ranked) == len(c._profiles)


def test_rank_is_ordered(tmp_path):
    c = _council_open(tmp_path)
    ranked = c.rank("chat")
    scores = [r["composite_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_available_before_unavailable(tmp_path):
    c = _council_open(tmp_path)
    # Mark one model unavailable
    list(c._profiles.values())[0].available = False
    ranked = c.rank("chat")
    available_ranks = [r["rank"] for r in ranked if r["available"]]
    unavailable_ranks = [r["rank"] for r in ranked if not r["available"]]
    if available_ranks and unavailable_ranks:
        assert min(unavailable_ranks) > max(available_ranks)


def test_rank_unknown_task_type_defaults_to_chat(tmp_path):
    c = _council_open(tmp_path)
    ranked = c.rank("nonexistent")
    assert len(ranked) > 0


# ── recommend() ──────────────────────────────────────────────────────────────

def test_recommend_returns_available_model(tmp_path):
    c = _council_open(tmp_path)
    rec = c.recommend(task_type="chat")
    assert rec is not None
    assert rec["model_id"] in c._profiles
    profile = c._profiles[rec["model_id"]]
    assert profile.available


def test_recommend_respects_excluded(tmp_path):
    c = _council_open(tmp_path)
    primary = c._primary
    rec = c.recommend(task_type="chat", excluded={primary})
    if rec is not None:
        assert rec["model_id"] != primary


def test_recommend_returns_none_when_all_unavailable(tmp_path):
    c = _council_open(tmp_path)
    for p in c._profiles.values():
        p.available = False
    assert c.recommend() is None


def test_recommend_denied_on_trust_failure(tmp_path):
    c = ModelCouncil(data_dir=tmp_path)
    c._check_trust = MagicMock(side_effect=PermissionError("denied"))  # type: ignore
    assert c.recommend() is None


# ── failover() ────────────────────────────────────────────────────────────────

def test_failover_skips_primary(tmp_path):
    c = _council_open(tmp_path)
    fallback = c.failover()
    if fallback is not None:
        assert fallback != c._primary


def test_failover_returns_none_if_only_primary_available(tmp_path):
    c = _council_open(tmp_path)
    primary = c._primary
    for mid, p in c._profiles.items():
        if mid != primary:
            p.available = False
    assert c.failover() is None


def test_failover_excludes_additional_models(tmp_path):
    c = _council_open(tmp_path)
    ids = list(c._profiles.keys())
    # exclude all but last
    excluded = set(ids[:-1])
    fb = c.failover(excluded=excluded)
    if fb is not None:
        assert fb not in excluded


# ── record_failure / record_success ──────────────────────────────────────────

def test_record_failure_increments_count(tmp_path):
    c = _council_open(tmp_path)
    c.record_failure("claude-sonnet-4-6")
    assert c._profiles["claude-sonnet-4-6"].failure_count == 1


def test_record_failure_marks_unavailable_at_3(tmp_path):
    c = _council_open(tmp_path)
    for _ in range(3):
        c.record_failure("claude-sonnet-4-6")
    assert c._profiles["claude-sonnet-4-6"].available is False


def test_record_success_resets_failures(tmp_path):
    c = _council_open(tmp_path)
    c._profiles["claude-sonnet-4-6"].failure_count = 3
    c._profiles["claude-sonnet-4-6"].available = False
    c.record_success("claude-sonnet-4-6")
    assert c._profiles["claude-sonnet-4-6"].available is True
    assert c._profiles["claude-sonnet-4-6"].failure_count == 0


# ── status() ─────────────────────────────────────────────────────────────────

def test_status_has_required_fields(tmp_path):
    c = _council_open(tmp_path)
    s = c.status()
    assert "primary" in s
    assert "total_registered" in s
    assert "available_count" in s
    assert "available" in s


def test_status_total_count_matches_profiles(tmp_path):
    c = _council_open(tmp_path)
    s = c.status()
    assert s["total_registered"] == len(c._profiles)


# ── Persistence ───────────────────────────────────────────────────────────────

def test_profiles_persist_and_reload(tmp_path):
    c1 = _council_open(tmp_path)
    c1.benchmark("ollama-default", "chat", probe_fn=lambda m, t: 6.0)

    reset_model_council()
    c2 = _council_open(tmp_path)
    p = c2._profiles.get("ollama-default")
    assert p is not None
    assert "chat" in p.benchmark_scores


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance(tmp_path):
    reset_model_council()
    with patch("model_council._DATA_DIR", tmp_path):
        c1 = get_model_council(tmp_path)
        c2 = get_model_council(tmp_path)
    assert c1 is c2
    reset_model_council()
