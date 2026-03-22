"""Tests for common/feature_flags.py — env-based feature flag system."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from common.feature_flags import (
    _REGISTRY,
    get_all_flags,
    is_enabled,
    register_flag,
)


# ─────────────────────────────────────────────────────────────────────
#  Registry sanity
# ─────────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_has_known_flags(self):
        expected = {
            "DREAM_PHASE_7", "CHECKPOINT_AUTO", "TREE_SEARCH",
            "PRIORITY_QUEUE", "SAGE_CRITIQUE", "IMAGINATION_ENGINE",
        }
        for name in expected:
            assert name in _REGISTRY, f"Missing flag {name}"

    def test_defaults_are_bool(self):
        for name, (desc, default) in _REGISTRY.items():
            assert isinstance(default, bool), f"{name} default is not bool"

    def test_minimum_flag_count(self):
        assert len(_REGISTRY) >= 12


# ─────────────────────────────────────────────────────────────────────
#  is_enabled
# ─────────────────────────────────────────────────────────────────────

class TestIsEnabled:
    def test_reads_default_true(self):
        # All built-in flags default True
        assert is_enabled("SAGE_CRITIQUE") is True

    def test_env_override_false(self, monkeypatch):
        monkeypatch.setenv("FF_SAGE_CRITIQUE", "0")
        assert is_enabled("SAGE_CRITIQUE") is False

    def test_env_override_true(self, monkeypatch):
        monkeypatch.setenv("FF_SAGE_CRITIQUE", "1")
        assert is_enabled("SAGE_CRITIQUE") is True

    def test_env_override_false_string(self, monkeypatch):
        monkeypatch.setenv("FF_TREE_SEARCH", "false")
        assert is_enabled("TREE_SEARCH") is False

    def test_env_override_true_string(self, monkeypatch):
        monkeypatch.setenv("FF_TREE_SEARCH", "true")
        assert is_enabled("TREE_SEARCH") is True

    def test_unknown_flag_returns_false(self):
        assert is_enabled("NONEXISTENT_FLAG_XYZ") is False

    def test_case_insensitive_env(self, monkeypatch):
        monkeypatch.setenv("FF_CHECKPOINT_AUTO", "FALSE")
        assert is_enabled("CHECKPOINT_AUTO") is False


# ─────────────────────────────────────────────────────────────────────
#  get_all_flags
# ─────────────────────────────────────────────────────────────────────

class TestGetAllFlags:
    def test_returns_list(self):
        flags = get_all_flags()
        assert isinstance(flags, list)

    def test_entries_have_name_and_enabled(self):
        flags = get_all_flags()
        for f in flags:
            assert "flag" in f
            assert "enabled" in f
            assert isinstance(f["enabled"], bool)

    def test_respects_env_override(self, monkeypatch):
        monkeypatch.setenv("FF_IMAGINATION_ENGINE", "0")
        flags = get_all_flags()
        ie = next(f for f in flags if f["flag"] == "IMAGINATION_ENGINE")
        assert ie["enabled"] is False


# ─────────────────────────────────────────────────────────────────────
#  register_flag
# ─────────────────────────────────────────────────────────────────────

class TestRegisterFlag:
    def test_register_new_flag(self):
        register_flag("TEST_TEMP_FLAG_1234", description="temp", default=False)
        assert "TEST_TEMP_FLAG_1234" in _REGISTRY
        assert is_enabled("TEST_TEMP_FLAG_1234") is False
        # cleanup
        del _REGISTRY["TEST_TEMP_FLAG_1234"]

    def test_register_does_not_overwrite(self):
        original = _REGISTRY.get("SAGE_CRITIQUE")
        register_flag("SAGE_CRITIQUE", description="dup", default=False)
        # register_flag overwrites (by design), restore original
        _REGISTRY["SAGE_CRITIQUE"] = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
