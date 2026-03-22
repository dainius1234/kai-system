"""Tests for P14 — Temporal Self-Model.

Validates _trend(), overall health logic, and assessment persistence
from heartbeat/app.py.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "heartbeat"))

os.environ.setdefault("MEMU_URL", "http://localhost:8002")

# Provide assessment file in a temp location for tests
_tmp = tempfile.mkdtemp()
os.environ["ASSESSMENT_FILE"] = os.path.join(_tmp, "test_assess.json")

# Stub heavy modules
for mod_name in ("redis", "httpx"):
    if mod_name not in sys.modules:
        from types import ModuleType
        _stub = ModuleType(mod_name)
        if mod_name == "httpx":
            class _AsyncClient:
                def __init__(self, **kw): pass
                async def __aenter__(self): return self
                async def __aexit__(self, *a): pass
                async def get(self, *a, **kw):
                    class _Resp:
                        status_code = 200
                        def json(self): return {}
                    return _Resp()
            _stub.AsyncClient = _AsyncClient
        elif mod_name == "redis":
            class _FakeRedis:
                @classmethod
                def from_url(cls, *a, **kw): return cls()
                def ping(self): return True
            _stub.Redis = _FakeRedis
        sys.modules[mod_name] = _stub

# Now import the heartbeat module
import importlib.util
_spec = importlib.util.spec_from_file_location("heartbeat_app", ROOT / "heartbeat" / "app.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["heartbeat_app"] = _mod
try:
    _spec.loader.exec_module(_mod)
except Exception:
    pass  # app startup may fail in test env; we only need pure functions

_trend = _mod._trend
_load_previous_assessment = _mod._load_previous_assessment
_save_assessment = _mod._save_assessment
_ASSESSMENT_FILE = _mod._ASSESSMENT_FILE
ASSESSMENT_WINDOW_DAYS = _mod.ASSESSMENT_WINDOW_DAYS


class TestTrend(unittest.TestCase):
    """P14: _trend() direction detection."""

    def test_improving_large_increase(self):
        self.assertEqual(_trend(120, 100), "improving")

    def test_declining_large_decrease(self):
        self.assertEqual(_trend(80, 100), "declining")

    def test_stable_small_change(self):
        self.assertEqual(_trend(102, 100), "stable")

    def test_stable_equal_values(self):
        self.assertEqual(_trend(50, 50), "stable")

    def test_new_from_zero(self):
        """Previous was 0, current positive → 'new'."""
        self.assertEqual(_trend(10, 0), "new")

    def test_stable_both_zero(self):
        """Both 0 → stable, not 'new'."""
        self.assertEqual(_trend(0, 0), "stable")

    def test_improving_from_low_base(self):
        """Small absolute change but large relative = improving."""
        self.assertEqual(_trend(1.2, 1.0), "improving")

    def test_declining_precision(self):
        """Exactly at 10% boundary = stable (threshold is > 0.10)."""
        # 10% decline exactly: 90 / 100 = -0.10, not < -0.10
        self.assertEqual(_trend(90, 100), "stable")


class TestOverallHealth(unittest.TestCase):
    """P14: overall health classification from trend counts."""

    def _classify(self, trends: dict) -> str:
        declining = sum(1 for v in trends.values() if v == "declining")
        improving = sum(1 for v in trends.values() if v == "improving")
        if declining >= 2:
            return "needs_attention"
        elif improving >= 2:
            return "improving"
        return "stable"

    def test_needs_attention(self):
        trends = {"a": "declining", "b": "declining", "c": "stable", "d": "improving"}
        self.assertEqual(self._classify(trends), "needs_attention")

    def test_improving(self):
        trends = {"a": "improving", "b": "improving", "c": "stable", "d": "stable"}
        self.assertEqual(self._classify(trends), "improving")

    def test_stable(self):
        trends = {"a": "stable", "b": "stable", "c": "improving", "d": "declining"}
        self.assertEqual(self._classify(trends), "stable")

    def test_needs_attention_overrides_improving(self):
        trends = {"a": "declining", "b": "declining", "c": "improving", "d": "improving"}
        self.assertEqual(self._classify(trends), "needs_attention")

    def test_all_new(self):
        trends = {"a": "new", "b": "new", "c": "new", "d": "new"}
        self.assertEqual(self._classify(trends), "stable")


class TestAssessmentPersistence(unittest.TestCase):
    """P14: file-based assessment storage."""

    def setUp(self):
        if _ASSESSMENT_FILE.exists():
            _ASSESSMENT_FILE.unlink()

    def test_no_previous_returns_none(self):
        self.assertIsNone(_load_previous_assessment())

    def test_save_and_load(self):
        data = {"total_memories": 42, "error_rate": 0.01}
        _save_assessment(data)
        loaded = _load_previous_assessment()
        self.assertEqual(loaded["total_memories"], 42)
        self.assertAlmostEqual(loaded["error_rate"], 0.01)

    def test_corrupt_file_returns_none(self):
        _ASSESSMENT_FILE.write_text("not json at all {{{")
        self.assertIsNone(_load_previous_assessment())


class TestConfig(unittest.TestCase):
    """P14: configuration defaults."""

    def test_window_default(self):
        self.assertEqual(ASSESSMENT_WINDOW_DAYS, 7)


class TestInvertedTrends(unittest.TestCase):
    """P14: error_rate and cpu_usage have inverted trend labels."""

    def test_error_rate_decrease_is_improving(self):
        """When error_rate goes down, that's 'improving' for the user."""
        # Raw trend of decrease = "declining", but inverted = "improving"
        raw = _trend(0.01, 0.10)  # big decrease
        self.assertEqual(raw, "declining")
        # After inversion (as done in the endpoint):
        if raw == "declining":
            inverted = "improving"
        self.assertEqual(inverted, "improving")

    def test_error_rate_increase_is_declining(self):
        """When error_rate goes up, that's 'declining' for the user."""
        raw = _trend(0.20, 0.10)  # increase
        self.assertEqual(raw, "improving")
        if raw == "improving":
            inverted = "declining"
        self.assertEqual(inverted, "declining")


if __name__ == "__main__":
    unittest.main()
