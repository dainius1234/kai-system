"""Tests for P5 planner preference injection + P6 boundary in planner context.

Validates that preferences are fetched and injected into plans,
and that the planner's gather_context includes the preferences field.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph"))
sys.path.insert(0, str(ROOT / "common"))

# Stub redis for kai_config import
if "redis" not in sys.modules:
    from types import ModuleType
    _r = ModuleType("redis")
    class _FakeRedis:
        @classmethod
        def from_url(cls, *a, **kw):
            return cls()
        def ping(self):
            return True
    _r.Redis = _FakeRedis
    sys.modules["redis"] = _r

from planner import (
    PlanContext,
    PlanDecision,
    build_enriched_plan,
    gather_context,
    _fetch_preferences,
    _keyword_similarity,
)


class TestPlanContextPreferences(unittest.TestCase):
    """PlanContext should include preferences field."""

    def test_plan_context_has_preferences(self):
        ctx = PlanContext(user_input="test", session_id="s1")
        self.assertEqual(ctx.preferences, [])

    def test_plan_context_stores_preferences(self):
        prefs = [{"content": {"text": "keeper prefers brevity"}}]
        ctx = PlanContext(user_input="test", session_id="s1", preferences=prefs)
        self.assertEqual(len(ctx.preferences), 1)


class TestBuildEnrichedPlanWithPreferences(unittest.TestCase):
    """build_enriched_plan should inject preferences into plan steps."""

    def test_no_preferences_no_preference_steps(self):
        ctx = PlanContext(user_input="deploy server", session_id="s1")
        decision = build_enriched_plan(ctx, "general")
        pref_steps = [s for s in decision.plan["steps"] if s.get("action") == "apply_preference"]
        self.assertEqual(len(pref_steps), 0)

    def test_preferences_injected_into_steps(self):
        prefs = [
            {"content": {"text": "keeper prefers conservative risk limits"}},
            {"content": {"text": "keeper wants daily reporting"}},
        ]
        ctx = PlanContext(user_input="set risk limits", session_id="s1", preferences=prefs)
        decision = build_enriched_plan(ctx, "general")
        pref_steps = [s for s in decision.plan["steps"] if s.get("action") == "apply_preference"]
        self.assertEqual(len(pref_steps), 2)
        self.assertIn("conservative", pref_steps[0]["preference"])

    def test_preferences_counted_in_plan(self):
        prefs = [{"content": {"text": "pref 1"}}, {"content": {"text": "pref 2"}}]
        ctx = PlanContext(user_input="test", session_id="s1", preferences=prefs)
        decision = build_enriched_plan(ctx, "general")
        self.assertEqual(decision.plan["preferences_applied"], 2)

    def test_max_five_preferences_injected(self):
        """At most 5 preferences should be injected."""
        prefs = [{"content": {"text": f"pref {i}"}} for i in range(10)]
        ctx = PlanContext(user_input="test", session_id="s1", preferences=prefs)
        decision = build_enriched_plan(ctx, "general")
        pref_steps = [s for s in decision.plan["steps"] if s.get("action") == "apply_preference"]
        self.assertLessEqual(len(pref_steps), 5)


class TestFetchPreferences(unittest.TestCase):
    """_fetch_preferences should GET /memory/preferences."""

    def test_returns_empty_on_failure(self):
        """Should return [] if memu-core is unavailable."""
        result = asyncio.get_event_loop().run_until_complete(
            _fetch_preferences("http://nonexistent:9999")
        )
        self.assertEqual(result, [])


class TestGatherContextIncludesPreferences(unittest.TestCase):
    """gather_context should fetch and include preferences."""

    def test_gather_context_populates_preferences(self):
        """Preferences should be populated in PlanContext after gather."""
        fake_prefs = [{"content": {"text": "keeper prefers X"}}]

        async def _run():
            with patch("planner._fetch_memory_chunks", new_callable=AsyncMock, return_value=[]), \
                 patch("planner._fetch_correction_memories", new_callable=AsyncMock, return_value=[]), \
                 patch("planner._fetch_nudges", new_callable=AsyncMock, return_value=[]), \
                 patch("planner._fetch_preferences", new_callable=AsyncMock, return_value=fake_prefs):
                ctx = await gather_context("test input", "s1", [])
                return ctx

        ctx = asyncio.get_event_loop().run_until_complete(_run())
        self.assertEqual(ctx.preferences, fake_prefs)


if __name__ == "__main__":
    unittest.main()
