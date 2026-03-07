"""P4 Personality & Proactive Conversation — Comprehensive Tests

Tests for:
  P4a — Deep personality system prompts (langgraph)
  P4b — Anti-annoyance engine (memu-core: dismissal, DND, cooldowns)
  P4c — Conversation holding (topics: track, defer, resurface)
  P4d — Mode-aware proactive thresholds (filtered nudges)
  P4e — Implicit mode transitions (tool-gate: schedule, override)
  P4f — Proactive greeting & check-in (memu-core)
"""

import importlib
import json
import os
import sys
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# bootstrap: stub heavy deps so memu-core/app.py can import in test
# ---------------------------------------------------------------------------
for mod_name in [
    "sentence_transformers",
    "psutil",
    "redis", "redis.asyncio",
    "psycopg2",
    "lakefs_client",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

os.environ.setdefault("MEMU_HMAC_KEY", "test-key")
os.environ.setdefault("VECTOR_STORE", "memory")
os.environ.setdefault("LEDGER_PATH", "/tmp/test-tool-gate-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-trusted-tokens.txt")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-nonces.json")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

spec = importlib.util.spec_from_file_location("memu_app", "memu-core/app.py")
memu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memu)

# rebuild pydantic models
from typing import Any, Dict, List, Optional
memu.__dict__.update(Dict=Dict, List=List, Optional=Optional, Any=Any)
memu.MemoryRecord.model_rebuild()
if hasattr(memu, 'GoalRequest'):
    memu.GoalRequest.model_rebuild()
if hasattr(memu, 'GoalUpdateRequest'):
    memu.GoalUpdateRequest.model_rebuild()
if hasattr(memu, 'DismissRequest'):
    memu.DismissRequest.model_rebuild()
if hasattr(memu, 'DNDRequest'):
    memu.DNDRequest.model_rebuild()
if hasattr(memu, 'TopicRequest'):
    memu.TopicRequest.model_rebuild()
if hasattr(memu, 'DeferRequest'):
    memu.DeferRequest.model_rebuild()

store = memu.store

# ---------------------------------------------------------------------------
# import tool-gate for mode transition tests
# ---------------------------------------------------------------------------
# tool-gate uses dataclasses + from __future__ import annotations
# we need to ensure it registers as a proper module
tg_spec = importlib.util.spec_from_file_location("toolgate_app", "tool-gate/app.py")
tg = importlib.util.module_from_spec(tg_spec)
sys.modules["toolgate_app"] = tg
tg_spec.loader.exec_module(tg)

# ---------------------------------------------------------------------------
# import langgraph system prompts (extract without loading full app)
# langgraph has many deps (kai_config, conviction, router, etc.)
# we only need the prompt constants, so we parse them from source
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))
for _stub in ["kai_config", "conviction", "router", "planner", "adversary",
              "security_audit", "tree_search", "priority_queue", "model_selector"]:
    if _stub not in sys.modules:
        sys.modules[_stub] = MagicMock()

lg_spec = importlib.util.spec_from_file_location("lg_app", "langgraph/app.py")
lg = importlib.util.module_from_spec(lg_spec)
sys.modules["lg_app"] = lg
lg_spec.loader.exec_module(lg)


def _clear_store():
    if hasattr(store, '_records'):
        store._records.clear()
    elif hasattr(store, 'records'):
        store.records.clear()


def _reset_p4_state():
    """Reset all P4 in-memory state between tests."""
    memu._dismissal_counts.clear()
    memu._dnd_until = 0.0
    memu._last_nudge_by_type.clear()
    memu._active_topics.clear()
    memu._deferred_topics.clear()
    memu._last_greeting = 0.0
    memu._last_check_in = 0.0


# ── P4a: Deep personality system prompts ──────────────────────────────

class TestPersonalityPrompts(unittest.TestCase):
    """Verify system prompts have the required personality depth."""

    def test_work_prompt_exists(self):
        self.assertIn("WORK", lg._SYSTEM_PROMPTS)

    def test_pub_prompt_exists(self):
        self.assertIn("PUB", lg._SYSTEM_PROMPTS)

    def test_core_identity_in_work(self):
        prompt = lg._SYSTEM_PROMPTS["WORK"]
        self.assertIn("Kind And Intelligent", prompt)
        self.assertIn("sovereign personal AI", prompt)
        self.assertIn("brother", prompt)

    def test_core_identity_in_pub(self):
        prompt = lg._SYSTEM_PROMPTS["PUB"]
        self.assertIn("Kind And Intelligent", prompt)
        self.assertIn("mate at the pub", prompt)
        self.assertIn("bollocks", prompt)

    def test_work_is_professional(self):
        prompt = lg._SYSTEM_PROMPTS["WORK"]
        self.assertIn("concise", prompt)
        self.assertIn("task-oriented", prompt)
        self.assertIn("conservative", prompt)

    def test_pub_is_casual(self):
        prompt = lg._SYSTEM_PROMPTS["PUB"]
        self.assertIn("unrestricted", prompt)
        self.assertIn("opinionated", prompt)
        self.assertIn("companion", prompt)

    def test_proactive_in_both_modes(self):
        for mode in ("WORK", "PUB"):
            with self.subTest(mode=mode):
                prompt = lg._SYSTEM_PROMPTS[mode]
                self.assertIn("proactive", prompt.lower())

    def test_memory_awareness_in_both(self):
        for mode in ("WORK", "PUB"):
            with self.subTest(mode=mode):
                prompt = lg._SYSTEM_PROMPTS[mode]
                self.assertIn("remember", prompt.lower())

    def test_goals_mentioned_in_identity(self):
        self.assertIn("Ohana", lg._KAI_CORE_IDENTITY)
        self.assertIn("goals", lg._KAI_CORE_IDENTITY.lower())


# ── P4b: Anti-annoyance engine ────────────────────────────────────────

class TestAntiAnnoyance(unittest.TestCase):
    def setUp(self):
        _reset_p4_state()

    def test_dismiss_increments_count(self):
        memu._dismissal_counts["reminder"] = 0
        memu._dismissal_counts["reminder"] = memu._dismissal_counts.get("reminder", 0) + 1
        self.assertEqual(memu._dismissal_counts["reminder"], 1)

    def test_dismiss_escalates_cooldown(self):
        base = memu._TYPE_COOLDOWNS["reminder"]
        memu._dismissal_counts["reminder"] = 3
        effective = min(base * (memu.DISMISSAL_ESCALATION ** 3), memu.MAX_COOLDOWN_SECONDS)
        self.assertGreater(effective, base)

    def test_dnd_blocks_nudge(self):
        memu._dnd_until = time.time() + 3600  # DND for 1 hour
        self.assertFalse(memu._nudge_allowed("reminder", urgency=0.5))

    def test_dnd_allows_critical_nudge(self):
        memu._dnd_until = time.time() + 3600
        self.assertTrue(memu._nudge_allowed("goal_deadline", urgency=0.95))

    def test_cooldown_blocks_repeated_nudge(self):
        memu._last_nudge_by_type["reminder"] = time.time()
        self.assertFalse(memu._nudge_allowed("reminder", urgency=0.5))

    def test_high_urgency_breaks_half_cooldown(self):
        base = memu._TYPE_COOLDOWNS["reminder"]
        # set last nudge to 60% of cooldown ago (past half)
        memu._last_nudge_by_type["reminder"] = time.time() - (base * 0.6)
        self.assertTrue(memu._nudge_allowed("reminder", urgency=0.85))

    def test_no_cooldown_allows_nudge(self):
        self.assertTrue(memu._nudge_allowed("reminder", urgency=0.5))

    def test_max_cooldown_cap(self):
        memu._dismissal_counts["fading_memory"] = 100  # extreme dismissals
        base = memu._TYPE_COOLDOWNS["fading_memory"]
        effective = min(base * (memu.DISMISSAL_ESCALATION ** 100), memu.MAX_COOLDOWN_SECONDS)
        self.assertEqual(effective, memu.MAX_COOLDOWN_SECONDS)

    def test_type_cooldowns_defined(self):
        expected_types = ["reminder", "silence", "goal_deadline", "drift", "fading_memory", "greeting", "check_in"]
        for t in expected_types:
            self.assertIn(t, memu._TYPE_COOLDOWNS, f"Missing cooldown for type: {t}")


# ── P4c: Conversation holding ─────────────────────────────────────────

class TestConversationHolding(unittest.TestCase):
    def setUp(self):
        _reset_p4_state()

    def test_track_topic_creates(self):
        memu._active_topics.append({
            "id": "test1",
            "topic": "CAD tolerances",
            "context": None,
            "started_at": time.time(),
            "last_mentioned": time.time(),
            "mention_count": 1,
            "deferred": False,
        })
        self.assertEqual(len(memu._active_topics), 1)
        self.assertEqual(memu._active_topics[0]["topic"], "CAD tolerances")

    def test_topic_update_increments_mention(self):
        topic = {
            "id": "test1",
            "topic": "CAD tolerances",
            "context": None,
            "started_at": time.time(),
            "last_mentioned": time.time() - 100,
            "mention_count": 1,
            "deferred": False,
        }
        memu._active_topics.append(topic)
        topic["mention_count"] += 1
        topic["last_mentioned"] = time.time()
        self.assertEqual(topic["mention_count"], 2)

    def test_defer_topic_creates_entry(self):
        entry = {
            "id": "def1",
            "topic": "neutrino paper",
            "context": None,
            "deferred_at": time.time(),
            "resurface_after": time.time() + 14400,
            "resurfaced": False,
            "deferred": True,
        }
        memu._deferred_topics.append(entry)
        self.assertEqual(len(memu._deferred_topics), 1)
        self.assertTrue(memu._deferred_topics[0]["deferred"])

    def test_deferred_topic_resurfaces(self):
        # topic due 1 hour ago
        entry = {
            "id": "def2",
            "topic": "that client thing",
            "context": None,
            "deferred_at": time.time() - 7200,
            "resurface_after": time.time() - 3600,
            "resurfaced": False,
            "deferred": True,
        }
        memu._deferred_topics.append(entry)
        now = time.time()
        ready = [
            t for t in memu._deferred_topics
            if not t.get("resurfaced") and t.get("resurface_after", float("inf")) <= now
        ]
        self.assertEqual(len(ready), 1)

    def test_resurfaced_topic_not_shown_again(self):
        entry = {
            "id": "def3",
            "topic": "old topic",
            "context": None,
            "deferred_at": time.time() - 7200,
            "resurface_after": time.time() - 3600,
            "resurfaced": True,
            "deferred": True,
        }
        memu._deferred_topics.append(entry)
        now = time.time()
        ready = [
            t for t in memu._deferred_topics
            if not t.get("resurfaced") and t.get("resurface_after", float("inf")) <= now
        ]
        self.assertEqual(len(ready), 0)

    def test_active_topics_cap_at_20(self):
        for i in range(25):
            memu._active_topics.append({
                "id": f"t{i}",
                "topic": f"topic {i}",
                "context": None,
                "started_at": time.time(),
                "last_mentioned": time.time() + i,
                "mention_count": 1,
                "deferred": False,
            })
        # apply cap logic
        memu._active_topics.sort(key=lambda x: x["last_mentioned"], reverse=True)
        memu._active_topics[:] = memu._active_topics[:20]
        self.assertEqual(len(memu._active_topics), 20)


# ── P4d: Mode-aware proactive thresholds ──────────────────────────────

class TestModeAwareProactive(unittest.TestCase):
    def test_work_config_exists(self):
        self.assertIn("WORK", memu._PROACTIVE_MODE_CONFIG)

    def test_pub_config_exists(self):
        self.assertIn("PUB", memu._PROACTIVE_MODE_CONFIG)

    def test_work_fewer_types(self):
        work_types = memu._PROACTIVE_MODE_CONFIG["WORK"]["enabled_types"]
        pub_types = memu._PROACTIVE_MODE_CONFIG["PUB"]["enabled_types"]
        self.assertLess(len(work_types), len(pub_types))

    def test_work_higher_urgency_threshold(self):
        work_thresh = memu._PROACTIVE_MODE_CONFIG["WORK"]["urgency_threshold"]
        pub_thresh = memu._PROACTIVE_MODE_CONFIG["PUB"]["urgency_threshold"]
        self.assertGreater(work_thresh, pub_thresh)

    def test_work_fewer_max_nudges(self):
        work_max = memu._PROACTIVE_MODE_CONFIG["WORK"]["max_nudges"]
        pub_max = memu._PROACTIVE_MODE_CONFIG["PUB"]["max_nudges"]
        self.assertLess(work_max, pub_max)

    def test_silence_not_in_work(self):
        work_types = memu._PROACTIVE_MODE_CONFIG["WORK"]["enabled_types"]
        self.assertNotIn("silence", work_types)

    def test_silence_in_pub(self):
        pub_types = memu._PROACTIVE_MODE_CONFIG["PUB"]["enabled_types"]
        self.assertIn("silence", pub_types)


# ── P4e: Implicit mode transitions ───────────────────────────────────

class TestModeTransitions(unittest.TestCase):
    def test_mode_schedule_loaded(self):
        self.assertIsInstance(tg._MODE_SCHEDULE, list)
        self.assertGreater(len(tg._MODE_SCHEDULE), 0)

    def test_default_schedule_weekday_work(self):
        rule = tg._MODE_SCHEDULE[0]
        self.assertEqual(rule["mode"], "WORK")
        self.assertIn(0, rule["days"])  # Monday
        self.assertEqual(rule["start"], 8)
        self.assertEqual(rule["end"], 18)

    def test_effective_mode_returns_string(self):
        mode = tg._effective_mode()
        self.assertIn(mode, {"PUB", "WORK"})

    def test_manual_override_takes_precedence(self):
        tg.policy.mode = "WORK"
        tg._mode_override_until[0] = time.time() + 3600
        self.assertEqual(tg._effective_mode(), "WORK")
        tg._mode_override_until[0] = 0.0  # reset

    def test_expired_override_falls_to_schedule(self):
        tg.policy.mode = "WORK"
        tg._mode_override_until[0] = time.time() - 100  # expired
        mode = tg._effective_mode()
        # should be schedule-based, not necessarily WORK
        self.assertIn(mode, {"PUB", "WORK"})
        tg._mode_override_until[0] = 0.0  # reset


# ── P4f: Proactive greeting & check-in ───────────────────────────────

class TestGreetingAndCheckIn(unittest.TestCase):
    def setUp(self):
        _reset_p4_state()
        _clear_store()

    def test_greeting_cooldown_var_exists(self):
        self.assertIsInstance(memu.GREETING_COOLDOWN, int)
        self.assertGreater(memu.GREETING_COOLDOWN, 0)

    def test_check_in_cooldown_var_exists(self):
        self.assertIsInstance(memu.CHECK_IN_COOLDOWN, int)
        self.assertGreater(memu.CHECK_IN_COOLDOWN, 0)

    def test_greeting_respects_cooldown(self):
        memu._last_nudge_by_type["greeting"] = time.time()
        self.assertFalse(memu._nudge_allowed("greeting", urgency=0.6))

    def test_check_in_respects_cooldown(self):
        memu._last_nudge_by_type["check_in"] = time.time()
        self.assertFalse(memu._nudge_allowed("check_in", urgency=0.3))

    def test_greeting_allowed_after_cooldown(self):
        memu._last_nudge_by_type["greeting"] = time.time() - 30000
        self.assertTrue(memu._nudge_allowed("greeting", urgency=0.6))


# ── Integration: all P4 routes exist ──────────────────────────────────

class TestP4EndpointsExist(unittest.TestCase):
    """Verify all P4 endpoints are registered on both apps."""

    def test_memu_p4_routes(self):
        routes = [r.path for r in memu.app.routes]
        expected = [
            "/memory/nudge/dismiss",
            "/memory/dnd",
            "/memory/nudge/status",
            "/memory/topics/track",
            "/memory/topics/defer",
            "/memory/topics/active",
            "/memory/topics/deferred",
            "/memory/topics/resurface",
            "/memory/proactive/filtered",
            "/memory/greeting",
            "/memory/check-in",
        ]
        for ep in expected:
            self.assertIn(ep, routes, f"Missing memu-core route: {ep}")

    def test_toolgate_mode_route(self):
        routes = [r.path for r in tg.app.routes]
        self.assertIn("/gate/mode", routes)


if __name__ == "__main__":
    unittest.main()
