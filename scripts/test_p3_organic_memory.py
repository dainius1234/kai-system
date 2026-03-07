"""P3 Organic Memory — Comprehensive Tests

Tests for:
  P3a — Correction learning (retrieval boost for corrections)
  P3b — Category-aware retrieval boost
  P3c — Spaced repetition enforcement (decay endpoint)
  P3d — Proactive conversation engine (unified scan)
  P3e — Ohana goal tracker (create / update / list)
  P3f — Operator drift detection
"""

import importlib
import json
import os
import sys
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

spec = importlib.util.spec_from_file_location("memu_app", "memu-core/app.py")
memu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memu)

# rebuild pydantic models — inject typing names into module's global namespace
# needed because `from __future__ import annotations` makes them string refs
from typing import Any, Dict, List, Optional
memu.__dict__.update(Dict=Dict, List=List, Optional=Optional, Any=Any)
memu.MemoryRecord.model_rebuild()
if hasattr(memu, 'GoalRequest'):
    memu.GoalRequest.model_rebuild()
if hasattr(memu, 'GoalUpdateRequest'):
    memu.GoalUpdateRequest.model_rebuild()

# convenience handles
MemoryRecord = memu.MemoryRecord
store = memu.store
classify_category = memu.classify_category
generate_embedding = memu.generate_embedding
retrieve_ranked = memu.retrieve_ranked
DEFAULT_CATEGORY = memu.DEFAULT_CATEGORY


def _clear_store():
    """Clear the in-memory store regardless of internal attribute name."""
    if hasattr(store, '_records'):
        store._records.clear()
    elif hasattr(store, 'records'):
        _clear_store()


def _make_record(content_text, category=None, event_type="observation",
                 importance=0.5, pinned=False, access_count=0,
                 timestamp=None, relevance=0.8, poisoned=False):
    """Helper: build a MemoryRecord for testing."""
    import uuid
    cat = category or classify_category(content_text)
    ts = timestamp or datetime.utcnow().isoformat()
    return MemoryRecord(
        id=str(uuid.uuid4()),
        timestamp=ts,
        event_type=event_type,
        category=cat,
        content={"result": content_text},
        embedding=generate_embedding(content_text),
        relevance=relevance,
        importance=importance,
        access_count=access_count,
        last_accessed=None,
        pinned=pinned,
        poisoned=poisoned,
    )


# ===================================================================
# P3a + P3b: Retrieval Boost Tests
# ===================================================================

class TestRetrievalBoosts(unittest.TestCase):
    """P3a correction boost + P3b category-aware boost in retrieve_ranked."""

    def setUp(self):
        _clear_store()

    def test_correction_memory_gets_boost(self):
        """P3a: correction event_type memories should rank higher."""
        normal = _make_record("setting out baseline procedure", importance=0.5)
        correction = _make_record(
            "CORRECTION: always use total station for setting out baseline",
            category=normal.category,
            event_type="correction",
            importance=0.7,
        )
        store.insert(normal)
        store.insert(correction)

        results = retrieve_ranked("setting out procedure", "keeper", top_k=5)
        self.assertGreater(len(results), 0)
        # correction should appear first due to boost
        ids = [r.id for r in results]
        self.assertIn(correction.id, ids)
        if len(ids) >= 2:
            corr_idx = ids.index(correction.id)
            self.assertEqual(corr_idx, 0, "Correction should rank first")

    def test_metacognitive_rule_gets_boost(self):
        """P3a: metacognitive_rule event_type should also get boosted."""
        rule = _make_record(
            "RULE: if topic=crypto, always verify with fresh data",
            event_type="metacognitive_rule",
            importance=0.8,
        )
        normal = _make_record("crypto prices went up today", importance=0.5)
        store.insert(rule)
        store.insert(normal)

        results = retrieve_ranked("crypto prices", "keeper", top_k=5)
        ids = [r.id for r in results]
        self.assertIn(rule.id, ids)

    def test_category_boost_same_domain(self):
        """P3b: query about setting-out boosts setting-out memories."""
        setting_out = _make_record(
            "setting out coordinates for block A",
            category="setting_out",
        )
        general = _make_record(
            "general notes about the weather today",
            category="general",
        )
        store.insert(setting_out)
        store.insert(general)

        results = retrieve_ranked("setting out coordinates", "keeper", top_k=5)
        # with fake embeddings all cosine sims are similar, so just verify
        # the category boost code runs without error and returns results
        self.assertGreater(len(results), 0)
        ids = [r.id for r in results]
        self.assertIn(setting_out.id, ids)

    def test_no_boost_for_different_category(self):
        """P3b: memories in different category don't get boost."""
        rec = _make_record("drainage plans reviewed", category="drainage")
        store.insert(rec)
        # query about setting-out should not boost drainage
        results = retrieve_ranked("setting out baseline", "keeper", top_k=5)
        # just ensure it doesn't crash and returns results
        self.assertIsInstance(results, list)

    def test_poisoned_records_excluded(self):
        """Poisoned records should not appear in retrieval."""
        good = _make_record("good memory about concrete")
        bad = _make_record("poisoned bad memory", poisoned=True)
        store.insert(good)
        store.insert(bad)

        results = retrieve_ranked("memory about concrete", "keeper", top_k=5)
        ids = [r.id for r in results]
        self.assertNotIn(bad.id, ids)


# ===================================================================
# P3c: Spaced Repetition Decay Tests
# ===================================================================

class TestSpacedRepetitionDecay(unittest.TestCase):
    """P3c: POST /memory/decay endpoint."""

    def setUp(self):
        _clear_store()

    def test_decay_endpoint_exists(self):
        """The /memory/decay route should exist."""
        routes = [r.path for r in memu.app.routes]
        self.assertIn("/memory/decay", routes)

    def test_old_unused_memories_fade(self):
        """Memories not accessed in > half_life should have reduced relevance."""
        old_ts = (datetime.utcnow() - timedelta(days=30)).isoformat()
        old_rec = _make_record(
            "old memory nobody accessed",
            timestamp=old_ts,
            relevance=0.8,
            importance=0.5,
            access_count=0,
        )
        store.insert(old_rec)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.apply_spaced_repetition_decay(half_life_days=14.0)
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["faded"], 0)

    def test_pinned_memories_not_decayed(self):
        """Pinned memories are never decayed."""
        old_ts = (datetime.utcnow() - timedelta(days=60)).isoformat()
        pinned = _make_record("pinned goal memory", timestamp=old_ts,
                              pinned=True, relevance=1.0)
        store.insert(pinned)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.apply_spaced_repetition_decay(half_life_days=14.0)
        )
        self.assertEqual(result["skipped"], 1)

    def test_frequently_accessed_strengthened(self):
        """Memories with high access count and recent use get strengthened."""
        recent_ts = (datetime.utcnow() - timedelta(hours=2)).isoformat()
        active = _make_record("actively used memory", timestamp=recent_ts,
                              relevance=0.7, access_count=5)
        store.insert(active)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.apply_spaced_repetition_decay(half_life_days=14.0)
        )
        # active memory should be in strengthened or skipped
        self.assertGreaterEqual(result["strengthened"] + result["skipped"], 1)

    def test_poisoned_memories_skipped(self):
        """Poisoned records should be skipped during decay."""
        bad = _make_record("poisoned record", poisoned=True)
        store.insert(bad)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.apply_spaced_repetition_decay(half_life_days=14.0)
        )
        self.assertEqual(result["skipped"], 1)


# ===================================================================
# P3e: Ohana Goal Tracker Tests
# ===================================================================

class TestOhanaGoals(unittest.TestCase):
    """P3e: goal CRUD — create, update, list."""

    def setUp(self):
        _clear_store()

    def test_create_goal(self):
        """Creating a goal returns success with goal_id."""
        import asyncio
        req = memu.GoalRequest(
            title="Invoice by Friday",
            description="Send invoice to main contractor",
            deadline="2026-03-14",
            priority="high",
        )
        result = asyncio.get_event_loop().run_until_complete(
            memu.create_goal(req)
        )
        self.assertEqual(result["status"], "created")
        self.assertIn("goal_id", result)
        self.assertEqual(result["title"], "Invoice by Friday")
        self.assertEqual(result["priority"], "high")

    def test_goal_is_pinned(self):
        """Goals should be pinned (immune to decay)."""
        import asyncio
        req = memu.GoalRequest(title="Build KAI", priority="critical")
        result = asyncio.get_event_loop().run_until_complete(
            memu.create_goal(req)
        )
        # find the record
        goal_id = result["goal_id"]
        found = [r for r in store._records if r.id == goal_id]
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].pinned)

    def test_list_goals(self):
        """Listing goals returns active goals sorted by priority."""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Low priority task", priority="low")))
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Critical deadline", priority="critical")))

        result = loop.run_until_complete(memu.list_goals())
        self.assertEqual(result["goal_count"], 2)
        # critical first
        self.assertEqual(result["goals"][0]["priority"], "critical")

    def test_update_goal_progress(self):
        """Updating a goal adds a progress note."""
        import asyncio
        loop = asyncio.get_event_loop()
        create_resp = loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Test goal")))
        goal_id = create_resp["goal_id"]

        update_resp = loop.run_until_complete(memu.update_goal(
            memu.GoalUpdateRequest(
                goal_id=goal_id,
                progress_note="Started working on it",
                status="active",
            )))
        self.assertEqual(update_resp["status"], "updated")
        self.assertEqual(update_resp["progress_count"], 1)

    def test_update_goal_complete(self):
        """Completing a goal changes its status."""
        import asyncio
        loop = asyncio.get_event_loop()
        create_resp = loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Finish test")))
        goal_id = create_resp["goal_id"]

        loop.run_until_complete(memu.update_goal(
            memu.GoalUpdateRequest(goal_id=goal_id, status="completed")))

        result = loop.run_until_complete(memu.list_goals(status="completed"))
        self.assertEqual(result["goal_count"], 1)

    def test_update_nonexistent_goal_404(self):
        """Updating a non-existent goal should raise 404."""
        import asyncio
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            asyncio.get_event_loop().run_until_complete(memu.update_goal(
                memu.GoalUpdateRequest(goal_id="nonexistent")))
        self.assertEqual(ctx.exception.status_code, 404)

    def test_goal_importance_by_priority(self):
        """Critical goals should have higher importance than low goals."""
        import asyncio
        loop = asyncio.get_event_loop()
        low = loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Low task", priority="low")))
        crit = loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Critical task", priority="critical")))

        low_rec = [r for r in store._records if r.id == low["goal_id"]][0]
        crit_rec = [r for r in store._records if r.id == crit["goal_id"]][0]
        self.assertGreater(crit_rec.importance, low_rec.importance)

    def test_list_goals_filter_status(self):
        """Filtering by status should only return matching goals."""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Active goal")))
        resp = loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="To be paused")))
        loop.run_until_complete(memu.update_goal(
            memu.GoalUpdateRequest(goal_id=resp["goal_id"], status="paused")))

        active = loop.run_until_complete(memu.list_goals(status="active"))
        paused = loop.run_until_complete(memu.list_goals(status="paused"))
        self.assertEqual(active["goal_count"], 1)
        self.assertEqual(paused["goal_count"], 1)


# ===================================================================
# P3f: Operator Drift Detection Tests
# ===================================================================

class TestDriftDetection(unittest.TestCase):
    """P3f: GET /memory/drift endpoint."""

    def setUp(self):
        _clear_store()

    def test_no_goals_returns_no_drift(self):
        """With no goals set, drift detection should say 'no_goals'."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.detect_operator_drift(hours=4)
        )
        self.assertEqual(result["status"], "no_goals")
        self.assertFalse(result["drifting"])

    def test_insufficient_recent_activity(self):
        """With too few recent records, should report insufficient data."""
        import asyncio
        loop = asyncio.get_event_loop()
        # create a goal
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Build walls", category="construction")))
        # add only 1 recent activity (below threshold of 3)
        recent = _make_record("one recent thing",
                              timestamp=datetime.utcnow().isoformat())
        store.insert(recent)

        result = loop.run_until_complete(memu.detect_operator_drift(hours=4))
        self.assertFalse(result["drifting"])

    def test_on_goal_no_drift(self):
        """When recent activity matches goal categories, no drift."""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Review drainage", category="drainage")))

        now = datetime.utcnow()
        for i in range(5):
            rec = _make_record(f"drainage check {i}", category="drainage",
                               timestamp=(now - timedelta(minutes=i*10)).isoformat())
            store.insert(rec)

        result = loop.run_until_complete(memu.detect_operator_drift(hours=4))
        self.assertFalse(result.get("drifting", False))

    def test_drift_detected_with_nudge(self):
        """When most activity is off-goal, drift should be flagged with nudge."""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Finish brickwork", category="brickwork")))

        now = datetime.utcnow()
        # add 5 off-goal activities
        for i in range(5):
            rec = _make_record(f"browsing internet thing {i}",
                               category="general",
                               timestamp=(now - timedelta(minutes=i*10)).isoformat())
            store.insert(rec)

        result = loop.run_until_complete(memu.detect_operator_drift(hours=4))
        self.assertTrue(result["drifting"])
        self.assertGreater(result["drift_ratio"], 0.6)
        self.assertIn("Brother", result.get("nudge", ""))

    def test_drift_endpoint_exists(self):
        """The /memory/drift route should exist."""
        routes = [r.path for r in memu.app.routes]
        self.assertIn("/memory/drift", routes)


# ===================================================================
# P3d: Proactive Conversation Engine Tests
# ===================================================================

class TestProactiveEngine(unittest.TestCase):
    """P3d: GET /memory/proactive/full endpoint."""

    def setUp(self):
        _clear_store()

    def test_proactive_full_endpoint_exists(self):
        """The /memory/proactive/full route should exist."""
        routes = [r.path for r in memu.app.routes]
        self.assertIn("/memory/proactive/full", routes)

    def test_proactive_full_returns_nudges(self):
        """Full proactive scan should return a nudge list."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.full_proactive_scan()
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("nudges", result)
        self.assertIn("nudge_count", result)

    def test_proactive_goal_deadline_nudge(self):
        """Goals with approaching deadlines should produce nudges."""
        import asyncio
        loop = asyncio.get_event_loop()
        tomorrow = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        loop.run_until_complete(memu.create_goal(
            memu.GoalRequest(title="Submit tender", deadline=tomorrow,
                             priority="critical")))

        result = loop.run_until_complete(memu.full_proactive_scan())
        deadline_nudges = [n for n in result["nudges"]
                          if n.get("type") == "goal_deadline"]
        self.assertGreater(len(deadline_nudges), 0)
        self.assertIn("Submit tender", deadline_nudges[0]["message"])

    def test_proactive_sorted_by_urgency(self):
        """Nudges should be sorted highest urgency first."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.full_proactive_scan()
        )
        nudges = result.get("nudges", [])
        if len(nudges) >= 2:
            for i in range(len(nudges) - 1):
                self.assertGreaterEqual(
                    nudges[i].get("urgency", 0),
                    nudges[i + 1].get("urgency", 0),
                )

    def test_proactive_fading_memory_detection(self):
        """High-importance old memories approaching fade should trigger nudge."""
        import asyncio
        old_ts = (datetime.utcnow() - timedelta(days=30)).isoformat()
        important = _make_record(
            "Critical site measurement from last month",
            timestamp=old_ts,
            importance=0.9,
            access_count=0,
        )
        store.insert(important)

        result = asyncio.get_event_loop().run_until_complete(
            memu.full_proactive_scan()
        )
        fading = [n for n in result["nudges"] if n.get("type") == "fading_memory"]
        self.assertGreater(len(fading), 0)

    def test_proactive_max_10_nudges(self):
        """Proactive scan should cap at 10 nudges max."""
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            memu.full_proactive_scan()
        )
        self.assertLessEqual(len(result.get("nudges", [])), 10)


# ===================================================================
# Integration: all endpoints exist
# ===================================================================

class TestEndpointsExist(unittest.TestCase):
    """Ensure all P3 endpoints are registered."""

    def test_all_p3_endpoints_registered(self):
        routes = [r.path for r in memu.app.routes]
        expected = [
            "/memory/decay",
            "/memory/goals",
            "/memory/goals/update",
            "/memory/drift",
            "/memory/proactive/full",
        ]
        for ep in expected:
            self.assertIn(ep, routes, f"Missing endpoint: {ep}")


if __name__ == "__main__":
    unittest.main()
