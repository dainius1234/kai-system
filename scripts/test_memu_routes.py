"""S3 — FastAPI route tests for memu-core/app.py (target: 65%+ coverage).

Exercises the in-memory (no DB/Redis/LLM) routes that account for the
largest uncovered surface.  VECTOR_STORE=memory keeps it offline-safe.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Bootstrap: ensure redis stub is present before module load ────────
# (conftest.py does this globally but re-guard for isolated runs)
if "redis" not in sys.modules:
    _rs = MagicMock()
    _rs.from_url.return_value = MagicMock(ping=MagicMock(side_effect=ConnectionError("stub")))
    _rs.asyncio = MagicMock()
    sys.modules["redis"] = _rs
    sys.modules["redis.asyncio"] = _rs.asyncio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Use in-memory store for all tests
os.environ["VECTOR_STORE"] = "memory"
os.environ["MEMU_ALLOW_FAKE_EMBEDDINGS"] = "true"
os.environ["LOG_PATH"] = "/tmp/test-memu-routes.log"
os.environ["FF_GRAPH_INGEST"] = "false"
os.environ["REQUIRE_VERDICT_PASS"] = "false"
os.environ["AUDIT_REQUIRED"] = "false"


def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app_routes", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app_routes"] = mod
    spec.loader.exec_module(mod)
    mod._redis_client = None  # force in-memory session path
    return mod


memu = _load_memu()

from fastapi.testclient import TestClient

client = TestClient(memu.app, raise_server_exceptions=True)


# ─── helpers ────────────────────────────────────────────────────────

def _memorize(text: str = "test event", event_type: str = "test") -> dict:
    return client.post("/memory/memorize", json={
        "timestamp": "2026-07-23T00:00:00Z",
        "event_type": event_type,
        "result_raw": text,
        "user_id": "keeper",
    }).json()


# ═════════════════════════════════════════════════════════════════════
# /health + /metrics + /logs
# ═════════════════════════════════════════════════════════════════════

class TestInfra:
    def test_health_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status(self):
        data = client.get("/health").json()
        assert data["status"] in ("ok", "degraded")
        assert "storage" in data

    def test_metrics_200(self):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_is_dict(self):
        assert isinstance(client.get("/metrics").json(), dict)

    def test_logs_200(self):
        r = client.get("/logs")
        assert r.status_code == 200

    def test_logs_has_entries_key(self):
        data = client.get("/logs").json()
        assert "entries" in data
        assert "count" in data

    def test_logs_level_filter(self):
        r = client.get("/logs?level=INFO")
        assert r.status_code == 200

    def test_logs_since_future(self):
        r = client.get(f"/logs?since={time.time() + 9999}")
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_logs_limit(self):
        r = client.get("/logs?limit=5")
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /recover
# ═════════════════════════════════════════════════════════════════════

class TestRecover:
    def test_post_200(self):
        r = client.post("/recover")
        assert r.status_code == 200

    def test_has_status(self):
        assert client.post("/recover").json().get("status") == "ok"


# ═════════════════════════════════════════════════════════════════════
# /memory/memorize + /memory/retrieve
# ═════════════════════════════════════════════════════════════════════

class TestMemorize:
    def test_memorize_200(self):
        r = client.post("/memory/memorize", json={
            "timestamp": "2026-07-23T00:00:00Z",
            "event_type": "chat",
            "result_raw": "hello world",
            "user_id": "keeper",
        })
        assert r.status_code == 200

    def test_memorize_returns_stored(self):
        data = _memorize("important CIS deduction note")
        assert data.get("status") in ("stored", "appended")

    def test_memorize_auto_classifies_category(self):
        data = _memorize("scaffolding invoice payment received", "financial")
        assert "category" in data

    def test_retrieve_returns_list(self):
        _memorize("vector retrieval test content")
        r = client.get("/memory/retrieve?query=test&user_id=keeper&top_k=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_retrieve_empty_store_ok(self):
        r = client.get("/memory/retrieve?query=nonexistent_xyz&user_id=keeper&top_k=1")
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/note
# ═════════════════════════════════════════════════════════════════════

class TestNote:
    def test_note_200(self):
        r = client.post("/memory/note", json={"text": "remember to review budget"})
        assert r.status_code == 200

    def test_note_empty_400(self):
        r = client.post("/memory/note", json={"text": ""})
        assert r.status_code == 400

    def test_note_returns_id(self):
        data = client.post("/memory/note", json={"text": "test note content"}).json()
        assert "id" in data
        assert data.get("status") == "noted"

    def test_note_pinned(self):
        r = client.post("/memory/note", json={"text": "pinned note", "pin": True})
        assert r.status_code == 200

    def test_note_with_category(self):
        r = client.post("/memory/note", json={"text": "CIS note", "category": "construction"})
        assert r.status_code == 200
        assert r.json().get("category") == "construction"


# ═════════════════════════════════════════════════════════════════════
# /memory/preferences
# ═════════════════════════════════════════════════════════════════════

class TestPreferences:
    def test_store_preference_200(self):
        r = client.post("/memory/preferences", json={"preference": "prefer concise replies"})
        assert r.status_code == 200

    def test_store_preference_empty_400(self):
        r = client.post("/memory/preferences", json={"preference": ""})
        assert r.status_code == 400

    def test_store_preference_returns_status(self):
        data = client.post("/memory/preferences", json={"preference": "use bullets"}).json()
        assert "status" in data

    def test_get_preferences_200(self):
        r = client.get("/memory/preferences")
        assert r.status_code == 200

    def test_get_preferences_has_preferences(self):
        data = client.get("/memory/preferences").json()
        assert "preferences" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# /session/{id} — working memory buffer
# ═════════════════════════════════════════════════════════════════════

class TestSession:
    SID = "test-session-routes-001"

    def test_get_empty_session_200(self):
        r = client.get(f"/session/{self.SID}-empty")
        assert r.status_code == 200

    def test_get_session_has_turns(self):
        data = client.get(f"/session/{self.SID}-empty2").json()
        assert "turns" in data
        assert "messages" in data

    def test_append_user_turn_200(self):
        r = client.post(
            f"/session/{self.SID}/append",
            json={"role": "user", "content": "hello kai"},
        )
        assert r.status_code == 200

    def test_append_assistant_turn_200(self):
        r = client.post(
            f"/session/{self.SID}/append",
            json={"role": "assistant", "content": "hello operator"},
        )
        assert r.status_code == 200

    def test_append_invalid_role_400(self):
        r = client.post(
            f"/session/{self.SID}/append",
            json={"role": "bot", "content": "bad role"},
        )
        assert r.status_code == 400

    def test_session_grows_after_append(self):
        sid = f"{self.SID}-grow"
        client.post(f"/session/{sid}/append", json={"role": "user", "content": "msg1"})
        client.post(f"/session/{sid}/append", json={"role": "assistant", "content": "reply1"})
        data = client.get(f"/session/{sid}").json()
        assert data["turns"] >= 2

    def test_delete_session_200(self):
        sid = f"{self.SID}-del"
        client.post(f"/session/{sid}/append", json={"role": "user", "content": "to delete"})
        r = client.delete(f"/session/{sid}")
        assert r.status_code == 200
        assert r.json().get("cleared") is True

    def test_session_context_200(self):
        r = client.get(f"/session/{self.SID}/context?query=hello&top_k=3")
        assert r.status_code == 200

    def test_session_context_has_keys(self):
        data = client.get(f"/session/{self.SID}/context").json()
        assert "session_messages" in data or "long_term_memories" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/consolidate + /memory/token-budget + /memory/drift
# ═════════════════════════════════════════════════════════════════════

class TestMemoryManagement:
    def test_consolidate_200(self):
        r = client.post("/memory/consolidate")
        assert r.status_code == 200

    def test_consolidate_has_stats(self):
        data = client.post("/memory/consolidate").json()
        assert "pruned" in data or "status" in data

    def test_token_budget_200(self):
        r = client.get("/memory/token-budget")
        assert r.status_code == 200

    def test_token_budget_has_usage(self):
        data = client.get("/memory/token-budget").json()
        assert "total_tokens" in data
        assert "token_budget" in data
        assert "usage_pct" in data

    def test_drift_200(self):
        r = client.get("/memory/drift")
        assert r.status_code == 200

    def test_drift_has_keys(self):
        data = client.get("/memory/drift").json()
        assert "drift_detected" in data or "status" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/goals
# ═════════════════════════════════════════════════════════════════════

class TestGoals:
    def test_post_goal_200(self):
        r = client.post("/memory/goals", json={
            "title": "Earn £100k this year",
            "description": "hit monthly targets",
            "priority": "high",
        })
        assert r.status_code == 200

    def test_post_goal_returns_id(self):
        data = client.post("/memory/goals", json={
            "title": "Grow construction portfolio",
            "priority": "medium",
        }).json()
        assert "id" in data or "goal_id" in data

    def test_get_goals_200(self):
        r = client.get("/memory/goals")
        assert r.status_code == 200

    def test_get_goals_has_goals(self):
        data = client.get("/memory/goals").json()
        assert "goals" in data
        assert "count" in data or "goal_count" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/dnd + /memory/nudge/status
# ═════════════════════════════════════════════════════════════════════

class TestDND:
    def test_post_dnd_200(self):
        r = client.post("/memory/dnd", json={"hours": 2.0})
        assert r.status_code == 200

    def test_post_dnd_has_hours(self):
        data = client.post("/memory/dnd", json={"hours": 1.0}).json()
        assert data.get("hours") == 1.0
        assert data.get("status") == "ok"

    def test_get_dnd_200(self):
        r = client.get("/memory/dnd")
        assert r.status_code == 200

    def test_get_dnd_has_active(self):
        client.post("/memory/dnd", json={"hours": 1.0})
        data = client.get("/memory/dnd").json()
        assert "active" in data
        assert "remaining_seconds" in data

    def test_nudge_status_200(self):
        r = client.get("/memory/nudge/status")
        assert r.status_code == 200

    def test_nudge_dismiss_200(self):
        r = client.post("/memory/nudge/dismiss", json={"nudge_type": "reminder"})
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/topics
# ═════════════════════════════════════════════════════════════════════

class TestTopics:
    def test_track_topic_200(self):
        r = client.post("/memory/topics/track", json={
            "topic": "CIS deduction rates",
            "context": "discussing subcontractor payments",
        })
        assert r.status_code == 200

    def test_track_topic_created(self):
        data = client.post("/memory/topics/track", json={"topic": "new unique topic xyz"}).json()
        assert data.get("action") in ("created", "updated")

    def test_track_topic_updated_on_repeat(self):
        t = "repeated-topic-abc"
        client.post("/memory/topics/track", json={"topic": t})
        data = client.post("/memory/topics/track", json={"topic": t}).json()
        assert data.get("action") == "updated"

    def test_defer_topic_200(self):
        r = client.post("/memory/topics/defer", json={
            "topic": "tax return prep",
            "resurface_after_hours": 24,
        })
        assert r.status_code == 200

    def test_active_topics_200(self):
        r = client.get("/memory/topics/active")
        assert r.status_code == 200

    def test_active_topics_has_topics(self):
        data = client.get("/memory/topics/active").json()
        assert "topics" in data

    def test_deferred_topics_200(self):
        r = client.get("/memory/topics/deferred")
        assert r.status_code == 200

    def test_resurface_topics_missing_404(self):
        r = client.post("/memory/topics/resurface?topic_id=nonexistent-xyz")
        assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════
# /memory/greeting + /memory/actions
# ═════════════════════════════════════════════════════════════════════

class TestGreetingAndActions:
    def test_greeting_200(self):
        r = client.get("/memory/greeting")
        assert r.status_code == 200

    def test_greeting_has_status(self):
        data = client.get("/memory/greeting").json()
        assert "status" in data

    def test_actions_200(self):
        r = client.get("/memory/actions")
        assert r.status_code == 200

    def test_actions_has_count(self):
        data = client.get("/memory/actions").json()
        assert "count" in data
        assert "actions" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/schedule/task
# ═════════════════════════════════════════════════════════════════════

class TestSchedule:
    def test_schedule_task_200(self):
        r = client.post("/memory/schedule/task", json={
            "title": "send invoice",
            "type": "reminder",
            "frequency": "once",
        })
        assert r.status_code == 200

    def test_schedule_task_no_title_400(self):
        r = client.post("/memory/schedule/task", json={"type": "reminder"})
        assert r.status_code == 400

    def test_list_tasks_200(self):
        r = client.get("/memory/schedule/tasks")
        assert r.status_code == 200

    def test_list_tasks_has_tasks(self):
        data = client.get("/memory/schedule/tasks").json()
        assert "tasks" in data

    def test_schedule_due_200(self):
        r = client.get("/memory/schedule/due")
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/reminders
# ═════════════════════════════════════════════════════════════════════

class TestReminders:
    def test_set_reminder_200(self):
        r = client.post("/memory/reminders/set", json={
            "text": "submit CIS return",
            "remind_at": "2026-08-01T09:00:00Z",
        })
        assert r.status_code == 200

    def test_set_reminder_no_text_400(self):
        r = client.post("/memory/reminders/set", json={"remind_at": "2026-08-01T09:00:00Z"})
        assert r.status_code == 400

    def test_get_reminders_200(self):
        r = client.get("/memory/reminders")
        assert r.status_code == 200

    def test_get_reminders_has_list(self):
        data = client.get("/memory/reminders").json()
        assert "reminders" in data

    def test_due_reminders_200(self):
        r = client.get("/memory/reminders/due")
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/briefing
# ═════════════════════════════════════════════════════════════════════

class TestBriefing:
    def test_morning_briefing_route_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/briefing/morning" in routes

    def test_evening_briefing_route_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/briefing/evening" in routes

    def test_briefing_history_200(self):
        r = client.get("/memory/briefing/history")
        assert r.status_code == 200

    def test_briefing_history_has_items(self):
        data = client.get("/memory/briefing/history").json()
        assert "history" in data or "briefings" in data or "status" in data


# ═════════════════════════════════════════════════════════════════════
# P20: Values + Conscience + Loyalty + Gratitude
# ═════════════════════════════════════════════════════════════════════

class TestValuesConscience:
    def test_values_learn_200(self):
        r = client.post("/memory/values/learn", json={
            "experience": "paid subcontractor on time",
            "outcome": "positive",
        })
        assert r.status_code == 200

    def test_values_get_200(self):
        r = client.get("/memory/values")
        assert r.status_code == 200

    def test_values_has_values(self):
        data = client.get("/memory/values").json()
        assert "values" in data or "formed_values" in data or "status" in data

    def test_conscience_check_200(self):
        r = client.post("/memory/conscience/check", json={
            "action": "delay payment to subcontractor",
            "context": "cash flow issue this month",
        })
        assert r.status_code == 200

    def test_conscience_audit_200(self):
        r = client.get("/memory/conscience/audit")
        assert r.status_code == 200

    def test_conscience_summary_200(self):
        r = client.get("/memory/conscience/summary")
        assert r.status_code == 200

    def test_loyalty_record_200(self):
        r = client.post("/memory/loyalty/record", json={
            "person": "John Smith",
            "act": "referred new client",
            "type": "commitment",
        })
        assert r.status_code == 200

    def test_loyalty_get_200(self):
        r = client.get("/memory/loyalty")
        assert r.status_code == 200

    def test_gratitude_record_200(self):
        r = client.post("/memory/gratitude/record", json={
            "reason": "helped with the scaffolding design",
            "recipient": "John Smith",
        })
        assert r.status_code == 200

    def test_gratitude_get_200(self):
        r = client.get("/memory/gratitude")
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/agent/summary
# ═════════════════════════════════════════════════════════════════════

class TestAgentSummary:
    def test_agent_summary_200(self):
        r = client.get("/memory/agent/summary")
        assert r.status_code == 200

    def test_agent_summary_has_keys(self):
        data = client.get("/memory/agent/summary").json()
        assert "status" in data or "summary" in data or "total_memories" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/boundary (P3c proactive boundary)
# ═════════════════════════════════════════════════════════════════════

class TestBoundary:
    def test_boundary_200(self):
        r = client.get("/memory/boundary")
        assert r.status_code == 200

    def test_boundary_has_status(self):
        data = client.get("/memory/boundary").json()
        assert "status" in data
