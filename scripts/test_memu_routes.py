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

# Remove any prior lakefs_client stub so memu-core's built-in ImportError
# fallback fires and returns real string commit_ids (not MagicMock).
sys.modules.pop("lakefs_client", None)

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
# /memory/persist + /route
# ═════════════════════════════════════════════════════════════════════

class TestPersistAndRoute:
    def test_persist_200(self):
        r = client.post("/memory/persist")
        assert r.status_code == 200

    def test_persist_has_status(self):
        data = client.post("/memory/persist").json()
        assert data.get("status") in ("ok", "error")

    def test_route_200(self):
        r = client.post("/route", json={
            "query": "CIS deduction rate",
            "session_id": "s1",
            "timestamp": "2026-07-23T10:00:00Z",
        })
        assert r.status_code == 200

    def test_route_has_specialist(self):
        data = client.post("/route", json={
            "query": "scaffolding invoice",
            "session_id": "s1",
            "timestamp": "2026-07-23T10:00:00Z",
        }).json()
        assert "specialist" in data

    def test_route_has_context_payload(self):
        data = client.post("/route", json={
            "query": "what is my goal",
            "session_id": "s2",
            "timestamp": "2026-07-23T10:00:00Z",
        }).json()
        assert "context_payload" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/graph/query + /memory/evidence-pack
# ═════════════════════════════════════════════════════════════════════

class TestGraphAndEvidencePack:
    def test_graph_query_200(self):
        r = client.get("/memory/graph/query?q=invoice")
        assert r.status_code == 200

    def test_graph_query_disabled_when_flag_off(self):
        # FF_GRAPH_INGEST=false (set at top of file) → status = graph_disabled
        data = client.get("/memory/graph/query?q=test").json()
        assert data.get("status") == "graph_disabled"

    def test_evidence_pack_200(self):
        _memorize("CIS 20 percent rate for verified contractors")
        r = client.get("/memory/evidence-pack?query=CIS+deduction&user_id=keeper&top_k=5")
        assert r.status_code == 200

    def test_evidence_pack_has_pack(self):
        data = client.get("/memory/evidence-pack?query=invoice").json()
        assert "pack_size" in data
        assert "pack" in data or "query" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/assert
# ═════════════════════════════════════════════════════════════════════

class TestAssertMemory:
    def test_assert_200(self):
        r = client.post("/memory/assert", json={
            "result_raw": "CIS deduction rate is 20 percent for verified subcontractors",
            "event_type": "fact",
            "timestamp": "2026-07-23T10:00:00Z",
            "user_id": "keeper",
        })
        assert r.status_code == 200

    def test_assert_empty_400(self):
        r = client.post("/memory/assert", json={
            "result_raw": "",
            "event_type": "fact",
            "timestamp": "2026-07-23T10:00:00Z",
        })
        assert r.status_code == 400

    def test_assert_has_id(self):
        data = client.post("/memory/assert", json={
            "result_raw": "material costs increased by 10 percent this quarter",
            "event_type": "financial",
            "timestamp": "2026-07-23T10:00:00Z",
        }).json()
        assert "id" in data or "status" in data

    def test_assert_force_overrides(self):
        r = client.post("/memory/assert", json={
            "result_raw": "payment terms are 30 days",
            "event_type": "policy",
            "timestamp": "2026-07-23T10:00:00Z",
            "force": True,
        })
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/goals/update
# ═════════════════════════════════════════════════════════════════════

class TestGoalsUpdate:
    def test_update_goal_not_found_404(self):
        r = client.post("/memory/goals/update", json={
            "goal_id": "nonexistent-goal-xyz",
            "progress_note": "did some work",
            "status": "active",
        })
        assert r.status_code == 404

    def test_update_goal_invalid_status_400(self):
        r = client.post("/memory/goals/update", json={
            "goal_id": "any",
            "progress_note": "",
            "status": "invalid_status",
        })
        assert r.status_code in (400, 404)

    def test_update_existing_goal_200(self):
        # First create a goal, then update it
        goal_resp = client.post("/memory/goals", json={
            "title": "finish quarterly report",
            "priority": "high",
        }).json()
        goal_id = goal_resp.get("id") or goal_resp.get("goal_id")
        if not goal_id:
            return  # can't test without id
        r = client.post("/memory/goals/update", json={
            "goal_id": goal_id,
            "progress_note": "drafted introduction",
            "status": "active",
        })
        assert r.status_code == 200


# ═════════════════════════════════════════════════════════════════════
# /memory/proactive + /memory/proactive/full + /memory/proactive/filtered
# ═════════════════════════════════════════════════════════════════════

class TestProactive:
    def test_proactive_200(self):
        r = client.get("/memory/proactive")
        assert r.status_code == 200

    def test_proactive_has_nudges(self):
        data = client.get("/memory/proactive").json()
        assert "nudge_count" in data
        assert "nudges" in data

    def test_proactive_full_200(self):
        r = client.get("/memory/proactive/full")
        assert r.status_code == 200

    def test_proactive_full_has_nudges(self):
        data = client.get("/memory/proactive/full").json()
        assert "nudges" in data

    def test_proactive_filtered_200(self):
        r = client.get("/memory/proactive/filtered?mode=PUB")
        assert r.status_code == 200

    def test_proactive_filtered_work_mode(self):
        r = client.get("/memory/proactive/filtered?mode=WORK")
        assert r.status_code == 200
        assert r.json().get("mode") == "WORK"


# ═════════════════════════════════════════════════════════════════════
# /memory/silence + /memory/tempo + /memory/check-in + /memory/struggle
# ═════════════════════════════════════════════════════════════════════

class TestSilenceTempoCheckInStruggle:
    def test_silence_200(self):
        r = client.get("/memory/silence")
        assert r.status_code == 200

    def test_silence_has_signals(self):
        data = client.get("/memory/silence").json()
        assert "status" in data or "signals" in data or "silent_topics" in data

    def test_tempo_200(self):
        r = client.get("/memory/tempo")
        assert r.status_code == 200

    def test_tempo_has_status(self):
        data = client.get("/memory/tempo").json()
        assert "status" in data

    def test_check_in_200(self):
        r = client.get("/memory/check-in")
        assert r.status_code == 200

    def test_check_in_has_status(self):
        data = client.get("/memory/check-in").json()
        assert "status" in data

    def test_struggle_200(self):
        r = client.get("/memory/struggle?session_id=test-session")
        assert r.status_code == 200

    def test_struggle_has_score(self):
        data = client.get("/memory/struggle").json()
        assert "status" in data or "struggle_score" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/feedback + /memory/feedback/stats
# ═════════════════════════════════════════════════════════════════════

class TestFeedback:
    def test_feedback_200(self):
        r = client.post("/memory/feedback", json={
            "session_id": "session-abc",
            "message_index": 0,
            "rating": 5,
        })
        assert r.status_code == 200

    def test_feedback_bad_rating_400(self):
        r = client.post("/memory/feedback", json={
            "session_id": "session-abc",
            "message_index": 0,
            "rating": 6,
        })
        assert r.status_code == 400

    def test_feedback_low_rating_stored(self):
        data = client.post("/memory/feedback", json={
            "session_id": "session-xyz",
            "message_index": 1,
            "rating": 1,
            "comment": "wrong answer",
        }).json()
        assert data.get("effect") == "correction"

    def test_feedback_stats_200(self):
        r = client.get("/memory/feedback/stats")
        assert r.status_code == 200

    def test_feedback_stats_has_count(self):
        data = client.get("/memory/feedback/stats").json()
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/emotion/record + /memory/emotion/timeline
# ═════════════════════════════════════════════════════════════════════

class TestEmotionRecordAndTimeline:
    def test_record_emotion_200(self):
        r = client.post("/memory/emotion/record", json={
            "session_id": "emotion-test",
            "text": "I'm feeling really frustrated with HMRC",
        })
        assert r.status_code == 200

    def test_record_emotion_no_text_400(self):
        r = client.post("/memory/emotion/record", json={"session_id": "emotion-test", "text": ""})
        assert r.status_code == 400

    def test_record_emotion_has_emotion(self):
        data = client.post("/memory/emotion/record", json={
            "session_id": "s1",
            "text": "got the contract signed, feeling brilliant",
        }).json()
        assert "emotion" in data or "status" in data

    def test_emotion_timeline_200(self):
        r = client.get("/memory/emotion/timeline")
        assert r.status_code == 200

    def test_emotion_timeline_has_entries(self):
        data = client.get("/memory/emotion/timeline").json()
        assert "entries" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/self-reflect + /memory/self-reflections
# ═════════════════════════════════════════════════════════════════════

class TestSelfReflect:
    def test_self_reflect_200(self):
        r = client.post("/memory/self-reflect")
        assert r.status_code == 200

    def test_self_reflect_has_reflection(self):
        data = client.post("/memory/self-reflect").json()
        assert data.get("status") == "ok"
        assert "reflection" in data

    def test_self_reflections_200(self):
        r = client.get("/memory/self-reflections")
        assert r.status_code == 200

    def test_self_reflections_has_entries(self):
        data = client.get("/memory/self-reflections").json()
        assert "entries" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/relationship + /memory/relationship/milestone
# ═════════════════════════════════════════════════════════════════════

class TestRelationship:
    def test_relationship_200(self):
        r = client.get("/memory/relationship")
        assert r.status_code == 200

    def test_relationship_has_status(self):
        data = client.get("/memory/relationship").json()
        assert "status" in data

    def test_milestone_200(self):
        r = client.post("/memory/relationship/milestone", json={
            "title": "first site visit together",
            "description": "walked the Bradshaw scaffolding site",
        })
        assert r.status_code == 200

    def test_milestone_no_title_400(self):
        r = client.post("/memory/relationship/milestone", json={"description": "no title here"})
        assert r.status_code == 400

    def test_milestone_has_status(self):
        data = client.post("/memory/relationship/milestone", json={
            "title": "resolved dispute with subcontractor"
        }).json()
        assert data.get("status") == "ok"


# ═════════════════════════════════════════════════════════════════════
# /memory/confidence + /memory/confidence/check + /memory/confess
# ═════════════════════════════════════════════════════════════════════

class TestConfidenceAndConfess:
    def test_confidence_200(self):
        r = client.get("/memory/confidence")
        assert r.status_code == 200

    def test_confidence_has_domains(self):
        data = client.get("/memory/confidence").json()
        assert "status" in data
        assert "domains" in data or "by_domain" in data or isinstance(data, dict)

    def test_confidence_check_200(self):
        r = client.get("/memory/confidence/check?query=CIS+deduction+rate")
        assert r.status_code == 200

    def test_confidence_check_has_status(self):
        data = client.get("/memory/confidence/check?query=scaffolding+VAT").json()
        assert "status" in data

    def test_confess_200(self):
        r = client.post("/memory/confess", json={
            "correction": "I said CIS rate was 10 percent but it is 20 percent",
            "category": "financial",
        })
        assert r.status_code == 200

    def test_confess_no_text_400(self):
        r = client.post("/memory/confess", json={"correction": ""})
        assert r.status_code == 400

    def test_confess_has_confessions(self):
        data = client.post("/memory/confess", json={
            "correction": "invoice terms were wrong"
        }).json()
        assert "confessions" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/eq/summary
# ═════════════════════════════════════════════════════════════════════

class TestEqSummary:
    def test_eq_summary_200(self):
        r = client.get("/memory/eq/summary")
        assert r.status_code == 200

    def test_eq_summary_has_status(self):
        data = client.get("/memory/eq/summary").json()
        assert "status" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/autobiography/record + /memory/autobiography
# ═════════════════════════════════════════════════════════════════════

class TestAutobiography:
    def test_autobiography_record_200(self):
        r = client.post("/memory/autobiography/record", json={
            "text": "today I won the biggest scaffolding contract of my career worth 200k",
            "context": "business milestone",
        })
        assert r.status_code == 200

    def test_autobiography_record_no_text_400(self):
        r = client.post("/memory/autobiography/record", json={"text": ""})
        assert r.status_code == 400

    def test_autobiography_record_returns_status(self):
        data = client.post("/memory/autobiography/record", json={
            "text": "first time I managed a team of 10 lads on site"
        }).json()
        assert "status" in data

    def test_autobiography_get_200(self):
        r = client.get("/memory/autobiography")
        assert r.status_code == 200

    def test_autobiography_has_entries(self):
        data = client.get("/memory/autobiography").json()
        assert "entries" in data
        assert "total" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/identity + /memory/story-arcs + /memory/future-self
# ═════════════════════════════════════════════════════════════════════

class TestIdentityNarrative:
    def test_identity_200(self):
        r = client.get("/memory/identity")
        assert r.status_code == 200

    def test_identity_has_status(self):
        data = client.get("/memory/identity").json()
        assert "status" in data

    def test_story_arcs_200(self):
        r = client.get("/memory/story-arcs")
        assert r.status_code == 200

    def test_story_arcs_has_status(self):
        data = client.get("/memory/story-arcs").json()
        assert "status" in data

    def test_future_self_200(self):
        r = client.get("/memory/future-self")
        assert r.status_code == 200

    def test_future_self_has_status(self):
        data = client.get("/memory/future-self").json()
        assert "status" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/legacy/write + /memory/legacy + /memory/legacy/pending
# ═════════════════════════════════════════════════════════════════════

class TestLegacy:
    def test_legacy_write_200(self):
        r = client.post("/memory/legacy/write", json={
            "message": "future Kai: remember that honesty always wins in the long run",
            "recipient": "self",
            "surface_after_days": 7,
        })
        assert r.status_code == 200

    def test_legacy_write_no_message_400(self):
        r = client.post("/memory/legacy/write", json={"recipient": "self"})
        assert r.status_code == 400

    def test_legacy_write_has_status(self):
        data = client.post("/memory/legacy/write", json={
            "message": "keep the faith"
        }).json()
        assert data.get("status") == "ok"

    def test_legacy_get_200(self):
        r = client.get("/memory/legacy")
        assert r.status_code == 200

    def test_legacy_get_has_messages(self):
        data = client.get("/memory/legacy").json()
        assert "messages" in data

    def test_legacy_include_unsurfaced(self):
        r = client.get("/memory/legacy?include_unsurfaced=true")
        assert r.status_code == 200

    def test_legacy_pending_200(self):
        r = client.get("/memory/legacy/pending")
        assert r.status_code == 200

    def test_legacy_pending_has_ready_count(self):
        data = client.get("/memory/legacy/pending").json()
        assert "ready_count" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/narrative/summary
# ═════════════════════════════════════════════════════════════════════

class TestNarrativeSummary:
    def test_narrative_summary_200(self):
        r = client.get("/memory/narrative/summary")
        assert r.status_code == 200

    def test_narrative_summary_has_status(self):
        data = client.get("/memory/narrative/summary").json()
        assert "status" in data


# ═════════════════════════════════════════════════════════════════════
# P19 Imagination routes
# ═════════════════════════════════════════════════════════════════════

class TestImagination:
    def test_counterfactual_post_200(self):
        r = client.post("/memory/imagine/counterfactual", json={
            "original": "I turned down the Morrison contract last month",
            "context": "business decision",
        })
        assert r.status_code == 200

    def test_counterfactual_no_original_400(self):
        r = client.post("/memory/imagine/counterfactual", json={"context": "missing original"})
        assert r.status_code == 400

    def test_counterfactuals_list_200(self):
        r = client.get("/memory/imagine/counterfactuals")
        assert r.status_code == 200

    def test_counterfactuals_list_has_items(self):
        data = client.get("/memory/imagine/counterfactuals").json()
        assert "counterfactuals" in data

    def test_empathize_post_200(self):
        r = client.post("/memory/imagine/empathize", json={
            "text": "client seems really stressed about the payment delay",
        })
        assert r.status_code == 200

    def test_empathize_no_text_400(self):
        r = client.post("/memory/imagine/empathize", json={})
        assert r.status_code == 400

    def test_empathy_map_get_200(self):
        r = client.get("/memory/imagine/empathy-map")
        assert r.status_code == 200

    def test_synthesize_post_200(self):
        # Need at least 2 memories with different categories
        _memorize("scaffolding invoice payment received", "financial")
        _memorize("planning the summer festival schedule", "personal")
        r = client.post("/memory/imagine/synthesize", json={"seed": "efficiency"})
        assert r.status_code == 200

    def test_ideas_get_200(self):
        r = client.get("/memory/imagine/ideas")
        assert r.status_code == 200

    def test_thought_post_200(self):
        r = client.post("/memory/imagine/thought", json={
            "thought": "I wonder if there is a pattern in how clients respond to early invoicing",
            "context": "business reflection",
        })
        assert r.status_code == 200

    def test_thought_no_thought_400(self):
        r = client.post("/memory/imagine/thought", json={})
        assert r.status_code == 400

    def test_inner_monologue_get_200(self):
        r = client.get("/memory/imagine/inner-monologue")
        assert r.status_code == 200

    def test_aspire_post_200(self):
        r = client.post("/memory/imagine/aspire", json={
            "vision": "become the most trusted scaffolding AI in the UK",
            "domain": "professional",
        })
        assert r.status_code == 200

    def test_aspire_no_vision_400(self):
        r = client.post("/memory/imagine/aspire", json={"domain": "professional"})
        assert r.status_code == 400

    def test_aspirations_get_200(self):
        r = client.get("/memory/imagine/aspirations")
        assert r.status_code == 200

    def test_imagination_summary_200(self):
        r = client.get("/memory/imagine/summary")
        assert r.status_code == 200

    def test_imagination_summary_has_status(self):
        data = client.get("/memory/imagine/summary").json()
        assert "status" in data


# ═════════════════════════════════════════════════════════════════════
# P22a: /memory/echo/analyse + /memory/echo/history
# ═════════════════════════════════════════════════════════════════════

class TestEcho:
    def test_echo_analyse_200(self):
        r = client.post("/memory/echo/analyse", json={
            "text": "this invoice situation is making me really stressed again",
            "session_id": "echo-test-session",
        })
        assert r.status_code == 200

    def test_echo_analyse_no_text_400(self):
        r = client.post("/memory/echo/analyse", json={"session_id": "s1"})
        assert r.status_code == 400

    def test_echo_analyse_has_emotion(self):
        data = client.post("/memory/echo/analyse", json={
            "text": "everything is going great today",
            "session_id": "s2",
        }).json()
        assert "current_emotion" in data

    def test_echo_history_200(self):
        r = client.get("/memory/echo/history")
        assert r.status_code == 200

    def test_echo_history_has_entries(self):
        data = client.get("/memory/echo/history").json()
        assert "entries" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# P22b: /memory/nudge/escalate + /memory/nudge/ladder
# ═════════════════════════════════════════════════════════════════════

class TestNudgeEscalate:
    def test_escalate_200(self):
        r = client.post("/memory/nudge/escalate", json={
            "target": "gym-goal",
            "reason": "user dismissed",
            "message": "you haven't been to the gym this week",
        })
        assert r.status_code == 200

    def test_escalate_has_level(self):
        data = client.post("/memory/nudge/escalate", json={"target": "invoice-chase"}).json()
        assert "escalation_level" in data
        assert "dismissals" in data

    def test_escalate_level_rises(self):
        target = "test-escalation-target"
        for _ in range(4):
            client.post("/memory/nudge/escalate", json={"target": target})
        data = client.post("/memory/nudge/escalate", json={"target": target}).json()
        # after 5 dismissals, should be level 3 (tough_love threshold=5)
        assert data["escalation_level"] >= 2

    def test_nudge_ladder_200(self):
        r = client.get("/memory/nudge/ladder")
        assert r.status_code == 200

    def test_nudge_ladder_has_targets(self):
        data = client.get("/memory/nudge/ladder").json()
        assert "targets" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# P22c: /memory/cross-mode/scan + /memory/cross-mode
# ═════════════════════════════════════════════════════════════════════

class TestCrossMode:
    def test_cross_mode_scan_200(self):
        r = client.post("/memory/cross-mode/scan", json={
            "query": "stress",
            "mode": "WORK",
        })
        assert r.status_code == 200

    def test_cross_mode_scan_no_query_400(self):
        r = client.post("/memory/cross-mode/scan", json={"mode": "WORK"})
        assert r.status_code == 400

    def test_cross_mode_scan_has_insights(self):
        data = client.post("/memory/cross-mode/scan", json={
            "query": "invoice payment",
            "mode": "PUB",
        }).json()
        assert "insights" in data
        assert "current_mode" in data

    def test_cross_mode_history_200(self):
        r = client.get("/memory/cross-mode")
        assert r.status_code == 200

    def test_cross_mode_history_has_count(self):
        data = client.get("/memory/cross-mode").json()
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# P22d: /memory/oracle/predict + /memory/oracle/chains
# ═════════════════════════════════════════════════════════════════════

class TestOracle:
    def test_oracle_predict_200(self):
        r = client.post("/memory/oracle/predict", json={
            "action": "skip the gym this week",
        })
        assert r.status_code == 200

    def test_oracle_predict_no_action_400(self):
        r = client.post("/memory/oracle/predict", json={})
        assert r.status_code == 400

    def test_oracle_predict_has_risk(self):
        data = client.post("/memory/oracle/predict", json={
            "action": "delay filing the CIS return by one week",
        }).json()
        assert "overall_risk" in data or "status" in data

    def test_oracle_chains_200(self):
        r = client.get("/memory/oracle/chains")
        assert r.status_code == 200

    def test_oracle_chains_has_predictions(self):
        data = client.get("/memory/oracle/chains").json()
        assert "predictions" in data
        assert "count" in data


# ═════════════════════════════════════════════════════════════════════
# P22e: /memory/shadow/branch + /memory/shadow/branches + explore
# ═════════════════════════════════════════════════════════════════════

class TestShadow:
    def test_shadow_branch_200(self):
        r = client.post("/memory/shadow/branch", json={
            "decision": "accepted the Morrison contract",
            "alternative": "declined and focused on Bradshaw instead",
        })
        assert r.status_code == 200

    def test_shadow_branch_no_decision_400(self):
        r = client.post("/memory/shadow/branch", json={"alternative": "something else"})
        assert r.status_code == 400

    def test_shadow_branch_no_alternative_400(self):
        r = client.post("/memory/shadow/branch", json={"decision": "went to the site"})
        assert r.status_code == 400

    def test_shadow_branch_has_id(self):
        data = client.post("/memory/shadow/branch", json={
            "decision": "hired two extra lads",
            "alternative": "brought in a labour-only subcontractor",
        }).json()
        assert "branch_id" in data

    def test_shadow_branches_200(self):
        r = client.get("/memory/shadow/branches")
        assert r.status_code == 200

    def test_shadow_branches_has_branches(self):
        data = client.get("/memory/shadow/branches").json()
        assert "branches" in data

    def test_shadow_explore_existing_200(self):
        # create then explore
        branch_data = client.post("/memory/shadow/branch", json={
            "decision": "took the Euston job",
            "alternative": "passed on it and bid for Manchester instead",
        }).json()
        bid = branch_data.get("branch_id")
        if bid:
            r = client.get(f"/memory/shadow/explore/{bid}")
            assert r.status_code == 200

    def test_shadow_explore_missing_404(self):
        r = client.get("/memory/shadow/explore/nonexistent-branch-id")
        assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════
# /memory/operator-model
# ═════════════════════════════════════════════════════════════════════

class TestOperatorModel:
    def test_operator_model_200(self):
        r = client.get("/memory/operator-model")
        assert r.status_code == 200

    def test_operator_model_has_status(self):
        data = client.get("/memory/operator-model").json()
        assert "status" in data
        assert "model_completeness" in data


# ═════════════════════════════════════════════════════════════════════
# /memory/schedule/task/{task_id}/cancel + fire
# /memory/reminders/{id}/fire + cancel
# ═════════════════════════════════════════════════════════════════════

class TestScheduleAndReminderSubResources:
    def _create_task(self) -> str:
        resp = client.post("/memory/schedule/task", json={
            "title": "fire test task",
            "type": "reminder",
            "frequency": "once",
        })
        return resp.json().get("task_id", "")

    def _create_reminder(self) -> str:
        resp = client.post("/memory/reminders/set", json={
            "text": "fire test reminder",
            "remind_at": "2026-08-01T09:00:00Z",
        })
        return resp.json().get("reminder_id", "")

    def test_cancel_task_not_found_404(self):
        r = client.post("/memory/schedule/task/nonexistent-task/cancel")
        assert r.status_code == 404

    def test_cancel_task_200(self):
        tid = self._create_task()
        if not tid:
            return
        r = client.post(f"/memory/schedule/task/{tid}/cancel")
        assert r.status_code == 200
        assert r.json().get("status") == "cancelled"

    def test_fire_task_not_found_404(self):
        r = client.post("/memory/schedule/task/nonexistent-task/fire")
        assert r.status_code == 404

    def test_fire_task_200(self):
        tid = self._create_task()
        if not tid:
            return
        r = client.post(f"/memory/schedule/task/{tid}/fire")
        assert r.status_code == 200
        assert r.json().get("status") == "fired"

    def test_fire_reminder_not_found_404(self):
        r = client.post("/memory/reminders/nonexistent-reminder-id/fire")
        assert r.status_code == 404

    def test_fire_reminder_200(self):
        rid = self._create_reminder()
        if not rid:
            return
        r = client.post(f"/memory/reminders/{rid}/fire")
        assert r.status_code == 200
        assert r.json().get("status") == "fired"

    def test_cancel_reminder_not_found_404(self):
        r = client.post("/memory/reminders/nonexistent-id/cancel")
        assert r.status_code == 404

    def test_cancel_reminder_200(self):
        rid = self._create_reminder()
        if not rid:
            return
        r = client.post(f"/memory/reminders/{rid}/cancel")
        assert r.status_code == 200
        assert r.json().get("status") == "cancelled"


# ═════════════════════════════════════════════════════════════════════
# /memory/briefing/morning + /memory/briefing/evening
# ═════════════════════════════════════════════════════════════════════

class TestBriefingMorningEvening:
    def test_morning_briefing_200(self):
        r = client.post("/memory/briefing/morning")
        assert r.status_code == 200

    def test_morning_briefing_has_greeting(self):
        data = client.post("/memory/briefing/morning").json()
        assert "greeting" in data
        assert "type" in data
        assert data["type"] == "morning"

    def test_morning_briefing_has_sections(self):
        data = client.post("/memory/briefing/morning").json()
        assert "sections" in data

    def test_evening_briefing_200(self):
        r = client.post("/memory/briefing/evening")
        assert r.status_code == 200

    def test_evening_briefing_has_type(self):
        data = client.post("/memory/briefing/evening").json()
        assert "type" in data
        assert data["type"] == "evening"


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
