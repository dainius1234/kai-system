"""P16 Operational Intelligence tests.

Tests for:
 - P16a: Struggle detection engine
 - P16b: Log aggregation
 - P16c: Dashboard goals API proxies
 - P16d: Dashboard memory browser API proxies
 - P16e: Feedback rating loop
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.module_stubs import (  # noqa: E402
    AGENTIC_HEAVY_DEPS, absent_stubs, stubbed,
)

# ── Bootstrap stubs ──────────────────────────────────────────────────
# Stub heavy deps that aren't installed in the test env

# Declared once in scripts/module_stubs.py rather than copied into five
# suites, and scoped to the imports below rather than left installed.
_STUBS = absent_stubs(AGENTIC_HEAVY_DEPS)

# Ensure common package is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agentic"))

# Set env vars before importing
os.environ.setdefault("LEDGER_PATH", "/tmp/test-p16-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-p16-tokens.json")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-p16-nonces.json")


# ── Load memu-core ───────────────────────────────────────────────────
def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app"] = mod
    with stubbed(_STUBS):
        spec.loader.exec_module(mod)
    # Force in-memory sessions (Redis is mocked, not real)
    mod._redis_client = None
    return mod


memu = _load_memu()

# ── Load agentic ─────────────────────────────────────────────────────


def _load_agentic():
    spec = importlib.util.spec_from_file_location(
        "agentic_app", os.path.join(ROOT, "agentic", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agentic_app"] = mod
    with stubbed(_STUBS):
        spec.loader.exec_module(mod)
    return mod


ag = _load_agentic()

# ── Load dashboard ───────────────────────────────────────────────────


def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "dashboard_app", os.path.join(ROOT, "dashboard", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app"] = mod
    with stubbed(_STUBS):
        spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()


# ═════════════════════════════════════════════════════════════════════
# P16a: Struggle Detection
# ═════════════════════════════════════════════════════════════════════

class TestStruggleDetection:
    """Test the struggle detection engine in memu-core."""

    def test_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/struggle" in routes

    def test_insufficient_data_returns_zero(self):
        """With no session messages, score should be 0."""
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.get("/memory/struggle?session_id=test_empty_session")
        assert resp.status_code == 200
        data = resp.json()
        assert data["struggle_score"] == 0.0
        assert data["reason"] == "insufficient_data"

    def test_frustration_keywords_boost_score(self):
        """Session with frustration keywords should raise score."""
        sid = f"test_frustrated_{time.time()}"
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)

        # Build frustrated session — ensure enough user messages
        frustrated_msgs = [
            "help me with this",
            "this is broken again",
            "still not working properly",
            "why doesn't this work",
            "stuck again and confused",
        ]
        for msg in frustrated_msgs:
            client.post(f"/session/{sid}/append", json={"role": "user", "content": msg})
            client.post(f"/session/{sid}/append", json={"role": "assistant", "content": "Let me help."})

        resp = client.get(f"/memory/struggle?session_id={sid}")
        data = resp.json()
        assert data["struggle_score"] > 0.0
        assert len(data["signals"]) > 0

        client.delete(f"/session/{sid}")

    def test_short_messages_detected(self):
        """Multiple short messages signal frustration."""
        sid = f"test_short_{time.time()}"
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)

        # All short user messages
        for msg in ["?", "??", "help", "no", "???", "ugh", "wtf"]:
            client.post(f"/session/{sid}/append", json={"role": "user", "content": msg})

        resp = client.get(f"/memory/struggle?session_id={sid}")
        data = resp.json()
        assert data["struggle_score"] > 0.0
        assert any("short" in s for s in data["signals"])

        client.delete(f"/session/{sid}")

    def test_repeated_questions_detected(self):
        """Asking the same thing twice signals frustration."""
        sid = f"test_repeat_{time.time()}"
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)

        for msg in ["how do I fix the database connection error",
                     "how do I fix the database connection error",
                     "still the same error with database"]:
            client.post(f"/session/{sid}/append", json={"role": "user", "content": msg})
            client.post(f"/session/{sid}/append", json={"role": "assistant", "content": "Here's a suggestion."})

        resp = client.get(f"/memory/struggle?session_id={sid}")
        data = resp.json()
        assert data["struggle_score"] > 0
        assert any("repeated" in s for s in data.get("signals", []))

        client.delete(f"/session/{sid}")

    def test_calm_session_low_score(self):
        """Normal conversation should have low struggle score."""
        sid = f"test_calm_{time.time()}"
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)

        for msg in ["What's the weather like today in London?",
                     "Tell me about quantum physics and entanglement",
                     "That's really interesting, can you tell me more about it"]:
            client.post(f"/session/{sid}/append", json={"role": "user", "content": msg})
            client.post(f"/session/{sid}/append", json={"role": "assistant", "content": "Here's the info."})

        resp = client.get(f"/memory/struggle?session_id={sid}")
        data = resp.json()
        assert data["struggle_score"] < 0.3

        client.delete(f"/session/{sid}")

    def test_question_density_signal(self):
        """Lots of question marks signal confusion."""
        sid = f"test_qmarks_{time.time()}"
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)

        for msg in ["what is happening???", "why is this broken???", "how do I fix this??"]:
            client.post(f"/session/{sid}/append", json={"role": "user", "content": msg})
            client.post(f"/session/{sid}/append", json={"role": "assistant", "content": "Let me check."})

        resp = client.get(f"/memory/struggle?session_id={sid}")
        data = resp.json()
        assert any("question" in s for s in data.get("signals", []))

        client.delete(f"/session/{sid}")


# ═════════════════════════════════════════════════════════════════════
# P16e: Feedback Rating Loop
# ═════════════════════════════════════════════════════════════════════

class TestFeedbackRating:
    """Test the feedback/rating system."""

    def test_feedback_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/feedback" in routes
        assert "/memory/feedback/stats" in routes

    def test_submit_positive_feedback(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.post("/memory/feedback", json={
            "session_id": "test_fb", "message_index": 0, "rating": 5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["rating"] == 5
        assert data["effect"] == "boost"

    def test_submit_negative_feedback(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.post("/memory/feedback", json={
            "session_id": "test_fb", "message_index": 0, "rating": 1,
            "comment": "Not helpful at all"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effect"] == "correction"

    def test_a_failed_durable_write_is_not_reported_as_a_boost(self):
        """H-6: the handler used to be `except Exception: pass`, so the
        caller was told `effect: boost` whether or not the memory that
        the boost consists of was ever written.

        This is the half that matters — three tests above assert the
        happy path and would all still pass if the except-branch were
        unreachable. Without this one, the branch is a claim nothing
        checks."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        # A session message must exist, or the write is never attempted
        # and "no write" is the correct, non-degraded answer.
        memu._append_session_message("test_fb_fail", "assistant",
                                     "something to rate")
        client = TestClient(memu.app)
        with patch.object(memu.store, "insert",
                          side_effect=RuntimeError("store is down")):
            resp = client.post("/memory/feedback", json={
                "session_id": "test_fb_fail", "message_index": 0, "rating": 5,
            })
        data = resp.json()
        assert data["effect"] == "not_persisted", data
        assert data["degraded"] is True, data
        assert data["status"] == "degraded", data

        # ...and the failure is aggregatable, not just this-request news.
        from common.degraded import degradation_report
        entries = [e for e in degradation_report()
                   if e["operation"] == "submit_feedback"]
        assert entries, degradation_report()

    def test_submit_neutral_feedback(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.post("/memory/feedback", json={
            "session_id": "test_fb", "message_index": 0, "rating": 3
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["effect"] == "noted"

    def test_invalid_rating_rejected(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.post("/memory/feedback", json={
            "session_id": "test_fb", "message_index": 0, "rating": 0
        })
        assert resp.status_code == 400

    def test_feedback_stats(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.get("/memory/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "avg_rating" in data

    def test_feedback_stats_distribution(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        # Submit a few ratings to have distribution
        for r in [5, 4, 3, 5, 2]:
            client.post("/memory/feedback", json={
                "session_id": "test_dist", "message_index": 0, "rating": r
            })
        resp = client.get("/memory/feedback/stats")
        data = resp.json()
        assert data["count"] > 0
        assert "distribution" in data


# ═════════════════════════════════════════════════════════════════════
# P16b: Log Aggregation
# ═════════════════════════════════════════════════════════════════════

class TestLogAggregation:
    """Test log capture and serving."""

    def test_memu_logs_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/logs" in routes

    def test_agentic_logs_endpoint_exists(self):
        routes = [r.path for r in ag.app.routes]
        assert "/logs" in routes

    def test_memu_logs_returns_entries(self):
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.get("/logs?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "count" in data

    def test_agentic_logs_returns_entries(self):
        from fastapi.testclient import TestClient
        client = TestClient(ag.app)
        resp = client.get("/logs?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data

    def test_logs_filter_by_level(self):
        """Filtering by level should return only matching entries."""
        from fastapi.testclient import TestClient
        client = TestClient(memu.app)
        resp = client.get("/logs?level=ERROR&limit=50")
        assert resp.status_code == 200
        data = resp.json()
        for entry in data["entries"]:
            assert entry["level"] == "ERROR"


# ═════════════════════════════════════════════════════════════════════
# P16c/d: Dashboard API Proxies
# ═════════════════════════════════════════════════════════════════════

class TestDashboardAPIs:
    """Test dashboard API proxy endpoints exist."""

    def test_goals_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/goals" in routes

    def test_goals_update_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/goals/update" in routes

    def test_drift_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/drift" in routes

    def test_memories_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/memories" in routes

    def test_memory_stats_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/memory/stats" in routes

    def test_struggle_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/struggle" in routes

    def test_feedback_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/feedback" in routes

    def test_feedback_stats_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/feedback/stats" in routes

    def test_logs_endpoint_exists(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/logs" in routes


# ═════════════════════════════════════════════════════════════════════
# Dashboard HTML View Checks
# ═════════════════════════════════════════════════════════════════════

class TestDashboardViews:
    """Test the dashboard SPA has the new views."""

    @pytest.fixture
    def html_content(self):
        path = os.path.join(ROOT, "dashboard", "static", "app.html")
        with open(path, "r") as f:
            return f.read()

    def test_goals_nav_exists(self, html_content):
        assert 'data-view="goals"' in html_content

    def test_memory_nav_exists(self, html_content):
        assert 'data-view="memory"' in html_content

    def test_logs_nav_exists(self, html_content):
        assert 'data-view="logs"' in html_content

    def test_goals_view_exists(self, html_content):
        assert 'id="goalsView"' in html_content

    def test_memory_view_exists(self, html_content):
        assert 'id="memoryView"' in html_content

    def test_logs_view_exists(self, html_content):
        assert 'id="logsView"' in html_content

    def test_feedback_buttons_js(self, html_content):
        assert 'addFeedbackButtons' in html_content

    def test_struggle_check_js(self, html_content):
        assert 'checkStruggle' in html_content

    def test_goal_form_exists(self, html_content):
        assert 'id="goalForm"' in html_content

    def test_memory_search_exists(self, html_content):
        assert 'id="memorySearch"' in html_content

    def test_log_level_filter(self, html_content):
        assert 'id="logLevel"' in html_content

    def test_keyboard_shortcuts_extended(self, html_content):
        assert "'5': 'goals'" in html_content
        assert "'6': 'memory'" in html_content
        assert "'7': 'logs'" in html_content
