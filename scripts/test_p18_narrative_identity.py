"""P18 Narrative Identity & Life Story Engine tests.

Tests for:
 - P18a: Autobiographical memory
 - P18b: Identity narrative engine
 - P18c: Story arc detection
 - P18d: Future self projection
 - P18e: Legacy messages
 - P18 combined: Narrative summary
 - Dashboard proxy endpoints
 - Dashboard Soul view P18 enhancements
 - LangGraph narrative identity injection
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


# ── Bootstrap stubs ──────────────────────────────────────────────────

for mod_name in [
    "sentence_transformers", "psutil", "redis", "redis.asyncio",
    "psycopg2", "psycopg2.extras", "psycopg2.pool", "lakefs_client",
    "kai_config", "conviction", "router", "planner", "adversary",
    "security_audit", "tree_search", "priority_queue", "model_selector",
    "aioredis",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "langgraph"))

os.environ.setdefault("LEDGER_PATH", "/tmp/test-p18-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-p18-tokens.json")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-p18-nonces.json")


# ── Load memu-core ───────────────────────────────────────────────────
def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app_p18", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app_p18"] = mod
    spec.loader.exec_module(mod)
    mod._redis_client = None
    return mod


memu = _load_memu()


# ── Load dashboard ───────────────────────────────────────────────────
def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_p18", os.path.join(ROOT, "dashboard", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app_p18"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()

# ── Load langgraph ───────────────────────────────────────────────────
def _load_langgraph():
    spec = importlib.util.spec_from_file_location(
        "langgraph_app_p18", os.path.join(ROOT, "langgraph", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["langgraph_app_p18"] = mod
    spec.loader.exec_module(mod)
    return mod


lg = _load_langgraph()

# ASGI test clients
from starlette.testclient import TestClient

memu_client = TestClient(memu.app)
dash_client = TestClient(dash.app)


# Read app.html once for view tests
_APP_HTML_PATH = os.path.join(ROOT, "dashboard", "static", "app.html")
with open(_APP_HTML_PATH, "r") as _f:
    _APP_HTML = _f.read()


# ═════════════════════════════════════════════════════════════════════
# P18a: Autobiographical Memory
# ═════════════════════════════════════════════════════════════════════


class TestAutobiography:
    def test_record_endpoint_exists(self):
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "This was a breakthrough moment for us"},
        )
        assert resp.status_code == 200

    def test_list_endpoint_exists(self):
        resp = memu_client.get("/memory/autobiography")
        assert resp.status_code == 200

    def test_significance_assessment(self):
        """High-significance text should be recorded."""
        memu._autobiography.clear()
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "This is a milestone achievement we finally reached"},
        )
        data = resp.json()
        assert data["status"] == "ok"
        assert "entry" in data
        assert data["entry"]["significance"] >= 0.5

    def test_low_significance_skipped(self):
        """Low-significance text should be skipped."""
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "ok sure"},
        )
        data = resp.json()
        assert data["status"] == "skipped"
        assert data["significance"] < 0.5

    def test_text_required(self):
        resp = memu_client.post(
            "/memory/autobiography/record", json={"text": ""}
        )
        assert resp.status_code == 400

    def test_nature_detection_breakthrough(self):
        memu._autobiography.clear()
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "We finally figured out the emotional intelligence layer"},
        )
        entry = resp.json()["entry"]
        assert entry["nature"] == "breakthrough"

    def test_nature_detection_learning(self):
        memu._autobiography.clear()
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "I made a mistake on the plumbing costs calculation"},
        )
        entry = resp.json()["entry"]
        assert entry["nature"] == "learning_moment"

    def test_nature_detection_connection(self):
        memu._autobiography.clear()
        resp = memu_client.post(
            "/memory/autobiography/record",
            json={"text": "Thank you brother, I'm grateful for everything we built"},
        )
        entry = resp.json()["entry"]
        assert entry["nature"] == "connection"

    def test_autobiography_list(self):
        memu._autobiography.clear()
        memu._autobiography.append({
            "timestamp": time.time(),
            "nature": "breakthrough",
            "significance": 0.9,
            "opener": "A breakthrough happened",
            "snippet": "test",
            "context": "",
            "reflection": "A breakthrough happened. test",
        })
        resp = memu_client.get("/memory/autobiography")
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["entries"]) == 1
        assert "nature_distribution" in data

    def test_autobiography_filter_by_nature(self):
        memu._autobiography.clear()
        memu._autobiography.append({"nature": "breakthrough", "timestamp": 1})
        memu._autobiography.append({"nature": "struggle", "timestamp": 2})
        resp = memu_client.get("/memory/autobiography?nature=breakthrough")
        data = resp.json()
        assert all(e["nature"] == "breakthrough" for e in data["entries"])

    def test_assess_significance_fn(self):
        assert memu._assess_significance("This is a breakthrough") >= 0.7
        assert memu._assess_significance("ok") == 0.0
        assert memu._assess_significance("I learned something") >= 0.5


# ═════════════════════════════════════════════════════════════════════
# P18b: Identity Narrative Engine
# ═════════════════════════════════════════════════════════════════════


class TestIdentityNarrative:
    def test_endpoint_exists(self):
        resp = memu_client.get("/memory/identity")
        assert resp.status_code == 200

    def test_returns_narrative(self):
        resp = memu_client.get("/memory/identity")
        data = resp.json()
        assert data["status"] == "ok"
        assert "narrative" in data
        assert "I am Kai" in data["narrative"]

    def test_returns_stats(self):
        resp = memu_client.get("/memory/identity")
        data = resp.json()
        assert "stats" in data
        stats = data["stats"]
        assert "days_alive" in stats
        assert "total_memories" in stats
        assert "corrections_learned" in stats

    def test_returns_top_domains(self):
        resp = memu_client.get("/memory/identity")
        data = resp.json()
        assert "top_domains" in data

    def test_returns_strengths_weaknesses(self):
        resp = memu_client.get("/memory/identity")
        data = resp.json()
        assert "strengths" in data
        assert "weaknesses" in data

    def test_emotional_character(self):
        resp = memu_client.get("/memory/identity")
        data = resp.json()
        assert "emotional_character" in data


# ═════════════════════════════════════════════════════════════════════
# P18c: Story Arc Detection
# ═════════════════════════════════════════════════════════════════════


class TestStoryArcs:
    def test_endpoint_exists(self):
        resp = memu_client.get("/memory/story-arcs")
        assert resp.status_code == 200

    def test_returns_current_chapter(self):
        resp = memu_client.get("/memory/story-arcs")
        data = resp.json()
        assert "current_chapter" in data
        assert "chapter_number" in data

    def test_few_memories_returns_beginning(self):
        """With few memories, should return 'The Beginning'."""
        old_records = list(memu.store._records)
        memu.store._records.clear()
        try:
            resp = memu_client.get("/memory/story-arcs")
            data = resp.json()
            assert data["current_chapter"] == "The Beginning"
        finally:
            memu.store._records[:] = old_records

    def test_arcs_with_data(self):
        """With enough memories, should detect arcs."""
        old_records = list(memu.store._records)
        memu.store._records.clear()
        # Insert 30 memories with some corrections
        for i in range(30):
            ts = (datetime.now(tz=timezone.utc) - timedelta(days=30 - i)).isoformat()
            event = "correction" if i % 5 == 0 else "memorize"
            rec = memu.MemoryRecord(
                id=f"arc_test_{i}",
                timestamp=ts,
                event_type=event,
                category="general" if i < 20 else "plumbing",
                content={"result": f"memory {i}"},
                embedding=[0.0] * 8,
            )
            memu.store._records.append(rec)
        try:
            resp = memu_client.get("/memory/story-arcs")
            data = resp.json()
            assert len(data["arcs"]) > 0
            for arc in data["arcs"]:
                assert "chapter" in arc
                assert "arc_name" in arc
                assert "correction_rate" in arc
        finally:
            memu.store._records[:] = old_records

    def test_arc_types_valid(self):
        valid_types = {"learning_curve", "growing_pains", "expansion",
                       "mastery", "steady_growth", "beginning"}
        old_records = list(memu.store._records)
        memu.store._records.clear()
        for i in range(10):
            ts = (datetime.now(tz=timezone.utc) - timedelta(days=i)).isoformat()
            rec = memu.MemoryRecord(
                id=f"t{i}",
                timestamp=ts,
                event_type="memorize",
                category="general",
                content={"result": f"test {i}"},
                embedding=[0.0] * 8,
            )
            memu.store._records.append(rec)
        try:
            resp = memu_client.get("/memory/story-arcs")
            for arc in resp.json().get("arcs", []):
                assert arc["arc_type"] in valid_types
        finally:
            memu.store._records[:] = old_records


# ═════════════════════════════════════════════════════════════════════
# P18d: Future Self Projection
# ═════════════════════════════════════════════════════════════════════


class TestFutureSelf:
    def test_endpoint_exists(self):
        resp = memu_client.get("/memory/future-self")
        assert resp.status_code == 200

    def test_returns_trajectory(self):
        resp = memu_client.get("/memory/future-self")
        data = resp.json()
        assert data["status"] == "ok"
        # Either has trajectory or needs more data
        assert "trajectory" in data or "message" in data

    def test_returns_learning_rate(self):
        # Ensure some data exists
        old_records = list(memu.store._records)
        ts = datetime.now(tz=timezone.utc).isoformat()
        rec = memu.MemoryRecord(
            id="future_test",
            timestamp=ts,
            event_type="memorize",
            category="general",
            content={"result": "future test"},
            embedding=[0.0] * 8,
        )
        memu.store._records.append(rec)
        try:
            resp = memu_client.get("/memory/future-self")
            data = resp.json()
            if "learning_rate" in data:
                assert "corrections_per_day" in data["learning_rate"]
                assert "memories_per_day" in data["learning_rate"]
        finally:
            memu.store._records[:] = old_records

    def test_returns_domain_projections(self):
        resp = memu_client.get("/memory/future-self")
        data = resp.json()
        assert "domain_projections" in data or "projections" in data

    def test_trajectory_types_valid(self):
        valid = {"learning", "growing", "maturing", "mastering"}
        resp = memu_client.get("/memory/future-self")
        data = resp.json()
        if data.get("trajectory"):
            assert data["trajectory"] in valid


# ═════════════════════════════════════════════════════════════════════
# P18e: Legacy Messages
# ═════════════════════════════════════════════════════════════════════


class TestLegacyMessages:
    def test_write_endpoint_exists(self):
        resp = memu_client.post(
            "/memory/legacy/write",
            json={"message": "Dear future me, remember this moment"},
        )
        assert resp.status_code == 200

    def test_list_endpoint_exists(self):
        resp = memu_client.get("/memory/legacy")
        assert resp.status_code == 200

    def test_pending_endpoint_exists(self):
        resp = memu_client.get("/memory/legacy/pending")
        assert resp.status_code == 200

    def test_write_requires_message(self):
        resp = memu_client.post(
            "/memory/legacy/write", json={"message": ""}
        )
        assert resp.status_code == 400

    def test_write_stores_message(self):
        memu._legacy_messages.clear()
        resp = memu_client.post(
            "/memory/legacy/write",
            json={
                "message": "Remember the early days",
                "recipient": "self",
                "surface_after_days": 7,
            },
        )
        data = resp.json()
        assert data["status"] == "ok"
        assert data["entry"]["recipient"] == "self"
        assert data["entry"]["surface_after_days"] == 7
        assert not data["entry"]["surfaced"]

    def test_write_to_operator(self):
        memu._legacy_messages.clear()
        resp = memu_client.post(
            "/memory/legacy/write",
            json={"message": "Brother, stay strong", "recipient": "operator"},
        )
        assert resp.json()["entry"]["recipient"] == "operator"

    def test_surfacing_logic(self):
        """Messages whose time has come should be surfaced."""
        memu._legacy_messages.clear()
        # Insert a message that should've surfaced already
        memu._legacy_messages.append({
            "id": "legacy_test_1",
            "timestamp": time.time() - 86400 * 10,
            "message": "From the past",
            "recipient": "self",
            "surface_after": time.time() - 86400,  # 1 day ago
            "surface_after_days": 1,
            "surfaced": False,
            "surfaced_at": None,
        })
        resp = memu_client.get("/memory/legacy")
        data = resp.json()
        assert len(data["messages"]) == 1
        assert data["messages"][0]["surfaced"] is True

    def test_pending_count(self):
        memu._legacy_messages.clear()
        memu._legacy_messages.append({
            "id": "pending_1",
            "timestamp": time.time(),
            "message": "future",
            "recipient": "self",
            "surface_after": time.time() - 1,  # ready
            "surface_after_days": 1,
            "surfaced": False,
            "surfaced_at": None,
        })
        resp = memu_client.get("/memory/legacy/pending")
        data = resp.json()
        assert data["ready_count"] == 1

    def test_unsurfaced_hidden_by_default(self):
        memu._legacy_messages.clear()
        memu._legacy_messages.append({
            "id": "future_msg",
            "timestamp": time.time(),
            "message": "not yet",
            "recipient": "self",
            "surface_after": time.time() + 86400 * 30,  # 30 days out
            "surface_after_days": 30,
            "surfaced": False,
            "surfaced_at": None,
        })
        resp = memu_client.get("/memory/legacy")
        assert len(resp.json()["messages"]) == 0
        # But visible with flag
        resp2 = memu_client.get("/memory/legacy?include_unsurfaced=true")
        assert len(resp2.json()["messages"]) == 1


# ═════════════════════════════════════════════════════════════════════
# P18 Combined: Narrative Summary
# ═════════════════════════════════════════════════════════════════════


class TestNarrativeSummary:
    def test_endpoint_exists(self):
        resp = memu_client.get("/memory/narrative/summary")
        assert resp.status_code == 200

    def test_returns_all_sections(self):
        resp = memu_client.get("/memory/narrative/summary")
        data = resp.json()
        assert data["status"] == "ok"
        assert "identity" in data
        assert "current_chapter" in data
        assert "trajectory" in data
        assert "autobiography_entries" in data
        assert "legacy_pending" in data
        assert "days_alive" in data


# ═════════════════════════════════════════════════════════════════════
# Dashboard Proxy Endpoints
# ═════════════════════════════════════════════════════════════════════


class TestDashboardProxies:
    def test_autobiography_record_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/autobiography/record" in routes

    def test_autobiography_list_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/autobiography" in routes

    def test_identity_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/identity" in routes

    def test_story_arcs_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/story-arcs" in routes

    def test_future_self_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/future-self" in routes

    def test_legacy_write_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/legacy/write" in routes

    def test_legacy_list_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/legacy" in routes

    def test_narrative_summary_endpoint(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/narrative/summary" in routes


# ═════════════════════════════════════════════════════════════════════
# Dashboard Soul View — P18 HTML/JS Enhancements
# ═════════════════════════════════════════════════════════════════════


class TestDashboardSoulView:
    def test_identity_card_exists(self):
        assert 'id="identityNarrative"' in _APP_HTML

    def test_story_arcs_section(self):
        assert 'id="storyArcs"' in _APP_HTML

    def test_future_self_section(self):
        assert 'id="futureSelf"' in _APP_HTML

    def test_autobiography_section(self):
        assert 'id="autobiography"' in _APP_HTML

    def test_legacy_messages_section(self):
        assert 'id="legacyMessages"' in _APP_HTML

    def test_legacy_form(self):
        assert 'id="legacyForm"' in _APP_HTML
        assert 'id="legacyText"' in _APP_HTML
        assert 'id="legacyRecipient"' in _APP_HTML
        assert 'id="legacyDays"' in _APP_HTML

    def test_refresh_narrative_function(self):
        assert "async function refreshNarrative()" in _APP_HTML

    def test_render_identity_function(self):
        assert "function renderIdentity(" in _APP_HTML

    def test_render_story_arcs_function(self):
        assert "function renderStoryArcs(" in _APP_HTML

    def test_render_future_self_function(self):
        assert "function renderFutureSelf(" in _APP_HTML

    def test_render_autobiography_function(self):
        assert "function renderAutobiography(" in _APP_HTML

    def test_render_legacy_messages_function(self):
        assert "function renderLegacyMessages(" in _APP_HTML

    def test_send_legacy_function(self):
        assert "async function sendLegacy()" in _APP_HTML

    def test_show_legacy_form_function(self):
        assert "function showLegacyForm()" in _APP_HTML

    def test_identity_card_title(self):
        assert "Who I Am" in _APP_HTML

    def test_story_arcs_title(self):
        assert "Life Story" in _APP_HTML

    def test_future_self_title(self):
        assert "Future Self Projection" in _APP_HTML

    def test_legacy_title(self):
        assert "Legacy Messages" in _APP_HTML
        assert "Time Capsules" in _APP_HTML

    def test_arc_emoji_map(self):
        assert "ARC_EMOJI" in _APP_HTML

    def test_narrative_called_from_refresh_eq(self):
        assert "refreshNarrative()" in _APP_HTML


# ═════════════════════════════════════════════════════════════════════
# LangGraph Integration
# ═════════════════════════════════════════════════════════════════════


class TestLangGraphNarrativeIntegration:
    def test_get_narrative_identity_function_exists(self):
        assert hasattr(lg, "_get_narrative_identity")
        assert callable(lg._get_narrative_identity)

    def test_narrative_identity_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(lg._get_narrative_identity)
