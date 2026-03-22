"""P17 Emotional Intelligence & Self-Awareness tests.

Tests for:
 - P17a: Emotional memory tracking
 - P17b: Self-reflection journal
 - P17c: Relationship timeline
 - P17d: Epistemic humility (domain confidence)
 - P17e: Confession engine
 - Dashboard EQ proxies and views
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

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

os.environ.setdefault("LEDGER_PATH", "/tmp/test-p17-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-p17-tokens.json")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-p17-nonces.json")


# ── Load memu-core ───────────────────────────────────────────────────
def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app_p17", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app_p17"] = mod
    spec.loader.exec_module(mod)
    mod._redis_client = None
    return mod


memu = _load_memu()


# ── Load dashboard ───────────────────────────────────────────────────
def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_p17", os.path.join(ROOT, "dashboard", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app_p17"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()

# ASGI test clients
from starlette.testclient import TestClient

memu_client = TestClient(memu.app)
dash_client = TestClient(dash.app)


# ═════════════════════════════════════════════════════════════════════
# P17a: Emotional Memory Tests
# ═════════════════════════════════════════════════════════════════════

class TestEmotionalMemory:
    """Test emotion detection and timeline tracking."""

    def test_emotion_record_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/emotion/record" in routes

    def test_emotion_timeline_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/emotion/timeline" in routes

    def test_detect_frustrated(self):
        emo, intensity = memu._detect_emotion("This is so frustrating and useless, ugh!")
        assert emo == "frustrated"
        assert intensity > 0.3

    def test_detect_happy(self):
        emo, intensity = memu._detect_emotion("That's amazing, brilliant work brother!")
        assert emo in ("happy", "grateful")
        assert intensity > 0.2

    def test_detect_neutral(self):
        emo, intensity = memu._detect_emotion("Please pass me the data file")
        assert emo == "neutral"

    def test_record_emotion_builds_timeline(self):
        memu._emotional_timeline.clear()
        memu._record_emotion("test_session", "I'm so stressed about this deadline")
        assert len(memu._emotional_timeline) >= 1
        assert memu._emotional_timeline[-1]["emotion"] == "stressed"

    def test_emotion_record_api(self):
        memu._emotional_timeline.clear()
        resp = memu_client.post("/memory/emotion/record", json={
            "session_id": "api_test",
            "text": "This is great, I love it!"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["emotion"] in ("happy", "grateful")

    def test_emotion_timeline_api(self):
        memu._emotional_timeline.clear()
        memu._record_emotion("tl_test", "So frustrated")
        memu._record_emotion("tl_test", "Wait, that's actually great!")
        resp = memu_client.get("/memory/emotion/timeline", params={"session_id": "tl_test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2
        assert data["arc"] != "stable"  # emotion changed

    def test_emotion_text_required(self):
        resp = memu_client.post("/memory/emotion/record", json={
            "session_id": "test", "text": ""
        })
        assert resp.status_code == 400


# ═════════════════════════════════════════════════════════════════════
# P17b: Self-Reflection Journal Tests
# ═════════════════════════════════════════════════════════════════════

class TestSelfReflection:
    """Test self-reflection generation and retrieval."""

    def test_reflect_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/self-reflect" in routes

    def test_reflections_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/self-reflections" in routes

    def test_generate_reflection(self):
        resp = memu_client.post("/memory/self-reflect", json={})
        assert resp.status_code == 200
        data = resp.json()
        ref = data["reflection"]
        assert "strengths" in ref
        assert "weaknesses" in ref
        assert "insights" in ref
        assert "emotional_balance" in ref

    def test_reflections_list(self):
        memu._reflection_journal.clear()
        memu_client.post("/memory/self-reflect", json={})
        memu_client.post("/memory/self-reflect", json={})
        resp = memu_client.get("/memory/self-reflections", params={"limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 2

    def test_reflection_uses_feedback(self):
        memu._feedback_store.clear()
        memu._feedback_store.extend([
            {"rating": 1, "category": "general"},
            {"rating": 1, "category": "general"},
            {"rating": 5, "category": "general"},
        ])
        memu._reflection_journal.clear()
        resp = memu_client.post("/memory/self-reflect", json={})
        data = resp.json()
        ref = data["reflection"]
        assert ref["corrections_total"] == 2
        assert ref["positives_total"] == 1


# ═════════════════════════════════════════════════════════════════════
# P17c: Relationship Timeline Tests
# ═════════════════════════════════════════════════════════════════════

class TestRelationshipTimeline:
    """Test relationship narrative and milestones."""

    def test_relationship_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/relationship" in routes

    def test_milestone_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/relationship/milestone" in routes

    def test_relationship_returns_stats(self):
        resp = memu_client.get("/memory/relationship")
        assert resp.status_code == 200
        data = resp.json()
        assert "days_together" in data
        assert "total_memories" in data
        assert "corrections_given" in data
        assert "top_categories" in data
        assert "emotional_journey" in data

    def test_add_milestone(self):
        memu._relationship_milestones.clear()
        resp = memu_client.post("/memory/relationship/milestone", json={
            "title": "First voice conversation",
            "description": "Kai responded to spoken word for the first time"
        })
        assert resp.status_code == 200
        assert len(memu._relationship_milestones) == 1

    def test_milestone_requires_title(self):
        resp = memu_client.post("/memory/relationship/milestone", json={
            "title": "", "description": "test"
        })
        assert resp.status_code == 400


# ═════════════════════════════════════════════════════════════════════
# P17d: Epistemic Humility Tests
# ═════════════════════════════════════════════════════════════════════

class TestEpistemicHumility:
    """Test domain confidence tracking and warnings."""

    def test_confidence_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/confidence" in routes

    def test_confidence_check_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/confidence/check" in routes

    def test_domain_confidence_computes(self):
        resp = memu_client.get("/memory/confidence")
        assert resp.status_code == 200
        data = resp.json()
        assert "domains" in data
        assert "low_confidence_domains" in data
        assert "high_confidence_domains" in data

    def test_confidence_check_returns_assessment(self):
        resp = memu_client.get("/memory/confidence/check", params={"query": "scaffolding regulations"})
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence" in data
        assert "flag" in data
        assert "should_warn" in data
        assert "query_category" in data

    def test_compute_domain_confidence_fn(self):
        result = memu._compute_domain_confidence()
        # should return a dict of domains
        assert isinstance(result, dict)
        for domain, info in result.items():
            assert "confidence" in info
            assert "flag" in info
            assert info["flag"] in ("low", "medium", "high")


# ═════════════════════════════════════════════════════════════════════
# P17e: Confession Engine Tests
# ═════════════════════════════════════════════════════════════════════

class TestConfessionEngine:
    """Test proactive accountability checking."""

    def test_confess_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/confess" in routes

    def test_confess_requires_text(self):
        resp = memu_client.post("/memory/confess", json={"correction": ""})
        assert resp.status_code == 400

    def test_confess_returns_structure(self):
        resp = memu_client.post("/memory/confess", json={
            "correction": "scaffolding regs require edge protection at 2m not 3m"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "confessions" in data
        assert "correction_category" in data
        assert isinstance(data["confessions"], list)

    def test_confession_cooldown(self):
        memu._confession_cooldown.clear()
        # first call sets cooldown
        resp1 = memu_client.post("/memory/confess", json={
            "correction": "VAT rate is 20% not 17.5%",
            "category": "test_cooldown_cat",
        })
        data1 = resp1.json()
        # second call with same category should hit cooldown
        resp2 = memu_client.post("/memory/confess", json={
            "correction": "another VAT issue",
            "category": "test_cooldown_cat",
        })
        data2 = resp2.json()
        # only one of them should NOT be in cooldown
        # (first might have confessions or not, but second should be cooldown)
        if data1.get("confessions"):
            assert data2.get("reason") == "cooldown_active"


# ═════════════════════════════════════════════════════════════════════
# EQ Summary Tests
# ═════════════════════════════════════════════════════════════════════

class TestEQSummary:
    """Test the combined EQ summary endpoint."""

    def test_eq_summary_endpoint_exists(self):
        routes = [r.path for r in memu.app.routes]
        assert "/memory/eq/summary" in routes

    def test_eq_summary_returns_all_sections(self):
        resp = memu_client.get("/memory/eq/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "emotional_state" in data
        assert "self_awareness" in data
        assert "epistemic_humility" in data
        assert "relationship" in data

    def test_eq_summary_emotional_state(self):
        memu._emotional_timeline.clear()
        memu._record_emotion("eq_test", "I'm really happy today!")
        resp = memu_client.get("/memory/eq/summary")
        data = resp.json()
        assert data["emotional_state"]["current_mood"] == "happy"


# ═════════════════════════════════════════════════════════════════════
# Dashboard API Proxy Tests
# ═════════════════════════════════════════════════════════════════════

class TestDashboardEQAPIs:
    """Test that dashboard proxy endpoints exist for all P17 features."""

    def test_emotion_timeline_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/emotion/timeline" in routes

    def test_emotion_record_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/emotion/record" in routes

    def test_reflect_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/reflect" in routes

    def test_reflections_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/reflections" in routes

    def test_relationship_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/relationship" in routes

    def test_milestone_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/relationship/milestone" in routes

    def test_confidence_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/confidence" in routes

    def test_eq_summary_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/eq/summary" in routes

    def test_confess_proxy(self):
        routes = [r.path for r in dash.app.routes]
        assert "/api/confess" in routes


# ═════════════════════════════════════════════════════════════════════
# Dashboard View Tests
# ═════════════════════════════════════════════════════════════════════

class TestDashboardEQView:
    """Test that the EQ/Soul view exists in the dashboard HTML."""

    @pytest.fixture(autouse=True)
    def load_html(self):
        html_path = os.path.join(ROOT, "dashboard", "static", "app.html")
        with open(html_path) as f:
            self.html = f.read()

    def test_eq_nav_exists(self):
        assert 'data-view="eq"' in self.html

    def test_eq_view_section(self):
        assert 'id="eqView"' in self.html

    def test_eq_mood_card(self):
        assert 'id="eqMood"' in self.html

    def test_eq_timeline(self):
        assert 'id="eqTimeline"' in self.html

    def test_eq_confidence_section(self):
        assert 'id="eqConfidence"' in self.html

    def test_eq_reflections_section(self):
        assert 'id="eqReflections"' in self.html

    def test_eq_milestones_section(self):
        assert 'id="eqMilestones"' in self.html

    def test_refresh_eq_js(self):
        assert 'function refreshEQ' in self.html

    def test_trigger_reflection_js(self):
        assert 'function triggerReflection' in self.html

    def test_render_emotion_timeline_js(self):
        assert 'function renderEmotionTimeline' in self.html

    def test_render_domain_confidence_js(self):
        assert 'function renderDomainConfidence' in self.html

    def test_keyboard_shortcut_eq(self):
        assert "'8': 'eq'" in self.html

    def test_soul_dashboard_title(self):
        assert "Soul Dashboard" in self.html
