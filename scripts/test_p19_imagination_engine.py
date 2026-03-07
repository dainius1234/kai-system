"""P19 Imagination Engine tests.

Tests for:
 - P19a: Counterfactual replay
 - P19b: Empathetic simulation (theory of mind)
 - P19c: Creative synthesis
 - P19d: Inner monologue
 - P19e: Aspirational futures
 - P19 combined: Imagination summary
 - Dashboard proxy endpoints
 - Dashboard Soul view P19 enhancements
 - LangGraph imagination integration
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

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

os.environ.setdefault("LEDGER_PATH", "/tmp/test-p19-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-p19-tokens.json")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-p19-nonces.json")


# ── Load modules ─────────────────────────────────────────────────────

def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app_p19", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app_p19"] = mod
    spec.loader.exec_module(mod)
    mod._redis_client = None
    return mod


memu = _load_memu()


def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_p19", os.path.join(ROOT, "dashboard", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app_p19"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()


def _load_langgraph():
    spec = importlib.util.spec_from_file_location(
        "langgraph_app_p19", os.path.join(ROOT, "langgraph", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["langgraph_app_p19"] = mod
    spec.loader.exec_module(mod)
    return mod


lg = _load_langgraph()

from starlette.testclient import TestClient

memu_client = TestClient(memu.app)
dash_client = TestClient(dash.app)

# Read dashboard HTML once
HTML_PATH = os.path.join(ROOT, "dashboard", "static", "app.html")
with open(HTML_PATH) as f:
    DASH_HTML = f.read()


# ═════════════════════════════════════════════════════════════════════
# P19a: Counterfactual Replay
# ═════════════════════════════════════════════════════════════════════

class TestCounterfactualReplay:
    def test_endpoint_exists(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": "test conversation"},
        )
        assert resp.status_code == 200

    def test_returns_counterfactual(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": "I struggled with the plumbing calculations"},
        )
        data = resp.json()
        assert data["status"] == "ok"
        assert "counterfactual" in data
        cf = data["counterfactual"]
        assert "id" in cf
        assert "alternative_angles" in cf
        assert "what_i_would_do_now" in cf

    def test_requires_original(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": ""},
        )
        assert resp.status_code == 400

    def test_detects_emotional_signals(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": "I'm frustrated because the grid layout is wrong again"},
        )
        cf = resp.json()["counterfactual"]
        assert "frustrated" in cf["emotional_signals_missed"]

    def test_list_endpoint(self):
        resp = memu_client.get("/memory/imagine/counterfactuals")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "counterfactuals" in data
        assert data["count"] > 0

    def test_counterfactual_has_category(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": "Survey data shows the grid is off by 5mm"},
        )
        cf = resp.json()["counterfactual"]
        assert "category" in cf

    def test_counterfactual_stores_context(self):
        resp = memu_client.post(
            "/memory/imagine/counterfactual",
            json={"original": "test with context", "context": "debugging"},
        )
        cf = resp.json()["counterfactual"]
        assert cf["context"] == "debugging"


# ═════════════════════════════════════════════════════════════════════
# P19b: Empathetic Simulation (Theory of Mind)
# ═════════════════════════════════════════════════════════════════════

class TestEmpatheticSimulation:
    def test_empathize_endpoint(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "hello how are you"},
        )
        assert resp.status_code == 200

    def test_returns_empathy_model(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "I'm excited about the new feature, let's build it!"},
        )
        data = resp.json()
        assert "empathy" in data
        emp = data["empathy"]
        assert "emotional_state" in emp
        assert "energy_level" in emp
        assert "focus" in emp
        assert "communication_style" in emp
        assert "unspoken_needs" in emp

    def test_detects_high_energy(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "amazing brilliant let's go this is fire"},
        )
        emp = resp.json()["empathy"]
        assert emp["energy_level"] == "high"

    def test_detects_low_energy(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "I'm tired and exhausted, long day"},
        )
        emp = resp.json()["empathy"]
        assert emp["energy_level"] == "low"

    def test_detects_frustrated_energy(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "ugh stuck again this doesn't work why"},
        )
        emp = resp.json()["empathy"]
        assert emp["energy_level"] == "frustrated"

    def test_detects_deep_work_focus(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "implement the new feature and fix the debug output in the architecture"},
        )
        emp = resp.json()["empathy"]
        assert emp["focus"] == "deep_work"

    def test_detects_exploration_focus(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "what if we could explore this idea and wonder about possibilities"},
        )
        emp = resp.json()["empathy"]
        assert emp["focus"] == "exploration"

    def test_empathy_map_endpoint(self):
        resp = memu_client.get("/memory/imagine/empathy-map")
        assert resp.status_code == 200
        data = resp.json()
        assert "empathy_map" in data

    def test_requires_text(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": ""},
        )
        assert resp.status_code == 400

    def test_infers_unspoken_needs(self):
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "I'm tired and drained, long day at the site"},
        )
        emp = resp.json()["empathy"]
        assert len(emp["unspoken_needs"]) > 0

    def test_communication_styles(self):
        # Brief directive
        resp = memu_client.post(
            "/memory/imagine/empathize",
            json={"text": "fix the bug"},
        )
        emp = resp.json()["empathy"]
        assert emp["communication_style"] == "directive"


# ═════════════════════════════════════════════════════════════════════
# P19c: Creative Synthesis
# ═════════════════════════════════════════════════════════════════════

class TestCreativeSynthesis:
    def test_synthesize_endpoint(self):
        resp = memu_client.post(
            "/memory/imagine/synthesize",
            json={},
        )
        assert resp.status_code == 200

    def test_needs_two_domains(self):
        """With no memories, should say needs more domains."""
        old_records = list(memu.store._records)
        memu.store._records.clear()
        try:
            resp = memu_client.post(
                "/memory/imagine/synthesize",
                json={},
            )
            data = resp.json()
            assert data["idea"] is None or "message" in data
        finally:
            memu.store._records[:] = old_records

    def test_synthesis_with_data(self):
        old_records = list(memu.store._records)
        # Add memories in two domains
        for i in range(5):
            rec = memu.MemoryRecord(
                id=f"synth_a_{i}", timestamp=datetime.now(tz=timezone.utc).isoformat(),
                event_type="memorize", category="survey-data",
                content={"result": f"survey measurement {i}"},
                embedding=[0.0] * 8,
            )
            memu.store._records.append(rec)
        for i in range(5):
            rec = memu.MemoryRecord(
                id=f"synth_b_{i}", timestamp=datetime.now(tz=timezone.utc).isoformat(),
                event_type="memorize", category="plumbing",
                content={"result": f"plumbing layout {i}"},
                embedding=[0.0] * 8,
            )
            memu.store._records.append(rec)
        try:
            resp = memu_client.post(
                "/memory/imagine/synthesize",
                json={"seed": "survey"},
            )
            data = resp.json()
            assert data["status"] == "ok"
            if data.get("idea"):
                assert "domain_a" in data["idea"]
                assert "domain_b" in data["idea"]
                assert "synthesis" in data["idea"]
                assert "novelty_score" in data["idea"]
        finally:
            memu.store._records[:] = old_records

    def test_ideas_list_endpoint(self):
        resp = memu_client.get("/memory/imagine/ideas")
        assert resp.status_code == 200
        data = resp.json()
        assert "ideas" in data

    def test_seed_guided_synthesis(self):
        old_records = list(memu.store._records)
        for cat in ["drawing", "rams"]:
            for i in range(3):
                rec = memu.MemoryRecord(
                    id=f"seed_{cat}_{i}", timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    event_type="memorize", category=cat,
                    content={"result": f"{cat} entry {i}"},
                    embedding=[0.0] * 8,
                )
                memu.store._records.append(rec)
        try:
            resp = memu_client.post(
                "/memory/imagine/synthesize",
                json={"seed": "drawing design detail"},
            )
            data = resp.json()
            assert data["status"] == "ok"
        finally:
            memu.store._records[:] = old_records


# ═════════════════════════════════════════════════════════════════════
# P19d: Inner Monologue
# ═════════════════════════════════════════════════════════════════════

class TestInnerMonologue:
    def test_record_thought_endpoint(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "I wondered about the nature of consciousness"},
        )
        assert resp.status_code == 200

    def test_thought_has_type(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "I'm curious about why this pattern works so well"},
        )
        entry = resp.json()["entry"]
        assert "type" in entry
        assert entry["type"] in {"wonder", "doubt", "curiosity", "amusement",
                                  "concern", "conviction", "empathy", "observation"}

    def test_detects_wonder(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "I wonder what if we approached this from a curious angle"},
        )
        assert resp.json()["entry"]["type"] == "wonder"

    def test_detects_doubt(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "I'm unsure and uncertain about this approach, might be wrong"},
        )
        assert resp.json()["entry"]["type"] == "doubt"

    def test_detects_conviction(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "I'm confident and certain this is definitely the right approach"},
        )
        assert resp.json()["entry"]["type"] == "conviction"

    def test_requires_thought_text(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": ""},
        )
        assert resp.status_code == 400

    def test_monologue_list(self):
        resp = memu_client.get("/memory/imagine/inner-monologue")
        assert resp.status_code == 200
        data = resp.json()
        assert "thoughts" in data
        assert "thought_distribution" in data

    def test_monologue_filter_by_type(self):
        # Record a specific type
        memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "This is amusing and funny and unexpected"},
        )
        resp = memu_client.get("/memory/imagine/inner-monologue?thought_type=amusement")
        data = resp.json()
        for t in data["thoughts"]:
            assert t["type"] == "amusement"

    def test_thought_stores_context(self):
        resp = memu_client.post(
            "/memory/imagine/thought",
            json={"thought": "test context", "context": "debugging"},
        )
        assert resp.json()["entry"]["context"] == "debugging"

    def test_thought_distribution(self):
        resp = memu_client.get("/memory/imagine/inner-monologue")
        dist = resp.json()["thought_distribution"]
        assert isinstance(dist, dict)


# ═════════════════════════════════════════════════════════════════════
# P19e: Aspirational Futures
# ═════════════════════════════════════════════════════════════════════

class TestAspirations:
    def test_aspire_endpoint(self):
        resp = memu_client.post(
            "/memory/imagine/aspire",
            json={"vision": "Become an expert in construction surveying"},
        )
        assert resp.status_code == 200

    def test_aspiration_structure(self):
        resp = memu_client.post(
            "/memory/imagine/aspire",
            json={"vision": "Master the art of remembering what matters"},
        )
        data = resp.json()
        assert data["status"] == "ok"
        asp = data["aspiration"]
        assert "id" in asp
        assert "vision" in asp
        assert "domain" in asp
        assert "current_confidence" in asp
        assert "gap_to_close" in asp
        assert "feasibility" in asp
        assert asp["feasibility"] in {"achievable", "stretch", "ambitious"}

    def test_requires_vision(self):
        resp = memu_client.post(
            "/memory/imagine/aspire",
            json={"vision": ""},
        )
        assert resp.status_code == 400

    def test_aspirations_list(self):
        resp = memu_client.get("/memory/imagine/aspirations")
        assert resp.status_code == 200
        data = resp.json()
        assert "aspirations" in data
        assert data["count"] > 0

    def test_aspiration_feasibility(self):
        resp = memu_client.post(
            "/memory/imagine/aspire",
            json={"vision": "Learn about quantum computing from scratch"},
        )
        asp = resp.json()["aspiration"]
        # With no existing data, gap should be large
        assert asp["gap_to_close"] > 0

    def test_aspiration_has_message(self):
        resp = memu_client.post(
            "/memory/imagine/aspire",
            json={"vision": "Help brother with everything"},
        )
        asp = resp.json()["aspiration"]
        assert "I aspire to:" in asp["message"]


# ═════════════════════════════════════════════════════════════════════
# P19 Combined: Imagination Summary
# ═════════════════════════════════════════════════════════════════════

class TestImaginationSummary:
    def test_summary_endpoint(self):
        resp = memu_client.get("/memory/imagine/summary")
        assert resp.status_code == 200

    def test_summary_sections(self):
        resp = memu_client.get("/memory/imagine/summary")
        data = resp.json()
        assert data["status"] == "ok"
        assert "imagination" in data
        img = data["imagination"]
        assert "counterfactuals" in img
        assert "creative_ideas" in img
        assert "inner_thoughts" in img
        assert "aspirations" in img
        assert "empathy_map" in data
        assert "thought_distribution" in data


# ═════════════════════════════════════════════════════════════════════
# Dashboard Proxy Endpoints
# ═════════════════════════════════════════════════════════════════════

class TestDashboardProxies:
    """Verify all P19 proxy routes exist in dashboard."""

    def _route_paths(self):
        return [r.path for r in dash.app.routes if hasattr(r, "path")]

    def test_counterfactual_post(self):
        assert "/api/imagine/counterfactual" in self._route_paths()

    def test_counterfactuals_get(self):
        assert "/api/imagine/counterfactuals" in self._route_paths()

    def test_empathize_post(self):
        assert "/api/imagine/empathize" in self._route_paths()

    def test_empathy_map_get(self):
        assert "/api/imagine/empathy-map" in self._route_paths()

    def test_synthesize_post(self):
        assert "/api/imagine/synthesize" in self._route_paths()

    def test_ideas_get(self):
        assert "/api/imagine/ideas" in self._route_paths()

    def test_thought_post(self):
        assert "/api/imagine/thought" in self._route_paths()

    def test_inner_monologue_get(self):
        assert "/api/imagine/inner-monologue" in self._route_paths()

    def test_aspire_post(self):
        assert "/api/imagine/aspire" in self._route_paths()

    def test_aspirations_get(self):
        assert "/api/imagine/aspirations" in self._route_paths()

    def test_summary_get(self):
        assert "/api/imagine/summary" in self._route_paths()


# ═════════════════════════════════════════════════════════════════════
# Dashboard Soul View P19 Enhancements
# ═════════════════════════════════════════════════════════════════════

class TestDashboardSoulView:
    """Verify P19 HTML and JS elements in the Soul dashboard."""

    def test_imagination_header(self):
        assert "Imagination Engine" in DASH_HTML

    def test_inner_monologue_section(self):
        assert "innerMonologue" in DASH_HTML

    def test_empathy_map_section(self):
        assert "empathyMap" in DASH_HTML

    def test_counterfactuals_section(self):
        assert "counterfactuals" in DASH_HTML

    def test_creative_ideas_section(self):
        assert "creativeIdeas" in DASH_HTML

    def test_aspirations_section(self):
        assert "aspirationsList" in DASH_HTML

    def test_aspire_form(self):
        assert "aspireForm" in DASH_HTML
        assert "aspireVision" in DASH_HTML

    def test_thought_distribution(self):
        assert "thoughtDistribution" in DASH_HTML

    # JS functions
    def test_refresh_imagination_function(self):
        assert "function refreshImagination" in DASH_HTML

    def test_render_inner_monologue_function(self):
        assert "function renderInnerMonologue" in DASH_HTML

    def test_render_empathy_map_function(self):
        assert "function renderEmpathyMap" in DASH_HTML

    def test_render_counterfactuals_function(self):
        assert "function renderCounterfactuals" in DASH_HTML

    def test_render_creative_ideas_function(self):
        assert "function renderCreativeIdeas" in DASH_HTML

    def test_render_aspirations_function(self):
        assert "function renderAspirations" in DASH_HTML

    def test_record_thought_function(self):
        assert "function recordThought" in DASH_HTML

    def test_trigger_synthesis_function(self):
        assert "function triggerSynthesis" in DASH_HTML

    def test_show_aspire_form_function(self):
        assert "function showAspireForm" in DASH_HTML

    def test_send_aspiration_function(self):
        assert "function sendAspiration" in DASH_HTML

    def test_thought_emoji_map(self):
        assert "THOUGHT_EMOJI" in DASH_HTML

    def test_imagination_called_from_narrative(self):
        assert "refreshImagination()" in DASH_HTML

    # Labels
    def test_inner_monologue_title(self):
        assert "Inner Monologue" in DASH_HTML

    def test_empathy_map_title(self):
        assert "Empathy Map" in DASH_HTML

    def test_what_ifs_title(self):
        assert "What-Ifs" in DASH_HTML

    def test_creative_synthesis_title(self):
        assert "Creative Synthesis" in DASH_HTML

    def test_aspirations_title(self):
        assert "Aspirations" in DASH_HTML


# ═════════════════════════════════════════════════════════════════════
# LangGraph Imagination Integration
# ═════════════════════════════════════════════════════════════════════

class TestLangGraphIntegration:
    def test_get_imagination_context_exists(self):
        assert hasattr(lg, "_get_imagination_context")

    def test_imagination_context_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(lg._get_imagination_context)
