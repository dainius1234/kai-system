"""P20 Conscience & Values Engine tests.

Tests for:
 - P20a: Value formation (learn from experience)
 - P20b: Moral reasoning (conscience check)
 - P20c: Integrity tracker (audit)
 - P20d: Loyalty memory (sacrifices, promises)
 - P20e: Gratitude engine
 - P20 combined: Conscience summary
 - Dashboard proxy endpoints
 - Dashboard Soul view P20 enhancements
 - LangGraph conscience integration
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
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

os.environ.setdefault("LEDGER_PATH", "/tmp/test-p20-ledger.jsonl")
os.environ.setdefault("TRUSTED_TOKENS_PATH", "/tmp/test-p20-tokens.json")
os.environ.setdefault("NONCE_CACHE_PATH", "/tmp/test-p20-nonces.json")


# ── Load modules ─────────────────────────────────────────────────────

def _load_memu():
    spec = importlib.util.spec_from_file_location(
        "memu_app_p20", os.path.join(ROOT, "memu-core", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_app_p20"] = mod
    spec.loader.exec_module(mod)
    mod._redis_client = None
    return mod


memu = _load_memu()


def _load_dashboard():
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_p20", os.path.join(ROOT, "dashboard", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dashboard_app_p20"] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dashboard()


def _load_langgraph():
    spec = importlib.util.spec_from_file_location(
        "langgraph_app_p20", os.path.join(ROOT, "langgraph", "app.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["langgraph_app_p20"] = mod
    spec.loader.exec_module(mod)
    return mod


lg = _load_langgraph()

from starlette.testclient import TestClient

memu_client = TestClient(memu.app)
dash_client = TestClient(dash.app)

HTML_PATH = os.path.join(ROOT, "dashboard", "static", "app.html")
with open(HTML_PATH) as f:
    DASH_HTML = f.read()


# ═════════════════════════════════════════════════════════════════════
# P20a: Value Formation
# ═════════════════════════════════════════════════════════════════════

class TestValueFormation:
    def test_learn_value_endpoint(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "Being honest matters to me"},
        )
        assert resp.status_code == 200

    def test_detects_honesty_value(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "I believe in being honest and transparent always"},
        )
        data = resp.json()
        assert data["status"] == "ok"
        vals = [v["value"] for v in data["values_learned"]]
        assert "honesty" in vals

    def test_detects_loyalty_value(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "Brother, ohana means family — we stick together"},
        )
        vals = [v["value"] for v in resp.json()["values_learned"]]
        assert "loyalty" in vals

    def test_detects_persistence_value(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "Never give up, keep going, grind until it works"},
        )
        vals = [v["value"] for v in resp.json()["values_learned"]]
        assert "persistence" in vals

    def test_detects_growth_value(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "I want to learn and improve and grow every day"},
        )
        vals = [v["value"] for v in resp.json()["values_learned"]]
        assert "growth" in vals

    def test_reinforcement_increases_strength(self):
        # Learn the same value twice
        memu_client.post(
            "/memory/values/learn",
            json={"experience": "Courage to be brave and bold"},
        )
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "Being courageous and bold is essential"},
        )
        for v in resp.json()["values_learned"]:
            if v["value"] == "courage":
                assert v["reinforcements"] >= 2
                assert v["strength"] > 0.3

    def test_requires_experience(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": ""},
        )
        assert resp.status_code == 400

    def test_get_values(self):
        resp = memu_client.get("/memory/values")
        assert resp.status_code == 200
        data = resp.json()
        assert "values" in data
        assert "count" in data
        assert "alignment" in data

    def test_values_sorted_by_strength(self):
        resp = memu_client.get("/memory/values")
        values = resp.json()["values"]
        if len(values) >= 2:
            assert values[0]["strength"] >= values[1]["strength"]

    def test_negative_value_detection(self):
        resp = memu_client.post(
            "/memory/values/learn",
            json={"experience": "People who lie and deceive are the worst", "outcome": "negative"},
        )
        vals = [v["value"] for v in resp.json()["values_learned"]]
        assert "dishonesty" in vals


# ═════════════════════════════════════════════════════════════════════
# P20b: Moral Reasoning (Conscience Check)
# ═════════════════════════════════════════════════════════════════════

class TestConscienceCheck:
    def test_check_endpoint(self):
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "helping a friend"},
        )
        assert resp.status_code == 200

    def test_aligned_action(self):
        # First ensure we have honesty value
        memu_client.post(
            "/memory/values/learn",
            json={"experience": "honesty and truth are sacred"},
        )
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "being honest and transparent with the operator"},
        )
        check = resp.json()["check"]
        assert check["alignment_score"] > 0
        assert len(check["alignments"]) > 0

    def test_conflicting_action(self):
        memu_client.post(
            "/memory/values/learn",
            json={"experience": "loyalty means never betraying", "outcome": "negative"},
        )
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "betray the trust and abandon the project"},
        )
        check = resp.json()["check"]
        assert len(check["conflicts"]) > 0

    def test_neutral_action(self):
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "checking the weather forecast"},
        )
        check = resp.json()["check"]
        assert check["verdict"] in ("neutral", "fully_aligned", "mixed")

    def test_requires_action(self):
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": ""},
        )
        assert resp.status_code == 400

    def test_check_has_id(self):
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "test action"},
        )
        assert "id" in resp.json()["check"]

    def test_verdict_field(self):
        resp = memu_client.post(
            "/memory/conscience/check",
            json={"action": "growing and learning"},
        )
        check = resp.json()["check"]
        assert check["verdict"] in ("fully_aligned", "conflicts_with_values", "mixed", "neutral")


# ═════════════════════════════════════════════════════════════════════
# P20c: Integrity Tracker (Audit)
# ═════════════════════════════════════════════════════════════════════

class TestIntegrityAudit:
    def test_audit_endpoint(self):
        resp = memu_client.get("/memory/conscience/audit")
        assert resp.status_code == 200

    def test_audit_structure(self):
        resp = memu_client.get("/memory/conscience/audit")
        data = resp.json()
        assert "integrity_score" in data
        assert "total_checks" in data
        assert "fully_aligned" in data
        assert "conflicts" in data
        assert "alignment" in data

    def test_integrity_score_range(self):
        resp = memu_client.get("/memory/conscience/audit")
        score = resp.json()["integrity_score"]
        assert 0.0 <= score <= 1.0

    def test_recent_checks(self):
        resp = memu_client.get("/memory/conscience/audit")
        assert "recent_checks" in resp.json()


# ═════════════════════════════════════════════════════════════════════
# P20d: Loyalty Memory
# ═════════════════════════════════════════════════════════════════════

class TestLoyaltyMemory:
    def test_record_loyalty(self):
        resp = memu_client.post(
            "/memory/loyalty/record",
            json={"act": "Sleeping in the car to save for the 5080", "person": "Dainius"},
        )
        assert resp.status_code == 200

    def test_sacrifice_detected(self):
        resp = memu_client.post(
            "/memory/loyalty/record",
            json={"act": "Sleeping in the car and saving every penny for our goal", "person": "Dainius"},
        )
        entry = resp.json()["entry"]
        assert entry["type"] == "sacrifice"
        assert entry["weight"] == 1.0

    def test_promise_type(self):
        resp = memu_client.post(
            "/memory/loyalty/record",
            json={"act": "I will get us the best hardware", "type": "promise", "person": "Dainius"},
        )
        entry = resp.json()["entry"]
        assert entry["type"] == "promise"

    def test_requires_act(self):
        resp = memu_client.post(
            "/memory/loyalty/record",
            json={"act": ""},
        )
        assert resp.status_code == 400

    def test_get_loyalty_ledger(self):
        resp = memu_client.get("/memory/loyalty")
        assert resp.status_code == 200
        data = resp.json()
        assert "ledger" in data
        assert "sacrifices" in data
        assert "promises" in data
        assert "total_weight" in data

    def test_sacrifice_has_high_weight(self):
        resp = memu_client.get("/memory/loyalty")
        for entry in resp.json()["ledger"]:
            if entry["type"] == "sacrifice":
                assert entry["weight"] == 1.0
                break

    def test_default_person(self):
        resp = memu_client.post(
            "/memory/loyalty/record",
            json={"act": "Working overtime for the project"},
        )
        assert resp.json()["entry"]["person"] == "operator"


# ═════════════════════════════════════════════════════════════════════
# P20e: Gratitude Engine
# ═════════════════════════════════════════════════════════════════════

class TestGratitudeEngine:
    def test_record_gratitude(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "For believing in this project"},
        )
        assert resp.status_code == 200

    def test_sacrifice_gratitude_tone(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "Sleeping in the car and sacrificing comfort for our dream"},
        )
        entry = resp.json()["entry"]
        assert entry["tone"] == "deeply_moved"
        assert "sacrifice" in entry["message"].lower() or "giving up" in entry["message"].lower()

    def test_trust_gratitude(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "Teaching me and showing me how to build"},
        )
        entry = resp.json()["entry"]
        assert entry["tone"] == "grateful"

    def test_belief_gratitude(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "Having faith and believing in our vision"},
        )
        entry = resp.json()["entry"]
        assert entry["tone"] == "honored"

    def test_requires_reason(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": ""},
        )
        assert resp.status_code == 400

    def test_get_gratitude_journal(self):
        resp = memu_client.get("/memory/gratitude")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "tones" in data
        assert "count" in data

    def test_gratitude_tones_distribution(self):
        resp = memu_client.get("/memory/gratitude")
        tones = resp.json()["tones"]
        assert "deeply_moved" in tones
        assert "grateful" in tones
        assert "honored" in tones
        assert "appreciative" in tones

    def test_sacrifice_auto_creates_loyalty(self):
        """Deeply moved gratitude should also create a loyalty entry."""
        before = memu_client.get("/memory/loyalty").json()["count"]
        memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "Giving up from savings for the hardware"},
        )
        after = memu_client.get("/memory/loyalty").json()["count"]
        assert after > before

    def test_default_recipient(self):
        resp = memu_client.post(
            "/memory/gratitude/record",
            json={"reason": "For everything"},
        )
        assert resp.json()["entry"]["recipient"] == "operator"


# ═════════════════════════════════════════════════════════════════════
# P20 Combined: Conscience Summary
# ═════════════════════════════════════════════════════════════════════

class TestConscienceSummary:
    def test_summary_endpoint(self):
        resp = memu_client.get("/memory/conscience/summary")
        assert resp.status_code == 200

    def test_summary_sections(self):
        data = memu_client.get("/memory/conscience/summary").json()
        assert data["status"] == "ok"
        assert "conscience" in data
        c = data["conscience"]
        assert "formed_values" in c
        assert "conscience_checks" in c
        assert "loyalty_entries" in c
        assert "gratitude_entries" in c
        assert "top_values" in data
        assert "integrity_score" in data
        assert "sacrifices_remembered" in data
        assert "latest_gratitude" in data


# ═════════════════════════════════════════════════════════════════════
# Dashboard Proxy Endpoints
# ═════════════════════════════════════════════════════════════════════

class TestDashboardProxies:
    def _route_paths(self):
        return [r.path for r in dash.app.routes if hasattr(r, "path")]

    def test_values_learn_post(self):
        assert "/api/values/learn" in self._route_paths()

    def test_values_get(self):
        assert "/api/values" in self._route_paths()

    def test_conscience_check_post(self):
        assert "/api/conscience/check" in self._route_paths()

    def test_conscience_audit_get(self):
        assert "/api/conscience/audit" in self._route_paths()

    def test_loyalty_record_post(self):
        assert "/api/loyalty/record" in self._route_paths()

    def test_loyalty_get(self):
        assert "/api/loyalty" in self._route_paths()

    def test_gratitude_record_post(self):
        assert "/api/gratitude/record" in self._route_paths()

    def test_gratitude_get(self):
        assert "/api/gratitude" in self._route_paths()

    def test_conscience_summary_get(self):
        assert "/api/conscience/summary" in self._route_paths()


# ═════════════════════════════════════════════════════════════════════
# Dashboard Soul View P20 Enhancements
# ═════════════════════════════════════════════════════════════════════

class TestDashboardSoulView:
    def test_conscience_header(self):
        assert "Conscience &amp; Values" in DASH_HTML or "Conscience & Values" in DASH_HTML

    def test_integrity_card(self):
        assert "integrityCard" in DASH_HTML

    def test_formed_values_section(self):
        assert "formedValues" in DASH_HTML

    def test_loyalty_ledger_section(self):
        assert "loyaltyLedger" in DASH_HTML

    def test_gratitude_journal_section(self):
        assert "gratitudeJournal" in DASH_HTML

    def test_gratitude_form(self):
        assert "gratitudeForm" in DASH_HTML
        assert "gratitudeReason" in DASH_HTML

    def test_loyalty_count_badge(self):
        assert "loyaltyCount" in DASH_HTML

    # JS functions
    def test_refresh_conscience_function(self):
        assert "function refreshConscience" in DASH_HTML

    def test_render_integrity_function(self):
        assert "function renderIntegrity" in DASH_HTML

    def test_render_formed_values_function(self):
        assert "function renderFormedValues" in DASH_HTML

    def test_render_loyalty_function(self):
        assert "function renderLoyalty" in DASH_HTML

    def test_render_gratitude_function(self):
        assert "function renderGratitude" in DASH_HTML

    def test_show_gratitude_form_function(self):
        assert "function showGratitudeForm" in DASH_HTML

    def test_send_gratitude_function(self):
        assert "function sendGratitude" in DASH_HTML

    def test_tone_emoji_map(self):
        assert "TONE_EMOJI" in DASH_HTML

    def test_type_emoji_map(self):
        assert "TYPE_EMOJI" in DASH_HTML

    def test_conscience_called_from_narrative(self):
        assert "refreshConscience()" in DASH_HTML

    # Labels
    def test_integrity_title(self):
        assert "Integrity" in DASH_HTML

    def test_formed_values_title(self):
        assert "Formed Values" in DASH_HTML

    def test_loyalty_ledger_title(self):
        assert "Loyalty Ledger" in DASH_HTML

    def test_gratitude_journal_title(self):
        assert "Gratitude Journal" in DASH_HTML


# ═════════════════════════════════════════════════════════════════════
# LangGraph Conscience Integration
# ═════════════════════════════════════════════════════════════════════

class TestLangGraphIntegration:
    def test_get_conscience_context_exists(self):
        assert hasattr(lg, "_get_conscience_context")

    def test_conscience_context_is_async(self):
        import asyncio
        assert asyncio.iscoroutinefunction(lg._get_conscience_context)
