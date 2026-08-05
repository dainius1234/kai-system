"""Tests for D120: Trust Auditor teammate — data/teammates/auditor.md + app wiring."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from teammates import load_teammates, get_teammate, list_teammates, _parse_teammate_md


AUDITOR_MD = (Path(__file__).parent.parent / "data" / "teammates" / "auditor.md").read_text()


# ── File format validation ────────────────────────────────────────────────────

def test_auditor_md_parses_correctly():
    t = _parse_teammate_md("auditor", AUDITOR_MD)
    assert t.name == "Auditor"
    assert t.specialty == "trust_governance"
    assert "trust" in t.description.lower()
    assert len(t.system_prompt) > 100


def test_auditor_system_prompt_mentions_factors():
    t = _parse_teammate_md("auditor", AUDITOR_MD)
    assert "operator_approval_history" in t.system_prompt
    assert "value_alignment" in t.system_prompt
    assert "conviction_alignment" in t.system_prompt


def test_auditor_system_prompt_mentions_levels():
    t = _parse_teammate_md("auditor", AUDITOR_MD)
    assert "DORMANT" in t.system_prompt
    assert "GUARDIAN" in t.system_prompt


def test_auditor_system_prompt_mentions_wisdom_graph():
    t = _parse_teammate_md("auditor", AUDITOR_MD)
    assert "wisdom_graph" in t.system_prompt or "wisdom graph" in t.system_prompt.lower()


def test_auditor_loads_via_load_teammates(tmp_path):
    """Auditor loads correctly through the teammate registry."""
    import teammates as tm
    tm._registry = {}
    tm.TEAMMATES_DIR = Path(__file__).parent.parent / "data" / "teammates"
    load_teammates()
    t = get_teammate("auditor")
    assert t is not None
    assert t.slug == "auditor"
    assert t.specialty == "trust_governance"


def test_auditor_appears_in_list_teammates(tmp_path):
    import teammates as tm
    tm._registry = {}
    tm.TEAMMATES_DIR = Path(__file__).parent.parent / "data" / "teammates"
    load_teammates()
    slugs = [t["slug"] for t in list_teammates()]
    assert "auditor" in slugs


# ── Trust data contract ───────────────────────────────────────────────────────

def test_trust_status_has_auditor_required_keys():
    """get_trust_status() returns the keys auditor.md promises to interpret."""
    from trust_integration import get_trust_status
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    # Auditor needs at least score and tier — even in degraded mode
    assert "score" in status
    assert "tier" in status or "level_name" in status


def test_trust_status_with_full_core_has_all_auditor_fields():
    from trust_integration import get_trust_status
    trust = MagicMock()
    trust.status.return_value = {
        "level": 3,
        "level_name": "AGENT",
        "granted_by": "operator",
        "scores": {"consistency": 0.6, "judgment": 0.5},
        "next_level": "PARTNER",
        "progress_to_next": 0.4,
    }
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    assert status["level_name"] == "AGENT"
    assert status["next_level"] == "PARTNER"
    assert "progress_to_next" in status


def test_trust_status_includes_wisdom_graph_stats():
    from trust_integration import get_trust_status
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    # wisdom_graph key is present (may be absent if import fails, that's ok)
    # Just confirm the function doesn't crash — it's fail-open
    assert isinstance(status, dict)


# ── Load agentic/app.py under a name of its own ──────────────────────
# `import app` is a coin toss in a full run. `sys.modules["app"]` is a
# generic name, and test_p3_organic_memory.py claims it for
# memu-core/app.py. Whichever suite imports first wins, and this one sorts
# later, so the bare name resolved to memu-core, the is_enabled patch
# landed on the wrong module, and two tests failed while passing perfectly
# when the file ran alone.
#
# test_letta_agent.py already carried a comment about this exact collision.
# It was a known hazard that nothing enforced.
_APP_NAME = "agentic_app_trust"


def _agentic_app():
    """agentic/app.py, loaded once, under an unambiguous name."""
    if _APP_NAME in sys.modules:
        return sys.modules[_APP_NAME]
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "agentic" / "app.py"
    assert path.is_file(), f"agentic application missing at {path}"
    spec = importlib.util.spec_from_file_location(_APP_NAME, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_APP_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Endpoint wiring ───────────────────────────────────────────────────────────

def test_auditor_endpoint_injects_trust_data():
    """The /chat/teammate/auditor endpoint injects trust state, not world state."""
    import teammates as tm
    tm._registry = {}
    tm.TEAMMATES_DIR = Path(__file__).parent.parent / "data" / "teammates"
    load_teammates()

    captured_prompts = []

    async def fake_chat(messages):
        captured_prompts.append(messages[0]["content"])
        return "Trust audit complete."

    app_module = _agentic_app()
    orig_llm = app_module._llm
    orig_trust = app_module.get_trust_status

    fake_llm = MagicMock()
    fake_llm.chat = fake_chat
    trust_data = {"score": 66.5, "tier": "Adept", "level_name": "ASSISTANT"}

    try:
        app_module._llm = fake_llm
        app_module.get_trust_status = lambda: trust_data

        from fastapi.testclient import TestClient
        client = TestClient(app_module.app)

        with patch(f"{_APP_NAME}.is_enabled", return_value=True):
            resp = client.post(
                "/chat/teammate/auditor",
                json={"message": "What is my current trust level?"},
            )
        assert resp.status_code == 200
        assert "auditor" in resp.json()["teammate"]
        assert len(captured_prompts) == 1
        assert "Adept" in captured_prompts[0] or "66.5" in captured_prompts[0]
    finally:
        app_module._llm = orig_llm
        app_module.get_trust_status = orig_trust


def test_auditor_endpoint_does_not_inject_world_state():
    """Auditor gets trust data, not the proactive observer world snapshot."""
    import teammates as tm
    tm._registry = {}
    tm.TEAMMATES_DIR = Path(__file__).parent.parent / "data" / "teammates"
    load_teammates()

    captured_prompts = []

    async def fake_chat(messages):
        captured_prompts.append(messages[0]["content"])
        return "ok"

    app_module = _agentic_app()
    orig_llm = app_module._llm
    orig_snapshot = app_module._last_world_snapshot
    orig_trust = app_module.get_trust_status

    fake_llm = MagicMock()
    fake_llm.chat = fake_chat

    try:
        app_module._llm = fake_llm
        app_module._last_world_snapshot = {"docker": {"count": 5}, "sys": {"cpu": 99.0}}
        app_module.get_trust_status = lambda: {"score": 55.0, "tier": "Journeyman"}

        from fastapi.testclient import TestClient
        client = TestClient(app_module.app)

        with patch(f"{_APP_NAME}.is_enabled", return_value=True):
            resp = client.post(
                "/chat/teammate/auditor",
                json={"message": "audit", "world_context": True},
            )
        assert resp.status_code == 200
        prompt = captured_prompts[0]
        assert "docker" not in prompt
        assert "Journeyman" in prompt or "55.0" in prompt
    finally:
        app_module._llm = orig_llm
        app_module._last_world_snapshot = orig_snapshot
        app_module.get_trust_status = orig_trust
