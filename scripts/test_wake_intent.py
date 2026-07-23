from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))


def _load_wake_module():
    spec = importlib.util.spec_from_file_location("wake_app_test", ROOT / "perception" / "wake" / "app.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_agentic_module():
    import types

    sys.modules.setdefault("redis", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("agentic_app_wake_test", ROOT / "agentic" / "app.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wake_mod = _load_wake_module()
client = TestClient(wake_mod.app)


def test_detect_positive_simple_kai():
    wake_mod._last_wake_ts = 0.0
    detected, confidence, word = wake_mod.detect_wake_word("Kai, hello there")
    assert detected is True
    assert confidence >= 0.8
    assert word == "kai"


def test_detect_case_insensitive():
    wake_mod._last_wake_ts = 0.0
    detected, _, _ = wake_mod.detect_wake_word("hey KAI can you help")
    assert detected is True


def test_detect_negative_text():
    wake_mod._last_wake_ts = 0.0
    detected, confidence, word = wake_mod.detect_wake_word("good morning team")
    assert detected is False
    assert confidence == 0.0
    assert word is None


def test_detect_multiple_wake_words_phrase():
    wake_mod._last_wake_ts = 0.0
    wake_mod.WAKE_WORDS = ["kai", "hey kai", "ok kai"]
    detected, confidence, word = wake_mod.detect_wake_word("ok kai open status")
    assert detected is True
    assert confidence >= 0.9
    assert word == "ok kai"


def test_detect_partial_word_does_not_match():
    wake_mod._last_wake_ts = 0.0
    detected, _, _ = wake_mod.detect_wake_word("kaiser roll")
    assert detected is False


def test_detect_whitespace_normalization():
    wake_mod._last_wake_ts = 0.0
    detected, _, _ = wake_mod.detect_wake_word("  hey    kai   ")
    assert detected is True


def test_detect_cooldown_blocks_double_trigger():
    wake_mod._last_wake_ts = 0.0
    first = wake_mod.detect_wake_word("kai wake up")
    second = wake_mod.detect_wake_word("kai wake up")
    assert first[0] is True
    assert second[0] is False
    assert second[1] < 0.5


def test_detect_after_cooldown_allows_trigger():
    wake_mod._last_wake_ts = time.time() - wake_mod.WAKE_COOLDOWN_SECONDS - 0.1
    detected, _, _ = wake_mod.detect_wake_word("kai status")
    assert detected is True


def test_detect_empty_text_not_detected():
    wake_mod._last_wake_ts = 0.0
    detected, confidence, _ = wake_mod.detect_wake_word("")
    assert detected is False
    assert confidence == 0.0


def test_heuristic_command_label():
    result = wake_mod._heuristic_intent("Kai stop now")
    assert result["intent"] == "command"


def test_heuristic_question_label():
    result = wake_mod._heuristic_intent("What is my next reminder?")
    assert result["intent"] == "question"


def test_heuristic_task_label():
    result = wake_mod._heuristic_intent("Set a reminder for tomorrow morning")
    assert result["intent"] == "task"


def test_heuristic_emotional_label():
    result = wake_mod._heuristic_intent("I feel overwhelmed and stressed")
    assert result["intent"] == "emotional"


def test_heuristic_chat_label():
    result = wake_mod._heuristic_intent("hello mate")
    assert result["intent"] == "chat"


def test_heuristic_unknown_label():
    result = wake_mod._heuristic_intent("blue triangle satellite")
    assert result["intent"] == "unknown"


def test_validate_intent_payload_valid():
    payload = wake_mod._validate_intent_payload(
        json.dumps({"intent": "task", "confidence": 0.91, "reasoning": "Action verb found"})
    )
    assert payload is not None
    assert payload["intent"] == "task"


def test_validate_intent_payload_invalid_json():
    payload = wake_mod._validate_intent_payload("not-json")
    assert payload is None


def test_validate_intent_payload_invalid_intent():
    payload = wake_mod._validate_intent_payload(
        json.dumps({"intent": "other", "confidence": 0.4, "reasoning": "bad"})
    )
    assert payload is None


def test_validate_intent_payload_invalid_confidence_range():
    payload = wake_mod._validate_intent_payload(
        json.dumps({"intent": "task", "confidence": 1.4, "reasoning": "bad"})
    )
    assert payload is None


def test_validate_intent_payload_requires_reasoning():
    payload = wake_mod._validate_intent_payload(
        json.dumps({"intent": "task", "confidence": 0.4, "reasoning": ""})
    )
    assert payload is None


def test_classify_intent_fallback_when_llm_invalid(monkeypatch):
    class Dummy:
        text = "garbage"

    async def fake_query(*_args, **_kwargs):
        return Dummy()

    monkeypatch.setattr(wake_mod, "query_specialist", fake_query)
    result = asyncio.run(wake_mod.classify_intent("Set a reminder"))
    assert result["intent"] == "task"


def test_wake_detect_endpoint_text_smoke():
    wake_mod._last_wake_ts = 0.0
    resp = client.post("/wake/detect", json={"text": "Kai check status"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detected"] is True
    assert body["wake_word"] == "kai"
    assert "latency_ms" in body


def test_wake_detect_endpoint_negative_smoke():
    wake_mod._last_wake_ts = 0.0
    resp = client.post("/wake/detect", json={"text": "normal sentence only"})
    assert resp.status_code == 200
    assert resp.json()["detected"] is False


def test_wake_detect_endpoint_invalid_payload():
    resp = client.post("/wake/detect", json={})
    assert resp.status_code == 422


def test_wake_intent_endpoint_smoke():
    resp = client.post("/wake/intent", json={"text": "Can you tell me the weather?"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"intent", "confidence", "reasoning"}


def test_wake_process_endpoint_detects_and_classifies():
    wake_mod._last_wake_ts = 0.0
    resp = client.post("/wake/process", json={"text": "Kai, set a reminder"})
    assert resp.status_code == 200
    body = resp.json()
    assert "wake" in body and "intent" in body
    assert body["wake"]["detected"] is True


def test_health_endpoint_smoke():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "wake_words" in body


def test_langgraph_feature_flag_disabled(monkeypatch):
    lg_mod = _load_agentic_module()
    monkeypatch.setattr(lg_mod, "is_enabled", lambda _name: False)
    result = asyncio.run(lg_mod._preclassify_wake_intent("hello there"))
    assert result["reasoning"] == "feature_flag_disabled"


def test_langgraph_feature_flag_enabled_uses_service(monkeypatch):
    lg_mod = _load_agentic_module()
    monkeypatch.setattr(lg_mod, "is_enabled", lambda _name: True)

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"intent": "question", "confidence": 0.8, "reasoning": "question form"}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *_args, **_kwargs):
            return DummyResponse()

    monkeypatch.setattr(lg_mod.httpx, "AsyncClient", DummyClient)
    result = asyncio.run(lg_mod._preclassify_wake_intent("where are my tasks?"))
    assert result["intent"] == "question"
    assert result["confidence"] == 0.8


def test_langgraph_chat_route_override_from_wake_intent(monkeypatch):
    lg_mod = _load_agentic_module()
    lg_client = TestClient(lg_mod.app)
    decision_type = type(lg_mod.classify("hello"))

    async def fake_preclassify(_text: str):
        return {"intent": "command", "confidence": 0.95, "reasoning": "stop keyword"}

    def fake_classify(_text: str):
        return decision_type(
            route="GENERAL_CHAT",
            confidence=0.5,
            reason="fallback",
            bypass_llm=False,
            matched_keywords=[],
        )

    async def _empty_list(*_args, **_kwargs):
        return []

    async def _empty_dict(*_args, **_kwargs):
        return {}

    async def _noop(*_args, **_kwargs):
        return None

    async def fake_stream(*_args, **_kwargs):
        yield "ok"

    async def fake_get_mode():
        return "PUB"

    monkeypatch.setattr(lg_mod, "_preclassify_wake_intent", fake_preclassify)
    monkeypatch.setattr(lg_mod, "classify", fake_classify)
    monkeypatch.setattr(lg_mod, "_get_mode", fake_get_mode)
    monkeypatch.setattr(lg_mod, "_get_relevant_memories", _empty_list)
    monkeypatch.setattr(lg_mod, "_get_session_messages", _empty_list)
    monkeypatch.setattr(lg_mod, "_get_active_goals", _empty_list)
    monkeypatch.setattr(lg_mod, "_get_active_topics", _empty_list)
    monkeypatch.setattr(lg_mod, "_get_emotional_context", _empty_dict)
    monkeypatch.setattr(lg_mod, "_get_narrative_identity", _empty_dict)
    monkeypatch.setattr(lg_mod, "_get_imagination_context", _empty_dict)
    monkeypatch.setattr(lg_mod, "_get_conscience_context", _empty_dict)
    monkeypatch.setattr(lg_mod, "_get_agent_context", _empty_dict)
    monkeypatch.setattr(lg_mod, "_get_operator_model", _empty_dict)
    monkeypatch.setattr(lg_mod, "_append_session_turn", _noop)
    monkeypatch.setattr(lg_mod, "_auto_memorize", _noop)
    monkeypatch.setattr(lg_mod, "_trim_context", lambda messages, _budget: messages)
    monkeypatch.setattr(lg_mod._llm, "stream", fake_stream)

    response = lg_client.post("/chat", json={"message": "Kai stop now", "session_id": "s1", "mode": "PUB"})
    assert response.status_code == 200
    assert response.headers.get("X-Kai-Route") == "EXECUTE_ACTION"
    assert response.headers.get("X-Kai-Intent") == "command"
