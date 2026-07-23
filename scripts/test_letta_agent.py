"""Unit tests for letta-agent FastAPI service (Letta client mocked)."""
from __future__ import annotations

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# ── Stub letta before importing app ──────────────────────────────────
# Build minimal stubs so importing letta-agent/app.py succeeds without
# the real package being installed in the test environment.
def _make_letta_stubs() -> None:
    if "letta" in sys.modules:
        return
    stub = types.ModuleType("letta")
    stub.create_client = MagicMock()
    schemas = types.ModuleType("letta.schemas")
    llm_cfg = types.ModuleType("letta.schemas.llm_config")
    llm_cfg.LLMConfig = MagicMock()
    emb_cfg = types.ModuleType("letta.schemas.embedding_config")
    emb_cfg.EmbeddingConfig = MagicMock()
    sys.modules["letta"] = stub
    sys.modules["letta.schemas"] = schemas
    sys.modules["letta.schemas.llm_config"] = llm_cfg
    sys.modules["letta.schemas.embedding_config"] = emb_cfg

_make_letta_stubs()

# Load letta-agent/app.py by path to avoid collision with sys.modules["app"]
# which test_p3_organic_memory.py sets to memu-core/app.py.
# Must register in sys.modules BEFORE exec so Pydantic forward-ref resolution
# (TypeAdapter) can find the module when rebuilding models at request time.
import importlib.util as _ilu  # noqa: E402

_letta_spec = _ilu.spec_from_file_location(
    "_letta_agent_app",
    os.path.join(os.path.dirname(__file__), "..", "letta-agent", "app.py"),
)
letta_app = _ilu.module_from_spec(_letta_spec)
sys.modules["_letta_agent_app"] = letta_app
_letta_spec.loader.exec_module(letta_app)

from fastapi.testclient import TestClient  # noqa: E402


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(letta_app.app)

    def test_health_ok(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "letta-agent")
        self.assertIn("model", data)

    def test_health_agent_id_initially_none(self):
        # _agent_id is None before any /agent/run call initialises the client
        resp = self.client.get("/health")
        self.assertIsNone(resp.json()["agent_id"])


class TestAgentRun(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(letta_app.app)
        # Build a mock Letta response
        msg = MagicMock()
        msg.assistant_message = "Test reply from Letta."
        msg.function_call = None
        mock_response = MagicMock()
        mock_response.messages = [msg]

        self.mock_lc = MagicMock()
        self.mock_lc.send_message.return_value = mock_response

    def test_agent_run_returns_response(self):
        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.post("/agent/run", json={"task": "What is 2+2?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["response"], "Test reply from Letta.")
        self.assertFalse(data["memories_updated"])
        self.assertEqual(data["agent_id"], "test-agent-id")

    def test_agent_run_with_context_prepends_context(self):
        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.post(
                "/agent/run",
                json={"task": "Summarise.", "context": {"user": "alice", "lang": "en"}},
            )
        self.assertEqual(resp.status_code, 200)
        # Verify the context was prepended into the message sent to Letta
        call_args = self.mock_lc.send_message.call_args
        msg_sent = call_args.kwargs.get("message") or call_args.args[0]
        self.assertIn("context:", msg_sent)
        self.assertIn("user=alice", msg_sent)

    def test_agent_run_memories_updated_on_archival_call(self):
        mem_msg = MagicMock()
        mem_msg.assistant_message = None
        mem_msg.text = None
        fn_mock = MagicMock()
        fn_mock.name = "archival_memory_insert"
        mem_msg.function_call = fn_mock
        mock_response = MagicMock()
        mock_response.messages = [mem_msg]
        self.mock_lc.send_message.return_value = mock_response

        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.post("/agent/run", json={"task": "Remember this."})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["memories_updated"])

    def test_agent_run_502_on_exception(self):
        self.mock_lc.send_message.side_effect = RuntimeError("Ollama down")
        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.post("/agent/run", json={"task": "hello"})
        self.assertEqual(resp.status_code, 502)
        self.assertIn("agent/run failed", resp.json()["detail"])


class TestMemoryExport(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(letta_app.app)
        p1, p2 = MagicMock(), MagicMock()
        p1.text = "Kai likes coffee."
        p2.text = "Project deadline is Friday."
        self.mock_lc = MagicMock()
        self.mock_lc.get_archival_memory.return_value = [p1, p2]

    def test_export_returns_memories_list(self):
        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.get("/agent/memory/export")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["count"], 2)
        self.assertIn("Kai likes coffee.", data["memories"])

    def test_export_502_on_exception(self):
        self.mock_lc.get_archival_memory.side_effect = RuntimeError("DB error")
        with patch.object(letta_app, "_letta_client", self.mock_lc), \
             patch.object(letta_app, "_agent_id", "test-agent-id"):
            resp = self.client.get("/agent/memory/export")
        self.assertEqual(resp.status_code, 502)


if __name__ == "__main__":
    unittest.main()
