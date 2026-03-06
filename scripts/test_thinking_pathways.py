"""P8: Thinking Pathways Dashboard — tests for new API proxy endpoints."""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

# ── Import dashboard module ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "dashboard" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# Prevent tests from hitting real services
mod.NODES = {}

from fastapi.testclient import TestClient

client = TestClient(mod.app)


# ── Thinking Page Tests ──────────────────────────────────────────────

class TestThinkingPage(unittest.TestCase):
    """Test the thinking pathways content (now served via /app unified shell)."""

    def test_thinking_redirect(self):
        resp = client.get("/thinking", follow_redirects=False)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("/app", resp.text)

    def test_app_shell_has_thinking_content(self):
        resp = client.get("/app")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Conviction Pipeline", resp.text)
        self.assertIn("Operator Tempo", resp.text)
        self.assertIn("Knowledge Boundary", resp.text)
        self.assertIn("Silence-as-Signal", resp.text)
        self.assertIn("Temporal Self-Assessment", resp.text)

    def test_app_shell_has_thinking_api_endpoints(self):
        resp = client.get("/app")
        text = resp.text
        self.assertIn("/api/thinking", text)
        self.assertIn("/api/tempo", text)
        self.assertIn("/api/boundary", text)
        self.assertIn("/api/silence", text)
        self.assertIn("/api/self-assessment", text)

    def test_app_shell_has_navigation(self):
        resp = client.get("/app")
        text = resp.text
        self.assertIn('data-view="chat"', text)
        self.assertIn('data-view="dashboard"', text)
        self.assertIn('data-view="thinking"', text)

    def test_app_shell_has_dream_and_audit(self):
        resp = client.get("/app")
        self.assertIn("triggerDream()", resp.text)
        self.assertIn("runSecurityAudit()", resp.text)


# ── API /api/thinking Tests ──────────────────────────────────────────

class TestApiThinking(unittest.TestCase):
    """Test the /api/thinking endpoint."""

    @patch("dashboard_app.httpx.AsyncClient")
    def test_thinking_returns_episodes(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "count": 2,
            "episodes": [
                {
                    "episode_id": "ep1",
                    "input": "test question",
                    "output": "test answer",
                    "conviction_score": 8.5,
                    "final_conviction": 9.0,
                    "rethink_count": 0,
                    "learning_value": 0.3,
                    "ts": 1700000000,
                },
                {
                    "episode_id": "ep2",
                    "input": "another one",
                    "output": "another answer",
                    "conviction_score": 6.0,
                    "final_conviction": 7.5,
                    "rethink_count": 1,
                    "failure_class": "LOW_EVIDENCE",
                    "metacognitive_rule": "check sources first",
                    "learning_value": 0.8,
                    "ts": 1700001000,
                },
            ],
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/thinking")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["total_episodes"], 2)
        self.assertEqual(len(data["pathways"]), 2)
        self.assertEqual(data["pathways"][0]["conviction_score"], 8.5)
        self.assertEqual(data["pathways"][1]["failure_class"], "LOW_EVIDENCE")
        self.assertEqual(data["pathways"][1]["metacognitive_rule"], "check sources first")

    def test_thinking_returns_unavailable_on_error(self):
        resp = client.get("/api/thinking")
        data = resp.json()
        self.assertIn(data["status"], ["ok", "unavailable"])

    @patch("dashboard_app.httpx.AsyncClient")
    def test_thinking_truncates_long_input(self, mock_client_cls):
        long_input = "x" * 500
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "count": 1,
            "episodes": [{"episode_id": "ep1", "input": long_input, "output": "short", "conviction_score": 5.0, "ts": 0}],
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/thinking")
        data = resp.json()
        self.assertLessEqual(len(data["pathways"][0]["input"]), 200)

    @patch("dashboard_app.httpx.AsyncClient")
    def test_thinking_limits_to_10_episodes(self, mock_client_cls):
        eps = [{"episode_id": f"ep{i}", "input": f"q{i}", "output": f"a{i}", "conviction_score": 5.0, "ts": i} for i in range(20)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"count": 20, "episodes": eps}
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.post = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/thinking")
        data = resp.json()
        self.assertLessEqual(len(data["pathways"]), 10)


# ── API /api/tempo Tests ─────────────────────────────────────────────

class TestApiTempo(unittest.TestCase):
    """Test the /api/tempo proxy endpoint."""

    @patch("dashboard_app.httpx.AsyncClient")
    def test_tempo_proxies_data(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "tempo": "normal",
            "style_hint": "conversational pace",
            "distribution": {"rapid": 2, "normal": 5, "reflective": 1},
            "avg_gap_seconds": 120.5,
            "burst_count": 1,
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/tempo")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["tempo"], "normal")
        self.assertIn("style_hint", data)

    def test_tempo_returns_unavailable_on_error(self):
        resp = client.get("/api/tempo")
        data = resp.json()
        # Either proxied or fallback
        self.assertIn("tempo", data)


# ── API /api/boundary Tests ──────────────────────────────────────────

class TestApiBoundary(unittest.TestCase):
    """Test the /api/boundary proxy endpoint."""

    @patch("dashboard_app.httpx.AsyncClient")
    def test_boundary_proxies_zones(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "zones": [
                {"topic": "python", "confidence": 0.9},
                {"topic": "quantum physics", "confidence": 0.2},
            ],
            "overall_confidence": 0.55,
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/boundary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["zones"]), 2)
        self.assertEqual(data["zones"][0]["topic"], "python")

    def test_boundary_returns_fallback_on_error(self):
        resp = client.get("/api/boundary")
        data = resp.json()
        self.assertIn("zones", data)


# ── API /api/silence Tests ───────────────────────────────────────────

class TestApiSilence(unittest.TestCase):
    """Test the /api/silence proxy endpoint."""

    @patch("dashboard_app.httpx.AsyncClient")
    def test_silence_proxies_data(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "silence_topics": [{"topic": "classified", "reason": "no evidence"}],
            "silence_count": 3,
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/silence")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["silence_count"], 3)

    def test_silence_returns_fallback_on_error(self):
        resp = client.get("/api/silence")
        data = resp.json()
        self.assertIn("silence_topics", data)


# ── API /api/self-assessment Tests ───────────────────────────────────

class TestApiSelfAssessment(unittest.TestCase):
    """Test the /api/self-assessment proxy endpoint."""

    @patch("dashboard_app.httpx.AsyncClient")
    def test_assessment_proxies_data(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "status": "ok",
            "current": {
                "total_memories": 150,
                "error_rate": 0.02,
                "uptime_ratio": 0.98,
                "avg_response_ms": 320,
                "episodes_count": 45,
                "avg_conviction": 7.8,
            },
            "deltas": {"total_memories": 12, "error_rate": -0.01, "avg_conviction": 0.3},
            "trend": "improving",
            "narrative": "System is getting better",
        }
        mock_instance = AsyncMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_instance.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value = mock_instance

        resp = client.get("/api/self-assessment")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["trend"], "improving")
        self.assertEqual(data["current"]["total_memories"], 150)

    def test_assessment_returns_unavailable_on_error(self):
        resp = client.get("/api/self-assessment")
        data = resp.json()
        self.assertEqual(data["status"], "unavailable")


# ── Integration: all dashboard pages exist ───────────────────────────

class TestDashboardPages(unittest.TestCase):
    """Ensure all dashboard pages return 200."""

    def test_health(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_thinking_page(self):
        resp = client.get("/thinking")
        self.assertEqual(resp.status_code, 200)

    def test_chat_page(self):
        resp = client.get("/chat")
        self.assertEqual(resp.status_code, 200)

    def test_ui_page(self):
        resp = client.get("/ui")
        self.assertEqual(resp.status_code, 200)

    def test_static_thinking_html(self):
        resp = client.get("/static/thinking.html")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Thinking Pathways", resp.text)


if __name__ == "__main__":
    unittest.main()
