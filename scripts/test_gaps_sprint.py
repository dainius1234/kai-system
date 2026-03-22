"""Tests for gaps sprint: JSON logging, vector cleanup, ledger stats."""
import json
import logging
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch, MagicMock

# ── JSON Logging Tests ────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestJSONLogger(unittest.TestCase):
    """Test setup_json_logger produces JSON on both file and stdout."""

    def test_logger_has_two_handlers(self):
        from common.runtime import setup_json_logger
        logger = setup_json_logger("test-svc", "/tmp/test-svc.json.log")
        self.assertEqual(len(logger.handlers), 2)
        handler_types = {type(h).__name__ for h in logger.handlers}
        self.assertIn("TimedRotatingFileHandler", handler_types)
        self.assertIn("StreamHandler", handler_types)

    def test_stdout_handler_writes_json(self):
        from common.runtime import setup_json_logger
        import io
        buf = io.StringIO()
        logger = setup_json_logger("json-test", "/tmp/json-test.json.log")
        # replace stdout handler's stream with our buffer
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not hasattr(h, 'baseFilename'):
                h.stream = buf
        logger.info("hello world")
        output = buf.getvalue().strip()
        parsed = json.loads(output)
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["service"], "json-test")
        self.assertIn("hello world", parsed["msg"])

    def test_log_format_has_required_fields(self):
        from common.runtime import setup_json_logger
        import io
        buf = io.StringIO()
        logger = setup_json_logger("fields-test", "/tmp/fields-test.json.log")
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not hasattr(h, 'baseFilename'):
                h.stream = buf
        logger.warning("test message")
        parsed = json.loads(buf.getvalue().strip())
        for key in ("time", "level", "service", "msg"):
            self.assertIn(key, parsed, f"Missing key: {key}")


# ── Vector Cleanup Tests ─────────────────────────────────────────────

class TestVectorCleanup(unittest.TestCase):
    """Test InMemoryVectorStore.delete_old."""

    def _make_store(self):
        """Create an InMemoryVectorStore with test data."""
        # import here to avoid import-time side effects
        from memu_core_app import InMemoryVectorStore, MemoryRecord
        store = InMemoryVectorStore()
        now = datetime.now(tz=timezone.utc)
        # old record: 120 days ago
        store._records.append(MemoryRecord(
            id="old-1", timestamp=(now - timedelta(days=120)).isoformat(),
            event_type="observation", content={"text": "old memory"},
            embedding=[0.1] * 10, relevance=0.5, pinned=False,
        ))
        # recent record: 10 days ago
        store._records.append(MemoryRecord(
            id="recent-1", timestamp=(now - timedelta(days=10)).isoformat(),
            event_type="observation", content={"text": "recent memory"},
            embedding=[0.2] * 10, relevance=0.5, pinned=False,
        ))
        # pinned old record: 120 days ago but pinned
        store._records.append(MemoryRecord(
            id="pinned-old", timestamp=(now - timedelta(days=120)).isoformat(),
            event_type="observation", content={"text": "pinned old"},
            embedding=[0.3] * 10, relevance=0.5, pinned=True,
        ))
        return store

    def test_cleanup_deletes_old_unpinned(self):
        store = self._make_store()
        result = store.delete_old(max_age_days=90)
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["after"], 2)
        remaining_ids = {r.id for r in store._records}
        self.assertIn("recent-1", remaining_ids)
        self.assertIn("pinned-old", remaining_ids)
        self.assertNotIn("old-1", remaining_ids)

    def test_cleanup_preserves_pinned(self):
        store = self._make_store()
        result = store.delete_old(max_age_days=1)
        remaining_ids = {r.id for r in store._records}
        self.assertIn("pinned-old", remaining_ids)

    def test_cleanup_with_large_age_deletes_nothing(self):
        store = self._make_store()
        result = store.delete_old(max_age_days=365)
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(result["before"], 3)

    def test_cleanup_returns_cutoff_iso(self):
        store = self._make_store()
        result = store.delete_old(max_age_days=90)
        self.assertIn("cutoff", result)
        # verify it's a valid ISO string
        datetime.fromisoformat(result["cutoff"])


# ── Cleanup Endpoint Tests ────────────────────────────────────────────

class TestCleanupEndpoint(unittest.TestCase):
    """Test /memory/cleanup endpoint logic."""

    def test_cleanup_endpoint_rejects_zero_days(self):
        """max_age_days < 1 should return error."""
        import importlib
        mod = importlib.import_module("memu_core_app")
        import asyncio
        result = asyncio.run(
            mod.memory_cleanup(max_age_days=0)
        )
        self.assertEqual(result["status"], "error")

    def test_cleanup_endpoint_succeeds(self):
        """Normal cleanup should return status ok."""
        import importlib
        mod = importlib.import_module("memu_core_app")
        import asyncio
        result = asyncio.run(
            mod.memory_cleanup(max_age_days=90)
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("deleted", result)


# ── Dashboard Ledger Stats Tests ──────────────────────────────────────

class TestLedgerStatsEndpoint(unittest.TestCase):
    """Test /api/ledger-stats proxy endpoint."""

    def test_ledger_stats_returns_unavailable_on_error(self):
        """When ledger-worker is unreachable, should return unavailable."""
        import importlib
        dashboard = importlib.import_module("dashboard_app")
        import asyncio

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get.side_effect = Exception("connection refused")
            mock_client_cls.return_value = mock_client

            result = asyncio.run(
                dashboard.api_ledger_stats()
            )
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["total_entries"], 0)


if __name__ == "__main__":
    # Setup import aliases so modules can be imported without package structure
    # memu-core/app.py → memu_core_app
    memu_path = os.path.join(os.path.dirname(__file__), "..", "memu-core")
    if memu_path not in sys.path:
        sys.path.insert(0, memu_path)
    # create module alias
    import importlib.util
    spec = importlib.util.spec_from_file_location("memu_core_app",
                                                    os.path.join(memu_path, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["memu_core_app"] = mod
    spec.loader.exec_module(mod)

    # dashboard/app.py → dashboard_app
    dash_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
    if dash_path not in sys.path:
        sys.path.insert(0, dash_path)
    spec2 = importlib.util.spec_from_file_location("dashboard_app",
                                                     os.path.join(dash_path, "app.py"))
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["dashboard_app"] = mod2
    spec2.loader.exec_module(mod2)

    unittest.main()
