"""Tests for browser-agent/app.py.

All playwright calls are mocked — no real browser launched.

Run:
    PYTHONPATH=. python -m pytest scripts/test_browser_agent.py -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "browser_agent_app", ROOT / "browser-agent" / "app.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
app = _mod.app

from fastapi.testclient import TestClient

client = TestClient(app)


def _make_page_mock(title="Test Page", url="https://example.com", text="Hello world", status=200):
    page = AsyncMock()
    page.is_closed.return_value = False
    page.title = AsyncMock(return_value=title)
    page.url = url
    page.evaluate = AsyncMock(return_value=text)
    page.goto = AsyncMock(return_value=MagicMock(status=status))
    page.click = AsyncMock(return_value=None)
    page.fill = AsyncMock(return_value=None)
    page.screenshot = AsyncMock(return_value=b"\x89PNG\r\n")
    mock_el = AsyncMock()
    mock_el.click = AsyncMock(return_value=None)
    page.get_by_text = MagicMock(return_value=MagicMock(first=mock_el))
    return page


class TestHealth(unittest.TestCase):
    def test_health_returns_ok(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_metrics_returns_dict(self):
        r = client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), dict)


class TestNavigate(unittest.TestCase):
    def test_navigate_returns_title_and_url(self):
        page = _make_page_mock()
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/navigate", json={"url": "https://example.com"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["title"], "Test Page")
        self.assertEqual(data["url"], "https://example.com")

    def test_navigate_without_playwright_returns_503(self):
        with patch.object(_mod, "_PLAYWRIGHT_OK", False):
            r = client.post("/navigate", json={"url": "https://example.com"})
        self.assertEqual(r.status_code, 503)

    def test_navigate_failure_returns_502(self):
        page = _make_page_mock()
        page.goto = AsyncMock(side_effect=Exception("net::ERR_NAME_NOT_RESOLVED"))
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/navigate", json={"url": "https://does-not-exist.invalid"})
        self.assertEqual(r.status_code, 502)


class TestClick(unittest.TestCase):
    def test_click_by_selector(self):
        page = _make_page_mock()
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/click", json={"selector": "#submit"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_click_by_text(self):
        page = _make_page_mock()
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/click", json={"text": "Submit"})
        self.assertEqual(r.status_code, 200)

    def test_click_no_selector_or_text_returns_400(self):
        with patch.object(_mod, "_PLAYWRIGHT_OK", True):
            r = client.post("/click", json={})
        self.assertEqual(r.status_code, 400)


class TestType(unittest.TestCase):
    def test_type_fills_field(self):
        page = _make_page_mock()
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/type", json={"selector": "#search", "text": "hello"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])


class TestScrape(unittest.TestCase):
    def test_scrape_returns_title_and_text(self):
        page = _make_page_mock(title="Wiki", text="Wikipedia content here")
        page.evaluate = AsyncMock(side_effect=["Wikipedia content here", [{"text": "Link", "href": "https://wiki.example.com"}]])
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/scrape")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["title"], "Wiki")
        self.assertIn("Wikipedia", data["text"])


class TestScreenshot(unittest.TestCase):
    def test_screenshot_returns_png(self):
        page = _make_page_mock()
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/screenshot")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "image/png")


class TestRun(unittest.TestCase):
    def test_run_navigates_and_scrapes(self):
        page = _make_page_mock(title="Example", text="Some page text")
        page.evaluate = AsyncMock(return_value="Some page text")
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/run", json={"task": "find the main content", "url": "https://example.com"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("result", data)
        self.assertIn("steps", data)
        self.assertTrue(len(data["steps"]) >= 1)

    def test_run_without_url_scrapes_current_page(self):
        page = _make_page_mock(title="Current", text="Current page content")
        page.evaluate = AsyncMock(return_value="Current page content")
        with patch.object(_mod, "_PLAYWRIGHT_OK", True), \
             patch.object(_mod, "_get_page", AsyncMock(return_value=page)):
            r = client.post("/run", json={"task": "summarise"})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
