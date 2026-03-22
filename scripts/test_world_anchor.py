"""External World Anchor tests (Gap 4).

Tests date/time context, local news feed, events calendar,
and combined world context in calendar-sync/app.py.

Source: OpenClaw "world-anchor" skill pattern.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANCHOR_SRC = (ROOT / "calendar-sync" / "app.py").read_text()


# ── Date/Time Context ──────────────────────────────────────────────

class TestDateContext(unittest.TestCase):
    """Verify rich date/time context generation."""

    def test_function_defined(self):
        self.assertIn("def _date_context(", ANCHOR_SRC)

    def test_returns_time_of_day(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        self.assertIn("time_of_day", fn)

    def test_weekend_detection(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        self.assertIn("is_weekend", fn)

    def test_monday_detection(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        self.assertIn("is_monday", fn)

    def test_week_number(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        self.assertIn("week_number", fn)

    def test_suggestions_list(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        self.assertIn("suggestions", fn)

    def test_time_of_day_categories(self):
        fn = ANCHOR_SRC.split("def _date_context(")[1].split("\ndef ")[0]
        for period in ["morning", "afternoon", "evening", "night"]:
            self.assertIn(period, fn)


# ── Local News Feed ────────────────────────────────────────────────

class TestNewsFeed(unittest.TestCase):
    """Verify local file-based news system."""

    def test_recent_news_defined(self):
        self.assertIn("def _recent_news(", ANCHOR_SRC)

    def test_news_file_config(self):
        self.assertIn("NEWS_FILE", ANCHOR_SRC)

    def test_load_json_helper(self):
        self.assertIn("def _load_json(", ANCHOR_SRC)

    def test_save_json_helper(self):
        self.assertIn("def _save_json(", ANCHOR_SRC)

    def test_news_sorted_by_timestamp(self):
        fn = ANCHOR_SRC.split("def _recent_news(")[1].split("\ndef ")[0]
        self.assertIn("timestamp", fn)

    def test_news_has_limit(self):
        fn = ANCHOR_SRC.split("def _recent_news(")[1].split("\ndef ")[0]
        self.assertIn("limit", fn)


# ── Events Calendar ────────────────────────────────────────────────

class TestEventsCalendar(unittest.TestCase):
    """Verify local file-based events system."""

    def test_upcoming_events_defined(self):
        self.assertIn("def _upcoming_events(", ANCHOR_SRC)

    def test_events_file_config(self):
        self.assertIn("EVENTS_FILE", ANCHOR_SRC)

    def test_filters_by_date_range(self):
        fn = ANCHOR_SRC.split("def _upcoming_events(")[1].split("\ndef ")[0]
        self.assertIn("cutoff", fn)

    def test_sorts_events(self):
        fn = ANCHOR_SRC.split("def _upcoming_events(")[1].split("\ndef ")[0]
        self.assertIn("sort", fn)


# ── Combined World Context ─────────────────────────────────────────

class TestWorldContext(unittest.TestCase):
    """Verify combined world context snapshot."""

    def test_world_context_defined(self):
        self.assertIn("def _world_context(", ANCHOR_SRC)

    def test_includes_date(self):
        fn = ANCHOR_SRC.split("def _world_context(")[1].split("\ndef ")[0]
        self.assertIn("date", fn)

    def test_includes_news(self):
        fn = ANCHOR_SRC.split("def _world_context(")[1].split("\ndef ")[0]
        self.assertIn("news", fn)

    def test_includes_events(self):
        fn = ANCHOR_SRC.split("def _world_context(")[1].split("\ndef ")[0]
        self.assertIn("events", fn)


# ── Endpoints ───────────────────────────────────────────────────────

class TestAnchorEndpoints(unittest.TestCase):
    """Verify HTTP endpoints."""

    def test_health_endpoint(self):
        self.assertIn('"/health"', ANCHOR_SRC)

    def test_context_endpoint(self):
        self.assertIn('"/context"', ANCHOR_SRC)

    def test_date_endpoint(self):
        self.assertIn('"/date"', ANCHOR_SRC)

    def test_news_get_endpoint(self):
        self.assertIn('@app.get("/news")', ANCHOR_SRC)

    def test_news_post_endpoint(self):
        self.assertIn('@app.post("/news")', ANCHOR_SRC)

    def test_events_get_endpoint(self):
        self.assertIn('@app.get("/events")', ANCHOR_SRC)

    def test_events_post_endpoint(self):
        self.assertIn('@app.post("/events")', ANCHOR_SRC)


# ── Data Safety ─────────────────────────────────────────────────────

class TestDataSafety(unittest.TestCase):
    """Verify input bounds and file safety."""

    def test_news_title_truncation(self):
        """News title should be bounded."""
        fn = ANCHOR_SRC.split("async def add_news(")[1].split("\nasync def ")[0]
        self.assertIn("[:200]", fn)

    def test_events_limit(self):
        """Events list should be capped."""
        fn = ANCHOR_SRC.split("async def add_event(")[1].split("\n\n")[0]
        self.assertIn("500", fn)

    def test_seeds_empty_files(self):
        """Should seed empty JSON files if missing."""
        self.assertIn("NEWS_FILE.exists()", ANCHOR_SRC)
        self.assertIn("EVENTS_FILE.exists()", ANCHOR_SRC)


if __name__ == "__main__":
    unittest.main()
