"""Tests for calendar-service."""
import importlib.util
import os
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from scripts.module_stubs import stubbed  # noqa: E402
import types
import unittest
from datetime import date, timedelta

from fastapi.testclient import TestClient  # must be imported before any stubs


def _load_module(configured=False):
    _stubs = {}
    for key in list(sys.modules.keys()):
        if "calendar_service" in key:
            del sys.modules[key]

    # Stub caldav + icalendar so the module loads without real deps
    caldav_stub = types.ModuleType("caldav")
    caldav_stub.DAVClient = object
    _stubs["caldav"] = caldav_stub

    ical_stub = types.ModuleType("icalendar")
    ical_stub.Calendar = object
    _stubs["icalendar"] = ical_stub

    runtime = types.ModuleType("common.runtime")
    runtime.setup_json_logger = lambda *_, **__: __import__("logging").getLogger("cal-test")
    runtime.ErrorBudget = type("ErrorBudget", (), {
        "__init__": lambda self, **_: None,
        "record": lambda self, *a, **k: None,
        "snapshot": lambda self: {},
    })
    sys.modules.setdefault("common", types.ModuleType("common"))
    _stubs["common.runtime"] = runtime

    if configured:
        os.environ["CALDAV_URL"] = "https://caldav.example.com"
        os.environ["CALDAV_USER"] = "user"
        os.environ["CALDAV_PASS"] = "pass"
    else:
        os.environ.pop("CALDAV_URL", None)
        os.environ.pop("CALDAV_USER", None)
        os.environ.pop("CALDAV_PASS", None)

    spec = importlib.util.spec_from_file_location(
        "calendar_service",
        "/home/user/kai-system/calendar-service/app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    with stubbed(_stubs):
        spec.loader.exec_module(mod)
    return mod


def _make_event(start_offset_days=0, summary="Test Meeting"):
    today = date.today() + timedelta(days=start_offset_days)
    return {
        "uid": "test-uid-1",
        "summary": summary,
        "start": today.isoformat() + "T10:00:00+00:00",
        "end": today.isoformat() + "T11:00:00+00:00",
        "location": "Room A",
        "description": "",
        "calendar": "Personal",
    }


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module(configured=False)
        self.client = TestClient(self.mod.app)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("configured", data)
        self.assertFalse(data["configured"])

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


class TestStubMode(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module(configured=False)
        self.client = TestClient(self.mod.app)

    def test_events_today_empty_when_unconfigured(self):
        r = self.client.get("/events/today")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["events"], [])

    def test_events_upcoming_empty_when_unconfigured(self):
        r = self.client.get("/events/upcoming")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["events"], [])

    def test_summary_unconfigured(self):
        r = self.client.get("/summary")
        self.assertIn("not configured", r.json()["summary"])

    def test_refresh_503_when_unconfigured(self):
        r = self.client.post("/refresh")
        self.assertEqual(r.status_code, 503)


class TestEventFiltering(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module(configured=True)
        today_evt = _make_event(0, "Daily Standup")
        tomorrow_evt = _make_event(1, "Sprint Planning")
        next_week_evt = _make_event(8, "Quarterly Review")
        self.mod._events = [today_evt, tomorrow_evt, next_week_evt]
        self.client = TestClient(self.mod.app)

    def test_today_events_filter(self):
        r = self.client.get("/events/today")
        data = r.json()
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["summary"], "Daily Standup")

    def test_upcoming_7_days(self):
        r = self.client.get("/events/upcoming?days=7")
        data = r.json()
        self.assertEqual(len(data["events"]), 2)

    def test_upcoming_14_days(self):
        r = self.client.get("/events/upcoming?days=14")
        data = r.json()
        self.assertEqual(len(data["events"]), 3)

    def test_today_date_in_response(self):
        r = self.client.get("/events/today")
        self.assertEqual(r.json()["date"], date.today().isoformat())


class TestSummaryWithEvents(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module(configured=True)
        self.client = TestClient(self.mod.app)

    def test_summary_no_events(self):
        self.mod._events = []
        r = self.client.get("/summary")
        self.assertIn("No upcoming", r.json()["summary"])

    def test_summary_with_today_events(self):
        self.mod._events = [_make_event(0, "Morning Sync")]
        r = self.client.get("/summary")
        self.assertIn("Today", r.json()["summary"])
        self.assertIn("Morning Sync", r.json()["summary"])

    def test_summary_with_future_event(self):
        self.mod._events = [_make_event(3, "Client Call")]
        r = self.client.get("/summary")
        self.assertIn("Next", r.json()["summary"])


class TestDtToIso(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_date_to_iso(self):
        d = date(2026, 7, 24)
        self.assertEqual(self.mod._dt_to_iso(d), "2026-07-24")

    def test_string_fallback(self):
        self.assertEqual(self.mod._dt_to_iso("already-string"), "already-string")


if __name__ == "__main__":
    unittest.main()
