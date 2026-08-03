"""Tests for weather-service."""
import importlib.util
import sys
from pathlib import Path
import types
import unittest

from fastapi.testclient import TestClient  # must be imported before httpx stub


_SAMPLE_WEATHER = {
    "current_weather": {
        "temperature": 18.5,
        "windspeed": 15.0,
        "winddirection": 270,
        "weathercode": 1,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-07-24", "2026-07-25"],
        "temperature_2m_max": [20.0, 22.0],
        "temperature_2m_min": [14.0, 16.0],
        "precipitation_sum": [0.0, 2.5],
        "precipitation_probability_max": [5, 40],
        "weathercode": [1, 61],
    },
}


def _load_module():
    for key in list(sys.modules.keys()):
        if "weather_service" in key:
            del sys.modules[key]

    httpx_stub = types.ModuleType("httpx")
    real_httpx = sys.modules.get("httpx")

    class _AsyncClient:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def get(self, *_, **__): return _Response()

    class _Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return _SAMPLE_WEATHER

    # Copy essential attributes from real httpx so TestClient still works
    if real_httpx:
        for attr in dir(real_httpx):
            if not attr.startswith("__"):
                setattr(httpx_stub, attr, getattr(real_httpx, attr))
    httpx_stub.AsyncClient = _AsyncClient

    sys.modules["httpx"] = httpx_stub

    runtime = types.ModuleType("common.runtime")
    runtime.setup_json_logger = lambda *_, **__: __import__("logging").getLogger("weather-test")
    runtime.ErrorBudget = type("ErrorBudget", (), {
        "__init__": lambda self, **_: None,
        "record": lambda self, *a, **k: None,
        "snapshot": lambda self: {},
    })
    # The `common` stub must remain a *package*: without `__path__`,
    # Python cannot resolve any `common.X` submodule, so every service
    # that imports a new shared module fails to load under this test —
    # which is how `common.http_hygiene` broke three suites at once.
    # Giving it the real package path means only what is explicitly
    # replaced below is stubbed; everything else resolves normally.
    _common = sys.modules.setdefault("common", types.ModuleType("common"))
    if not hasattr(_common, "__path__"):
        _common.__path__ = [str(Path(__file__).resolve().parents[1] / "common")]
    sys.modules["common.runtime"] = runtime

    spec = importlib.util.spec_from_file_location(
        "weather_service",
        "/home/user/kai-system/weather-service/app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Restore real httpx for TestClient
    if real_httpx:
        sys.modules["httpx"] = real_httpx

    return mod


class TestWMOCodes(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_known_code(self):
        self.assertEqual(self.mod._wmo_desc(0), "clear sky")
        self.assertEqual(self.mod._wmo_desc(61), "light rain")
        self.assertEqual(self.mod._wmo_desc(95), "thunderstorm")

    def test_unknown_code(self):
        self.assertIn("99999", self.mod._wmo_desc(99999))

    def test_wmo_dict_not_empty(self):
        self.assertGreater(len(self.mod._WMO_CODES), 10)


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("location", data)
        self.assertIn("uptime_seconds", data)

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


class TestCurrentEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.mod._cache = _SAMPLE_WEATHER
        self.mod._last_fetch = 1234567890.0
        self.client = TestClient(self.mod.app)

    def test_current_with_cache(self):
        r = self.client.get("/current")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("temp_c", data)
        self.assertIn("description", data)
        self.assertIn("weathercode", data)

    def test_current_without_cache(self):
        self.mod._cache = {}
        r = self.client.get("/current")
        self.assertEqual(r.status_code, 503)

    def test_current_description_matches_code(self):
        r = self.client.get("/current")
        data = r.json()
        self.assertEqual(data["description"], "mainly clear")

    def test_current_is_day_bool(self):
        r = self.client.get("/current")
        data = r.json()
        self.assertIsInstance(data["is_day"], bool)


class TestForecastEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.mod._cache = _SAMPLE_WEATHER
        self.client = TestClient(self.mod.app)

    def test_forecast_with_cache(self):
        r = self.client.get("/forecast")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("forecast", data)
        self.assertIsInstance(data["forecast"], list)
        self.assertEqual(len(data["forecast"]), 2)

    def test_forecast_without_cache(self):
        self.mod._cache = {}
        r = self.client.get("/forecast")
        self.assertEqual(r.status_code, 503)

    def test_forecast_fields(self):
        r = self.client.get("/forecast")
        day0 = r.json()["forecast"][0]
        for field in ("date", "temp_max_c", "temp_min_c", "precipitation_mm", "weathercode", "description"):
            self.assertIn(field, day0)

    def test_forecast_rain_prob(self):
        r = self.client.get("/forecast")
        day1 = r.json()["forecast"][1]
        self.assertEqual(day1["rain_prob_pct"], 40)


class TestSummaryEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_summary_loading_state(self):
        self.mod._cache = {}
        r = self.client.get("/summary")
        self.assertEqual(r.status_code, 200)
        self.assertIn("loading", r.json()["summary"])

    def test_summary_with_data(self):
        self.mod._cache = _SAMPLE_WEATHER
        r = self.client.get("/summary")
        data = r.json()
        self.assertIn("summary", data)
        self.assertIsInstance(data["summary"], str)
        self.assertGreater(len(data["summary"]), 5)

    def test_summary_includes_rain_chance(self):
        self.mod._cache = _SAMPLE_WEATHER
        r = self.client.get("/summary")
        self.assertIn("rain", r.json()["summary"])


if __name__ == "__main__":
    unittest.main()
