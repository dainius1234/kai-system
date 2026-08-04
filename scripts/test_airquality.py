"""Tests for airquality-service."""
import importlib.util
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from scripts.module_stubs import stubbed  # noqa: E402
from pathlib import Path
import types
import unittest

from fastapi.testclient import TestClient  # must be imported before httpx stub


_SAMPLE_AQ = {
    "hourly": {
        "pm2_5": [5.0, 8.0, 10.0],
        "pm10": [15.0, 20.0, 25.0],
        "ozone": [80.0, 90.0, 85.0],
        "nitrogen_dioxide": [20.0, 25.0, 22.0],
        "uv_index": [1.0, 3.0, 5.0],
    }
}

_SAMPLE_AQ_WITH_NONE = {
    "hourly": {
        "pm2_5": [None, None, 12.0],
        "pm10": [None, None, 20.0],
        "ozone": [None, None, None],
        "nitrogen_dioxide": [None],
        "uv_index": [None, 0.0],
    }
}


def _load_module():
    for key in list(sys.modules.keys()):
        if "airquality" in key:
            del sys.modules[key]

    real_httpx = sys.modules.get("httpx")
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:
        def __init__(self, **_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
        async def get(self, *_, **__): return _Response()

    class _Response:
        def raise_for_status(self): pass
        def json(self): return _SAMPLE_AQ

    if real_httpx:
        for attr in dir(real_httpx):
            if not attr.startswith("__"):
                setattr(httpx_stub, attr, getattr(real_httpx, attr))
    httpx_stub.AsyncClient = _AsyncClient

    runtime = types.ModuleType("common.runtime")
    runtime.setup_json_logger = lambda *_, **__: __import__("logging").getLogger("aq-test")
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
    stubs = {"common.runtime": runtime, "httpx": httpx_stub}
    if "common" not in sys.modules:
        _common = types.ModuleType("common")
        _common.__path__ = [str(Path(__file__).resolve().parents[1] / "common")]
        stubs["common"] = _common

    spec = importlib.util.spec_from_file_location(
        "airquality_service",
        str(Path(__file__).resolve().parents[1] / "airquality-service" / "app.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # `httpx` was already being put back by hand here; `common.runtime`
    # was not, and it stayed replaced for the rest of the session. Every
    # later `from common.runtime import detect_pii` then failed — 26 of
    # them in test_j_series alone, none of which is a bug in that file.
    with stubbed(stubs):
        spec.loader.exec_module(mod)

    return mod


class TestAqiCategory(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_good(self):
        self.assertEqual(self.mod._aqi_category(5.0), "good")

    def test_moderate(self):
        self.assertEqual(self.mod._aqi_category(20.0), "moderate")

    def test_sensitive(self):
        self.assertEqual(self.mod._aqi_category(40.0), "unhealthy for sensitive groups")

    def test_unhealthy(self):
        self.assertEqual(self.mod._aqi_category(100.0), "unhealthy")

    def test_very_unhealthy(self):
        self.assertEqual(self.mod._aqi_category(200.0), "very unhealthy")

    def test_hazardous(self):
        self.assertEqual(self.mod._aqi_category(300.0), "hazardous")

    def test_none_value(self):
        self.assertEqual(self.mod._aqi_category(None), "unknown")


class TestLatestHelper(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()

    def test_returns_last_non_none(self):
        val = self.mod._latest(_SAMPLE_AQ_WITH_NONE, "pm2_5")
        self.assertEqual(val, 12.0)

    def test_all_none_returns_none(self):
        val = self.mod._latest(_SAMPLE_AQ_WITH_NONE, "ozone")
        self.assertIsNone(val)

    def test_missing_key_returns_none(self):
        val = self.mod._latest(_SAMPLE_AQ, "nonexistent_key")
        self.assertIsNone(val)


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")
        self.assertIn("uptime_seconds", r.json())

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


class TestCurrentEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_current_no_cache(self):
        self.mod._cache = {}
        r = self.client.get("/current")
        self.assertEqual(r.status_code, 503)

    def test_current_with_cache(self):
        self.mod._cache = _SAMPLE_AQ
        r = self.client.get("/current")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("aqi_category", data)
        self.assertIn("pm2_5_ugm3", data)
        self.assertIn("uv_index", data)

    def test_current_aqi_category_good(self):
        self.mod._cache = _SAMPLE_AQ
        r = self.client.get("/current")
        self.assertEqual(r.json()["aqi_category"], "good")

    def test_current_location_included(self):
        self.mod._cache = _SAMPLE_AQ
        r = self.client.get("/current")
        self.assertIn("location", r.json())


class TestSummaryEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_summary_loading(self):
        self.mod._cache = {}
        r = self.client.get("/summary")
        self.assertIn("loading", r.json()["summary"])

    def test_summary_with_data(self):
        self.mod._cache = _SAMPLE_AQ
        r = self.client.get("/summary")
        s = r.json()["summary"]
        self.assertIn("Air quality", s)
        self.assertIn("PM2.5", s)

    def test_summary_uv_label_low(self):
        self.mod._cache = _SAMPLE_AQ
        r = self.client.get("/summary")
        self.assertIn("UV", r.json()["summary"])


if __name__ == "__main__":
    unittest.main()
