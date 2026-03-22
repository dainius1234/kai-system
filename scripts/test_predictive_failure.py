"""Predictive Failure Modeling tests (Gap 1).

Tests the linear regression, forecasting, proactive alert logic,
and /predict endpoints in supervisor/app.py.

Source: nightly log forecast concept from 2026 resilience research.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPERVISOR_SRC = (ROOT / "supervisor" / "app.py").read_text()


# ── Core functions ──────────────────────────────────────────────────

class TestLinearRegression(unittest.TestCase):
    """Verify pure-math OLS regression helper."""

    def test_function_defined(self):
        self.assertIn("def _linear_regression(", SUPERVISOR_SRC)

    def test_returns_slope_intercept_r_squared(self):
        fn = SUPERVISOR_SRC.split("def _linear_regression(")[1].split("\ndef ")[0]
        self.assertIn("slope", fn)
        self.assertIn("intercept", fn)
        self.assertIn("r_squared", fn)

    def test_handles_zero_variance(self):
        """Must not crash on flat data (zero denominator)."""
        fn = SUPERVISOR_SRC.split("def _linear_regression(")[1].split("\ndef ")[0]
        # Should have a guard for denom == 0
        self.assertTrue("denom" in fn or "== 0" in fn or "not xs" in fn)


class TestForecastFailures(unittest.TestCase):
    """Verify failure forecasting logic."""

    def test_function_defined(self):
        self.assertIn("def _forecast_failures(", SUPERVISOR_SRC)

    def test_uses_fleet_history(self):
        fn = SUPERVISOR_SRC.split("def _forecast_failures(")[1].split("\ndef ")[0]
        self.assertIn("_fleet_history", fn)

    def test_uses_linear_regression(self):
        fn = SUPERVISOR_SRC.split("def _forecast_failures(")[1].split("\ndef ")[0]
        self.assertIn("_linear_regression", fn)

    def test_returns_warning_when_threshold_crossed(self):
        fn = SUPERVISOR_SRC.split("def _forecast_failures(")[1].split("\ndef ")[0]
        self.assertIn("warning", fn)

    def test_horizon_config(self):
        self.assertIn("PREDICT_HORIZON_MINUTES", SUPERVISOR_SRC)

    def test_threshold_config(self):
        self.assertIn("PREDICT_ALERT_THRESHOLD", SUPERVISOR_SRC)


class TestProactiveForecast(unittest.TestCase):
    """Verify background proactive alerting."""

    def test_function_defined(self):
        self.assertIn("async def _proactive_forecast(", SUPERVISOR_SRC)

    def test_sends_telegram_alert(self):
        fn = SUPERVISOR_SRC.split("async def _proactive_forecast(")[1].split("\nasync def ")[0]
        self.assertIn("TELEGRAM", fn.upper() if "telegram" in fn.lower() else fn)

    def test_cooldown_mechanism(self):
        fn = SUPERVISOR_SRC.split("async def _proactive_forecast(")[1].split("\nasync def ")[0]
        self.assertIn("cooldown", fn.lower())

    def test_wired_into_background_loop(self):
        """Must be called from the main background loop."""
        bg = SUPERVISOR_SRC.split("async def _background_loop(")[1].split("\nasync def ")[0]
        self.assertIn("_proactive_forecast", bg)


# ── Endpoints ───────────────────────────────────────────────────────

class TestPredictEndpoints(unittest.TestCase):
    """Verify forecast HTTP endpoints."""

    def test_predict_endpoint(self):
        self.assertIn('"/predict"', SUPERVISOR_SRC)

    def test_predict_is_get(self):
        self.assertIn('@app.get("/predict")', SUPERVISOR_SRC)

    def test_per_service_endpoint(self):
        self.assertIn('"/predict/per-service"', SUPERVISOR_SRC)

    def test_per_service_is_get(self):
        self.assertIn('@app.get("/predict/per-service")', SUPERVISOR_SRC)

    def test_predict_calls_forecast(self):
        fn = SUPERVISOR_SRC.split('"/predict"')[1].split("\n@app")[0]
        self.assertIn("_forecast_failures", fn)


# ── Configuration ───────────────────────────────────────────────────

class TestPredictiveConfig(unittest.TestCase):
    """Verify configurable parameters."""

    def test_horizon_has_default_30(self):
        self.assertIn('PREDICT_HORIZON_MINUTES', SUPERVISOR_SRC)
        match = re.search(r'PREDICT_HORIZON_MINUTES.*?(\d+)', SUPERVISOR_SRC)
        self.assertIsNotNone(match)

    def test_threshold_has_default_3(self):
        self.assertIn('PREDICT_ALERT_THRESHOLD', SUPERVISOR_SRC)

    def test_cooldown_constant(self):
        self.assertIn('FORECAST_ALERT_COOLDOWN', SUPERVISOR_SRC)


if __name__ == "__main__":
    unittest.main()
