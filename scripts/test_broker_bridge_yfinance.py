"""
yfinance stocks / forex endpoint tests — yfinance is stubbed in sys.modules
so no network calls are made and no real yfinance install is required.

Isolation pattern: each test class loads broker-bridge/app.py via importlib
with a fresh module name, so multiple loads don't collide.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient  # import BEFORE any stubs

# ── Sentinel (used to detect "key was absent") ────────────────────────────────

_SENTINEL = object()

# ── Fake fast_info ────────────────────────────────────────────────────────────


class _FakeFastInfo:
    """Attribute-access wrapper around a plain dict."""

    def __init__(self, data: dict) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str):
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(name)


_FAST_INFO = {
    "currency": "USD",
    "last_price": 195.89,
    "previous_close": 193.00,
    "day_high": 197.00,
    "day_low": 193.50,
    "fifty_two_week_high": 220.00,
    "fifty_two_week_low": 160.00,
    "three_month_average_volume": 55_000_000,
    "market_cap": 3_000_000_000_000,
    "exchange": "NMS",
}


def _make_yf_stub(data: dict) -> types.ModuleType:
    """Build a sys.modules-compatible yfinance stub."""
    stub = types.ModuleType("yfinance")
    _d = data

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.fast_info = _FakeFastInfo(_d)

    stub.Ticker = FakeTicker
    return stub


# ── Module loader ─────────────────────────────────────────────────────────────

_APP_PATH = Path(__file__).parent.parent / "broker-bridge" / "app.py"
_load_counter = 0


def _load(yf_stub: "types.ModuleType | None") -> object:
    """
    Load broker-bridge/app.py in isolation.

    If *yf_stub* is None, ``sys.modules["yfinance"]`` is set to None which
    causes Python to raise ImportError on ``import yfinance``, so the guard
    sets ``_YF_OK = False``.
    """
    global _load_counter
    _load_counter += 1
    module_name = f"_bb_yf_app_{_load_counter}"

    import os
    os.environ["BINANCE_API_KEY"] = "test-key"
    os.environ["BINANCE_API_SECRET"] = "test-secret"
    os.environ.setdefault("BINANCE_MODE", "spot")

    if yf_stub is None:
        # None in sys.modules blocks the import → ImportError → _YF_OK=False
        sys.modules["yfinance"] = None  # type: ignore[assignment]
    else:
        sys.modules["yfinance"] = yf_stub

    spec = importlib.util.spec_from_file_location(module_name, _APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Tests — Stocks ────────────────────────────────────────────────────────────

class TestStocksEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._yf_stub = _make_yf_stub(_FAST_INFO)
        cls.mod = _load(cls._yf_stub)
        cls.client = TestClient(cls.mod.app)

    # ------------------------------------------------------------------

    def test_stocks_returns_200(self) -> None:
        r = self.client.get("/stocks/AAPL")
        self.assertEqual(r.status_code, 200)

    def test_stocks_fields_present(self) -> None:
        r = self.client.get("/stocks/AAPL")
        data = r.json()
        for field in (
            "symbol", "currency", "last_price", "previous_close",
            "day_high", "day_low", "fifty_two_week_high", "fifty_two_week_low",
            "volume", "market_cap", "exchange",
        ):
            self.assertIn(field, data, f"Missing field: {field}")

    def test_stocks_symbol_uppercased(self) -> None:
        r = self.client.get("/stocks/aapl")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["symbol"], "AAPL")

    def test_stocks_yf_unavailable_503(self) -> None:
        saved = sys.modules.get("yfinance", _SENTINEL)
        try:
            mod = _load(None)
            client = TestClient(mod.app)
            r = client.get("/stocks/AAPL")
            self.assertEqual(r.status_code, 503)
        finally:
            if saved is _SENTINEL:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = saved  # type: ignore[assignment]


# ── Tests — Forex ─────────────────────────────────────────────────────────────

class TestForexEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._yf_stub = _make_yf_stub(_FAST_INFO)
        cls.mod = _load(cls._yf_stub)
        cls.client = TestClient(cls.mod.app)

    # ------------------------------------------------------------------

    def test_forex_returns_200(self) -> None:
        r = self.client.get("/forex/EURUSD")
        self.assertEqual(r.status_code, 200)

    def test_forex_fields_present(self) -> None:
        r = self.client.get("/forex/EURUSD")
        data = r.json()
        for field in ("pair", "symbol", "rate", "previous_close", "day_high", "day_low", "currency"):
            self.assertIn(field, data, f"Missing field: {field}")

    def test_forex_pair_normalized(self) -> None:
        """EURUSD should be normalised to pair='EURUSD', symbol='EURUSD=X'."""
        r = self.client.get("/forex/EURUSD")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["pair"], "EURUSD")
        self.assertEqual(data["symbol"], "EURUSD=X")

    def test_forex_yf_unavailable_503(self) -> None:
        saved = sys.modules.get("yfinance", _SENTINEL)
        try:
            mod = _load(None)
            client = TestClient(mod.app)
            r = client.get("/forex/EURUSD")
            self.assertEqual(r.status_code, 503)
        finally:
            if saved is _SENTINEL:
                sys.modules.pop("yfinance", None)
            else:
                sys.modules["yfinance"] = saved  # type: ignore[assignment]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main()
