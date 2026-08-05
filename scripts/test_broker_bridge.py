"""
Broker-bridge service tests — all Binance HTTP calls are patched at the
_public_get / _signed_get module-level functions so no real network requests
are made.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Module loader ─────────────────────────────────────────────────────────────

def _load_module(monkeypatch):
    """Load broker-bridge/app.py with httpx stubbed and env vars neutral."""
    stub_httpx = types.ModuleType("httpx")
    stub_httpx.AsyncClient = object
    stub_httpx.HTTPStatusError = Exception
    stub_httpx.RequestError = Exception
    sys.modules.setdefault("httpx", stub_httpx)

    spec = importlib.util.spec_from_file_location(
        "broker_bridge_app",
        Path(__file__).parent.parent / "broker-bridge" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    monkeypatch.setenv("BINANCE_MODE", "spot")
    spec.loader.exec_module(mod)
    return mod


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def spot_client(monkeypatch):
    mod = _load_module(monkeypatch)
    return TestClient(mod.app), mod


@pytest.fixture()
def futures_client(monkeypatch):
    mod = _load_module(monkeypatch)
    mod.MODE = "futures"
    return TestClient(mod.app), mod


@pytest.fixture()
def no_key_client(monkeypatch):
    mod = _load_module(monkeypatch)
    mod.API_KEY = ""
    mod.API_SECRET = ""
    return TestClient(mod.app), mod


# ── Health / metrics ──────────────────────────────────────────────────────────

def test_health_spot(spot_client):
    client, _ = spot_client
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["mode"] == "spot"
    assert data["api_configured"] is True


def test_health_no_key(no_key_client):
    client, _ = no_key_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["api_configured"] is False


def test_metrics(spot_client):
    client, _ = spot_client
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "uptime_seconds" in data
    assert "requests" in data
    assert "errors" in data


# ── Signing ───────────────────────────────────────────────────────────────────

def test_sign_params_deterministic(spot_client):
    _, mod = spot_client
    result = mod._sign_params({"symbol": "BTCUSDT", "timestamp": 1000}, "secret")
    assert "signature" in result
    # deterministic: same inputs → same sig
    result2 = mod._sign_params({"symbol": "BTCUSDT", "timestamp": 1000}, "secret")
    assert result["signature"] == result2["signature"]


def test_sign_params_changes_with_secret(spot_client):
    _, mod = spot_client
    r1 = mod._sign_params({"x": 1}, "secret1")
    r2 = mod._sign_params({"x": 1}, "secret2")
    assert r1["signature"] != r2["signature"]


def test_sign_params_preserves_original(spot_client):
    _, mod = spot_client
    original = {"symbol": "ETH", "timestamp": 99}
    signed = mod._sign_params(original, "s")
    assert signed["symbol"] == "ETH"
    assert "signature" in signed
    assert "signature" not in original   # original not mutated


# ── Ticker ────────────────────────────────────────────────────────────────────

def test_ticker_symbol(spot_client):
    client, mod = spot_client
    with patch.object(mod, "_public_get", new=AsyncMock(
            return_value={"symbol": "BTCUSDT", "price": "95000.00"})):
        r = client.get("/ticker/btcusdt")
    assert r.status_code == 200
    assert r.json()["symbol"] == "BTCUSDT"
    assert r.json()["price"] == 95000.0


def test_ticker_all_default(spot_client):
    client, mod = spot_client
    with patch.object(mod, "_public_get", new=AsyncMock(
            return_value={"symbol": "BTCUSDT", "price": "95000.00"})):
        r = client.get("/ticker")
    assert r.status_code == 200
    assert isinstance(r.json()["tickers"], list)


def test_ticker_custom_symbols(spot_client):
    client, mod = spot_client
    with patch.object(mod, "_public_get", new=AsyncMock(
            return_value={"symbol": "ETHUSDT", "price": "3500.00"})):
        r = client.get("/ticker?symbols=ETHUSDT")
    assert r.status_code == 200
    assert r.json()["tickers"][0]["symbol"] == "ETHUSDT"


# ── Balance ───────────────────────────────────────────────────────────────────

def test_balance_spot(spot_client):
    client, mod = spot_client
    mock_account = {
        "balances": [
            {"asset": "BTC", "free": "0.5", "locked": "0.0"},
            {"asset": "USDT", "free": "1000.0", "locked": "0.0"},
            {"asset": "XRP", "free": "0.0", "locked": "0.0"},  # zero — filtered
        ]
    }
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_account)):
        r = client.get("/balance")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "spot"
    syms = [a["asset"] for a in data["assets"]]
    assert "BTC" in syms
    assert "USDT" in syms
    assert "XRP" not in syms   # filtered (zero balance)


def test_balance_futures(futures_client):
    client, mod = futures_client
    mock_balance = [
        {"asset": "USDT", "availableBalance": "800.0", "balance": "1000.0"},
        {"asset": "BNB", "availableBalance": "0.0", "balance": "0.0"},  # zero — filtered
    ]
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_balance)):
        r = client.get("/balance")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "futures"
    assert len(data["assets"]) == 1
    assert data["assets"][0]["asset"] == "USDT"


def test_balance_no_credentials(no_key_client):
    client, _ = no_key_client
    r = client.get("/balance")
    assert r.status_code == 503


# ── Positions ─────────────────────────────────────────────────────────────────

def test_positions_futures(futures_client):
    client, mod = futures_client
    mock_pos = [
        {
            "symbol": "BTCUSDT", "positionAmt": "0.01",
            "entryPrice": "90000", "markPrice": "95000",
            "unRealizedProfit": "50.0", "leverage": "10",
        },
        {
            "symbol": "ETHUSDT", "positionAmt": "0.0",  # zero — filtered
            "entryPrice": "0", "markPrice": "3500",
            "unRealizedProfit": "0.0", "leverage": "5",
        },
    ]
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_pos)):
        r = client.get("/positions")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "futures"
    assert len(data["positions"]) == 1
    p = data["positions"][0]
    assert p["symbol"] == "BTCUSDT"
    assert p["side"] == "LONG"
    assert p["unrealized_pnl"] == 50.0


def test_positions_spot(spot_client):
    client, mod = spot_client
    mock_account = {
        "balances": [
            {"asset": "BTC", "free": "0.5", "locked": "0.0"},
            {"asset": "USDT", "free": "1000.0", "locked": "0.0"},  # stable — skipped
        ]
    }
    mock_price = {"symbol": "BTCUSDT", "price": "95000.00"}
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_account)), \
            patch.object(mod, "_public_get", new=AsyncMock(return_value=mock_price)):
        r = client.get("/positions")
    assert r.status_code == 200
    data = r.json()
    assert data["mode"] == "spot"
    assert len(data["positions"]) == 1
    p = data["positions"][0]
    assert p["symbol"] == "BTC"
    assert p["value_usdt"] == round(0.5 * 95000.0, 2)


# ── Orders ────────────────────────────────────────────────────────────────────

def test_orders_spot(spot_client):
    client, mod = spot_client
    mock_orders = [
        {
            "symbol": "BTCUSDT", "orderId": 123, "side": "BUY", "type": "LIMIT",
            "price": "90000", "origQty": "0.01", "executedQty": "0.0", "status": "NEW",
        }
    ]
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_orders)):
        r = client.get("/orders")
    assert r.status_code == 200
    orders = r.json()["orders"]
    assert len(orders) == 1
    assert orders[0]["symbol"] == "BTCUSDT"
    assert orders[0]["side"] == "BUY"


def test_orders_with_symbol_filter(spot_client):
    client, mod = spot_client
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=[])) as mock:
        r = client.get("/orders?symbol=ETHUSDT")
    assert r.status_code == 200
    # verify symbol was uppercased and passed through
    call_kwargs = mock.call_args
    assert "ETHUSDT" in str(call_kwargs)


# ── PnL summary ───────────────────────────────────────────────────────────────

def test_pnl_summary_futures(futures_client):
    client, mod = futures_client
    mock_pos = [
        {"symbol": "BTCUSDT", "positionAmt": "0.01", "unRealizedProfit": "50.0",
         "entryPrice": "90000", "markPrice": "95000", "leverage": "10"},
        {"symbol": "ETHUSDT", "positionAmt": "-0.5", "unRealizedProfit": "-25.0",
         "entryPrice": "4000", "markPrice": "3950", "leverage": "5"},
        {"symbol": "SOLUSDT", "positionAmt": "0.0", "unRealizedProfit": "0.0",
         "entryPrice": "0", "markPrice": "150", "leverage": "1"},
    ]
    with patch.object(mod, "_signed_get", new=AsyncMock(return_value=mock_pos)):
        r = client.get("/pnl/summary")
    assert r.status_code == 200
    data = r.json()
    assert data["total_unrealized_pnl"] == 25.0
    assert len(data["positions"]) == 2


def test_pnl_summary_spot_mode_rejected(spot_client):
    client, _ = spot_client
    r = client.get("/pnl/summary")
    assert r.status_code == 400


# ── Templates ─────────────────────────────────────────────────────────────────

def test_templates(spot_client):
    client, _ = spot_client
    r = client.get("/templates")
    assert r.status_code == 200
    templates = r.json()["templates"]
    assert len(templates) >= 4
    ids = {t["id"] for t in templates}
    assert "price-above" in ids
    assert "price-below" in ids
    assert "price-drop-pct" in ids
    assert "whale-volume" in ids


def test_templates_have_rule_structure(spot_client):
    client, _ = spot_client
    r = client.get("/templates")
    for t in r.json()["templates"]:
        rule = t["rule"]
        assert "source" in rule
        assert "condition" in rule
        assert "actions" in rule
        assert "interval" in rule
        assert "cooldown" in rule
