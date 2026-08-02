"""Tests for D127: Market Data Feed — agentic/market_data.py."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from market_data import (
    MarketDataFeed,
    PriceQuote,
    _SYMBOL_MAP,
    get_market_data,
    reset_market_data,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feed(ttl_s: float = 60.0) -> MarketDataFeed:
    return MarketDataFeed(ttl_s=ttl_s, timeout_s=2.0)


def _mock_coingecko(prices: dict) -> MagicMock:
    """Return a mock httpx.Client.get that returns the given {coin_id: usd} prices."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {k: {"usd": v} for k, v in prices.items()}
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


# ── known_symbols ─────────────────────────────────────────────────────────────

def test_known_symbols_returns_sorted():
    f = _feed()
    syms = f.known_symbols()
    assert syms == sorted(syms)
    assert "BTCUSD" in syms
    assert "ETHUSD" in syms


def test_known_symbols_all_uppercase():
    f = _feed()
    for s in f.known_symbols():
        assert s == s.upper()


# ── get_price: unknown symbol ─────────────────────────────────────────────────

def test_get_price_unknown_symbol_returns_none():
    f = _feed()
    assert f.get_price("FAKEUSD") is None


def test_get_price_normalises_to_upper():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 50000.0})):
        price = f.get_price("btcusd")
    assert price == 50000.0


# ── get_prices: basic fetch ───────────────────────────────────────────────────

def test_get_prices_empty_list():
    f = _feed()
    assert f.get_prices([]) == {}


def test_get_prices_fetches_coingecko(monkeypatch):
    f = _feed()
    client = _mock_coingecko({"bitcoin": 50000.0, "ethereum": 3000.0})
    with patch("httpx.Client", return_value=client):
        result = f.get_prices(["BTCUSD", "ETHUSD"])
    assert result["BTCUSD"] == pytest.approx(50000.0)
    assert result["ETHUSD"] == pytest.approx(3000.0)


def test_get_prices_skips_unknown_symbols():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 50000.0})):
        result = f.get_prices(["BTCUSD", "FAKEUSD"])
    assert "BTCUSD" in result
    assert "FAKEUSD" not in result


def test_get_prices_case_insensitive():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 50000.0})):
        result = f.get_prices(["btcusd"])
    assert result.get("BTCUSD") == pytest.approx(50000.0)


# ── Cache behaviour ───────────────────────────────────────────────────────────

def test_cache_hit_skips_network():
    f = _feed(ttl_s=60.0)
    f._cache["BTCUSD"] = PriceQuote("BTCUSD", 50000.0, time.time())
    with patch("httpx.Client") as mock_client:
        result = f.get_prices(["BTCUSD"])
    mock_client.assert_not_called()
    assert result["BTCUSD"] == pytest.approx(50000.0)


def test_cache_stale_triggers_fetch():
    f = _feed(ttl_s=1.0)
    f._cache["BTCUSD"] = PriceQuote("BTCUSD", 49000.0, time.time() - 5.0)
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 51000.0})):
        result = f.get_prices(["BTCUSD"])
    assert result["BTCUSD"] == pytest.approx(51000.0)


def test_cache_updated_after_fetch():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 50000.0})):
        f.get_prices(["BTCUSD"])
    assert "BTCUSD" in f._cache
    assert f._cache["BTCUSD"].price_usd == pytest.approx(50000.0)


def test_cache_mix_fresh_and_stale():
    f = _feed(ttl_s=60.0)
    # BTC is fresh in cache
    f._cache["BTCUSD"] = PriceQuote("BTCUSD", 50000.0, time.time())
    # ETH is not in cache — must be fetched
    client = _mock_coingecko({"ethereum": 3000.0})
    with patch("httpx.Client", return_value=client):
        result = f.get_prices(["BTCUSD", "ETHUSD"])
    assert result["BTCUSD"] == pytest.approx(50000.0)
    assert result["ETHUSD"] == pytest.approx(3000.0)
    # CoinGecko was only called once (for ETH)
    client.get.assert_called_once()


# ── Fail-open: network errors ─────────────────────────────────────────────────

def test_network_error_returns_empty():
    f = _feed()
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(side_effect=ConnectionError("network down"))
    with patch("httpx.Client", return_value=client):
        result = f.get_prices(["BTCUSD"])
    assert result == {}


def test_http_error_status_returns_empty():
    f = _feed()
    resp = MagicMock()
    resp.status_code = 429  # rate limited
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    with patch("httpx.Client", return_value=client):
        result = f.get_prices(["BTCUSD"])
    assert result == {}


def test_malformed_response_returns_empty():
    f = _feed()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"bitcoin": {"gbp": 40000.0}}  # no "usd" key
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    with patch("httpx.Client", return_value=client):
        result = f.get_prices(["BTCUSD"])
    assert result == {}


# ── PriceQuote ────────────────────────────────────────────────────────────────

def test_price_quote_to_dict():
    q = PriceQuote("BTCUSD", 50000.0, time.time() - 5.0)
    d = q.to_dict()
    assert d["symbol"] == "BTCUSD"
    assert d["price_usd"] == pytest.approx(50000.0)
    assert d["source"] == "coingecko"
    assert d["age_s"] >= 5.0


# ── status ────────────────────────────────────────────────────────────────────

def test_status_empty():
    f = _feed()
    s = f.status()
    assert s["cached_symbols"] == 0
    assert s["quotes"] == []


def test_status_after_fetch():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 50000.0})):
        f.get_prices(["BTCUSD"])
    s = f.status()
    assert s["cached_symbols"] == 1
    assert s["quotes"][0]["symbol"] == "BTCUSD"
    assert s["quotes"][0]["fresh"] is True


def test_status_fresh_flag_false_when_stale():
    f = _feed(ttl_s=1.0)
    f._cache["BTCUSD"] = PriceQuote("BTCUSD", 50000.0, time.time() - 10.0)
    s = f.status()
    assert s["quotes"][0]["fresh"] is False


# ── prices_for_positions ──────────────────────────────────────────────────────
#
# `mark_positions()` used to live here and imported the paper trader, which
# made this Perception Provider reach into an Actuator — roadmap §15 rule 1.
# The composition now lives in agentic.app._mark_paper_positions(); this
# provider supplies prices and nothing else.

def test_prices_for_positions_returns_prices():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({"bitcoin": 51000.0})):
        result = f.prices_for_positions(["BTCUSD"])
    assert result.get("BTCUSD") == 51000.0


def test_prices_for_positions_empty_symbols():
    f = _feed()
    assert f.prices_for_positions([]) == {}


def test_prices_for_positions_unknown_symbol():
    f = _feed()
    with patch("httpx.Client", return_value=_mock_coingecko({})):
        assert f.prices_for_positions(["FAKEUSD"]) == {}


def test_provider_does_not_import_actuator():
    """The §15 rule-1 fix must stay fixed."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "agentic" / "market_data.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[-1])
    assert "paper_trader" not in imported, (
        "market_data (Perception Provider) must not import paper_trader "
        "(Actuator) — roadmap §15 rule 1"
    )


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_same_instance():
    reset_market_data()
    f1 = get_market_data()
    f2 = get_market_data()
    assert f1 is f2
    reset_market_data()


def test_reset_clears_singleton():
    reset_market_data()
    f1 = get_market_data()
    reset_market_data()
    f2 = get_market_data()
    assert f1 is not f2
    reset_market_data()
