"""Tests for D130: Alpha Signal Engine — agentic/alpha_signals.py."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from alpha_signals import (
    AlphaSignalFeed,
    FundingRate,
    LongShortRatio,
    MarkPremium,
    OpenInterest,
    _bnb_symbol,
    get_alpha_signals,
    reset_alpha_signals,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _feed() -> AlphaSignalFeed:
    return AlphaSignalFeed(timeout_s=1.0)


def _mock_resp(status: int, body) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


_PREMIUM_INDEX_BODY = {
    "symbol": "BTCUSDT",
    "markPrice": "50100.00",
    "indexPrice": "50000.00",
    "lastFundingRate": "0.0001",
    "nextFundingTime": 1700000000000,
}

_OI_BODY = {
    "symbol": "BTCUSDT",
    "openInterest": "120000.00",
    "time": 1700000000000,
}

_LS_BODY = [
    {
        "symbol": "BTCUSDT",
        "longAccount": "0.65",
        "shortAccount": "0.35",
        "longShortRatio": "1.857",
        "timestamp": "1700000000000",
    }
]


# ── _bnb_symbol ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("inp,expected", [
    ("BTCUSD",  "BTCUSDT"),
    ("ETHUSD",  "ETHUSDT"),
    ("btcusd",  "BTCUSDT"),
    ("BTCUSDT", "BTCUSDT"),   # already correct
    ("SOLUSDT", "SOLUSDT"),   # already correct
])
def test_bnb_symbol_normalises(inp, expected):
    assert _bnb_symbol(inp) == expected


# ── FundingRate ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rate,expected", [
    ( 0.0015,  "extremely_long"),
    ( 0.0005,  "crowded_long"),
    ( 0.0002,  "mild_long"),
    ( 0.00005, "neutral"),
    (-0.00005, "neutral"),
    (-0.0002,  "mild_short"),
    (-0.0005,  "crowded_short"),
    (-0.0015,  "extremely_short"),
])
def test_funding_rate_sentiment(rate, expected):
    fr = FundingRate("BTCUSD", rate, rate * 3 * 365 * 100, 0, time.time())
    assert fr.sentiment() == expected


def test_funding_rate_to_dict_keys():
    fr = FundingRate("BTCUSD", 0.0001, 10.95, 0, time.time())
    d = fr.to_dict()
    for key in ("symbol", "rate", "rate_pct", "annualised_pct", "sentiment",
                "next_funding_time", "timestamp"):
        assert key in d


def test_funding_rate_annualised():
    rate = 0.0001
    fr = FundingRate("BTCUSD", rate, rate * 3 * 365 * 100, 0, time.time())
    assert fr.annualised_pct == pytest.approx(10.95, abs=0.01)


# ── LongShortRatio ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("long_pct,expected", [
    (80, "extremely_crowded_long"),
    (65, "crowded_long"),
    (50, "balanced"),
    (35, "crowded_short"),
    (20, "extremely_crowded_short"),
])
def test_ls_ratio_sentiment(long_pct, expected):
    lsr = LongShortRatio("BTCUSD", long_pct, 100 - long_pct, long_pct / (100 - long_pct), "1h", time.time())
    assert lsr.sentiment() == expected


def test_ls_ratio_to_dict_keys():
    lsr = LongShortRatio("BTCUSD", 65.0, 35.0, 1.857, "1h", time.time())
    d = lsr.to_dict()
    for key in ("symbol", "long_pct", "short_pct", "ls_ratio", "period",
                "sentiment", "timestamp"):
        assert key in d


# ── MarkPremium ────────────────────────────────────────────────────────────────

def test_mark_premium_contango():
    mp = MarkPremium("BTCUSD", 50100.0, 50000.0, 0.2, time.time())
    assert mp.to_dict()["basis"] == "contango"
    assert mp.to_dict()["premium_pct"] > 0


def test_mark_premium_backwardation():
    mp = MarkPremium("BTCUSD", 49900.0, 50000.0, -0.2, time.time())
    assert mp.to_dict()["basis"] == "backwardation"
    assert mp.to_dict()["premium_pct"] < 0


# ── get_funding_rate ───────────────────────────────────────────────────────────

def test_get_funding_rate_success():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)):
        fr = feed.get_funding_rate("BTCUSD")
    assert fr is not None
    assert fr.rate == pytest.approx(0.0001)
    assert fr.symbol == "BTCUSD"


def test_get_funding_rate_list_response():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, [_PREMIUM_INDEX_BODY])):
        fr = feed.get_funding_rate("BTCUSD")
    assert fr is not None
    assert fr.rate == pytest.approx(0.0001)


def test_get_funding_rate_http_error_returns_none():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(503, {})):
        fr = feed.get_funding_rate("BTCUSD")
    assert fr is None


def test_get_funding_rate_network_error_returns_none():
    feed = _feed()
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(side_effect=ConnectionError("down"))
    with patch("httpx.Client", return_value=client):
        fr = feed.get_funding_rate("BTCUSD")
    assert fr is None


def test_get_funding_rate_cached():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)) as mock_cls:
        feed.get_funding_rate("BTCUSD")
        feed.get_funding_rate("BTCUSD")
    assert mock_cls.call_count == 1


def test_get_funding_rate_empty_list_returns_none():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, [])):
        fr = feed.get_funding_rate("BTCUSD")
    assert fr is None


# ── get_funding_rates (batch) ──────────────────────────────────────────────────

def test_get_funding_rates_batch():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)):
        rates = feed.get_funding_rates(["BTCUSD", "ETHUSD"])
    assert "BTCUSD" in rates
    assert "ETHUSD" in rates


def test_get_funding_rates_partial_failure():
    feed = _feed()
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_resp(200, _PREMIUM_INDEX_BODY)
        return _mock_resp(503, {})

    with patch("httpx.Client", side_effect=side_effect):
        rates = feed.get_funding_rates(["BTCUSD", "ETHUSD"])
    # Only the successful one appears
    assert "BTCUSD" in rates
    assert "ETHUSD" not in rates


# ── get_open_interest ──────────────────────────────────────────────────────────

def test_get_open_interest_success():
    feed = _feed()

    def side_effect(*args, **kwargs):
        return _mock_resp(200, _OI_BODY) if call_count[0] == 0 else _mock_resp(200, _PREMIUM_INDEX_BODY)

    call_count = [0]

    def patched_get(path, params=None):
        if "openInterest" in path:
            return _OI_BODY
        return _PREMIUM_INDEX_BODY

    feed._get = patched_get  # type: ignore[method-assign]
    oi = feed.get_open_interest("BTCUSD")
    assert oi is not None
    assert oi.oi_contracts == pytest.approx(120000.0)
    assert oi.oi_usd is not None  # mark price was available


def test_get_open_interest_cached():
    feed = _feed()
    feed._get = MagicMock(return_value=_OI_BODY)  # type: ignore[method-assign]

    # Prime the cache
    oi1 = feed.get_open_interest("BTCUSD")
    call_count_after_first = feed._get.call_count  # type: ignore[union-attr]
    oi2 = feed.get_open_interest("BTCUSD")
    assert feed._get.call_count == call_count_after_first  # no additional calls  # type: ignore[union-attr]
    assert oi1 is oi2


def test_get_open_interest_fail_open():
    feed = _feed()
    feed._get = MagicMock(return_value=None)  # type: ignore[method-assign]
    oi = feed.get_open_interest("BTCUSD")
    assert oi is None


# ── get_long_short_ratio ───────────────────────────────────────────────────────

def test_get_long_short_ratio_success():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _LS_BODY)):
        lsr = feed.get_long_short_ratio("BTCUSD")
    assert lsr is not None
    assert lsr.long_pct == pytest.approx(65.0)
    assert lsr.short_pct == pytest.approx(35.0)
    assert lsr.period == "1h"


def test_get_long_short_ratio_http_error_returns_none():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(429, {})):
        lsr = feed.get_long_short_ratio("BTCUSD")
    assert lsr is None


def test_get_long_short_ratio_cached():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _LS_BODY)) as mock_cls:
        feed.get_long_short_ratio("BTCUSD")
        feed.get_long_short_ratio("BTCUSD")
    assert mock_cls.call_count == 1


def test_get_long_short_ratio_different_periods_cached_separately():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _LS_BODY)) as mock_cls:
        feed.get_long_short_ratio("BTCUSD", "1h")
        feed.get_long_short_ratio("BTCUSD", "4h")
    assert mock_cls.call_count == 2


def test_get_long_short_ratio_empty_list_returns_none():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, [])):
        lsr = feed.get_long_short_ratio("BTCUSD")
    assert lsr is None


# ── get_mark_premium ───────────────────────────────────────────────────────────

def test_get_mark_premium_success():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)):
        mp = feed.get_mark_premium("BTCUSD")
    assert mp is not None
    assert mp.mark_price == pytest.approx(50100.0)
    assert mp.index_price == pytest.approx(50000.0)
    assert mp.premium_pct == pytest.approx(0.2, abs=0.01)


def test_get_mark_premium_zero_index_returns_none():
    feed = _feed()
    body = {**_PREMIUM_INDEX_BODY, "indexPrice": "0"}
    with patch("httpx.Client", return_value=_mock_resp(200, body)):
        mp = feed.get_mark_premium("BTCUSD")
    assert mp is None


def test_get_mark_premium_cached():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)) as mock_cls:
        feed.get_mark_premium("BTCUSD")
        feed.get_mark_premium("BTCUSD")
    assert mock_cls.call_count == 1


def test_get_mark_premium_http_error_returns_none():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(500, {})):
        mp = feed.get_mark_premium("BTCUSD")
    assert mp is None


# ── composite ──────────────────────────────────────────────────────────────────

def test_composite_returns_all_keys():
    feed = _feed()
    feed.get_funding_rate  = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_open_interest = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_long_short_ratio = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_mark_premium  = MagicMock(return_value=None)  # type: ignore[method-assign]
    result = feed.composite("BTCUSD")
    for key in ("symbol", "funding", "open_interest", "long_short_ratio",
                "mark_premium", "timestamp"):
        assert key in result


def test_composite_none_signals_present_as_none():
    feed = _feed()
    feed.get_funding_rate  = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_open_interest = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_long_short_ratio = MagicMock(return_value=None)  # type: ignore[method-assign]
    feed.get_mark_premium  = MagicMock(return_value=None)  # type: ignore[method-assign]
    result = feed.composite("BTCUSD")
    assert result["funding"] is None
    assert result["open_interest"] is None


# ── status ─────────────────────────────────────────────────────────────────────

def test_status_empty():
    feed = _feed()
    s = feed.status()
    assert s["cached_keys"] == []


def test_status_after_fetch():
    feed = _feed()
    with patch("httpx.Client", return_value=_mock_resp(200, _PREMIUM_INDEX_BODY)):
        feed.get_funding_rate("BTCUSD")
    s = feed.status()
    assert any("funding" in k for k in s["cached_keys"])


# ── Singleton ──────────────────────────────────────────────────────────────────

def test_singleton_same_instance():
    reset_alpha_signals()
    f1 = get_alpha_signals()
    f2 = get_alpha_signals()
    assert f1 is f2
    reset_alpha_signals()


def test_reset_clears_singleton():
    reset_alpha_signals()
    f1 = get_alpha_signals()
    reset_alpha_signals()
    f2 = get_alpha_signals()
    assert f1 is not f2
    reset_alpha_signals()
