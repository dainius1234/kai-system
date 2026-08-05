"""Tests for D129: Market Intelligence Module — agentic/market_intel.py."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from market_intel import (
    FearGreedReading,
    GlobalStats,
    MarketIntelligence,
    TrendingCoin,
    _classify_tone,
    _fng_label,
    get_market_intel,
    reset_market_intel,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _intel() -> MarketIntelligence:
    return MarketIntelligence(timeout_s=1.0)


def _mock_resp(status: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


_FNG_BODY = {
    "data": [{"value": "25", "value_classification": "Extreme Fear", "time_until_update": "3600"}]
}
_GLOBAL_BODY = {
    "data": {
        "total_market_cap": {"usd": 1_000_000_000_000},
        "total_volume": {"usd": 50_000_000_000},
        "market_cap_percentage": {"btc": 52.5, "eth": 17.3},
        "market_cap_change_percentage_24h_usd": -2.1,
        "active_cryptocurrencies": 10000,
    }
}
_TRENDING_BODY = {
    "coins": [
        {"item": {"name": "Bitcoin", "symbol": "BTC", "market_cap_rank": 1}},
        {"item": {"name": "Ethereum", "symbol": "ETH", "market_cap_rank": 2}},
        {"item": {"name": "Pepe", "symbol": "PEPE", "market_cap_rank": 50}},
    ]
}


# ── _fng_label ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (0, "Extreme Fear"),
    (10, "Extreme Fear"),
    (25, "Extreme Fear"),
    (26, "Fear"),
    (46, "Fear"),
    (47, "Neutral"),
    (53, "Neutral"),
    (54, "Greed"),
    (74, "Greed"),
    (75, "Extreme Greed"),
    (100, "Extreme Greed"),
])
def test_fng_label(value, expected):
    assert _fng_label(value) == expected


# ── _classify_tone ────────────────────────────────────────────────────────────

def test_classify_tone_bullish():
    assert _classify_tone("Bitcoin rally surge adoption bull market") == "bullish"


def test_classify_tone_bearish():
    assert _classify_tone("crash dump fear panic sell regulation ban") == "bearish"


def test_classify_tone_neutral_empty():
    assert _classify_tone("") == "neutral"


def test_classify_tone_neutral_mixed():
    # Equal bullish and bearish words → neutral
    assert _classify_tone("rally crash") == "neutral"


# ── FearGreedReading ──────────────────────────────────────────────────────────

def test_fear_greed_regime_extreme_fear():
    r = FearGreedReading(value=10, label="Extreme Fear", timestamp=time.time())
    assert r.to_dict()["regime"] == "extreme_fear"


def test_fear_greed_regime_greed():
    r = FearGreedReading(value=70, label="Greed", timestamp=time.time())
    assert r.to_dict()["regime"] == "greed"


def test_fear_greed_to_dict_keys():
    r = FearGreedReading(value=50, label="Neutral", timestamp=time.time())
    d = r.to_dict()
    assert "value" in d and "label" in d and "regime" in d and "age_s" in d


# ── GlobalStats ───────────────────────────────────────────────────────────────

def test_global_stats_trend_down():
    s = GlobalStats(1e12, 5e10, 52.0, 17.0, -2.1, 10000, time.time())
    assert s.to_dict()["trend_24h"] == "down"


def test_global_stats_trend_up():
    s = GlobalStats(1e12, 5e10, 52.0, 17.0, 1.5, 10000, time.time())
    assert s.to_dict()["trend_24h"] == "up"


# ── TrendingCoin ──────────────────────────────────────────────────────────────

def test_trending_coin_symbol_uppercase():
    tc = TrendingCoin(rank=1, name="Bitcoin", symbol="btc", market_cap_rank=1)
    assert tc.to_dict()["symbol"] == "BTC"


# ── get_fear_greed ────────────────────────────────────────────────────────────

def test_fear_greed_success():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _FNG_BODY)):
        r = mi.get_fear_greed()
    assert r is not None
    assert r.value == 25
    assert r.label == "Extreme Fear"


def test_fear_greed_http_error_returns_none():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(503, {})):
        r = mi.get_fear_greed()
    assert r is None


def test_fear_greed_network_error_returns_none():
    mi = _intel()
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(side_effect=ConnectionError("down"))
    with patch("httpx.Client", return_value=client):
        r = mi.get_fear_greed()
    assert r is None


def test_fear_greed_cached():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _FNG_BODY)) as mock_cls:
        mi.get_fear_greed()
        mi.get_fear_greed()   # second call should hit cache
    assert mock_cls.call_count == 1


def test_fear_greed_empty_data_returns_none():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, {"data": []})):
        r = mi.get_fear_greed()
    assert r is None


# ── get_global_stats ──────────────────────────────────────────────────────────

def test_global_stats_success():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _GLOBAL_BODY)):
        s = mi.get_global_stats()
    assert s is not None
    assert s.btc_dominance_pct == pytest.approx(52.5)
    assert s.market_cap_change_pct_24h == pytest.approx(-2.1)


def test_global_stats_http_error_returns_none():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(429, {})):
        s = mi.get_global_stats()
    assert s is None


def test_global_stats_cached():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _GLOBAL_BODY)) as mock_cls:
        mi.get_global_stats()
        mi.get_global_stats()
    assert mock_cls.call_count == 1


# ── get_trending ──────────────────────────────────────────────────────────────

def test_trending_success():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _TRENDING_BODY)):
        coins = mi.get_trending()
    assert len(coins) == 3
    assert coins[0].symbol == "BTC"
    assert coins[0].rank == 1


def test_trending_http_error_returns_empty():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(500, {})):
        coins = mi.get_trending()
    assert coins == []


def test_trending_cached():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _TRENDING_BODY)) as mock_cls:
        mi.get_trending()
        mi.get_trending()
    assert mock_cls.call_count == 1


# ── get_news_sentiment ────────────────────────────────────────────────────────

def _mock_search_result(abstract: str = "", topics: list = None):
    from web_scout import SearchResult
    return SearchResult(
        query="btc crypto news",
        abstract=abstract,
        abstract_url="",
        topics=topics or [],
        answer="",
        elapsed_ms=50.0,
    )


def test_news_sentiment_success():
    mi = _intel()
    sr = _mock_search_result(abstract="Bitcoin rally adoption surge bull market")
    with patch("web_scout.search", return_value=sr):
        result = mi.get_news_sentiment("BTCUSD")
    assert result["symbol"] == "BTCUSD"
    assert result["tone"] == "bullish"


def test_news_sentiment_bearish():
    mi = _intel()
    sr = _mock_search_result(abstract="crash dump panic fear sell regulation ban")
    with patch("web_scout.search", return_value=sr):
        result = mi.get_news_sentiment("BTCUSD")
    assert result["tone"] == "bearish"


def test_news_sentiment_fail_open():
    mi = _intel()
    with patch("web_scout.search", side_effect=RuntimeError("unavailable")):
        result = mi.get_news_sentiment("BTCUSD")
    assert result["symbol"] == "BTCUSD"
    assert result["tone"] == "neutral"


def test_news_sentiment_cached():
    mi = _intel()
    sr = _mock_search_result()
    with patch("web_scout.search", return_value=sr) as mock_search:
        mi.get_news_sentiment("BTCUSD")
        mi.get_news_sentiment("BTCUSD")
    assert mock_search.call_count == 1


def test_news_sentiment_different_symbols_cached_separately():
    mi = _intel()
    sr = _mock_search_result()
    with patch("web_scout.search", return_value=sr) as mock_search:
        mi.get_news_sentiment("BTCUSD")
        mi.get_news_sentiment("ETHUSD")
    assert mock_search.call_count == 2


# ── get_macro_context ─────────────────────────────────────────────────────────

def _mock_search(abstract: str = ""):
    from web_scout import SearchResult
    return SearchResult(query="q", abstract=abstract, abstract_url="",
                        topics=[], answer="", elapsed_ms=10.0)


def test_macro_context_returns_all_topics():
    mi = _intel()
    sr = _mock_search("oil gold rally surge adoption")
    with patch("web_scout.search", return_value=sr):
        result = mi.get_macro_context()
    assert "topics" in result
    assert "gold" in result["topics"]
    assert "oil" in result["topics"]
    assert "dollar_dxy" in result["topics"]
    assert "fed_rates" in result["topics"]
    assert "geopolitical" in result["topics"]


def test_macro_context_overall_tone_bullish():
    mi = _intel()
    sr = _mock_search("rally surge adoption bull growth profit")
    with patch("web_scout.search", return_value=sr):
        result = mi.get_macro_context()
    # All queries return bullish → overall bullish
    assert result["overall_tone"] == "bullish"


def test_macro_context_overall_tone_bearish():
    mi = _intel()
    sr = _mock_search("crash dump fear panic ban regulation risk lose")
    with patch("web_scout.search", return_value=sr):
        result = mi.get_macro_context()
    assert result["overall_tone"] == "bearish"


def test_macro_context_fail_open():
    mi = _intel()
    with patch("web_scout.search", side_effect=RuntimeError("unavailable")):
        result = mi.get_macro_context()
    assert result["overall_tone"] == "neutral"
    assert "topics" in result


def test_macro_context_cached():
    mi = _intel()
    sr = _mock_search()
    with patch("web_scout.search", return_value=sr) as mock_s:
        mi.get_macro_context()
        mi.get_macro_context()
    # 5 queries first time, 0 second time
    assert mock_s.call_count == 5


# ── context ───────────────────────────────────────────────────────────────────

def test_context_returns_all_keys():
    mi = _intel()
    mi.get_fear_greed = MagicMock(return_value=None)               # type: ignore[method-assign]
    mi.get_global_stats = MagicMock(return_value=None)               # type: ignore[method-assign]
    mi.get_trending = MagicMock(return_value=[])                 # type: ignore[method-assign]
    mi.get_news_sentiment = MagicMock(return_value={"tone": "neutral"})  # type: ignore[method-assign]
    mi.get_macro_context = MagicMock(return_value={"overall_tone": "neutral", "topics": {}})  # type: ignore[method-assign]
    ctx = mi.context("BTCUSD")
    assert "symbol" in ctx
    assert "fear_greed" in ctx
    assert "global" in ctx
    assert "is_trending" in ctx
    assert "trending_coins" in ctx
    assert "news_sentiment" in ctx
    assert "macro" in ctx


def _stub_macro():
    return MagicMock(return_value={"overall_tone": "neutral", "topics": {}})


def test_context_is_trending_detected():
    mi = _intel()
    mi.get_fear_greed = MagicMock(return_value=None)                # type: ignore[method-assign]
    mi.get_global_stats = MagicMock(return_value=None)                # type: ignore[method-assign]
    mi.get_trending = MagicMock(return_value=[TrendingCoin(1, "Bitcoin", "BTC", 1)])  # type: ignore[method-assign]
    mi.get_news_sentiment = MagicMock(return_value={})                  # type: ignore[method-assign]
    mi.get_macro_context = _stub_macro()                               # type: ignore[method-assign]
    ctx = mi.context("BTCUSD")
    assert ctx["is_trending"] is True


def test_context_not_trending():
    mi = _intel()
    mi.get_fear_greed = MagicMock(return_value=None)                # type: ignore[method-assign]
    mi.get_global_stats = MagicMock(return_value=None)                # type: ignore[method-assign]
    mi.get_trending = MagicMock(return_value=[TrendingCoin(1, "Pepe", "PEPE", 50)])  # type: ignore[method-assign]
    mi.get_news_sentiment = MagicMock(return_value={})                  # type: ignore[method-assign]
    mi.get_macro_context = _stub_macro()                               # type: ignore[method-assign]
    ctx = mi.context("BTCUSD")
    assert ctx["is_trending"] is False


# ── status ────────────────────────────────────────────────────────────────────

def test_status_empty():
    mi = _intel()
    s = mi.status()
    assert s["cached_keys"] == []


def test_status_after_fetch():
    mi = _intel()
    with patch("httpx.Client", return_value=_mock_resp(200, _FNG_BODY)):
        mi.get_fear_greed()
    s = mi.status()
    assert "fng" in s["cached_keys"]


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_same_instance():
    reset_market_intel()
    m1 = get_market_intel()
    m2 = get_market_intel()
    assert m1 is m2
    reset_market_intel()


def test_reset_clears_singleton():
    reset_market_intel()
    m1 = get_market_intel()
    reset_market_intel()
    m2 = get_market_intel()
    assert m1 is not m2
    reset_market_intel()
