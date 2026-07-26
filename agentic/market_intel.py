"""D129: Market Intelligence Module — regime awareness for Kai's strategy engine.

Phase 3: Sustainability. Raw price signals alone are sheep-following — the same
RSI reading means different things in a bull-market euphoria vs a capitulation
crash. This module aggregates free public feeds so strategies can be
regime-aware:

  Fear & Greed Index  — market emotion (Alternative.me, updated daily)
  Global stats        — BTC dominance, total market cap, 24h trend (CoinGecko)
  Trending coins      — crowd momentum / contra-indicator (CoinGecko)
  News sentiment      — per-symbol news tone (web_scout DuckDuckGo search)

Everything is read-only; the outputs are context dicts consumed by the
strategy engine to adjust signal weights, not to act directly.

Trust gating:
    All methods → OBSERVER (1) — read-only intelligence

Feature-flagged: FF_MARKET_INTEL=true
Fail-open: every external call is wrapped; partial data is returned rather
than raising. Cache TTL: Fear/Greed = 3600 s (daily); Global/Trending = 300 s;
News sentiment = 1800 s.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("kai.market_intel")

_FNG_URL    = "https://api.alternative.me/fng/"
_CG_BASE    = "https://api.coingecko.com/api/v3"
_TIMEOUT_S  = 5.0

_TTL_FNG      = 3600.0   # Fear & Greed changes once a day
_TTL_GLOBAL   = 300.0    # Global stats every 5 min is plenty
_TTL_TRENDING = 300.0
_TTL_SENTIMENT= 1800.0   # News sentiment; DuckDuckGo caches well


# ── Fear & Greed ───────────────────────────────────────────────────────────────

FNG_LABELS = {
    range(0,  26): "Extreme Fear",
    range(26, 47): "Fear",
    range(47, 54): "Neutral",
    range(54, 75): "Greed",
    range(75,101): "Extreme Greed",
}

def _fng_label(value: int) -> str:
    for r, label in FNG_LABELS.items():
        if value in r:
            return label
    return "Unknown"


@dataclass
class FearGreedReading:
    value: int           # 0 (extreme fear) – 100 (extreme greed)
    label: str           # human-readable classification
    timestamp: float
    next_update: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "regime": self._regime(),
            "timestamp": self.timestamp,
            "age_s": round(time.time() - self.timestamp, 0),
        }

    def _regime(self) -> str:
        if self.value < 26:   return "extreme_fear"
        if self.value < 47:   return "fear"
        if self.value < 54:   return "neutral"
        if self.value < 75:   return "greed"
        return "extreme_greed"


# ── Global market stats ────────────────────────────────────────────────────────

@dataclass
class GlobalStats:
    total_market_cap_usd: float
    total_volume_24h_usd: float
    btc_dominance_pct: float
    eth_dominance_pct: float
    market_cap_change_pct_24h: float
    active_cryptocurrencies: int
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_market_cap_usd": self.total_market_cap_usd,
            "total_volume_24h_usd": self.total_volume_24h_usd,
            "btc_dominance_pct": round(self.btc_dominance_pct, 2),
            "eth_dominance_pct": round(self.eth_dominance_pct, 2),
            "market_cap_change_pct_24h": round(self.market_cap_change_pct_24h, 2),
            "active_cryptocurrencies": self.active_cryptocurrencies,
            "trend_24h": "up" if self.market_cap_change_pct_24h >= 0 else "down",
            "timestamp": self.timestamp,
            "age_s": round(time.time() - self.timestamp, 0),
        }


# ── Trending coin ──────────────────────────────────────────────────────────────

@dataclass
class TrendingCoin:
    rank: int
    name: str
    symbol: str
    market_cap_rank: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "symbol": self.symbol.upper(),
            "market_cap_rank": self.market_cap_rank,
        }


# ── Market Intelligence ────────────────────────────────────────────────────────

class MarketIntelligence:
    """Aggregates regime-level market context from free public APIs."""

    def __init__(self, timeout_s: float = _TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}

    # ── Cache helpers ──────────────────────────────────────────────────

    def _cached(self, key: str, ttl: float) -> Optional[Any]:
        ts = self._cache_ts.get(key, 0.0)
        if time.time() - ts < ttl:
            return self._cache.get(key)
        return None

    def _store(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    # ── Fear & Greed ───────────────────────────────────────────────────

    def get_fear_greed(self) -> Optional[FearGreedReading]:
        """Fetch the current Crypto Fear & Greed Index from Alternative.me."""
        cached = self._cached("fng", _TTL_FNG)
        if cached is not None:
            return cached

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(_FNG_URL, params={"limit": 1, "format": "json"})
            if resp.status_code != 200:
                logger.warning("Fear & Greed API returned %d", resp.status_code)
                return None
            data = resp.json().get("data", [])
            if not data:
                return None
            entry = data[0]
            reading = FearGreedReading(
                value=int(entry["value"]),
                label=entry.get("value_classification", _fng_label(int(entry["value"]))),
                timestamp=time.time(),
                next_update=entry.get("time_until_update"),
            )
            self._store("fng", reading)
            logger.info("Fear & Greed: %d (%s)", reading.value, reading.label)
            return reading
        except Exception as exc:
            logger.debug("Fear & Greed fetch failed (fail-open): %s", exc)
            return None

    # ── Global stats ───────────────────────────────────────────────────

    def get_global_stats(self) -> Optional[GlobalStats]:
        """Fetch global crypto market stats from CoinGecko."""
        cached = self._cached("global", _TTL_GLOBAL)
        if cached is not None:
            return cached

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(f"{_CG_BASE}/global")
            if resp.status_code != 200:
                logger.warning("CoinGecko /global returned %d", resp.status_code)
                return None
            d = resp.json().get("data", {})
            cap = d.get("total_market_cap", {})
            vol = d.get("total_volume", {})
            dom = d.get("market_cap_percentage", {})
            stats = GlobalStats(
                total_market_cap_usd=float(cap.get("usd", 0)),
                total_volume_24h_usd=float(vol.get("usd", 0)),
                btc_dominance_pct=float(dom.get("btc", 0)),
                eth_dominance_pct=float(dom.get("eth", 0)),
                market_cap_change_pct_24h=float(
                    d.get("market_cap_change_percentage_24h_usd", 0)
                ),
                active_cryptocurrencies=int(d.get("active_cryptocurrencies", 0)),
                timestamp=time.time(),
            )
            self._store("global", stats)
            logger.info(
                "Global: BTC dom=%.1f%% market_cap_chg=%.2f%%",
                stats.btc_dominance_pct, stats.market_cap_change_pct_24h,
            )
            return stats
        except Exception as exc:
            logger.debug("CoinGecko /global fetch failed (fail-open): %s", exc)
            return None

    # ── Trending coins ─────────────────────────────────────────────────

    def get_trending(self) -> List[TrendingCoin]:
        """Fetch top trending coins from CoinGecko."""
        cached = self._cached("trending", _TTL_TRENDING)
        if cached is not None:
            return cached

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(f"{_CG_BASE}/search/trending")
            if resp.status_code != 200:
                logger.warning("CoinGecko /trending returned %d", resp.status_code)
                return []
            coins_raw = resp.json().get("coins", [])
            coins = []
            for i, entry in enumerate(coins_raw[:10]):
                item = entry.get("item", {})
                coins.append(TrendingCoin(
                    rank=i + 1,
                    name=item.get("name", ""),
                    symbol=item.get("symbol", ""),
                    market_cap_rank=item.get("market_cap_rank"),
                ))
            self._store("trending", coins)
            logger.info("Trending: %s", [c.symbol for c in coins[:5]])
            return coins
        except Exception as exc:
            logger.debug("CoinGecko /trending fetch failed (fail-open): %s", exc)
            return []

    # ── News sentiment ─────────────────────────────────────────────────

    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Fetch news headline sentiment for a symbol via DuckDuckGo search.

        Uses the existing web_scout module — no new network dependency.
        Returns a lightweight sentiment dict with tone and top headlines.
        """
        symbol = symbol.upper()
        cache_key = f"sentiment:{symbol}"
        cached = self._cached(cache_key, _TTL_SENTIMENT)
        if cached is not None:
            return cached

        coin_name = symbol.replace("USD", "").lower()
        query = f"{coin_name} crypto news sentiment today"
        result: Dict[str, Any] = {"symbol": symbol, "query": query,
                                   "abstract": "", "topics": [], "tone": "neutral",
                                   "timestamp": time.time()}
        try:
            try:
                from web_scout import search as ws_search
            except ImportError:
                from agentic.web_scout import search as ws_search  # type: ignore

            sr = ws_search(query, max_results=5, autonomous=False)
            if sr.error:
                logger.debug("Sentiment search failed for %s: %s", symbol, sr.error)
            else:
                result["abstract"] = sr.abstract[:300] if sr.abstract else ""
                result["topics"] = [t.get("Text", "")[:80] for t in sr.topics[:5]
                                    if isinstance(t, dict)]
                result["tone"] = _classify_tone(result["abstract"] + " " + " ".join(result["topics"]))
        except Exception as exc:
            logger.debug("News sentiment unavailable for %s (fail-open): %s", symbol, exc)

        self._store(cache_key, result)
        return result

    # ── Macro context ──────────────────────────────────────────────────

    def get_macro_context(self) -> Dict[str, Any]:
        """Fetch macro market context via web_scout news searches.

        Searches DuckDuckGo for signals across the macro layers that
        drive crypto: gold, oil, USD/DXY strength, Fed policy, and
        geopolitical risk. Returns tone + raw headlines per topic.
        Cached for 1800 s (same as news sentiment).
        """
        cache_key = "macro"
        cached = self._cached(cache_key, _TTL_SENTIMENT)
        if cached is not None:
            return cached

        _MACRO_QUERIES: Dict[str, str] = {
            "gold":        "gold price today crypto impact",
            "oil":         "oil price today market impact crypto",
            "dollar_dxy":  "US dollar DXY index crypto bitcoin impact today",
            "fed_rates":   "Federal Reserve interest rates crypto market today",
            "geopolitical":"geopolitical risk crypto market impact today",
        }

        result: Dict[str, Any] = {
            "timestamp": time.time(),
            "topics": {},
            "overall_tone": "neutral",
        }

        try:
            try:
                from web_scout import search as ws_search
            except ImportError:
                from agentic.web_scout import search as ws_search  # type: ignore

            tones: List[str] = []
            for topic, query in _MACRO_QUERIES.items():
                try:
                    sr = ws_search(query, max_results=3, autonomous=False)
                    text = (sr.abstract or "") + " " + " ".join(
                        t.get("Text", "") if isinstance(t, dict) else str(t)
                        for t in (sr.topics or [])
                    )
                    tone = _classify_tone(text)
                    tones.append(tone)
                    result["topics"][topic] = {
                        "query": query,
                        "abstract": (sr.abstract or "")[:200],
                        "tone": tone,
                    }
                except Exception as exc:
                    logger.debug("Macro query '%s' failed: %s", topic, exc)
                    result["topics"][topic] = {"query": query, "abstract": "", "tone": "neutral"}

            # Overall macro tone: majority vote across topics
            bull = tones.count("bullish")
            bear = tones.count("bearish")
            if bull > bear:
                result["overall_tone"] = "bullish"
            elif bear > bull:
                result["overall_tone"] = "bearish"

        except Exception as exc:
            logger.debug("Macro context unavailable (fail-open): %s", exc)

        self._store(cache_key, result)
        logger.info("Macro context: overall_tone=%s", result["overall_tone"])
        return result

    # ── Combined context ───────────────────────────────────────────────

    def context(self, symbol: str) -> Dict[str, Any]:
        """Return a combined intelligence context dict for a symbol.

        Strategy engine consumers can inspect this before weighting signals.
        All fields are present; missing data defaults to None / [].
        Includes coin-level news sentiment + macro context (gold, oil,
        DXY, Fed rates, geopolitical) so signals can be regime-aware.
        """
        fng   = self.get_fear_greed()
        glbl  = self.get_global_stats()
        trend = self.get_trending()
        news  = self.get_news_sentiment(symbol)
        macro = self.get_macro_context()

        trending_symbols = {c.symbol.upper() for c in trend}
        is_trending = symbol.replace("USD", "") in trending_symbols

        return {
            "symbol": symbol,
            "fear_greed": fng.to_dict() if fng else None,
            "global": glbl.to_dict() if glbl else None,
            "is_trending": is_trending,
            "trending_coins": [c.to_dict() for c in trend[:5]],
            "news_sentiment": news,
            "macro": macro,
            "timestamp": time.time(),
        }

    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "cached_keys": list(self._cache_ts.keys()),
            "cache_ages_s": {
                k: round(now - ts, 0) for k, ts in self._cache_ts.items()
            },
        }


# ── Sentiment tone classifier ──────────────────────────────────────────────────

_BEARISH_WORDS = {
    "crash", "dump", "bear", "sell", "down", "drop", "fear", "panic",
    "hack", "scam", "fraud", "ban", "regulation", "lawsuit", "fud",
    "lose", "loss", "plunge", "tumble", "decline", "fall", "risk",
}
_BULLISH_WORDS = {
    "rally", "bull", "buy", "up", "surge", "pump", "gain", "moon",
    "adoption", "partnership", "launch", "upgrade", "breakout", "all-time",
    "institutional", "etf", "approval", "positive", "growth", "profit",
}

def _classify_tone(text: str) -> str:
    """Simple bag-of-words tone classifier. Returns 'bullish', 'bearish', or 'neutral'."""
    words = set(text.lower().split())
    bull_score = len(words & _BULLISH_WORDS)
    bear_score = len(words & _BEARISH_WORDS)
    if bull_score > bear_score:
        return "bullish"
    if bear_score > bull_score:
        return "bearish"
    return "neutral"


# ── Singleton ──────────────────────────────────────────────────────────────────

_intel: Optional[MarketIntelligence] = None


def get_market_intel(timeout_s: float = _TIMEOUT_S) -> MarketIntelligence:
    global _intel
    if _intel is None:
        _intel = MarketIntelligence(timeout_s=timeout_s)
    return _intel


def reset_market_intel() -> None:
    global _intel
    _intel = None
