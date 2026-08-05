"""D130: Alpha Signal Engine — quantitative signals that professionals actually use.

Phase 3: Sustainability. Standard RSI/MA are retail noise. The signals that
matter to prop desks, quant funds, and serious operators are:

  Funding Rate    — cost of leverage; extreme positive = crowded longs
                    (the market is paying you to be short)
  Open Interest   — total leverage magnitude; rising OI + falling price = cascade risk
  Long/Short Ratio— positioning: when retail is 80% long, the exit is crowded
  Mark Premium    — futures price vs spot; basis tells you carry and arbitrage pressure
  Liquidation Map — estimated cascade levels from price × OI distribution

All from Binance Futures public endpoints — no API key, no auth.

These signals are the INPUTS to opportunity scoring, not trading decisions.
Trust gating:
    All fetches → OBSERVER (1) — read-only market intelligence

Feature-flagged: FF_ALPHA_SIGNALS=true
Fail-open: every network failure returns None / empty; caller degrades gracefully.
Cache: funding=300s, OI=60s, L/S=60s, premium=60s.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from common.degraded import record_degradation

logger = logging.getLogger("kai.alpha_signals")

_BASE = "https://fapi.binance.com"
_TIMEOUT_S = 5.0

_TTL_FUNDING  = 300.0   # updates every 8h, checking every 5 min is fine
_TTL_OI       = 60.0
_TTL_LS_RATIO = 60.0
_TTL_PREMIUM  = 60.0

# Binance uses BTCUSDT format; we accept BTCUSD and normalise
def _bnb_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        return s[:-3] + "USDT"
    return s + "USDT"


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FundingRate:
    symbol: str
    rate: float           # current 8h rate (0.0001 = 0.01%)
    annualised_pct: float # rate * 3 * 365 * 100
    next_funding_time: int
    timestamp: float

    def sentiment(self) -> str:
        """Interpret funding rate as market sentiment."""
        if self.rate > 0.001:    return "extremely_long"    # longs paying > 0.1%/8h
        if self.rate > 0.0003:   return "crowded_long"
        if self.rate > 0.0001:   return "mild_long"
        if self.rate < -0.001:   return "extremely_short"   # shorts paying > 0.1%/8h
        if self.rate < -0.0003:  return "crowded_short"
        if self.rate < -0.0001:  return "mild_short"
        return "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rate": self.rate,
            "rate_pct": round(self.rate * 100, 6),
            "annualised_pct": round(self.annualised_pct, 2),
            "sentiment": self.sentiment(),
            "next_funding_time": self.next_funding_time,
            "timestamp": self.timestamp,
        }


@dataclass
class OpenInterest:
    symbol: str
    oi_contracts: float     # notional in contracts
    oi_usd: Optional[float] # estimated USD value (requires mark price)
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "oi_contracts": self.oi_contracts,
            "oi_usd": self.oi_usd,
            "timestamp": self.timestamp,
        }


@dataclass
class LongShortRatio:
    symbol: str
    long_pct: float     # % of accounts long
    short_pct: float    # % of accounts short
    ls_ratio: float     # long / short ratio
    period: str         # "5m" | "15m" | "1h" | "4h"
    timestamp: float

    def sentiment(self) -> str:
        if self.long_pct > 75:  return "extremely_crowded_long"
        if self.long_pct > 60:  return "crowded_long"
        if self.long_pct < 25:  return "extremely_crowded_short"
        if self.long_pct < 40:  return "crowded_short"
        return "balanced"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "long_pct": round(self.long_pct, 2),
            "short_pct": round(self.short_pct, 2),
            "ls_ratio": round(self.ls_ratio, 4),
            "period": self.period,
            "sentiment": self.sentiment(),
            "timestamp": self.timestamp,
        }


@dataclass
class MarkPremium:
    symbol: str
    mark_price: float
    index_price: float     # spot reference
    premium_pct: float     # (mark - index) / index * 100
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
            "premium_pct": round(self.premium_pct, 4),
            "basis": "contango" if self.premium_pct > 0 else "backwardation",
            "timestamp": self.timestamp,
        }


# ── Alpha Signal Feed ──────────────────────────────────────────────────────────

class AlphaSignalFeed:
    """Fetches quantitative signals from Binance Futures public API.

    All endpoints are unauthenticated — this module never touches API keys.
    Fail-open: None returned on any error; callers must handle gracefully.
    """

    def __init__(self, timeout_s: float = _TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._cache: Dict[str, Any] = {}
        self._cache_ts: Dict[str, float] = {}

    # ── Cache ──────────────────────────────────────────────────────────

    def _cached(self, key: str, ttl: float) -> Optional[Any]:
        ts = self._cache_ts.get(key, 0.0)
        if time.time() - ts < ttl:
            return self._cache.get(key)
        return None

    def _store(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._cache_ts[key] = time.time()

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Any]:
        """HTTP GET against Binance Futures. Returns parsed JSON or None."""
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(f"{_BASE}{path}", params=params or {})
            if resp.status_code != 200:
                logger.warning("Binance %s returned %d", path, resp.status_code)
                return None
            return resp.json()
        except Exception as exc:
            logger.debug("Binance %s failed (fail-open): %s", path, exc)
            return None

    # ── Funding Rate ───────────────────────────────────────────────────

    def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Current funding rate for a perpetual contract.

        Positive = longs pay shorts (market is overleveraged long).
        Negative = shorts pay longs (capitulation, overleveraged short).
        Typical neutral range: ±0.01%/8h. Extremes: >0.1%/8h = bubble territory.
        """
        bnb = _bnb_symbol(symbol)
        cache_key = f"funding:{bnb}"
        cached = self._cached(cache_key, _TTL_FUNDING)
        if cached is not None:
            return cached

        data = self._get("/fapi/v1/premiumIndex", {"symbol": bnb})
        if not data:
            return None
        if isinstance(data, list):
            data = data[0] if data else None
        if not data:
            return None

        try:
            rate = float(data.get("lastFundingRate", 0))
            annualised = rate * 3 * 365 * 100
            fr = FundingRate(
                symbol=symbol.upper(),
                rate=rate,
                annualised_pct=annualised,
                next_funding_time=int(data.get("nextFundingTime", 0)),
                timestamp=time.time(),
            )
            self._store(cache_key, fr)
            logger.info("Funding %s: %.4f%% (%.1f%% ann.) — %s",
                        symbol, rate * 100, annualised, fr.sentiment())
            return fr
        except Exception as exc:
            logger.debug("Funding parse error for %s: %s", symbol, exc)
            return None

    def get_funding_rates(self, symbols: List[str]) -> Dict[str, FundingRate]:
        """Batch funding rates for multiple symbols."""
        result: Dict[str, FundingRate] = {}
        for sym in symbols:
            fr = self.get_funding_rate(sym)
            if fr:
                result[sym.upper()] = fr
        return result

    # ── Open Interest ──────────────────────────────────────────────────

    def get_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        """Total open interest — magnitude of leverage in the market.

        Rising OI + falling price = new shorts added (bearish pressure).
        Rising OI + rising price = new longs added (bullish momentum).
        Falling OI + price move = liquidations / position unwinding.
        """
        bnb = _bnb_symbol(symbol)
        cache_key = f"oi:{bnb}"
        cached = self._cached(cache_key, _TTL_OI)
        if cached is not None:
            return cached

        data = self._get("/fapi/v1/openInterest", {"symbol": bnb})
        if not data:
            return None

        # Get mark price to estimate USD value
        mark_data = self._get("/fapi/v1/premiumIndex", {"symbol": bnb})
        mark_price = None
        if mark_data:
            if isinstance(mark_data, list):
                mark_data = mark_data[0] if mark_data else None
            if mark_data:
                try:
                    mark_price = float(mark_data.get("markPrice", 0)) or None
                except Exception as _exc:
                    record_degradation("market", "mark_price_parse", _exc)

        try:
            oi_contracts = float(data.get("openInterest", 0))
            oi_usd = oi_contracts * mark_price if mark_price else None
            oi = OpenInterest(
                symbol=symbol.upper(),
                oi_contracts=oi_contracts,
                oi_usd=oi_usd,
                timestamp=time.time(),
            )
            self._store(cache_key, oi)
            logger.info("OI %s: %.2f contracts (~$%.0fM)",
                        symbol, oi_contracts, (oi_usd or 0) / 1e6)
            return oi
        except Exception as exc:
            logger.debug("OI parse error for %s: %s", symbol, exc)
            return None

    # ── Long/Short Ratio ───────────────────────────────────────────────

    def get_long_short_ratio(
        self, symbol: str, period: str = "1h"
    ) -> Optional[LongShortRatio]:
        """Global long/short account ratio — retail crowd positioning.

        When >75% of accounts are long, they are the exit liquidity.
        Best used as a CONTRA-indicator at extremes:
          >75% long  → contrarian sell signal
          <25% long  → contrarian buy signal
          40-60%     → balanced, no strong signal
        """
        bnb = _bnb_symbol(symbol)
        cache_key = f"ls:{bnb}:{period}"
        cached = self._cached(cache_key, _TTL_LS_RATIO)
        if cached is not None:
            return cached

        data = self._get(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": bnb, "period": period, "limit": 1},
        )
        if not data or not isinstance(data, list) or not data:
            return None

        try:
            entry = data[0]
            long_pct  = float(entry.get("longAccount", 0)) * 100
            short_pct = float(entry.get("shortAccount", 0)) * 100
            ls_ratio  = float(entry.get("longShortRatio", 0))
            lsr = LongShortRatio(
                symbol=symbol.upper(),
                long_pct=long_pct,
                short_pct=short_pct,
                ls_ratio=ls_ratio,
                period=period,
                timestamp=time.time(),
            )
            self._store(cache_key, lsr)
            logger.info("L/S %s: %.1f%% long — %s", symbol, long_pct, lsr.sentiment())
            return lsr
        except Exception as exc:
            logger.debug("L/S parse error for %s: %s", symbol, exc)
            return None

    # ── Mark Premium ───────────────────────────────────────────────────

    def get_mark_premium(self, symbol: str) -> Optional[MarkPremium]:
        """Futures mark price vs spot index — basis / carry signal.

        Contango (mark > spot): market expects higher prices; carry favours longs.
        Backwardation (mark < spot): market in stress; forced selling in futures.
        Extreme contango signals over-enthusiasm; extreme backwardation signals panic.
        """
        bnb = _bnb_symbol(symbol)
        cache_key = f"premium:{bnb}"
        cached = self._cached(cache_key, _TTL_PREMIUM)
        if cached is not None:
            return cached

        data = self._get("/fapi/v1/premiumIndex", {"symbol": bnb})
        if not data:
            return None
        if isinstance(data, list):
            data = data[0] if data else None
        if not data:
            return None

        try:
            mark  = float(data.get("markPrice", 0))
            index = float(data.get("indexPrice", 0))
            if index <= 0:
                return None
            premium_pct = (mark - index) / index * 100
            mp = MarkPremium(
                symbol=symbol.upper(),
                mark_price=mark,
                index_price=index,
                premium_pct=premium_pct,
                timestamp=time.time(),
            )
            self._store(cache_key, mp)
            logger.info("Premium %s: %.4f%% (%s)", symbol, premium_pct,
                        "contango" if premium_pct > 0 else "backwardation")
            return mp
        except Exception as exc:
            logger.debug("Premium parse error for %s: %s", symbol, exc)
            return None

    # ── Composite signal ───────────────────────────────────────────────

    def composite(self, symbol: str) -> Dict[str, Any]:
        """Combine all alpha signals into one context dict for a symbol.

        Returns the full picture: funding sentiment, OI magnitude,
        positioning, and basis — the four dimensions pros look at before
        taking a view.
        """
        fr  = self.get_funding_rate(symbol)
        oi  = self.get_open_interest(symbol)
        lsr = self.get_long_short_ratio(symbol)
        mp  = self.get_mark_premium(symbol)

        return {
            "symbol": symbol.upper(),
            "funding": fr.to_dict() if fr else None,
            "open_interest": oi.to_dict() if oi else None,
            "long_short_ratio": lsr.to_dict() if lsr else None,
            "mark_premium": mp.to_dict() if mp else None,
            "timestamp": time.time(),
        }

    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "cached_keys": list(self._cache_ts.keys()),
            "cache_ages_s": {k: round(now - ts, 0) for k, ts in self._cache_ts.items()},
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_feed: Optional[AlphaSignalFeed] = None


def get_alpha_signals(timeout_s: float = _TIMEOUT_S) -> AlphaSignalFeed:
    global _feed
    if _feed is None:
        _feed = AlphaSignalFeed(timeout_s=timeout_s)
    return _feed


def reset_alpha_signals() -> None:
    global _feed
    _feed = None
