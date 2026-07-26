"""D127: Market Data Feed — real-time price discovery for Kai's paper trading engine.

Phase 3: Sustainability. Kai fetches cryptocurrency prices from public APIs
(CoinGecko, no key required) and marks open paper positions to market
automatically. This closes the paper-trading loop — positions now carry live
unrealised P&L without operator intervention.

Trust gating:
    get_price / get_prices / mark_positions / status → OBSERVER (1) — read-only
    (no writes, no financial actions — prices are information, not decisions)

Feature-flagged: FF_MARKET_DATA=true
Fail-open: API failures return empty dict, never crash.
Cache: in-memory only (prices are ephemeral; TTL default = 60 s).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("kai.market_data")

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_DEFAULT_TTL_S = 60.0
_DEFAULT_TIMEOUT_S = 5.0

# Symbol → CoinGecko coin-id mapping (all USD-quoted)
_SYMBOL_MAP: Dict[str, str] = {
    "BTCUSD":  "bitcoin",
    "ETHUSD":  "ethereum",
    "SOLUSD":  "solana",
    "ADAUSD":  "cardano",
    "DOTUSD":  "polkadot",
    "LINKUSD": "chainlink",
    "AVAXUSD": "avalanche-2",
    "MATICUSD":"matic-network",
    "UNIUSD":  "uniswap",
    "AAVEUSD": "aave",
    "BNBUSD":  "binancecoin",
    "XRPUSD":  "ripple",
    "LTCUSD":  "litecoin",
    "BCHUSD":  "bitcoin-cash",
    "DOGEUSD": "dogecoin",
}


# ── Price quote ────────────────────────────────────────────────────────────────

@dataclass
class PriceQuote:
    symbol: str
    price_usd: float
    fetched_at: float
    source: str = "coingecko"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price_usd": self.price_usd,
            "fetched_at": self.fetched_at,
            "source": self.source,
            "age_s": round(time.time() - self.fetched_at, 1),
        }


# ── Market Data Feed ───────────────────────────────────────────────────────────

class MarketDataFeed:
    """Fetches and caches real-time prices from public APIs.

    Trust: OBSERVER (1) — read-only.
    Fail-open: every network/parse error returns empty dict.
    """

    def __init__(
        self,
        ttl_s: float = _DEFAULT_TTL_S,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        symbol_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._ttl_s = ttl_s
        self._timeout_s = timeout_s
        self._symbol_map = symbol_map or _SYMBOL_MAP
        self._cache: Dict[str, PriceQuote] = {}

    # ── Cache helpers ──────────────────────────────────────────────────

    def _fresh(self, symbol: str) -> Optional[PriceQuote]:
        q = self._cache.get(symbol)
        if q is not None and (time.time() - q.fetched_at) < self._ttl_s:
            return q
        return None

    # ── Public API ─────────────────────────────────────────────────────

    def get_price(self, symbol: str) -> Optional[float]:
        """Return current USD price for symbol, or None if unavailable."""
        symbol = symbol.upper()
        cached = self._fresh(symbol)
        if cached:
            return cached.price_usd
        result = self.get_prices([symbol])
        return result.get(symbol)

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return {symbol: price_usd} for the given symbols.

        Serves fresh cache entries without a network call.
        Fetches the rest from CoinGecko in one batched request.
        Unknown symbols (not in the mapping) are silently skipped.
        """
        symbols = [s.upper() for s in symbols]
        result: Dict[str, float] = {}
        need_fetch: List[str] = []

        for sym in symbols:
            cached = self._fresh(sym)
            if cached:
                result[sym] = cached.price_usd
            elif sym in self._symbol_map:
                need_fetch.append(sym)
            else:
                logger.debug("No CoinGecko mapping for symbol: %s", sym)

        if need_fetch:
            fetched = self._fetch_coingecko(need_fetch)
            result.update(fetched)
        return result

    def mark_positions(self) -> Dict[str, float]:
        """Mark all open paper positions to market.

        Convenience wrapper: fetches prices for all open symbols, then
        calls paper_trader.mark_to_market(). Returns {position_id: unrealised_pnl}.
        Trust: OBSERVER — prices are read-only.
        """
        try:
            try:
                from paper_trader import get_paper_trader
            except ImportError:
                from agentic.paper_trader import get_paper_trader  # type: ignore
            trader = get_paper_trader()
            positions = trader.get_positions()
            if not positions:
                return {}
            symbols = list({p["symbol"] for p in positions})
            prices = self.get_prices(symbols)
            if not prices:
                return {}
            return trader.mark_to_market(prices)
        except Exception as exc:
            logger.debug("mark_positions failed (fail-open): %s", exc)
            return {}

    def status(self) -> Dict[str, Any]:
        """Return cache summary."""
        now = time.time()
        quotes = [
            {**q.to_dict(), "fresh": (now - q.fetched_at) < self._ttl_s}
            for q in self._cache.values()
        ]
        return {
            "cached_symbols": len(self._cache),
            "ttl_s": self._ttl_s,
            "quotes": sorted(quotes, key=lambda q: q["symbol"]),
        }

    def known_symbols(self) -> List[str]:
        """List of symbols Kai can fetch prices for."""
        return sorted(self._symbol_map.keys())

    # ── CoinGecko fetch ────────────────────────────────────────────────

    def _fetch_coingecko(self, symbols: List[str]) -> Dict[str, float]:
        """Batch-fetch from CoinGecko /simple/price. Updates cache. Fail-open."""
        coin_ids = [self._symbol_map[s] for s in symbols if s in self._symbol_map]
        if not coin_ids:
            return {}

        url = f"{_COINGECKO_BASE}/simple/price"
        params = {"ids": ",".join(coin_ids), "vs_currencies": "usd"}

        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("CoinGecko returned HTTP %d", resp.status_code)
                return {}
            data = resp.json()
        except Exception as exc:
            logger.debug("CoinGecko fetch failed (fail-open): %s", exc)
            return {}

        reverse_map = {v: k for k, v in self._symbol_map.items()}
        now = time.time()
        result: Dict[str, float] = {}

        for coin_id, prices in data.items():
            usd = prices.get("usd")
            if usd is None:
                continue
            sym = reverse_map.get(coin_id)
            if sym and sym in symbols:
                quote = PriceQuote(symbol=sym, price_usd=float(usd), fetched_at=now)
                self._cache[sym] = quote
                result[sym] = float(usd)

        if result:
            logger.info("CoinGecko: %d price(s) fetched %s", len(result), sorted(result))
        return result


# ── Singleton ──────────────────────────────────────────────────────────────────

_feed: Optional[MarketDataFeed] = None


def get_market_data(
    ttl_s: float = _DEFAULT_TTL_S,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> MarketDataFeed:
    global _feed
    if _feed is None:
        _feed = MarketDataFeed(ttl_s=ttl_s, timeout_s=timeout_s)
    return _feed


def reset_market_data() -> None:
    global _feed
    _feed = None
