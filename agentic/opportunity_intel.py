"""D130: Opportunity Intelligence — cross-domain scanner for Kai.

Smartest operators don't just watch price charts. They scan the full
opportunity landscape: where is momentum building, what trend is just
emerging, which affiliate category is about to spike, what content gap
exists right now? This module synthesises every feed Kai has access to
and scores opportunities across four domains:

  financial   — futures positioning + sentiment + macro alignment
  content     — trending topics with high search interest and low supply
  affiliate   — product/category trends with monetisation potential
  trend_arb   — cross-market signal: what's moving in crypto also moves
                 in related markets (energy, tech, commodities)

Conviction scale (mirrors TrustLevel convention):
  0–2   NOISE        — too weak, ignore
  3–4   WATCH        — monitor, not actionable yet
  5–6   SPECULATIVE  — worth sizing small
  7–8   CONFIDENT    — well-supported, material position
  9–10  CONVICTION   — multiple independent signals aligned

Trust gating:
  All reads → OBSERVER (1) — read-only intelligence
  Scan outputs feed into strategy decisions but never act directly

Feature-flagged: FF_OPPORTUNITY_INTEL=true
Fail-open: every scan returns a result; empty/low-conviction on failures.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kai.opportunity_intel")

_TTL_FINANCIAL = 120.0    # alpha signals update frequently
_TTL_CONTENT   = 1800.0   # trend searches can be cached longer
_TTL_AFFILIATE = 3600.0   # affiliate trends are slow-moving
_TTL_TREND_ARB = 300.0    # macro cross-market 5-min refresh


# ── OpportunitySignal ──────────────────────────────────────────────────────────

@dataclass
class OpportunitySignal:
    domain: str          # "financial" | "content" | "affiliate" | "trend_arb"
    subject: str         # symbol, niche, topic, category
    conviction: int      # 0–10 (see scale above)
    time_horizon: str    # "immediate" | "hours" | "days" | "weeks"
    headline: str        # one-sentence human-readable rationale
    signals: List[str]   # contributing evidence bullets
    recommended_action: str   # verb phrase: "enter long", "create video", "promote X"
    evidence: Dict[str, Any]  # raw supporting data
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "subject": self.subject,
            "conviction": self.conviction,
            "conviction_label": self._conviction_label(),
            "time_horizon": self.time_horizon,
            "headline": self.headline,
            "signals": self.signals,
            "recommended_action": self.recommended_action,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
        }

    def _conviction_label(self) -> str:
        if self.conviction <= 2:  return "noise"
        if self.conviction <= 4:  return "watch"
        if self.conviction <= 6:  return "speculative"
        if self.conviction <= 8:  return "confident"
        return "conviction"


# ── Conviction scoring helpers ─────────────────────────────────────────────────

def _score_financial(
    funding_sentiment: Optional[str],
    ls_sentiment: Optional[str],
    fng_value: Optional[int],
    macro_tone: str,
    mark_premium_pct: Optional[float],
) -> tuple[int, str, List[str]]:
    """Score a financial opportunity. Returns (conviction, direction, signals)."""
    bull_points = 0
    bear_points = 0
    signals: List[str] = []

    # Funding rate: contrarian signal (extremely_long → bearish opportunity)
    if funding_sentiment in ("extremely_long", "crowded_long"):
        bear_points += 2
        signals.append(f"Funding {funding_sentiment} → crowd paying to be long (contrarian short setup)")
    elif funding_sentiment in ("extremely_short", "crowded_short"):
        bull_points += 2
        signals.append(f"Funding {funding_sentiment} → crowd paying to be short (contrarian long setup)")
    elif funding_sentiment in ("mild_long",):
        bear_points += 1
        signals.append("Funding mildly positive — slight long bias in market")
    elif funding_sentiment in ("mild_short",):
        bull_points += 1
        signals.append("Funding mildly negative — slight short bias in market")

    # L/S ratio: contrarian signal
    if ls_sentiment in ("extremely_crowded_long",):
        bear_points += 3
        signals.append("80%+ accounts long — exit liquidity crowded, reversal risk high")
    elif ls_sentiment in ("crowded_long",):
        bear_points += 1
        signals.append("L/S crowded long — watch for mean reversion")
    elif ls_sentiment in ("extremely_crowded_short",):
        bull_points += 3
        signals.append("20%+ accounts short — potential short squeeze fuel")
    elif ls_sentiment in ("crowded_short",):
        bull_points += 1
        signals.append("L/S crowded short — squeeze potential building")

    # Fear & Greed: contrarian at extremes, momentum in middle
    if fng_value is not None:
        if fng_value <= 15:
            bull_points += 2
            signals.append(f"F&G Extreme Fear ({fng_value}) — capitulation zone, contrarian buy")
        elif fng_value <= 25:
            bull_points += 1
            signals.append(f"F&G Fear ({fng_value}) — depressed sentiment, asymmetric upside")
        elif fng_value >= 85:
            bear_points += 2
            signals.append(f"F&G Extreme Greed ({fng_value}) — euphoria, late-cycle risk")
        elif fng_value >= 75:
            bear_points += 1
            signals.append(f"F&G Greed ({fng_value}) — sentiment stretched")

    # Macro tone alignment
    if macro_tone == "bullish":
        bull_points += 1
        signals.append("Macro tone bullish — gold/oil/DXY conditions supportive")
    elif macro_tone == "bearish":
        bear_points += 1
        signals.append("Macro tone bearish — risk-off environment")

    # Mark premium (basis)
    if mark_premium_pct is not None:
        if mark_premium_pct > 0.5:
            bull_points += 1
            signals.append(f"Mark premium +{mark_premium_pct:.2f}% (contango) — carry favours longs")
        elif mark_premium_pct < -0.3:
            bear_points += 1
            signals.append(f"Mark premium {mark_premium_pct:.2f}% (backwardation) — stress, futures discount")

    total = bull_points + bear_points
    if total == 0:
        return 0, "neutral", signals

    # Net conviction: winner takes the score, scaled to 0-10
    net = abs(bull_points - bear_points)
    conviction = min(10, round(net * 10 / max(total, 1)))
    direction = "long" if bull_points > bear_points else "short"
    return conviction, direction, signals


def _score_content(topic: str, tone: str, abstract: str) -> tuple[int, List[str]]:
    """Score a content opportunity based on trend signal strength."""
    signals: List[str] = []
    points = 0

    if tone == "bullish":
        points += 3
        signals.append(f"Topic '{topic}' trending bullish — positive sentiment")
    elif tone == "bearish":
        points += 2   # controversy also drives views
        signals.append(f"Topic '{topic}' controversy — high engagement potential")
    else:
        points += 1
        signals.append(f"Topic '{topic}' has neutral coverage — niche opportunity")

    # Length of abstract as crude proxy for coverage depth
    if len(abstract) > 200:
        points += 1
        signals.append("Rich search result — topic has traction")

    # Financial keywords in topic boost content-finance crossover appeal
    finance_kw = {"bitcoin", "crypto", "ethereum", "defi", "nft", "ai", "stock",
                  "market", "trading", "invest", "money", "earn", "passive"}
    topic_words = set(topic.lower().split())
    if topic_words & finance_kw:
        points += 2
        signals.append("Finance/crypto crossover topic — high monetisable audience")

    conviction = min(10, points)
    return conviction, signals


def _score_affiliate(category: str, tone: str) -> tuple[int, List[str]]:
    """Score an affiliate opportunity by category trend strength."""
    signals: List[str] = []
    points = 0

    # Check both individual keywords and full phrases
    high_value_kw = {"exchange", "broker", "vpn", "trading", "software",
                     "saas", "crypto", "defi", "course", "wallet"}
    high_value_phrases = {"hardware wallet", "ai tool"}
    cat_lower = category.lower()
    cat_words = set(cat_lower.split())

    if (cat_words & high_value_kw) or any(p in cat_lower for p in high_value_phrases):
        points += 3
        signals.append(f"Category '{category}' in high-commission tier (finance/software)")

    if tone == "bullish":
        points += 2
        signals.append("Category trending positively — buyer intent likely elevated")
    elif tone == "bearish":
        points += 1
        signals.append("Bearish coverage may suppress buyers — time opportunity carefully")

    conviction = min(10, points)
    return conviction, signals


# ── OpportunityIntelligence ────────────────────────────────────────────────────

class OpportunityIntelligence:
    """Synthesises all Kai feeds into cross-domain opportunity signals.

    Financial signals come from AlphaSignalFeed + MarketIntelligence.
    Content, affiliate, and trend-arb signals come from web_scout searches
    interpreted through the same tone classifier MarketIntelligence uses.

    All scans are fail-open: a network failure produces a low-conviction
    result rather than an exception.
    """

    def __init__(self) -> None:
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

    # ── Financial scan ─────────────────────────────────────────────────

    def scan_financial(self, symbol: str) -> OpportunitySignal:
        """Score the financial opportunity for a symbol.

        Combines futures positioning (alpha signals), crowd emotion (F&G),
        and macro alignment into a single conviction score.
        """
        cache_key = f"fin:{symbol.upper()}"
        cached = self._cached(cache_key, _TTL_FINANCIAL)
        if cached is not None:
            return cached

        funding_sent = ls_sent = None
        fng_value    = None
        macro_tone   = "neutral"
        premium_pct  = None
        evidence: Dict[str, Any] = {}

        try:
            try:
                from alpha_signals import get_alpha_signals
            except ImportError:
                from agentic.alpha_signals import get_alpha_signals  # type: ignore

            feed = get_alpha_signals()
            comp = feed.composite(symbol)
            evidence["alpha"] = comp

            if comp.get("funding"):
                funding_sent = comp["funding"].get("sentiment")
            if comp.get("long_short_ratio"):
                ls_sent = comp["long_short_ratio"].get("sentiment")
            if comp.get("mark_premium"):
                premium_pct = comp["mark_premium"].get("premium_pct")

        except Exception as exc:
            logger.debug("Alpha signals unavailable for %s: %s", symbol, exc)

        try:
            try:
                from market_intel import get_market_intel
            except ImportError:
                from agentic.market_intel import get_market_intel  # type: ignore

            intel = get_market_intel()
            fng = intel.get_fear_greed()
            if fng:
                fng_value = fng.value
                evidence["fear_greed"] = fng.to_dict()

            macro = intel.get_macro_context()
            macro_tone = macro.get("overall_tone", "neutral")
            evidence["macro"] = macro.get("overall_tone")

        except Exception as exc:
            logger.debug("Market intel unavailable for %s: %s", symbol, exc)

        conviction, direction, signals = _score_financial(
            funding_sent, ls_sent, fng_value, macro_tone, premium_pct
        )

        if conviction >= 7:
            time_horizon = "immediate"
        elif conviction >= 5:
            time_horizon = "hours"
        elif conviction >= 3:
            time_horizon = "days"
        else:
            time_horizon = "weeks"

        if direction == "long":
            action = f"consider long {symbol.upper()} — monitor entry"
        elif direction == "short":
            action = f"consider short {symbol.upper()} — monitor entry"
        else:
            action = f"hold / observe {symbol.upper()}"

        headline = (
            f"{symbol.upper()} financial conviction {conviction}/10 "
            f"({direction}) — {len(signals)} signals aligned"
        )

        result = OpportunitySignal(
            domain="financial",
            subject=symbol.upper(),
            conviction=conviction,
            time_horizon=time_horizon,
            headline=headline,
            signals=signals,
            recommended_action=action,
            evidence=evidence,
        )
        self._store(cache_key, result)
        logger.info("Financial scan %s: conviction=%d (%s)", symbol, conviction, direction)
        return result

    # ── Content scan ───────────────────────────────────────────────────

    def scan_content(self, topic: str) -> OpportunitySignal:
        """Detect a content creation opportunity for a topic.

        Uses web_scout to assess current search coverage and tone.
        Returns conviction + recommended video/article angle.
        """
        cache_key = f"content:{topic.lower()}"
        cached = self._cached(cache_key, _TTL_CONTENT)
        if cached is not None:
            return cached

        abstract = ""
        tone = "neutral"
        evidence: Dict[str, Any] = {}

        try:
            try:
                from web_scout import search as ws_search
            except ImportError:
                from agentic.web_scout import search as ws_search  # type: ignore
            from market_intel import _classify_tone

            query = f"{topic} trending 2025 content opportunity"
            sr = ws_search(query, max_results=5, autonomous=False)
            abstract = (sr.abstract or "")[:300]
            tone = _classify_tone(abstract)
            evidence = {"query": query, "abstract": abstract, "tone": tone}

        except Exception as exc:
            logger.debug("Content scan unavailable for '%s': %s", topic, exc)

        conviction, signals = _score_content(topic, tone, abstract)
        time_horizon = "days" if conviction >= 6 else "weeks"

        headline = (
            f"Content opportunity: '{topic}' — conviction {conviction}/10, "
            f"tone {tone}"
        )
        action = f"create content on '{topic}' — {tone} angle, publish within {time_horizon}"

        result = OpportunitySignal(
            domain="content",
            subject=topic,
            conviction=conviction,
            time_horizon=time_horizon,
            headline=headline,
            signals=signals,
            recommended_action=action,
            evidence=evidence,
        )
        self._store(cache_key, result)
        logger.info("Content scan '%s': conviction=%d", topic, conviction)
        return result

    # ── Affiliate scan ─────────────────────────────────────────────────

    def scan_affiliate(self, category: str) -> OpportunitySignal:
        """Detect an affiliate marketing opportunity for a product category.

        Searches for trending products and assesses commission tier.
        Finance / software / hardware wallet categories score highest.
        """
        cache_key = f"aff:{category.lower()}"
        cached = self._cached(cache_key, _TTL_AFFILIATE)
        if cached is not None:
            return cached

        tone = "neutral"
        abstract = ""
        evidence: Dict[str, Any] = {}

        try:
            try:
                from web_scout import search as ws_search
            except ImportError:
                from agentic.web_scout import search as ws_search  # type: ignore
            from market_intel import _classify_tone

            query = f"best affiliate programs {category} high commission 2025"
            sr = ws_search(query, max_results=5, autonomous=False)
            abstract = (sr.abstract or "")[:300]
            tone = _classify_tone(abstract)
            evidence = {"query": query, "abstract": abstract, "tone": tone}

        except Exception as exc:
            logger.debug("Affiliate scan unavailable for '%s': %s", category, exc)

        conviction, signals = _score_affiliate(category, tone)
        time_horizon = "weeks" if conviction >= 5 else "months"

        headline = (
            f"Affiliate opportunity: '{category}' — conviction {conviction}/10"
        )
        action = (
            f"research top affiliate programs in '{category}', "
            f"launch content funnel within {time_horizon}"
        )

        result = OpportunitySignal(
            domain="affiliate",
            subject=category,
            conviction=conviction,
            time_horizon=time_horizon,
            headline=headline,
            signals=signals,
            recommended_action=action,
            evidence=evidence,
        )
        self._store(cache_key, result)
        logger.info("Affiliate scan '%s': conviction=%d", category, conviction)
        return result

    # ── Trend arbitrage scan ───────────────────────────────────────────

    def scan_trend_arb(self, symbol: str) -> OpportunitySignal:
        """Cross-market trend arbitrage: what's moving crypto moves related markets.

        When BTC rallies, GPU demand rises, mining stocks move, energy demand
        spikes. When gold rallies alongside BTC, macro risk-off narrative
        dominates. This scan identifies cross-domain opportunities from a
        single crypto signal.
        """
        cache_key = f"arb:{symbol.upper()}"
        cached = self._cached(cache_key, _TTL_TREND_ARB)
        if cached is not None:
            return cached

        signals: List[str] = []
        evidence: Dict[str, Any] = {}
        conviction = 0

        try:
            try:
                from market_intel import get_market_intel, _classify_tone
            except ImportError:
                from agentic.market_intel import get_market_intel, _classify_tone  # type: ignore

            intel = get_market_intel()
            macro = intel.get_macro_context()
            topics = macro.get("topics", {})
            overall = macro.get("overall_tone", "neutral")
            evidence["macro"] = macro

            # Score cross-market alignment
            bullish_topics = [k for k, v in topics.items() if v.get("tone") == "bullish"]
            bearish_topics = [k for k, v in topics.items() if v.get("tone") == "bearish"]

            if len(bullish_topics) >= 3:
                conviction += 3
                signals.append(f"Macro alignment: {len(bullish_topics)}/5 macro topics bullish — risk-on across markets")
            elif len(bullish_topics) >= 2:
                conviction += 2
                signals.append(f"{len(bullish_topics)}/5 macro topics bullish — partial risk-on")

            if len(bearish_topics) >= 3:
                conviction += 2
                signals.append(f"{len(bearish_topics)}/5 macro topics bearish — risk-off pressure")

            # Gold + crypto divergence is actionable
            gold_tone = topics.get("gold", {}).get("tone", "neutral")
            if gold_tone == "bullish" and overall == "bullish":
                conviction += 2
                signals.append("Gold + crypto both bullish — safe-haven + risk-on confluence (rare, high value)")
            elif gold_tone == "bullish" and overall == "bearish":
                conviction += 1
                signals.append("Gold bullish while crypto bearish — flight to safety, watch BTC as late-cycle hedge")

            # Oil: high oil = energy cost for miners → bearish BTC, bullish energy stocks
            oil_tone = topics.get("oil", {}).get("tone", "neutral")
            if oil_tone == "bullish":
                signals.append("Oil trending up — mining energy costs rising, watch hash rate")

            # Fed rates: tightening = risk-off = crypto headwind
            fed_tone = topics.get("fed_rates", {}).get("tone", "neutral")
            if fed_tone == "bearish":
                conviction += 1
                signals.append("Fed policy bearish tone — tightening pressure on risk assets")
            elif fed_tone == "bullish":
                conviction += 1
                signals.append("Fed policy bullish tone — easing / pause supportive for risk assets")

        except Exception as exc:
            logger.debug("Trend arb scan unavailable for %s: %s", symbol, exc)

        conviction = min(10, conviction)
        time_horizon = "hours" if conviction >= 7 else "days"

        headline = (
            f"Cross-market trend arb for {symbol.upper()} — "
            f"conviction {conviction}/10, {len(signals)} macro signals"
        )
        action = (
            f"monitor {symbol.upper()} in context of macro alignment; "
            "size accordingly to cross-market narrative"
        )

        result = OpportunitySignal(
            domain="trend_arb",
            subject=symbol.upper(),
            conviction=conviction,
            time_horizon=time_horizon,
            headline=headline,
            signals=signals,
            recommended_action=action,
            evidence=evidence,
        )
        self._store(cache_key, result)
        logger.info("Trend arb %s: conviction=%d", symbol, conviction)
        return result

    # ── Full scan ──────────────────────────────────────────────────────

    def full_scan(
        self,
        symbol: str,
        content_topics: Optional[List[str]] = None,
        affiliate_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all domain scans and return a ranked opportunity report.

        Defaults: if no content/affiliate targets provided, derives them
        from the symbol (e.g. BTCUSD → "bitcoin investing tutorial",
        "hardware wallet").
        """
        coin = symbol.upper().replace("USD", "").replace("USDT", "")

        default_topics = content_topics or [
            f"{coin.lower()} price prediction 2025",
            f"{coin.lower()} investing beginner guide",
        ]
        default_categories = affiliate_categories or [
            "hardware wallet",
            "crypto exchange",
        ]

        financial  = self.scan_financial(symbol)
        trend_arb  = self.scan_trend_arb(symbol)
        content    = [self.scan_content(t) for t in default_topics]
        affiliate  = [self.scan_affiliate(c) for c in default_categories]

        all_signals: List[OpportunitySignal] = (
            [financial, trend_arb] + content + affiliate
        )
        all_signals.sort(key=lambda s: s.conviction, reverse=True)

        top = [s for s in all_signals if s.conviction >= 5]
        watchlist = [s for s in all_signals if 3 <= s.conviction < 5]

        return {
            "symbol": symbol.upper(),
            "timestamp": time.time(),
            "top_opportunities": [s.to_dict() for s in top],
            "watchlist": [s.to_dict() for s in watchlist],
            "all_signals": [s.to_dict() for s in all_signals],
            "max_conviction": max((s.conviction for s in all_signals), default=0),
        }

    def status(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "cached_keys": list(self._cache_ts.keys()),
            "cache_ages_s": {k: round(now - ts, 0) for k, ts in self._cache_ts.items()},
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_opp_intel: Optional[OpportunityIntelligence] = None


def get_opportunity_intel() -> OpportunityIntelligence:
    global _opp_intel
    if _opp_intel is None:
        _opp_intel = OpportunityIntelligence()
    return _opp_intel


def reset_opportunity_intel() -> None:
    global _opp_intel
    _opp_intel = None
