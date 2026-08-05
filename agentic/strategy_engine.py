"""D128: Strategy Engine — rule-based trading signals for Kai's paper trading.

Phase 3: Sustainability. Kai now has a brain for trading decisions.
Pluggable strategies evaluate price history and emit BUY/SELL/HOLD signals.
The consensus aggregator combines multiple strategies; auto_trade() acts on
consensus when trust level permits.

Trust gating:
    evaluate() / consensus()   → OBSERVER (1) — signal generation only
    auto_trade()               → AGENT (3)    — opens/closes paper positions

Feature-flagged: FF_STRATEGY_ENGINE=true
Fail-open: individual strategy errors return HOLD, never crash the engine.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kai.strategy_engine")

MIN_PRICES = 2   # minimum price history length for any strategy


# ── Signal ─────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    symbol: str
    action: str           # "buy" | "sell" | "hold"
    confidence: float     # 0.0–1.0
    strategy_name: str
    reason: str
    price: float          # latest price at signal time
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "strategy_name": self.strategy_name,
            "reason": self.reason,
            "price": self.price,
            "timestamp": self.timestamp,
        }


# ── Strategy base ──────────────────────────────────────────────────────────────

class Strategy:
    """Base class for all trading strategies."""

    name: str = "base"

    def evaluate(self, symbol: str, prices: List[float]) -> Signal:
        raise NotImplementedError

    def _hold(self, symbol: str, price: float, reason: str) -> Signal:
        return Signal(symbol=symbol, action="hold", confidence=0.0,
                      strategy_name=self.name, reason=reason, price=price)


# ── Momentum strategy ──────────────────────────────────────────────────────────

class MomentumStrategy(Strategy):
    """Buy when price is up >= threshold over lookback window; sell when down >= threshold.

    Confidence scales linearly with magnitude up to 2× threshold.
    """

    name = "momentum"

    def __init__(self, lookback: int = 10, threshold_pct: float = 2.0) -> None:
        self.lookback = lookback
        self.threshold_pct = threshold_pct

    def evaluate(self, symbol: str, prices: List[float]) -> Signal:
        if len(prices) < self.lookback + 1:
            return self._hold(symbol, prices[-1] if prices else 0.0,
                              f"not enough data (need {self.lookback + 1})")
        price_now = prices[-1]
        price_then = prices[-(self.lookback + 1)]
        if price_then <= 0:
            return self._hold(symbol, price_now, "zero reference price")
        pct = (price_now - price_then) / price_then * 100
        confidence = min(1.0, abs(pct) / (self.threshold_pct * 2))
        if pct >= self.threshold_pct:
            return Signal(symbol=symbol, action="buy", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"price up {pct:.2f}% over {self.lookback} periods",
                          price=price_now)
        if pct <= -self.threshold_pct:
            return Signal(symbol=symbol, action="sell", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"price down {abs(pct):.2f}% over {self.lookback} periods",
                          price=price_now)
        return self._hold(symbol, price_now,
                          f"momentum {pct:.2f}% within ±{self.threshold_pct}% band")


# ── Moving-average cross strategy ─────────────────────────────────────────────

class MovingAverageCrossStrategy(Strategy):
    """Buy when short MA crosses above long MA; sell when it crosses below."""

    name = "ma_cross"

    def __init__(self, short: int = 5, long: int = 20) -> None:
        if short >= long:
            raise ValueError(f"short ({short}) must be < long ({long})")
        self.short = short
        self.long = long

    @staticmethod
    def _ma(prices: List[float], n: int) -> Optional[float]:
        if len(prices) < n:
            return None
        return sum(prices[-n:]) / n

    def evaluate(self, symbol: str, prices: List[float]) -> Signal:
        if len(prices) < self.long + 1:
            return self._hold(symbol, prices[-1] if prices else 0.0,
                              f"not enough data (need {self.long + 1})")
        price = prices[-1]
        short_now = self._ma(prices, self.short)
        long_now = self._ma(prices, self.long)
        # Previous bar — use prices[:-1]
        prev = prices[:-1]
        short_prev = self._ma(prev, self.short)
        long_prev = self._ma(prev, self.long)

        if None in (short_now, long_now, short_prev, long_prev):
            return self._hold(symbol, price, "insufficient history for MA")

        cross_up = short_prev <= long_prev and short_now > long_now  # type: ignore[operator]
        cross_down = short_prev >= long_prev and short_now < long_now  # type: ignore[operator]

        spread_pct = abs(short_now - long_now) / long_now * 100  # type: ignore[operator]
        confidence = min(1.0, spread_pct / 2.0)

        if cross_up:
            return Signal(symbol=symbol, action="buy", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"MA{self.short} crossed above MA{self.long}",
                          price=price)
        if cross_down:
            return Signal(symbol=symbol, action="sell", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"MA{self.short} crossed below MA{self.long}",
                          price=price)
        direction = "above" if short_now > long_now else "below"  # type: ignore[operator]
        return self._hold(symbol, price,
                          f"MA{self.short} {direction} MA{self.long}, no cross")


# ── RSI strategy ───────────────────────────────────────────────────────────────

class RSIStrategy(Strategy):
    """Buy when RSI < oversold; sell when RSI > overbought.

    Uses simple (non-smoothed) RSI for simplicity and determinism.
    """

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _rsi(self, prices: List[float]) -> Optional[float]:
        if len(prices) < self.period + 1:
            return None
        recent = prices[-(self.period + 1):]
        gains = [max(0.0, recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        losses = [max(0.0, recent[i - 1] - recent[i]) for i in range(1, len(recent))]
        avg_gain = sum(gains) / self.period
        avg_loss = sum(losses) / self.period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def evaluate(self, symbol: str, prices: List[float]) -> Signal:
        price = prices[-1] if prices else 0.0
        rsi = self._rsi(prices)
        if rsi is None:
            return self._hold(symbol, price,
                              f"not enough data (need {self.period + 1})")
        if rsi < self.oversold:
            confidence = min(1.0, (self.oversold - rsi) / self.oversold)
            return Signal(symbol=symbol, action="buy", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"RSI={rsi:.1f} below oversold threshold {self.oversold}",
                          price=price)
        if rsi > self.overbought:
            confidence = min(1.0, (rsi - self.overbought) / (100.0 - self.overbought))
            return Signal(symbol=symbol, action="sell", confidence=confidence,
                          strategy_name=self.name,
                          reason=f"RSI={rsi:.1f} above overbought threshold {self.overbought}",
                          price=price)
        return self._hold(symbol, price, f"RSI={rsi:.1f} in neutral zone")


# ── Strategy Engine ────────────────────────────────────────────────────────────

_DEFAULT_STRATEGIES: List[Strategy] = [
    MomentumStrategy(lookback=10, threshold_pct=2.0),
    MovingAverageCrossStrategy(short=5, long=20),
    RSIStrategy(period=14),
]


class StrategyEngine:
    """Runs multiple strategies and aggregates their signals.

    Trust: evaluate/consensus = OBSERVER (1).

    UH-INV-02: this class is a Proposal Specialist only. It produces
    an ActionProposal dict; it does not execute trades. Execution is the
    paper_trader's (Actuator) responsibility, gated by the approval path.
    """

    def __init__(self, strategies: Optional[List[Strategy]] = None) -> None:
        self._strategies = strategies if strategies is not None else list(_DEFAULT_STRATEGIES)

    # ── Public API ─────────────────────────────────────────────────────

    def evaluate(self, symbol: str, prices: List[float]) -> List[Signal]:
        """Run all strategies. Returns one Signal per strategy. Trust: OBSERVER."""
        results: List[Signal] = []
        for strategy in self._strategies:
            try:
                results.append(strategy.evaluate(symbol, prices))
            except Exception as exc:
                logger.debug("Strategy %s failed (fail-open): %s", strategy.name, exc)
                if prices:
                    results.append(strategy._hold(symbol, prices[-1], f"error: {exc}"))
        return results

    def consensus(self, symbol: str, prices: List[float]) -> Signal:
        """Aggregate strategy signals by majority vote.

        Returns the majority action. On a tie, HOLD wins.
        Confidence = fraction of strategies that agree on the winning action.
        Trust: OBSERVER.
        """
        signals = self.evaluate(symbol, prices)
        if not signals:
            price = prices[-1] if prices else 0.0
            return Signal(symbol=symbol, action="hold", confidence=0.0,
                          strategy_name="consensus", reason="no strategies available", price=price)

        counts: Dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        for s in signals:
            counts[s.action] = counts.get(s.action, 0) + 1

        # Majority: strictly more than half. Tie → hold.
        total = len(signals)
        winner = "hold"
        for action in ("buy", "sell"):
            if counts[action] > total / 2:
                winner = action
                break

        confidence = counts[winner] / total
        reasons = [s.reason for s in signals if s.action == winner]
        price = signals[-1].price
        return Signal(symbol=symbol, action=winner, confidence=round(confidence, 4),
                      strategy_name="consensus",
                      reason=f"{winner.upper()} ({counts[winner]}/{total}): " + "; ".join(reasons[:2]),
                      price=price)

    def generate_proposal(
        self,
        symbol: str,
        prices: List[float],
        quantity: float = 1.0,
        strategy_tag: str = "auto",
    ) -> Dict[str, Any]:
        """Produce an ActionProposal from the consensus signal.

        Returns a proposal dict. Does NOT execute any trade — that is the
        Actuator's (paper_trader) responsibility after the approval path.
        The caller must route this through the Workspace → Policy → Capability
        chain before any execution occurs.

        Trust: OBSERVER (1) — read-only signal production.
        """
        signal = self.consensus(symbol, prices)
        proposal_type = "no_action"
        if signal.action == "buy" and signal.confidence >= 0.5:
            proposal_type = "open_long"
        elif signal.action == "sell" and signal.confidence >= 0.5:
            proposal_type = "close_long"

        return {
            "proposal_type": proposal_type,
            "symbol": symbol,
            "quantity": quantity,
            "strategy_tag": strategy_tag,
            "signal": signal.to_dict(),
            "requires_approval": proposal_type != "no_action",
        }

    def status(self) -> Dict[str, Any]:
        return {
            "strategies": [s.name for s in self._strategies],
            "strategy_count": len(self._strategies),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
_engine: Optional[StrategyEngine] = None


def get_strategy_engine(strategies: Optional[List[Strategy]] = None) -> StrategyEngine:
    global _engine
    if _engine is None:
        _engine = StrategyEngine(strategies=strategies)
    return _engine


def reset_strategy_engine() -> None:
    global _engine
    _engine = None
