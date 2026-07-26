"""Tests for D128: Strategy Engine — agentic/strategy_engine.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from strategy_engine import (
    MomentumStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
    Signal,
    StrategyEngine,
    get_strategy_engine,
    reset_strategy_engine,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rising(n: int, start: float = 100.0, step: float = 1.0) -> list:
    return [start + i * step for i in range(n)]


def _falling(n: int, start: float = 100.0, step: float = 1.0) -> list:
    return [start - i * step for i in range(n)]


def _flat(n: int, price: float = 100.0) -> list:
    return [price] * n


# ── Signal dataclass ──────────────────────────────────────────────────────────

def test_signal_to_dict():
    s = Signal("BTCUSD", "buy", 0.8, "test", "reason", 50000.0)
    d = s.to_dict()
    assert d["symbol"] == "BTCUSD"
    assert d["action"] == "buy"
    assert d["confidence"] == pytest.approx(0.8, abs=0.001)
    assert d["strategy_name"] == "test"


# ── MomentumStrategy ──────────────────────────────────────────────────────────

class TestMomentum:
    def test_not_enough_data_returns_hold(self):
        s = MomentumStrategy(lookback=10)
        sig = s.evaluate("BTCUSD", [100.0] * 5)
        assert sig.action == "hold"

    def test_rising_prices_returns_buy(self):
        s = MomentumStrategy(lookback=5, threshold_pct=2.0)
        # 100 → 110 over 5 steps = 10% gain
        prices = _rising(10, start=100.0, step=2.0)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "buy"

    def test_falling_prices_returns_sell(self):
        s = MomentumStrategy(lookback=5, threshold_pct=2.0)
        prices = _falling(10, start=100.0, step=2.0)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "sell"

    def test_flat_prices_returns_hold(self):
        s = MomentumStrategy(lookback=5, threshold_pct=2.0)
        prices = _flat(15)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "hold"

    def test_confidence_scales_with_magnitude(self):
        s = MomentumStrategy(lookback=5, threshold_pct=2.0)
        # Small move: barely over threshold
        small = [100.0] * 5 + [102.5]   # 2.5% gain
        big   = [100.0] * 5 + [110.0]   # 10% gain
        sig_small = s.evaluate("BTCUSD", small)
        sig_big   = s.evaluate("BTCUSD", big)
        assert sig_big.confidence > sig_small.confidence

    def test_confidence_capped_at_one(self):
        s = MomentumStrategy(lookback=5, threshold_pct=1.0)
        prices = [100.0] * 5 + [200.0]   # 100% gain
        sig = s.evaluate("BTCUSD", prices)
        assert sig.confidence <= 1.0

    def test_zero_reference_price_returns_hold(self):
        s = MomentumStrategy(lookback=2)
        sig = s.evaluate("BTCUSD", [0.0, 0.0, 100.0])
        assert sig.action == "hold"


# ── MovingAverageCrossStrategy ────────────────────────────────────────────────

class TestMACross:
    def test_invalid_short_ge_long_raises(self):
        with pytest.raises(ValueError):
            MovingAverageCrossStrategy(short=20, long=5)

    def test_not_enough_data_returns_hold(self):
        s = MovingAverageCrossStrategy(short=5, long=20)
        sig = s.evaluate("BTCUSD", [100.0] * 10)
        assert sig.action == "hold"

    def test_bullish_cross_returns_buy(self):
        s = MovingAverageCrossStrategy(short=3, long=5)
        # Verified crossover: prev short(6.0) < prev long(7.0); now short(10.3) > now long(9.2)
        # prices = [10, 9, 8, 7, 6, 5, 20]
        # prev=[10,9,8,7,6,5]: short_prev=mean([7,6,5])=6.0, long_prev=mean([9,8,7,6,5])=7.0
        # curr=[10,9,8,7,6,5,20]: short_now=mean([6,5,20])=10.3, long_now=mean([8,7,6,5,20])=9.2
        prices = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 20.0]
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "buy"

    def test_bearish_cross_returns_sell(self):
        s = MovingAverageCrossStrategy(short=3, long=5)
        # Verified crossover: prev short(9.0) > prev long(8.0); now short(6.7) < now long(7.0)
        # prices = [5, 6, 7, 8, 9, 10, 1]
        # prev=[5,6,7,8,9,10]: short_prev=mean([8,9,10])=9.0, long_prev=mean([6,7,8,9,10])=8.0
        # curr=[5,6,7,8,9,10,1]: short_now=mean([9,10,1])=6.7, long_now=mean([7,8,9,10,1])=7.0
        prices = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0]
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "sell"

    def test_no_cross_returns_hold(self):
        s = MovingAverageCrossStrategy(short=3, long=5)
        # Steady rise — short always above long, no crossover
        prices = _rising(25, start=100.0, step=0.5)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "hold"

    def test_strategy_name(self):
        s = MovingAverageCrossStrategy()
        assert s.name == "ma_cross"


# ── RSIStrategy ───────────────────────────────────────────────────────────────

class TestRSI:
    def test_not_enough_data_returns_hold(self):
        s = RSIStrategy(period=14)
        sig = s.evaluate("BTCUSD", [100.0] * 5)
        assert sig.action == "hold"

    def test_oversold_returns_buy(self):
        s = RSIStrategy(period=5, oversold=30.0)
        # Strong falling prices → low RSI
        prices = _falling(20, start=100.0, step=3.0)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "buy"

    def test_overbought_returns_sell(self):
        s = RSIStrategy(period=5, overbought=70.0)
        # Strong rising prices → high RSI
        prices = _rising(20, start=100.0, step=3.0)
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "sell"

    def test_neutral_returns_hold(self):
        s = RSIStrategy(period=5)
        # Alternating prices → RSI near 50
        prices = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0]
        sig = s.evaluate("BTCUSD", prices)
        assert sig.action == "hold"

    def test_all_gains_returns_rsi_100(self):
        s = RSIStrategy(period=5)
        prices = _rising(10, start=100.0, step=1.0)
        sig = s.evaluate("BTCUSD", prices)
        # RSI = 100 → overbought → sell
        assert sig.action == "sell"

    def test_strategy_name(self):
        assert RSIStrategy().name == "rsi"


# ── StrategyEngine.evaluate ───────────────────────────────────────────────────

class TestEvaluate:
    def test_evaluate_returns_one_signal_per_strategy(self):
        engine = StrategyEngine(strategies=[MomentumStrategy(), RSIStrategy()])
        sigs = engine.evaluate("BTCUSD", _flat(30))
        assert len(sigs) == 2

    def test_evaluate_fail_open_on_strategy_error(self):
        bad = MomentumStrategy()
        bad.evaluate = MagicMock(side_effect=RuntimeError("boom"))
        engine = StrategyEngine(strategies=[bad, MomentumStrategy()])
        sigs = engine.evaluate("BTCUSD", _flat(30))
        assert len(sigs) == 2   # bad strategy → hold, good strategy → normal

    def test_evaluate_all_strategies_default(self):
        engine = StrategyEngine()
        assert len(engine._strategies) == 3


# ── StrategyEngine.consensus ──────────────────────────────────────────────────

class TestConsensus:
    def test_consensus_hold_when_no_strategies(self):
        engine = StrategyEngine(strategies=[])
        sig = engine.consensus("BTCUSD", [100.0])
        assert sig.action == "hold"

    def test_consensus_unanimous_buy(self):
        # Patch all strategies to return buy
        s1 = MomentumStrategy(lookback=2, threshold_pct=0.1)
        s2 = MomentumStrategy(lookback=2, threshold_pct=0.1)
        engine = StrategyEngine(strategies=[s1, s2])
        prices = [100.0, 100.0, 105.0]  # clear upward momentum
        sig = engine.consensus("BTCUSD", prices)
        assert sig.action == "buy"
        assert sig.confidence == pytest.approx(1.0)

    def test_consensus_majority_wins(self):
        s_buy  = MomentumStrategy(lookback=2, threshold_pct=0.1)
        s_hold = RSIStrategy(period=14)  # needs 15 prices — returns hold with short list
        engine = StrategyEngine(strategies=[s_buy, s_buy, s_hold])
        prices = [100.0, 100.0, 105.0]
        sig = engine.consensus("BTCUSD", prices)
        assert sig.action == "buy"

    def test_consensus_tie_returns_hold(self):
        s_buy  = MomentumStrategy(lookback=2, threshold_pct=0.1)
        s_sell = MomentumStrategy(lookback=2, threshold_pct=0.1)
        s_sell.evaluate = MagicMock(return_value=Signal("X", "sell", 0.8, "x", "x", 100.0))
        engine = StrategyEngine(strategies=[s_buy, s_sell])
        prices = [100.0, 100.0, 105.0]
        sig = engine.consensus("BTCUSD", prices)
        assert sig.action == "hold"

    def test_consensus_strategy_name(self):
        engine = StrategyEngine(strategies=[MomentumStrategy()])
        sig = engine.consensus("BTCUSD", _flat(30))
        assert sig.strategy_name == "consensus"


# ── StrategyEngine.auto_trade ─────────────────────────────────────────────────

class TestAutoTrade:
    def _engine_with_signal(self, action: str, confidence: float = 0.8) -> StrategyEngine:
        engine = StrategyEngine(strategies=[])
        engine.consensus = MagicMock(return_value=Signal(  # type: ignore[method-assign]
            "BTCUSD", action, confidence, "mock", "mock reason", 50000.0
        ))
        engine._check_trust = MagicMock()  # type: ignore[method-assign]
        return engine

    def test_auto_trade_hold_returns_hold(self):
        engine = self._engine_with_signal("hold")
        result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "hold"

    def test_auto_trade_low_confidence_returns_hold(self):
        engine = self._engine_with_signal("buy", confidence=0.3)
        result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "hold"

    def test_auto_trade_buy_opens_long(self):
        engine = self._engine_with_signal("buy")
        mock_trader = MagicMock()
        mock_pos = MagicMock()
        mock_pos.position_id = "pos-123"
        mock_trader.open_position.return_value = mock_pos
        with patch("paper_trader.get_paper_trader", return_value=mock_trader):
            result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "opened"
        assert result["position_id"] == "pos-123"
        mock_trader.open_position.assert_called_once_with("BTCUSD", "long", 1.0, 50000.0, "auto")

    def test_auto_trade_sell_closes_long(self):
        engine = self._engine_with_signal("sell")
        mock_trader = MagicMock()
        mock_trader.get_positions.return_value = [
            {"position_id": "pos-abc", "symbol": "BTCUSD", "side": "long"}
        ]
        mock_trade = MagicMock()
        mock_trade.trade_id = "trade-xyz"
        mock_trade.pnl = 500.0
        mock_trader.close_position.return_value = mock_trade
        with patch("paper_trader.get_paper_trader", return_value=mock_trader):
            result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "closed"
        assert "trade-xyz" in result["trade_ids"]

    def test_auto_trade_sell_no_open_positions(self):
        engine = self._engine_with_signal("sell")
        mock_trader = MagicMock()
        mock_trader.get_positions.return_value = []
        with patch("paper_trader.get_paper_trader", return_value=mock_trader):
            result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "no_position"

    def test_auto_trade_trust_denied(self):
        engine = StrategyEngine(strategies=[])
        engine.consensus = MagicMock(return_value=Signal(  # type: ignore[method-assign]
            "BTCUSD", "buy", 0.9, "mock", "mock", 50000.0
        ))
        engine._check_trust = MagicMock(side_effect=PermissionError("denied"))  # type: ignore[method-assign]
        result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "denied"

    def test_auto_trade_paper_trader_error_fail_open(self):
        engine = self._engine_with_signal("buy")
        with patch("paper_trader.get_paper_trader", side_effect=RuntimeError("boom")):
            result = engine.auto_trade("BTCUSD", [50000.0])
        assert result["action"] == "error"


# ── StrategyEngine.status ─────────────────────────────────────────────────────

def test_status_lists_strategy_names():
    engine = StrategyEngine(strategies=[MomentumStrategy(), RSIStrategy()])
    s = engine.status()
    assert "momentum" in s["strategies"]
    assert "rsi" in s["strategies"]
    assert s["strategy_count"] == 2


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_same_instance():
    reset_strategy_engine()
    e1 = get_strategy_engine()
    e2 = get_strategy_engine()
    assert e1 is e2
    reset_strategy_engine()


def test_reset_clears_singleton():
    reset_strategy_engine()
    e1 = get_strategy_engine()
    reset_strategy_engine()
    e2 = get_strategy_engine()
    assert e1 is not e2
    reset_strategy_engine()
