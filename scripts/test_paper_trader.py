"""Tests for D125: Paper Trading Engine — agentic/paper_trader.py."""
from __future__ import annotations

import json
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from paper_trader import (
    PaperTrader,
    Position,
    Trade,
    get_paper_trader,
    reset_paper_trader,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trader(tmp_path: Path | None = None) -> PaperTrader:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / "paper-trading"
    t = PaperTrader(data_dir=tmp_path)
    t._check_trust = lambda *a, **kw: None   # type: ignore[method-assign]
    return t


def _open(t: PaperTrader, symbol="BTCUSD", side="long", qty=1.0, price=50000.0, tag="") -> Position:
    return t.open_position(symbol, side, qty, price, tag)


# ── Validation ────────────────────────────────────────────────────────────────

def test_open_rejects_invalid_side(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(ValueError, match="side"):
        t.open_position("BTCUSD", "buy", 1.0, 50000.0)


def test_open_rejects_zero_quantity(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(ValueError, match="quantity"):
        t.open_position("BTCUSD", "long", 0.0, 50000.0)


def test_open_rejects_negative_quantity(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(ValueError, match="quantity"):
        t.open_position("BTCUSD", "long", -1.0, 50000.0)


def test_open_rejects_zero_price(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(ValueError, match="price"):
        t.open_position("BTCUSD", "long", 1.0, 0.0)


def test_open_rejects_empty_symbol(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(ValueError, match="symbol"):
        t.open_position("", "long", 1.0, 50000.0)


def test_close_rejects_unknown_position(tmp_path):
    t = _trader(tmp_path)
    with pytest.raises(KeyError):
        t.close_position("nonexistent-id", 50000.0)


def test_close_rejects_zero_price(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    with pytest.raises(ValueError, match="price"):
        t.close_position(pos.position_id, 0.0)


# ── open_position ─────────────────────────────────────────────────────────────

def test_open_returns_position(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    assert isinstance(pos, Position)
    assert pos.symbol == "BTCUSD"
    assert pos.side == "long"
    assert pos.quantity == 1.0
    assert pos.entry_price == 50000.0


def test_open_normalises_symbol_to_upper(tmp_path):
    t = _trader(tmp_path)
    pos = t.open_position("btcusd", "long", 1.0, 50000.0)
    assert pos.symbol == "BTCUSD"


def test_open_assigns_unique_ids(tmp_path):
    t = _trader(tmp_path)
    p1 = _open(t)
    p2 = _open(t)
    assert p1.position_id != p2.position_id


def test_open_stores_strategy_tag(tmp_path):
    t = _trader(tmp_path)
    pos = t.open_position("BTCUSD", "long", 1.0, 50000.0, strategy_tag="momentum")
    assert pos.strategy_tag == "momentum"


def test_open_adds_to_positions(tmp_path):
    t = _trader(tmp_path)
    assert len(t.get_positions()) == 0
    _open(t)
    assert len(t.get_positions()) == 1


def test_open_trust_denied_raises(tmp_path):
    t = PaperTrader(data_dir=tmp_path)
    t._check_trust = MagicMock(side_effect=PermissionError("denied"))  # type: ignore
    with pytest.raises(PermissionError):
        _open(t)


# ── close_position ────────────────────────────────────────────────────────────

def test_close_long_profit(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t, price=50000.0)
    trade = t.close_position(pos.position_id, 55000.0)
    assert trade.pnl == pytest.approx(5000.0)
    assert trade.pnl_pct == pytest.approx(10.0, abs=0.01)


def test_close_long_loss(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t, price=50000.0)
    trade = t.close_position(pos.position_id, 45000.0)
    assert trade.pnl == pytest.approx(-5000.0)
    assert trade.pnl_pct < 0


def test_close_short_profit(tmp_path):
    t = _trader(tmp_path)
    pos = t.open_position("BTCUSD", "short", 1.0, 50000.0)
    trade = t.close_position(pos.position_id, 45000.0)
    assert trade.pnl == pytest.approx(5000.0)


def test_close_short_loss(tmp_path):
    t = _trader(tmp_path)
    pos = t.open_position("BTCUSD", "short", 1.0, 50000.0)
    trade = t.close_position(pos.position_id, 55000.0)
    assert trade.pnl == pytest.approx(-5000.0)


def test_close_removes_from_positions(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    assert len(t.get_positions()) == 1
    t.close_position(pos.position_id, 55000.0)
    assert len(t.get_positions()) == 0


def test_close_records_duration(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    trade = t.close_position(pos.position_id, 55000.0)
    assert trade.duration_s >= 0.0


def test_close_trust_denied_raises(tmp_path):
    t = PaperTrader(data_dir=tmp_path)
    call_count = [0]
    def _trust(*a, **kw):
        call_count[0] += 1
        if call_count[0] > 1:   # allow open, deny close
            raise PermissionError("denied")
    t._check_trust = _trust  # type: ignore[method-assign]
    pos = _open(t)
    with pytest.raises(PermissionError):
        t.close_position(pos.position_id, 55000.0)


# ── mark_to_market ────────────────────────────────────────────────────────────

def test_mark_to_market_updates_unrealised(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t, symbol="BTCUSD", price=50000.0)
    result = t.mark_to_market({"BTCUSD": 52000.0})
    assert pos.position_id in result
    assert result[pos.position_id] == pytest.approx(2000.0)


def test_mark_to_market_short_position(tmp_path):
    t = _trader(tmp_path)
    pos = t.open_position("ETHUSD", "short", 2.0, 3000.0)
    result = t.mark_to_market({"ETHUSD": 2800.0})
    # short: (entry - mark) * qty = (3000 - 2800) * 2 = 400
    assert result[pos.position_id] == pytest.approx(400.0)


def test_mark_to_market_ignores_missing_symbols(tmp_path):
    t = _trader(tmp_path)
    _open(t, symbol="BTCUSD")
    result = t.mark_to_market({"ETHUSD": 3000.0})
    assert result == {}


def test_mark_to_market_ignores_zero_price(tmp_path):
    t = _trader(tmp_path)
    _open(t, symbol="BTCUSD")
    result = t.mark_to_market({"BTCUSD": 0.0})
    assert result == {}


# ── status ────────────────────────────────────────────────────────────────────

def test_status_empty(tmp_path):
    t = _trader(tmp_path)
    s = t.status()
    assert s["closed_trades"] == 0
    assert s["total_pnl"] == 0.0
    assert s["win_rate"] is None


def test_status_after_trades(tmp_path):
    t = _trader(tmp_path)
    pos1 = _open(t, price=50000.0)
    t.close_position(pos1.position_id, 55000.0)   # +5000 win
    pos2 = _open(t, price=50000.0)
    t.close_position(pos2.position_id, 48000.0)   # -2000 loss
    s = t.status()
    assert s["closed_trades"] == 2
    assert s["total_pnl"] == pytest.approx(3000.0)
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["best_trade"]["pnl"] == pytest.approx(5000.0)
    assert s["worst_trade"]["pnl"] == pytest.approx(-2000.0)


def test_status_open_position_count(tmp_path):
    t = _trader(tmp_path)
    _open(t)
    _open(t)
    s = t.status()
    assert s["open_positions"] == 2


# ── Persistence ───────────────────────────────────────────────────────────────

def test_positions_survive_reload(tmp_path):
    t1 = _trader(tmp_path)
    pos = _open(t1, symbol="BTCUSD", price=50000.0)
    # New instance — should load from disk
    t2 = PaperTrader(data_dir=tmp_path)
    t2._check_trust = lambda *a, **kw: None  # type: ignore[method-assign]
    loaded = t2.get_positions()
    assert len(loaded) == 1
    assert loaded[0]["position_id"] == pos.position_id


def test_trades_appended_to_jsonl(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    t.close_position(pos.position_id, 55000.0)
    f = tmp_path / "trades.jsonl"
    assert f.exists()
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["symbol"] == "BTCUSD"
    assert d["pnl"] == pytest.approx(5000.0)


def test_multiple_trades_all_in_jsonl(tmp_path):
    t = _trader(tmp_path)
    for _ in range(3):
        pos = _open(t)
        t.close_position(pos.position_id, 55000.0)
    f = tmp_path / "trades.jsonl"
    lines = [l for l in f.read_text().splitlines() if l.strip()]
    assert len(lines) == 3


# ── get_trades / get_positions ────────────────────────────────────────────────

def test_get_trades_returns_dicts(tmp_path):
    t = _trader(tmp_path)
    pos = _open(t)
    t.close_position(pos.position_id, 55000.0)
    trades = t.get_trades()
    assert len(trades) == 1
    assert "trade_id" in trades[0]


def test_get_trades_limit(tmp_path):
    t = _trader(tmp_path)
    for _ in range(5):
        pos = _open(t)
        t.close_position(pos.position_id, 55000.0)
    assert len(t.get_trades(limit=3)) == 3


def test_get_positions_returns_dicts(tmp_path):
    t = _trader(tmp_path)
    _open(t)
    positions = t.get_positions()
    assert len(positions) == 1
    assert "position_id" in positions[0]


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_same_instance(tmp_path):
    reset_paper_trader()
    t1 = get_paper_trader(tmp_path)
    t2 = get_paper_trader(tmp_path)
    assert t1 is t2
    reset_paper_trader()


def test_reset_clears_singleton(tmp_path):
    reset_paper_trader()
    t1 = get_paper_trader(tmp_path)
    reset_paper_trader()
    t2 = get_paper_trader(tmp_path)
    assert t1 is not t2
    reset_paper_trader()
