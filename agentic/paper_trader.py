"""D125: Paper Trading Engine — Kai's first financial autonomy layer.

Phase 3: Sustainability. Kai can simulate trades with zero real-money risk,
build a track record, and develop strategy intuition before real capital is
ever involved. The paper ledger is the proving ground — real trading only
unlocks when the paper record is strong and trust is at PARTNER+.

Trust gating:
    open_position / close_position  → PARTNER (4) — financial action
    mark_to_market / status         → OBSERVER (1) — read-only

Feature-flagged: FF_PAPER_TRADING=true
Fail-open: trust infra missing → fail-open (log warning, continue).
Storage:
    data/paper-trading/positions.json  — open positions (mutable)
    data/paper-trading/trades.jsonl    — closed trade log (append-only)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from common.degraded import record_degradation

logger = logging.getLogger("kai.paper_trader")

_DATA_DIR = Path("data/paper-trading")
_POSITIONS_FILE = _DATA_DIR / "positions.json"
_TRADES_FILE = _DATA_DIR / "trades.jsonl"

SIDES = {"long", "short"}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Position:
    position_id: str
    symbol: str
    side: str               # "long" | "short"
    quantity: float
    entry_price: float
    opened_at: float        # epoch seconds
    strategy_tag: str = ""  # optional label (e.g. "momentum", "mean_revert")
    unrealised_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def mark(self, current_price: float) -> float:
        """Compute unrealised P&L at the given mark price."""
        if self.side == "long":
            pnl = (current_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - current_price) * self.quantity
        self.unrealised_pnl = round(pnl, 6)
        return self.unrealised_pnl


@dataclass
class Trade:
    trade_id: str
    position_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float              # realised
    pnl_pct: float          # as % of entry cost
    opened_at: float
    closed_at: float
    strategy_tag: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Paper Trader ───────────────────────────────────────────────────────────────

class PaperTrader:
    """Simulates trades with no real capital at risk.

    All writes are fail-open — a broken disk never crashes the pipeline.
    Trust is enforced before any financial action.
    """

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._positions: Dict[str, Position] = {}
        self._load_positions()

    # ── Persistence ────────────────────────────────────────────────────

    def _load_positions(self) -> None:
        f = self._dir / "positions.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text())
            for entry in data.get("positions", []):
                try:
                    p = Position(**{k: v for k, v in entry.items()
                                    if k in Position.__dataclass_fields__})  # type: ignore
                    self._positions[p.position_id] = p
                except Exception as _exc:
                    record_degradation("filesystem", "load_paper_position", _exc)
        except Exception as exc:
            logger.debug("Paper trader position load failed: %s", exc)

    def _save_positions(self) -> None:
        try:
            payload = {"positions": [p.to_dict() for p in self._positions.values()]}
            tmp = self._dir / "positions.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._dir / "positions.json")
        except Exception as exc:
            logger.debug("Paper trader position save failed: %s", exc)

    def _append_trade(self, trade: Trade) -> None:
        try:
            f = self._dir / "trades.jsonl"
            with f.open("a") as fh:
                fh.write(json.dumps(trade.to_dict()) + "\n")
        except Exception as exc:
            logger.debug("Paper trader trade append failed: %s", exc)

    def _load_trades(self) -> List[Trade]:
        f = self._dir / "trades.jsonl"
        if not f.exists():
            return []
        trades = []
        try:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    trades.append(Trade(**{k: v for k, v in d.items()
                                           if k in Trade.__dataclass_fields__}))  # type: ignore
                except Exception as _exc:
                    record_degradation("filesystem", "load_paper_trade", _exc)
        except Exception as exc:
            logger.debug("Paper trader trade load failed: %s", exc)
        return trades

    # ── Trust gate ──────────────────────────────────────────────────────

    def _check_trust(self, capability: str, context: Dict[str, Any]) -> None:
        try:
            try:
                from trust_integration import gate_autonomous_action
            except ImportError:
                from agentic.trust_integration import gate_autonomous_action  # type: ignore
            allowed, reason = gate_autonomous_action(capability, context, conviction=7.0)
            if not allowed:
                raise PermissionError(f"Paper trader trust gate denied {capability}: {reason}")
        except PermissionError:
            raise
        except Exception as exc:
            logger.debug("Trust gate unavailable (fail-open for paper_trader): %s", exc)

    # ── Public API ─────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_tag: str = "",
    ) -> Position:
        """Open a simulated position.

        Trust: PARTNER (4).
        Raises: PermissionError if trust denied.
                ValueError on invalid arguments.
        """
        if side not in SIDES:
            raise ValueError(f"side must be 'long' or 'short', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol must not be empty")

        self._check_trust(
            "paper_trading_open",
            {"symbol": symbol, "side": side, "quantity": quantity, "price": price},
        )

        position = Position(
            position_id=str(uuid.uuid4()),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            opened_at=time.time(),
            strategy_tag=strategy_tag,
        )
        self._positions[position.position_id] = position
        self._save_positions()
        logger.info(
            "Paper trade OPEN: %s %s %.4f @ %.4f [%s]",
            side.upper(), symbol, quantity, price, strategy_tag or "untagged",
        )
        return position

    def close_position(self, position_id: str, price: float) -> Trade:
        """Close a simulated position and record the realised P&L.

        Trust: PARTNER (4).
        Raises: PermissionError if trust denied.
                KeyError if position_id not found.
                ValueError on invalid price.
        """
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        pos = self._positions.get(position_id)
        if pos is None:
            raise KeyError(f"No open position with id={position_id!r}")

        self._check_trust(
            "paper_trading_close",
            {"symbol": pos.symbol, "position_id": position_id, "price": price},
        )

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - price) * pos.quantity
        cost = pos.entry_price * pos.quantity
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
        now = time.time()

        trade = Trade(
            trade_id=str(uuid.uuid4()),
            position_id=position_id,
            symbol=pos.symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=round(pnl, 6),
            pnl_pct=round(pnl_pct, 4),
            opened_at=pos.opened_at,
            closed_at=now,
            strategy_tag=pos.strategy_tag,
            duration_s=round(now - pos.opened_at, 1),
        )

        del self._positions[position_id]
        self._save_positions()
        self._append_trade(trade)

        logger.info(
            "Paper trade CLOSE: %s %s pnl=%.4f (%.2f%%) duration=%.0fs",
            pos.side.upper(), pos.symbol, pnl, pnl_pct, trade.duration_s,
        )
        return trade

    def mark_to_market(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Update unrealised P&L for all open positions.

        Trust: OBSERVER (1) — read-only computation.
        prices: {symbol → current_price}
        Returns: {position_id → unrealised_pnl}
        """
        result: Dict[str, float] = {}
        for pid, pos in self._positions.items():
            current = prices.get(pos.symbol)
            if current is not None and current > 0:
                result[pid] = pos.mark(current)
        return result

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions as dicts. Trust: OBSERVER."""
        return [p.to_dict() for p in self._positions.values()]

    def get_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent closed trades. Trust: OBSERVER."""
        trades = self._load_trades()
        return [t.to_dict() for t in trades[-limit:]]

    def status(self) -> Dict[str, Any]:
        """Overall P&L stats across all closed trades."""
        trades = self._load_trades()
        if not trades:
            return {
                "open_positions": len(self._positions),
                "closed_trades": 0,
                "total_pnl": 0.0,
                "win_rate": None,
                "avg_pnl_per_trade": None,
                "best_trade": None,
                "worst_trade": None,
            }
        pnls = [t.pnl for t in trades]
        wins = [p for p in pnls if p > 0]
        best = max(trades, key=lambda t: t.pnl)
        worst = min(trades, key=lambda t: t.pnl)
        return {
            "open_positions": len(self._positions),
            "closed_trades": len(trades),
            "total_pnl": round(sum(pnls), 6),
            "win_rate": round(len(wins) / len(trades), 4) if trades else None,
            "avg_pnl_per_trade": round(sum(pnls) / len(trades), 6),
            "best_trade": {
                "symbol": best.symbol, "pnl": best.pnl, "pnl_pct": best.pnl_pct,
            },
            "worst_trade": {
                "symbol": worst.symbol, "pnl": worst.pnl, "pnl_pct": worst.pnl_pct,
            },
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_trader: Optional[PaperTrader] = None


def get_paper_trader(data_dir: Path = _DATA_DIR) -> PaperTrader:
    global _trader
    if _trader is None:
        _trader = PaperTrader(data_dir=data_dir)
    return _trader


def reset_paper_trader() -> None:
    global _trader
    _trader = None
