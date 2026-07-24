from __future__ import annotations

import hashlib
import hmac
import os
import time
import urllib.parse
from contextlib import suppress
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title="Broker Bridge", version="0.1.0")

BASE_URL  = os.getenv("BINANCE_BASE_URL",  "https://api.binance.com")
FAPI_URL  = os.getenv("BINANCE_FAPI_URL",  "https://fapi.binance.com")
API_KEY   = os.getenv("BINANCE_API_KEY",   "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
MODE      = os.getenv("BINANCE_MODE", "spot").lower()   # "spot" | "futures"

_start = time.time()
_req_count = 0
_err_count = 0


# ── Signing ──────────────────────────────────────────────────────────────────

def _sign_params(params: dict, secret: str) -> dict:
    qs = urllib.parse.urlencode(params)
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return {**params, "signature": sig}


# ── Transport helpers (mockable in tests) ────────────────────────────────────

async def _public_get(path: str, params: dict | None = None, base: str = "") -> Any:
    global _req_count, _err_count
    url = (base or BASE_URL) + path
    _req_count += 1
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        _err_count += 1
        raise HTTPException(status_code=502, detail=f"Binance unreachable: {exc}") from exc


async def _signed_get(path: str, params: dict | None = None, base: str = "") -> Any:
    if not API_KEY or not API_SECRET:
        raise HTTPException(status_code=503, detail="Binance API credentials not configured")
    global _req_count, _err_count
    url = (base or BASE_URL) + path
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    p["recvWindow"] = 5000
    signed = _sign_params(p, API_SECRET)
    _req_count += 1
    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"X-MBX-APIKEY": API_KEY}) as client:
            resp = await client.get(url, params=signed)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        _err_count += 1
        raise HTTPException(status_code=exc.response.status_code,
                            detail=f"Binance API error: {exc.response.text}") from exc
    except Exception as exc:
        _err_count += 1
        raise HTTPException(status_code=502, detail=f"Binance unreachable: {exc}") from exc


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mode": MODE, "api_configured": bool(API_KEY and API_SECRET)}


@app.get("/metrics")
async def metrics():
    return {
        "uptime_seconds": round(time.time() - _start, 1),
        "requests": _req_count,
        "errors": _err_count,
        "mode": MODE,
    }


@app.get("/ticker/{symbol}")
async def ticker_symbol(symbol: str):
    data = await _public_get("/api/v3/ticker/price", params={"symbol": symbol.upper()})
    return {"symbol": data["symbol"], "price": float(data["price"])}


@app.get("/ticker")
async def ticker_all(symbols: str = Query(default="")):
    """Return prices for a comma-separated list of symbols, or top BTC/ETH/BNB/SOL."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    ]
    results = []
    for sym in syms:
        with suppress(Exception):
            d = await _public_get("/api/v3/ticker/price", params={"symbol": sym})
            results.append({"symbol": d["symbol"], "price": float(d["price"])})
    return {"tickers": results}


@app.get("/balance")
async def balance():
    if MODE == "futures":
        raw = await _signed_get("/fapi/v2/balance", base=FAPI_URL)
        assets = [
            {"asset": b["asset"], "free": float(b["availableBalance"]),
             "total": float(b["balance"])}
            for b in raw
            if float(b["balance"]) > 0
        ]
    else:
        raw = await _signed_get("/api/v3/account")
        assets = [
            {"asset": b["asset"], "free": float(b["free"]), "locked": float(b["locked"]),
             "total": float(b["free"]) + float(b["locked"])}
            for b in raw.get("balances", [])
            if float(b["free"]) + float(b["locked"]) > 0
        ]
    return {"mode": MODE, "assets": assets}


@app.get("/positions")
async def positions():
    if MODE == "futures":
        raw = await _signed_get("/fapi/v2/positionRisk", base=FAPI_URL)
        pos = [
            {
                "symbol": p["symbol"],
                "size": float(p["positionAmt"]),
                "entry_price": float(p["entryPrice"]),
                "mark_price": float(p["markPrice"]),
                "unrealized_pnl": float(p["unRealizedProfit"]),
                "leverage": int(p.get("leverage", 1)),
                "side": "LONG" if float(p["positionAmt"]) > 0 else "SHORT",
            }
            for p in raw
            if abs(float(p["positionAmt"])) > 0
        ]
    else:
        # Spot: balances with non-trivial value, enriched with price
        raw = await _signed_get("/api/v3/account")
        non_stables = [
            b for b in raw.get("balances", [])
            if float(b["free"]) + float(b["locked"]) > 0
            and b["asset"] not in ("USDT", "BUSD", "USDC", "FDUSD", "TUSD")
        ]
        pos = []
        for b in non_stables:
            qty = float(b["free"]) + float(b["locked"])
            sym = b["asset"] + "USDT"
            price = None
            with suppress(Exception):
                d = await _public_get("/api/v3/ticker/price", params={"symbol": sym})
                price = float(d["price"])
            pos.append({
                "symbol": b["asset"],
                "size": qty,
                "price_usdt": price,
                "value_usdt": round(qty * price, 2) if price else None,
            })
    return {"mode": MODE, "positions": pos}


@app.get("/orders")
async def orders(symbol: str = Query(default="")):
    params = {}
    if symbol:
        params["symbol"] = symbol.upper()
    if MODE == "futures":
        raw = await _signed_get("/fapi/v1/openOrders", params=params, base=FAPI_URL)
    else:
        raw = await _signed_get("/api/v3/openOrders", params=params)
    result = [
        {
            "symbol": o["symbol"],
            "order_id": o["orderId"],
            "side": o["side"],
            "type": o["type"],
            "price": float(o["price"]),
            "qty": float(o["origQty"]),
            "filled": float(o["executedQty"]),
            "status": o["status"],
        }
        for o in raw
    ]
    return {"mode": MODE, "orders": result}


@app.get("/pnl/summary")
async def pnl_summary():
    if MODE != "futures":
        raise HTTPException(status_code=400, detail="PnL summary only available in futures mode")
    raw = await _signed_get("/fapi/v2/positionRisk", base=FAPI_URL)
    active = [p for p in raw if abs(float(p["positionAmt"])) > 0]
    total_pnl = sum(float(p["unRealizedProfit"]) for p in active)
    breakdown = [
        {"symbol": p["symbol"], "pnl": float(p["unRealizedProfit"])}
        for p in active
    ]
    return {"total_unrealized_pnl": round(total_pnl, 4), "positions": breakdown}


@app.get("/templates")
async def templates():
    """Pre-built monitor rule templates for common trading scenarios."""
    return {
        "templates": [
            {
                "id": "price-above",
                "name": "Price alert (above)",
                "description": "Alert when asset price rises above a threshold",
                "rule": {
                    "source": {"type": "http", "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "field": "price"},
                    "condition": {"op": "gt", "threshold": 100000},
                    "actions": [{"type": "notify", "message": "BTC above $100k"}, {"type": "tts", "text": "Bitcoin is above one hundred thousand dollars"}],
                    "interval": 60,
                    "cooldown": 3600,
                },
            },
            {
                "id": "price-below",
                "name": "Price alert (below)",
                "description": "Alert when asset price drops below a threshold",
                "rule": {
                    "source": {"type": "http", "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "field": "price"},
                    "condition": {"op": "lt", "threshold": 90000},
                    "actions": [{"type": "notify", "message": "BTC below $90k"}, {"type": "tts", "text": "Bitcoin dropped below ninety thousand dollars"}],
                    "interval": 60,
                    "cooldown": 3600,
                },
            },
            {
                "id": "price-drop-pct",
                "name": "Price drop % alert",
                "description": "Alert when asset drops more than N% since last check",
                "rule": {
                    "source": {"type": "http", "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", "field": "price"},
                    "condition": {"op": "decreased_pct", "threshold": 3},
                    "actions": [{"type": "notify", "message": "BTC dropped 3%"}, {"type": "tts", "text": "Bitcoin price dropped three percent"}],
                    "interval": 300,
                    "cooldown": 900,
                },
            },
            {
                "id": "whale-volume",
                "name": "Volume spike alert",
                "description": "Alert when 24hr quote volume spikes 50%",
                "rule": {
                    "source": {"type": "http", "url": "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", "field": "quoteVolume"},
                    "condition": {"op": "increased_pct", "threshold": 50},
                    "actions": [{"type": "notify", "message": "BTC volume spike"}, {"type": "tts", "text": "Bitcoin volume spike detected"}],
                    "interval": 600,
                    "cooldown": 1800,
                },
            },
            {
                "id": "futures-pnl-loss",
                "name": "Futures PnL loss alert",
                "description": "Alert when unrealized PnL falls below a threshold",
                "rule": {
                    "source": {"type": "http", "url": "http://broker-bridge:8034/pnl/summary", "field": "total_unrealized_pnl"},
                    "condition": {"op": "lt", "threshold": -100},
                    "actions": [{"type": "notify", "message": "Futures PnL below -$100"}, {"type": "tts", "text": "Warning: futures loss exceeds one hundred dollars"}],
                    "interval": 60,
                    "cooldown": 900,
                },
            },
        ]
    }


@app.get("/depth/{symbol}")
async def depth(symbol: str, limit: int = Query(default=20, ge=5, le=1000)):
    """Order book depth (bids + asks)."""
    data = await _public_get("/api/v3/depth", params={"symbol": symbol.upper(), "limit": limit})
    bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
    asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
    return {"symbol": symbol.upper(), "bids": bids, "asks": asks, "last_update_id": data.get("lastUpdateId")}


@app.get("/stats/24hr/{symbol}")
async def stats_24hr(symbol: str):
    """24-hour rolling window statistics."""
    data = await _public_get("/api/v3/ticker/24hr", params={"symbol": symbol.upper()})
    return {
        "symbol": data["symbol"],
        "price_change": float(data["priceChange"]),
        "price_change_pct": float(data["priceChangePercent"]),
        "high": float(data["highPrice"]),
        "low": float(data["lowPrice"]),
        "volume": float(data["volume"]),
        "quote_volume": float(data["quoteVolume"]),
        "open": float(data["openPrice"]),
        "close": float(data["lastPrice"]),
        "trades": int(data["count"]),
    }


@app.get("/trades/{symbol}")
async def recent_trades(symbol: str, limit: int = Query(default=20, ge=1, le=500)):
    """Most recent trades for a symbol."""
    data = await _public_get("/api/v3/trades", params={"symbol": symbol.upper(), "limit": limit})
    return {
        "symbol": symbol.upper(),
        "trades": [
            {
                "id": t["id"],
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "side": "BUY" if t["isBuyerMaker"] is False else "SELL",
                "time": t["time"],
            }
            for t in data
        ],
    }


@app.get("/futures/funding/{symbol}")
async def funding_rate(symbol: str):
    """Current and predicted funding rate (futures mode only)."""
    if MODE != "futures":
        raise HTTPException(status_code=400, detail="Funding rate only available in futures mode")
    data = await _public_get("/fapi/v1/premiumIndex", params={"symbol": symbol.upper()}, base=FAPI_URL)
    return {
        "symbol": data["symbol"],
        "mark_price": float(data["markPrice"]),
        "index_price": float(data["indexPrice"]),
        "funding_rate": float(data["lastFundingRate"]),
        "next_funding_time": data["nextFundingTime"],
    }


@app.get("/futures/openinterest/{symbol}")
async def open_interest(symbol: str):
    """Open interest for a futures symbol."""
    if MODE != "futures":
        raise HTTPException(status_code=400, detail="Open interest only available in futures mode")
    data = await _public_get("/fapi/v1/openInterest", params={"symbol": symbol.upper()}, base=FAPI_URL)
    return {
        "symbol": data["symbol"],
        "open_interest": float(data["openInterest"]),
        "time": data["time"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8034")))
