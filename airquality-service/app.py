"""Air quality service — PM2.5, O3, NO2, UV index via OpenMeteo AQ API (no key needed).

Endpoints:
  GET /health    → {status, location, uptime_seconds}
  GET /metrics   → error budget
  GET /current   → {pm2_5, pm10, o3, no2, uv_index, aqi_category, fetched_at}
  GET /summary   → one-sentence summary for agentic context injection
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("airquality-service", os.getenv("LOG_PATH", "/tmp/airquality-service.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("airquality-service")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

PORT = int(os.getenv("PORT", "8042"))
LAT  = float(os.getenv("WEATHER_LAT", "51.5074"))
LON  = float(os.getenv("WEATHER_LON", "-0.1278"))
LOCATION_NAME = os.getenv("WEATHER_LOCATION", "London")
REFRESH_INTERVAL = int(os.getenv("AQ_REFRESH_SECONDS", "1800"))  # 30 min

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

_start = time.time()
_cache: dict = {}
_last_fetch: float = 0.0
_fetch_error: Optional[str] = None
_refresh_task: Optional[asyncio.Task] = None


def _aqi_category(pm2_5: Optional[float]) -> str:
    if pm2_5 is None:
        return "unknown"
    if pm2_5 <= 12:
        return "good"
    if pm2_5 <= 35.4:
        return "moderate"
    if pm2_5 <= 55.4:
        return "unhealthy for sensitive groups"
    if pm2_5 <= 150.4:
        return "unhealthy"
    if pm2_5 <= 250.4:
        return "very unhealthy"
    return "hazardous"


async def _fetch_aq() -> dict:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "pm2_5,pm10,ozone,nitrogen_dioxide,uv_index",
        "timezone": "auto",
        "forecast_days": 1,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()


def _latest(data: dict, key: str) -> Optional[float]:
    values = (data.get("hourly") or {}).get(key, [])
    for v in reversed(values):
        if v is not None:
            return round(v, 2)
    return None


async def _refresh_loop():
    global _cache, _last_fetch, _fetch_error
    while True:
        try:
            data = await _fetch_aq()
            _cache = data
            _last_fetch = time.time()
            _fetch_error = None
            logger.info("air quality refreshed for %s", LOCATION_NAME)
        except Exception as exc:
            _fetch_error = str(exc)
            logger.warning("air quality fetch error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _refresh_task
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()


app = FastAPI(title="airquality-service", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "location": LOCATION_NAME,
        "last_fetch": _last_fetch,
        "fetch_error": _fetch_error,
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/current")
def current():
    if not _cache:
        raise HTTPException(503, "air quality data not yet available")
    pm2_5 = _latest(_cache, "pm2_5")
    pm10 = _latest(_cache, "pm10")
    o3 = _latest(_cache, "ozone")
    no2 = _latest(_cache, "nitrogen_dioxide")
    uv = _latest(_cache, "uv_index")
    return {
        "location": LOCATION_NAME,
        "pm2_5_ugm3": pm2_5,
        "pm10_ugm3": pm10,
        "o3_ugm3": o3,
        "no2_ugm3": no2,
        "uv_index": uv,
        "aqi_category": _aqi_category(pm2_5),
        "fetched_at": _last_fetch,
    }


@app.get("/summary")
def summary():
    if not _cache:
        return {"summary": f"Air quality data for {LOCATION_NAME} is loading."}
    pm2_5 = _latest(_cache, "pm2_5")
    uv = _latest(_cache, "uv_index")
    cat = _aqi_category(pm2_5)
    parts = [f"Air quality in {LOCATION_NAME}: {cat}"]
    if pm2_5 is not None:
        parts.append(f"PM2.5 {pm2_5} µg/m³")
    if uv is not None:
        uv_label = "low" if uv < 3 else "moderate" if uv < 6 else "high" if uv < 8 else "very high"
        parts.append(f"UV index {uv} ({uv_label})")
    return {"summary": ". ".join(parts) + "."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
