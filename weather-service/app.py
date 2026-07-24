"""Weather service — current conditions + forecast via OpenMeteo (no API key).

Endpoints:
  GET /health     → {status, location, uptime_seconds}
  GET /metrics    → error budget snapshot
  GET /current    → {temp_c, feels_like_c, wind_kph, wind_dir_deg, weathercode, description, is_day}
  GET /forecast   → [{date, temp_max_c, temp_min_c, precipitation_mm, rain_prob_pct, weathercode}] (7 days)
  GET /summary    → one-sentence human-readable summary for agentic context injection
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
    logger = setup_json_logger("weather-service", os.getenv("LOG_PATH", "/tmp/weather-service.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("weather-service")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

PORT = int(os.getenv("PORT", "8039"))
LAT  = float(os.getenv("WEATHER_LAT", "51.5074"))   # default: London
LON  = float(os.getenv("WEATHER_LON", "-0.1278"))
LOCATION_NAME = os.getenv("WEATHER_LOCATION", "London")
REFRESH_INTERVAL = int(os.getenv("WEATHER_REFRESH_SECONDS", "600"))  # 10 min

BASE_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes → human description
_WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "icy fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    77: "snow grains",
    80: "light showers", 81: "moderate showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

_start = time.time()
_cache: dict = {}
_last_fetch: float = 0.0
_fetch_error: Optional[str] = None
_refresh_task: Optional[asyncio.Task] = None


async def _fetch_weather() -> dict:
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current_weather": "true",
        "hourly": "apparent_temperature,precipitation_probability",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weathercode",
        "timezone": "auto",
        "forecast_days": 7,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(BASE_URL, params=params)
        resp.raise_for_status()
        return resp.json()


async def _refresh_loop():
    global _cache, _last_fetch, _fetch_error
    while True:
        try:
            data = await _fetch_weather()
            _cache = data
            _last_fetch = time.time()
            _fetch_error = None
            logger.info("weather refreshed for %s (%.4f, %.4f)", LOCATION_NAME, LAT, LON)
        except Exception as exc:
            _fetch_error = str(exc)
            logger.warning("weather fetch error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _refresh_task
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()


app = FastAPI(title="weather-service", version="0.1.0", lifespan=_lifespan)


def _wmo_desc(code: int) -> str:
    return _WMO_CODES.get(code, f"code {code}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "location": LOCATION_NAME,
        "lat": LAT,
        "lon": LON,
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
        raise HTTPException(503, "weather data not yet available")
    cw = _cache.get("current_weather", {})
    code = int(cw.get("weathercode", 0))
    return {
        "temp_c": cw.get("temperature"),
        "wind_kph": round(cw.get("windspeed", 0) * 1.0, 1),
        "wind_dir_deg": cw.get("winddirection"),
        "weathercode": code,
        "description": _wmo_desc(code),
        "is_day": bool(cw.get("is_day", 1)),
        "location": LOCATION_NAME,
        "fetched_at": _last_fetch,
    }


@app.get("/forecast")
def forecast():
    if not _cache:
        raise HTTPException(503, "weather data not yet available")
    daily = _cache.get("daily", {})
    dates = daily.get("time", [])
    result = []
    for i, date in enumerate(dates):
        code = int((daily.get("weathercode") or [])[i] or 0) if i < len(daily.get("weathercode") or []) else 0
        result.append({
            "date": date,
            "temp_max_c": (daily.get("temperature_2m_max") or [])[i] if i < len(daily.get("temperature_2m_max") or []) else None,
            "temp_min_c": (daily.get("temperature_2m_min") or [])[i] if i < len(daily.get("temperature_2m_min") or []) else None,
            "precipitation_mm": (daily.get("precipitation_sum") or [])[i] if i < len(daily.get("precipitation_sum") or []) else None,
            "rain_prob_pct": (daily.get("precipitation_probability_max") or [])[i] if i < len(daily.get("precipitation_probability_max") or []) else None,
            "weathercode": code,
            "description": _wmo_desc(code),
        })
    return {"location": LOCATION_NAME, "forecast": result}


@app.get("/summary")
def summary():
    """One-sentence summary for agentic context injection."""
    if not _cache:
        return {"summary": f"Weather data for {LOCATION_NAME} is loading."}
    cw = _cache.get("current_weather", {})
    daily = _cache.get("daily", {})
    temp = cw.get("temperature")
    code = int(cw.get("weathercode", 0))
    desc = _wmo_desc(code)
    rain_prob = None
    if daily.get("precipitation_probability_max"):
        rain_prob = daily["precipitation_probability_max"][0]
    parts = [f"In {LOCATION_NAME}: {desc}, {temp}°C"]
    if rain_prob is not None:
        parts.append(f"{rain_prob}% chance of rain today")
    return {"summary": ". ".join(parts) + "."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
