"""metrics-gateway — unified metrics aggregator.

Collects /metrics and /health from all sovereign AI services and exposes
a single aggregated endpoint for Prometheus scraping and dashboard display.

Features:
  - Parallel health sweep across all services
  - Aggregated error-budget snapshots
  - Prometheus-compatible /metrics/text output
  - Service discovery (auto-detect from env or defaults)
  - Historical uptime tracking
"""
from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from common.runtime import ErrorBudget, setup_json_logger

logger = setup_json_logger("metrics-gateway", os.getenv("LOG_PATH", "/tmp/metrics-gateway.json.log"))

app = FastAPI(title="Metrics Gateway", version="0.5.0")

SCRAPE_TIMEOUT = float(os.getenv("METRICS_SCRAPE_TIMEOUT", "5.0"))
SCRAPE_INTERVAL = int(os.getenv("METRICS_SCRAPE_INTERVAL", "30"))
budget = ErrorBudget(window_seconds=600)

# ── service registry ──────────────────────────────────────────────────
# Each entry: (name, url, metrics_path, health_path)
DEFAULT_SERVICES: List[Tuple[str, str, str, str]] = [
    ("tool-gate", "http://tool-gate:8000", "/metrics", "/health"),
    ("memu-core", "http://memu-core:8001", "/metrics", "/health"),
    ("executor", "http://executor:8002", "/metrics", "/health"),
    ("langgraph", "http://langgraph:8007", "/metrics", "/health"),
    ("heartbeat", "http://heartbeat:8010", "/metrics", "/health"),
    ("supervisor", "http://supervisor:8051", "/metrics", "/health"),
    ("verifier", "http://verifier:8052", "/metrics", "/health"),
    ("fusion-engine", "http://fusion-engine:8053", "/metrics", "/health"),
    ("dashboard", "http://dashboard:8080", "/health", "/health"),
    ("memory-compressor", "http://memory-compressor:8057", "/metrics", "/health"),
    ("ledger-worker", "http://ledger-worker:8056", "/metrics", "/health"),
]


def _build_registry() -> List[Tuple[str, str, str, str]]:
    """Build service registry from env overrides or defaults."""
    custom = os.getenv("METRICS_SERVICES")
    if custom:
        # format: name=url,name=url,...
        entries = []
        for pair in custom.split(","):
            if "=" in pair:
                name, url = pair.strip().split("=", 1)
                entries.append((name.strip(), url.strip(), "/metrics", "/health"))
        return entries if entries else DEFAULT_SERVICES
    return DEFAULT_SERVICES


SERVICES = _build_registry()

# ── cached state ──────────────────────────────────────────────────────
_latest_metrics: Dict[str, Dict[str, Any]] = {}
_latest_health: Dict[str, Dict[str, Any]] = {}
_uptime_history: Dict[str, List[bool]] = defaultdict(list)  # rolling window
_last_scrape: float = 0.0
_scraper_task: asyncio.Task | None = None

UPTIME_WINDOW = 100  # keep last 100 checks


async def _scrape_one(client: httpx.AsyncClient, name: str, url: str,
                      metrics_path: str, health_path: str) -> Tuple[
                          str, Optional[Dict], Optional[Dict]]:
    """Scrape a single service for health and metrics."""
    health_data: Optional[Dict] = None
    metrics_data: Optional[Dict] = None

    try:
        resp = await client.get(f"{url}{health_path}")
        if resp.status_code == 200:
            health_data = resp.json()
            health_data["_reachable"] = True
        else:
            health_data = {"_reachable": False, "_status_code": resp.status_code}
    except Exception as exc:
        health_data = {"_reachable": False, "_error": str(exc)}

    if metrics_path != health_path:
        try:
            resp = await client.get(f"{url}{metrics_path}")
            if resp.status_code == 200:
                metrics_data = resp.json()
        except Exception:
            metrics_data = None

    return name, health_data, metrics_data


async def scrape_all() -> Dict[str, Any]:
    """Scrape all registered services in parallel."""
    global _last_scrape
    started = time.time()

    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT) as client:
        tasks = [
            _scrape_one(client, name, url, mp, hp)
            for name, url, mp, hp in SERVICES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    reachable = 0
    total = len(SERVICES)

    for item in results:
        if isinstance(item, Exception):
            continue
        name, health_data, metrics_data = item
        if health_data:
            _latest_health[name] = health_data
            up = health_data.get("_reachable", False)
            _uptime_history[name].append(up)
            if len(_uptime_history[name]) > UPTIME_WINDOW:
                _uptime_history[name] = _uptime_history[name][-UPTIME_WINDOW:]
            if up:
                reachable += 1
        if metrics_data:
            _latest_metrics[name] = metrics_data

    _last_scrape = time.time()
    duration_ms = int((_last_scrape - started) * 1000)
    logger.info("Scrape complete: %d/%d reachable in %dms", reachable, total, duration_ms)

    return {
        "status": "ok",
        "reachable": reachable,
        "total": total,
        "duration_ms": duration_ms,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _compute_uptime(name: str) -> float:
    """Compute uptime percentage from rolling window."""
    history = _uptime_history.get(name, [])
    if not history:
        return 0.0
    return round(sum(1 for x in history if x) / len(history), 4)


async def _scraper_loop() -> None:
    """Background scraper loop."""
    logger.info("Background metrics scraper started (interval=%ds)", SCRAPE_INTERVAL)
    while True:
        try:
            await scrape_all()
        except Exception as exc:
            logger.error("Scraper loop error: %s", exc)
        await asyncio.sleep(SCRAPE_INTERVAL)


# ── HTTP endpoints ────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    global _scraper_task
    if os.getenv("METRICS_SCRAPE_ENABLED", "true").lower() == "true":
        _scraper_task = asyncio.create_task(_scraper_loop())


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "metrics-gateway",
        "scraper": "active" if _scraper_task and not _scraper_task.done() else "inactive",
    }


@app.get("/metrics")
async def aggregated_metrics() -> Dict[str, Any]:
    """Return aggregated metrics from all services."""
    return {
        "status": "ok",
        "last_scrape": datetime.fromtimestamp(_last_scrape).isoformat() if _last_scrape else None,
        "services": {
            name: {
                "health": _latest_health.get(name, {}),
                "metrics": _latest_metrics.get(name, {}),
                "uptime": _compute_uptime(name),
            }
            for name, *_ in SERVICES
        },
        "gateway": budget.snapshot(),
    }


@app.get("/metrics/text", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Prometheus-compatible text format output."""
    lines: List[str] = [
        "# HELP sovereign_service_up Whether service is reachable (1=up, 0=down)",
        "# TYPE sovereign_service_up gauge",
    ]
    for name, *_ in SERVICES:
        h = _latest_health.get(name, {})
        up = 1 if h.get("_reachable") else 0
        lines.append(f'sovereign_service_up{{service="{name}"}} {up}')

    lines.extend([
        "",
        "# HELP sovereign_service_uptime_ratio Rolling uptime ratio",
        "# TYPE sovereign_service_uptime_ratio gauge",
    ])
    for name, *_ in SERVICES:
        uptime = _compute_uptime(name)
        lines.append(f'sovereign_service_uptime_ratio{{service="{name}"}} {uptime}')

    # per-service error budget metrics
    lines.extend([
        "",
        "# HELP sovereign_error_rate Per-service error rate from /metrics",
        "# TYPE sovereign_error_rate gauge",
    ])
    for name, *_ in SERVICES:
        m = _latest_metrics.get(name, {})
        err_rate = m.get("error_rate", 0.0)
        lines.append(f'sovereign_error_rate{{service="{name}"}} {err_rate}')

    lines.append("")
    return "\n".join(lines)


@app.post("/scrape")
async def trigger_scrape() -> Dict[str, Any]:
    """Manually trigger a scrape of all services."""
    return await scrape_all()


@app.get("/fleet")
async def fleet_status() -> Dict[str, Any]:
    """Fleet overview — health + uptime of all services."""
    fleet = {}
    for name, url, *_ in SERVICES:
        h = _latest_health.get(name, {})
        fleet[name] = {
            "url": url,
            "reachable": h.get("_reachable", False),
            "uptime": _compute_uptime(name),
            "status": h.get("status", "unknown"),
            "last_check": datetime.fromtimestamp(_last_scrape).isoformat() if _last_scrape else None,
        }
    return {"status": "ok", "fleet": fleet, "total": len(SERVICES)}


@app.get("/registry")
async def service_registry() -> List[Dict[str, str]]:
    """List all registered services."""
    return [
        {"name": name, "url": url, "metrics_path": mp, "health_path": hp}
        for name, url, mp, hp in SERVICES
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8058")))
