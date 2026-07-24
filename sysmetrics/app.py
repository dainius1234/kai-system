"""Sysmetrics service — system health snapshot via psutil.

Endpoints:
  GET /health    → {status, uptime_seconds}
  GET /metrics   → error budget snapshot
  GET /snapshot  → {cpu, memory, disk, network, load_avg, uptime}
  GET /processes → [{pid, name, cpu_percent, mem_mb, status}] (top N by CPU)
"""
from __future__ import annotations

import os
import time

from fastapi import FastAPI

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("sysmetrics", os.getenv("LOG_PATH", "/tmp/sysmetrics.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("sysmetrics")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    logger.warning("psutil not available — sysmetrics in stub mode")

PORT = int(os.getenv("PORT", "8035"))
TOP_PROCESSES = int(os.getenv("TOP_PROCESSES", "20"))
_start = time.time()

app = FastAPI(title="sysmetrics", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "psutil": _PSUTIL_OK, "uptime_seconds": round(time.time() - _start, 1)}


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/snapshot")
def snapshot():
    if not _PSUTIL_OK:
        return {"error": "psutil not available", "stub": True}
    cpu = {
        "percent": psutil.cpu_percent(interval=0.2),
        "count_logical": psutil.cpu_count(logical=True),
        "count_physical": psutil.cpu_count(logical=False),
        "freq_mhz": round(psutil.cpu_freq().current, 1) if psutil.cpu_freq() else None,
    }
    vm = psutil.virtual_memory()
    memory = {
        "total_mb": round(vm.total / 1e6, 1),
        "used_mb": round(vm.used / 1e6, 1),
        "available_mb": round(vm.available / 1e6, 1),
        "percent": vm.percent,
    }
    disk_parts = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disk_parts.append({
                "mountpoint": part.mountpoint,
                "total_gb": round(usage.total / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2),
                "percent": usage.percent,
            })
        except PermissionError:
            pass
    net_io = psutil.net_io_counters()
    network = {
        "bytes_sent_mb": round(net_io.bytes_sent / 1e6, 2),
        "bytes_recv_mb": round(net_io.bytes_recv / 1e6, 2),
        "packets_sent": net_io.packets_sent,
        "packets_recv": net_io.packets_recv,
    }
    load = list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else []
    return {
        "cpu": cpu,
        "memory": memory,
        "disk": disk_parts,
        "network": network,
        "load_avg_1_5_15": load,
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/processes")
def processes():
    if not _PSUTIL_OK:
        return {"error": "psutil not available", "processes": []}
    procs = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "status"]):
        try:
            info = proc.info
            procs.append({
                "pid": info["pid"],
                "name": info["name"],
                "cpu_percent": info["cpu_percent"] or 0.0,
                "mem_mb": round((info["memory_info"].rss if info["memory_info"] else 0) / 1e6, 2),
                "status": info["status"],
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda p: p["cpu_percent"], reverse=True)
    return {"processes": procs[:TOP_PROCESSES]}


@app.get("/temperature")
def temperature():
    if not _PSUTIL_OK:
        return {"error": "psutil not available", "sensors": {}}
    if not hasattr(psutil, "sensors_temperatures"):
        return {"sensors": {}, "note": "not supported on this OS"}
    raw = psutil.sensors_temperatures() or {}
    sensors: dict = {}
    for name, entries in raw.items():
        sensors[name] = [
            {"label": e.label or name, "current_c": round(e.current, 1),
             "high_c": round(e.high, 1) if e.high else None,
             "critical_c": round(e.critical, 1) if e.critical else None}
            for e in entries
        ]
    return {"sensors": sensors}


@app.get("/battery")
def battery():
    if not _PSUTIL_OK:
        return {"error": "psutil not available", "battery": None}
    if not hasattr(psutil, "sensors_battery"):
        return {"battery": None, "note": "not supported on this OS"}
    b = psutil.sensors_battery()
    if b is None:
        return {"battery": None, "note": "no battery detected"}
    return {
        "battery": {
            "percent": round(b.percent, 1),
            "power_plugged": b.power_plugged,
            "secs_left": b.secsleft if b.secsleft not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN) else None,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
