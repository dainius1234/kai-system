"""Docker-watcher service — container health and resource monitoring.

Mounts /var/run/docker.sock to observe all running containers.

Endpoints:
  GET /health      → {status, containers_running, uptime_seconds}
  GET /metrics     → error budget
  GET /containers  → [{id, name, image, status, health, restarts, started_at}]
  GET /unhealthy   → containers not in running/healthy state
  GET /summary     → one-sentence summary for agentic context
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("docker-watcher", os.getenv("LOG_PATH", "/tmp/docker-watcher.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("docker-watcher")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

try:
    import docker as docker_sdk
    _DOCKER_SDK = True
except ImportError:
    _DOCKER_SDK = False

PORT = int(os.getenv("PORT", "8041"))
REFRESH_INTERVAL = int(os.getenv("DOCKER_REFRESH_SECONDS", "30"))
DOCKER_HOST = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")

_start = time.time()
_containers: List[dict] = []
_last_poll: float = 0.0
_poll_error: Optional[str] = None
_refresh_task: Optional[asyncio.Task] = None


def _poll_via_sdk() -> List[dict]:
    client = docker_sdk.from_env()
    result = []
    for c in client.containers.list(all=False):
        attrs = c.attrs or {}
        state = attrs.get("State", {})
        result.append({
            "id": c.short_id,
            "name": c.name,
            "image": c.image.tags[0] if c.image.tags else str(c.image.short_id),
            "status": c.status,
            "health": state.get("Health", {}).get("Status", "none") if state.get("Health") else "none",
            "restarts": state.get("RestartCount", 0),
            "started_at": state.get("StartedAt", ""),
            "exit_code": state.get("ExitCode", 0),
        })
    return result


def _poll_via_subprocess() -> List[dict]:
    cmd = ["docker", "ps", "--format",
           "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.RunningFor}}"]
    out = subprocess.check_output(cmd, timeout=10, text=True)
    result = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        cid, name, image, status = parts[0], parts[1], parts[2], parts[3]
        result.append({
            "id": cid,
            "name": name,
            "image": image,
            "status": status,
            "health": "none",
            "restarts": 0,
            "started_at": parts[4] if len(parts) > 4 else "",
            "exit_code": 0,
        })
    return result


def _poll_containers() -> List[dict]:
    if _DOCKER_SDK:
        return _poll_via_sdk()
    return _poll_via_subprocess()


async def _refresh_loop():
    global _containers, _last_poll, _poll_error
    while True:
        try:
            loop = asyncio.get_running_loop()
            containers = await loop.run_in_executor(None, _poll_containers)
            _containers = containers
            _last_poll = time.time()
            _poll_error = None
            logger.info("docker-watcher: %d containers", len(containers))
        except Exception as exc:
            _poll_error = str(exc)
            logger.warning("docker poll error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _refresh_task
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()


app = FastAPI(title="docker-watcher", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    running = sum(1 for c in _containers if c["status"] == "running" or c["status"].startswith("Up"))
    return {
        "status": "ok",
        "containers_running": running,
        "docker_sdk": _DOCKER_SDK,
        "last_poll": _last_poll,
        "poll_error": _poll_error,
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/containers")
def containers():
    return {"containers": _containers, "total": len(_containers), "last_poll": _last_poll}


@app.get("/unhealthy")
def unhealthy():
    bad = [
        c for c in _containers
        if c["health"] not in ("healthy", "none")
        or c.get("exit_code", 0) != 0
        or c.get("restarts", 0) > 3
    ]
    return {"unhealthy": bad, "count": len(bad)}


@app.get("/summary")
def summary():
    total = len(_containers)
    running = sum(1 for c in _containers if c["status"] == "running" or c["status"].startswith("Up"))
    unhealthy_count = sum(1 for c in _containers if c["health"] not in ("healthy", "none"))
    high_restarts = [c["name"] for c in _containers if c.get("restarts", 0) > 3]
    if not _containers:
        msg = "Docker data not yet available."
    elif unhealthy_count or high_restarts:
        issues = []
        if unhealthy_count:
            issues.append(f"{unhealthy_count} unhealthy")
        if high_restarts:
            issues.append(f"high restarts: {', '.join(high_restarts)}")
        msg = f"{running}/{total} containers running — issues: {'; '.join(issues)}."
    else:
        msg = f"All {running} containers running normally."
    return {"summary": msg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
