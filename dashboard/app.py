from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI


app = FastAPI(title="Sovereign Dashboard", version="0.2.0")

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("No GPU — running on CPU only")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
LEDGER_URL = os.getenv("LEDGER_URL", "postgresql://keeper:***@postgres:5432/sovereign")


NODES = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": os.getenv("MEMU_URL", "http://memu-core:8001") + "/health",
    "langgraph": os.getenv("LANGGRAPH_URL", "http://langgraph:8007") + "/health",
    "executor": os.getenv("EXECUTOR_URL", "http://executor:8002") + "/health",
    "heartbeat": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
    "tts": os.getenv("TTS_URL", "http://tts-service:8030") + "/health",
    "avatar": os.getenv("AVATAR_URL", "http://avatar-service:8081") + "/health",
}


def local_status_text() -> str:
    return "running (CPU)" if DEVICE == "cpu" else "running (CUDA)"


async def fetch_status() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for name, url in NODES.items():
            try:
                resp = await client.get(url, timeout=2.0)
                resp.raise_for_status()
                results[name] = {"status": "ok", "details": resp.json()}
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "down", "error": str(exc)}
    return results


async def fetch_ledger_size() -> int:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()
    return int(payload.get("count", 0))


async def fetch_memory_count() -> int:
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{memu_url}/memory/stats", timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()
    return int(payload.get("records", 0))


async def fetch_policy_mode() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{TOOL_GATE_URL}/health", timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()
    return payload.get("mode", "unknown")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": local_status_text(),
        "tool_gate_url": TOOL_GATE_URL,
    }


@app.get("/readiness")
async def readiness() -> Dict[str, Any]:
    node_status = await fetch_status()
    required = ["tool-gate", "memu-core", "executor"]
    core_ready = all(node_status.get(name, {}).get("status") == "ok" for name in required)

    ledger_size = -1
    memory_count = -1
    try:
        ledger_size = await fetch_ledger_size()
        memory_count = await fetch_memory_count()
    except Exception:  # noqa: BLE001
        core_ready = False

    return {
        "status": "ready" if core_ready and ledger_size >= 0 and memory_count >= 0 else "not_ready",
        "core_ready": core_ready,
        "ledger_size": ledger_size,
        "memory_count": memory_count,
        "required_nodes": required,
    }


@app.get("/")
async def index() -> Dict[str, Any]:
    statuses = await fetch_status()
    alive_nodes = [name for name, payload in statuses.items() if payload.get("status") == "ok"]
    ledger_size = await fetch_ledger_size()
    memory_count = await fetch_memory_count()
    policy_mode = await fetch_policy_mode()
    core_ready = all(statuses.get(name, {}).get("status") == "ok" for name in ["tool-gate", "memu-core", "executor"])
    return {
        "service": "dashboard",
        "status": local_status_text(),
        "device_summary": local_status_text(),
        "policy_mode": policy_mode,
        "core_ready": core_ready,
        "tool_gate_url": TOOL_GATE_URL,
        "ledger_url": LEDGER_URL,
        "alive_nodes": alive_nodes,
        "node_status": statuses,
        "ledger_size": ledger_size,
        "memory_count": memory_count,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
