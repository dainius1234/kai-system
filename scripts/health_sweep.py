#!/usr/bin/env python3
"""Health sweep — hit every service /health endpoint and produce a scorecard.

Usage:
    python scripts/health_sweep.py                # default: docker-compose.full.yml ports
    python scripts/health_sweep.py --host 172.20.0.3  # custom host

Returns exit code 0 if all services are healthy, 1 if any are down.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Dict, List, Tuple

import httpx

# (service_name, host_port) — matches docker-compose.full.yml port mappings
SERVICES: List[Tuple[str, int]] = [
    ("tool-gate",          8000),
    ("memu-core",          8001),
    ("executor",           8002),
    ("langgraph",          8007),
    ("heartbeat",          8010),
    ("audio-service",      8021),
    ("camera-service",     8020),
    ("tts-service",        8030),
    ("supervisor",         8051),
    ("verifier",           8052),
    ("fusion-engine",      8053),
    ("backup-service",     8054),
    ("calendar-sync",      8055),
    ("ledger-worker",      8056),
    ("memory-compressor",  8057),
    ("metrics-gateway",    8058),
    ("screen-capture",     8059),
    ("orchestrator",       8050),
    ("workspace-manager",  8060),
    ("dashboard",          8080),
    ("avatar-service",     8081),
    ("kai-advisor",        8090),
]

HOST = "localhost"
TIMEOUT = 5.0
RETRIES = 3
RETRY_DELAY = 2.0


def check_health(name: str, port: int) -> Dict[str, str]:
    """Hit /health and return result dict."""
    url = f"http://{HOST}:{port}/health"
    for attempt in range(RETRIES):
        try:
            resp = httpx.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return {"service": name, "port": port, "status": "UP",
                        "detail": data.get("status", "ok")}
            else:
                return {"service": name, "port": port, "status": "DEGRADED",
                        "detail": f"HTTP {resp.status_code}"}
        except Exception as e:
            if attempt < RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {"service": name, "port": port, "status": "DOWN",
                    "detail": str(e)[:80]}
    return {"service": name, "port": port, "status": "DOWN", "detail": "exhausted retries"}


def main() -> int:
    print(f"\n{'='*60}")
    print(f"  Kai System — Health Sweep")
    print(f"  {len(SERVICES)} services | host={HOST}")
    print(f"{'='*60}\n")

    results = []
    for name, port in SERVICES:
        result = check_health(name, port)
        results.append(result)
        icon = "OK" if result["status"] == "UP" else "!!" if result["status"] == "DEGRADED" else "XX"
        print(f"  [{icon}] {name:24s} :{port}  {result['status']:10s}  {result['detail']}")

    up = sum(1 for r in results if r["status"] == "UP")
    down = sum(1 for r in results if r["status"] == "DOWN")
    degraded = sum(1 for r in results if r["status"] == "DEGRADED")

    print(f"\n{'='*60}")
    print(f"  SCORECARD: {up} UP / {degraded} DEGRADED / {down} DOWN — {len(SERVICES)} total")

    if down == 0 and degraded == 0:
        print(f"  RESULT: ALL GREEN")
    elif down == 0:
        print(f"  RESULT: PARTIAL (some degraded)")
    else:
        print(f"  RESULT: FAILURES DETECTED")
    print(f"{'='*60}\n")

    # write JSON scorecard
    scorecard = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(SERVICES),
        "up": up,
        "degraded": degraded,
        "down": down,
        "services": results,
    }
    try:
        with open("output/health_scorecard.json", "w") as f:
            json.dump(scorecard, f, indent=2)
        print(f"  Scorecard written: output/health_scorecard.json\n")
    except Exception:
        pass

    return 0 if down == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
