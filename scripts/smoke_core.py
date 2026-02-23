from __future__ import annotations

import time
from typing import List

import httpx


def check(url: str) -> bool:
    try:
        r = httpx.get(url, timeout=2.0)
        r.raise_for_status()
        print(f"{url} -> ok")
        return True
    except Exception as e:
        print(f"{url} -> ERROR: {e}")
        return False


def main() -> int:
    core_services = [
        "http://localhost:5432",  # postgres tcp check
        "http://localhost:8000/health",  # tool-gate
        "http://localhost:8001/health",  # memu-core
        "http://localhost:8010/health",  # heartbeat
        "http://localhost:8080/health",  # dashboard
    ]
    optional_services = [
        "http://localhost:8002/health",  # executor
        "http://localhost:8007/health",  # langgraph
        "http://localhost:8021/health",  # audio service
        "http://localhost:8040/health",  # camera
        "http://localhost:8090/health",  # grok
        "http://localhost:8030/health",  # tts service
        "http://localhost:8081/health",  # avatar service
    ]
    print("waiting for services to warm up... (30s)")
    time.sleep(5)
    success: List[bool] = []
    for url in core_services:
        success.append(check(url))
    if all(success):
        print("all core services healthy")
    else:
        print("some core services failed")
    # probe optional services but don't treat them as fatal
    for url in optional_services:
        check(url)
    return 0 if all(success) else 1


if __name__ == "__main__":
    raise SystemExit(main())
