"""Shared resilience primitives for Sovereign AI services.

Layer 1 (Process-Level) building blocks:
  - resilient_call()  : HTTP client with retry, backoff, and circuit-breaker
  - ServiceHealth     : deep /health probe that checks real dependencies
  - TaskWatchdog      : detects frozen asyncio background tasks
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import httpx

from common.runtime import CircuitBreaker


# ═══════════════════════════════════════════════════════════════════
#  Resilient HTTP Caller
# ═══════════════════════════════════════════════════════════════════

# Per-target circuit breakers (lazy-created, shared across calls)
_breakers: Dict[str, CircuitBreaker] = {}
_breaker_lock = asyncio.Lock()


def _get_breaker(name: str) -> CircuitBreaker:
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(failure_threshold=3, recovery_seconds=30)
    return _breakers[name]


async def resilient_call(
    method: str,
    url: str,
    *,
    service_name: str = "",
    json: Any = None,
    params: Any = None,
    timeout: float = 5.0,
    retries: int = 2,
    backoff: float = 0.5,
    fallback: Any = None,
    logger: Any = None,
) -> Any:
    """HTTP call with retry, exponential backoff, and circuit breaker.

    Returns the parsed JSON response on success, or *fallback* when all
    attempts fail or the circuit is open.
    """
    name = service_name or url.split("//")[-1].split("/")[0].split(":")[0]
    cb = _get_breaker(name)

    if not cb.allow():
        if logger:
            logger.debug("Circuit open for %s — returning fallback", name)
        return fallback

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method.upper() == "GET":
                    resp = await client.get(url, params=params)
                else:
                    resp = await client.post(url, json=json)
            if resp.status_code < 500:
                cb.record_success()
                return resp.json()
            last_exc = Exception(f"HTTP {resp.status_code}")
        except Exception as exc:
            last_exc = exc
        # backoff before retry (skip sleep on last attempt)
        if attempt < retries:
            await asyncio.sleep(backoff * (2 ** (attempt - 1)))

    cb.record_failure()
    if logger:
        logger.warning("resilient_call %s %s failed after %d attempts: %s",
                        method, url, retries, last_exc)
    return fallback


def get_breaker_snapshot() -> Dict[str, Dict]:
    """Snapshot of all client-side circuit breakers for diagnostics."""
    return {name: cb.snapshot() for name, cb in _breakers.items()}


# ═══════════════════════════════════════════════════════════════════
#  Deep Health Probe
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HealthCheck:
    """One dependency to probe during a deep health check."""
    name: str
    check: Callable[[], Any]  # async callable returning truthy on success


@dataclass
class ServiceHealth:
    """Manages deep health checks for a service.

    Register dependency checks (postgres, redis, sibling services)
    and call ``probe()`` to get honest health status.
    """
    service_name: str
    checks: List[HealthCheck] = field(default_factory=list)
    _last_result: Dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, check: Callable) -> None:
        self.checks.append(HealthCheck(name=name, check=check))

    async def probe(self) -> Dict[str, Any]:
        """Run all registered checks and return honest health status."""
        results: Dict[str, str] = {}
        degraded = False
        for hc in self.checks:
            try:
                ok = await hc.check()
                results[hc.name] = "ok" if ok else "degraded"
                if not ok:
                    degraded = True
            except Exception as exc:
                results[hc.name] = f"fail: {str(exc)[:80]}"
                degraded = True

        status = "degraded" if degraded else "ok"
        self._last_result = {
            "status": status,
            "service": self.service_name,
            "checks": results,
            "ts": time.time(),
        }
        return self._last_result


# ═══════════════════════════════════════════════════════════════════
#  Task Watchdog — detects frozen background tasks
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TaskWatchdog:
    """Tracks liveness of named background loops.

    Each background task calls ``heartbeat(task_name)`` each iteration.
    ``frozen()`` returns the list of tasks that haven't reported in
    more than ``stale_seconds``.
    """
    stale_seconds: float = 60.0
    _beats: Dict[str, float] = field(default_factory=dict)

    def heartbeat(self, task_name: str) -> None:
        self._beats[task_name] = time.time()

    def frozen(self) -> List[str]:
        now = time.time()
        return [name for name, ts in self._beats.items()
                if now - ts > self.stale_seconds]

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        return {
            name: {
                "last_beat": ts,
                "age_seconds": round(now - ts, 1),
                "frozen": (now - ts) > self.stale_seconds,
            }
            for name, ts in self._beats.items()
        }
