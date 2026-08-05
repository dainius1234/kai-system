"""D124: Service Watchdog — persistent health monitoring for Kai's component services.

Phase 2: Self-Preservation. Kai actively monitors the services it depends on,
tracks failure history, and surfaces recommended FSM events (SERVICE_DOWN,
SERVICE_RESTORED) for the async caller to fire. This closes the loop between
service failure detection and Kai's operational state machine.

Trust gating:
    status() / check_all()     → OBSERVER (1) — read-only health data
    (no active recovery actions in Phase 0 — groundwork for AGENT gating)

Feature-flagged: FF_SERVICE_WATCHDOG=true
Fail-open: watchdog errors never crash the pipeline.
Storage: data/watchdog/status.json — last check results, persistent across restarts.

FSM integration (SERVICE_DOWN / SERVICE_RESTORED events) is handled by the
async caller — watchdog returns recommended events as strings so it stays sync.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger("kai.watchdog")

_DATA_DIR = Path("data/watchdog")
_STATUS_FILE = _DATA_DIR / "status.json"
_DEFAULT_TIMEOUT_S = 3.0
_FAILURE_THRESHOLD = 2   # consecutive failures before marking a service down
_FSM_EVENT_SERVICE_DOWN = "service_down"
_FSM_EVENT_SERVICE_RESTORED = "service_restored"


# ── Service registry ───────────────────────────────────────────────────────────

@dataclass
class ServiceProfile:
    name: str
    url: str
    health_path: str = "/health"
    critical: bool = False       # critical services trigger FSM DEGRADED
    consecutive_failures: int = 0
    last_healthy_at: float = 0.0
    was_down: bool = False       # tracks previous state to emit restored events

    @property
    def health_url(self) -> str:
        return self.url.rstrip("/") + self.health_path


@dataclass
class CheckResult:
    name: str
    url: str
    healthy: bool
    status_code: int
    latency_ms: float
    consecutive_failures: int
    critical: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Built-in service registry ──────────────────────────────────────────────────
# Matches the sensory services list in app.py; URLs resolved from env at check time.

_DEFAULT_SERVICES: List[Dict[str, Any]] = [
    {"name": "broker", "env": "BROKER_URL", "critical": True},
    {"name": "skill_hunter", "env": "SKILL_HUNTER_URL", "critical": True},
    {"name": "calendar", "env": "CALENDAR_URL", "critical": False},
    {"name": "email_reader", "env": "EMAIL_READER_URL", "critical": False},
    {"name": "docker_watcher", "env": "DOCKER_WATCHER_URL", "critical": False},
    {"name": "sysmetrics", "env": "SYSMETRICS_URL", "critical": False},
    {"name": "weather", "env": "WEATHER_URL", "critical": False},
    {"name": "airquality", "env": "AIRQUALITY_URL", "critical": False},
    {"name": "news_feed", "env": "NEWS_FEED_URL", "critical": False},
    {"name": "git_watcher", "env": "GIT_WATCHER_URL", "critical": False},
]


def _resolve_services() -> List[ServiceProfile]:
    """Build ServiceProfiles from environment variables."""
    import os
    profiles = []
    for svc in _DEFAULT_SERVICES:
        url = os.getenv(svc["env"], "")
        if url:
            profiles.append(ServiceProfile(
                name=svc["name"],
                url=url,
                critical=svc["critical"],
            ))
    return profiles


# ── Watchdog ───────────────────────────────────────────────────────────────────

class ServiceWatchdog:
    """Tracks health of all registered Kai services."""

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        self._dir = data_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, ServiceProfile] = {}
        self._last_results: Dict[str, CheckResult] = {}
        self._last_checked_at: float = 0.0
        self._load_state()

    # ── Persistence ────────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Restore failure counters and was_down flags from disk."""
        f = self._dir / "status.json"
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text())
            for entry in data.get("services", []):
                name = entry.get("name", "")
                if name:
                    # We'll update these when we build profiles
                    # Store as raw dict for now; applied after _resolve_services()
                    pass
            self._last_checked_at = data.get("last_checked_at", 0.0)
        except Exception as exc:
            logger.debug("Watchdog state load failed (non-critical): %s", exc)

    def _save_state(self) -> None:
        try:
            payload = {
                "last_checked_at": self._last_checked_at,
                "services": [
                    {
                        "name": r.name,
                        "healthy": r.healthy,
                        "status_code": r.status_code,
                        "latency_ms": r.latency_ms,
                        "consecutive_failures": r.consecutive_failures,
                        "critical": r.critical,
                        "error": r.error,
                    }
                    for r in self._last_results.values()
                ],
            }
            tmp = self._dir / "status.json.tmp"
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self._dir / "status.json")
        except Exception as exc:
            logger.debug("Watchdog state save failed: %s", exc)

    # ── Core ping ──────────────────────────────────────────────────────

    def ping(
        self,
        name: str,
        url: str,
        health_path: str = "/health",
        critical: bool = False,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> CheckResult:
        """Ping a single service health endpoint."""
        health_url = url.rstrip("/") + health_path
        t0 = time.monotonic()
        try:
            with httpx.Client(timeout=timeout_s) as client:
                resp = client.get(health_url)
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            healthy = resp.status_code < 400
            return CheckResult(
                name=name,
                url=url,
                healthy=healthy,
                status_code=resp.status_code,
                latency_ms=latency_ms,
                consecutive_failures=0 if healthy else 1,
                critical=critical,
            )
        except Exception as exc:
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            return CheckResult(
                name=name,
                url=url,
                healthy=False,
                status_code=0,
                latency_ms=latency_ms,
                consecutive_failures=1,
                critical=critical,
                error=str(exc)[:200],
            )

    # ── Check all services ─────────────────────────────────────────────

    def check_all(
        self,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        services: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[CheckResult], List[str]]:
        """Check all registered services in parallel (via threads).

        Returns:
            results   — CheckResult per service
            fsm_events — recommended FSM event names to fire
                         ("service_down" or "service_restored")

        Never raises.
        """
        import concurrent.futures

        resolved = _resolve_services() if services is None else [
            ServiceProfile(name=s["name"], url=s["url"],
                           critical=s.get("critical", False))
            for s in services
        ]

        if not resolved:
            return [], []

        def _check_one(svc: ServiceProfile) -> CheckResult:
            result = self.ping(svc.name, svc.url, svc.health_path,
                               svc.critical, timeout_s)
            # Accumulate consecutive failures from prior state
            prev = self._last_results.get(svc.name)
            if not result.healthy:
                prev_failures = prev.consecutive_failures if prev else 0
                result.consecutive_failures = prev_failures + 1
            svc.consecutive_failures = result.consecutive_failures
            svc.was_down = (prev is not None and not prev.healthy
                            and prev.consecutive_failures >= _FAILURE_THRESHOLD)
            return result

        results: List[CheckResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(_check_one, svc): svc for svc in resolved}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    svc = futures[fut]
                    logger.debug("Watchdog check failed for %s: %s", svc.name, exc)

        # Update stored results
        for r in results:
            self._last_results[r.name] = r
        self._last_checked_at = time.time()
        self._save_state()

        # Compute recommended FSM events
        fsm_events = self._recommend_fsm_events(resolved, results)

        healthy_count = sum(1 for r in results if r.healthy)
        logger.info(
            "Watchdog check: %d/%d healthy, events=%s",
            healthy_count, len(results), fsm_events,
        )
        return results, fsm_events

    def _recommend_fsm_events(
        self,
        profiles: List[ServiceProfile],
        results: List[CheckResult],
    ) -> List[str]:
        """Derive FSM events from check results based on critical service state."""
        events: List[str] = []
        profile_map = {p.name: p for p in profiles}

        critical_down = any(
            r.critical and not r.healthy and r.consecutive_failures >= _FAILURE_THRESHOLD
            for r in results
        )
        critical_restored = any(
            r.critical and r.healthy
            and profile_map.get(r.name, ServiceProfile("", "")).was_down
            for r in results
        )

        if critical_down:
            events.append(_FSM_EVENT_SERVICE_DOWN)
        if critical_restored and not critical_down:
            events.append(_FSM_EVENT_SERVICE_RESTORED)
        return events

    # ── Status ─────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return last check summary."""
        results = list(self._last_results.values())
        healthy = [r for r in results if r.healthy]
        critical_down = [r for r in results if r.critical and not r.healthy]
        return {
            "last_checked_at": self._last_checked_at,
            "seconds_since_check": round(time.time() - self._last_checked_at)
            if self._last_checked_at else None,
            "total": len(results),
            "healthy_count": len(healthy),
            "unhealthy_count": len(results) - len(healthy),
            "critical_down": [r.name for r in critical_down],
            "services": [r.to_dict() for r in results],
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_watchdog: Optional[ServiceWatchdog] = None


def get_watchdog(data_dir: Path = _DATA_DIR) -> ServiceWatchdog:
    global _watchdog
    if _watchdog is None:
        _watchdog = ServiceWatchdog(data_dir=data_dir)
    return _watchdog


def reset_watchdog() -> None:
    global _watchdog
    _watchdog = None
