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


# ═══════════════════════════════════════════════════════════════════
#  Bio-inspired Self-Healing (ReCiSt 4-Phase)
# ═══════════════════════════════════════════════════════════════════
# Modelled after multi-agent self-healing patterns:
#   Phase 1 — Containment   : isolate the failing component
#   Phase 2 — Diagnosis     : identify root cause from recent failures
#   Phase 3 — Meta-Cognitive: reflect on whether usual fixes work
#   Phase 4 — Knowledge     : record what worked for future reference
#
# Low-CPU design: pure Python dicts, no ML inference, no background
# threads.  The heal() coroutine runs only when a failure is detected.

PHASE_CONTAINMENT = "containment"
PHASE_DIAGNOSIS = "diagnosis"
PHASE_META_COGNITIVE = "meta_cognitive"
PHASE_KNOWLEDGE = "knowledge"
PHASE_HEALTHY = "healthy"

_PHASE_ORDER = [
    PHASE_CONTAINMENT,
    PHASE_DIAGNOSIS,
    PHASE_META_COGNITIVE,
    PHASE_KNOWLEDGE,
    PHASE_HEALTHY,
]


@dataclass
class FailureRecord:
    """Single failure observation."""
    service: str
    error: str
    ts: float = 0.0
    phase: str = PHASE_CONTAINMENT
    fix_applied: str = ""

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()


@dataclass
class HealingEngine:
    """4-phase bio-inspired self-healing engine.

    Tracks per-service failure history and progresses through healing
    phases.  Each phase is a lightweight heuristic — no heavy compute.

    Usage::

        healer = HealingEngine()
        result = await healer.heal("memu-core", "connection refused")
        # result = {"phase": "containment", "action": "circuit_open", ...}
    """
    # How many recent failures to keep per service
    history_limit: int = 50
    # After this many consecutive same-error failures, escalate to
    # meta-cognitive phase (usual fixes aren't working)
    escalation_threshold: int = 3

    _history: Dict[str, List[FailureRecord]] = field(default_factory=dict)
    _knowledge: Dict[str, Dict[str, str]] = field(default_factory=dict)
    _phase: Dict[str, str] = field(default_factory=dict)

    # ── Phase 1: Containment ──────────────────────────────────────
    def _containment(self, service: str) -> Dict[str, Any]:
        """Isolate: open circuit breaker, mark degraded."""
        cb = _get_breaker(service)
        # Force-open circuit
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self._phase[service] = PHASE_DIAGNOSIS
        return {
            "phase": PHASE_CONTAINMENT,
            "action": "circuit_open",
            "service": service,
            "detail": "circuit breaker opened to isolate failure",
        }

    # ── Phase 2: Diagnosis ────────────────────────────────────────
    def _diagnosis(self, service: str, error: str) -> Dict[str, Any]:
        """Analyse recent failures to identify pattern."""
        history = self._history.get(service, [])
        recent_errors = [r.error for r in history[-10:]]
        unique_errors = list(set(recent_errors))

        # Check knowledge base for known fix
        known_fix = self._knowledge.get(service, {}).get(error)
        if known_fix:
            self._phase[service] = PHASE_KNOWLEDGE
            return {
                "phase": PHASE_DIAGNOSIS,
                "action": "known_fix_found",
                "service": service,
                "fix": known_fix,
                "detail": f"knowledge base has fix for '{error}'",
            }

        # Count consecutive identical errors
        consecutive = 0
        for r in reversed(history):
            if r.error == error:
                consecutive += 1
            else:
                break

        if consecutive >= self.escalation_threshold:
            self._phase[service] = PHASE_META_COGNITIVE
        else:
            self._phase[service] = PHASE_KNOWLEDGE

        return {
            "phase": PHASE_DIAGNOSIS,
            "action": "pattern_analysis",
            "service": service,
            "error": error,
            "consecutive": consecutive,
            "unique_errors": unique_errors,
            "escalated": consecutive >= self.escalation_threshold,
            "detail": f"{consecutive} consecutive '{error}' failures",
        }

    # ── Phase 3: Meta-Cognitive ───────────────────────────────────
    def _meta_cognitive(self, service: str, error: str) -> Dict[str, Any]:
        """Reflect: usual fixes aren't working, try alternative."""
        history = self._history.get(service, [])
        fixes_tried = [r.fix_applied for r in history if r.fix_applied]
        unique_fixes = list(set(fixes_tried))

        # Suggest escalation
        suggestion = "restart_service"
        if "restart_service" in unique_fixes:
            suggestion = "rebuild_container"
        if "rebuild_container" in unique_fixes:
            suggestion = "alert_operator"

        self._phase[service] = PHASE_KNOWLEDGE
        return {
            "phase": PHASE_META_COGNITIVE,
            "action": "reflect_and_escalate",
            "service": service,
            "fixes_tried": unique_fixes,
            "suggestion": suggestion,
            "detail": (
                f"standard fixes exhausted ({len(unique_fixes)} tried), "
                f"suggesting: {suggestion}"
            ),
        }

    # ── Phase 4: Knowledge ────────────────────────────────────────
    def _record_knowledge(
        self, service: str, error: str, fix: str
    ) -> Dict[str, Any]:
        """Store what worked for future use."""
        if service not in self._knowledge:
            self._knowledge[service] = {}
        self._knowledge[service][error] = fix
        self._phase[service] = PHASE_HEALTHY
        return {
            "phase": PHASE_KNOWLEDGE,
            "action": "fix_recorded",
            "service": service,
            "error": error,
            "fix": fix,
            "detail": f"recorded fix '{fix}' for '{error}'",
        }

    # ── Main Entry Point ──────────────────────────────────────────
    async def heal(
        self, service: str, error: str, fix_applied: str = ""
    ) -> Dict[str, Any]:
        """Run the 4-phase healing cycle for a service failure.

        Call this when a failure is detected.  If *fix_applied* is
        provided, skip straight to the knowledge phase.
        """
        record = FailureRecord(service=service, error=error,
                               fix_applied=fix_applied)
        if service not in self._history:
            self._history[service] = []
        self._history[service].append(record)
        if len(self._history[service]) > self.history_limit:
            self._history[service] = self._history[service][
                -self.history_limit:
            ]

        # If caller already applied a fix → record knowledge
        if fix_applied:
            return self._record_knowledge(service, error, fix_applied)

        # Determine current phase
        current = self._phase.get(service, PHASE_CONTAINMENT)

        if current == PHASE_CONTAINMENT:
            return self._containment(service)
        elif current == PHASE_DIAGNOSIS:
            return self._diagnosis(service, error)
        elif current == PHASE_META_COGNITIVE:
            return self._meta_cognitive(service, error)
        elif current == PHASE_KNOWLEDGE:
            return self._record_knowledge(service, error, "auto_recovery")
        else:
            # Reset and start fresh
            self._phase[service] = PHASE_CONTAINMENT
            return self._containment(service)

    def reset(self, service: str) -> None:
        """Mark a service as healthy again."""
        self._phase[service] = PHASE_HEALTHY

    def status(self, service: Optional[str] = None) -> Dict[str, Any]:
        """Snapshot of healing state."""
        if service:
            return {
                "service": service,
                "phase": self._phase.get(service, PHASE_HEALTHY),
                "failures": len(self._history.get(service, [])),
                "knowledge_entries": len(
                    self._knowledge.get(service, {})
                ),
            }
        return {
            name: {
                "phase": self._phase.get(name, PHASE_HEALTHY),
                "failures": len(recs),
                "knowledge_entries": len(
                    self._knowledge.get(name, {})
                ),
            }
            for name, recs in self._history.items()
        }

    def knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """Return the full knowledge base for inspection."""
        return dict(self._knowledge)
