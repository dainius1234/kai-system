from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Deque, Dict


def build_health_payload(
    *,
    device: str,
    memu_cb: Dict[str, Any],
    tool_gate_cb: Dict[str, Any],
    memu_guard: Dict[str, Any],
    tool_gate_guard: Dict[str, Any],
    degraded: bool,
) -> Dict[str, Any]:
    return {
        "status": "degraded" if degraded else "ok",
        "device": device,
        "dependencies": {"memu": memu_cb, "tool_gate": tool_gate_cb},
        "error_guards": {"memu": memu_guard, "tool_gate": tool_gate_guard},
    }


def checkpoint_pre_recover(
    *,
    create_checkpoint: Callable[..., Any],
    budget_snapshot: Dict[str, Any],
    conviction_overrides: list[str],
    logger: Any,
    memu_breaker: Any,
    tool_gate_breaker: Any,
    memu_error_guard: Any,
    tool_error_guard: Any,
) -> None:
    try:
        create_checkpoint(
            label="pre-recover",
            trigger="pre_recover",
            breaker_states={
                "memu": {**memu_breaker.snapshot(), "opened_at": memu_breaker.opened_at},
                "tool_gate": {**tool_gate_breaker.snapshot(), "opened_at": tool_gate_breaker.opened_at},
            },
            guard_states={"memu": memu_error_guard.snapshot(), "tool_gate": tool_error_guard.snapshot()},
            budget_state=budget_snapshot,
            conviction_overrides=conviction_overrides,
        )
    except Exception:
        logger.debug("Pre-recover checkpoint failed (non-critical)")


def reset_breakers(*, memu_breaker: Any, tool_gate_breaker: Any) -> None:
    memu_breaker.failures = 0
    memu_breaker.state = "closed"
    tool_gate_breaker.failures = 0
    tool_gate_breaker.state = "closed"


def get_metrics_payload(budget: Any) -> Dict[str, float]:
    return budget.snapshot()


def get_queue_stats_payload(get_queue: Callable[[], Any]) -> Dict[str, Any]:
    q = get_queue()
    s = q.stats()
    return {
        "pending": s.pending,
        "active": s.active,
        "total_processed": s.total_processed,
        "avg_wait_ms": s.avg_wait_ms,
    }


def get_models_payload(
    *,
    available_live: list[str],
    get_profile: Callable[[str], Any],
    list_models: Callable[[], list[str]],
) -> Dict[str, Any]:
    profiles = {}
    for name in list_models():
        profile = get_profile(name)
        if profile:
            profiles[name] = {
                "strengths": profile.strengths,
                "speed_tier": profile.speed_tier,
                "quality_tier": profile.quality_tier,
                "moe_experts": profile.moe_expert_count,
            }
    return {"available_live": available_live, "registered": profiles}


class LogCapture(logging.Handler):
    def __init__(self, log_buffer: Deque[Dict[str, Any]], service_name: str) -> None:
        super().__init__()
        self._log_buffer = log_buffer
        self._service_name = service_name

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._log_buffer.append(
                {
                    "time": record.created,
                    "level": record.levelname,
                    "service": self._service_name,
                    "msg": record.getMessage()[:500],
                }
            )
        except Exception:
            pass


def install_log_capture(service_name: str) -> tuple[Deque[Dict[str, Any]], LogCapture]:
    log_buffer: Deque[Dict[str, Any]] = deque(maxlen=500)
    log_capture = LogCapture(log_buffer, service_name)
    log_capture.setLevel(logging.INFO)
    logging.getLogger().addHandler(log_capture)
    return log_buffer, log_capture


def query_logs(log_buffer: Deque[Dict[str, Any]], *, level: str = "", limit: int = 100, since: float = 0) -> Dict[str, Any]:
    entries = list(log_buffer)
    if level:
        entries = [entry for entry in entries if entry["level"] == level.upper()]
    if since:
        entries = [entry for entry in entries if entry["time"] >= since]
    entries.reverse()
    entries = entries[:limit]
    return {"status": "ok", "count": len(entries), "entries": entries}
