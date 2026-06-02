from __future__ import annotations

from collections import deque
from typing import Any, Callable, Dict

from fastapi import APIRouter

from routes_ops import (
    get_metrics_payload,
    get_models_payload,
    get_queue_stats_payload,
    query_logs,
)


def build_router(
    *,
    available_live: list[str],
    budget: Any,
    get_queue: Callable[[], Any],
    log_buffer: deque[Dict[str, Any]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/metrics")
    async def metrics() -> Dict[str, float]:
        return get_metrics_payload(budget)

    @router.get("/queue/stats")
    async def queue_stats() -> Dict[str, Any]:
        """HP5: Priority queue statistics."""
        return get_queue_stats_payload(get_queue)

    @router.get("/models")
    async def models_info() -> Dict[str, Any]:
        """HP2: Available models and selection info."""
        from model_selector import get_profile, list_models

        return get_models_payload(
            available_live=available_live,
            get_profile=get_profile,
            list_models=list_models,
        )

    @router.get("/logs")
    async def get_logs(limit: int = 100, level: str = "", since: float = 0) -> Dict[str, Any]:
        """Query recent log entries from the agentic service."""
        return query_logs(log_buffer, level=level, limit=limit, since=since)

    return router
