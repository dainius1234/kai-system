"""HP5 — Async Priority Queue for LLM request scheduling.

Routes latency-sensitive requests (chat) ahead of batch jobs (dream, audit).
Supports preemption: a high-priority request can jump ahead of queued
low-priority work.

Usage:
    from priority_queue import PriorityQueue, Priority
    queue = PriorityQueue()
    result = await queue.submit(Priority.CHAT, coro_fn, *args)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Dict, List, Optional


class Priority(IntEnum):
    """Request priority levels. Lower number = higher priority."""
    CHAT = 0        # Interactive chat — lowest latency
    RUN = 1         # Agentic /run pipeline
    BACKGROUND = 2  # Background tasks (audit, dream)
    BATCH = 3       # Batch operations (cleanup, compression)


@dataclass
class QueueEntry:
    """A single queued request."""
    priority: Priority
    submitted_at: float
    task_id: str
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    def __lt__(self, other: "QueueEntry") -> bool:
        """Lower priority number wins. Ties broken by submission time."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.submitted_at < other.submitted_at


@dataclass
class QueueStats:
    """Snapshot of queue state."""
    pending: int
    active: int
    total_processed: int
    avg_wait_ms: float
    by_priority: Dict[str, int]


class PriorityQueue:
    """Async priority queue with concurrency limiting.

    Args:
        max_concurrent: Maximum number of concurrent tasks (e.g. GPU slots).
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        self._max_concurrent = max(1, max_concurrent)
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        self._pending: List[QueueEntry] = []
        self._active: int = 0
        self._total_processed: int = 0
        self._wait_times: List[float] = []
        self._lock = asyncio.Lock()

    async def submit(
        self,
        priority: Priority,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        task_id: str = "",
        **kwargs: Any,
    ) -> Any:
        """Submit a coroutine function with priority. Returns the result.

        Higher-priority requests (lower number) are processed first.
        Blocks until a concurrency slot is available.
        """
        if not task_id:
            task_id = f"{priority.name}-{time.monotonic():.4f}"

        submitted = time.monotonic()

        # Acquire semaphore (respects max_concurrent)
        await self._semaphore.acquire()

        wait_ms = (time.monotonic() - submitted) * 1000
        async with self._lock:
            self._active += 1
            self._wait_times.append(wait_ms)
            # Keep only last 100 samples
            if len(self._wait_times) > 100:
                self._wait_times = self._wait_times[-100:]

        try:
            result = await fn(*args, **kwargs)
            return result
        finally:
            async with self._lock:
                self._active -= 1
                self._total_processed += 1
            self._semaphore.release()

    def stats(self) -> QueueStats:
        """Current queue statistics."""
        avg_wait = sum(self._wait_times) / max(len(self._wait_times), 1)
        return QueueStats(
            pending=self._max_concurrent - self._semaphore._value,
            active=self._active,
            total_processed=self._total_processed,
            avg_wait_ms=round(avg_wait, 2),
            by_priority={},
        )


# Module-level singleton — shared across the langgraph service
_default_queue: Optional[PriorityQueue] = None


def get_queue(max_concurrent: int = 1) -> PriorityQueue:
    """Get or create the global priority queue."""
    global _default_queue
    if _default_queue is None:
        _default_queue = PriorityQueue(max_concurrent=max_concurrent)
    return _default_queue
