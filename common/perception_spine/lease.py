"""Leader lease with fencing tokens.

Closes the fencing half of roadmap §16.27.  Multiple workers may run the
perception spine, but only one may write at a time.  A lock alone is not
enough: a leader that stalls (GC pause, network partition, suspended
container) can wake up believing it is still leader and write stale data
over a newer leader's work.

The fence is a monotonic token.  Every acquisition increments it, and a
writer must present its token to write.  A stale leader's token is lower
than the current one, so its write is rejected even though it still holds
what it thinks is a valid lease.

Time is injectable so clock-change behaviour is testable.  Expiry uses a
monotonic clock, not wall-clock, so an operator moving the system clock
backwards cannot extend a lease.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional


class LeaseError(Exception):
    pass


class FencedLease:
    """Single-writer lease guarded by a monotonically increasing token.

    Parameters:
        ttl_seconds: how long an acquisition stays valid without renewal
        clock: monotonic time source, injectable for tests.  Must be
            monotonic — a wall-clock source would let a backwards clock
            change extend a live lease.
    """

    def __init__(
        self,
        ttl_seconds: float = 30.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise LeaseError("lease ttl must be positive")
        self._ttl = ttl_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._token = 0
        self._holder: Optional[str] = None
        self._expires_at: float = 0.0

    # ── Acquisition ─────────────────────────────────────────────────

    def acquire(self, worker_id: str) -> int:
        """Take the lease, returning a fencing token.

        Succeeds when the lease is free or expired.  Every successful
        acquisition returns a strictly higher token than the last.
        """
        if not worker_id or not worker_id.strip():
            raise LeaseError("worker_id must not be empty")

        with self._lock:
            now = self._clock()
            held = self._holder is not None and now < self._expires_at
            if held and self._holder != worker_id:
                raise LeaseError(
                    f"lease held by '{self._holder}' for "
                    f"{self._expires_at - now:.1f}s more"
                )

            self._token += 1
            self._holder = worker_id
            self._expires_at = now + self._ttl
            return self._token

    def renew(self, worker_id: str, token: int) -> int:
        """Extend the lease without changing the token.

        Renewal keeps the same token so downstream fencing checks stay
        stable for a healthy leader.
        """
        with self._lock:
            if self._holder != worker_id:
                raise LeaseError(
                    f"'{worker_id}' does not hold the lease "
                    f"(holder: {self._holder})"
                )
            if token != self._token:
                raise LeaseError(
                    f"stale token {token}, current is {self._token}"
                )
            if self._clock() >= self._expires_at:
                raise LeaseError("lease already expired — reacquire instead")

            self._expires_at = self._clock() + self._ttl
            return self._token

    def release(self, worker_id: str, token: int) -> None:
        with self._lock:
            if self._holder != worker_id or token != self._token:
                return
            self._holder = None
            self._expires_at = 0.0

    # ── Fencing ─────────────────────────────────────────────────────

    def check_token(self, token: int) -> None:
        """Raise unless ``token`` is the current fencing token.

        This is the call a writer must make before every write.  A stale
        leader fails here even though it still believes it holds a lease.
        """
        with self._lock:
            if token != self._token:
                raise LeaseError(
                    f"fencing token {token} is stale (current {self._token}) "
                    f"— write rejected"
                )
            if self._holder is None or self._clock() >= self._expires_at:
                raise LeaseError("lease expired — write rejected")

    def is_valid(self, worker_id: str, token: int) -> bool:
        try:
            self.check_token(token)
        except LeaseError:
            return False
        with self._lock:
            return self._holder == worker_id

    # ── Inspection ──────────────────────────────────────────────────

    @property
    def current_token(self) -> int:
        with self._lock:
            return self._token

    @property
    def holder(self) -> Optional[str]:
        with self._lock:
            if self._holder is not None and self._clock() >= self._expires_at:
                return None
            return self._holder
