"""Perception ingress — validated event intake with dedup and staleness.

The ingress is the single entry point for all perception events.  It:

  1. Validates the event against the PerceptionEvent schema (rejects malformed).
  2. Checks for duplicates (by raw_hash or event id).
  3. Marks stale events based on configurable freshness windows.
  4. Enforces principal isolation (cross-principal events cannot leak).
  5. Appends accepted events to the durable journal.
  6. Returns an IngressResult indicating accept/reject/duplicate/stale.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional, Set

from pydantic import ValidationError

from common.contracts.base import ContractState, Principal, Provenance
from common.contracts.perception import EventSource, PerceptionEvent

from common.perception_spine.journal import EventJournal


class IngressVerdict(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_INVALID = "rejected_invalid"
    REJECTED_DUPLICATE = "rejected_duplicate"
    ACCEPTED_STALE = "accepted_stale"
    REJECTED_CROSS_PRINCIPAL = "rejected_cross_principal"
    REJECTED_OVERSIZED = "rejected_oversized"


# Payload bounds (roadmap §16.4: oversized/deep/high-cardinality events).
# A compromised or faulty source must not be able to exhaust memory in the
# reducers by sending one enormous event.
MAX_PAYLOAD_BYTES = 256 * 1024
MAX_PAYLOAD_DEPTH = 16
MAX_PAYLOAD_KEYS = 1_000
MAX_STRING_LENGTH = 64 * 1024


def _payload_depth(obj: Any, _depth: int = 0) -> int:
    """Maximum nesting depth, short-circuiting past the limit.

    Stops descending once the cap is exceeded so a maliciously deep
    payload cannot blow the Python stack during the check itself.
    """
    if _depth > MAX_PAYLOAD_DEPTH:
        return _depth
    if isinstance(obj, dict):
        if not obj:
            return _depth
        return max(_payload_depth(v, _depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        if not obj:
            return _depth
        return max(_payload_depth(v, _depth + 1) for v in obj)
    return _depth


def _payload_cardinality(obj: Any, _budget: int = MAX_PAYLOAD_KEYS) -> int:
    """Total key/element count, stopping once the budget is spent."""
    count = 0
    stack = [obj]
    while stack and count <= _budget:
        current = stack.pop()
        if isinstance(current, dict):
            count += len(current)
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            count += len(current)
            stack.extend(current)
    return count


def _longest_string(obj: Any) -> int:
    longest = 0
    stack = [obj]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            longest = max(longest, len(current))
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return longest


def check_payload_bounds(payload: Any) -> Optional[str]:
    """Return a rejection reason, or None when the payload is within bounds.

    Ordered cheapest-check-first: depth and cardinality are bounded walks,
    so they run before the full serialisation needed for a byte count.
    """
    depth = _payload_depth(payload)
    if depth > MAX_PAYLOAD_DEPTH:
        return f"payload too deep: {depth} > {MAX_PAYLOAD_DEPTH}"

    cardinality = _payload_cardinality(payload)
    if cardinality > MAX_PAYLOAD_KEYS:
        return f"payload too high-cardinality: {cardinality} > {MAX_PAYLOAD_KEYS}"

    longest = _longest_string(payload)
    if longest > MAX_STRING_LENGTH:
        return f"payload string too long: {longest} > {MAX_STRING_LENGTH}"

    try:
        size = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        return f"payload not serialisable: {exc}"
    if size > MAX_PAYLOAD_BYTES:
        return f"payload too large: {size} > {MAX_PAYLOAD_BYTES} bytes"

    return None


class IngressResult:
    __slots__ = ("verdict", "offset", "event", "reason")

    def __init__(
        self,
        verdict: IngressVerdict,
        offset: Optional[int] = None,
        event: Optional[PerceptionEvent] = None,
        reason: str = "",
    ):
        self.verdict = verdict
        self.offset = offset
        self.event = event
        self.reason = reason


class _LRUSet:
    """Bounded set with LRU eviction for dedup tracking."""

    def __init__(self, capacity: int = 10_000):
        self._cap = capacity
        self._data: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        if key in self._data:
            self._data.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            return
        self._data[key] = None
        if len(self._data) > self._cap:
            self._data.popitem(last=False)


class PerceptionIngress:
    """Validated perception event intake.

    Parameters:
        journal: durable event journal to append accepted events to.
        principal: the principal this ingress is scoped to.  Events from
            a different principal are rejected (cross-principal isolation).
        freshness_seconds: events with a source_timestamp older than this
            are marked stale.
        dedup_capacity: max number of event hashes to remember for dedup.
    """

    def __init__(
        self,
        journal: EventJournal,
        principal: Principal,
        freshness_seconds: int = 600,
        dedup_capacity: int = 10_000,
    ) -> None:
        self._journal = journal
        self._principal = principal
        self._freshness = timedelta(seconds=freshness_seconds)
        self._seen_hashes = _LRUSet(dedup_capacity)
        self._seen_ids: _LRUSet = _LRUSet(dedup_capacity)
        self._stats: Dict[str, int] = {v.value: 0 for v in IngressVerdict}

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def submit(self, event: PerceptionEvent) -> IngressResult:
        if event.principal.identity != self._principal.identity:
            self._stats[IngressVerdict.REJECTED_CROSS_PRINCIPAL.value] += 1
            return IngressResult(
                IngressVerdict.REJECTED_CROSS_PRINCIPAL,
                reason=(
                    f"principal mismatch: event={event.principal.identity} "
                    f"ingress={self._principal.identity}"
                ),
            )

        bounds_error = check_payload_bounds(event.payload)
        if bounds_error is not None:
            self._stats[IngressVerdict.REJECTED_OVERSIZED.value] += 1
            return IngressResult(
                IngressVerdict.REJECTED_OVERSIZED,
                event=event,
                reason=bounds_error,
            )

        dedup_key = event.raw_hash or event.id
        if dedup_key in self._seen_hashes:
            self._stats[IngressVerdict.REJECTED_DUPLICATE.value] += 1
            return IngressResult(
                IngressVerdict.REJECTED_DUPLICATE,
                event=event,
                reason=f"duplicate key: {dedup_key}",
            )

        if event.id in self._seen_ids:
            self._stats[IngressVerdict.REJECTED_DUPLICATE.value] += 1
            return IngressResult(
                IngressVerdict.REJECTED_DUPLICATE,
                event=event,
                reason=f"duplicate id: {event.id}",
            )

        now = datetime.now(timezone.utc)
        is_stale = False
        updates: Dict[str, Any] = {"received_at": now}
        if event.source_timestamp and (now - event.source_timestamp) > self._freshness:
            is_stale = True
            updates["stale"] = True

        self._seen_hashes.add(dedup_key)
        self._seen_ids.add(event.id)

        received = event.model_copy(update=updates)
        received.digest = received._make_digest()

        offset = self._journal.append(received)

        if is_stale:
            self._stats[IngressVerdict.ACCEPTED_STALE.value] += 1
            return IngressResult(
                IngressVerdict.ACCEPTED_STALE,
                offset=offset,
                event=received,
                reason="event accepted but marked stale",
            )

        self._stats[IngressVerdict.ACCEPTED.value] += 1
        return IngressResult(
            IngressVerdict.ACCEPTED,
            offset=offset,
            event=received,
        )

    def submit_raw(self, data: Dict[str, Any]) -> IngressResult:
        try:
            event = PerceptionEvent.model_validate(data)
        except (ValidationError, TypeError) as exc:
            self._stats[IngressVerdict.REJECTED_INVALID.value] += 1
            return IngressResult(
                IngressVerdict.REJECTED_INVALID,
                reason=str(exc),
            )
        return self.submit(event)
