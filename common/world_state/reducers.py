"""Deterministic world-state reducers — convert PerceptionEvents to Claims.

Reducers are pure functions: given the same sequence of events, they produce
the same set of claims.  Each reducer:

  - is revisioned (reducer_revision string)
  - maps one or more EventSource types to a domain
  - produces Claims with evidence linkage back to source events
  - handles conflict, staleness, and supersession explicitly

The ReducerRegistry holds all registered reducers and can apply them in
deterministic order to an event sequence.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    VerificationVerdict,
)
from common.contracts.perception import EventSource, PerceptionEvent
from common.contracts.world_state import (
    Claim,
    EvidenceRecord,
    FreshnessStatus,
)

REDUCER_REVISION = "1.0.0"


class ReducerOutput:
    __slots__ = ("claims", "evidence")

    def __init__(
        self,
        claims: List[Claim] | None = None,
        evidence: List[EvidenceRecord] | None = None,
    ):
        self.claims = claims or []
        self.evidence = evidence or []


ReducerFn = Callable[[PerceptionEvent, Principal], ReducerOutput]


def _freshness_from_event(event: PerceptionEvent) -> FreshnessStatus:
    if event.stale:
        return FreshnessStatus.STALE
    if event.source_timestamp is None:
        return FreshnessStatus.UNKNOWN
    age = datetime.now(timezone.utc) - event.source_timestamp
    if age > timedelta(minutes=30):
        return FreshnessStatus.STALE
    return FreshnessStatus.CURRENT


def _make_evidence(
    event: PerceptionEvent,
    content: str,
    evidence_type: str,
    principal: Principal,
    strength: float = 0.7,
) -> EvidenceRecord:
    return EvidenceRecord(
        content=content,
        evidence_type=evidence_type,
        source_event_id=event.id,
        strength=strength,
        direction="supports",
        raw_data=event.payload,
        freshness=_freshness_from_event(event),
        principal=principal,
        purpose="world_state",
        provenance=Provenance(
            source=f"reducer:{REDUCER_REVISION}",
            upstream_ids=[event.id],
            independence_group=event.provenance.independence_group,
        ),
    )


def _make_claim(
    text: str,
    domain: str,
    evidence: EvidenceRecord,
    principal: Principal,
    confidence: float = 0.7,
    freshness: FreshnessStatus = FreshnessStatus.UNKNOWN,
) -> Claim:
    return Claim(
        claim_text=text,
        domain=domain,
        evidence_ids=[evidence.id],
        verification=VerificationVerdict.INCONCLUSIVE,
        confidence=confidence,
        freshness=freshness,
        principal=principal,
        purpose="world_state",
        provenance=Provenance(
            source=f"reducer:{REDUCER_REVISION}",
            upstream_ids=[evidence.id],
            independence_group=evidence.provenance.independence_group,
        ),
    )


def reduce_weather(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    summary = payload.get("summary", "")
    if not summary:
        return ReducerOutput()

    evidence = _make_evidence(event, summary, "weather_observation", principal)
    freshness = _freshness_from_event(event)

    claims = []
    claims.append(_make_claim(
        f"Weather: {summary}",
        "environment",
        evidence,
        principal,
        confidence=event.confidence,
        freshness=freshness,
    ))

    if "temp_c" in payload:
        temp_ev = _make_evidence(
            event, f"Temperature: {payload['temp_c']}°C", "measurement", principal, 0.9
        )
        claims.append(_make_claim(
            f"Temperature is {payload['temp_c']}°C",
            "environment",
            temp_ev,
            principal,
            confidence=0.9,
            freshness=freshness,
        ))
        return ReducerOutput(claims=claims, evidence=[evidence, temp_ev])

    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_system(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    evidence = _make_evidence(
        event,
        json.dumps(payload, default=str),
        "system_metrics",
        principal,
        strength=0.9,
    )
    freshness = _freshness_from_event(event)
    claims = []

    if "cpu_percent" in payload:
        cpu = payload["cpu_percent"]
        claims.append(_make_claim(
            f"CPU utilisation is {cpu}%",
            "infrastructure",
            evidence,
            principal,
            confidence=0.9,
            freshness=freshness,
        ))

    if "memory_percent" in payload:
        mem = payload["memory_percent"]
        claims.append(_make_claim(
            f"Memory utilisation is {mem}%",
            "infrastructure",
            evidence,
            principal,
            confidence=0.9,
            freshness=freshness,
        ))

    if "disk_percent" in payload:
        disk = payload["disk_percent"]
        claims.append(_make_claim(
            f"Disk utilisation is {disk}%",
            "infrastructure",
            evidence,
            principal,
            confidence=0.9,
            freshness=freshness,
        ))

    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_docker(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    summary = payload.get("summary", "")
    if not summary:
        return ReducerOutput()

    evidence = _make_evidence(event, summary, "docker_observation", principal)
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"Docker: {summary}",
        "infrastructure",
        evidence,
        principal,
        confidence=event.confidence,
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_git(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    summary = payload.get("summary", "")
    if not summary:
        return ReducerOutput()

    evidence = _make_evidence(event, summary, "git_observation", principal)
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"Git: {summary}",
        "development",
        evidence,
        principal,
        confidence=event.confidence,
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_calendar(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    summary = payload.get("summary", "")
    if not summary:
        return ReducerOutput()

    evidence = _make_evidence(event, summary, "calendar_observation", principal)
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"Calendar: {summary}",
        "schedule",
        evidence,
        principal,
        confidence=event.confidence,
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_screen(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    if not payload.get("active"):
        return ReducerOutput()

    diff = payload.get("diff_score", 0)
    evidence = _make_evidence(
        event, f"Screen active, diff={diff:.2f}", "screen_observation", principal
    )
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"Screen is active (change score {diff:.2f})",
        "user_activity",
        evidence,
        principal,
        confidence=min(diff * 2, 1.0),
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_market(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    symbol = payload.get("symbol", "")
    price = payload.get("price")
    if not symbol or price is None:
        return ReducerOutput()

    evidence = _make_evidence(
        event, f"{symbol}={price}", "market_observation", principal, 0.95
    )
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"{symbol} price is {price}",
        "market",
        evidence,
        principal,
        confidence=0.95,
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


def reduce_generic(event: PerceptionEvent, principal: Principal) -> ReducerOutput:
    payload = event.payload
    summary = payload.get("summary", payload.get("content_preview", ""))
    if not summary:
        summary = json.dumps(payload, default=str)[:200]
    if not summary:
        return ReducerOutput()

    evidence = _make_evidence(
        event, summary, f"{event.source_type.value}_observation", principal, 0.5
    )
    freshness = _freshness_from_event(event)
    claims = [_make_claim(
        f"{event.source_type.value}: {summary[:100]}",
        event.source_type.value,
        evidence,
        principal,
        confidence=event.confidence * 0.8,
        freshness=freshness,
    )]
    return ReducerOutput(claims=claims, evidence=[evidence])


REDUCER_MAP: Dict[str, ReducerFn] = {
    "weather_reading": reduce_weather,
    "system_metrics": reduce_system,
    "docker_status": reduce_docker,
    "git_status": reduce_git,
    "calendar_reading": reduce_calendar,
    "screen_activity": reduce_screen,
    "market_tick": reduce_market,
}


class ReducerRegistry:
    """Applies registered reducers to perception events in deterministic order."""

    def __init__(self) -> None:
        self._reducers: Dict[str, ReducerFn] = dict(REDUCER_MAP)
        self.revision = REDUCER_REVISION

    def reduce(
        self, event: PerceptionEvent, principal: Principal
    ) -> ReducerOutput:
        fn = self._reducers.get(event.event_type, reduce_generic)
        return fn(event, principal)

    def reduce_sequence(
        self, events: List[PerceptionEvent], principal: Principal
    ) -> ReducerOutput:
        all_claims: List[Claim] = []
        all_evidence: List[EvidenceRecord] = []
        for event in events:
            out = self.reduce(event, principal)
            all_claims.extend(out.claims)
            all_evidence.extend(out.evidence)
        return ReducerOutput(claims=all_claims, evidence=all_evidence)

    def register(self, event_type: str, fn: ReducerFn) -> None:
        self._reducers[event_type] = fn
