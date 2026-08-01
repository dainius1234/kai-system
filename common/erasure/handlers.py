"""Layer handlers wiring real subsystems into the erasure coordinator.

Each handler is a ``(find, erase)`` pair over one subsystem.  They are
deliberately thin: the subsystem owns its own storage, and the handler
only knows how to scope a query to one subject.

``build_full_coordinator`` assembles the standard set covering all five
roadmap §16.30 layers.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from common.contracts.base import Principal
from common.contracts.erasure import ErasureLayer
from common.erasure.coordinator import ErasureCoordinator


# ── Source events (perception journal) ───────────────────────────────

def journal_handlers(journal) -> Tuple[Callable, Callable]:
    def find(subject: str) -> List[Tuple[str, Any]]:
        return [
            (entry.event.id, entry.event.payload)
            for entry in journal.replay()
            if entry.event.principal.identity == subject
        ]

    def erase(subject: str) -> int:
        return journal.erase_subject(subject)

    return find, erase


# ── World state (claims and snapshots) ───────────────────────────────

def world_state_handlers(
    store,
    event_owner: Optional[dict] = None,
) -> Tuple[Callable, Callable]:
    """Handlers for derived world-state claims.

    A world-state store is scoped to one principal, so every claim it
    holds carries *that* principal — not the principal of the event the
    claim was derived from.  Subject-scoping therefore has to follow
    lineage rather than the claim's own principal field.

    ``event_owner`` maps source event id → subject identity.  When
    supplied, a claim belongs to a subject if its lineage traces back to
    that subject's events.  Without it, the claim's own principal is
    used, which is correct when each subject has a dedicated store.

    Lineage is two hops, not one: a reducer emits ``event → evidence →
    claim``, so the claim's ``upstream_ids`` name evidence records, and
    it is the *evidence* that names the source event.  Following only
    the first hop would find nothing and silently leave derived claims
    in place — the exact failure §16.30 is about.
    """
    def _owner_of(record_id: str, seen: set) -> Optional[str]:
        """Resolve a lineage id back to its subject, walking evidence."""
        if record_id in seen:
            return None
        seen.add(record_id)

        owner = (event_owner or {}).get(record_id)
        if owner is not None:
            return owner

        evidence = store.evidence_by_id.get(record_id)
        if evidence is None:
            return None
        for upstream in evidence.provenance.upstream_ids:
            resolved = _owner_of(upstream, seen)
            if resolved is not None:
                return resolved
        return None

    def _belongs(claim, subject: str) -> bool:
        if event_owner is None:
            return claim.principal.identity == subject
        return any(
            _owner_of(upstream, set()) == subject
            for upstream in claim.provenance.upstream_ids
        )

    def find(subject: str) -> List[Tuple[str, Any]]:
        return [
            (claim.id, claim.claim_text)
            for claim in store.active_claims()
            if _belongs(claim, subject)
        ]

    def erase(subject: str) -> int:
        doomed = [
            claim.id for claim in list(store.active_claims())
            if _belongs(claim, subject)
        ]
        return store.erase_claims(doomed)

    return find, erase


# ── Proposals ────────────────────────────────────────────────────────

def proposal_handlers(proposals: dict) -> Tuple[Callable, Callable]:
    def find(subject: str) -> List[Tuple[str, Any]]:
        return [
            (pid, p.description)
            for pid, p in proposals.items()
            if p.principal.identity == subject
        ]

    def erase(subject: str) -> int:
        doomed = [
            pid for pid, p in proposals.items()
            if p.principal.identity == subject
        ]
        for pid in doomed:
            del proposals[pid]
        return len(doomed)

    return find, erase


# ── Learning derivatives (graded evidence) ───────────────────────────

def evidence_handlers(service) -> Tuple[Callable, Callable]:
    def find(subject: str) -> List[Tuple[str, Any]]:
        return [
            (record.id, record.content)
            for record in service.all_evidence(include_superseded=True)
            if record.principal.identity == subject
        ]

    def erase(subject: str) -> int:
        return service.erase_subject(subject)

    return find, erase


# ── Audit references (tombstoned, never removed outright) ────────────

def audit_handlers(audit_log: list) -> Tuple[Callable, Callable]:
    """Audit records are redacted in place, not deleted.

    The entry survives so the audit trail stays continuous; only the
    subject-identifying payload is replaced.
    """
    def find(subject: str) -> List[Tuple[str, Any]]:
        return [
            (entry["id"], entry.get("payload"))
            for entry in audit_log
            if entry.get("subject") == subject and not entry.get("redacted")
        ]

    def erase(subject: str) -> int:
        redacted = 0
        for entry in audit_log:
            if entry.get("subject") == subject and not entry.get("redacted"):
                entry["payload"] = None
                entry["redacted"] = True
                redacted += 1
        return redacted

    return find, erase


# ── Assembly ─────────────────────────────────────────────────────────

def build_full_coordinator(
    principal: Principal,
    journal=None,
    world_state=None,
    event_owner: Optional[dict] = None,
    proposals: Optional[dict] = None,
    evidence=None,
    audit_log: Optional[list] = None,
) -> ErasureCoordinator:
    """Coordinator covering every §16.30 layer that is supplied."""
    coordinator = ErasureCoordinator(principal=principal)

    if journal is not None:
        find, erase = journal_handlers(journal)
        coordinator.register_layer(ErasureLayer.SOURCE_EVENTS, find, erase)

    if world_state is not None:
        find, erase = world_state_handlers(world_state, event_owner)
        coordinator.register_layer(ErasureLayer.WORLD_STATE, find, erase)

    if proposals is not None:
        find, erase = proposal_handlers(proposals)
        coordinator.register_layer(ErasureLayer.PROPOSALS, find, erase)

    if evidence is not None:
        find, erase = evidence_handlers(evidence)
        coordinator.register_layer(
            ErasureLayer.LEARNING_DERIVATIVES, find, erase
        )

    if audit_log is not None:
        find, erase = audit_handlers(audit_log)
        coordinator.register_layer(
            ErasureLayer.AUDIT_REFERENCES, find, erase, tombstone=True
        )

    return coordinator
