"""Scoped world-state snapshot store — immutable snapshots with retention.

Produces WorldStateSnapshot instances from reduced claims and evidence.
Snapshots are:

  - immutable once created (no in-place mutation)
  - scoped by principal, purpose, and data classification
  - reproducible from the same event sequence and reducer revision
  - digest-verified
  - subject to bounded retention with deletion lineage

Conflict semantics:
  - claims from different domains or independence groups with contradicting
    content are preserved as conflicts, not averaged away
  - stale claims are kept but marked with freshness=STALE
  - superseded claims are moved to state=SUPERSEDED and excluded from
    active views
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    VerificationVerdict,
)
from common.contracts.world_state import (
    Claim,
    EvidenceRecord,
    FreshnessStatus,
    WorldStateSnapshot,
)
from common.contracts.perception import PerceptionEvent

from common.world_state.reducers import ReducerOutput, ReducerRegistry


class SnapshotStore:
    """Produces and stores immutable world-state snapshots.

    Parameters:
        principal: the principal scope for all snapshots
        purpose: the purpose scope for all snapshots
        max_snapshots: bounded retention — oldest snapshots beyond this
            count are deleted (with lineage recorded)
        max_claims_per_domain: cap on active claims per domain to prevent
            unbounded growth
    """

    def __init__(
        self,
        principal: Principal,
        purpose: str = "world_state",
        classification: str = "internal",
        max_snapshots: int = 100,
        max_claims_per_domain: int = 50,
    ) -> None:
        self._principal = principal
        self._purpose = purpose
        self._classification = classification
        self._max_snapshots = max_snapshots
        self._max_claims_per_domain = max_claims_per_domain
        self._registry = ReducerRegistry()

        self._claims: Dict[str, Claim] = OrderedDict()
        self._evidence: Dict[str, EvidenceRecord] = OrderedDict()
        self._snapshots: List[WorldStateSnapshot] = []
        self._deleted_snapshot_ids: List[str] = []
        self._superseded_claim_ids: Set[str] = set()
        self._ingested_event_ids: List[str] = []
        self._event_offset: int = 0

    @property
    def reducer_registry(self) -> ReducerRegistry:
        return self._registry

    def ingest_event(self, event: PerceptionEvent) -> ReducerOutput:
        output = self._registry.reduce(event, self._principal)
        for ev in output.evidence:
            self._evidence[ev.id] = ev
        for claim in output.claims:
            self._supersede_conflicting(claim)
            self._claims[claim.id] = claim
            self._enforce_domain_cap(claim.domain)
        self._ingested_event_ids.append(event.id)
        self._event_offset += 1
        return output

    def ingest_events(self, events: List[PerceptionEvent]) -> ReducerOutput:
        all_claims: List[Claim] = []
        all_evidence: List[EvidenceRecord] = []
        for event in events:
            out = self.ingest_event(event)
            all_claims.extend(out.claims)
            all_evidence.extend(out.evidence)
        return ReducerOutput(claims=all_claims, evidence=all_evidence)

    def _supersede_conflicting(self, new_claim: Claim) -> None:
        """Mark older claims in the same domain as superseded if they
        are from the same independence group.  Claims from different
        independence groups are preserved as conflicts."""
        new_group = new_claim.provenance.independence_group
        to_supersede: List[str] = []

        for cid, existing in self._claims.items():
            if cid in self._superseded_claim_ids:
                continue
            if existing.domain != new_claim.domain:
                continue
            existing_group = existing.provenance.independence_group
            if new_group is not None and existing_group == new_group:
                to_supersede.append(cid)

        for cid in to_supersede:
            old = self._claims[cid]
            self._claims[cid] = old.model_copy(
                update={
                    "state": ContractState.SUPERSEDED,
                    "supersedes": None,
                    "digest": None,
                }
            )
            self._claims[cid].digest = self._claims[cid]._make_digest()
            self._superseded_claim_ids.add(cid)
            new_claim_updated = new_claim.model_copy(
                update={"supersedes": cid, "digest": None}
            )
            new_claim_updated.digest = new_claim_updated._make_digest()
            new_claim.__dict__.update(new_claim_updated.__dict__)

    def _enforce_domain_cap(self, domain: str) -> None:
        domain_claims = [
            (cid, c) for cid, c in self._claims.items()
            if c.domain == domain and cid not in self._superseded_claim_ids
        ]
        excess = len(domain_claims) - self._max_claims_per_domain
        if excess <= 0:
            return
        for cid, c in domain_claims[:excess]:
            self._claims[cid] = c.model_copy(
                update={"state": ContractState.SUPERSEDED, "digest": None}
            )
            self._claims[cid].digest = self._claims[cid]._make_digest()
            self._superseded_claim_ids.add(cid)

    def active_claims(self) -> List[Claim]:
        return [
            c for cid, c in self._claims.items()
            if cid not in self._superseded_claim_ids
            and c.state == ContractState.ACTIVE
        ]

    def conflicts(self) -> List[Dict[str, Any]]:
        by_domain: Dict[str, List[Claim]] = {}
        for claim in self.active_claims():
            by_domain.setdefault(claim.domain, []).append(claim)

        conflicts: List[Dict[str, Any]] = []
        for domain, claims in by_domain.items():
            groups: Dict[Optional[str], List[Claim]] = {}
            for c in claims:
                g = c.provenance.independence_group
                groups.setdefault(g, []).append(c)
            if len(groups) > 1:
                conflicts.append({
                    "domain": domain,
                    "groups": {
                        str(g): [c.id for c in cs] for g, cs in groups.items()
                    },
                })
        return conflicts

    def take_snapshot(self) -> WorldStateSnapshot:
        now = datetime.now(timezone.utc)
        active = self.active_claims()
        active_evidence_ids: Set[str] = set()
        for c in active:
            active_evidence_ids.update(c.evidence_ids)
        evidence = [
            e for eid, e in self._evidence.items() if eid in active_evidence_ids
        ]

        snapshot = WorldStateSnapshot(
            snapshot_at=now,
            scope_principal=self._principal.identity,
            scope_purpose=self._purpose,
            scope_classification=self._classification,
            claims=active,
            evidence=evidence,
            conflicts=self.conflicts(),
            degraded_sources=[
                c.provenance.source for c in active
                if c.freshness == FreshnessStatus.STALE
            ],
            event_sequence_digest=self._compute_event_sequence_digest(),
            principal=self._principal,
            purpose=self._purpose,
            provenance=Provenance(
                source=f"snapshot_store:{self._registry.revision}",
            ),
        )

        self._snapshots.append(snapshot)
        self._enforce_retention()
        return snapshot

    def _compute_event_sequence_digest(self) -> str:
        raw = json.dumps(self._ingested_event_ids, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _enforce_retention(self) -> None:
        while len(self._snapshots) > self._max_snapshots:
            removed = self._snapshots.pop(0)
            self._deleted_snapshot_ids.append(removed.id)

    def get_snapshot(self, index: int = -1) -> Optional[WorldStateSnapshot]:
        if not self._snapshots:
            return None
        try:
            return self._snapshots[index]
        except IndexError:
            return None

    def snapshot_count(self) -> int:
        return len(self._snapshots)

    @property
    def deleted_snapshot_ids(self) -> List[str]:
        return list(self._deleted_snapshot_ids)

    def scoped_view(
        self,
        principal_identity: str,
        purpose: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> Optional[WorldStateSnapshot]:
        if principal_identity != self._principal.identity:
            return None
        latest = self.get_snapshot()
        if latest is None:
            return None
        if purpose and purpose != self._purpose:
            return None
        if classification and classification != self._classification:
            return None
        return latest

    def replay_from_events(
        self, events: List[PerceptionEvent]
    ) -> WorldStateSnapshot:
        store = SnapshotStore(
            principal=self._principal,
            purpose=self._purpose,
            classification=self._classification,
            max_snapshots=self._max_snapshots,
            max_claims_per_domain=self._max_claims_per_domain,
        )
        store.ingest_events(events)
        return store.take_snapshot()
