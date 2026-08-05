"""End-to-end erasure tests — closes UH tracker gap G-06 / roadmap §16.30.

"End-to-end data deletion across source events, views, proposals,
audit-allowed references and learning derivatives."

The properties that matter:
  - deletion reaches **every** layer, not just the obvious one;
  - the audit trail **survives** as tombstones carrying no content;
  - erasure is **verified independently**, not trusted;
  - a partial failure reports PARTIAL, never COMPLETE;
  - other subjects' data is **untouched**.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import ContractState, Principal, Provenance, RiskTier
from common.contracts.perception import EventSource, PerceptionEvent
from common.contracts.action import ActionProposal
from common.contracts.autonomy import EvidenceGrade
from common.contracts.erasure import (
    ErasureLayer,
    ErasureStatus,
)
from common.perception_spine.journal import EventJournal
from common.world_state.snapshot_store import SnapshotStore
from common.autonomy.evidence_service import EvidenceService
from common.erasure.coordinator import (
    ErasureCoordinator,
    ErasureError,
    content_digest,
)
from common.erasure.handlers import build_full_coordinator

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


SUBJECT = "alice"
OTHER = "bob"

_tmpdir = tempfile.mkdtemp(prefix="erasure_test_")
_counter = 0


def _principal(identity: str = "kai") -> Principal:
    return Principal(identity=identity, role="system")


def _path() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"journal_{_counter}.jsonl")


def _event(identity: str, i: int) -> PerceptionEvent:
    return PerceptionEvent(
        source_type=EventSource.SYSTEM,
        event_type="test",
        payload={"secret": f"{identity}-data-{i}"},
        principal=_principal(identity),
        purpose="test",
        provenance=Provenance(source="sensor"),
        source_timestamp=datetime.now(timezone.utc),
        raw_hash=f"{identity}-{i}",
    )


def _proposal(identity: str, i: int) -> ActionProposal:
    return ActionProposal(
        action_type="t", description=f"{identity} proposal {i}",
        risk_tier=RiskTier.OBSERVE, rationale="t", alternatives=["n"],
        principal=_principal(identity), purpose="test",
        provenance=Provenance(source="test"),
    )


class _Fixture:
    """A populated system holding data for two subjects."""

    def __init__(self, per_subject: int = 3) -> None:
        self.journal = EventJournal(_path())
        self.store = SnapshotStore(principal=_principal(SUBJECT))
        self.evidence = EvidenceService(principal=_principal(SUBJECT))
        self.proposals: dict = {}
        self.audit: list = []
        self.event_owner: dict = {}

        for identity in (SUBJECT, OTHER):
            for i in range(per_subject):
                ev = _event(identity, i)
                self.event_owner[ev.id] = identity
                self.journal.append(ev)
                self.store.ingest_event(ev)

                p = _proposal(identity, i)
                self.proposals[p.id] = p

                self.evidence.record(
                    grade=EvidenceGrade.EXTERNAL_OBSERVED,
                    domain="d", task_type="t", observed_by="sensor",
                    content={"v": f"{identity}-{i}"},
                    principal=_principal(identity),
                    provenance=Provenance(source="sensor"),
                )

                self.audit.append({
                    "id": f"audit-{identity}-{i}",
                    "subject": identity,
                    "payload": {"detail": f"{identity} did thing {i}"},
                })

    def coordinator(self) -> ErasureCoordinator:
        return build_full_coordinator(
            principal=_principal(),
            journal=self.journal,
            world_state=self.store,
            event_owner=self.event_owner,
            proposals=self.proposals,
            evidence=self.evidence,
            audit_log=self.audit,
        )


# ═══════════════════════════════════════════════════════════════════
# 1. Full cascade
# ═══════════════════════════════════════════════════════════════════

def test_all_five_layers_registered():
    coordinator = _Fixture().coordinator()
    layers = set(coordinator.registered_layers())
    check("all_five_layers", layers == {
        ErasureLayer.SOURCE_EVENTS, ErasureLayer.WORLD_STATE,
        ErasureLayer.PROPOSALS, ErasureLayer.AUDIT_REFERENCES,
        ErasureLayer.LEARNING_DERIVATIVES,
    }, str(layers))


def test_erasure_reaches_every_layer():
    fx = _Fixture()
    receipt = fx.coordinator().erase(SUBJECT, "operator request", "dainius")

    check("erasure_complete", receipt.status == ErasureStatus.COMPLETE,
          f"status={receipt.status.value} residue={receipt.verification_residue}")
    check("erasure_verified", receipt.verified)
    check("erasure_no_residue", receipt.verification_residue == [])
    check("erasure_all_layers_reported", len(receipt.layer_results) == 5)

    by_layer = {r.layer: r for r in receipt.layer_results}
    check("events_erased", by_layer[ErasureLayer.SOURCE_EVENTS].records_erased == 3)
    check("claims_erased", by_layer[ErasureLayer.WORLD_STATE].records_erased >= 1)
    check("proposals_erased", by_layer[ErasureLayer.PROPOSALS].records_erased == 3)
    check("evidence_erased",
          by_layer[ErasureLayer.LEARNING_DERIVATIVES].records_erased == 3)
    check("audit_redacted",
          by_layer[ErasureLayer.AUDIT_REFERENCES].records_erased == 3)


def test_subject_data_actually_gone():
    fx = _Fixture()
    fx.coordinator().erase(SUBJECT, "request", "dainius")

    remaining_events = [
        e for e in fx.journal.replay()
        if e.event.principal.identity == SUBJECT
    ]
    check("no_subject_events", not remaining_events)

    remaining_proposals = [
        p for p in fx.proposals.values() if p.principal.identity == SUBJECT
    ]
    check("no_subject_proposals", not remaining_proposals)

    remaining_evidence = [
        r for r in fx.evidence.all_evidence(include_superseded=True)
        if r.principal.identity == SUBJECT
    ]
    check("no_subject_evidence", not remaining_evidence)

    # Claims carry the *store's* principal, not the source event's, so
    # subject-scoping must follow lineage: claim → evidence → event.
    evidence = fx.store.evidence_by_id

    def _from_subject(claim) -> bool:
        for ev_id in claim.provenance.upstream_ids:
            record = evidence.get(ev_id)
            if record is None:
                continue
            if any(fx.event_owner.get(u) == SUBJECT
                   for u in record.provenance.upstream_ids):
                return True
        return False

    active = [c for c in fx.store.active_claims() if _from_subject(c)]
    check("no_subject_claims", not active, f"{len(active)} claims survived")


def test_subject_payload_not_recoverable():
    """The erased content must not survive anywhere readable."""
    fx = _Fixture()
    fx.coordinator().erase(SUBJECT, "request", "dainius")

    raw = open(fx.journal._path, encoding="utf-8").read()
    check("journal_file_has_no_subject_data",
          "alice-data-" not in raw)
    check("journal_file_keeps_other_subject",
          "bob-data-" in raw)


# ═══════════════════════════════════════════════════════════════════
# 2. Other subjects untouched
# ═══════════════════════════════════════════════════════════════════

def test_other_subject_untouched():
    fx = _Fixture()
    fx.coordinator().erase(SUBJECT, "request", "dainius")

    other_events = [
        e for e in fx.journal.replay()
        if e.event.principal.identity == OTHER
    ]
    check("other_events_intact", len(other_events) == 3)

    other_proposals = [
        p for p in fx.proposals.values() if p.principal.identity == OTHER
    ]
    check("other_proposals_intact", len(other_proposals) == 3)

    other_evidence = [
        r for r in fx.evidence.all_evidence()
        if r.principal.identity == OTHER
    ]
    check("other_evidence_intact", len(other_evidence) == 3)

    other_audit = [
        a for a in fx.audit
        if a["subject"] == OTHER and not a.get("redacted")
    ]
    check("other_audit_intact", len(other_audit) == 3)
    check("other_audit_payload_intact",
          all(a["payload"] is not None for a in other_audit))


# ═══════════════════════════════════════════════════════════════════
# 3. Audit survives as tombstones carrying no content
# ═══════════════════════════════════════════════════════════════════

def test_audit_entries_survive_redacted():
    fx = _Fixture()
    fx.coordinator().erase(SUBJECT, "request", "dainius")

    subject_audit = [a for a in fx.audit if a["subject"] == SUBJECT]
    check("audit_entries_retained", len(subject_audit) == 3)
    check("audit_marked_redacted", all(a["redacted"] for a in subject_audit))
    check("audit_payload_removed",
          all(a["payload"] is None for a in subject_audit))


def test_tombstones_created_for_audit_layer():
    fx = _Fixture()
    coordinator = fx.coordinator()
    receipt = coordinator.erase(SUBJECT, "request", "dainius")

    tombstones = coordinator.tombstones_for(receipt.request_id)
    check("tombstones_created", len(tombstones) == 3, f"got {len(tombstones)}")
    check("tombstones_are_audit_layer",
          all(t.layer == ErasureLayer.AUDIT_REFERENCES for t in tombstones))
    check("receipt_counts_tombstones", receipt.total_tombstones == 3)


def test_tombstone_carries_no_content():
    """A tombstone must not become a backdoor copy of deleted data."""
    fx = _Fixture()
    coordinator = fx.coordinator()
    receipt = coordinator.erase(SUBJECT, "request", "dainius")

    for tombstone in coordinator.tombstones_for(receipt.request_id):
        serialised = tombstone.model_dump_json()
        check(f"tombstone_{tombstone.original_id}_no_content",
              "did thing" not in serialised and "alice-data" not in serialised)
        check(f"tombstone_{tombstone.original_id}_has_digest",
              len(tombstone.content_digest) == 64)

    fields = set(type(receipt).model_fields)
    check("receipt_has_no_content_field", "content" not in fields)


def test_tombstone_digest_is_stable():
    d1 = content_digest({"a": 1, "b": 2})
    d2 = content_digest({"b": 2, "a": 1})
    check("digest_key_order_stable", d1 == d2)
    check("digest_differs_on_content", d1 != content_digest({"a": 1, "b": 3}))


def test_non_audit_layers_have_no_tombstones():
    fx = _Fixture()
    coordinator = fx.coordinator()
    receipt = coordinator.erase(SUBJECT, "request", "dainius")

    by_layer = {r.layer: r for r in receipt.layer_results}
    for layer in (ErasureLayer.SOURCE_EVENTS, ErasureLayer.PROPOSALS,
                  ErasureLayer.LEARNING_DERIVATIVES):
        check(f"no_tombstone_{layer.value}",
              by_layer[layer].tombstones_created == 0)


# ═══════════════════════════════════════════════════════════════════
# 4. Independent verification
# ═══════════════════════════════════════════════════════════════════

def test_verification_detects_residue():
    """A handler that reports success but leaves data is caught."""
    coordinator = ErasureCoordinator(principal=_principal())
    leftover = {"r1": "alice data"}

    coordinator.register_layer(
        ErasureLayer.PROPOSALS,
        find=lambda s: [(k, v) for k, v in leftover.items()],
        erase=lambda s: 1,           # claims success...
    )                                 # ...but never removes anything

    receipt = coordinator.erase(SUBJECT, "request", "dainius")
    check("lying_handler_caught", receipt.status == ErasureStatus.PARTIAL,
          f"status={receipt.status.value}")
    check("lying_handler_not_verified", not receipt.verified)
    check("residue_reported", len(receipt.verification_residue) == 1)
    check("residue_identifies_layer",
          receipt.verification_residue[0].startswith("proposals:"))


def test_failing_handler_reports_failed():
    coordinator = ErasureCoordinator(principal=_principal())

    def exploding(subject: str) -> int:
        raise RuntimeError("storage offline")

    coordinator.register_layer(
        ErasureLayer.SOURCE_EVENTS,
        find=lambda s: [],
        erase=exploding,
    )

    receipt = coordinator.erase(SUBJECT, "request", "dainius")
    check("failing_handler_status", receipt.status == ErasureStatus.FAILED)
    check("failing_handler_not_verified", not receipt.verified)
    check("failure_recorded",
          "storage offline" in (receipt.layer_results[0].error or ""))


def test_verify_erased_is_standalone():
    fx = _Fixture()
    coordinator = fx.coordinator()

    before = coordinator.verify_erased(SUBJECT)
    check("residue_before_erasure", len(before) > 0)

    coordinator.erase(SUBJECT, "request", "dainius")
    after = coordinator.verify_erased(SUBJECT)
    check("no_residue_after_erasure", after == [], str(after))

    other = coordinator.verify_erased(OTHER)
    check("other_subject_still_present", len(other) > 0)


# ═══════════════════════════════════════════════════════════════════
# 5. Scoping and validation
# ═══════════════════════════════════════════════════════════════════

def test_partial_layer_selection():
    fx = _Fixture()
    coordinator = fx.coordinator()
    receipt = coordinator.erase(
        SUBJECT, "request", "dainius",
        layers=[ErasureLayer.PROPOSALS],
    )

    check("single_layer_reported", len(receipt.layer_results) == 1)
    check("single_layer_erased",
          receipt.layer_results[0].records_erased == 3)

    remaining = [
        e for e in fx.journal.replay()
        if e.event.principal.identity == SUBJECT
    ]
    check("unselected_layer_untouched", len(remaining) == 3)


def test_unregistered_layer_rejected():
    coordinator = ErasureCoordinator(principal=_principal())
    coordinator.register_layer(
        ErasureLayer.PROPOSALS, find=lambda s: [], erase=lambda s: 0
    )
    try:
        coordinator.erase(SUBJECT, "r", "dainius",
                          layers=[ErasureLayer.SOURCE_EVENTS])
        check("unregistered_layer_rejected", False, "should have raised")
    except ErasureError as e:
        check("unregistered_layer_rejected", "no handler" in str(e))


def test_duplicate_layer_registration_rejected():
    coordinator = ErasureCoordinator(principal=_principal())
    coordinator.register_layer(
        ErasureLayer.PROPOSALS, find=lambda s: [], erase=lambda s: 0
    )
    try:
        coordinator.register_layer(
            ErasureLayer.PROPOSALS, find=lambda s: [], erase=lambda s: 0
        )
        check("duplicate_layer_rejected", False, "should have raised")
    except ErasureError as e:
        check("duplicate_layer_rejected", "already registered" in str(e))


def test_erasure_validation():
    coordinator = ErasureCoordinator(principal=_principal())
    coordinator.register_layer(
        ErasureLayer.PROPOSALS, find=lambda s: [], erase=lambda s: 0
    )
    for name, args in [
        ("empty_subject", ("", "reason", "dainius")),
        ("anonymous", ("alice", "reason", "")),
        ("no_reason", ("alice", "", "dainius")),
    ]:
        try:
            coordinator.erase(*args)
            check(f"erasure_{name}_rejected", False, "should have raised")
        except ErasureError:
            check(f"erasure_{name}_rejected", True)


def test_receipt_is_auditable():
    fx = _Fixture()
    coordinator = fx.coordinator()
    receipt = coordinator.erase(SUBJECT, "GDPR request", "dainius")

    check("receipt_names_subject", receipt.subject_identity == SUBJECT)
    check("receipt_has_request_id", receipt.request_id is not None)
    check("receipt_timestamped", receipt.completed_at is not None)
    check("receipt_counts_total", receipt.total_erased >= 12,
          f"got {receipt.total_erased}")
    check("receipt_retained", len(coordinator.receipts) == 1)
    check("receipt_digest_valid", receipt.verify_digest())


def test_repeated_erasure_is_idempotent():
    fx = _Fixture()
    coordinator = fx.coordinator()

    first = coordinator.erase(SUBJECT, "request", "dainius")
    second = coordinator.erase(SUBJECT, "request again", "dainius")

    check("first_erased_records", first.total_erased > 0)
    check("second_erased_nothing", second.total_erased == 0)
    check("second_still_complete", second.status == ErasureStatus.COMPLETE)
    check("second_verified", second.verified)


# ═══════════════════════════════════════════════════════════════════
# 6. Journal-level erasure detail
# ═══════════════════════════════════════════════════════════════════

def test_journal_erasure_preserves_other_offsets():
    """Erasure must not renumber surviving records."""
    journal = EventJournal(_path())
    journal.append(_event(OTHER, 0))
    journal.append(_event(SUBJECT, 0))
    journal.append(_event(OTHER, 1))

    before = {
        e.event.principal.identity: e.offset
        for e in journal.replay()
        if e.event.principal.identity == OTHER
    }
    removed = journal.erase_subject(SUBJECT)
    check("journal_removed_one", removed == 1)

    after = [e for e in journal.replay()]
    check("journal_two_remain", len(after) == 2)
    check("journal_offsets_preserved",
          [e.offset for e in after] == [0, 2],
          str([e.offset for e in after]))


def test_journal_erasure_on_empty_is_safe():
    journal = EventJournal(_path())
    check("empty_journal_erase_zero", journal.erase_subject(SUBJECT) == 0)


def test_evidence_erasure_removes_not_supersedes():
    """Erasure must not leave the original readable via a chain."""
    service = EvidenceService(principal=_principal(SUBJECT))
    original = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED, domain="d", task_type="t",
        observed_by="sensor", content={"secret": "alice-secret"},
        principal=_principal(SUBJECT),
    )
    service.correct(original.id, EvidenceGrade.EXTERNAL_OBSERVED,
                    {"secret": "alice-secret-v2"}, "sensor")

    removed = service.erase_subject(SUBJECT)
    check("evidence_chain_removed", removed == 2, f"got {removed}")
    check("evidence_original_gone", service.get(original.id) is None)
    check("evidence_store_empty", service.count == 0)
    check("evidence_lineage_empty", service.lineage(original.id) == [])


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_all_five_layers_registered()
    test_erasure_reaches_every_layer()
    test_subject_data_actually_gone()
    test_subject_payload_not_recoverable()
    test_other_subject_untouched()
    test_audit_entries_survive_redacted()
    test_tombstones_created_for_audit_layer()
    test_tombstone_carries_no_content()
    test_tombstone_digest_is_stable()
    test_non_audit_layers_have_no_tombstones()
    test_verification_detects_residue()
    test_failing_handler_reports_failed()
    test_verify_erased_is_standalone()
    test_partial_layer_selection()
    test_unregistered_layer_rejected()
    test_duplicate_layer_registration_rejected()
    test_erasure_validation()
    test_receipt_is_auditable()
    test_repeated_erasure_is_idempotent()
    test_journal_erasure_preserves_other_offsets()
    test_journal_erasure_on_empty_is_safe()
    test_evidence_erasure_removes_not_supersedes()

    print(f"\n{'='*60}")
    print(f"Erasure Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
