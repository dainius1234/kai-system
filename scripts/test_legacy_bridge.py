"""Legacy trust bridge tests — closes UH tracker gap G-04.

Two authority systems coexisted: the legacy ``TrustLevel`` scalar and
UH-8's scoped grants.  The bridge unifies them under one rule that these
tests exist to pin down:

    **the legacy scalar may only deny, never grant.**

Whatever the old system says, it cannot widen what the new authority
permits.  A bridge that could would recreate the "two authorities, most
permissive wins" problem it was built to remove.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# The two tests at the bottom call `gate_autonomous_action` for real,
# and that function appends a signed, hash-chained AUTONOMOUS_ACTION
# event to whatever ledger it resolves. Until 2026-08-05 that resolved
# to the repository's own `data/trust-ledger/events.jsonl` with no way
# to redirect it, so every `make test-uh` wrote two live events into a
# tracked file — noticed only when a commit meant to touch six files
# carried a seventh.
#
# Scoped through `_Env` rather than set at module scope. A module-scope
# `os.environ[...] = ...` is the precise defect the isolation plugin
# exists to catch: the first file to run wins and every file after it
# inherits a value it did not choose.
_SCRATCH_LEDGER = os.path.join(
    tempfile.gettempdir(), "kai-test-legacy-bridge-ledger.jsonl")

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import AutonomyLevel, EvidenceGrade
from common.autonomy.authority import AutonomyAuthority
from common.autonomy.calibration import CalibrationTracker
from common.autonomy.evidence_service import EvidenceService
from common.autonomy.legacy_bridge import (
    CAPABILITY_DOMAINS,
    ENFORCE_ENV,
    LegacyTrustBridge,
    enforcing,
)

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


def _principal() -> Principal:
    return Principal(identity="kai", role="system")


class _Env:
    def __init__(self, **overrides):
        self._o = overrides
        self._saved = {}

    def __enter__(self):
        for k, v in self._o.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _bridge(with_grant: bool = False) -> LegacyTrustBridge:
    principal = _principal()
    evidence = EvidenceService(principal=principal)
    calibration = CalibrationTracker(principal=principal)

    if with_grant:
        for i in range(30):
            ev = evidence.record(
                grade=EvidenceGrade.VERIFIED_OUTCOME, domain="trading",
                task_type="trade", observed_by="portfolio-verifier",
                provenance=Provenance(source="verifier:portfolio"),
            )
            calibration.observe("trade", "trading", "r1", 0.9, ev, True)

    authority = AutonomyAuthority(principal, evidence, calibration)

    if with_grant:
        authority.grant(
            level=AutonomyLevel.A2_REVERSIBLE,
            capability="paper_trade_open", domain="trading",
            task_type="trade", revision="r1", granted_by="dainius",
            max_invocations=100, independent_verifier_count=2,
        )

    return LegacyTrustBridge(authority, principal)


# ═══════════════════════════════════════════════════════════════════
# 1. The core rule: legacy can only subtract
# ═══════════════════════════════════════════════════════════════════

def test_legacy_cannot_grant_what_scoped_denies():
    """The central invariant, in enforcing mode."""
    with _Env(**{ENFORCE_ENV: "true"}):
        bridge = _bridge(with_grant=False)
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trust OK",
        )
        check("legacy_allow_cannot_override_scoped_deny", not allowed)
        check("denial_names_scoped", "scoped authority denies" in reason)


def test_legacy_deny_still_denies_when_scoped_allows():
    with _Env(**{ENFORCE_ENV: "true"}):
        bridge = _bridge(with_grant=True)
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=False,
            legacy_reason="trust too low",
        )
        check("legacy_deny_wins", not allowed)
        check("denial_names_legacy", "legacy gate denies" in reason)


def test_both_allow_permits():
    with _Env(**{ENFORCE_ENV: "true"}):
        bridge = _bridge(with_grant=True)
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trust OK",
        )
        check("both_allow_permits", allowed, reason)
        check("permit_names_grant", "granted at" in reason)


def test_enforcing_never_widens():
    """Across every combination, enforcing is never more permissive."""
    with _Env(**{ENFORCE_ENV: "true"}):
        for has_grant in (True, False):
            for legacy in (True, False):
                bridge = _bridge(with_grant=has_grant)
                allowed, _ = bridge.gate(
                    "paper_trade_open", legacy_allowed=legacy,
                    legacy_reason="r",
                )
                check(
                    f"never_widens_grant{has_grant}_legacy{legacy}",
                    allowed == (has_grant and legacy),
                )


# ═══════════════════════════════════════════════════════════════════
# 2. Advisory mode — observe without changing behaviour
# ═══════════════════════════════════════════════════════════════════

def test_advisory_preserves_legacy_decision():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=False)

        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trust OK",
        )
        check("advisory_keeps_legacy_allow", allowed)
        check("advisory_keeps_legacy_reason", reason == "trust OK")

        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=False, legacy_reason="nope",
        )
        check("advisory_keeps_legacy_deny", not allowed)
        check("advisory_keeps_deny_reason", reason == "nope")


def test_advisory_records_disagreement():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=False)
        bridge.gate("paper_trade_open", legacy_allowed=True, legacy_reason="ok")

        check("disagreement_recorded", len(bridge.disagreements) == 1)
        d = bridge.disagreements[0]
        check("disagreement_capability", d.capability == "paper_trade_open")
        check("disagreement_legacy_allowed", d.legacy_allowed)
        check("disagreement_scoped_denied", not d.scoped_allowed)
        check("disagreement_has_reason", "no active grant" in d.scoped_reason)


def test_agreement_records_nothing():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=True)
        bridge.gate("paper_trade_open", legacy_allowed=True, legacy_reason="ok")
        check("agreement_no_disagreement", bridge.disagreements == [])
        check("agreement_counted", bridge.consulted == 1)


def test_mode_flag_parsing():
    for value, expected in [("true", True), ("1", True), ("yes", True),
                            ("false", False), ("0", False), ("", False),
                            (None, False), ("maybe", False)]:
        with _Env(**{ENFORCE_ENV: value}):
            check(f"enforce_flag_{value or 'unset'}", enforcing() == expected)


# ═══════════════════════════════════════════════════════════════════
# 3. Scoped decisions
# ═══════════════════════════════════════════════════════════════════

def test_no_grant_means_denied():
    bridge = _bridge(with_grant=False)
    allowed, reason = bridge.scoped_decision("paper_trade_open")
    check("no_grant_denies", not allowed)
    check("no_grant_says_a0", "A0_NONE" in reason)


def test_grant_permits():
    bridge = _bridge(with_grant=True)
    allowed, reason = bridge.scoped_decision("paper_trade_open")
    check("grant_permits", allowed)
    check("grant_names_level", "A2_REVERSIBLE" in reason)


def test_unmapped_capability_denied():
    """A capability with no scoped mapping cannot be authorised."""
    bridge = _bridge(with_grant=True)
    allowed, reason = bridge.scoped_decision("some_new_capability")
    check("unmapped_denied", not allowed)
    check("unmapped_explains", "no scoped mapping" in reason)


def test_explicit_domain_override():
    bridge = _bridge(with_grant=True)
    allowed, _ = bridge.scoped_decision("paper_trade_open", domain="medical")
    check("wrong_domain_denied", not allowed)

    allowed, _ = bridge.scoped_decision("paper_trade_open", domain="trading")
    check("right_domain_allowed", allowed)


def test_revoked_grant_denies():
    bridge = _bridge(with_grant=True)
    allowed, _ = bridge.scoped_decision("paper_trade_open")
    check("granted_before_revoke", allowed)

    authority = bridge._authority
    for grant in authority.active_grants():
        authority.revoke(grant.id, "test")

    allowed, reason = bridge.scoped_decision("paper_trade_open")
    check("revoked_denies", not allowed)
    check("revoked_says_no_grant", "no active grant" in reason)


def test_capability_map_covers_financial():
    """The financial capabilities the audit flagged must be mapped."""
    for capability in ("paper_trade_open", "paper_trade_close", "auto_trade"):
        check(f"mapped_{capability}", capability in CAPABILITY_DOMAINS)
        check(f"{capability}_domain_trading",
              CAPABILITY_DOMAINS[capability][1] == "trading")


# ═══════════════════════════════════════════════════════════════════
# 4. Migration telemetry
# ═══════════════════════════════════════════════════════════════════

def test_migration_report_blocks_premature_enforcement():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=False)
        bridge.gate("paper_trade_open", legacy_allowed=True, legacy_reason="ok")

        report = bridge.migration_report()
        check("report_mode_advisory", report["mode"] == "advisory")
        check("report_counts_consulted", report["consulted"] == 1)
        check("report_counts_breakage",
              report["would_be_denied_under_enforcement"] == 1)
        check("report_not_ready", report["ready_to_enforce"] is False)


def test_migration_report_signals_readiness():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=True)
        bridge.gate("paper_trade_open", legacy_allowed=True, legacy_reason="ok")

        report = bridge.migration_report()
        check("report_no_disagreement", report["disagreements"] == 0)
        check("report_ready", report["ready_to_enforce"] is True)


def test_report_ready_requires_traffic():
    """Zero observations is not evidence of safety."""
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=True)
        report = bridge.migration_report()
        check("no_traffic_not_ready", report["ready_to_enforce"] is False)


def test_report_tracks_legacy_stricter():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=True)
        bridge.gate("paper_trade_open", legacy_allowed=False,
                    legacy_reason="legacy says no")
        report = bridge.migration_report()
        check("legacy_stricter_counted",
              report["legacy_stricter_than_scoped"] == 1)
        check("no_breakage_from_stricter_legacy",
              report["would_be_denied_under_enforcement"] == 0)


def test_disagreement_serialises():
    with _Env(**{ENFORCE_ENV: "false"}):
        bridge = _bridge(with_grant=False)
        bridge.gate("paper_trade_open", legacy_allowed=True, legacy_reason="ok")
        d = bridge.disagreements[0].as_dict()
        check("disagreement_dict_keys",
              set(d) == {"capability", "domain", "legacy_allowed",
                         "scoped_allowed", "scoped_reason"})


# ═══════════════════════════════════════════════════════════════════
# 5. Integration with the live gate
# ═══════════════════════════════════════════════════════════════════

def test_live_gate_uses_bridge():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic"))
    from trust_integration import (
        gate_autonomous_action,
        get_legacy_bridge,
        set_legacy_bridge,
    )

    with _Env(**{ENFORCE_ENV: "false", "TRUST_LEDGER_PATH": _SCRATCH_LEDGER}):
        bridge = _bridge(with_grant=False)
        set_legacy_bridge(bridge)

        gate_autonomous_action("paper_trade_open", {"t": 1}, conviction=7.0)
        check("live_gate_consulted_bridge", bridge.consulted >= 1)

        set_legacy_bridge(None)


def test_live_gate_enforcing_cannot_widen():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic"))
    from trust_integration import gate_autonomous_action, set_legacy_bridge

    with _Env(**{ENFORCE_ENV: "true", "TRUST_LEDGER_PATH": _SCRATCH_LEDGER}):
        bridge = _bridge(with_grant=False)
        set_legacy_bridge(bridge)

        allowed, reason = gate_autonomous_action(
            "paper_trade_open", {"t": 1}, conviction=10.0
        )
        check("live_enforcing_denies_without_grant", not allowed, reason)

        set_legacy_bridge(None)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_legacy_cannot_grant_what_scoped_denies()
    test_legacy_deny_still_denies_when_scoped_allows()
    test_both_allow_permits()
    test_enforcing_never_widens()
    test_advisory_preserves_legacy_decision()
    test_advisory_records_disagreement()
    test_agreement_records_nothing()
    test_mode_flag_parsing()
    test_no_grant_means_denied()
    test_grant_permits()
    test_unmapped_capability_denied()
    test_explicit_domain_override()
    test_revoked_grant_denies()
    test_capability_map_covers_financial()
    test_migration_report_blocks_premature_enforcement()
    test_migration_report_signals_readiness()
    test_report_ready_requires_traffic()
    test_report_tracks_legacy_stricter()
    test_disagreement_serialises()
    test_live_gate_uses_bridge()
    test_live_gate_enforcing_cannot_widen()

    print(f"\n{'='*60}")
    print(f"Legacy Bridge Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
