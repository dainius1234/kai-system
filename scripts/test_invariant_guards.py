"""Invariant guard tests — detect regression to fail-open or legacy authority.

Closes roadmap §16.26 ("rollback attempts to restore fail-open/legacy
authority") and enforces §15.14 ("no `except Exception: pass` or
fail-open behaviour in policy, approval, execution, verification or
persistence paths").

These are *source-level* guards, not behavioural tests.  Behavioural
tests prove the code does the right thing today; these prove that a
revert, a merge, or a well-meaning "temporary" patch cannot quietly put
the old permissive behaviour back without a test going red.

Each guard names the invariant it protects so a failure explains itself.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = Path(__file__).resolve().parent.parent

# Modules on the protected path.  Roadmap §15.14 forbids fail-open
# behaviour in policy, approval, execution, verification and persistence.
PROTECTED_PATHS = [
    "common/policy_bridge",
    "common/actuator_registry",
    "common/autonomy",
    "common/perception_spine",
    "common/world_state",
    "common/proposal_workspace",
    "common/vertical_slice",
]

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


def _protected_files() -> list[Path]:
    files: list[Path] = []
    for rel in PROTECTED_PATHS:
        base = REPO / rel
        if base.exists():
            files.extend(sorted(base.rglob("*.py")))
    return [f for f in files if "__pycache__" not in str(f)]


# ── 1. No silent exception swallowing on protected paths ────────────

def test_no_silent_except():
    """`except: pass` on a protected path is a fail-open in disguise."""
    offenders: list[str] = []

    for path in _protected_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{path.relative_to(REPO)}: unparseable ({exc})")
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = [s for s in node.body if not isinstance(s, ast.Expr)
                    or not isinstance(s.value, ast.Constant)]
            if not body:
                offenders.append(
                    f"{path.relative_to(REPO)}:{node.lineno} empty except body"
                )
                continue
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                offenders.append(
                    f"{path.relative_to(REPO)}:{node.lineno} `except: pass`"
                )

    check("no_silent_except_on_protected_paths", not offenders,
          "; ".join(offenders[:5]))


# ── 2. Fail-closed defaults survive ─────────────────────────────────

def test_policy_engine_fails_closed():
    """The unavailable-engine branch must still deny."""
    from common.contracts.base import Principal, Provenance, RiskTier
    from common.contracts.action import ActionProposal
    from common.policy_bridge.policy_engine import PolicyEngine

    principal = Principal(identity="kai", role="system")
    proposal = ActionProposal(
        action_type="t", description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=principal,
        purpose="test", provenance=Provenance(source="test"),
    )

    engine = PolicyEngine(principal=principal)
    engine.set_available(False)
    result = engine.evaluate(proposal)

    check("policy_unavailable_denies", result.decision.result == "deny")
    check("policy_unavailable_not_allow", result.decision.result != "allow")


def test_capability_bridge_has_no_bypass():
    """CapabilityBridge must expose no method that skips consumption."""
    from common.policy_bridge.capability import CapabilityBridge

    forbidden = {"force_consume", "bypass", "consume_unchecked",
                 "issue_without_approval", "skip_validation"}
    present = forbidden & set(dir(CapabilityBridge))
    check("no_capability_bypass_methods", not present, str(present))


def test_actuator_registry_has_no_bypass():
    """ActuatorRegistry must expose no method that skips capability checks."""
    from common.actuator_registry.registry import ActuatorRegistry

    forbidden = {"dispatch_unchecked", "force_dispatch", "bypass_capability",
                 "dispatch_legacy", "direct_execute"}
    present = forbidden & set(dir(ActuatorRegistry))
    check("no_actuator_bypass_methods", not present, str(present))


# ── 3. Legacy authority cannot be restored silently ─────────────────

def test_legacy_verification_gate_intact():
    """advance_migration to VERIFIED must still check legacy_disabled."""
    from common.contracts.base import Principal, RiskTier
    from common.actuator_registry.registry import (
        ActuatorDispatchError,
        ActuatorRegistration,
        ActuatorRegistry,
        MigrationStatus,
        MigrationTier,
    )

    principal = Principal(identity="kai", role="system")
    registry = ActuatorRegistry(principal=principal)
    registry.register(ActuatorRegistration(
        identity="guard-test", display_name="Guard", description="d",
        risk_tier=RiskTier.OBSERVE, migration_tier=MigrationTier.READ_ONLY,
        action_types=["a"], legacy_path="legacy:/old",
    ))
    registry.advance_migration("guard-test", MigrationStatus.MIGRATING, principal)

    raised = False
    try:
        registry.advance_migration("guard-test", MigrationStatus.VERIFIED, principal)
    except ActuatorDispatchError:
        raised = True

    check("legacy_gate_still_blocks_verification", raised,
          "VERIFIED was reachable with legacy path enabled")


def test_evidence_grading_gate_intact():
    """Self-generated and simulated evidence must still fail to qualify."""
    from common.contracts.autonomy import EvidenceGrade

    check("model_generated_still_disqualified",
          not EvidenceGrade.MODEL_GENERATED.qualifies())
    check("simulated_still_disqualified",
          not EvidenceGrade.SIMULATED.qualifies())
    check("unknown_still_disqualified",
          not EvidenceGrade.UNKNOWN.qualifies())


def test_assessor_cannot_express_permission():
    """The assessment layer must have no way to say 'allowed'.

    An assessor that could return a security allow would let values or
    loyalty manufacture permission (roadmap §7.4).
    """
    from common.contracts.assessment import AssessmentResult

    values = {r.value for r in AssessmentResult}
    check("no_bare_allow_result", "allow" not in values, str(values))
    check("advisory_allow_named_clearly",
          "allow_advisory" in values)

    from common.policy_bridge.assessment import AggregateAssessment
    check("aggregate_has_no_allow_field",
          "allowed" not in AggregateAssessment.__slots__)


def test_trust_scalar_not_reintroduced_into_new_path():
    """The new autonomy path must not depend on the legacy TrustLevel.

    Matches real coupling — imports of ``trust_core`` and uses of
    ``TrustLevel`` as a value — rather than any textual mention, so
    documentation that *names* the legacy system while explaining the
    migration away from it does not trip the guard.
    """
    offenders: list[str] = []
    coupling = re.compile(
        r"^\s*(from\s+\S*trust_core\s+import|import\s+\S*trust_core)"
        r"|TrustLevel\s*[.(\[]",
        re.MULTILINE,
    )

    for path in _protected_files():
        text = path.read_text(encoding="utf-8")
        if coupling.search(text):
            offenders.append(str(path.relative_to(REPO)))

    check("legacy_trust_level_not_in_new_path", not offenders,
          "; ".join(offenders))


def test_bridge_cannot_widen_authority():
    """The legacy bridge must be able to deny, never to grant.

    A bridge that could turn a scoped denial into an allow would restore
    the "two authorities, most permissive wins" problem it exists to
    remove.
    """
    import os as _os
    from common.autonomy.legacy_bridge import ENFORCE_ENV, LegacyTrustBridge
    from common.autonomy.authority import AutonomyAuthority
    from common.autonomy.calibration import CalibrationTracker
    from common.autonomy.evidence_service import EvidenceService
    from common.contracts.base import Principal

    principal = Principal(identity="kai", role="system")
    bridge = LegacyTrustBridge(
        AutonomyAuthority(
            principal,
            EvidenceService(principal=principal),
            CalibrationTracker(principal=principal),
        ),
        principal,
    )

    saved = _os.environ.get(ENFORCE_ENV)
    _os.environ[ENFORCE_ENV] = "true"
    try:
        allowed, _ = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trusted",
        )
        check("bridge_cannot_widen", not allowed,
              "legacy allow overrode a scoped denial")
    finally:
        if saved is None:
            _os.environ.pop(ENFORCE_ENV, None)
        else:
            _os.environ[ENFORCE_ENV] = saved


# ── 4. Ingress bounds cannot be silently removed ────────────────────

def test_payload_bounds_present():
    from common.perception_spine.ingress import (
        MAX_PAYLOAD_BYTES,
        MAX_PAYLOAD_DEPTH,
        MAX_PAYLOAD_KEYS,
        check_payload_bounds,
    )

    check("payload_byte_cap_positive", MAX_PAYLOAD_BYTES > 0)
    check("payload_depth_cap_positive", MAX_PAYLOAD_DEPTH > 0)
    check("payload_key_cap_positive", MAX_PAYLOAD_KEYS > 0)

    deep = {"a": {}}
    cursor = deep["a"]
    for _ in range(MAX_PAYLOAD_DEPTH + 5):
        cursor["a"] = {}
        cursor = cursor["a"]
    check("deep_payload_still_rejected",
          check_payload_bounds(deep) is not None)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_no_silent_except()
    test_policy_engine_fails_closed()
    test_capability_bridge_has_no_bypass()
    test_actuator_registry_has_no_bypass()
    test_legacy_verification_gate_intact()
    test_evidence_grading_gate_intact()
    test_assessor_cannot_express_permission()
    test_trust_scalar_not_reintroduced_into_new_path()
    test_bridge_cannot_widen_authority()
    test_payload_bounds_present()

    print(f"\n{'='*60}")
    print(f"Invariant Guard Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
