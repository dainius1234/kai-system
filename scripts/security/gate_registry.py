"""What each check in `scripts/security/` is, and what it claims.

Adding a check today means remembering three places — the `policy-check`
target, a CI workflow, and a suite proving it can fail — with nothing
cross-checking them. Two of twelve were already inconsistent when this
file was written. That is the `PUBLIC_ROUTES` problem: `dashboard/app.py`
solved it by making every route declare its own scope, so a new
unauthenticated route is *visibly* unsafe rather than invisibly so.

This is the same declaration for the watching layer. A check that is not
declared here fails `check_gate_registry.py`, so a new one is visibly
unregistered rather than quietly unwatched.

**There is deliberately no free-text `exception` field.** The operator's
rubric governs this file:

> If it can't be encoded so the system enforces it, it's not an
> exception — it's debt.

So every departure from the norm is a typed field the meta-check reads
and reports on, never a note explaining why a rule does not apply:

  - `kind=REPORT` says *this is not a gate* — the reason it is absent
    from `policy-check` is recorded, not inferred from its absence.
  - `optional_inputs` says *this input may legitimately be missing* —
    per path, rather than a blanket `continue` that makes every input
    optional.
  - `probe=False` requires `probe_skip_reason`, so a check that is too
    expensive to run says so out loud instead of being silently skipped.
  - `pending_wiring` names the step that will wire a gate up. A gate that
    is not yet enforced is reported on every run until it is.

Each of those is visible in the meta-check's output every time it runs.
None of them can be satisfied by someone knowing why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# A gate fails the build when it finds a violation. A report informs and
# must *not* be wired as a gate — a report that gates makes the build
# depend on information nobody promised was actionable.
GATE = "gate"
REPORT = "report"

COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)


@dataclass(frozen=True)
class Gate:
    module: str
    kind: str
    summary: str

    # I-1 — what this check reads. A required input that is missing means
    # the check cannot answer its question, which is a failure, not a
    # pass. Optional inputs are declared per path so that "this one may
    # be absent" never generalises into "absence is fine".
    inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()

    # I-2 — a regex its output must match, naming how much it inspected.
    denominator: Optional[str] = None
    probe: bool = True
    probe_skip_reason: Optional[str] = None

    # I-3 — the suite that injects a violation and asserts it fires.
    proven_by: Optional[str] = None

    # I-4 — where it is enforced. Discovered independently and compared.
    in_policy_check: bool = False
    in_workflows: Tuple[str, ...] = ()
    pending_wiring: Optional[str] = None

    # The register entry this check's own defects are tracked under.
    findings: Tuple[str, ...] = field(default_factory=tuple)


_COMPOSE_GATE = dict(
    kind=GATE,
    inputs=COMPOSE_FILES,
    in_policy_check=True,
    in_workflows=("policy-checks.yml",),
    findings=("KAI-GATE-001", "KAI-GATE-002", "KAI-GATE-003"),
)

REGISTRY: Tuple[Gate, ...] = (
    # ── The eight pre-incident compose gates ─────────────────────────
    # None reports a denominator; none has ever been observed failing.
    # All eight skip absent inputs. They are the retrofit backlog.
    Gate(module="check_port_bindings",
         summary="only the dashboard may publish a port, on 127.0.0.1",
         **_COMPOSE_GATE),
    Gate(module="check_default_profiles",
         summary="no dangerous service in the default profile",
         **_COMPOSE_GATE),
    Gate(module="check_secret_fallbacks",
         summary="no insecure fallback for a missing secret",
         **_COMPOSE_GATE),
    Gate(module="check_network_zones",
         summary="network zone segmentation holds",
         **_COMPOSE_GATE),
    Gate(module="check_turbovec_writers",
         summary="TurboVec has a single writer",
         **_COMPOSE_GATE),
    Gate(module="check_restart_recovery",
         summary="restart and recovery stay contained",
         **_COMPOSE_GATE),
    Gate(module="check_image_tags",
         summary="pre-built images use pinned tags",
         **_COMPOSE_GATE),
    Gate(module="check_compose_drift",
         summary="compose files stay structurally consistent",
         **_COMPOSE_GATE),

    # ── The four built or repaired after an incident ─────────────────
    Gate(module="check_architecture_rules",
         kind=GATE,
         summary="the 15 §15 architecture rules (A-01)",
         inputs=("common/contracts", "common/policy_bridge"),
         denominator=r"rules accounted for:\s*\d+/\d+",
         proven_by="scripts/test_architecture_rules.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-001",)),

    Gate(module="hygiene_survey",
         kind=GATE,
         summary="repo-wide HTTP/time hygiene ratchet (H-5)",
         inputs=("scripts/security/hygiene_baseline.json",),
         denominator=r"\d+ of \d+ services carry none of these",
         proven_by="scripts/test_hygiene_gate.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-001",)),

    Gate(module="check_dashboard_findings",
         kind=REPORT,
         summary="revalidates all 96 KAI-DASH findings — a status report",
         inputs=("dashboard/app.py", "common/dashboard_auth.py"),
         denominator=r"Coverage: all \d+ findings accounted for",
         proven_by="scripts/test_dashboard_findings.py",
         in_policy_check=False,
         in_workflows=(),
         findings=("KAI-GATE-001", "KAI-GATE-005")),

    Gate(module="check_assertion_floors",
         kind=GATE,
         summary="assertion-count ratchet (A-02)",
         inputs=("scripts/security/assertion_floors.json",),
         denominator=r"\d+ suites, \d+ assertions",
         probe=False,
         probe_skip_reason="runs `make test-uh` — minutes, not seconds; "
                           "probed by scripts/test_assertion_floors.py "
                           "against synthetic logs instead",
         proven_by="scripts/test_assertion_floors.py",
         in_policy_check=False,
         in_workflows=("unified-hunter.yml",)),

    # ── The meta-check, bound by the rules it enforces ───────────────
    # It appears in its own registry on purpose. The recursion is
    # depth-one and closed: it declares a denominator, fails closed on an
    # unreadable registry, and is proven by a synthetic-registry suite.
    # `pending_wiring` is the encoded form of "not enforced yet" — it is
    # reported on every run until A-04e flips it.
    Gate(module="check_gate_registry",
         kind=GATE,
         summary="the four instrumentation invariants (A-04)",
         inputs=("scripts/security/gate_registry.py", "Makefile"),
         denominator=r"\d+ checks cross-checked",
         proven_by="scripts/test_gate_registry.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="A-04e — flips to enforcing once the register is clear"),
)

BY_MODULE = {gate.module: gate for gate in REGISTRY}
