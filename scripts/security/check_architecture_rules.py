"""Architecture dependency rules — roadmap §15 enforcement.

The roadmap closes §15 with:

    "A CI dependency rule should enforce forbidden imports/calls,
     supported by the side-effect registry and architecture tests."

Until now there was no such rule.  Fifteen architectural invariants — the
ones the entire Unified Hunter design rests on — were enforced by
convention, which is to say not enforced at all.

Module roles come from the UH-0 evidence manifest's six-role taxonomy.
Roles are declared here rather than inferred, because inferring a
module's role from its contents is exactly the ambiguity these rules
exist to remove.

Statically checkable rules are enforced.  Rules that need runtime
behaviour (idempotency design, reconciliation) are listed as
NOT_STATICALLY_CHECKABLE rather than silently omitted — an unenforced
rule that looks enforced is worse than an acknowledged gap.

Exit 0 clean, 1 on any violation.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
AGENTIC = REPO / "agentic"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

# ── Role taxonomy (UH-0 evidence manifest §6) ────────────────────────

PERCEPTION_PROVIDERS = {
    "alpha_signals", "market_intel", "market_data", "web_scout",
    "cortex", "forecaster",
}
TRANSFORMERS = {
    "global_workspace", "policy_memory", "causal_world_model",
    "cognitive_fingerprint", "wisdom_graph",
}
PROPOSAL_SPECIALISTS = {
    "strategy_engine", "opportunity_intel", "adversary", "hypothesis",
    "counterfactual", "planner", "dialectic", "analogy", "model_council",
}
POLICY_AUTHORITIES = {"trust_integration", "trust_core", "moral_core"}
ACTUATORS = {"paper_trader", "teammates", "swarm"}

# Modules that must never be imported by providers, specialists or
# transformers: importing an actuator is how a decider becomes a doer.
FORBIDDEN_ACTUATOR_IMPORTS = ACTUATORS

# Credential-bearing names D102 must not touch (§15 rule 3).
CREDENTIAL_MARKERS = {
    "BINANCE_API_KEY", "BINANCE_API_SECRET", "API_SECRET",
    "INTERSERVICE_HMAC_SECRET", "KAI_SERVICE_TOKEN", "BEARER_TOKEN",
}

# Rules that cannot be decided from source alone.  Listed so the gap is
# visible rather than mistaken for coverage.
NOT_STATICALLY_CHECKABLE = {
    9: "idempotency/reconciliation design for external effects",
    13: "missing mandatory dependency produces blocked/unavailable state",
    15: "feature flags disable capability but are not the authority boundary",
}


class Violation:
    __slots__ = ("rule", "module", "line", "message")

    def __init__(self, rule: int, module: str, line: int, message: str) -> None:
        self.rule = rule
        self.module = module
        self.line = line
        self.message = message

    def __str__(self) -> str:
        loc = f"{self.module}:{self.line}" if self.line else self.module
        return f"  RULE {self.rule:<2} {loc:<44} {self.message}"


# Files this gate could not read. Emptied at the start of every run.
#
# `_parse` used to answer `None` and every caller wrote `continue`, so a
# file that failed to parse was invisible to **all twelve** enforced
# rules — and the gate still printed `15/15 rules accounted for` and
# PASS. Proven by planting a syntax error in `common/policy_bridge`:
# exit 0, no mention of the file, clean bill of health.
#
# Note what that says about denominators. This gate *had* one, which is
# why it satisfied I-2. But it counted **rules**, not **files** — so it
# could not reveal a scanner that had gone blind to half its inputs. A
# denominator only falsifies a pass along the dimension it measures.
UNREADABLE: List[str] = []


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError) as exc:
        rel = str(path.relative_to(REPO)) if REPO in path.parents else str(path)
        UNREADABLE.append(f"{rel}: {type(exc).__name__}")
        return None


def _imported_names(tree) -> List[Tuple[str, int]]:
    """Every module name this file imports, with line numbers."""
    names: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append((node.module.split(".")[-1], node.lineno))
    return names


# ── Rule 1: a provider may not import an actuator ────────────────────

def rule_1_provider_imports_actuator() -> List[Violation]:
    violations = []
    for module in sorted(PERCEPTION_PROVIDERS | TRANSFORMERS):
        path = AGENTIC / f"{module}.py"
        tree = _parse(path)
        if tree is None:
            continue
        for name, line in _imported_names(tree):
            if name in FORBIDDEN_ACTUATOR_IMPORTS:
                violations.append(Violation(
                    1, f"agentic/{module}.py", line,
                    f"provider/transformer imports actuator '{name}'",
                ))
    return violations


# ── Rule 2: a proposal specialist may not call a side-effect endpoint ─

_SIDE_EFFECT_CALLS = {"post", "put", "delete", "patch"}


def _side_effect_markers() -> Set[str]:
    """Service names and paths the side-effect registry knows about.

    §15 says this rule is "supported by the side-effect registry", so the
    registry is the authority — not the HTTP verb.  POST is also how a
    read-only service receives a request body: a verifier being asked to
    check a claim causes no side effect, and flagging it would train
    people to ignore the gate.
    """
    markers: Set[str] = set()
    try:
        sys.path.insert(0, str(REPO))
        from common.actuator_registry.mutating_handlers import (
            MUTATING_ENDPOINTS,
        )
    except Exception:
        return markers

    for actuator, (env_key, default_url, actions) in MUTATING_ENDPOINTS.items():
        markers.add(actuator)
        markers.add(env_key)
        host = default_url.split("//", 1)[-1].split(":", 1)[0]
        if host:
            markers.add(host)
        for _method, path, effects in actions.values():
            if effects:                      # only paths that actually mutate
                markers.add(path.split("{")[0].rstrip("/"))
    markers.discard("")
    return markers


def rule_2_specialist_side_effects() -> List[Violation]:
    violations = []
    markers = _side_effect_markers()
    for module in sorted(PROPOSAL_SPECIALISTS):
        path = AGENTIC / f"{module}.py"
        tree = _parse(path)
        if tree is None:
            continue

        for name, line in _imported_names(tree):
            if name in FORBIDDEN_ACTUATOR_IMPORTS:
                violations.append(Violation(
                    2, f"agentic/{module}.py", line,
                    f"proposal specialist imports actuator '{name}'",
                ))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr not in _SIDE_EFFECT_CALLS:
                continue
            source = ast.unparse(func)
            if not any(c in source for c in
                       ("client", "httpx", "requests", "session")):
                continue

            # Registry-backed: flag only when the call targets something the
            # side-effect registry recognises as mutating.
            call_text = ast.unparse(node)
            hit = next((m for m in markers if m and m in call_text), None)
            if hit:
                violations.append(Violation(
                    2, f"agentic/{module}.py", node.lineno,
                    f"proposal specialist {func.attr.upper()}s to "
                    f"side-effect target '{hit}'",
                ))
    return violations


# ── Rule 3: D102 may not import or possess actuator credentials ──────

def rule_3_d102_credentials() -> List[Violation]:
    violations = []
    path = AGENTIC / "global_workspace.py"
    tree = _parse(path)
    if tree is None:
        return violations

    text = path.read_text(encoding="utf-8")
    for marker in sorted(CREDENTIAL_MARKERS):
        # Word-bounded so a shorter marker does not also match inside a
        # longer one (API_SECRET inside BINANCE_API_SECRET), which would
        # report the same occurrence twice.
        pattern = re.compile(rf"\b{re.escape(marker)}\b")
        match_line = next(
            (i for i, l in enumerate(text.splitlines(), 1) if pattern.search(l)),
            None,
        )
        if match_line is not None:
            violations.append(Violation(
                3, "agentic/global_workspace.py", match_line,
                f"D102 references credential '{marker}'",
            ))
    for name, line in _imported_names(tree):
        if name in FORBIDDEN_ACTUATOR_IMPORTS:
            violations.append(Violation(
                3, "agentic/global_workspace.py", line,
                f"D102 imports actuator '{name}'",
            ))
    return violations


# ── Rule 4: Ohana may block, but cannot issue security permission ────

def rule_4_ohana_cannot_permit() -> List[Violation]:
    violations = []
    from_assessment = REPO / "common" / "contracts" / "assessment.py"
    if from_assessment.exists():
        text = from_assessment.read_text(encoding="utf-8")
        # A bare ALLOW result would let the values layer manufacture
        # permission.  ALLOW_ADVISORY is deliberately named.
        if '= "allow"' in text:
            violations.append(Violation(
                4, "common/contracts/assessment.py", 0,
                "assessment layer exposes a bare 'allow' result",
            ))
    return violations


# ── Rule 12: privileged schemas reject unknown fields ────────────────

def rule_12_privileged_schemas_forbid_extra() -> List[Violation]:
    violations = []
    contracts = REPO / "common" / "contracts"
    if not contracts.exists():
        # `main()` calls `require(DECLARED_INPUTS)` first, so this cannot
        # be reached there. It can be reached when a rule is invoked
        # directly — by the test suite, or by a future caller — and a
        # rule that answers "no violations" because it found nothing to
        # inspect is the defect this gate exists to enforce against.
        return [Violation(12, "common/contracts", 0,
                          "directory not found — rule 12 inspected nothing")]

    for path in sorted(contracts.glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {ast.unparse(b) for b in node.bases}
            if not bases & {"BaseModel", "ContractBase"}:
                continue
            # ContractBase already forbids extras; direct BaseModel
            # subclasses must declare it themselves.
            if "ContractBase" in bases:
                continue
            body = ast.unparse(node)
            if 'extra' not in body or 'forbid' not in body:
                violations.append(Violation(
                    12, f"common/contracts/{path.name}", node.lineno,
                    f"privileged schema '{node.name}' does not forbid extra fields",
                ))
    return violations


# ── Rule 14: no fail-open on protected paths ─────────────────────────

PROTECTED_DIRS = [
    "common/policy_bridge", "common/actuator_registry", "common/autonomy",
    "common/perception_spine", "common/world_state",
    "common/proposal_workspace", "common/vertical_slice", "common/erasure",
]


def rule_14_no_fail_open() -> List[Violation]:
    violations = []
    for rel in PROTECTED_DIRS:
        base = REPO / rel
        if not base.exists():
            violations.append(Violation(14, rel, 0, "protected directory "
                                        "not found — inspected nothing"))
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                body = [
                    s for s in node.body
                    if not (isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant))
                ]
                if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    violations.append(Violation(
                        14, str(path.relative_to(REPO)), node.lineno,
                        "silent exception swallow on a protected path",
                    ))
    return violations


# ── Rule 5: trust/conviction may not bypass policy or approval ───────

def rule_5_trust_cannot_bypass() -> List[Violation]:
    """The legacy trust scalar must only be able to deny, never grant.

    Checked structurally: the bridge that unifies legacy trust with
    scoped grants must have no path that turns a scoped denial into an
    allow.  A grant-shaped return from the legacy side would recreate the
    "two authorities, most permissive wins" problem.
    """
    violations = []
    bridge = REPO / "common" / "autonomy" / "legacy_bridge.py"
    if not bridge.exists():
        return [Violation(5, "common/autonomy/legacy_bridge.py", 0,
                          "legacy trust bridge missing — trust is ungoverned")]

    text = bridge.read_text(encoding="utf-8")
    # In enforcing mode a scoped denial must short-circuit to False before
    # the legacy verdict is consulted.
    if "if not scoped_allowed:" not in text:
        violations.append(Violation(
            5, "common/autonomy/legacy_bridge.py", 0,
            "bridge does not short-circuit on a scoped denial",
        ))

    # Conviction must not appear as a standalone gate in policy paths.
    for rel in ("common/policy_bridge/policy_engine.py",
                "common/policy_bridge/approval.py"):
        path = REPO / rel
        if not path.exists():
            violations.append(Violation(5, rel, 0, "file not found — the "
                                        "conviction check inspected nothing "
                                        "here"))
            continue
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            source = ast.unparse(node)
            if ("conviction" in source or "trust_level" in source) and \
                    "result" not in source:
                violations.append(Violation(
                    5, rel, node.lineno,
                    f"policy path gates on a trust/conviction value: {source}",
                ))
    return violations


# ── Rule 6: every action route is registered and boundary-enforced ───

def rule_6_action_routes_registered() -> List[Violation]:
    """A side-effecting route must be authenticated at its own boundary.

    §15 rule 6 pairs registration with *final-boundary* enforcement: a
    central policy that a service does not itself enforce is advisory.
    """
    violations = []
    services = {
        "backup-service/app.py", "browser-agent/app.py",
        "telegram-bot/app.py", "monitor-service/app.py",
        "output/notify/app.py", "vault-sync/app.py", "executor/app.py",
    }
    # Routes that legitimately mutate nothing.
    READ_ONLY_ROUTES = {"/health", "/metrics", "/status", "/ready"}

    for rel in sorted(services):
        path = REPO / rel
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            for dec in getattr(node, "decorator_list", []):
                if not (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)):
                    continue
                if dec.func.attr not in {"post", "put", "delete", "patch"}:
                    continue
                if not dec.args or not isinstance(dec.args[0], ast.Constant):
                    continue
                route = dec.args[0].value
                if route in READ_ONLY_ROUTES:
                    continue
                if "require_service_auth" not in ast.unparse(dec):
                    violations.append(Violation(
                        6, rel, dec.lineno,
                        f"{dec.func.attr.upper()} {route} is not "
                        f"boundary-enforced",
                    ))
    return violations


# ── Rule 7: legacy action APIs are disabled ──────────────────────────

def rule_7_legacy_paths_closed() -> List[Violation]:
    violations = []
    try:
        sys.path.insert(0, str(REPO))
        from common.actuator_registry.legacy_verification import (
            open_legacy_paths,
        )
    except Exception as exc:
        return [Violation(7, "common/actuator_registry", 0,
                          f"legacy verification unavailable: {exc}")]

    for actuator, reason in sorted(open_legacy_paths().items()):
        violations.append(Violation(
            7, f"actuator:{actuator}", 0, f"legacy path still open — {reason}",
        ))
    return violations


# ── Rule 8: state-changing methods return typed state ────────────────

_SUCCESS_SHAPED_KEYS = {"success", "ok", "status"}


def rule_8_typed_operation_state() -> List[Violation]:
    """A success-shaped dict hides failure; a typed state cannot.

    Only flags *literal* returns of a bare success dict on protected
    paths.  A dict assembled from real values is a payload, not a
    success shape.
    """
    violations = []
    for rel in PROTECTED_DIRS:
        base = REPO / rel
        if not base.exists():
            violations.append(Violation(14, rel, 0, "protected directory "
                                        "not found — inspected nothing"))
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                keys = {
                    k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                if not keys or not keys <= _SUCCESS_SHAPED_KEYS:
                    continue
                values_are_literals = all(
                    isinstance(v, ast.Constant) for v in node.value.values
                )
                if values_are_literals:
                    violations.append(Violation(
                        8, str(path.relative_to(REPO)), node.lineno,
                        f"returns a success-shaped dict {sorted(keys)} "
                        f"instead of a typed operation state",
                    ))
    return violations


# ── Rule 10: persistent records carry principal/purpose/provenance ───

_REQUIRED_RECORD_FIELDS = {
    "principal", "purpose", "classification", "provenance", "revision",
}


def rule_10_records_carry_context() -> List[Violation]:
    violations = []
    base_path = REPO / "common" / "contracts" / "base.py"
    tree = _parse(base_path)
    if tree is None:
        return [Violation(10, "common/contracts/base.py", 0,
                          "contract base unreadable")]

    contract_base = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "ContractBase"), None,
    )
    if contract_base is None:
        return [Violation(10, "common/contracts/base.py", 0,
                          "ContractBase not found")]

    declared = {
        t.target.id for t in contract_base.body
        if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
    }
    for field in sorted(_REQUIRED_RECORD_FIELDS - declared):
        violations.append(Violation(
            10, "common/contracts/base.py", contract_base.lineno,
            f"ContractBase does not carry '{field}'",
        ))

    # Persistent contracts must inherit it rather than bare BaseModel.
    contracts_dir = REPO / "common" / "contracts"
    for path in sorted(contracts_dir.glob("*.py")):
        if path.name in {"__init__.py", "base.py"}:
            continue
        sub = _parse(path)
        if sub is None:
            continue
        for node in ast.walk(sub):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {ast.unparse(b) for b in node.bases}
            if bases == {"BaseModel"}:
                violations.append(Violation(
                    10, f"common/contracts/{path.name}", node.lineno,
                    f"'{node.name}' subclasses BaseModel directly, so it "
                    f"carries no principal/purpose/provenance",
                ))
    return violations


# ── Rule 11: model-generated fields are labelled ─────────────────────

def rule_11_model_output_labelled() -> List[Violation]:
    """Model output must be distinguishable from observation.

    Enforced through the evidence grading system: MODEL_GENERATED and
    SIMULATED must exist and must not qualify to grant trust.  Without
    that distinction, a model's own text can be laundered into evidence.
    """
    violations = []
    path = REPO / "common" / "contracts" / "autonomy.py"
    if not path.exists():
        return [Violation(11, "common/contracts/autonomy.py", 0,
                          "evidence grading missing — model output "
                          "indistinguishable from observation")]

    text = path.read_text(encoding="utf-8")
    for required in ("MODEL_GENERATED", "SIMULATED", "def qualifies"):
        if required not in text:
            violations.append(Violation(
                11, "common/contracts/autonomy.py", 0,
                f"evidence grading lacks '{required}'",
            ))

    try:
        sys.path.insert(0, str(REPO))
        from common.contracts.autonomy import EvidenceGrade
        for grade in (EvidenceGrade.MODEL_GENERATED, EvidenceGrade.SIMULATED):
            if grade.qualifies():
                violations.append(Violation(
                    11, "common/contracts/autonomy.py", 0,
                    f"{grade.name} qualifies to grant trust",
                ))
    except Exception as exc:
        violations.append(Violation(
            11, "common/contracts/autonomy.py", 0,
            f"evidence grading not importable: {exc}",
        ))
    return violations


# ── Runner ───────────────────────────────────────────────────────────

CHECKS = [
    ("1  provider/transformer may not import an actuator",
     rule_1_provider_imports_actuator),
    ("2  proposal specialist may not cause side effects",
     rule_2_specialist_side_effects),
    ("3  D102 may not hold actuator credentials",
     rule_3_d102_credentials),
    ("4  Ohana cannot issue security permission",
     rule_4_ohana_cannot_permit),
    ("5  trust/conviction cannot bypass policy or approval",
     rule_5_trust_cannot_bypass),
    ("6  every action route is boundary-enforced",
     rule_6_action_routes_registered),
    ("7  legacy action APIs are disabled",
     rule_7_legacy_paths_closed),
    ("8  state-changing methods return typed state",
     rule_8_typed_operation_state),
    ("10 persistent records carry principal/purpose/provenance",
     rule_10_records_carry_context),
    ("11 model-generated output is labelled and cannot grant trust",
     rule_11_model_output_labelled),
    ("12 privileged schemas reject unknown fields",
     rule_12_privileged_schemas_forbid_extra),
    ("14 no fail-open on protected paths",
     rule_14_no_fail_open),
]

ALL_RULES = set(range(1, 16))


def accounted_rules() -> Set[int]:
    """Every §15 rule this gate either enforces or declares uncheckable."""
    enforced = {int(re.match(r"(\d+)", label).group(1)) for label, _ in CHECKS}
    return enforced | set(NOT_STATICALLY_CHECKABLE)


# Everything these rules read. A missing protected directory means the
# rules over it inspected nothing, which is not a clean bill of health.
DECLARED_INPUTS = tuple(PROTECTED_DIRS) + (
    "common/contracts",
    "common/policy_bridge/policy_engine.py",
    "common/policy_bridge/approval.py",
)


def _python_files() -> int:
    total = 0
    for rel in DECLARED_INPUTS:
        base = REPO / rel
        if base.is_file():
            total += 1
        elif base.is_dir():
            total += sum(1 for p in base.rglob("*.py")
                         if "__pycache__" not in str(p))
    return total


def main() -> int:
    UNREADABLE.clear()
    require(DECLARED_INPUTS)
    all_violations: List[Violation] = []

    print("Architecture dependency rules (roadmap §15)\n")
    for label, check in CHECKS:
        found = check()
        all_violations.extend(found)
        mark = "FAIL" if found else "ok  "
        print(f"  {mark}  rule {label}")
        for violation in found:
            print(str(violation))

    # A second denominator, along the dimension the first one misses.
    # `15/15 rules` says every rule ran; it says nothing about whether
    # every file was readable by them.
    print()
    print(inspected(_python_files(), "python files",
                    f"across {len(DECLARED_INPUTS)} declared inputs"))
    unreadable = sorted(set(UNREADABLE))   # one entry per file, not per rule
    if unreadable:
        print(f"\n  UNREADABLE ({len(unreadable)}) — these files were "
              f"invisible to every rule above:")
        for entry in unreadable:
            print(f"    - {entry}")
        print("  A file the parser cannot read has not been checked. "
              "Declining to\n  report that would make every verdict above "
              "narrower than it looks.")

    print()
    for rule, why in sorted(NOT_STATICALLY_CHECKABLE.items()):
        print(f"  n/a   rule {rule:<2} {why} — not statically checkable")

    # A rule that is neither enforced nor declared is invisible, which is
    # the failure this gate exists to prevent.  It must not happen to the
    # gate itself.
    unaccounted = sorted(ALL_RULES - accounted_rules())
    print()
    if unaccounted:
        print(f"  GAP   §15 rules neither enforced nor declared: {unaccounted}")
        all_violations.append(Violation(
            0, "check_architecture_rules.py", 0,
            f"rules {unaccounted} are silently unaccounted for",
        ))
    else:
        print(f"  cover §15 rules accounted for: "
              f"{len(accounted_rules())}/15 "
              f"({len(CHECKS)} enforced, "
              f"{len(NOT_STATICALLY_CHECKABLE)} declared uncheckable)")

    print()
    if unreadable:
        print(f"\nFAIL: {len(unreadable)} file(s) could not be parsed and "
              f"were checked by no rule.")
        return 1

    if all_violations:
        print(f"FAIL: {len(all_violations)} architecture violation(s)")
        return 1
    print("PASS: no architecture violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
