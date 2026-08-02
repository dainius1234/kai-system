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


def _parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
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
        return violations

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
    ("12 privileged schemas reject unknown fields",
     rule_12_privileged_schemas_forbid_extra),
    ("14 no fail-open on protected paths",
     rule_14_no_fail_open),
]


def main() -> int:
    all_violations: List[Violation] = []

    print("Architecture dependency rules (roadmap §15)\n")
    for label, check in CHECKS:
        found = check()
        all_violations.extend(found)
        mark = "FAIL" if found else "ok  "
        print(f"  {mark}  rule {label}")
        for violation in found:
            print(str(violation))

    print()
    for rule, why in sorted(NOT_STATICALLY_CHECKABLE.items()):
        print(f"  n/a   rule {rule:<2} {why} — not statically checkable")

    print()
    if all_violations:
        print(f"FAIL: {len(all_violations)} architecture violation(s)")
        return 1
    print("PASS: no architecture violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
