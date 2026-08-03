"""The meta-check: does the watching layer obey its own rules?

Nine defects in this programme were in the instrumentation, not the
system, and all nine were found by luck. The application layer has scoped
routes, a degraded envelope, a finding register and an audit trail. The
layer watching it had none of that — while being the only thing standing
between 1,947 assertions and silent atrophy.

Four invariants, each a property `dashboard/app.py` already has:

  I-1  **Fail closed on a missing input.** No token → 503, never open.
       The checks did the opposite: no input file → `continue` → PASS.
  I-2  **Report a denominator, or decline.** A `PASS` with no statement
       of how much was inspected reads identically whether it examined
       fifty services or zero.
  I-3  **Prove it can fail.** Eight checks have never been observed
       failing. They may be vacuous now; nothing would say so.
  I-4  **Declare in one place, not three.** A check nobody registered is
       a check nobody watches.

I-1 has a name, given by the operator: **boundary blindness** — a check
that cannot distinguish *the system is correct* from *the system is
absent*. It is the self-consuming guard moved from the state boundary to
the input boundary. A self-consuming guard erodes because the thing it
guards succeeded; a boundary-blind check erodes because its input moved.
Both end at "passes while checking nothing", and the reason is the same
in both: the script answers *"of the things I looked at, were any
wrong?"* while claiming to answer *"are the things correct?"*.

**Depth-one recursion, deliberately.** This file is in its own registry
and bound by its own rules — it reports a denominator, fails closed on an
unreadable registry, and is proven by a synthetic-registry suite. The
criterion for stopping is the operator's: *does the watcher survive the
same scrutiny it applies?* It does, so there is no seventh gate. Adding
one to watch this would be the mistake; making this obey the rule it
enforces is not.

Modes:
  (default)  report — always exits 0. Makes the debt visible without
             blocking, the same way H-5 landed.
  --gate     enforce — exits 1 on any violation. A-04e flips to this.

Exit codes:
  0  reporting mode, or enforcing mode with nothing to report
  1  enforcing mode with violations, or the registry could not be read
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
SECURITY = REPO / "scripts" / "security"

# Mirrored from the registry so `cross_check` stays a pure function that
# the suite can call without importing the real registry.
GATE_KIND = "gate"
REPORT_KIND = "report"


def _load_registry():
    """Fail closed: an unreadable registry is a failure, not a skip.

    This is I-1 applied to this file. Returning an empty registry here
    would make the meta-check pass by inspecting nothing — precisely the
    defect it exists to find.
    """
    sys.path.insert(0, str(REPO))
    from scripts.security.gate_registry import GATE, REPORT, REGISTRY
    if not REGISTRY:
        raise SystemExit("REFUSED: the registry is empty. A meta-check "
                         "with nothing to check is not a passing "
                         "meta-check.")
    return GATE, REPORT, REGISTRY


# ── Source 1: the filesystem ─────────────────────────────────────────

def discover_modules() -> List[str]:
    """Every check script that exists, whatever anyone declared."""
    return sorted(
        p.stem for p in SECURITY.glob("*.py")
        if p.stem not in {"__init__", "gate_registry", "gate_inputs"}
    )


# ── Source 2: what actually runs ─────────────────────────────────────

def discover_policy_check() -> List[str]:
    """Modules named inside the Makefile's `policy-check` target."""
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    block: List[str] = []
    inside = False
    for line in makefile.splitlines():
        if line.startswith("policy-check:"):
            inside = True
            continue
        if inside:
            if line and not line.startswith(("\t", " ")):
                break
            block.append(line)
    return sorted(set(re.findall(r"scripts/security/([a-z_]+)\.py",
                                 "\n".join(block))))


def discover_workflows() -> Dict[str, List[str]]:
    """Map module -> the workflow files that invoke it."""
    found: Dict[str, List[str]] = {}
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        for module in re.findall(r"scripts/security/([a-z_]+)\.py", text):
            found.setdefault(module, []).append(path.name)
    return {m: sorted(set(v)) for m, v in found.items()}


# ── I-1: boundary blindness, detected in the source ──────────────────

def _is_absence_test(node: ast.expr) -> bool:
    """`not X.exists()`, or a chain of them joined by and/or."""
    if isinstance(node, ast.BoolOp):
        return any(_is_absence_test(v) for v in node.values)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and isinstance(node.operand, ast.Call)
        and isinstance(node.operand.func, ast.Attribute)
        and node.operand.func.attr == "exists"
    )


def skips_absent_input(module: str) -> List[int]:
    """Lines where a missing input is skipped rather than refused.

    Deliberately conservative: it matches only the explicit shape
    ``if not X.exists(): continue|return|pass``. Positive guards
    (``if X.exists(): <do the work>``) have the same effect but also have
    innocent uses, and a survey with false positives invites someone to
    "fix" working code — which is exactly how defect 7 happened.

    So this is a **lower bound**, and the report says so rather than
    presenting it as complete.
    """
    path = SECURITY / f"{module}.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_absence_test(node.test):
            continue
        if all(isinstance(stmt, (ast.Continue, ast.Pass, ast.Return))
               for stmt in node.body):
            lines.append(node.lineno)
    return sorted(lines)


# ── I-2: does it say how much it inspected? ──────────────────────────

def probe_denominator(gate) -> Tuple[str, str]:
    """Run the check and see whether its output names a denominator.

    Returns (status, detail) where status is one of:
      ok        the output matched the declared pattern
      missing   no denominator declared
      absent    declared, but the output did not contain it
      skipped   declared too expensive to run, with a stated reason
      self      this module — see below

    **This module must never probe itself by subprocess.** It did, on the
    first run, and recursed until the process tree had to be killed. The
    depth-one recursion is a property of the *design*; nothing in the code
    enforced it, and a design property that the code does not enforce is
    exactly the class of defect this file exists to find.

    So the terminus is explicit: the meta-check's own denominator is
    verified by `scripts/test_gate_registry.py`, which drives `main()`
    in-process and asserts the header line it prints matches the pattern
    declared in the registry. That is not circular — the assertion comes
    from outside, against real output.
    """
    if gate.module == Path(__file__).stem:
        return "self", ("verified in-process by "
                        "scripts/test_gate_registry.py, never by subprocess")
    if gate.denominator is None:
        return "missing", "no denominator declared"
    if not gate.probe:
        return "skipped", gate.probe_skip_reason or "no reason given"
    result = subprocess.run(
        [sys.executable, str(SECURITY / f"{gate.module}.py")],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    output = result.stdout + result.stderr
    match = re.search(gate.denominator, output)
    if match:
        return "ok", match.group(0)
    return "absent", f"output did not match /{gate.denominator}/"


# ── The cross-check ──────────────────────────────────────────────────

def cross_check(registry, on_disk, in_policy, in_flows,
                blind=None, probe=None, repo_has=None,
                kinds=(GATE_KIND, REPORT_KIND)) -> Dict[str, List[str]]:
    """The rules, as a pure function of the four sources.

    Split out from `evaluate()` so the suite can drive it from synthetic
    registries and synthetic filesystems. A meta-check that could only be
    tested against the real repository would be guarded on state its own
    tests modify — the self-consuming shape, in the file written to
    detect it.
    """
    GATE, REPORT = kinds
    blind = blind or (lambda module: [])
    probe = probe or (lambda gate: ("ok", ""))
    repo_has = repo_has or (lambda rel: True)

    by_module = {g.module: g for g in registry}
    declared = set(by_module)
    on_disk = set(on_disk)
    in_policy = set(in_policy)

    problems: Dict[str, List[str]] = {
        "unregistered": [], "phantom": [], "wiring": [],
        "unproven": [], "blind": [], "denominator": [], "pending": [],
    }

    # I-4 — three sources must agree.
    for module in sorted(on_disk - declared):
        problems["unregistered"].append(
            f"{module}: exists but is not in the registry")
    for module in sorted(declared - on_disk):
        problems["phantom"].append(
            f"{module}: registered but no such file")

    for module in sorted(declared & on_disk):
        gate = by_module[module]
        actual_policy = module in in_policy
        actual_flows = in_flows.get(module, [])

        if actual_policy != gate.in_policy_check:
            problems["wiring"].append(
                f"{module}: declares in_policy_check="
                f"{gate.in_policy_check}, actually {actual_policy}")
        if sorted(gate.in_workflows) != sorted(actual_flows):
            problems["wiring"].append(
                f"{module}: declares workflows {list(gate.in_workflows)}, "
                f"actually {actual_flows}")

        if gate.kind == REPORT and actual_policy:
            problems["wiring"].append(
                f"{module}: declared a REPORT but wired into policy-check")
        if gate.kind == GATE and not (actual_policy or actual_flows):
            if gate.pending_wiring:
                problems["pending"].append(
                    f"{module}: not enforced yet — {gate.pending_wiring}")
            else:
                problems["wiring"].append(
                    f"{module}: a GATE that nothing invokes")

        # I-3 — proven able to fail.
        if gate.proven_by is None:
            problems["unproven"].append(
                f"{module}: no suite proves it can fail")
        elif not repo_has(gate.proven_by):
            problems["unproven"].append(
                f"{module}: proven_by {gate.proven_by} does not exist")

        # I-1 — boundary blindness.
        for line in blind(module):
            problems["blind"].append(
                f"{module}.py:{line}: skips an absent input instead of "
                f"refusing to certify it")

        # I-1 — and the inputs it declares must actually be there.
        for rel in gate.inputs:
            if not repo_has(rel):
                problems["blind"].append(
                    f"{module}: required input {rel} is missing — this "
                    f"check cannot answer its question right now")

        # I-2 — a denominator, or an explicit decline.
        status, detail = probe(gate)
        if status in ("missing", "absent"):
            problems["denominator"].append(f"{module}: {detail}")

    return problems


def evaluate() -> Dict[str, List[str]]:
    """`cross_check` wired to the real repository."""
    GATE, REPORT, REGISTRY = _load_registry()
    return cross_check(
        REGISTRY,
        on_disk=discover_modules(),
        in_policy=discover_policy_check(),
        in_flows=discover_workflows(),
        blind=skips_absent_input,
        probe=probe_denominator,
        repo_has=lambda rel: (REPO / rel).exists(),
        kinds=(GATE, REPORT),
    )


_HEADINGS = [
    ("unregistered", "UNREGISTERED — on disk, undeclared (I-4)"),
    ("phantom", "PHANTOM — declared, not on disk (I-4)"),
    ("wiring", "WIRING DISAGREES with the declaration (I-4)"),
    ("blind", "BOUNDARY BLINDNESS — absence reads as correctness (I-1)"),
    ("denominator", "NO DENOMINATOR — a pass that cannot be falsified (I-2)"),
    ("unproven", "NEVER OBSERVED FAILING (I-3)"),
]

# ── Per-invariant enforcement ────────────────────────────────────────
#
# A big-bang flip would make CI red for as long as the retrofit takes,
# and a permanently red gate is an ignored gate — which is defect 9
# again, wearing a fix's clothes. Waiting until all four are clean means
# no enforcement at all in the meantime.
#
# So the ratchet has teeth that harden one at a time. An invariant is
# enforced once it reaches zero, and from then on it can never regress.
# It does not need to cover the whole surface to be useful; it needs to
# never go backwards on the surface it covers.

INVARIANTS = {
    "I-1": ("fail closed on a missing input", ("blind",)),
    "I-2": ("report a denominator", ("denominator",)),
    "I-3": ("prove it can fail", ("unproven",)),
    "I-4": ("declare in one place", ("unregistered", "phantom", "wiring")),
}

# Enforced invariants fail the build. Adding one here is the ratchet
# turning; removing one is the thing this file exists to prevent, and
# `test_gate_registry.py` asserts the set never shrinks.
ENFORCED = ("I-4",)


def invariant_counts(problems: Dict[str, List[str]]) -> Dict[str, int]:
    return {name: sum(len(problems[k]) for k in keys)
            for name, (_, keys) in INVARIANTS.items()}


def verdict(counts: Dict[str, int], enforced=ENFORCED):
    """Return (breaches, ready_to_enforce).

    ``ready`` is the self-advancing half: an invariant that has reached
    zero but is not enforced is debt, because nothing stops it drifting
    back. Reaching zero obliges the flip.
    """
    breaches = sum(counts[n] for n in enforced)
    ready = [n for n in INVARIANTS
             if counts.get(n) == 0 and n not in enforced]
    return breaches, ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 on violations (A-04e). Default reports.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    problems = evaluate()
    _, _, REGISTRY = _load_registry()
    total = sum(len(v) for k, v in problems.items() if k != "pending")

    if args.json:
        print(json.dumps({"checked": len(REGISTRY), "problems": problems},
                         indent=2))
        return 1 if (args.gate and total) else 0

    counts = invariant_counts(problems)
    print(f"Instrumentation invariants — {len(REGISTRY)} checks "
          f"cross-checked\n")

    # Say which teeth are engaged, so "partially on" reads as a design
    # decision rather than an accident.
    for tier, names in (("Enforcing", ENFORCED),
                        ("Reporting ", tuple(n for n in INVARIANTS
                                             if n not in ENFORCED))):
        if names:
            print(f"  {tier}: " + ", ".join(
                f"{n} ({INVARIANTS[n][0]}, {counts[n]})" for n in names))
    print()

    for key, heading in _HEADINGS:
        if problems[key]:
            print(f"  {heading}:")
            for line in problems[key]:
                print(f"    - {line}")
            print()

    if problems["pending"]:
        print("  DECLARED BUT NOT YET ENFORCED — reported every run until wired:")
        for line in problems["pending"]:
            print(f"    - {line}")
        print()

    print(f"  {total} finding(s) across 4 invariants. "
          f"Register: KAI-GATE-001..005.")
    print("  Boundary-blindness scan is a **lower bound** — it matches only")
    print("  the explicit `if not X.exists(): continue` shape, because a")
    print("  survey with false positives invites fixes to working code.")

    # The ratchet advances itself. An invariant that has reached zero but
    # is not yet in ENFORCED is debt, not an achievement — leaving it
    # unenforced lets it silently regress, which is the whole defect
    # class. So reaching zero *obliges* the flip, and says so.
    breaches, ready = verdict(counts)
    if ready:
        print("  READY TO ENFORCE — at zero, so add to ENFORCED now:")
        for name in ready:
            print(f"    - {name} ({INVARIANTS[name][0]})")
        print("  An invariant at zero that nothing enforces will not stay "
              "at zero.\n")

    if not args.gate:
        print(f"  Reporting mode: exits 0 by design. Under --gate this "
              f"would {'FAIL' if breaches or ready else 'PASS'}.")
        return 0

    if breaches:
        print(f"  GATE FAILED: {breaches} breach(es) of "
              f"{', '.join(ENFORCED)}.")
        return 1
    if ready:
        print("  GATE FAILED: an invariant reached zero and is not enforced.")
        return 1
    print(f"  GATE PASSED: {', '.join(ENFORCED)} hold. The rest is "
          f"reported debt, not silence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
