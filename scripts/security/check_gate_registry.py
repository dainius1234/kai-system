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

# Invocation means *running* the script. A bare path match counted a
# comment explaining a gate, and an `echo` naming it in a log message, as
# though they configured it — so this file reported a gate as invoked by
# two workflows that only talked about it. A gate is invoked when a
# python interpreter is pointed at it, and not otherwise.
_INVOCATION = re.compile(
    r"python3?\s+(?:-\S+\s+)*scripts/(?:security/)?([a-z0-9_/]+)\.py")

#: Modules named without a path live in `scripts/security/`; anything
#: else carries its directory. Both forms resolve here, so the registry
#: never has to hold a second copy of where a file is.
#:
#: This existed as `SECURITY / f"{module}.py"` at four call sites, which
#: is the same assumption written four times — and it silently defined
#: the registry's *reach*: a check outside `scripts/security/` could not
#: be named, so it could not be registered, so it could not be found
#: missing. Eight instruments that can fail the build were outside it.


def module_path(module: str) -> Path:
    """Where a registered module's source lives.

    Resolved by looking, not by a naming convention. The convention —
    *a bare name means `scripts/security/`* — is the obvious rule and it
    is wrong for `scripts/sync_docs.py`, whose bare name is
    indistinguishable from a security module's. Encoding that as
    `if "/" in module` silently dropped three of the eight instruments
    this widening exists to reach, and it dropped them into the *other*
    directory, where the file does not exist and every AST check
    returned "no findings".

    `ambiguous_modules()` covers the one case looking cannot settle.
    """
    direct = SECURITY / f"{module}.py"
    if direct.exists():
        return direct
    return REPO / "scripts" / f"{module}.py"


def ambiguous_modules() -> List[str]:
    """Names that exist in both `scripts/` and `scripts/security/`.

    `module_path` resolves by existence, so a name in both places would
    resolve to one of them and inspect the wrong file — passing while
    checking something else, which is this programme's whole subject.
    None exist today; this is the finding that says so if one appears.
    """
    return sorted(
        p.stem for p in (REPO / "scripts").glob("*.py")
        if (SECURITY / p.name).exists() and not p.stem.startswith("_"))


def _swallows(line: str, step: dict, run: str) -> bool:
    """Does this invocation's exit code get discarded?

    A script whose exit code cannot fail the build is not enforcing, and
    holding it to a gate's invariants would report a defect in code
    behaving exactly as designed — the inverse error, and the worse one.

    Three shapes, because the third was found the hard way. The first
    draft knew `|| true` and `continue-on-error`, and classified
    `behavioral_scoreboard` as enforcing. It is not, twice over: its step
    is

        set +e
        out=$(python scripts/behavioral_scoreboard.py 2>&1)
        ...
        exit 0

    and the script itself ends `asyncio.run(run()); sys.exit(0)`, so the
    score it computes is deliberately advisory. The step even says so in
    its name. Writing the test for it is what surfaced that — which is
    what I-3 is for, aimed at my own detector.

    **This is a lower bound and says so.** A step could `set +e` and then
    `exit 1` on a condition, and deciding that needs shell semantics
    rather than a regex. Under-reporting is the safe direction here for
    the same reason the boundary-blindness scan under-reports: a survey
    with false positives invites people to fix working code.
    """
    if (step or {}).get("continue-on-error") is True:
        return True
    if re.search(r"\|\|\s*(true|echo)", line):
        return True
    # The step manages its own exit code, so the script's is not the
    # build's.
    if re.search(r"^\s*set\s+\+e\b", run, re.M):
        return True
    return bool(re.search(r"^\s*exit\s+0\s*$", run, re.M))

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
    """Every check script that exists, whatever anyone declared.

    A check is a module with a `main()`; a support module is one without.
    That was a hand-maintained list of four names until `isolation_plugin`
    became the fifth and I-4 fired on it — correctly, since a name-based
    exclusion is a second declaration of the same fact and drifts from the
    first. The rule reads the property that actually distinguishes them,
    so a new helper needs no edit here and a new *check* cannot hide as
    one.
    """
    in_security = [
        p.stem for p in SECURITY.glob("*.py")
        if re.search(r"^def main\(", p.read_text(encoding="utf-8"), re.M)
    ]
    return sorted(set(in_security) | set(enforcing_elsewhere()))


def enforcing_elsewhere() -> List[str]:
    """Scripts outside `scripts/security/` that can fail the build.

    The denominator was `scripts/security/*.py` — a *directory*, which is
    where the checks happened to be put, not what makes something an
    instrument. What makes it one is that CI runs it and a non-zero exit
    stops the build.

    Measured on 2026-08-06: 30 modules in that directory, and **eight**
    outside it that can fail the build —

        scripts/behavioral_scoreboard      scripts/ci/kill_isolation
        scripts/ci/assert_clean_bringup    scripts/ci/live_smoke
        scripts/ci/compose_probe           scripts/ci/make_dev_secrets
        scripts/sync_docs                  scripts/test_restart_persistence

    none of them registered, none held to I-1 through I-7, and the
    meta-check printing `GATE PASSED: I-1 … I-7 hold` over all of it.
    The seventeenth venue of this programme's one finding, and this time
    in the file whose entire job is to catch it: **a check whose scope
    was smaller than its name implied.**

    `assert_clean_bringup` made it concrete. It was written this morning
    to enforce in CI, it is the guard that decides whether a bring-up
    succeeded — and because it lives in `scripts/ci/`, the registry could
    not see it, could not find it unregistered, and reported the
    instrumentation sound.

    An invocation whose exit code is swallowed is excluded: it cannot
    fail the build, so holding it to a gate's invariants would report a
    defect in code doing exactly what it was written to do.
    """
    import yaml

    found: Dict[str, bool] = {}
    make_targets: set = set()
    for path in workflow_files():
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue        # `cross_check` owns the unparseable-workflow finding
        for job in (doc.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                run = str((step or {}).get("run") or "")
                # Join shell continuations first: a `|| true` after a
                # trailing `\` belongs to the same command, and reading
                # it per physical line got that wrong the first time.
                joined = re.sub(r"\\\s*\n\s*", " ", run)
                swallowed_step = _swallows("", step, run)
                for line in joined.splitlines():
                    for module in _INVOCATION.findall(line):
                        if (SECURITY / f"{module}.py").exists():
                            continue        # already in the directory scan
                        enforcing = not _swallows(line, step, run)
                        found[module] = found.get(module, False) or enforcing
                if not swallowed_step:
                    make_targets |= set(_MAKE_TARGET.findall(joined))

    # A script can also enforce *through* a make target — `check-docs`
    # runs `sync_docs.py --check`, and reading only workflow `run:` lines
    # missed it. Measured before it was added, because widening a scope
    # past the evidence is the worse defect: the 36 targets CI invokes
    # run exactly **two** scripts directly, so this is a small, bounded
    # extension rather than a floodgate.
    #
    # Recipe lines only, not prerequisites. `test-uh` is a target whose
    # prerequisites are forty suites, and those are watched by
    # `check_assertion_floors` and `check_suite_floor` — a different
    # instrument, verified to cover them, not an assumption made here.
    for module in _makefile_scripts(make_targets):
        if not (SECURITY / f"{module}.py").exists():
            found[module] = True
    return sorted(m for m, enforcing in found.items() if enforcing)


_MAKE_TARGET = re.compile(r"^\s*make\s+(?:--\S+\s+)*([a-z0-9][a-z0-9_-]*)",
                          re.M)


def _makefile_scripts(targets: set) -> List[str]:
    """Scripts run directly by the recipe of any of `targets`."""
    if not targets:
        return []
    lines = (REPO / "Makefile").read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for i, line in enumerate(lines):
        match = re.match(r"^([a-z0-9][a-z0-9_-]*):", line)
        if not match or match.group(1) not in targets:
            continue
        for j in range(i + 1, len(lines)):
            if lines[j] and not lines[j].startswith(("\t", " ")):
                break
            if lines[j].lstrip().startswith("#"):
                continue
            out.extend(_INVOCATION.findall(lines[j]))
    return sorted(set(out))


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
    # Comments mention scripts; they do not run them. Without this, a
    # line explaining *why* a gate is configured a certain way counts as
    # configuring it — which is how this file first reported a gate as
    # invoked by two workflows that only named it in a comment.
    body = "\n".join(l for l in block if not l.lstrip().startswith("#"))
    return sorted(set(_INVOCATION.findall(body)))


def workflow_files() -> List[Path]:
    """Every workflow, both extensions. GitHub accepts `.yaml` too."""
    root = REPO / ".github" / "workflows"
    return sorted(p for p in root.glob("*.y*ml") if p.is_file())


def discover_workflows() -> Dict[str, List[str]]:
    """Map module -> the workflow files that actually invoke it.

    **Parsed, not grepped.** This read the raw text and matched any line
    that looked like an invocation, which is not the same question as
    "does this step run". `policy-checks.yml` had:

        - name: Every compose bring-up supplies the variables it needs
          run: python scripts/security/check_compose_env.py

          run: python scripts/security/check_test_wiring.py

    — a step that lost its `- name:`, so its `run:` became a *second*
    `run:` key on the step above. YAML keeps the last one. The job
    displayed the compose-env name, executed the test-wiring gate, and
    went green; `check_compose_env.py` never ran in CI at all.

    The text said both were wired. Only the parse knows which one runs,
    and I-4's whole job is to make the registry agree with reality.
    """
    import yaml

    found: Dict[str, List[str]] = {}
    for path in workflow_files():
        # I-1: an unparseable workflow is a finding, not a file to skip.
        # `main()` surfaces it as a phantom/wiring disagreement rather
        # than silently shrinking the survey.
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job in (doc.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                run = str((step or {}).get("run") or "")
                for module in _INVOCATION.findall(run):
                    found.setdefault(module, []).append(path.name)
                # A workflow that runs `make check-docs` runs
                # `sync_docs.py` just as surely as one naming it
                # directly. Without this, a make-invoked gate could
                # never have a declaration that matches reality: it
                # would be discovered as enforcing by
                # `enforcing_elsewhere` and as invoked by nobody here,
                # so every possible `in_workflows` value was wrong. The
                # two discoveries have to share one idea of "invoked",
                # or the cross-check is comparing different questions.
                for module in _makefile_scripts(
                        set(_MAKE_TARGET.findall(run))):
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


def _is_skip(stmt: ast.stmt) -> bool:
    """True when this statement shrugs off a missing input.

    `continue` and `pass` always shrug. A `return` depends entirely on
    what it returns, and the first version of this detector did not look
    — it treated every `return` as a skip and over-counted by **4 of 15
    (27%)**. Those four were the opposite of the defect:

        return AnchorFailure("absent", ...)      # refuses to judge
        return [Violation(5, "legacy_bridge")]   # reports the violation
        return (LIVE, "no auth.js; the UI ...")  # reports the finding

    Sweeping mechanically would have replaced four correct fail-closed
    returns with `require()` calls and called it a fix. A survey with
    false positives invites exactly that.

    So: a bare `return`, `None`, an empty literal, or a lone name (the
    accumulator, empty at that point) is a skip. Anything constructed —
    a call, a populated list or tuple — is a refusal.
    """
    if isinstance(stmt, (ast.Continue, ast.Pass)):
        return True
    if not isinstance(stmt, ast.Return):
        return False
    value = stmt.value
    if value is None:
        return True
    if isinstance(value, ast.Constant) and value.value is None:
        return True
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and not value.elts:
        return True
    if isinstance(value, ast.Dict) and not value.keys:
        return True
    return isinstance(value, ast.Name)


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
    path = module_path(module)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not _is_absence_test(node.test):
            continue
        if all(_is_skip(stmt) for stmt in node.body):
            lines.append(node.lineno)
    return sorted(lines)


# ── I-5: rules that exist in syntax but have no effect ───────────────

def inert_rules(module: str) -> List[str]:
    """Named rules the code declares and never consults.

    Three instances of this appeared while reading the gates, all with
    the same signature — the code's self-description and its behaviour
    had diverged:

      - `check_compose_drift`: `if net_cfg.get("internal"): pass`
      - `check_restart_recovery`: `ALLOWED_RESTART` declared, never read,
        so `restart: nonsense-value` passed while the docstring promised
        an allowlist
      - `check_network_zones`: `if svc_nets is None: pass`, under a
        docstring claiming "every service has an explicit networks
        assignment"

    The operator's framing is the sharp one: an unused import can be
    cruft, but **a declared-but-unreferenced constant with a
    security-shaped name is a claim the code makes about itself that
    is not true.** The docstring says "we allowlist"; the constant exists
    to prove it; the constant is wired to nothing.

    Both shapes are detected: a policy-shaped constant nobody reads, and
    a conditional whose body is exactly `pass`.
    """
    path = module_path(module)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []

    findings: List[str] = []

    declared = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                declared[target.id] = node.lineno

    referenced = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced.add(node.id)

    # Only policy-shaped names. A leftover URL constant is cruft; an
    # unread ALLOWED_/REQUIRED_/BANNED_ constant is a lapsed assertion.
    policy = ("ALLOW", "DENY", "BANNED", "REQUIRED", "FORBIDDEN",
              "MUTABLE", "DANGEROUS", "PROTECTED", "ISOLATED", "INSECURE",
              "TRUSTED", "PRIMARY", "SECRET", "INTERNAL", "EXTERNAL")
    for name, lineno in sorted(declared.items()):
        if name in referenced or not any(p in name for p in policy):
            continue
        findings.append(
            f"{module}.py:{lineno}: {name} is declared and never read — "
            f"a rule the code claims and does not apply")

    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
                and not node.orelse):
            findings.append(
                f"{module}.py:{node.lineno}: a condition whose body is "
                f"`pass` — a rule that looks present and does nothing")

    return findings


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
        [sys.executable, str(module_path(gate.module))],
        capture_output=True, text=True, cwd=str(REPO), timeout=300,
    )
    output = result.stdout + result.stderr
    match = re.search(gate.denominator, output)
    if match:
        return "ok", match.group(0)
    return "absent", f"output did not match /{gate.denominator}/"


# ── The cross-check ──────────────────────────────────────────────────

def cross_check(registry, on_disk, in_policy, in_flows,
                blind=None, probe=None, repo_has=None, inert=None,
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
    inert = inert or (lambda module: [])
    probe = probe or (lambda gate: ("ok", ""))
    repo_has = repo_has or (lambda rel: True)

    by_module = {g.module: g for g in registry}
    declared = set(by_module)
    on_disk = set(on_disk)
    in_policy = set(in_policy)

    # Derived from `_HEADINGS`, not typed beside it. The literal list
    # this replaces was missing "misreported" the moment that category
    # was added, and a second copy in `test_gate_registry.py` raised a
    # KeyError instead of a finding — a hand-written list of what to
    # collect, drifting from the thing that collects. Same defect, fifth
    # venue.
    problems: Dict[str, List[str]] = empty_problems()

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

        # I-5 — rules declared and never applied.
        problems["inert"] += inert(module)

        # I-2 — a denominator, or an explicit decline.
        status, detail = probe(gate)
        if status in ("missing", "absent"):
            problems["denominator"].append(f"{module}: {detail}")

    return problems


def evaluate() -> Dict[str, List[str]]:
    """`cross_check` wired to the real repository, plus closure review.

    A closed finding is a claim that a defect cannot recur. If the thing
    preventing it is removed — I-5 dropped from `ENFORCED`, a gate lifted
    out of `policy-check` — then the claim is no longer true and the
    finding **re-opens**. Closure that nobody re-checks is exactly the
    kind of label that decays into a rubber stamp.
    """
    GATE, REPORT, REGISTRY = _load_registry()
    problems = cross_check(
        REGISTRY,
        on_disk=discover_modules(),
        in_policy=discover_policy_check(),
        in_flows=discover_workflows(),
        blind=skips_absent_input,
        probe=probe_denominator,
        inert=inert_rules,
        repo_has=lambda rel: (REPO / rel).exists(),
        kinds=(GATE, REPORT),
    )
    from scripts.security.closure_register import lapsed
    problems["lapsed"] += lapsed()
    problems["misreported"] += misreported_closures()
    problems["uncalibrated"] += uncalibrated_ratchets(
        lambda rel: (REPO / rel).exists())
    return problems


def uncalibrated_ratchets(repo_has) -> List[str]:
    """Ratchets that cannot show their instrument still measures (I-7).

    A gate bounding a MAXIMUM is satisfied by zero, and zero is exactly
    what a detector that has stopped detecting reports. The bound is
    enforced correctly; the silence is the danger. A tokenising bug took
    the hygiene survey's `clients` from 16 to 0 and its adoption count
    from 149 to 0, and the gate passed — doing precisely what a ratchet
    does. It was caught because 0 was implausible and somebody looked,
    which is not a control.

    So every `ratchet=True` gate must name a suite that points its
    detector at input whose answer is known *before* pointing it at the
    repository. A historical baseline says "this is what we saw last
    time"; it cannot say whether last time's instrument was working.

    Fails closed on a `calibrated_by` naming a file that is not there —
    otherwise deleting the calibration suite would satisfy the check
    that exists to require it.
    """
    _gate, _report, registry = _load_registry()
    out = []
    for gate in registry:
        if not getattr(gate, "ratchet", False):
            continue
        declared = getattr(gate, "calibrated_by", None)
        if not declared:
            out.append(f"{gate.module}: ratchets a stored baseline and "
                       f"declares no calibration")
            continue
        path = declared.split(" ")[0].split(" —")[0]
        if path.endswith(".py") and not repo_has(path):
            out.append(f"{gate.module}: calibrated_by names {path}, "
                       f"which is not in the repository")
    return out


ARCHITECTURE_DOC = REPO / "kai-pm" / "INSTRUMENTATION_ARCHITECTURE.md"


def misreported_closures() -> List[str]:
    """Findings the register calls CLOSED that the doc summary does not.

    The closure register is enforced — every record carries a
    `still_holds` predicate re-evaluated on each run. The status table in
    `INSTRUMENTATION_ARCHITECTURE.md` is a *second* declaration of the
    same state, hand-maintained, and on 2026-08-05 **all ten** closures
    were misreported there: the machine-checked source said CLOSED and
    the document a human reads said OPEN or REMEDIATED. A reader would
    have concluded nothing had ever closed.

    That is I-4 — declared in more than one place with nothing
    cross-checking them — surviving in the documentation layer, which is
    the one place the invariant had not been pointed at. Fails closed:
    an unreadable or table-less document is a finding, not a pass.
    """
    from scripts.security.closure_register import CLOSED
    try:
        doc = ARCHITECTURE_DOC.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{ARCHITECTURE_DOC.name}: unreadable ({exc})"]
    # Greedy up to the LAST `| **` on the line: the verdict is always the
    # final column, and a description may legitimately contain bold — the
    # row for KAI-GATE-023 opens with it, and a non-greedy match read the
    # "E" of "**Every ratchet..." as the status.
    rows = dict(re.findall(r"^\| `(KAI-GATE-\d+)` \|.*\| \*\*([A-Z]+)",
                           doc, re.M))
    if not rows:
        return [f"{ARCHITECTURE_DOC.name}: no finding table found — the "
                f"cross-check has nothing to compare against"]
    out = []
    for closure in CLOSED:
        stated = rows.get(closure.finding, "ABSENT")
        if stated != "CLOSED":
            out.append(f"{closure.finding}: register=CLOSED, "
                       f"{ARCHITECTURE_DOC.name}={stated}")
    return out


def empty_problems() -> Dict[str, List[str]]:
    """One empty bucket per reported category, derived from `_HEADINGS`.

    The single source of truth for what a problems-dict contains. Adding
    a heading adds its bucket everywhere, including in tests, so a new
    category cannot be half-wired.
    """
    buckets = {key: [] for key, _ in _HEADINGS}
    buckets.setdefault("pending", [])   # reported, but not a finding
    return buckets


_HEADINGS = [
    ("unregistered", "UNREGISTERED — on disk, undeclared (I-4)"),
    ("phantom", "PHANTOM — declared, not on disk (I-4)"),
    ("wiring", "WIRING DISAGREES with the declaration (I-4)"),
    ("blind", "BOUNDARY BLINDNESS — absence reads as correctness (I-1)"),
    ("denominator", "NO DENOMINATOR — a pass that cannot be falsified (I-2)"),
    ("unproven", "NEVER OBSERVED FAILING (I-3)"),
    ("inert", "INERT RULES — declared, never applied (I-5)"),
    ("lapsed", "CLOSURES RE-OPENED — the prevention no longer holds (I-6)"),
    ("misreported", "CLOSURE STATE MISREPORTED — the summary a human "
                    "reads disagrees with the register (I-4)"),
    ("uncalibrated", "RATCHET WITH NO CALIBRATION — a bound that zero "
                     "satisfies, and zero is what a blinded detector "
                     "reports (I-7)"),
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
    "I-4": ("declare in one place",
            ("unregistered", "phantom", "wiring", "misreported")),
    "I-5": ("no inert rules", ("inert",)),
    "I-6": ("closures still hold", ("lapsed",)),
    "I-7": ("a ratchet proves it still measures", ("uncalibrated",)),
}

# Enforced invariants fail the build. Adding one here is the ratchet
# turning; removing one is the thing this file exists to prevent, and
# `test_gate_registry.py` asserts the set never shrinks.
# I-5 joined here because it reached zero and the gate refused to pass
# until it did — the ratchet advancing itself, on its own author.
# Each name here was added because the gate refused to pass without it:
# I-5 when the inert-rule detector cleared, I-6 on its first run, I-2
# when the last six compose gates got a denominator. The ratchet has
# advanced itself three times; nobody remembered to flip anything.
# All six. Every one was added because the gate refused to pass without
# it — I-5 when the inert-rule detector cleared, I-6 on its first run,
# I-2 when the last compose gates got denominators, I-1 and I-3 when the
# retrofit finished. Nobody ever remembered to flip one.
ENFORCED = ("I-1", "I-2", "I-3", "I-4", "I-5", "I-6", "I-7")


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
    from scripts.security.closure_register import CLOSED
    # Both numbers, because they answer different questions and only
    # their disagreement is interesting. This printed `len(REGISTRY)`
    # alone — what somebody declared — so a headline of "30 checks
    # cross-checked" was true of the register and said nothing about
    # the eight build-failing instruments outside it. A denominator
    # taken from the declaration cannot reveal a missing declaration.
    on_disk = len(discover_modules())
    print(f"Instrumentation invariants — {len(REGISTRY)} declared, "
          f"{on_disk} found on disk, {len(CLOSED)} finding(s) closed "
          f"and re-verified\n")

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
