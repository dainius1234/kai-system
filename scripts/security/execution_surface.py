"""One definition of what actually runs.

Extracted verbatim from `check_gate_registry.py` under D376 §3, corrected
by D377. **Behaviour-preserving: the code is moved, not reimplemented.**

Two consumers need the same answer to the same question — *which scripts
does this repository actually execute, and can their exit codes fail the
build?* — and two independently maintained answers to one question is the
divergence mechanism this repository has spent a fortnight removing. So
the meta-gate and the operational-portability gate import from here
instead of each holding a copy.

**Why a shared helper rather than one importing the other.** The meta-gate
AUDITS the portability gate: that gate is a registry row it probes and
cross-checks. If the gate imported the meta-gate, the auditor would become
a dependency of the audited — a circular authority shape, and the class of
defect this file exists to prevent. Both import a helper that holds no
authority of its own.

Every scar in the functions below was paid for, and the comments that
record them travel with the code:

* `discover_workflows` is PARSED, not grepped. `policy-checks.yml` had a
  step that lost its `- name:`, so its `run:` became a second `run:` key
  on the step above. YAML keeps the last one: the job displayed one
  gate's name and executed a different gate, and went green.
  `check_compose_env.py` had never run in CI while the text said it had.
* `_makefile_scripts` exists because a workflow running `make check-docs`
  runs `sync_docs.py` just as surely as one naming it directly, and
  without it every possible `in_workflows` value for a make-invoked gate
  was wrong.
* `discover_policy_check` excludes comment lines because a line
  *explaining* why a gate is configured a certain way is not a line that
  configures it — which is how a gate was once reported as invoked by two
  workflows that only named it in a comment.
* `_swallows` knows three shapes and says it is a lower bound. The third
  was found by writing the test for it.

**This module makes no claim of its own and gates nothing.** It reports
what it can derive from workflow YAML and Makefile recipes. Root classes
it does NOT derive — pytest collection, `python -m` entrypoints, shell
entrypoints, Docker `ENTRYPOINT`/`CMD`, Compose `command:`, service
startup — are not covered here, and any consumer that needs them must say
so in its own report rather than assume this module found them.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
SECURITY = REPO / "scripts" / "security"

# Invocation means *running* the script. A bare path match counted a
# comment explaining a gate, and an `echo` naming it in a log message, as
# though they configured it — so this file reported a gate as invoked by
# two workflows that only talked about it. A gate is invoked when a
# python interpreter is pointed at it, and not otherwise.
_INVOCATION = re.compile(
    r"python3?\s+(?:-\S+\s+)*scripts/(?:security/)?([a-z0-9_/]+)\.py")

_MAKE_TARGET = re.compile(r"^\s*make\s+(?:--\S+\s+)*([a-z0-9][a-z0-9_-]*)",
                          re.M)


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
    `run:` key on the step above. YAML keeps the last one: the job
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


# ── NEW IN THIS TRANCHE — not moved from check_gate_registry.py ──────
# Everything above is extracted code. `module_file` is new: the meta-gate
# resolves module names against `scripts/` and `scripts/security/` only,
# because that is the reach of its registry, while the operational
# portability gate must follow a static import closure anywhere in the
# tree. Rather than let the portability gate grow a second resolver, the
# shared module owns the one resolution rule both can live with.
#
# It returns None rather than guessing. An unresolved name is reported by
# the caller, never silently dropped: a name we could not resolve and a
# name with nothing in it are different answers.

#: Directories a bare module name may live in, most specific first.
_SEARCH_ROOTS = ("scripts/security", "scripts", "scripts/ci", "common", "")


def module_file(name: str) -> Optional[Path]:
    """Resolve a module or script name to a file inside the repository.

    Accepts the forms this repository actually uses: a bare stem
    (`check_port_bindings`), a path-ish stem (`ci/live_smoke`), and a
    dotted import (`scripts.security.gate_registry`). Anything that
    resolves outside the repository — a standard-library or site-packages
    module — returns None, because a third-party module's paths are not
    this repository's portability.
    """
    if not name:
        return None
    candidates = []
    dotted = name.replace(".", "/")
    for stem in {name, dotted}:
        for root in _SEARCH_ROOTS:
            base = REPO / root if root else REPO
            candidates.append(base / f"{stem}.py")
            candidates.append(base / stem / "__init__.py")
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(REPO)
        except ValueError:
            continue            # outside the repository: not our subject
        return resolved
    return None
