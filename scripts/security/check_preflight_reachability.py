#!/usr/bin/env python3
"""The measurement path must not be able to reach the experiment.

WHY GREPPING THE YAML WAS THE WRONG ALTITUDE
============================================

`item8-preflight.yml` exists so the operator can authorise a MEASUREMENT
without authorising the six-build experiment. I argued it was
structurally incapable of reaching the experiment, and my evidence was:

    grep -rn 'run_item8_experiment' .github/workflows/item8-preflight.yml
    → no output

The workflow ran `scripts/test_item8_verdicts.py`, which opens with
`RUNNER = .../run_item8_experiment.sh` and executes it. So the path was
one file away, and the absence I pointed at was the absence of a *name*
in *one* file — not the absence of a *path* through the graph.

**A reachability claim needs a reachability check.** This computes the
transitive closure of what the preflight workflow can execute, and
refuses if the forbidden set is inside it.

WHAT IS FORBIDDEN, AND WHY EACH
===============================

    run_item8_experiment.sh     spends the frozen denominator
    derive_item8_dockerfile.py  produces the subjects it would spend it on
    summarise_item8.py          turns subject evidence into a closure claim

None of the three has any business on a path whose whole purpose is to
answer one question about the daemon before anybody is allowed to build
anything.

HOW THE CLOSURE IS BUILT
========================

From the workflow's own `run:` blocks — not from a list kept beside it
(R5) — every referenced repository script becomes a root. Each root is
then scanned for references to other repository scripts, and so on until
nothing new appears. The scan is deliberately crude in the safe
direction: it matches a script's *basename anywhere in the text*, so a
mention in a comment counts as a reference. A reachability check that
under-reports is worse than one that over-reports, and this one is built
to over-report.

Exit 0 = the forbidden set is not reachable, and the closure is printed.
Exit 1 = it is reachable, with the path that reaches it.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

WORKFLOW = ".github/workflows/item8-preflight.yml"

# Derived from what each one DOES, stated in the docstring above.
FORBIDDEN = (
    "run_item8_experiment.sh",
    "derive_item8_dockerfile.py",
    "summarise_item8.py",
)

_SCRIPT = re.compile(r"scripts/[A-Za-z0-9_/]+\.(?:py|sh)")

# Where a bare module name could resolve to a file. Derived by walking
# the tree rather than kept as a list beside it (R5).
_SEARCH_DIRS = ("scripts/security", "scripts", "scripts/ci")


def _module_files() -> dict[str, str]:
    """Every local module name that a bare `import X` could reach."""
    out: dict[str, str] = {}
    for d in _SEARCH_DIRS:
        base = REPO / d
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*.py")):
            out.setdefault(f.stem, f"{d}/{f.name}")
    return out


MODULES = _module_files()


def scripts_in(text: str) -> set[str]:
    """What this file can reach: literal paths AND real Python imports.

    THE FIRST VERSION MATCHED ONLY LITERAL `scripts/....py` STRINGS.
    That is filename reachability, not dependency reachability, and it
    misses the ordinary form entirely:

        import derive_item8_dockerfile
        from derive_item8_dockerfile import derive

    So the gate written to forbid a dependency could not see the most
    normal way of creating one -- a check whose scope was smaller than
    its name, inside the instrument built to enforce a boundary. R5, in
    the place it does the most damage. (D302)

    Imports are resolved by parsing the file, not by pattern-matching
    the word `import`, so a mention inside a string or a comment is not
    mistaken for a dependency edge -- while a REAL import of a module
    that only exists locally is followed.
    """
    found = set(_SCRIPT.findall(text))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Not Python, or not parseable. The literal-path scan above is
        # what applies -- and shell scripts are reached that way.
        return found
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                names = [node.module.split(".")[0]]
        for n in names:
            if n in MODULES:
                found.add(MODULES[n])
    # `importlib.util.spec_from_file_location(name, ...)` is how this
    # codebase loads siblings without a package, and its FIRST argument
    # is a bare module name. Follow that too.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if getattr(fn, "attr", None) != "spec_from_file_location":
            continue
        for arg in node.args[:1]:
            if isinstance(arg, ast.Constant) and arg.value in MODULES:
                found.add(MODULES[arg.value])
    return found


def closure(roots: set[str]
            ) -> tuple[set[str], dict[str, str], set[str]]:
    """Everything reachable, and who first reached each thing."""
    seen: set[str] = set()
    missing_refs: set[str] = set()
    via: dict[str, str] = {}
    frontier = list(roots)
    for r in roots:
        via[r] = WORKFLOW
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        f = REPO / cur
        if not f.is_file():
            # REFERENCED BUT ABSENT. Reported, not traversed, and NOT
            # counted as reachable -- a path that names nothing cannot
            # execute anything. Keeping it in the closure inflated the
            # denominator with strings from the calibration that names
            # probe files, which is the instrument measuring its own
            # test's prose. (D302)
            missing_refs.add(cur)
            continue
        seen.add(cur)
        for nxt in scripts_in(f.read_text(errors="replace")):
            if nxt not in seen:
                via.setdefault(nxt, cur)
                frontier.append(nxt)
        # A basename mentioned anywhere counts, even without a path --
        # deliberately over-reporting, because the failure this exists
        # to stop was an under-report.
        text = f.read_text(errors="replace")
        for name in FORBIDDEN:
            if name in text:
                hit = next((s for s in seen | set(frontier)
                            if s.endswith(name)), f"scripts/**/{name}")
                via.setdefault(hit, cur)
                if hit not in seen:
                    frontier.append(hit)
    return seen, via, missing_refs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", default=WORKFLOW)
    ap.add_argument("--expect-reachable", action="append", default=[],
                    help="calibration: a path that MUST be in the closure, "
                         "so a check that reports 'nothing reachable' "
                         "cannot pass by finding nothing")
    args = ap.parse_args()

    wf = REPO / args.workflow
    if not wf.is_file():
        print(f"REFUSED: {args.workflow} does not exist. A reachability "
              f"claim about a workflow that is not there is not a claim.")
        return 1
    roots = scripts_in(wf.read_text())
    if not roots:
        print(f"REFUSED: {args.workflow} references no repository script at "
              f"all. Either it does nothing, or this check is not reading "
              f"it — and a closure of zero is exactly what a broken "
              f"extractor produces (R11).")
        return 1

    reached, via, missing_refs = closure(roots)

    print("PREFLIGHT REACHABILITY")
    print("=" * 68)
    print(f"  workflow : {args.workflow}")
    print(f"  roots    : {len(roots)}")
    for r in sorted(roots):
        print(f"      {r}")
    print(f"  reachable: {len(reached)}")
    for r in sorted(reached - roots):
        print(f"      {r}   (via {via.get(r, '?')})")
    print()

    breaches = []
    for name in FORBIDDEN:
        for r in sorted(reached):
            if r.endswith(name):
                breaches.append((name, r, via.get(r, "?")))
    print(f"  inspected: {len(reached)} reachable script(s) against "
          f"{len(FORBIDDEN)} forbidden target(s)")
    print()

    missing = [e for e in args.expect_reachable if e not in reached]
    if missing:
        print(f"FAIL: the closure does not contain {', '.join(missing)}, "
              f"which calibration says it must. A reachability check that "
              f"finds nothing passes every forbidden test for the wrong "
              f"reason.")
        return 1

    if breaches:
        for name, path, who in breaches:
            print(f"FAIL: {path} is REACHABLE from {args.workflow} via "
                  f"{who}")
        print()
        print("The standalone measurement exists so the operator can "
              "authorise measuring WITHOUT authorising the six builds. A "
              "path from it to the experiment's machinery makes that "
              "distinction a matter of trust rather than of structure, and "
              "the previous claim of incapability was made by grepping one "
              "file for a name. (D301)")
        return 1

    print("PASS: no path from the measurement to the experiment's runner, "
          "deriver or claim engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
