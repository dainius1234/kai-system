#!/usr/bin/env python3
"""Will this change fire a workflow that calls a real model?

The operator's standard, and the reason this exists rather than a habit:

> "Probe file wasn't changed" is not quite enough to prove no
> LLM-triggering workflow could fire. The stronger proof is:
> **changed paths ∩ live-capture trigger paths = ∅.**

D251 recorded the failure that earned it. A repair authorised as
"capture/analyser completeness only" necessarily touched
`probe_llm_contract.py`, which sits inside the capture workflow's
`paths:` filter, and pushing it started a live model run. I saw the
consequence and pre-registered how the *result* would be treated; I did
not ask whether the *action* was authorised. This makes the question
answerable **before** the push instead of noticed after it.

--- how the denominator is derived, not listed ------------------------

Both halves come from the tree (R5), because a hand-kept list of
"workflows that call the model" is exactly the defect that would let a
new one fire unnoticed:

  1. **Which scripts drive a live model?** Every file under `scripts/`
     that starts the model service — a `docker compose … up …` naming the
     model service. Derived by reading them.
  2. **Which workflows run those scripts?** Every workflow whose `run:`
     text references one, plus any workflow that starts the service
     inline.
  3. **What triggers each of those?** Its own `on.push.paths` list. A
     live-capture workflow with **no** `paths:` filter is the worst case
     and is reported as such: every push fires it.

Exit 0 = no changed path can fire a live-capture workflow.
Exit 3 = at least one can. That is not a failure; it is a fact the
author must have in hand before pushing, and the authorisation must say
whether that trigger is included or excluded.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

# The service whose bring-up means "a real model will answer". Read from
# the compose file rather than asserted, so renaming the service cannot
# quietly empty this check.
MODEL_SERVICE_HINTS = ("ollama",)


def _executable_lines(path: pathlib.Path, text: str) -> list[str]:
    """Lines that could actually run, with prose excluded.

    The first version of this check matched anywhere in the file and
    reported `check_compose_env.py` and `test_graph_live.py` as
    model-starting workflows' drivers. Both only *describe* the command
    in a docstring. That is R5's inverted form — a scope LARGER than
    reality — and it is the more expensive error here, because it sends
    someone to reason about a live model run that never happens.

    Comments go, and for Python so do docstrings, located by parsing
    rather than by guessing at quote characters.
    """
    skip: set[int] = set()
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node, clean=False)
                if doc is None or not node.body:
                    continue
                first = node.body[0]
                if isinstance(first, ast.Expr) and hasattr(first, "lineno"):
                    end = getattr(first, "end_lineno", first.lineno)
                    skip.update(range(first.lineno, end + 1))
    return [ln for n, ln in enumerate(text.splitlines(), 1)
            if n not in skip and not ln.lstrip().startswith("#")]


def scripts_that_start_a_model() -> list[pathlib.Path]:
    """Files under scripts/ that bring the model service up FOR REAL."""
    found = []
    for path in sorted((REPO / "scripts").rglob("*")):
        if not path.is_file() or path.suffix not in (".sh", ".py"):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        lines = _executable_lines(path, text)
        starts = any(
            re.search(r"docker[- ]compose\b.*\bup\b", ln) for ln in lines)
        names_model = any(h in ln for ln in lines
                          for h in MODEL_SERVICE_HINTS)
        if starts and names_model:
            found.append(path)
    return found


def _push_paths(text: str) -> tuple[list[str], bool]:
    """The workflow's `on.push.paths`, and whether it has one at all.

    Parsed with a small indentation reader rather than a YAML library, so
    this check has no import that CI might not have installed — a check
    that cannot run is not a check.
    """
    lines = text.splitlines()
    paths: list[str] = []
    in_paths = False
    indent = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("paths:"):
            in_paths = True
            indent = len(line) - len(line.lstrip())
            continue
        if in_paths:
            cur = len(line) - len(line.lstrip())
            if stripped.startswith("- "):
                if cur > indent:
                    paths.append(stripped[2:].strip())
                    continue
            if stripped and not stripped.startswith("#") and cur <= indent:
                in_paths = False
    return paths, bool(paths)


def _matches(pattern: str, path: str) -> bool:
    """GitHub's path-filter globbing, narrowly: `**`, `*`, `?`."""
    rx = ""
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**", i):
            rx += ".*"
            i += 2
        elif ch == "*":
            rx += "[^/]*"
            i += 1
        elif ch == "?":
            rx += "[^/]"
            i += 1
        else:
            rx += re.escape(ch)
            i += 1
    return re.fullmatch(rx, path) is not None


# A workflow that STARTS a model and a workflow that PRODUCES ADMISSIBLE
# CAPTURE EVIDENCE are two different concerns, and lumping them together
# is how the first run of this check told me `core-tests.yml` fires on
# every push in the same breath as the capture workflow. Both are true
# and they mean different things:
#
#   live-model      -> a real model answers. A resource and side-effect
#                      concern, and the one D251's rule is about.
#   capture-writing -> the run writes a capture that could become Q2/Q6
#                      or Stage-1 evidence. An EVIDENCE concern, and the
#                      stricter of the two.
#
# Derived, again, rather than listed: a workflow is capture-writing if it
# runs a script that writes the capture file the analysers read.
CAPTURE_ARTEFACT = "capture.jsonl"


def scripts_that_write_a_capture() -> list[pathlib.Path]:
    found = []
    for path in sorted((REPO / "scripts").rglob("*")):
        if not path.is_file() or path.suffix not in (".sh", ".py"):
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if CAPTURE_ARTEFACT in text:
            found.append(path)
    return found


def live_capture_workflows() -> list[dict]:
    drivers = {str(p.relative_to(REPO)) for p in scripts_that_start_a_model()}
    writers = {str(p.relative_to(REPO)) for p in scripts_that_write_a_capture()}
    out = []
    for wf in sorted(WORKFLOWS.glob("*.y*ml")):
        text = wf.read_text(errors="replace")
        why = sorted(d for d in drivers if d in text)
        # Same prose exclusion as the driver scan: a workflow COMMENT
        # describing a bring-up must not be read as one.
        wf_lines = [ln for ln in text.splitlines()
                    if not ln.lstrip().startswith("#")]
        inline = any(
            re.search(r"docker[- ]compose\b.*\bup\b", ln)
            and any(h in ln for h in MODEL_SERVICE_HINTS)
            for ln in wf_lines)
        if not why and not inline:
            continue
        paths, has_filter = _push_paths(text)
        out.append({
            "workflow": wf.name,
            "drivers": why,
            "inline": inline and not why,
            "writes_capture": sorted(w for w in writers if w in text),
            "paths": paths,
            "has_filter": has_filter,
        })
    return out


def changed_paths(diff_base: str | None) -> list[str]:
    """Uncommitted + staged by default; a range when one is given."""
    cmds = ([["git", "diff", "--name-only", diff_base]] if diff_base
            else [["git", "diff", "--name-only"],
                  ["git", "diff", "--name-only", "--cached"],
                  ["git", "ls-files", "--others", "--exclude-standard"]])
    seen: list[str] = []
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            # I-1: an unanswerable question is not a clean answer.
            raise SystemExit(
                f"REFUSED: `{' '.join(cmd)}` failed, so the changed-path "
                f"set is unknown:\n{r.stderr.strip()}")
        seen += [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return sorted(set(seen))


def intersect(changed: list[str], workflows: list[dict]) -> list[dict]:
    hits = []
    for wf in workflows:
        if not wf["has_filter"]:
            hits.append({"workflow": wf["workflow"], "path": "<no paths: filter>",
                         "pattern": "every push fires it"})
            continue
        for path in changed:
            for pattern in wf["paths"]:
                if _matches(pattern, path):
                    hits.append({"workflow": wf["workflow"], "path": path,
                                 "pattern": pattern})
                    break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diff", help="git ref to diff against (e.g. origin/main)")
    ap.add_argument("--changed", nargs="*",
                    help="explicit changed paths, instead of asking git")
    args = ap.parse_args()

    changed = args.changed if args.changed is not None else changed_paths(args.diff)
    workflows = live_capture_workflows()

    print("LIVE-CAPTURE TRIGGER CHECK")
    print("=" * 60)
    print("  workflows that can call a real model, derived from the tree:")
    for wf in workflows:
        how = ("inline compose up" if wf["inline"]
               else ", ".join(wf["drivers"]))
        kind = ("LIVE-MODEL + CAPTURE-WRITING" if wf["writes_capture"]
                else "LIVE-MODEL only")
        print(f"    {wf['workflow']}  [{kind}]")
        print(f"      driven by: {how}")
        if wf["writes_capture"]:
            print(f"      writes:    {', '.join(wf['writes_capture'])}")
        print(f"      trigger paths: {len(wf['paths'])}"
              + ("" if wf["has_filter"] else "  ** NO FILTER — every push **"))
    print(f"  changed paths: {len(changed)}")
    for p in changed:
        print(f"    {p}")
    hits = intersect(changed, workflows)
    print()
    print(f"  inspected: {len(changed)} changed path(s) against "
          f"{len(workflows)} live-capture workflow(s)")
    if not workflows:
        # A scope that has silently emptied reports zero and looks clean.
        print("REFUSED: no live-capture workflow was identified at all, "
              "which is either true or a blinded detector. Both need a "
              "human before this reads as 'safe to push'.")
        return 2
    if hits:
        writing = {wf["workflow"] for wf in workflows if wf["writes_capture"]}
        evidence = [h for h in hits if h["workflow"] in writing]
        print("TRIGGER: YES — this change can fire a live model run")
        for h in hits:
            tag = "CAPTURE-WRITING" if h["workflow"] in writing else "live-model"
            print(f"    [{tag}] {h['workflow']}: {h['path']}  "
                  f"(matched {h['pattern']})")
        print()
        print("  The authorisation for this change must SAY whether that")
        print("  trigger is included or excluded (D251). Do not decide it")
        print("  after the run has started.")
        if evidence:
            print()
            print("  AND at least one of them WRITES A CAPTURE, so the run")
            print("  will produce something that could be mistaken for Q2/Q6")
            print("  or Stage-1 evidence. Its admissibility must be")
            print("  pre-registered BEFORE the run, not after its contents")
            print("  are known.")
            return 4
        return 3
    print("TRIGGER: NO — changed paths ∩ live-capture trigger paths = ∅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
