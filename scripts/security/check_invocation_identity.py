#!/usr/bin/env python3
"""Did a repair change what the model is actually asked?

Stage 1 replays ONE captured request. A repair to the instrument around
it — where the output file lands, how a refusal is worded, what the CI
log may print — is only admissible if the model-facing invocation is
bit-for-bit the same as the one that was frozen. "I only touched the
plumbing" is an assertion (R1). This computes it.

HOW THE SURFACE IS DERIVED, RATHER THAN LISTED
==============================================

R5: a check's scope must come from the tree, not from a list kept beside
it. So the only thing named by hand here is the QUESTION —

    which code builds the request, and which code sends it

— expressed as two seed definitions. Everything else is derived: the
transitive closure of module-level names those seeds reference, plus any
imported module they reach into, whose file is then required to be
unchanged in its entirety.

The report prints EVERY top-level definition in the file with its status
and whether it is inside the surface, so the scope is visible and can be
disputed. A check whose scope you cannot see is a check you cannot
calibrate.

WHAT COUNTS AS A BREACH
=======================

Any in-surface definition whose source segment changed, appeared or
disappeared, and any in-surface module file whose bytes changed. Out-of-
surface changes are reported and are NOT failures — that is the whole
point of the exercise.

A caveat this cannot cover, and says so rather than implying otherwise:
it compares CODE. Values supplied from outside the file — a workflow's
`--url`, `--timeout` or model env — are not in these bytes, and must be
diffed where they live.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

# The QUESTION, not the scope: the definition that assembles the request
# body and the definition that transmits it. The scope is derived from
# these by `surface_of`.
SEEDS = ("freeze", "send_once")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def top_level(src: str) -> dict[str, str]:
    """Every module-level definition, mapped to its own source text."""
    tree = ast.parse(src)
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node) or ""
        elif isinstance(node, ast.Assign):
            seg = ast.get_source_segment(src, node) or ""
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = seg
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            out[node.target.id] = ast.get_source_segment(src, node) or ""
    return out


def import_aliases(src: str) -> dict[str, str]:
    """Local alias -> imported module name, for `import x as y` forms."""
    out: dict[str, str] = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def _names(segment: str) -> set[str]:
    if not segment.strip():
        return set()
    return {n.id for n in ast.walk(ast.parse(segment))
            if isinstance(n, ast.Name)}


def surface_of(defs: dict[str, str], seeds=SEEDS) -> set[str]:
    """Transitive closure of module-level names reachable from the seeds."""
    seen: set[str] = set()
    stack = [s for s in seeds]
    while stack:
        name = stack.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        stack.extend(n for n in _names(defs[name])
                     if n in defs and n not in seen)
    return seen


def reached_modules(defs: dict[str, str], surface: set[str],
                    aliases: dict[str, str]) -> dict[str, str]:
    """Modules the surface reaches into: local alias -> imported name.

    The alias is what appears in the code (`s1.select`); the imported
    name is what a file is found by (`select_replay_subject`). Keeping
    only the alias makes every aliased repo module look like a stdlib
    one, which is a scope smaller than the check's name implies (R5).
    """
    out: dict[str, str] = {}
    for name in surface:
        for n in _names(defs[name]):
            if n in aliases:
                out[n] = aliases[n]
    return out


def module_file(module: str) -> str | None:
    """The repo file an imported module resolves to, if it is ours."""
    tail = module.split(".")[-1]
    for rel in (f"scripts/security/{tail}.py", f"scripts/{tail}.py"):
        if (REPO / rel).is_file():
            return rel
    return None


def compare(old_src: str, new_src: str, seeds=SEEDS) -> dict:
    """The full OLD->NEW table, and which rows breach the surface."""
    old, new = top_level(old_src), top_level(new_src)
    # The surface is derived from BOTH revisions: a definition deleted in
    # the new tree is still in scope, or a removal would hide itself.
    surface = surface_of(old, seeds) | surface_of(new, seeds)
    rows = []
    for name in sorted(set(old) | set(new)):
        o, n = old.get(name), new.get(name)
        if o is None:
            status = "ADDED"
        elif n is None:
            status = "REMOVED"
        elif o == n:
            status = "unchanged"
        else:
            status = "CHANGED"
        rows.append({"name": name, "status": status,
                     "in_surface": name in surface,
                     "old": _sha(o)[:12] if o is not None else "-",
                     "new": _sha(n)[:12] if n is not None else "-"})
    breaches = [r for r in rows
                if r["in_surface"] and r["status"] != "unchanged"]
    return {"rows": rows, "surface": surface, "breaches": breaches,
            "modules": reached_modules(new, surface & set(new),
                                       import_aliases(new_src))}


def git_show(rev: str, path: str) -> str:
    out = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"cannot read {path} at {rev}: {out.stderr.strip()}")
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True, help="git revision before")
    ap.add_argument("--new", default="", help="git revision after; "
                                              "default = the working tree")
    ap.add_argument("--file", default="scripts/security/stage1_replay.py")
    ap.add_argument("--seed", action="append", default=[],
                    help=f"override the seed definitions (default {SEEDS})")
    args = ap.parse_args()

    seeds = tuple(args.seed) or SEEDS
    old_src = git_show(args.old, args.file)
    new_src = (git_show(args.new, args.file) if args.new
               else (REPO / args.file).read_text())
    result = compare(old_src, new_src, seeds)

    print("MODEL-FACING INVOCATION IDENTITY")
    print("=" * 72)
    print(f"  file        : {args.file}")
    print(f"  OLD         : {args.old}")
    print(f"  NEW         : {args.new or '(working tree)'}")
    print(f"  seeds       : {', '.join(seeds)}  (the question; the scope "
          f"below is derived from them)")
    print()
    print(f"  {'definition':<28} {'status':<10} {'surface':<8} old -> new")
    for r in result["rows"]:
        print(f"  {r['name']:<28} {r['status']:<10} "
              f"{'YES' if r['in_surface'] else '-':<8} "
              f"{r['old']} -> {r['new']}")
    print()

    # Any module the surface reaches into must be unchanged in full: we
    # cannot reason about which of its definitions the surface touches
    # without importing it, so the whole file is the unit.
    mod_breaches = []
    for alias, module in sorted(result["modules"].items()):
        cand = module_file(module)
        if cand is None:
            print(f"  module reached by the surface: {alias} -> {module} "
                  f"(not a repo file — stdlib or third-party, pinned by the "
                  f"image)")
            continue
        o = git_show(args.old, cand)
        n = (git_show(args.new, cand) if args.new
             else (REPO / cand).read_text())
        same = o == n
        print(f"  module reached by the surface: {alias} -> {cand} "
              f"{'unchanged' if same else 'CHANGED'} "
              f"({_sha(o)[:12]} -> {_sha(n)[:12]})")
        if not same:
            mod_breaches.append(cand)
    print()
    print(f"  inspected: {len(result['rows'])} top-level definition(s), "
          f"{sum(1 for r in result['rows'] if r['in_surface'])} in the "
          f"model-facing surface")
    print("  NOT COVERED: values supplied from outside this file (a "
          "workflow's --url, --timeout, model env). Diff those where "
          "they live.")

    if result["breaches"] or mod_breaches:
        print()
        print("BREACH — the model-facing invocation is NOT identical:")
        for r in result["breaches"]:
            print(f"    {r['name']}: {r['status']} ({r['old']} -> {r['new']})")
        for m in mod_breaches:
            print(f"    {m}: file contents CHANGED")
        return 1
    print()
    print("  IDENTICAL — every definition that builds or sends the request "
          "is byte-for-byte unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
