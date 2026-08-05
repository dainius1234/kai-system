"""Repository-wide survey of HTTP and time hygiene defects.

Three defects the dashboard audit found are not dashboard defects — they
are repository-wide habits. This counts them everywhere, so the scale is
a command rather than a number somebody typed into a document once.

  per-request clients  a connection pool built and torn down per request
  unbounded bodies     `await request.json()` with no size or shape limit
  naive timestamps     `datetime.utcnow()` — no timezone
  200-on-failure       an except-path that returns a success-shaped body

See `kai-pm/W1_GLOBAL_HYGIENE_SUBPLAN.md`. The totals should only ever
fall; this is the measurement that makes that checkable.

**The ratchet.** With ``--gate`` the current totals are compared against
`scripts/security/hygiene_baseline.json`. The run fails if any count has
*risen*. That is deliberately weaker than "must be zero": a gate that
starts red is a gate people learn to ignore, and 136 pre-existing
instances cannot be fixed in one change. A ratchet is honest about the
debt while making it impossible to add more — and every time the debt
falls, ``--update-baseline`` locks the improvement in.

Exit codes:
  0  survey clean, or counts unchanged/improved under --gate
  1  --gate and a count has risen, or --max-total exceeded
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent

# Adopted replacements. A service using these is not counted against.
POOLED = "pooled_client("
BOUNDED = "bounded_json("

# Declared once. `COLUMNS` is derived from the detectors rather than typed
# beside them, because the two drifted the moment a fifth detector was
# added: `silent_swallows` landed in survey() and this survey reported
# **zero** across a repository holding 156 of them, purely because the
# tuple here had not been updated. Identical in shape to a deprecation
# rule, fixed the same afternoon, whose seven hand-written filenames could
# not see the five files that actually had the defect.
#
# Populated below, once the detector functions exist.
DETECTORS = {}
ADOPTION_DETECTORS = {}
COLUMNS: tuple = ()


# Every first-party module, not only the entry points.
#
# This started as `*/app.py`, which missed `agentic/introspect_app.py`
# and 2 of its naive timestamps. Widened to the four globs below, which
# then missed all 117 library modules — and a defect in `common/llm.py`
# or `common/market_cache.py` reaches *every* service, so the files least
# covered were the ones with the widest blast radius. Widening again
# added 16 per-request clients and 30 silent swallows the ratchet had
# never been able to see.
#
# Twice now the scope has been a hand-written list of where to look, and
# twice it has been narrower than the problem. It is derived from the
# tree now: everything first-party that is not a test, a tool, or
# vendored. Adding a service or a module cannot leave it unscanned,
# which is the property that matters — the previous versions failed open,
# and a survey that undercounts is worse than none because the number
# still looks authoritative.
_EXCLUDED_DIRS = {"node_modules", ".venv", "venv", "site-packages",
                  "scripts", "tests", "kai-pm", "_archive", "__pycache__",
                  ".git", "_migrations"}


def _service_files() -> List[Path]:
    """Every first-party module, excluding tests, tooling and vendored code."""
    found = set()
    for path in REPO.rglob("*.py"):
        if any(part in _EXCLUDED_DIRS for part in path.parts):
            continue
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        found.add(path)
    return sorted(found)


def _success_on_failure(text: str) -> int:
    """Except-paths returning a dict literal, in route handlers only."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    routed = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            src = ast.unparse(dec)
            if re.search(r"\.(get|post|put|delete|patch)\(", src):
                routed.add(node.name)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in routed:
            continue
        for handler in _own_except_handlers(node):
            body = ast.unparse(handler)
            if any(m in body for m in ("raise", "status_code", "degraded_response",
                                       "JSONResponse", "HTTPException", "_sse_error")):
                continue
            if re.search(r"return \{", body):
                count += 1
                break
    return count


def _own_except_handlers(node) -> List[ast.ExceptHandler]:
    """Except-handlers belonging to this function, not to nested ones.

    A nested helper returning a dict is not an HTTP 200 — `agentic`'s
    per-node `_ping()` legitimately returns ``{"reachable": False}`` and
    the route around it succeeds while reporting which nodes are down.
    Counting that as a success-shaped failure was a false positive, and a
    survey with false positives invites someone to "fix" correct code.
    """
    found: List[ast.ExceptHandler] = []

    def walk(current) -> None:
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda, ast.ClassDef)):
                continue  # a different scope owns this
            if isinstance(child, ast.ExceptHandler):
                found.append(child)
            walk(child)

    walk(node)
    return found


def _silent_swallows(text: str) -> int:
    """`except ...: pass` — a handler that discards the reason.

    A fifth column rather than a fifth gate: this repository already has
    three ratchets, and a fourth watching the same kind of thing would be
    the "declared in three places" problem the meta-check exists to stop.

    Distinct from `success_on_failure`, which is at zero. That one catches
    a handler that *returns a success shape*; this one catches a handler
    that returns nothing at all and simply forgets what went wrong. Both
    end with the operator learning nothing, by different routes.

    Not hypothetical, and not a style preference. `test_soul_identity`
    carried exactly this. The moment it was changed to *record* the
    exception rather than discard it, the message named the cause in a
    single line — "No module named 'system_fsm'" — and four failures
    became zero. In a service the same shape means it degrades quietly and
    nobody is told.

    Counts only broad handlers (bare, `Exception`, `BaseException`). A
    narrow `except FileNotFoundError: pass` is a decision about a named
    condition; a broad one is a decision not to look.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = [n for n in node.body
                if not (isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant))]
        if len(body) != 1 or not isinstance(body[0], ast.Pass):
            continue
        if node.type is None or (isinstance(node.type, ast.Name)
                                 and node.type.id in {"Exception",
                                                      "BaseException"}):
            count += 1
    return count


DETECTORS.update({
    "clients": lambda t: t.count("async with httpx.AsyncClient("),
    "unbounded_bodies": lambda t: t.count("await request.json()"),
    "naive_timestamps": lambda t: t.count("datetime.utcnow()"),
    "success_on_failure": _success_on_failure,
    "silent_swallows": _silent_swallows,
})
# Evidence of adoption, reported so progress is visible rather than only
# the remaining debt. Never part of the ratchet: these should *rise*.
ADOPTION_DETECTORS.update({
    "pooled": lambda t: t.count(POOLED),
    "bounded": lambda t: t.count(BOUNDED),
})
COLUMNS = tuple(DETECTORS)


def survey() -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for path in _service_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        service = str(path.parent.relative_to(REPO))
        counts = {name: fn(text) for name, fn
                  in {**DETECTORS, **ADOPTION_DETECTORS}.items()}
        # A service may ship more than one entry point; sum them rather
        # than letting the last file scanned overwrite the others.
        existing = results.get(service)
        if existing is None:
            results[service] = counts
        else:
            for key, value in counts.items():
                existing[key] += value
    return results


BASELINE = Path(__file__).resolve().parent / "hygiene_baseline.json"


def load_baseline() -> Optional[Dict[str, int]]:
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8"))["totals"]
    except (OSError, KeyError, ValueError):
        return None


def ratchet(totals: Dict[str, int]) -> List[str]:
    """Report any count that has risen above the recorded baseline."""
    baseline = load_baseline()
    if baseline is None:
        return ["no baseline recorded; run with --update-baseline"]
    risen = []
    for column in COLUMNS:
        was = baseline.get(column)
        if was is None:
            risen.append(f"{column}: absent from the baseline")
        elif totals[column] > was:
            risen.append(f"{column}: {was} → {totals[column]} (+{totals[column] - was})")
    return risen


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey HTTP/time hygiene")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-total", type=int, default=None,
                        help="exit non-zero if the total exceeds this")
    parser.add_argument("--gate", action="store_true",
                        help="fail if any count has risen above the baseline")
    parser.add_argument("--update-baseline", action="store_true",
                        help="record the current counts as the new ceiling")
    parser.add_argument("--widen-scope", metavar="REASON",
                        help="record a HIGHER ceiling because the survey now "
                             "looks at more files. Requires a written reason, "
                             "which is stored in the baseline.")
    args = parser.parse_args()

    results = survey()
    totals = {c: sum(r[c] for r in results.values()) for c in COLUMNS}
    adopted = {
        "pooled": sum(r["pooled"] for r in results.values()),
        "bounded": sum(r["bounded"] for r in results.values()),
    }
    grand = sum(totals.values())

    risen = ratchet(totals) if args.gate else []

    if args.update_baseline or args.widen_scope:
        baseline = load_baseline() or {}
        worse = [c for c in COLUMNS if totals[c] > baseline.get(c, totals[c])]
        if worse and not args.widen_scope:
            # The ratchet defends itself. A count that has risen is
            # either a regression or a change of denominator, and those
            # must not be spelled the same way — the second is legitimate
            # and the first is the thing this file exists to catch.
            print("REFUSED: the baseline may only be lowered. Risen: "
                  + ", ".join(worse))
            print("  If the survey now scans MORE files, that is a scope "
                  "change, not a regression: re-run with\n"
                  '  --widen-scope "why the scope grew". The reason is '
                  "stored in the baseline and shows up in review.")
            return 1
        note = ("Ceiling for scripts/security/hygiene_survey.py --gate. "
                "Lower only; see kai-pm/W1_GLOBAL_HYGIENE_SUBPLAN.md.")
        payload = {"note": note, "totals": totals, "grand_total": grand}
        if args.widen_scope:
            payload["scope_widened"] = {
                "reason": args.widen_scope,
                "raised": {c: [baseline.get(c), totals[c]]
                           for c in COLUMNS if totals[c] > baseline.get(c, totals[c])},
            }
        BASELINE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline updated: {totals} (total {grand})")
        if args.widen_scope:
            print(f"  scope widened: {args.widen_scope}")
        return 0

    if args.json:
        print(json.dumps({"services": results, "totals": totals,
                          "adopted": adopted, "grand_total": grand,
                          "baseline": load_baseline(), "risen": risen},
                         indent=2))
        if risen:
            return 1
        return 1 if args.max_total is not None and grand > args.max_total else 0

    print("Repository-wide HTTP and time hygiene survey\n")
    # Header and rows both derive from COLUMNS. Written out by hand, the
    # table printed four columns while the grand total summed five — so
    # every one of the 120 findings in the new column was invisible in the
    # place a reader actually looks. A table that omits the column holding
    # the debt is a denominator problem wearing a different hat.
    _LABELS = {"clients": "clients", "unbounded_bodies": "raw body",
               "naive_timestamps": "utcnow", "success_on_failure": "200-fail",
               "silent_swallows": "swallowed"}
    width = 11
    header = "".join(f"{_LABELS.get(c, c):>{width}}" for c in COLUMNS)
    rule = "  " + "─" * (26 + width * len(COLUMNS))
    print(f"  {'service':<26}{header}")
    print(rule)
    ranked = sorted(results.items(),
                    key=lambda kv: -sum(kv[1][c] for c in COLUMNS))
    for service, counts in ranked:
        if not sum(counts[c] for c in COLUMNS):
            continue
        print(f"  {service:<26}"
              + "".join(f"{counts[c]:>{width}}" for c in COLUMNS))
    print(rule)
    print(f"  {'TOTAL':<26}"
          + "".join(f"{totals[c]:>{width}}" for c in COLUMNS))

    clean = [s for s, c in results.items() if not sum(c[x] for x in COLUMNS)]
    print(f"\n  {len(clean)} of {len(results)} services carry none of these.")
    print(f"  Adoption so far: {adopted['pooled']} pooled call site(s), "
          f"{adopted['bounded']} bounded body read(s).")
    print(f"\n  Grand total: {grand}. This number should only ever fall.")
    print("  Plan: kai-pm/W1_GLOBAL_HYGIENE_SUBPLAN.md")
    print("  This is a survey, not a gate. It closes no findings (Rule 7).")

    baseline = load_baseline()
    if baseline is not None:
        recorded = sum(baseline.get(c, 0) for c in COLUMNS)
        delta = grand - recorded
        arrow = "unchanged" if delta == 0 else (f"+{delta}" if delta > 0 else str(delta))
        print(f"  Baseline: {recorded} ({arrow}).")

    if risen:
        print("\n  GATE FAILED — these counts have risen:")
        for line in risen:
            print(f"    - {line}")
        print("\n  Use common/http_hygiene.py and common/degraded.py rather "
              "than adding\n  new instances. If a rise is genuinely correct, "
              "say why in the commit.")
        return 1
    if args.gate:
        print("\n  GATE PASSED: nothing has got worse.")

    if args.max_total is not None and grand > args.max_total:
        print(f"\n  OVER BUDGET: {grand} > {args.max_total}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
