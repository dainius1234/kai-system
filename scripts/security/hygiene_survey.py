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

COLUMNS = ("clients", "unbounded_bodies", "naive_timestamps", "success_on_failure")


# Service entry points are not all called `app.py`: `agentic` also ships
# `introspect_app.py`, and scanning only `app.py` missed 2 of its naive
# timestamps entirely. A survey that undercounts is worse than no survey,
# because the number looks authoritative.
ENTRY_POINT_GLOBS = ("*/app.py", "*/*/app.py", "*/*_app.py", "*/*/*_app.py")
_EXCLUDED_DIRS = {"node_modules", ".venv", "venv", "site-packages",
                  "scripts", "tests", "kai-pm"}


def _service_files() -> List[Path]:
    """Every service entry point, excluding tests and vendored code."""
    found = set()
    for pattern in ENTRY_POINT_GLOBS:
        for path in REPO.glob(pattern):
            if any(p in _EXCLUDED_DIRS for p in path.parts):
                continue
            if path.name.startswith("test_"):
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


def survey() -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for path in _service_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        service = str(path.parent.relative_to(REPO))
        counts = {
            "clients": text.count("async with httpx.AsyncClient("),
            "unbounded_bodies": text.count("await request.json()"),
            "naive_timestamps": text.count("datetime.utcnow()"),
            "success_on_failure": _success_on_failure(text),
            # Evidence of adoption, reported so progress is visible rather
            # than only the remaining debt.
            "pooled": text.count(POOLED),
            "bounded": text.count(BOUNDED),
        }
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
    args = parser.parse_args()

    results = survey()
    totals = {c: sum(r[c] for r in results.values()) for c in COLUMNS}
    adopted = {
        "pooled": sum(r["pooled"] for r in results.values()),
        "bounded": sum(r["bounded"] for r in results.values()),
    }
    grand = sum(totals.values())

    risen = ratchet(totals) if args.gate else []

    if args.update_baseline:
        baseline = load_baseline() or {}
        worse = [c for c in COLUMNS if totals[c] > baseline.get(c, totals[c])]
        if worse:
            print("REFUSED: the baseline may only be lowered. Risen: "
                  + ", ".join(worse))
            return 1
        BASELINE.write_text(
            json.dumps({
                "note": "Ceiling for scripts/security/hygiene_survey.py --gate. "
                        "Lower only; see kai-pm/W1_GLOBAL_HYGIENE_SUBPLAN.md.",
                "totals": totals,
                "grand_total": grand,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Baseline updated: {totals} (total {grand})")
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
    print(f"  {'service':<26}{'clients':>9}{'raw body':>10}"
          f"{'utcnow':>8}{'200-fail':>10}")
    print("  " + "─" * 63)
    ranked = sorted(results.items(),
                    key=lambda kv: -sum(kv[1][c] for c in COLUMNS))
    for service, counts in ranked:
        if not sum(counts[c] for c in COLUMNS):
            continue
        print(f"  {service:<26}{counts['clients']:>9}"
              f"{counts['unbounded_bodies']:>10}"
              f"{counts['naive_timestamps']:>8}"
              f"{counts['success_on_failure']:>10}")
    print("  " + "─" * 63)
    print(f"  {'TOTAL':<26}{totals['clients']:>9}"
          f"{totals['unbounded_bodies']:>10}"
          f"{totals['naive_timestamps']:>8}"
          f"{totals['success_on_failure']:>10}")

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
