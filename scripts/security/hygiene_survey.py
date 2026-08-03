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

Exit codes:
  0  always by default — this is a survey, not a gate
  1  with --max-total N, when the total exceeds N
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent.parent

# Adopted replacements. A service using these is not counted against.
POOLED = "pooled_client("
BOUNDED = "bounded_json("

COLUMNS = ("clients", "unbounded_bodies", "naive_timestamps", "success_on_failure")


def _service_files() -> List[Path]:
    """Every service entry point, excluding tests and vendored code."""
    found = []
    for path in sorted(REPO.glob("*/app.py")) + sorted(REPO.glob("*/*/app.py")):
        parts = path.parts
        if any(p in {"node_modules", ".venv", "venv", "site-packages"} for p in parts):
            continue
        found.append(path)
    return found


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
        for handler in ast.walk(node):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            body = ast.unparse(handler)
            if any(m in body for m in ("raise", "status_code", "degraded_response",
                                       "JSONResponse", "HTTPException", "_sse_error")):
                continue
            if re.search(r"return \{", body):
                count += 1
                break
    return count


def survey() -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    for path in _service_files():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        service = str(path.parent.relative_to(REPO))
        results[service] = {
            "clients": text.count("async with httpx.AsyncClient("),
            "unbounded_bodies": text.count("await request.json()"),
            "naive_timestamps": text.count("datetime.utcnow()"),
            "success_on_failure": _success_on_failure(text),
            # Evidence of adoption, reported so progress is visible rather
            # than only the remaining debt.
            "pooled": text.count(POOLED),
            "bounded": text.count(BOUNDED),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Survey HTTP/time hygiene")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-total", type=int, default=None,
                        help="exit non-zero if the total exceeds this")
    args = parser.parse_args()

    results = survey()
    totals = {c: sum(r[c] for r in results.values()) for c in COLUMNS}
    adopted = {
        "pooled": sum(r["pooled"] for r in results.values()),
        "bounded": sum(r["bounded"] for r in results.values()),
    }
    grand = sum(totals.values())

    if args.json:
        print(json.dumps({"services": results, "totals": totals,
                          "adopted": adopted, "grand_total": grand}, indent=2))
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

    if args.max_total is not None and grand > args.max_total:
        print(f"\n  OVER BUDGET: {grand} > {args.max_total}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
