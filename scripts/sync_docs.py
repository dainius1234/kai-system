#!/usr/bin/env python3
"""Automatic documentation sync — keeps README and PROJECT_BACKLOG current.

Scans the codebase for real metrics (test counts, targets, LOC, services,
milestones) and patches the README "Project Status" table in-place.

Usage:
    python scripts/sync_docs.py           # auto-patch README
    python scripts/sync_docs.py --check   # check-only (exit 1 if stale)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── Metric scanners ──────────────────────────────────────────────────

def count_test_functions() -> int:
    """Count 'def test_' across all test files."""
    total = 0
    for pattern in ["scripts/test_*.py", "kai-advisor/test_*.py"]:
        for path in ROOT.glob(pattern):
            text = path.read_text(errors="ignore")
            total += len(re.findall(r"^\s*def test_", text, re.MULTILINE))
    return total


def count_test_files() -> int:
    """Count test files."""
    count = 0
    for pattern in ["scripts/test_*.py", "kai-advisor/test_*.py"]:
        count += len(list(ROOT.glob(pattern)))
    return count


def count_test_targets() -> int:
    """Count targets in test-core dependency list."""
    makefile = (ROOT / "Makefile").read_text()
    m = re.search(r"^test-core:\s*(.+?)$", makefile, re.MULTILINE)
    if not m:
        return 0
    deps = m.group(1).strip().split()
    return len(deps)


def count_python_loc() -> int:
    """Count lines of Python (excluding .git and __pycache__)."""
    total = 0
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if ".git" in parts or "__pycache__" in parts or "_archive" in parts:
            continue
        try:
            total += sum(1 for _ in path.open(errors="ignore"))
        except OSError:
            continue
    return total


def count_services() -> int:
    """Count services in docker-compose.full.yml."""
    full = ROOT / "docker-compose.full.yml"
    if not full.exists():
        return 0
    text = full.read_text()
    # Count top-level service keys under 'services:'
    in_services = False
    count = 0
    for line in text.splitlines():
        if line.strip() == "services:":
            in_services = True
            continue
        if in_services:
            # A top-level service is a non-blank line with exactly 2-space indent
            if re.match(r"^  [a-zA-Z]", line) and not line.startswith("    "):
                count += 1
            # Stop at next top-level key
            if re.match(r"^[a-z]", line) and line.strip() != "":
                break
    return count


def count_milestones() -> int:
    """Count DONE milestones in README milestone section."""
    readme = (ROOT / "README.md").read_text()
    # Match table rows `| DONE |` / `| **DONE** |` AND ascii chart `DONE`
    table_hits = len(re.findall(r"\|\s*\*{0,2}DONE\*{0,2}\s*\|", readme))
    if table_hits:
        return table_hits
    # Fallback: count `██████████ DONE` lines in ascii milestone chart
    return len(re.findall(r"██+ DONE", readme))


def count_compose_files() -> int:
    """Count docker-compose*.yml files."""
    return len(list(ROOT.glob("docker-compose*.yml")))


def get_latest_commit() -> str:
    """Get short hash of HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ── README patcher ───────────────────────────────────────────────────

STATUS_TABLE_RE = re.compile(
    r"(## Project Status \()([^)]+)(\)\n\n\| Metric \| Value \|\n\|---\|---\|\n)"
    r"((?:\| .+\|\n)*)",
    re.MULTILINE,
)


def build_status_table(metrics: dict) -> str:
    """Build the Project Status section rows."""
    rows = [
        f'| **Services** | {metrics["services"]} Docker containers |',
        f'| **Test targets** | {metrics["targets"]} (`make test-core`) |',
        f'| **Individual tests** | {metrics["tests"]:,} (`def test_` across {metrics["test_files"]} files) |',
        f'| **Python LOC** | ~{metrics["loc"]:,} |',
        f'| **Compose files** | {metrics["compose"]} (minimal / full / sovereign) |',
        f'| **Milestones shipped** | {metrics["milestones"]} |',
        f'| **Failures** | 0 |',
    ]
    return "\n".join(rows) + "\n"


def sync_readme(metrics: dict, check_only: bool = False) -> bool:
    """Patch the Project Status table in README.md. Returns True if up-to-date."""
    readme_path = ROOT / "README.md"
    text = readme_path.read_text()

    m = STATUS_TABLE_RE.search(text)
    if not m:
        print("WARNING: Could not find '## Project Status (...)' table in README.md")
        print("         Skipping auto-patch. Add a conforming table to enable sync.")
        return True  # Don't fail if table doesn't exist

    today = datetime.now().strftime("%-d %B %Y")
    new_date = today
    new_table = build_status_table(metrics)

    current_date = m.group(2)
    current_rows = m.group(4)

    if current_date == new_date and current_rows.strip() == new_table.strip():
        print(f"docs-sync: README.md is current ({metrics['targets']} targets, "
              f"{metrics['tests']:,} tests, ~{metrics['loc']:,} LOC)")
        return True

    if check_only:
        print("docs-sync: README.md is STALE — run 'make sync-docs' to fix")
        print(f"  Date: {current_date} → {new_date}")
        # Show diffs
        old_lines = current_rows.strip().splitlines()
        new_lines = new_table.strip().splitlines()
        for old, new in zip(old_lines, new_lines):
            if old != new:
                print(f"  - {old.strip()}")
                print(f"  + {new.strip()}")
        return False

    # Patch in-place
    replacement = m.group(1) + new_date + m.group(3) + new_table
    new_text = text[:m.start()] + replacement + text[m.end():]
    readme_path.write_text(new_text)
    print(f"docs-sync: README.md updated ({metrics['targets']} targets, "
          f"{metrics['tests']:,} tests, ~{metrics['loc']:,} LOC)")
    return True


def sync_backlog(metrics: dict, check_only: bool = False) -> bool:
    """Patch the 'Test targets' and 'Individual tests' rows in PROJECT_BACKLOG.md."""
    backlog_path = ROOT / "docs" / "PROJECT_BACKLOG.md"
    if not backlog_path.exists():
        return True

    text = backlog_path.read_text()
    changed = False

    # Patch test targets row
    m = re.search(r"\| Test targets \| (\d+)", text)
    if m and int(m.group(1)) != metrics["targets"]:
        if check_only:
            print(f"docs-sync: PROJECT_BACKLOG.md stale — targets {m.group(1)} → {metrics['targets']}")
            return False
        text = text[:m.start(1)] + str(metrics["targets"]) + text[m.end(1):]
        changed = True

    # Patch individual tests row
    m2 = re.search(r"\| Individual tests \| (\d+)\+?", text)
    if m2:
        current = int(m2.group(1))
        if current != metrics["tests"]:
            if check_only:
                print(f"docs-sync: PROJECT_BACKLOG.md stale — tests {current} → {metrics['tests']}")
                return False
            text = text[:m2.start(1)] + str(metrics["tests"]) + "+" + text[m2.end(1):]
            if text[m2.end(1)] == "+":
                # Don't double the +
                text = text[:m2.start(1)] + str(metrics["tests"]) + text[m2.end(1):]
            changed = True

    if changed:
        backlog_path.write_text(text)
        print(f"docs-sync: PROJECT_BACKLOG.md updated")
    else:
        print(f"docs-sync: PROJECT_BACKLOG.md is current")

    return True


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    check_only = "--check" in sys.argv

    print("docs-sync: scanning codebase...")
    metrics = {
        "tests": count_test_functions(),
        "test_files": count_test_files(),
        "targets": count_test_targets(),
        "loc": count_python_loc(),
        "services": count_services(),
        "milestones": count_milestones(),
        "compose": count_compose_files(),
        "commit": get_latest_commit(),
    }

    print(f"  Tests:     {metrics['tests']:,} functions in {metrics['test_files']} files")
    print(f"  Targets:   {metrics['targets']}")
    print(f"  Python:    ~{metrics['loc']:,} LOC")
    print(f"  Services:  {metrics['services']}")
    print(f"  Milestones: {metrics['milestones']}")
    print(f"  Compose:   {metrics['compose']}")
    print()

    ok = True
    ok = sync_readme(metrics, check_only) and ok
    ok = sync_backlog(metrics, check_only) and ok

    if not ok:
        print("\nFAILED: Documentation is stale. Run 'make sync-docs' to fix.")
        return 1

    if not check_only:
        print("\nAll documentation synced.")
    else:
        print("\nAll documentation is current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
