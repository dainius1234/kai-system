#!/usr/bin/env python3
"""Auto-append session summary to SESSION_BACKLOG.md from recent git log.

Reads commits since the last SESSION_BACKLOG entry date, groups them,
and appends a new session block. Idempotent — won't duplicate entries
for the same date.

Usage:
    python scripts/auto_session_log.py           # append today's session
    python scripts/auto_session_log.py --dry-run # show what would be appended
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION_LOG = ROOT / "SESSION_BACKLOG.md"


def get_last_session_date() -> str | None:
    """Extract the most recent session date from SESSION_BACKLOG.md."""
    if not SESSION_LOG.exists():
        return None
    text = SESSION_LOG.read_text()
    dates = re.findall(r"^## (\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    return dates[-1] if dates else None


def get_commits_today() -> list[str]:
    """Get today's commit messages (subject only)."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", "--format=%s",
             f"--since={today}T00:00:00", f"--until={today}T23:59:59"],
            capture_output=True, text=True, cwd=ROOT,
        )
        return [line.strip() for line in result.stdout.strip().splitlines()
                if line.strip()]
    except Exception:
        return []


def get_commits_since_date(date_str: str) -> list[str]:
    """Get commits since a given date."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--no-merges", "--format=%s",
             f"--since={date_str}T00:00:00"],
            capture_output=True, text=True, cwd=ROOT,
        )
        return [line.strip() for line in result.stdout.strip().splitlines()
                if line.strip()]
    except Exception:
        return []


def build_session_block(commits: list[str]) -> str:
    """Build a session block from commit messages."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"\n---\n\n## {today} — Auto-logged Session\n"]
    for msg in commits:
        lines.append(f"- {msg}")
    lines.append("")
    return "\n".join(lines)


def already_logged_today() -> bool:
    """Check if today already has a session entry."""
    today = datetime.now().strftime("%Y-%m-%d")
    if not SESSION_LOG.exists():
        return False
    text = SESSION_LOG.read_text()
    return f"## {today}" in text


def append_session(block: str, dry_run: bool = False) -> bool:
    """Append session block before the Open Items section."""
    if not SESSION_LOG.exists():
        print("auto-session: SESSION_BACKLOG.md not found")
        return False

    if dry_run:
        print("auto-session: would append:\n")
        print(block)
        return True

    text = SESSION_LOG.read_text()

    # Insert before "## Open Items" if it exists
    open_items_match = re.search(r"^## Open Items", text, re.MULTILINE)
    if open_items_match:
        insert_pos = open_items_match.start()
        new_text = text[:insert_pos] + block + "\n" + text[insert_pos:]
    else:
        new_text = text + block

    SESSION_LOG.write_text(new_text)
    print(f"auto-session: SESSION_BACKLOG.md updated")
    return True


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if already_logged_today() and not dry_run:
        print("auto-session: today already logged, skipping")
        return 0

    # Get commits since last session date, or just today's
    last_date = get_last_session_date()
    if last_date:
        commits = get_commits_since_date(last_date)
    else:
        commits = get_commits_today()

    if not commits:
        print("auto-session: no commits to log")
        return 0

    block = build_session_block(commits)
    ok = append_session(block, dry_run=dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
