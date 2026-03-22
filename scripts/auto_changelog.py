#!/usr/bin/env python3
"""Auto-update CHANGELOG.md [Unreleased] section from git commits.

Reads commits since the last tagged release (or last changelog entry) and
groups them by conventional commit type into Added/Fixed/Changed sections.

Usage:
    python scripts/auto_changelog.py           # patch CHANGELOG.md
    python scripts/auto_changelog.py --check   # check-only (exit 1 if stale)
    python scripts/auto_changelog.py --dry-run # show what would be written
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

TYPE_MAP = {
    "feat": "Added",
    "fix": "Fixed",
    "docs": "Changed",
    "refactor": "Changed",
    "perf": "Changed",
    "test": "Added",
    "chore": "Changed",
    "ci": "Changed",
    "style": "Changed",
    "build": "Changed",
}


def get_last_version_tag() -> str | None:
    """Find the latest semver tag, if any."""
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-v:refname"],
            capture_output=True, text=True, cwd=ROOT,
        )
        for line in result.stdout.strip().splitlines():
            if re.match(r"^v?\d+\.\d+", line):
                return line.strip()
    except Exception:
        pass
    return None


def get_last_changelog_version() -> str | None:
    """Extract the most recent version header from CHANGELOG.md."""
    if not CHANGELOG.exists():
        return None
    text = CHANGELOG.read_text()
    m = re.search(r"^## \[(\d+\.\d+\.\d+)\]", text, re.MULTILINE)
    return m.group(1) if m else None


def get_commits_since(ref: str | None) -> list[str]:
    """Get commit subject lines since ref (or all if None)."""
    cmd = ["git", "log", "--oneline", "--no-merges", "--format=%s"]
    if ref:
        cmd.append(f"{ref}..HEAD")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=ROOT,
        )
        return [line.strip() for line in result.stdout.strip().splitlines()
                if line.strip()]
    except Exception:
        return []


def parse_commits(commits: list[str]) -> dict[str, list[str]]:
    """Group commits by changelog section (Added/Fixed/Changed)."""
    sections: dict[str, list[str]] = {"Added": [], "Fixed": [], "Changed": []}
    pattern = re.compile(r"^(\w+)(\(.+?\))?: (.+)")

    for msg in commits:
        m = pattern.match(msg)
        if m:
            ctype = m.group(1).lower()
            desc = m.group(3).strip()
            section = TYPE_MAP.get(ctype, "Changed")
            sections[section].append(desc)
        else:
            sections["Changed"].append(msg)

    return {k: v for k, v in sections.items() if v}


def build_unreleased_block(sections: dict[str, list[str]]) -> str:
    """Build the [Unreleased] markdown block."""
    lines = ["## [Unreleased]\n"]
    for section in ["Added", "Fixed", "Changed"]:
        if section in sections:
            lines.append(f"\n### {section}")
            for item in sections[section]:
                lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def update_changelog(new_block: str, check_only: bool = False,
                     dry_run: bool = False) -> bool:
    """Replace the [Unreleased] section in CHANGELOG.md."""
    if not CHANGELOG.exists():
        print("auto-changelog: CHANGELOG.md not found")
        return False

    text = CHANGELOG.read_text()

    # Find [Unreleased] section boundaries
    unreleased_re = re.compile(
        r"(## \[Unreleased\]\n)(.*?)(?=\n## \[|\Z)",
        re.DOTALL,
    )
    m = unreleased_re.search(text)
    if not m:
        print("auto-changelog: no [Unreleased] section found")
        return False

    current_block = m.group(0).strip()
    if current_block == new_block.strip():
        print("auto-changelog: CHANGELOG.md is current")
        return True

    if check_only:
        print("auto-changelog: CHANGELOG.md is STALE")
        return False

    if dry_run:
        print("auto-changelog: would write:\n")
        print(new_block)
        return True

    new_text = text[:m.start()] + new_block + "\n" + text[m.end():]
    CHANGELOG.write_text(new_text)
    print("auto-changelog: CHANGELOG.md updated")
    return True


def main() -> int:
    check_only = "--check" in sys.argv
    dry_run = "--dry-run" in sys.argv

    # Find reference point
    tag = get_last_version_tag()
    version = get_last_changelog_version()
    ref = tag or (f"v{version}" if version else None)

    commits = get_commits_since(ref)
    if not commits:
        print("auto-changelog: no new commits since last release")
        return 0

    sections = parse_commits(commits)
    if not sections:
        print("auto-changelog: no conventional commits to log")
        return 0

    new_block = build_unreleased_block(sections)
    ok = update_changelog(new_block, check_only=check_only, dry_run=dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
