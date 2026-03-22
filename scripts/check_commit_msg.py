#!/usr/bin/env python3
"""Validate commit message follows conventional commit format.

Expected: <type>: <description>
Types: feat, fix, docs, test, refactor, chore, perf, ci, style, build

Usage (pre-commit hook):
    python scripts/check_commit_msg.py .git/COMMIT_EDITMSG

Usage (standalone):
    echo "feat: add thing" | python scripts/check_commit_msg.py -
"""
import re
import sys

TYPES = {"feat", "fix", "docs", "test", "refactor", "chore", "perf", "ci", "style", "build"}
PATTERN = re.compile(
    r"^(" + "|".join(TYPES) + r")(\(.+\))?: .{3,}",
    re.IGNORECASE,
)


def check(msg: str) -> bool:
    first_line = msg.strip().splitlines()[0] if msg.strip() else ""
    if not first_line:
        return False
    return bool(PATTERN.match(first_line))


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_commit_msg.py <commit-msg-file>")
        return 1

    path = sys.argv[1]
    if path == "-":
        msg = sys.stdin.read()
    else:
        with open(path) as f:
            msg = f.read()

    if check(msg):
        return 0

    print("ERROR: Commit message does not follow conventional format.")
    print("  Expected: <type>: <description>")
    print(f"  Types: {', '.join(sorted(TYPES))}")
    print(f"  Got: {msg.strip().splitlines()[0] if msg.strip() else '(empty)'}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
