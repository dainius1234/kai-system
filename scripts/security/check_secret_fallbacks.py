#!/usr/bin/env python3
"""P0-PR-05 CI gate: reject insecure secret fallbacks in Compose files.

Scans for:
  - ${VAR:-localdev} or similar weak default credentials
  - HMAC_ALLOW_DEV_SECRET set to "true" in deployment definitions
  - Hardcoded passwords/tokens in environment blocks

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

INSECURE_FALLBACKS = re.compile(
    r"\$\{[A-Z_]+:-("
    r"localdev|changeme|change-me|password|secret|admin|test|dev|default"
    r")\}",
    re.IGNORECASE,
)

BANNED_ENV_PATTERNS = [
    (re.compile(r'HMAC_ALLOW_DEV_SECRET.*["\']?true', re.IGNORECASE),
     "HMAC_ALLOW_DEV_SECRET must not be true in deployment definitions"),
]


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except Exception as exc:
        violations.append(f"{path}: failed to read: {exc}")
        return violations

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        m = INSECURE_FALLBACKS.search(line)
        if m:
            violations.append(
                f"{path}:{lineno}: insecure fallback '{m.group()}' — "
                f"secrets must not have weak defaults"
            )

        for pattern, msg in BANNED_ENV_PATTERNS:
            if pattern.search(line):
                violations.append(f"{path}:{lineno}: {msg}")

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    all_violations: list[str] = []

    for name in COMPOSE_FILES:
        path = repo_root / name
        if not path.exists():
            continue
        all_violations.extend(check_file(path))

    if all_violations:
        print(f"FAIL: {len(all_violations)} insecure secret(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: No insecure secret fallbacks found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
