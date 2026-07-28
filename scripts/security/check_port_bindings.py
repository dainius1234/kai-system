#!/usr/bin/env python3
"""P0-PR-02 CI gate: reject disallowed host-port bindings in Compose files.

Only the 'dashboard' service may publish a port, and only on 127.0.0.1.
All other services must communicate over Docker networks only.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ALLOWED_SERVICE = "dashboard"
ALLOWED_PREFIX = "127.0.0.1:"

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except Exception as exc:
        violations.append(f"{path}: failed to parse: {exc}")
        return violations

    services = doc.get("services", {}) or {}
    for svc_name, svc_def in services.items():
        if not isinstance(svc_def, dict):
            continue
        ports = svc_def.get("ports")
        if not ports:
            continue

        if svc_name == ALLOWED_SERVICE:
            for port in ports:
                binding = str(port)
                if not binding.startswith(ALLOWED_PREFIX):
                    violations.append(
                        f"{path}: {svc_name} port '{binding}' must bind to "
                        f"127.0.0.1 only (expected prefix '{ALLOWED_PREFIX}')"
                    )
        else:
            for port in ports:
                violations.append(
                    f"{path}: {svc_name} has disallowed port binding '{port}' "
                    f"— only '{ALLOWED_SERVICE}' may publish ports"
                )
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
        print(f"FAIL: {len(all_violations)} port-binding violation(s):\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: No disallowed port bindings found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
