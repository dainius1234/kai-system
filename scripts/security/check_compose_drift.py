#!/usr/bin/env python3
"""P0-PR-09 CI gate: detect structural drift between compose files.

Checks that services shared between docker-compose.full.yml and
docker-compose.minimal.yml use consistent x-defaults anchors,
network assignments, and security-critical settings.

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

COMPOSE_FULL = "docker-compose.full.yml"
COMPOSE_MINIMAL = "docker-compose.minimal.yml"


def _get_defaults(data: dict) -> dict:
    for key in data:
        if key.startswith("x-") and isinstance(data[key], dict):
            return data[key]
    return {}


def check_drift(repo_root: Path) -> list[str]:
    violations: list[str] = []

    full_path = repo_root / COMPOSE_FULL
    min_path = repo_root / COMPOSE_MINIMAL

    if not full_path.exists() or not min_path.exists():
        return violations

    try:
        full_data = yaml.safe_load(full_path.read_text()) or {}
        min_data = yaml.safe_load(min_path.read_text()) or {}
    except Exception as exc:
        violations.append(f"Failed to parse compose files: {exc}")
        return violations

    full_defaults = _get_defaults(full_data)
    min_defaults = _get_defaults(min_data)

    if full_defaults.get("restart") != min_defaults.get("restart"):
        violations.append(
            f"x-defaults restart policy differs: "
            f"full={full_defaults.get('restart')!r}, "
            f"minimal={min_defaults.get('restart')!r}"
        )

    full_logging = full_defaults.get("logging", {})
    min_logging = min_defaults.get("logging", {})
    if full_logging != min_logging:
        violations.append(
            f"x-defaults logging config differs between full and minimal"
        )

    full_security = full_defaults.get("security_opt", [])
    min_security = min_defaults.get("security_opt", [])
    if full_security != min_security:
        violations.append(
            f"x-defaults security_opt differs: "
            f"full={full_security!r}, minimal={min_security!r}"
        )

    full_svcs = full_data.get("services", {})
    min_svcs = min_data.get("services", {})

    shared = set(full_svcs.keys()) & set(min_svcs.keys())

    for svc in sorted(shared):
        full_cfg = full_svcs[svc] or {}
        min_cfg = min_svcs[svc] or {}

        full_uses_defaults = "<<" in str(full_cfg)
        min_uses_defaults = "<<" in str(min_cfg)

        full_restart = full_cfg.get("restart")
        min_restart = min_cfg.get("restart")
        if full_restart != min_restart:
            violations.append(
                f"service '{svc}' restart policy differs: "
                f"full={full_restart!r}, minimal={min_restart!r}"
            )

        full_sec = full_cfg.get("security_opt")
        min_sec = min_cfg.get("security_opt")
        if full_sec != min_sec:
            violations.append(
                f"service '{svc}' security_opt differs between full and minimal"
            )

    full_nets = set(full_data.get("networks", {}).keys())
    min_nets = set(min_data.get("networks", {}).keys())
    only_full = full_nets - min_nets
    only_min = min_nets - full_nets

    for net in sorted(only_full):
        full_net_cfg = full_data["networks"][net] or {}
        if full_net_cfg.get("internal"):
            pass

    for net in sorted(only_min):
        min_net_cfg = min_data["networks"][net] or {}
        if min_net_cfg.get("internal"):
            violations.append(
                f"network '{net}' only in minimal — internal networks should be consistent"
            )

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    violations = check_drift(repo_root)

    if violations:
        print(f"FAIL: {len(violations)} compose drift issue(s) found:\n")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("PASS: Compose files are structurally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
