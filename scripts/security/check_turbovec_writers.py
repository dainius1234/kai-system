#!/usr/bin/env python3
"""P0-PR-07 CI gate: enforce single-writer TurboVec containment.

Checks:
  - Any service using VECTOR_STORE=turbovec that is NOT the primary
    writer (memu-core) must set TURBOVEC_READ_ONLY=true
  - Read-only TurboVec services must mount the turbovec volume as :ro
  - The primary writer must NOT set TURBOVEC_READ_ONLY=true

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

PRIMARY_WRITER = "memu-core"


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        violations.append(f"{path}: failed to parse: {exc}")
        return violations

    if data is None:
        return violations

    services = data.get("services", {})

    for svc_name, svc_cfg in services.items():
        if svc_cfg is None:
            continue

        env = svc_cfg.get("environment", {})
        if isinstance(env, list):
            env_dict = {}
            for item in env:
                if "=" in item:
                    k, v = item.split("=", 1)
                    env_dict[k] = v
            env = env_dict

        vector_store = env.get("VECTOR_STORE", "")
        if vector_store != "turbovec":
            continue

        is_read_only = str(env.get("TURBOVEC_READ_ONLY", "false")).lower() == "true"

        if svc_name == PRIMARY_WRITER:
            if is_read_only:
                violations.append(
                    f"{path}: primary writer '{svc_name}' must NOT set "
                    f"TURBOVEC_READ_ONLY=true"
                )
            continue

        if not is_read_only:
            violations.append(
                f"{path}: service '{svc_name}' uses turbovec but does not set "
                f"TURBOVEC_READ_ONLY=true — only '{PRIMARY_WRITER}' may write"
            )

        volumes = svc_cfg.get("volumes", [])
        for vol in volumes:
            if isinstance(vol, str) and "turbovec" in vol:
                if not vol.rstrip().endswith(":ro"):
                    violations.append(
                        f"{path}: service '{svc_name}' mounts turbovec volume "
                        f"without :ro — read-only services must use read-only mounts"
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
        print(f"FAIL: {len(all_violations)} TurboVec writer violation(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: TurboVec single-writer containment is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
