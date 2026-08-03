#!/usr/bin/env python3
"""P0-PR-08 CI gate: validate restart and recovery containment.

Checks:
  - No service uses restart: always (must use unless-stopped or on-failure)
  - Supervisor has SUPERVISOR_RECOVERY_ENABLED=false
  - x-service-defaults includes logging limits (max-size, max-file)
  - All healthchecks include start_period
  - Executor does not use restart: always

Exit 0 = clean.  Exit 1 = violations found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.security.gate_inputs import (count_services, inspected,  # noqa: E402
                                          require)

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

ALLOWED_RESTART = {"unless-stopped", "on-failure", "no", '"no"', "always:profiled"}


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

    defaults_key = None
    for key in data:
        if key.startswith("x-") and isinstance(data[key], dict):
            defaults_key = key
            break

    if defaults_key:
        defaults = data[defaults_key]
        logging_cfg = defaults.get("logging", {})
        if not logging_cfg:
            violations.append(
                f"{path}: {defaults_key} missing logging limits"
            )
        else:
            opts = logging_cfg.get("options", {})
            if "max-size" not in opts or "max-file" not in opts:
                violations.append(
                    f"{path}: {defaults_key} logging missing max-size or max-file"
                )

    for svc_name, svc_cfg in services.items():
        if svc_cfg is None:
            continue

        # `ALLOWED_RESTART` was declared and never referenced: the
        # docstring promised an allowlist ("must use unless-stopped or
        # on-failure") while the code denied exactly one string, so
        # `restart: nonsense-value` passed. A declared-but-unwired
        # constant is the same defect as the `if ...: pass` dead branch
        # found in `check_compose_drift` — an implementation simpler than
        # it reads. The allowlist it declares is the one enforced now.
        restart = svc_cfg.get("restart")
        if restart is not None and str(restart) not in ALLOWED_RESTART:
            violations.append(
                f"{path}: service '{svc_name}' uses restart: {restart!r} — "
                f"must be one of {sorted(ALLOWED_RESTART)}"
            )

        if svc_name == "supervisor":
            env = svc_cfg.get("environment", {})
            if isinstance(env, list):
                env_dict = {}
                for item in env:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        env_dict[k] = v
                env = env_dict
            recovery_val = str(env.get("SUPERVISOR_RECOVERY_ENABLED", "")).lower()
            if recovery_val != "false":
                violations.append(
                    f"{path}: supervisor must have SUPERVISOR_RECOVERY_ENABLED=false "
                    f"during Phase 0 containment"
                )

        hc = svc_cfg.get("healthcheck")
        if hc and isinstance(hc, dict):
            if "start_period" not in hc:
                violations.append(
                    f"{path}: service '{svc_name}' healthcheck missing start_period"
                )

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    all_violations: list[str] = []

    # A missing compose file is not a clean bill of health: this gate
    # would inspect nothing and print PASS, byte-identical to a real one.
    paths = require(COMPOSE_FILES)
    for path in paths:
        all_violations.extend(check_file(path))

    print(inspected(count_services(paths), "service definitions",
                    f"across {len(paths)} compose files"))

    if all_violations:
        print(f"FAIL: {len(all_violations)} restart/recovery violation(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: Restart and recovery containment is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
