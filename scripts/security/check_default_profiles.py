#!/usr/bin/env python3
"""P0-PR-03 CI gate: verify default compose profile contains no dangerous services.

Services with consequential capabilities must be behind an explicit profile.
Only the contained core may start with a bare `docker compose up`.

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

COMPOSE_FILES = [
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
]

DANGEROUS_SERVICES = frozenset({
    # Executor and generic subprocess execution
    "executor",
    # Browser Agent, Web Scout, broad external egress
    "browser-agent",
    "email-reader",
    "news-feed",
    "telegram-bot",
    "perception-telegram",
    "tailscale",
    # Vault Sync and arbitrary file ingestion
    "vault-sync",
    # Introspection and graph mutation
    "agentic-introspect",
    "memu-graph",
    "letta-agent",
    "cortex",
    # Broker/live finance mutation
    "broker-bridge",
    "financial-awareness",
    # Camera, screen, clipboard, audio, wake ingestion
    "audio-service",
    "camera-service",
    "wake-service",
    "screen-capture",
    "screen-watcher",
    "clipboard-service",
    "files-service",
    "vision-service",
    # Supervisor-initiated recovery
    "supervisor",
    "verifier",
    "fusion-engine",
    # Host Docker/Git/process watchers
    "docker-watcher",
    "git-watcher",
    "sysmetrics",
    "monitor-service",
})


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
        profiles = svc_def.get("profiles")
        if profiles:
            continue

        if svc_name in DANGEROUS_SERVICES:
            violations.append(
                f"{path}: '{svc_name}' is a dangerous service but has no "
                f"profile — it would start with default `docker compose up`"
            )

    return violations


def check_cross_profile_deps(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        with open(path) as f:
            doc = yaml.safe_load(f)
    except Exception:
        return violations

    services = doc.get("services", {}) or {}
    profiled = {k for k, v in services.items()
                if isinstance(v, dict) and v.get("profiles")}

    for svc_name, svc_def in services.items():
        if not isinstance(svc_def, dict):
            continue
        svc_profiles = svc_def.get("profiles", [])
        deps = svc_def.get("depends_on", {})
        if isinstance(deps, list):
            dep_names = deps
        elif isinstance(deps, dict):
            dep_names = list(deps.keys())
        else:
            continue

        for dep in dep_names:
            if dep not in profiled:
                continue
            dep_profiles = services.get(dep, {}).get("profiles", [])
            if not svc_profiles:
                violations.append(
                    f"{path}: core service '{svc_name}' depends_on profiled "
                    f"service '{dep}' [{dep_profiles}] — would pull it into "
                    f"default startup"
                )
            elif svc_profiles != dep_profiles:
                violations.append(
                    f"{path}: '{svc_name}' [{svc_profiles}] depends_on "
                    f"'{dep}' [{dep_profiles}] — cross-profile leak"
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
        all_violations.extend(check_cross_profile_deps(path))

    if all_violations:
        print(f"FAIL: {len(all_violations)} profile violation(s):\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: No dangerous services in default profile, no cross-profile leaks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
