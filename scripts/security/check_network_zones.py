#!/usr/bin/env python3
"""P0-PR-06 CI gate: validate trust-zone network segmentation.

Checks:
  - No service uses the old flat 'sovereign-net'
  - No static IP addresses (ipv4_address) in service network config
  - Dangerous zones (execution-net, egress-net, sensor-net) cannot reach
    data-net or control-net without an explicit bridge service
  - Every service has an explicit networks assignment
  - Internal zones are marked internal: true
  - egress-net and edge-net are NOT internal (need outbound/inbound)

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

INTERNAL_ZONES = frozenset({
    "data-net",
    "control-net",
    "agent-net",
    "execution-net",
    "sensor-net",
    "observability-net",
})

EXTERNAL_ZONES = frozenset({
    "egress-net",
    "edge-net",
})

BRIDGE_SERVICES = frozenset({
    "tool-gate",
    "dashboard",
    "heartbeat",
    "redis",
    "memu-core",
    "memu-core-introspect",
    "backup-service",
    "tts-service",
    "telegram-bot",
    "perception-telegram",
    "cortex",
    "memory-compressor",
    "ledger-worker",
    "supervisor",
    "agentic",
    "vault-sync",
})

ISOLATED_ZONES = frozenset({"execution-net", "egress-net", "sensor-net"})
PROTECTED_ZONES = frozenset({"data-net", "control-net"})


def check_file(path: Path) -> list[str]:
    violations: list[str] = []
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        violations.append(f"{path}: failed to parse: {exc}")
        return violations

    if data is None:
        violations.append(f"{path}: empty file")
        return violations

    networks = data.get("networks", {})
    services = data.get("services", {})

    if "sovereign-net" in networks:
        violations.append(
            f"{path}: top-level networks still defines 'sovereign-net' — "
            f"must use trust-zone networks"
        )

    for zone in INTERNAL_ZONES:
        if zone in networks:
            net_cfg = networks[zone] or {}
            if not net_cfg.get("internal", False):
                violations.append(
                    f"{path}: network '{zone}' must be internal: true"
                )

    for zone in EXTERNAL_ZONES:
        if zone in networks:
            net_cfg = networks[zone] or {}
            if net_cfg.get("internal", False):
                violations.append(
                    f"{path}: network '{zone}' must NOT be internal"
                )

    for svc_name, svc_cfg in services.items():
        if svc_cfg is None:
            continue

        svc_nets = svc_cfg.get("networks")

        if svc_nets is None:
            # The docstring has always claimed "every service has an
            # explicit networks assignment"; this branch was `pass`, so
            # the rule existed in prose only. A service with no
            # `networks:` key joins Compose's implicit `default` bridge,
            # which is not a trust zone and is not internal — it bypasses
            # the entire segmentation model this gate exists to enforce.
            violations.append(
                f"{path}: service '{svc_name}' has no networks assignment "
                f"— it would join the implicit default bridge, outside "
                f"every trust zone"
            )
        elif isinstance(svc_nets, dict):
            if "sovereign-net" in svc_nets:
                violations.append(
                    f"{path}: service '{svc_name}' still uses sovereign-net"
                )
            for net_name, net_cfg in svc_nets.items():
                if isinstance(net_cfg, dict) and "ipv4_address" in net_cfg:
                    violations.append(
                        f"{path}: service '{svc_name}' has static IP on "
                        f"'{net_name}' — static IPs must be removed"
                    )
        elif isinstance(svc_nets, list):
            if "sovereign-net" in svc_nets:
                violations.append(
                    f"{path}: service '{svc_name}' still uses sovereign-net"
                )

            svc_zone_set = set(svc_nets)
            isolated = svc_zone_set & ISOLATED_ZONES
            protected = svc_zone_set & PROTECTED_ZONES

            if isolated and protected and svc_name not in BRIDGE_SERVICES:
                violations.append(
                    f"{path}: service '{svc_name}' bridges isolated zone(s) "
                    f"{sorted(isolated)} to protected zone(s) "
                    f"{sorted(protected)} — not an approved bridge service"
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
        print(f"FAIL: {len(all_violations)} network zone violation(s) found:\n")
        for v in all_violations:
            print(f"  - {v}")
        return 1

    print("PASS: Network zone segmentation is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
