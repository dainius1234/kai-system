"""Deployment preflight for the Unified Hunter migration (UH tracker E-03).

Eight services now fail closed on their side-effecting endpoints, and the
dashboard gateway fails closed on all 179 of its protected routes. That is
the right default, but it means a deploy that forgets ``KAI_SERVICE_TOKEN``
gets 503s from things like the database restore, and one that forgets
``KAI_DASHBOARD_TOKEN`` ships a dashboard that answers nothing. This
catches both before they ship rather than after.

Checks, in order of consequence:

  1. ``KAI_SERVICE_TOKEN`` is set, and is not a placeholder or too short
  2. ``KAI_DASHBOARD_TOKEN`` is set, distinct from the service token, and
     names a known role
  3. ``KAI_ALLOW_UNAUTHENTICATED`` is not enabled outside development
  4. every migration flag holds a recognised value
  5. ``KAI_AUTONOMY_ENFORCE`` is only on when grants actually exist
  6. both tokens are wired into every compose profile that needs them

Check 4 is the one worth understanding: enforcing scoped autonomy with no
grants issued denies everything, so turning it on early is a self-inflicted
outage. The preflight refuses it until the authority reports it is ready.

Exit codes: 0 ready, 1 blocking problem, 2 warnings only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = Path(__file__).resolve().parent.parent

PLACEHOLDER_TOKENS = {
    "", "changeme", "change-me", "your-token-here", "token", "secret",
    "test", "dev", "localdev", "password", "REPLACE_ME",
}
MIN_TOKEN_LENGTH = 32

COMPOSE_FILES = [
    "docker-compose.minimal.yml",
    "docker-compose.full.yml",
    "docker-compose.sovereign.yml",
]

# Services whose side-effecting endpoints fail closed without the token.
PROTECTED_SERVICES = {
    "docker-compose.minimal.yml": [
        "agentic", "browser-agent", "notify-service", "monitor-service",
    ],
    "docker-compose.full.yml": ["agentic", "backup-service", "telegram-bot"],
    "docker-compose.sovereign.yml": ["agentic"],
}


class Finding:
    __slots__ = ("level", "check", "message", "fix")

    def __init__(self, level: str, check: str, message: str, fix: str = "") -> None:
        self.level = level        # "block" | "warn" | "ok"
        self.check = check
        self.message = message
        self.fix = fix


def check_service_token() -> list[Finding]:
    findings = []
    token = os.getenv("KAI_SERVICE_TOKEN", "")

    if not token:
        findings.append(Finding(
            "block", "service_token",
            "KAI_SERVICE_TOKEN is not set — eight services will return 503 "
            "on their side-effecting endpoints",
            "make setup-service-token",
        ))
        return findings

    if token.lower() in PLACEHOLDER_TOKENS:
        findings.append(Finding(
            "block", "service_token",
            f"KAI_SERVICE_TOKEN is the placeholder value {token!r}",
            "make setup-service-token",
        ))
        return findings

    if len(token) < MIN_TOKEN_LENGTH:
        findings.append(Finding(
            "block", "service_token",
            f"KAI_SERVICE_TOKEN is {len(token)} chars; minimum is "
            f"{MIN_TOKEN_LENGTH}",
            "openssl rand -hex 32",
        ))
        return findings

    findings.append(Finding(
        "ok", "service_token",
        f"set, {len(token)} chars",
    ))
    return findings


def check_dashboard_credentials() -> list[Finding]:
    """The dashboard gateway fails closed (Wave 1 Track A).

    Without credentials it answers 503 on all 179 protected routes, so a
    deploy that forgets them ships a dashboard that cannot be used at all.
    """
    principals = os.getenv("KAI_DASHBOARD_PRINCIPALS", "").strip()
    token = os.getenv("KAI_DASHBOARD_TOKEN", "").strip()

    if principals:
        try:
            entries = json.loads(principals)
        except json.JSONDecodeError as exc:
            return [Finding(
                "block", "dashboard_creds",
                f"KAI_DASHBOARD_PRINCIPALS is not valid JSON: {exc}",
                "make setup-service-token, or fix the JSON",
            )]
        if not isinstance(entries, list) or not entries:
            return [Finding(
                "block", "dashboard_creds",
                "KAI_DASHBOARD_PRINCIPALS must be a non-empty JSON list",
                "make setup-service-token",
            )]
        weak = [e.get("identity", "?") for e in entries
                if isinstance(e, dict)
                and len(str(e.get("token", ""))) < MIN_TOKEN_LENGTH]
        if weak:
            return [Finding(
                "block", "dashboard_creds",
                f"principal token(s) shorter than {MIN_TOKEN_LENGTH} chars: "
                f"{', '.join(weak)}",
                "openssl rand -hex 32",
            )]
        return [Finding("ok", "dashboard_creds",
                        f"{len(entries)} principal(s) configured")]

    if not token:
        return [Finding(
            "block", "dashboard_creds",
            "KAI_DASHBOARD_TOKEN is not set — the dashboard will return 503 "
            "on every protected route",
            "make setup-service-token",
        )]

    if token.lower() in PLACEHOLDER_TOKENS:
        return [Finding(
            "block", "dashboard_creds",
            f"KAI_DASHBOARD_TOKEN is the placeholder value {token!r}",
            "make setup-service-token",
        )]

    if len(token) < MIN_TOKEN_LENGTH:
        return [Finding(
            "block", "dashboard_creds",
            f"KAI_DASHBOARD_TOKEN is {len(token)} chars; minimum is "
            f"{MIN_TOKEN_LENGTH}",
            "openssl rand -hex 32",
        )]

    findings = [Finding("ok", "dashboard_creds", f"set, {len(token)} chars")]

    # A browser-held credential must not also authorise service-to-service
    # calls: leaking one would then hand over the other.
    if token == os.getenv("KAI_SERVICE_TOKEN", "").strip():
        findings = [Finding(
            "block", "dashboard_creds",
            "KAI_DASHBOARD_TOKEN and KAI_SERVICE_TOKEN are the same value — "
            "a browser-held credential must not also authorise service calls",
            "make setup-service-token FORCE=--force",
        )]

    role = os.getenv("KAI_DASHBOARD_ROLE", "keeper").strip().lower()
    if role not in {"viewer", "operator", "keeper"}:
        findings.append(Finding(
            "block", "dashboard_role",
            f"KAI_DASHBOARD_ROLE={role!r} is not a known role",
            "set viewer, operator or keeper",
        ))
    return findings


def check_unauthenticated_bypass() -> list[Finding]:
    enabled = os.getenv("KAI_ALLOW_UNAUTHENTICATED", "false").lower() in {
        "1", "true", "yes"
    }
    if enabled:
        return [Finding(
            "block", "auth_bypass",
            "KAI_ALLOW_UNAUTHENTICATED=true serves every protected endpoint "
            "without a token — development only",
            "unset KAI_ALLOW_UNAUTHENTICATED",
        )]
    return [Finding("ok", "auth_bypass", "not enabled")]


def check_flag_values() -> list[Finding]:
    findings = []
    valid = {
        "KAI_PERCEPTION_MODE": {"shadow", "active"},
        "KAI_CORTEX_SOURCE": {"poll", "world_state"},
    }
    for name, allowed in valid.items():
        value = os.getenv(name)
        if value is None:
            findings.append(Finding("ok", name, "unset (uses default)"))
            continue
        if value.strip().lower() not in allowed:
            findings.append(Finding(
                "warn", name,
                f"{value!r} is not one of {sorted(allowed)}; the code will "
                f"fall back to the safe default",
                f"set {name} to one of {sorted(allowed)}",
            ))
        else:
            findings.append(Finding("ok", name, value.strip().lower()))

    for name in ("KAI_AUTONOMY_ENFORCE", "KAI_ALLOW_UNAUTHENTICATED"):
        value = os.getenv(name)
        if value is not None and value.strip().lower() not in {
            "1", "true", "yes", "0", "false", "no", ""
        }:
            findings.append(Finding(
                "warn", name,
                f"{value!r} is not a recognised boolean; treated as false",
            ))
    return findings


def check_autonomy_enforcement() -> list[Finding]:
    """Enforcing with no grants issued denies everything."""
    enforcing = os.getenv("KAI_AUTONOMY_ENFORCE", "false").lower() in {
        "1", "true", "yes"
    }
    if not enforcing:
        return [Finding(
            "ok", "autonomy_enforce",
            "advisory mode — scoped decisions recorded, legacy verdict stands",
        )]

    try:
        from common.contracts.base import Principal
        from common.autonomy.authority import AutonomyAuthority
        from common.autonomy.calibration import CalibrationTracker
        from common.autonomy.evidence_service import EvidenceService

        principal = Principal(identity="kai", role="system")
        authority = AutonomyAuthority(
            principal,
            EvidenceService(principal=principal),
            CalibrationTracker(principal=principal),
        )
        active = authority.active_grants()
    except Exception as exc:
        return [Finding(
            "block", "autonomy_enforce",
            f"enforcement is on but the authority could not be inspected: {exc}",
            "set KAI_AUTONOMY_ENFORCE=false until this resolves",
        )]

    if not active:
        return [Finding(
            "block", "autonomy_enforce",
            "KAI_AUTONOMY_ENFORCE=true with no active grants — every scoped "
            "capability will be denied",
            "set KAI_AUTONOMY_ENFORCE=false until migration_report() "
            "reports ready_to_enforce: true",
        )]

    return [Finding(
        "ok", "autonomy_enforce",
        f"enforcing with {len(active)} active grant(s)",
    )]


def check_compose_wiring() -> list[Finding]:
    findings = []
    try:
        import yaml
    except ImportError:
        return [Finding("warn", "compose_wiring",
                        "pyyaml not installed; skipped")]

    for filename in COMPOSE_FILES:
        path = REPO / filename
        if not path.exists():
            findings.append(Finding("warn", "compose_wiring",
                                    f"{filename} not found"))
            continue
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            findings.append(Finding("block", "compose_wiring",
                                    f"{filename} is not valid YAML: {exc}"))
            continue

        services = document.get("services", {}) or {}
        missing = []
        for service in PROTECTED_SERVICES.get(filename, []):
            environment = services.get(service, {}).get("environment") or {}
            if isinstance(environment, list):
                keys = {e.split("=", 1)[0] for e in environment}
            else:
                keys = set(environment)
            if "KAI_SERVICE_TOKEN" not in keys:
                missing.append(service)

        # The dashboard fails closed on its own credential, not the
        # shared service token, so it is checked separately.
        dashboard_env = services.get("dashboard", {}).get("environment") or {}
        if isinstance(dashboard_env, list):
            dashboard_keys = {e.split("=", 1)[0] for e in dashboard_env}
        else:
            dashboard_keys = set(dashboard_env)
        if "dashboard" in services and "KAI_DASHBOARD_TOKEN" not in dashboard_keys:
            findings.append(Finding(
                "block", "compose_wiring",
                f"{filename}: KAI_DASHBOARD_TOKEN missing from the dashboard "
                f"service — every protected route would answer 503",
                "add KAI_DASHBOARD_TOKEN to the dashboard environment",
            ))

        if missing:
            findings.append(Finding(
                "block", "compose_wiring",
                f"{filename}: KAI_SERVICE_TOKEN missing from {missing}",
                "add KAI_SERVICE_TOKEN to those service environments",
            ))
        else:
            findings.append(Finding(
                "ok", "compose_wiring",
                f"{filename}: all protected services wired",
            ))
    return findings


def run_all() -> list[Finding]:
    findings = []
    findings += check_service_token()
    findings += check_dashboard_credentials()
    findings += check_unauthenticated_bypass()
    findings += check_flag_values()
    findings += check_autonomy_enforcement()
    findings += check_compose_wiring()
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true",
                        help="only show problems")
    args = parser.parse_args()

    findings = run_all()
    blocks = [f for f in findings if f.level == "block"]
    warns = [f for f in findings if f.level == "warn"]

    for finding in findings:
        if args.quiet and finding.level == "ok":
            continue
        mark = {"block": "BLOCK", "warn": "WARN ", "ok": "OK   "}[finding.level]
        print(f"  {mark} {finding.check:22} {finding.message}")
        if finding.fix:
            print(f"        └─ fix: {finding.fix}")

    print()
    if blocks:
        print(f"  NOT READY: {len(blocks)} blocking issue(s), "
              f"{len(warns)} warning(s)")
        return 1
    if warns:
        print(f"  READY with {len(warns)} warning(s)")
        return 2
    print("  READY TO DEPLOY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
