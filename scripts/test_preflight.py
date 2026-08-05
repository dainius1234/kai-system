"""Deployment preflight tests (UH tracker E-03).

The preflight exists to catch three deployment mistakes that are cheap to
make and expensive to discover in production:

  - shipping without ``KAI_SERVICE_TOKEN``, so eight services 503;
  - shipping with the development auth bypass still on;
  - turning on scoped-autonomy enforcement before any grant exists, which
    denies every gated capability at once.

These tests check that it actually catches each, because a preflight that
passes everything is worse than none — it manufactures confidence.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.preflight_deploy import (
    MIN_TOKEN_LENGTH,
    PROTECTED_SERVICES,
    check_autonomy_enforcement,
    check_compose_wiring,
    check_flag_values,
    check_service_token,
    check_dashboard_credentials,
    check_unauthenticated_bypass,
    run_all,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


class _Env:
    def __init__(self, **overrides):
        self._o = overrides
        self._saved = {}

    def __enter__(self):
        for k, v in self._o.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


GOOD_TOKEN = "a" * 64
DASHBOARD_TOKEN = "d" * 64  # distinct from GOOD_TOKEN by construction


def _levels(findings, check_name):
    return [f.level for f in findings if f.check == check_name]


# ═══════════════════════════════════════════════════════════════════
# 1. Service token
# ═══════════════════════════════════════════════════════════════════

def test_missing_token_blocks():
    with _Env(KAI_SERVICE_TOKEN=None):
        findings = check_service_token()
        check("missing_token_blocks", findings[0].level == "block")
        check("missing_token_explains_503", "503" in findings[0].message)
        check("missing_token_offers_fix", bool(findings[0].fix))


def test_placeholder_token_blocks():
    for placeholder in ("changeme", "secret", "test", "REPLACE_ME"):
        with _Env(KAI_SERVICE_TOKEN=placeholder):
            findings = check_service_token()
            check(f"placeholder_{placeholder}_blocks",
                  findings[0].level == "block", placeholder)


def test_short_token_blocks():
    with _Env(KAI_SERVICE_TOKEN="a" * (MIN_TOKEN_LENGTH - 1)):
        findings = check_service_token()
        check("short_token_blocks", findings[0].level == "block")
        check("short_token_states_minimum",
              str(MIN_TOKEN_LENGTH) in findings[0].message)


def test_good_token_passes():
    with _Env(KAI_SERVICE_TOKEN=GOOD_TOKEN):
        findings = check_service_token()
        check("good_token_ok", findings[0].level == "ok")


# ═══════════════════════════════════════════════════════════════════
# 1b. Dashboard credentials (Wave 1 Track A)
# ═══════════════════════════════════════════════════════════════════

def test_missing_dashboard_token_blocks():
    with _Env(KAI_DASHBOARD_TOKEN=None, KAI_DASHBOARD_PRINCIPALS=None):
        findings = check_dashboard_credentials()
        check("missing_dashboard_token_blocks", findings[0].level == "block")
        check("missing_dashboard_token_explains_503", "503" in findings[0].message)
        check("missing_dashboard_token_offers_fix", bool(findings[0].fix))


def test_short_dashboard_token_blocks():
    with _Env(KAI_DASHBOARD_TOKEN="a" * (MIN_TOKEN_LENGTH - 1),
              KAI_DASHBOARD_PRINCIPALS=None):
        findings = check_dashboard_credentials()
        check("short_dashboard_token_blocks", findings[0].level == "block")


def test_placeholder_dashboard_token_blocks():
    for placeholder in ("changeme", "secret", "test"):
        with _Env(KAI_DASHBOARD_TOKEN=placeholder, KAI_DASHBOARD_PRINCIPALS=None):
            findings = check_dashboard_credentials()
            check(f"placeholder_dashboard_{placeholder}_blocks",
                  findings[0].level == "block", placeholder)


def test_dashboard_token_must_differ_from_service_token():
    """A browser-held credential must not also authorise service calls."""
    with _Env(KAI_DASHBOARD_TOKEN=GOOD_TOKEN, KAI_SERVICE_TOKEN=GOOD_TOKEN,
              KAI_DASHBOARD_PRINCIPALS=None):
        findings = check_dashboard_credentials()
        check("shared_token_blocks", findings[0].level == "block")
        check("shared_token_explains_why",
              "service calls" in findings[0].message, findings[0].message)


def test_distinct_dashboard_token_passes():
    with _Env(KAI_DASHBOARD_TOKEN="b" * 64, KAI_SERVICE_TOKEN=GOOD_TOKEN,
              KAI_DASHBOARD_PRINCIPALS=None, KAI_DASHBOARD_ROLE=None):
        findings = check_dashboard_credentials()
        check("distinct_dashboard_token_ok", findings[0].level == "ok",
              findings[0].message)


def test_unknown_dashboard_role_blocks():
    with _Env(KAI_DASHBOARD_TOKEN="b" * 64, KAI_SERVICE_TOKEN=GOOD_TOKEN,
              KAI_DASHBOARD_PRINCIPALS=None, KAI_DASHBOARD_ROLE="superuser"):
        findings = check_dashboard_credentials()
        check("unknown_role_blocks",
              any(f.level == "block" for f in findings),
              "; ".join(f.message for f in findings))


def test_malformed_principals_block():
    for bad, label in [("{not json", "invalid JSON"), ("[]", "empty list"),
                       ('{"identity": "a"}', "not a list")]:
        with _Env(KAI_DASHBOARD_PRINCIPALS=bad, KAI_DASHBOARD_TOKEN=None):
            findings = check_dashboard_credentials()
            check(f"principals_{label}_blocks", findings[0].level == "block", label)


def test_weak_principal_token_blocks():
    weak = '[{"identity": "a", "role": "keeper", "token": "short"}]'
    with _Env(KAI_DASHBOARD_PRINCIPALS=weak, KAI_DASHBOARD_TOKEN=None):
        findings = check_dashboard_credentials()
        check("weak_principal_token_blocks", findings[0].level == "block")


def test_valid_principals_pass():
    good = '[{"identity": "a", "role": "keeper", "token": "%s"}]' % ("c" * 64)
    with _Env(KAI_DASHBOARD_PRINCIPALS=good, KAI_DASHBOARD_TOKEN=None):
        findings = check_dashboard_credentials()
        check("valid_principals_ok", findings[0].level == "ok", findings[0].message)


def test_compose_wires_dashboard_credentials():
    """Every compose profile must pass the dashboard its credential."""
    import yaml
    from scripts.preflight_deploy import COMPOSE_FILES, REPO
    for filename in COMPOSE_FILES:
        path = REPO / filename
        if not path.exists():
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        services = document.get("services", {}) or {}
        if "dashboard" not in services:
            continue
        env = services["dashboard"].get("environment") or {}
        keys = ({e.split("=", 1)[0] for e in env} if isinstance(env, list)
                else set(env))
        check(f"{filename}_wires_dashboard_token",
              "KAI_DASHBOARD_TOKEN" in keys, filename)

# ═══════════════════════════════════════════════════════════════════
# 2. Auth bypass
# ═══════════════════════════════════════════════════════════════════


def test_bypass_enabled_blocks():
    for value in ("true", "1", "yes"):
        with _Env(KAI_ALLOW_UNAUTHENTICATED=value):
            findings = check_unauthenticated_bypass()
            check(f"bypass_{value}_blocks", findings[0].level == "block")


def test_bypass_disabled_passes():
    for value in ("false", "0", "no", None):
        with _Env(KAI_ALLOW_UNAUTHENTICATED=value):
            findings = check_unauthenticated_bypass()
            check(f"bypass_{value}_ok", findings[0].level == "ok")


# ═══════════════════════════════════════════════════════════════════
# 3. Autonomy enforcement — the dangerous one
# ═══════════════════════════════════════════════════════════════════

def test_enforcement_without_grants_blocks():
    """Enforcing with no grants denies every gated capability."""
    with _Env(KAI_AUTONOMY_ENFORCE="true"):
        findings = check_autonomy_enforcement()
        check("enforce_no_grants_blocks", findings[0].level == "block")
        check("enforce_explains_consequence",
              "denied" in findings[0].message)
        check("enforce_names_readiness_signal",
              "ready_to_enforce" in findings[0].fix)


def test_advisory_mode_passes():
    for value in ("false", None):
        with _Env(KAI_AUTONOMY_ENFORCE=value):
            findings = check_autonomy_enforcement()
            check(f"advisory_{value}_ok", findings[0].level == "ok")
            check(f"advisory_{value}_described",
                  "advisory" in findings[0].message)


# ═══════════════════════════════════════════════════════════════════
# 4. Flag values
# ═══════════════════════════════════════════════════════════════════

def test_valid_flag_values_pass():
    with _Env(KAI_PERCEPTION_MODE="active", KAI_CORTEX_SOURCE="world_state"):
        findings = check_flag_values()
        check("valid_perception_ok",
              "ok" in _levels(findings, "KAI_PERCEPTION_MODE"))
        check("valid_cortex_ok",
              "ok" in _levels(findings, "KAI_CORTEX_SOURCE"))


def test_invalid_flag_values_warn():
    with _Env(KAI_PERCEPTION_MODE="nonsense", KAI_CORTEX_SOURCE="bogus"):
        findings = check_flag_values()
        check("invalid_perception_warns",
              "warn" in _levels(findings, "KAI_PERCEPTION_MODE"))
        check("invalid_cortex_warns",
              "warn" in _levels(findings, "KAI_CORTEX_SOURCE"))


def test_invalid_flag_is_warn_not_block():
    """An unrecognised value falls back safely, so it must not block."""
    with _Env(KAI_PERCEPTION_MODE="nonsense", KAI_SERVICE_TOKEN=GOOD_TOKEN,
              KAI_ALLOW_UNAUTHENTICATED=None, KAI_AUTONOMY_ENFORCE=None,
              KAI_DASHBOARD_TOKEN=DASHBOARD_TOKEN,
              KAI_DASHBOARD_PRINCIPALS=None, KAI_DASHBOARD_ROLE=None):
        findings = run_all()
        blocks = [f for f in findings if f.level == "block"]
        check("bad_flag_does_not_block", not blocks, str([f.check for f in blocks]))


# ═══════════════════════════════════════════════════════════════════
# 5. Compose wiring
# ═══════════════════════════════════════════════════════════════════

def test_compose_wiring_passes():
    findings = check_compose_wiring()
    blocks = [f for f in findings if f.level == "block"]
    check("compose_wiring_ok", not blocks,
          "; ".join(f.message for f in blocks))


def test_every_protected_service_listed():
    total = sum(len(v) for v in PROTECTED_SERVICES.values())
    check("protected_services_listed", total == 8, str(total))


# ═══════════════════════════════════════════════════════════════════
# 6. End to end
# ═══════════════════════════════════════════════════════════════════

def test_clean_config_is_ready():
    with _Env(KAI_SERVICE_TOKEN=GOOD_TOKEN, KAI_ALLOW_UNAUTHENTICATED=None,
              KAI_AUTONOMY_ENFORCE=None, KAI_PERCEPTION_MODE=None,
              KAI_CORTEX_SOURCE=None, KAI_DASHBOARD_TOKEN=DASHBOARD_TOKEN,
              KAI_DASHBOARD_PRINCIPALS=None, KAI_DASHBOARD_ROLE=None):
        findings = run_all()
        blocks = [f for f in findings if f.level == "block"]
        warns = [f for f in findings if f.level == "warn"]
        check("clean_config_no_blocks", not blocks,
              str([f.check for f in blocks]))
        check("clean_config_no_warns", not warns,
              str([f.check for f in warns]))


def test_worst_config_blocks_everything():
    with _Env(KAI_SERVICE_TOKEN=None, KAI_ALLOW_UNAUTHENTICATED="true",
              KAI_AUTONOMY_ENFORCE="true", KAI_DASHBOARD_TOKEN=None,
              KAI_DASHBOARD_PRINCIPALS=None):
        findings = run_all()
        blocked = {f.check for f in findings if f.level == "block"}
        check("worst_blocks_token", "service_token" in blocked)
        check("worst_blocks_dashboard_creds", "dashboard_creds" in blocked)
        check("worst_blocks_bypass", "auth_bypass" in blocked)
        check("worst_blocks_enforcement", "autonomy_enforce" in blocked)


def test_preflight_can_actually_fail():
    """A preflight that passes everything manufactures false confidence."""
    with _Env(KAI_SERVICE_TOKEN=None):
        findings = run_all()
        check("preflight_can_fail",
              any(f.level == "block" for f in findings))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_missing_token_blocks()
    test_placeholder_token_blocks()
    test_short_token_blocks()
    test_good_token_passes()
    test_missing_dashboard_token_blocks()
    test_short_dashboard_token_blocks()
    test_placeholder_dashboard_token_blocks()
    test_dashboard_token_must_differ_from_service_token()
    test_distinct_dashboard_token_passes()
    test_unknown_dashboard_role_blocks()
    test_malformed_principals_block()
    test_weak_principal_token_blocks()
    test_valid_principals_pass()
    test_compose_wires_dashboard_credentials()
    test_bypass_enabled_blocks()
    test_bypass_disabled_passes()
    test_enforcement_without_grants_blocks()
    test_advisory_mode_passes()
    test_valid_flag_values_pass()
    test_invalid_flag_values_warn()
    test_invalid_flag_is_warn_not_block()
    test_compose_wiring_passes()
    test_every_protected_service_listed()
    test_clean_config_is_ready()
    test_worst_config_blocks_everything()
    test_preflight_can_actually_fail()

    print(f"\n{'='*60}")
    print(f"Deployment Preflight Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
