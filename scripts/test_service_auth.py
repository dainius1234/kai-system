"""Service authentication tests — closes UH tracker gap G-03.

The audit found six side-effecting endpoints with no authentication,
including a PostgreSQL restore that overwrites the live database.

The property that matters most is **fail closed**: an unconfigured token
must produce 503, never an open endpoint.  A "temporarily open because
the secret is missing" path is exactly how a destructive endpoint ends
up exposed, so these tests assert it does not exist.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.service_auth import (
    ALLOW_UNAUTH_ENV,
    DASHBOARD_OPERATIONS,
    DASHBOARD_TOKEN_ENV,
    TOKEN_ENV,
    check_token,
    unauthenticated_allowed,
)

REPO = Path(__file__).resolve().parent.parent

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
    """Scoped environment override."""

    def __init__(self, **overrides: str | None) -> None:
        self._overrides = overrides
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for k, v in self._overrides.items():
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


# ═══════════════════════════════════════════════════════════════════
# 1. Fail closed — the core property
# ═══════════════════════════════════════════════════════════════════

def test_unconfigured_token_fails_closed():
    with _Env(**{TOKEN_ENV: None, ALLOW_UNAUTH_ENV: None}):
        ok, status, detail = check_token("Bearer anything", "db_restore")
        check("unconfigured_denies", not ok)
        check("unconfigured_503", status == 503, f"got {status}")
        check("unconfigured_explains", "not" in detail.lower())

        ok, status, _ = check_token(None, "db_restore")
        check("unconfigured_denies_no_header", not ok)
        check("unconfigured_no_header_503", status == 503)


def test_no_implicit_open_path():
    """Empty-string token must not be treated as 'no auth required'."""
    with _Env(**{TOKEN_ENV: "", ALLOW_UNAUTH_ENV: None}):
        ok, status, _ = check_token("Bearer ", "db_restore")
        check("empty_token_denies", not ok)
        check("empty_token_503", status == 503)


def test_explicit_dev_bypass_works():
    with _Env(**{TOKEN_ENV: None, ALLOW_UNAUTH_ENV: "true"}):
        check("dev_bypass_recognised", unauthenticated_allowed())
        ok, status, detail = check_token(None, "dev_test")
        check("dev_bypass_allows", ok)
        check("dev_bypass_200", status == 200)
        check("dev_bypass_labelled", "explicitly allowed" in detail)


def test_dev_bypass_requires_explicit_optin():
    for value in ("false", "0", "no", "", "yes-please"):
        with _Env(**{TOKEN_ENV: None, ALLOW_UNAUTH_ENV: value}):
            expected = value in {"1", "true", "yes"}
            check(f"bypass_value_{value or 'empty'}",
                  unauthenticated_allowed() == expected)


def test_configured_token_ignores_bypass_flag():
    """A configured token is enforced even if the dev flag is set."""
    with _Env(**{TOKEN_ENV: "real-token", ALLOW_UNAUTH_ENV: "true"}):
        ok, status, _ = check_token(None, "op")
        check("configured_still_requires_header", not ok)
        check("configured_401", status == 401)

        ok, _, _ = check_token("Bearer real-token", "op")
        check("configured_accepts_valid", ok)


# ═══════════════════════════════════════════════════════════════════
# 2. Token validation
# ═══════════════════════════════════════════════════════════════════

def test_valid_token_accepted():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        ok, status, detail = check_token("Bearer s3cret", "op")
        check("valid_token_ok", ok)
        check("valid_token_200", status == 200)
        check("valid_token_labelled", detail == "authenticated")


def test_invalid_token_rejected():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        ok, status, _ = check_token("Bearer wrong", "op")
        check("wrong_token_denied", not ok)
        check("wrong_token_403", status == 403, f"got {status}")


def test_missing_header_rejected():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        ok, status, _ = check_token(None, "op")
        check("missing_header_denied", not ok)
        check("missing_header_401", status == 401)


def test_malformed_header_rejected():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        for header in ("s3cret", "Basic s3cret", "Bearer", "Bearer ",
                       "bearer", "Token s3cret"):
            ok, status, _ = check_token(header, "op")
            check(f"malformed_{header.replace(' ', '_') or 'empty'}_denied",
                  not ok, f"header={header!r} status={status}")


def test_bearer_scheme_case_insensitive():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
            ok, _, _ = check_token(f"{scheme} s3cret", "op")
            check(f"scheme_{scheme}_accepted", ok)


def test_token_whitespace_tolerated():
    with _Env(**{TOKEN_ENV: "s3cret", ALLOW_UNAUTH_ENV: None}):
        ok, _, _ = check_token("Bearer  s3cret  ", "op")
        check("surrounding_whitespace_ok", ok)


def test_prefix_token_rejected():
    """A token that is a prefix of the real one must not pass."""
    with _Env(**{TOKEN_ENV: "s3cret-long-value", ALLOW_UNAUTH_ENV: None}):
        for attempt in ("s3cret", "s3cret-long", "s3cret-long-value-extra"):
            ok, _, _ = check_token(f"Bearer {attempt}", "op")
            check(f"prefix_{attempt[:12]}_denied", not ok)


def test_dashboard_key_rejected_where_not_configured():
    """Known-negative for privilege spread: a service holding only the
    shared token (e.g. the executor) must reject the dashboard's key."""
    with _Env(**{TOKEN_ENV: "shared-AAA", DASHBOARD_TOKEN_ENV: None, ALLOW_UNAUTH_ENV: None}):
        ok, status, _ = check_token("Bearer dash-BBB", "executor_run")
        check("dashboard_key_denied_at_unkeyed_service", not ok and status == 403, status)


def test_dashboard_key_accepted_where_configured():
    """Known-positive: browser-agent/monitor/notify accept the dashboard key
    and still accept the shared token; anything else is refused."""
    with _Env(**{TOKEN_ENV: "shared-AAA", DASHBOARD_TOKEN_ENV: "dash-BBB", ALLOW_UNAUTH_ENV: None}):
        ok_d, _, _ = check_token("Bearer dash-BBB", "monitor_rule_create")
        ok_s, _, _ = check_token("Bearer shared-AAA", "monitor_rule_create")
        ok_x, status_x, _ = check_token("Bearer dash-BB", "monitor_rule_create")
        check("dashboard_key_accepted", ok_d)
        check("shared_token_still_accepted", ok_s)
        check("near_miss_rejected", not ok_x and status_x == 403, status_x)


def test_dashboard_key_alone_keeps_fail_closed_semantics():
    """With no token of either kind and no dev bypass, still 503."""
    with _Env(**{TOKEN_ENV: None, DASHBOARD_TOKEN_ENV: None, ALLOW_UNAUTH_ENV: None}):
        ok, status, _ = check_token("Bearer anything", "op")
        check("no_tokens_fail_closed", not ok and status == 503, status)
    with _Env(**{TOKEN_ENV: None, DASHBOARD_TOKEN_ENV: "dash-BBB", ALLOW_UNAUTH_ENV: None}):
        ok, _, _ = check_token("Bearer dash-BBB", "desktop_notify")
        ok2, status2, _ = check_token(None, "desktop_notify")
        check("dashboard_only_service_accepts_key", ok)
        check("dashboard_only_service_requires_header", not ok2 and status2 == 401, status2)


_DASHBOARD_SERVICE_FILES = ("browser-agent/app.py", "monitor-service/app.py", "output/notify/app.py")
_EXCLUDED_FROM_DASHBOARD = {"browser_click", "browser_type", "monitor_rule_update"}


def _declared_operations() -> set:
    """Operation names declared with require_service_auth("...") in the three
    services that accept the dashboard key — read from their source, so the
    expected set does not come from the constant under test."""
    import re
    ops = set()
    for rel in _DASHBOARD_SERVICE_FILES:
        ops |= set(re.findall(r'require_service_auth\("([a-z_]+)"\)', (REPO / rel).read_text()))
    return ops


def test_dashboard_operation_list_matches_the_services():
    declared = _declared_operations()
    check("dashboard_services_declare_17_operations", len(declared) == 17, sorted(declared))
    check("dashboard_ops_all_declared", DASHBOARD_OPERATIONS <= declared,
          sorted(DASHBOARD_OPERATIONS - declared))
    check("dashboard_ops_are_declared_minus_excluded",
          DASHBOARD_OPERATIONS == declared - _EXCLUDED_FROM_DASHBOARD,
          sorted(DASHBOARD_OPERATIONS ^ (declared - _EXCLUDED_FROM_DASHBOARD)))


def test_dashboard_key_only_for_its_14_operations():
    with _Env(**{TOKEN_ENV: "shared-AAA", DASHBOARD_TOKEN_ENV: "dash-BBB", ALLOW_UNAUTH_ENV: None}):
        for op in sorted(DASHBOARD_OPERATIONS):
            ok, _, _ = check_token("Bearer dash-BBB", op)
            check(f"dashboard_key_allows_{op}", ok)
        for op in sorted(_EXCLUDED_FROM_DASHBOARD):
            ok, status, _ = check_token("Bearer dash-BBB", op)
            check(f"dashboard_key_denies_{op}", not ok and status == 403, status)
            ok_s, _, _ = check_token("Bearer shared-AAA", op)
            check(f"shared_key_still_allows_{op}", ok_s)
        check("count_allowed_is_14", len(DASHBOARD_OPERATIONS) == 14, len(DASHBOARD_OPERATIONS))


def test_dashboard_key_denied_at_unkeyed_service_even_for_dashboard_ops():
    """The executor and agentic are not given the dashboard variable, so the
    key fails there even for an operation name on the dashboard list."""
    with _Env(**{TOKEN_ENV: "shared-AAA", DASHBOARD_TOKEN_ENV: None, ALLOW_UNAUTH_ENV: None}):
        for op in ("executor_run", "browser_run"):
            ok, status, _ = check_token("Bearer dash-BBB", op)
            check(f"unkeyed_service_denies_dashboard_key_{op}", not ok and status == 403, status)


def test_identical_keys_fail_closed():
    """If both variables hold the same value, the dashboard key could pass as
    the shared key and skip the operation list — so refuse everything."""
    with _Env(**{TOKEN_ENV: "same-CCC", DASHBOARD_TOKEN_ENV: "same-CCC", ALLOW_UNAUTH_ENV: None}):
        for op in ("browser_click", "browser_run"):
            ok, status, _ = check_token("Bearer same-CCC", op)
            check(f"identical_keys_refuse_{op}", not ok and status == 503, status)


def test_compose_gives_dashboard_key_only_to_dashboard_and_three_services():
    import yaml
    allowed = {"dashboard", "browser-agent", "monitor-service", "notify-service"}
    for cf in ("docker-compose.full.yml", "docker-compose.minimal.yml", "docker-compose.sovereign.yml"):
        services = yaml.safe_load((REPO / cf).read_text()).get("services") or {}
        holders = {n for n, s in services.items()
                   if isinstance(s.get("environment"), dict) and DASHBOARD_TOKEN_ENV in s["environment"]}
        check(f"{cf}_dashboard_key_holders_allowed", holders <= allowed, sorted(holders - allowed))
        check(f"{cf}_executor_agentic_lack_dashboard_key", not holders & {"executor", "agentic"}, sorted(holders))


# ═══════════════════════════════════════════════════════════════════
# 3. Coverage — every audited endpoint is actually protected
# ═══════════════════════════════════════════════════════════════════

# (file, http method, path fragment) → must carry require_service_auth
AUDITED_ENDPOINTS = [
    ("backup-service/app.py", "post", "/restore/postgres"),
    ("backup-service/app.py", "post", "/backup/postgres"),
    ("backup-service/app.py", "post", "/backup/redis"),
    ("browser-agent/app.py", "post", "/click"),
    ("browser-agent/app.py", "post", "/type"),
    ("browser-agent/app.py", "post", "/navigate"),
    ("telegram-bot/app.py", "post", "/alert"),
    ("monitor-service/app.py", "post", "/rules"),
    ("monitor-service/app.py", "put", "/rules/{rule_id}"),
    ("monitor-service/app.py", "delete", "/rules/{rule_id}"),
    ("output/notify/app.py", "post", "/notify"),
    ("agentic/app.py", "post", "/checkpoint/{checkpoint_id}/restore"),
    ("agentic/app.py", "delete", "/checkpoint/{checkpoint_id}"),
    ("agentic/app.py", "post", "/uh/erasure"),
    ("agentic/app.py", "post", "/uh/paper-trade"),
    ("vault-sync/app.py", "post", "/export"),
    ("executor/app.py", "post", "/execute"),
    ("executor/app.py", "post", "/recover"),
    ("browser-agent/app.py", "post", "/scrape"),
    ("browser-agent/app.py", "post", "/screenshot"),
    ("browser-agent/app.py", "post", "/search"),
    ("monitor-service/app.py", "post", "/rules/{rule_id}/check"),
    ("output/notify/app.py", "delete", "/pending"),
    ("vault-sync/app.py", "post", "/ingest"),
]


def _decorator_guards(path: Path, method: str, route: str) -> bool | None:
    """Whether the decorator for (method, route) declares service auth.

    Returns None when the route is not found at all, so a renamed route
    surfaces as a distinct failure rather than a silent pass.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            if not isinstance(func, ast.Attribute) or func.attr != method:
                continue
            if not dec.args:
                continue
            first = dec.args[0]
            if not isinstance(first, ast.Constant) or first.value != route:
                continue
            source = ast.unparse(dec)
            return "require_service_auth" in source
    return None


def test_every_audited_endpoint_protected():
    missing: list[str] = []
    notfound: list[str] = []

    for rel, method, route in AUDITED_ENDPOINTS:
        result = _decorator_guards(REPO / rel, method, route)
        label = f"{rel}:{method.upper()} {route}"
        if result is None:
            notfound.append(label)
        elif not result:
            missing.append(label)

    check("all_audited_endpoints_found", not notfound,
          "; ".join(notfound))
    check("all_audited_endpoints_protected", not missing,
          "; ".join(missing))


def test_protection_count():
    """Sanity: each service carries the expected number of guards."""
    expected = {
        "backup-service/app.py": 7,
        "browser-agent/app.py": 7,
        "telegram-bot/app.py": 1,
        "monitor-service/app.py": 7,
        "output/notify/app.py": 3,
        "vault-sync/app.py": 2,
        "executor/app.py": 2,
        # 2 checkpoint routes + /uh/erasure + /uh/paper-trade
        "agentic/app.py": 4,
    }
    for rel, want in expected.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        got = text.count("Depends(require_service_auth")
        check(f"guard_count_{rel.split('/')[0]}", got == want,
              f"expected {want}, got {got}")


def test_services_import_helper():
    for rel in ["backup-service/app.py", "browser-agent/app.py",
                "telegram-bot/app.py", "monitor-service/app.py",
                "output/notify/app.py", "agentic/app.py",
                "vault-sync/app.py", "executor/app.py"]:
        text = (REPO / rel).read_text(encoding="utf-8")
        check(f"imports_helper_{rel.split('/')[0]}",
              "from common.service_auth import require_service_auth" in text)


def test_no_local_auth_reimplementation():
    """Services must use the shared helper, not hand-rolled checks."""
    offenders: list[str] = []
    for rel in ["backup-service/app.py", "browser-agent/app.py",
                "telegram-bot/app.py", "monitor-service/app.py",
                "output/notify/app.py"]:
        text = (REPO / rel).read_text(encoding="utf-8")
        if "hmac.compare_digest" in text and "service_auth" not in text:
            offenders.append(rel)
    check("no_local_auth_reimplementation", not offenders, "; ".join(offenders))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_unconfigured_token_fails_closed()
    test_no_implicit_open_path()
    test_explicit_dev_bypass_works()
    test_dev_bypass_requires_explicit_optin()
    test_configured_token_ignores_bypass_flag()
    test_valid_token_accepted()
    test_invalid_token_rejected()
    test_missing_header_rejected()
    test_malformed_header_rejected()
    test_bearer_scheme_case_insensitive()
    test_token_whitespace_tolerated()
    test_prefix_token_rejected()
    test_dashboard_key_rejected_where_not_configured()
    test_dashboard_key_accepted_where_configured()
    test_dashboard_key_alone_keeps_fail_closed_semantics()
    test_dashboard_operation_list_matches_the_services()
    test_dashboard_key_only_for_its_14_operations()
    test_dashboard_key_denied_at_unkeyed_service_even_for_dashboard_ops()
    test_identical_keys_fail_closed()
    test_compose_gives_dashboard_key_only_to_dashboard_and_three_services()
    test_every_audited_endpoint_protected()
    test_protection_count()
    test_services_import_helper()
    test_no_local_auth_reimplementation()

    print(f"\n{'='*60}")
    print(f"Service Auth Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
