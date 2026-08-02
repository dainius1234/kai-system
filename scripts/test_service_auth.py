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
        "browser-agent/app.py": 4,
        "telegram-bot/app.py": 1,
        "monitor-service/app.py": 6,
        "output/notify/app.py": 1,
        "vault-sync/app.py": 1,
        "executor/app.py": 1,
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
