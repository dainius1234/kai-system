"""Dashboard inbound authentication tests — Wave 1 Track A.

Covers `KAI-DASH-001`, `011`, `012`, `018`.

The point of these tests is the refusals. An auth module that accepts
valid credentials is easy; one that fails closed on a misconfiguration,
refuses an under-privileged role, and cannot be bypassed by a malformed
header is the thing worth proving. So most of what follows asserts that
something is *denied*.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.dashboard_auth import (
    ALLOW_UNAUTH_ENV,
    IDENTITY_ENV,
    PRINCIPALS_ENV,
    ROLE_ENV,
    ROLE_SCOPES,
    TOKEN_ENV,
    ConfigError,
    DashboardPrincipal,
    Role,
    Scope,
    authenticate,
    load_principals,
    require_dashboard_auth,
    unauthenticated_allowed,
)

passed = 0
failed = 0

ALL_ENV = (PRINCIPALS_ENV, TOKEN_ENV, IDENTITY_ENV, ROLE_ENV, ALLOW_UNAUTH_ENV)


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
    """Set dashboard credential environment, restoring it afterwards.

    Clears every credential variable on entry, so a test cannot
    accidentally pass because of state left by an earlier one.
    """

    def __init__(self, **values: str) -> None:
        self.values = values
        self._saved: dict = {}

    def __enter__(self):
        for key in ALL_ENV:
            self._saved[key] = os.environ.get(key)
            os.environ.pop(key, None)
        for key, value in self.values.items():
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


GOOD = "s3cret-token-value"
BEARER = f"Bearer {GOOD}"


# ── Fail-closed behaviour ────────────────────────────────────────────

def test_unconfigured_fails_closed():
    with _Env():
        principal, status, detail = authenticate(BEARER, Scope.READ_OPERATIONAL, "op")
    check("no credentials configured returns 503", status == 503, str(status))
    check("no principal is issued when unconfigured", principal is None)
    check("503 explains it fails closed by design", "fails closed" in detail, detail)


def test_unconfigured_refuses_even_with_no_header():
    with _Env():
        _, status, _ = authenticate(None, Scope.READ_OPERATIONAL, "op")
    check("unconfigured refuses an anonymous request too", status == 503, str(status))


def test_escape_hatch_is_explicit_and_single():
    with _Env(**{ALLOW_UNAUTH_ENV: "true"}):
        principal, status, detail = authenticate(
            None, Scope.READ_OPERATIONAL, "op")
    check("explicit escape hatch allows unauthenticated", status == 200, detail)
    check("escape hatch issues a named local principal",
          principal is not None and principal.identity == "local-development")
    check("escape hatch is labelled in the detail",
          "explicitly allowed" in detail, detail)


def test_escape_hatch_is_still_subject_to_scopes():
    """The bypass is for authentication, not for authorisation.

    Returning early on this path would have made setting one development
    flag grant more authority than any configured credential ever could.
    """
    with _Env(**{ALLOW_UNAUTH_ENV: "true"}):
        _, external, d1 = authenticate(None, Scope.WRITE_EXTERNAL, "browser")
        _, identity, d2 = authenticate(None, Scope.WRITE_IDENTITY, "soul")
        _, allowed, _ = authenticate(None, Scope.READ_SENSITIVE, "memory")
    check("unauthenticated cannot drive external action", external == 403, d1)
    check("unauthenticated cannot rewrite identity state", identity == 403, d2)
    check("unauthenticated retains operator-level reads", allowed == 200)


def test_empty_token_is_not_a_bypass():
    """An empty configured token must not mean 'allow anything'."""
    with _Env(**{TOKEN_ENV: "   "}):
        _, status, _ = authenticate(BEARER, Scope.READ_OPERATIONAL, "op")
    check("whitespace-only token is treated as unconfigured, not as a match",
          status == 503, str(status))


def test_misconfiguration_fails_closed_not_open():
    with _Env(**{PRINCIPALS_ENV: "{not json"}):
        principal, status, detail = authenticate(BEARER, Scope.READ_OPERATIONAL, "op")
    check("malformed principal config returns 503", status == 503, str(status))
    check("malformed config issues no principal", principal is None)
    check("503 names the misconfiguration", "misconfigured" in detail, detail)


# ── Credential validation ────────────────────────────────────────────

def test_valid_token_authenticates():
    with _Env(**{TOKEN_ENV: GOOD, IDENTITY_ENV: "dainius", ROLE_ENV: "keeper"}):
        principal, status, detail = authenticate(BEARER, Scope.WRITE_IDENTITY, "op")
    check("valid token authenticates", status == 200, detail)
    check("principal carries the configured identity",
          principal is not None and principal.identity == "dainius")
    check("principal carries the configured role",
          principal is not None and principal.role is Role.KEEPER)


def test_missing_header_is_401():
    with _Env(**{TOKEN_ENV: GOOD}):
        _, status, detail = authenticate(None, Scope.READ_OPERATIONAL, "op")
    check("missing Authorization header is 401", status == 401, str(status))
    check("401 says what is missing", "Authorization" in detail, detail)


def test_wrong_scheme_is_401():
    with _Env(**{TOKEN_ENV: GOOD}):
        _, basic, _ = authenticate(f"Basic {GOOD}", Scope.READ_OPERATIONAL, "op")
        _, bare, _ = authenticate(GOOD, Scope.READ_OPERATIONAL, "op")
        _, empty, _ = authenticate("Bearer ", Scope.READ_OPERATIONAL, "op")
    check("Basic scheme is refused", basic == 401, str(basic))
    check("a bare token with no scheme is refused", bare == 401, str(bare))
    check("Bearer with no credential is refused", empty == 401, str(empty))


def test_wrong_token_is_refused():
    with _Env(**{TOKEN_ENV: GOOD}):
        principal, status, _ = authenticate("Bearer wrong", Scope.READ_OPERATIONAL, "op")
    check("wrong token is refused", status == 401, str(status))
    check("wrong token issues no principal", principal is None)


def test_token_prefix_is_not_accepted():
    """Guards against a truncating or prefix comparison."""
    with _Env(**{TOKEN_ENV: GOOD}):
        _, short, _ = authenticate(f"Bearer {GOOD[:-1]}", Scope.READ_OPERATIONAL, "op")
        _, long, _ = authenticate(f"Bearer {GOOD}x", Scope.READ_OPERATIONAL, "op")
    check("a token one character short is refused", short == 401, str(short))
    check("a token one character long is refused", long == 401, str(long))


def test_scheme_is_case_insensitive_but_token_is_not():
    with _Env(**{TOKEN_ENV: GOOD}):
        _, lower, _ = authenticate(f"bearer {GOOD}", Scope.READ_OPERATIONAL, "op")
        _, upper_token, _ = authenticate(
            f"Bearer {GOOD.upper()}", Scope.READ_OPERATIONAL, "op")
    check("'bearer' in lower case is accepted", lower == 200, str(lower))
    check("the token itself is case sensitive", upper_token == 401, str(upper_token))


# ── Multi-principal configuration ────────────────────────────────────

def _principals_json(*entries) -> str:
    return json.dumps([
        {"identity": i, "role": r, "token": t} for i, r, t in entries
    ])


def test_multiple_principals_resolve_separately():
    cfg = _principals_json(("alice", "keeper", "tok-a"), ("bob", "viewer", "tok-b"))
    with _Env(**{PRINCIPALS_ENV: cfg}):
        alice, a_status, _ = authenticate("Bearer tok-a", Scope.READ_OPERATIONAL, "op")
        bob, b_status, _ = authenticate("Bearer tok-b", Scope.READ_OPERATIONAL, "op")
    check("first principal resolves", a_status == 200 and alice.identity == "alice")
    check("second principal resolves", b_status == 200 and bob.identity == "bob")
    check("principals keep distinct roles",
          alice.role is Role.KEEPER and bob.role is Role.VIEWER)


def test_duplicate_tokens_are_a_configuration_error():
    cfg = _principals_json(("alice", "keeper", "same"), ("bob", "viewer", "same"))
    with _Env(**{PRINCIPALS_ENV: cfg}):
        _, status, detail = authenticate("Bearer same", Scope.READ_OPERATIONAL, "op")
    check("a token shared by two identities is refused, not silently resolved",
          status == 503, f"{status}: {detail}")


def test_principal_entries_require_identity_and_token():
    for cfg, label in [
        (json.dumps([{"role": "keeper", "token": "t"}]), "no identity"),
        (json.dumps([{"identity": "a", "role": "keeper"}]), "no token"),
        (json.dumps([{"identity": "a", "role": "wizard", "token": "t"}]), "unknown role"),
        (json.dumps([]), "empty list"),
        (json.dumps({"identity": "a"}), "not a list"),
    ]:
        with _Env(**{PRINCIPALS_ENV: cfg}):
            _, status, _ = authenticate("Bearer t", Scope.READ_OPERATIONAL, "op")
        check(f"principal config with {label} fails closed", status == 503, str(status))


def test_unknown_role_in_single_token_form_fails_closed():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "superuser"}):
        _, status, detail = authenticate(BEARER, Scope.READ_OPERATIONAL, "op")
    check("an unknown role fails closed rather than defaulting",
          status == 503, f"{status}: {detail}")


def test_the_default_role_is_not_keeper():
    """The role model is decorative if the default hands out the top role.

    A single leaked token defaulting to `keeper` would carry authority to
    rewrite SOUL.md and drive the browser agent. Granting that must be a
    conscious act, so it has to be asked for by name.
    """
    with _Env(**{TOKEN_ENV: GOOD}):
        pairs = load_principals()
    check("single-token form yields exactly one principal", len(pairs) == 1)
    check("the default role is NOT keeper",
          pairs and pairs[0][1].role is not Role.KEEPER,
          pairs[0][1].role.value if pairs else "none")
    check("the default role is operator",
          pairs and pairs[0][1].role is Role.OPERATOR,
          pairs[0][1].role.value if pairs else "none")
    check("the default cannot rewrite identity state",
          pairs and not pairs[0][1].may(Scope.WRITE_IDENTITY))
    check("the default cannot drive external action",
          pairs and not pairs[0][1].may(Scope.WRITE_EXTERNAL))
    check("single-token form has a non-empty identity",
          pairs and bool(pairs[0][1].identity))


def test_keeper_must_be_asked_for_by_name():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "keeper"}):
        pairs = load_principals()
    check("keeper is granted when explicitly configured",
          pairs and pairs[0][1].role is Role.KEEPER)


def test_the_escape_hatch_does_not_confer_keeper():
    """Development convenience must not also be privilege escalation."""
    with _Env(**{ALLOW_UNAUTH_ENV: "true"}):
        principal, status, _ = authenticate(None, Scope.READ_OPERATIONAL, "op")
        _, identity_status, _ = authenticate(None, Scope.WRITE_IDENTITY, "soul")
    check("the escape hatch still authenticates", status == 200)
    check("the escape hatch does not confer keeper",
          principal is not None and principal.role is not Role.KEEPER,
          principal.role.value if principal else "none")
    check("an unauthenticated caller still cannot rewrite identity",
          identity_status == 403, str(identity_status))


# ── Authorisation — least privilege (KAI-DASH-018) ───────────────────

def test_viewer_cannot_write():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "viewer"}):
        for scope in (Scope.WRITE_ROUTINE, Scope.WRITE_IDENTITY,
                      Scope.WRITE_EXTERNAL, Scope.READ_SENSITIVE):
            principal, status, detail = authenticate(BEARER, scope, "op")
            check(f"viewer is denied {scope.value}", status == 403, f"{status}: {detail}")
            check(f"viewer gets no principal for {scope.value}", principal is None)


def test_viewer_can_read_operational():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "viewer"}):
        _, status, detail = authenticate(BEARER, Scope.READ_OPERATIONAL, "op")
    check("viewer may read operational status", status == 200, detail)


def test_operator_cannot_rewrite_identity_or_act_externally():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "operator"}):
        _, identity, d1 = authenticate(BEARER, Scope.WRITE_IDENTITY, "soul")
        _, external, d2 = authenticate(BEARER, Scope.WRITE_EXTERNAL, "browser")
        _, sensitive, _ = authenticate(BEARER, Scope.READ_SENSITIVE, "memory")
    check("operator cannot rewrite identity state", identity == 403, f"{identity}: {d1}")
    check("operator cannot drive external action", external == 403, f"{external}: {d2}")
    check("operator may read sensitive data", sensitive == 200, str(sensitive))


def test_keeper_holds_every_scope():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "keeper"}):
        for scope in Scope:
            _, status, detail = authenticate(BEARER, scope, "op")
            check(f"keeper may {scope.value}", status == 200, detail)


def test_denial_names_the_missing_scope():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "viewer"}):
        _, _, detail = authenticate(BEARER, Scope.WRITE_EXTERNAL, "browser_navigate")
    check("403 names the scope that was required",
          Scope.WRITE_EXTERNAL.value in detail, detail)
    check("403 names the operation", "browser_navigate" in detail, detail)


def test_role_scope_table_is_total_and_ordered():
    check("every role has a scope set", set(ROLE_SCOPES) == set(Role))
    check("keeper holds every scope", ROLE_SCOPES[Role.KEEPER] == frozenset(Scope))
    check("viewer scopes are a subset of operator scopes",
          ROLE_SCOPES[Role.VIEWER] < ROLE_SCOPES[Role.OPERATOR])
    check("operator scopes are a subset of keeper scopes",
          ROLE_SCOPES[Role.OPERATOR] < ROLE_SCOPES[Role.KEEPER])
    check("no role beyond keeper holds write:identity",
          {r for r, s in ROLE_SCOPES.items() if Scope.WRITE_IDENTITY in s} == {Role.KEEPER})


# ── Principal semantics (KAI-DASH-011, 012) ──────────────────────────

def test_session_is_carried_but_grants_nothing():
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "viewer"}):
        principal, status, _ = authenticate(
            BEARER, Scope.READ_OPERATIONAL, "op", session="sess-123")
        _, denied, _ = authenticate(
            BEARER, Scope.WRITE_EXTERNAL, "op", session="sess-123")
    check("session is attached to the principal",
          status == 200 and principal.session == "sess-123")
    check("session does not confer authority", denied == 403, str(denied))


def test_principal_is_immutable():
    p = DashboardPrincipal("a", Role.VIEWER)
    try:
        p.identity = "b"  # type: ignore[misc]
        mutated = True
    except Exception:
        mutated = False
    check("a resolved principal cannot be mutated after the fact", not mutated)


def test_principal_describes_itself_for_audit():
    p = DashboardPrincipal("dainius", Role.KEEPER)
    check("principal renders identity and role for the audit trail",
          p.describe() == "dainius(keeper)", p.describe())


def test_may_matches_role_scopes():
    for role in Role:
        p = DashboardPrincipal("x", role)
        for scope in Scope:
            check(f"{role.value}.may({scope.value}) matches the scope table",
                  p.may(scope) == (scope in ROLE_SCOPES[role]))


# ── Dependency wiring ────────────────────────────────────────────────

def test_dependency_returns_the_principal():
    import asyncio

    dep = require_dashboard_auth(Scope.READ_OPERATIONAL, "status")
    with _Env(**{TOKEN_ENV: GOOD, IDENTITY_ENV: "dainius"}):
        principal = asyncio.run(dep(authorization=BEARER, x_kai_session="s1"))
    check("dependency returns the resolved principal",
          principal.identity == "dainius", principal.identity)
    check("dependency propagates the session header", principal.session == "s1")


def test_dependency_raises_http_exception_on_refusal():
    import asyncio
    from fastapi import HTTPException

    dep = require_dashboard_auth(Scope.WRITE_EXTERNAL, "browser")
    with _Env(**{TOKEN_ENV: GOOD, ROLE_ENV: "viewer"}):
        try:
            asyncio.run(dep(authorization=BEARER, x_kai_session=""))
            raised = None
        except HTTPException as exc:
            raised = exc
    check("dependency raises HTTPException when denied", raised is not None)
    check("dependency raises 403 for an insufficient role",
          raised is not None and raised.status_code == 403,
          str(raised.status_code) if raised else "none")


def test_dependency_fails_closed_when_unconfigured():
    import asyncio
    from fastapi import HTTPException

    dep = require_dashboard_auth(Scope.READ_OPERATIONAL, "status")
    with _Env():
        try:
            asyncio.run(dep(authorization=BEARER, x_kai_session=""))
            status = 200
        except HTTPException as exc:
            status = exc.status_code
    check("dependency fails closed with 503 when unconfigured", status == 503, str(status))


def test_unauthenticated_allowed_reads_the_flag():
    with _Env(**{ALLOW_UNAUTH_ENV: "true"}):
        on = unauthenticated_allowed()
    with _Env(**{ALLOW_UNAUTH_ENV: "false"}):
        off = unauthenticated_allowed()
    with _Env():
        default = unauthenticated_allowed()
    check("escape hatch reads true", on is True)
    check("escape hatch reads false", off is False)
    check("escape hatch defaults to off", default is False)


def run() -> None:
    test_unconfigured_fails_closed()
    test_unconfigured_refuses_even_with_no_header()
    test_escape_hatch_is_explicit_and_single()
    test_escape_hatch_is_still_subject_to_scopes()
    test_empty_token_is_not_a_bypass()
    test_misconfiguration_fails_closed_not_open()
    test_valid_token_authenticates()
    test_missing_header_is_401()
    test_wrong_scheme_is_401()
    test_wrong_token_is_refused()
    test_token_prefix_is_not_accepted()
    test_scheme_is_case_insensitive_but_token_is_not()
    test_multiple_principals_resolve_separately()
    test_duplicate_tokens_are_a_configuration_error()
    test_principal_entries_require_identity_and_token()
    test_unknown_role_in_single_token_form_fails_closed()
    test_the_default_role_is_not_keeper()
    test_keeper_must_be_asked_for_by_name()
    test_the_escape_hatch_does_not_confer_keeper()
    test_viewer_cannot_write()
    test_viewer_can_read_operational()
    test_operator_cannot_rewrite_identity_or_act_externally()
    test_keeper_holds_every_scope()
    test_denial_names_the_missing_scope()
    test_role_scope_table_is_total_and_ordered()
    test_session_is_carried_but_grants_nothing()
    test_principal_is_immutable()
    test_principal_describes_itself_for_audit()
    test_may_matches_role_scopes()
    test_dependency_returns_the_principal()
    test_dependency_raises_http_exception_on_refusal()
    test_dependency_fails_closed_when_unconfigured()
    test_unauthenticated_allowed_reads_the_flag()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Dashboard Auth Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
