"""Inbound authentication and authorisation for the Dashboard gateway.

Closes the Wave 1 Track A findings: `KAI-DASH-001` (open privileged
gateway), `011` (no principal model), `012` (no delegation evidence) and
`018` (no least privilege). See `kai-pm/W1_DASHBOARD_REMEDIATION_PLAN.md`.

The dashboard proxies to Agentic, memU, Supervisor, Tool Gate, Financial
Awareness, Browser Agent, Monitor, Files, Notify, Email and Broker. It is
a single unified control plane for the whole stack, and until now every
one of its routes was anonymous.

Design decisions:

**Fail closed.** No credentials configured means 503, not 200. A missing
secret is a misconfiguration, and a misconfigured control plane must not
be open. Same rule as `common/service_auth.py` (roadmap §15.14).

**A principal, not just a token.** A shared bearer token would satisfy
`KAI-DASH-001` while leaving `011` and `012` exactly as they are. Every
request resolves to a `DashboardPrincipal` carrying identity, role and
session, so backend calls can carry *who asked* rather than borrowing the
dashboard's own privilege — which is the confused-deputy shape the audit
found in `KAI-DASH-002`.

**Scopes per route, declared at the route.** `require_dashboard_auth`
takes the scope the route needs. Authenticating everything uniformly
would close `001` and leave `018` untouched: least privilege means a
viewer token cannot drive the browser agent. Declaring it at the route
also keeps a new route unauthenticated *visibly*, rather than silently
inheriting protection from middleware nobody reads.

**No CSRF token.** Deliberate, not an omission. Credentials travel in the
`Authorization` header, which browsers never attach automatically, so
cross-site requests cannot borrow them. A CSRF token here would be
ceremony that implies protection it is not providing. This reasoning
changes the moment any credential moves to a cookie.

**Broker credentials are not read here and must never be.** They stay
inside the broker-bridge service. `scripts/security/check_dashboard_findings.py`
verifies this on every run.

Usage::

    from common.dashboard_auth import require_dashboard_auth, Scope

    @app.post("/api/browser/navigate",
              dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
    async def api_browser_navigate(...): ...

    # Or to use the caller's identity in the handler:
    @app.post("/api/soul")
    async def api_soul_post(
        principal: DashboardPrincipal = Depends(
            require_dashboard_auth(Scope.WRITE_IDENTITY)),
    ): ...
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("kai.dashboard_auth")

PRINCIPALS_ENV = "KAI_DASHBOARD_PRINCIPALS"
TOKEN_ENV = "KAI_DASHBOARD_TOKEN"
IDENTITY_ENV = "KAI_DASHBOARD_IDENTITY"
ROLE_ENV = "KAI_DASHBOARD_ROLE"
ALLOW_UNAUTH_ENV = "KAI_ALLOW_UNAUTHENTICATED"

SESSION_HEADER = "x-kai-session"

_WARNED_UNAUTH: set = set()


class Scope(str, Enum):
    """What a route is allowed to do.

    Ordered by consequence, not by convenience. A route declares the
    least it needs; a role grants the most it should have.
    """

    READ_OPERATIONAL = "read:operational"   # fleet status, health rollups
    READ_SENSITIVE = "read:sensitive"       # memory, finance, broker, email, logs
    WRITE_ROUTINE = "write:routine"         # goals, feedback, clipboard, reads refreshed
    WRITE_IDENTITY = "write:identity"       # SOUL, values, conscience, narrative
    WRITE_EXTERNAL = "write:external"       # browser, schedules, monitors, notifications


class Role(str, Enum):
    """Who the caller is, and therefore what they may do."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    KEEPER = "keeper"


ROLE_SCOPES: Dict[Role, frozenset] = {
    Role.VIEWER: frozenset({Scope.READ_OPERATIONAL}),
    Role.OPERATOR: frozenset({
        Scope.READ_OPERATIONAL,
        Scope.READ_SENSITIVE,
        Scope.WRITE_ROUTINE,
    }),
    # Identity rewrite and external action stay with the keeper. An
    # operator who can rewrite SOUL.md can rewrite what the system is.
    Role.KEEPER: frozenset(Scope),
}


@dataclass(frozen=True)
class DashboardPrincipal:
    """A verified caller.

    ``session`` is caller-supplied and therefore *not* an authorisation
    input — it exists so an action can be correlated across requests in
    the audit trail. Authority comes from ``role`` alone.
    """

    identity: str
    role: Role
    session: Optional[str] = None

    @property
    def scopes(self) -> frozenset:
        return ROLE_SCOPES[self.role]

    def may(self, scope: Scope) -> bool:
        return scope in self.scopes

    def describe(self) -> str:
        return f"{self.identity}({self.role.value})"


class ConfigError(Exception):
    """Credential configuration is present but unusable."""


# ── Credential loading ───────────────────────────────────────────────

def _read_env(name: str, default: str = "") -> str:
    """Read a secret, supporting Docker secrets via common.auth."""
    try:
        from common.auth import load_secret
        return load_secret(name, default)
    except Exception:
        return os.getenv(name, default)


def load_principals() -> List[Tuple[str, DashboardPrincipal]]:
    """Return ``(token, principal)`` pairs from the environment.

    Two forms, both fail closed when absent:

    ``KAI_DASHBOARD_PRINCIPALS`` — JSON list, for multiple callers::

        [{"identity": "dainius", "role": "keeper", "token": "..."}]

    ``KAI_DASHBOARD_TOKEN`` — single caller, with identity and role from
    ``KAI_DASHBOARD_IDENTITY`` / ``KAI_DASHBOARD_ROLE``.
    """
    raw = _read_env(PRINCIPALS_ENV).strip()
    if raw:
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{PRINCIPALS_ENV} is not valid JSON: {exc}") from exc
        if not isinstance(entries, list) or not entries:
            raise ConfigError(f"{PRINCIPALS_ENV} must be a non-empty JSON list")

        pairs: List[Tuple[str, DashboardPrincipal]] = []
        seen_tokens = set()
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigError(f"{PRINCIPALS_ENV}[{i}] is not an object")
            token = str(entry.get("token", "")).strip()
            identity = str(entry.get("identity", "")).strip()
            role_name = str(entry.get("role", "")).strip().lower()
            if not token:
                raise ConfigError(f"{PRINCIPALS_ENV}[{i}] has no token")
            if not identity:
                raise ConfigError(f"{PRINCIPALS_ENV}[{i}] has no identity")
            try:
                role = Role(role_name)
            except ValueError:
                raise ConfigError(
                    f"{PRINCIPALS_ENV}[{i}] has unknown role {role_name!r}; "
                    f"expected one of {', '.join(r.value for r in Role)}"
                ) from None
            if token in seen_tokens:
                raise ConfigError(
                    f"{PRINCIPALS_ENV}[{i}] reuses a token already assigned to "
                    f"another identity; a token must identify exactly one caller"
                )
            seen_tokens.add(token)
            pairs.append((token, DashboardPrincipal(identity=identity, role=role)))
        return pairs

    token = _read_env(TOKEN_ENV).strip()
    if not token:
        return []

    identity = _read_env(IDENTITY_ENV, "operator").strip() or "operator"
    role_name = _read_env(ROLE_ENV, Role.KEEPER.value).strip().lower()
    try:
        role = Role(role_name)
    except ValueError:
        raise ConfigError(
            f"{ROLE_ENV}={role_name!r} is not a known role; expected one of "
            f"{', '.join(r.value for r in Role)}"
        ) from None
    return [(token, DashboardPrincipal(identity=identity, role=role))]


def unauthenticated_allowed() -> bool:
    return os.getenv(ALLOW_UNAUTH_ENV, "false").lower() in {"1", "true", "yes"}


# ── Authentication ───────────────────────────────────────────────────

def authenticate(
    authorization: Optional[str],
    scope: Scope,
    operation: str = "unspecified",
    session: Optional[str] = None,
) -> Tuple[Optional[DashboardPrincipal], int, str]:
    """Resolve a request to a principal, or explain the refusal.

    Returns ``(principal, status_code, detail)``. ``principal`` is None
    whenever the status is not 200. Pure and framework-free so it can be
    tested without a server.
    """
    try:
        principals = load_principals()
    except ConfigError as exc:
        logger.error("SECURITY: dashboard credentials are misconfigured: %s", exc)
        return None, 503, (
            f"{operation} is unavailable: dashboard authentication is "
            f"misconfigured ({exc}). This gateway fails closed by design."
        )

    if not principals:
        if unauthenticated_allowed():
            if operation not in _WARNED_UNAUTH:
                _WARNED_UNAUTH.add(operation)
                logger.warning(
                    "SECURITY: dashboard route '%s' is serving UNAUTHENTICATED "
                    "because %s=true. Never set this outside local development.",
                    operation, ALLOW_UNAUTH_ENV,
                )
            return (
                DashboardPrincipal("local-development", Role.KEEPER, session),
                200,
                "unauthenticated (explicitly allowed)",
            )
        logger.error(
            "SECURITY: dashboard route '%s' refused — neither %s nor %s is "
            "configured. Set credentials, or set %s=true for local development only.",
            operation, PRINCIPALS_ENV, TOKEN_ENV, ALLOW_UNAUTH_ENV,
        )
        return None, 503, (
            f"{operation} is unavailable: dashboard authentication is not "
            f"configured. This gateway fails closed by design."
        )

    if not authorization:
        return None, 401, "missing Authorization header"

    auth_scheme, _, credential = authorization.partition(" ")
    if auth_scheme.lower() != "bearer" or not credential:
        return None, 401, "expected 'Authorization: Bearer <token>'"

    credential = credential.strip()
    matched: Optional[DashboardPrincipal] = None
    for token, principal in principals:
        # Compare against every principal rather than breaking early, so
        # the work done does not depend on which token was presented.
        if hmac.compare_digest(credential, token):
            matched = principal
    if matched is None:
        logger.warning("SECURITY: dashboard rejected unknown token for '%s'", operation)
        return None, 401, "invalid dashboard credentials"

    if not matched.may(scope):
        logger.warning(
            "SECURITY: %s denied '%s' — role %s lacks scope %s",
            matched.describe(), operation, matched.role.value, scope.value,
        )
        return None, 403, (
            f"{matched.role.value} is not permitted to {scope.value}; "
            f"'{operation}' requires it"
        )

    return (
        DashboardPrincipal(matched.identity, matched.role, session),
        200,
        "authenticated",
    )


def require_dashboard_auth(scope: Scope, operation: Optional[str] = None) -> Callable:
    """FastAPI dependency enforcing dashboard authentication.

    Returns the resolved `DashboardPrincipal`, so a handler that needs to
    attribute its backend call can simply depend on it.
    """
    from fastapi import Header, HTTPException

    label = operation or scope.value

    async def _dependency(
        authorization: str = Header(default=""),
        x_kai_session: str = Header(default=""),
    ) -> DashboardPrincipal:
        principal, status, detail = authenticate(
            authorization or None, scope, label, x_kai_session or None,
        )
        if principal is None:
            raise HTTPException(status_code=status, detail=detail)
        return principal

    return _dependency
