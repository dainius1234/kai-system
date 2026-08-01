"""Shared service authentication for side-effecting endpoints.

Closes UH tracker gap G-03.  The audit found six endpoints that perform
real side effects — including a PostgreSQL restore that overwrites the
live database — with no authentication at all.

Design decisions:

**Fail closed.**  An endpoint protected by ``require_service_auth`` with
no token configured returns 503, not 200.  A missing secret is a
misconfiguration, and a misconfigured destructive endpoint must not be
open.  This follows roadmap §15.14 (no fail-open in execution paths).

**One explicit, greppable escape hatch.**  Local development sets
``KAI_ALLOW_UNAUTHENTICATED=true``.  It logs a loud warning on every
request and is the only way to bypass.  There is deliberately no
"allow if token happens to be empty" path.

**Constant-time comparison.**  Token checks use ``hmac.compare_digest``
so a wrong token cannot be recovered by timing.

Usage::

    from common.service_auth import require_service_auth

    @app.post("/restore/postgres",
              dependencies=[Depends(require_service_auth("db_restore"))])
    async def restore_postgres(...): ...
"""
from __future__ import annotations

import hmac
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger("kai.service_auth")

TOKEN_ENV = "KAI_SERVICE_TOKEN"
ALLOW_UNAUTH_ENV = "KAI_ALLOW_UNAUTHENTICATED"

_WARNED_UNAUTH: set[str] = set()


def _load_token() -> str:
    """Read the shared service token, supporting Docker secrets."""
    try:
        from common.auth import load_secret
        return load_secret(TOKEN_ENV, "")
    except Exception:
        return os.getenv(TOKEN_ENV, "")


def unauthenticated_allowed() -> bool:
    return os.getenv(ALLOW_UNAUTH_ENV, "false").lower() in {"1", "true", "yes"}


def check_token(
    authorization: Optional[str],
    operation: str = "unspecified",
) -> tuple[bool, int, str]:
    """Validate an Authorization header.

    Returns ``(ok, status_code, detail)``.  Pure and framework-free so it
    can be unit-tested without spinning up a server.
    """
    expected = _load_token()

    if not expected:
        if unauthenticated_allowed():
            if operation not in _WARNED_UNAUTH:
                _WARNED_UNAUTH.add(operation)
                logger.warning(
                    "SECURITY: '%s' is serving UNAUTHENTICATED because %s=true. "
                    "Never set this outside local development.",
                    operation, ALLOW_UNAUTH_ENV,
                )
            return True, 200, "unauthenticated (explicitly allowed)"
        logger.error(
            "SECURITY: '%s' refused — %s is not configured. "
            "Set the token, or set %s=true for local development only.",
            operation, TOKEN_ENV, ALLOW_UNAUTH_ENV,
        )
        return False, 503, (
            f"{operation} is unavailable: service authentication is not "
            f"configured. This endpoint fails closed by design."
        )

    if not authorization:
        return False, 401, "missing Authorization header"

    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential:
        return False, 401, "expected 'Authorization: Bearer <token>'"

    if not hmac.compare_digest(credential.strip(), expected):
        logger.warning("SECURITY: rejected bad token for '%s'", operation)
        return False, 403, "invalid service token"

    return True, 200, "authenticated"


def require_service_auth(operation: str) -> Callable:
    """FastAPI dependency enforcing service authentication.

    ``operation`` names the protected action and appears in logs and in
    the 503 body, so a misconfiguration says which endpoint refused.
    """
    from fastapi import Header, HTTPException

    async def _dependency(authorization: str = Header(default="")) -> None:
        ok, status, detail = check_token(authorization or None, operation)
        if not ok:
            raise HTTPException(status_code=status, detail=detail)

    return _dependency
