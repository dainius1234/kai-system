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
    """FastAPI dependency enforcing service *membership*.

    Correct for the six class-A endpoints — read-only, no attribution,
    identical response for any authorised caller. For the twenty-six
    class-B endpoints, where the caller's identity changes what should be
    recorded or permitted, use ``require_service_identity`` instead.

    ``operation`` names the protected action and appears in logs and in
    the 503 body, so a misconfiguration says which endpoint refused.
    """
    from fastapi import Header, HTTPException

    async def _dependency(authorization: str = Header(default="")) -> None:
        ok, status, detail = check_token(authorization or None, operation)
        if not ok:
            raise HTTPException(status_code=status, detail=detail)

    return _dependency


# ── identity: who called, not merely that someone did ───────────────────
#
# The measurement that produced this (2026-08-07): 26 of 32 protected
# endpoints need the caller's identity, and the shared token cannot
# supply it. `common/service_identity` derives the principal from the key
# that signed the request. Nothing below reads a name from a header.

REQUIRE_IDENTITY_ENV = "KAI_REQUIRE_SERVICE_IDENTITY"

_keymap = None
_nonce_cache = None
_keymap_error = ""
_WARNED_TRANSITION: set = set()


def identity_required() -> bool:
    """Whether an unsigned caller is refused outright.

    False during the migration window, when a class-B endpoint still
    accepts the shared token but records the caller as **unverified** —
    which never reaches a provenance record, because
    ``ServicePrincipal.usable_for_provenance`` is False for it.

    This defaults to False so the migration does not break every caller
    on the commit that lands it. That is a deliberate, temporary widening
    and it is loud: every such request logs, and
    `scripts/security/check_service_identity_rollout.py` prints how many
    services are still in the window.
    """
    return os.getenv(REQUIRE_IDENTITY_ENV, "false").lower() in {
        "1", "true", "yes"}


def _identity_context():
    """Load the key map and replay cache once, not per request."""
    global _keymap, _nonce_cache, _keymap_error
    if _keymap is not None or _keymap_error:
        return _keymap, _nonce_cache
    try:
        from common.service_identity import KeyMap, NonceCache
        _keymap = KeyMap.load()
        _nonce_cache = NonceCache()
    except Exception as exc:
        # Recorded, not raised. Whether an absent key map is fatal
        # depends on identity_required(), which is decided per request.
        _keymap_error = f"{type(exc).__name__}: {exc}"
        logger.warning("service identity unavailable: %s", _keymap_error)
    return _keymap, _nonce_cache


def reset_identity_context() -> None:
    """Drop the cached key map. For tests and for key rotation."""
    global _keymap, _nonce_cache, _keymap_error
    _keymap = _nonce_cache = None
    _keymap_error = ""


def check_identity(
    headers, operation: str, *, destination: str, method: str, path: str,
    body: bytes,
):
    """Validate a request and return ``(principal, status, detail)``.

    Order matters. A **valid signature always wins**, in both directions:
    it is tried first so a signed caller is identified even during the
    transition window, and a *bad* signature is refused outright rather
    than falling back to the shared token. Without that second half, an
    attacker could strip a signature and downgrade to anonymous
    membership, which would make the whole mechanism optional.
    """
    from common.service_identity import (unverified_principal,
                                         verify_request, SIGNATURE_HEADER)

    lower = {str(k).lower(): v for k, v in dict(headers).items()}
    keymap, cache = _identity_context()

    if lower.get(SIGNATURE_HEADER):
        if keymap is None:
            logger.error(
                "SECURITY: '%s' received a signed request but has no usable "
                "key map (%s) — refusing rather than downgrading",
                operation, _keymap_error)
            return None, 503, (
                f"{operation} cannot verify caller identity: "
                f"the service key map is unavailable")
        try:
            principal, status, detail = verify_request(
                lower, destination=destination, method=method, path=path,
                body=body, keymap=keymap, cache=cache)
        except Exception as exc:
            # Reached when the key map loads but the Ed25519 backend does
            # not — a missing `cryptography` in the image. Without this,
            # an absent dependency surfaced as a 500 from somewhere deep,
            # which is the least useful shape a configuration error can
            # take. It is still a refusal: no fallback.
            logger.error("SECURITY: '%s' could not evaluate a signature "
                         "(%s: %s)", operation, type(exc).__name__, exc)
            return None, 503, (
                f"{operation} cannot verify caller identity: "
                f"the signing backend is unavailable")
        # No fallback on failure. See docstring.
        return principal, status, detail

    if identity_required():
        logger.warning("SECURITY: '%s' refused an unsigned caller", operation)
        return None, 401, (
            f"{operation} requires a signed request: this endpoint's "
            f"behaviour depends on which service called")

    ok, status, detail = check_token(lower.get("authorization"), operation)
    if not ok:
        return None, status, detail
    if operation not in _WARNED_TRANSITION:
        _WARNED_TRANSITION.add(operation)
        logger.warning(
            "SECURITY: '%s' accepted a shared-token caller. Identity is "
            "UNVERIFIED and will not appear in any provenance record. Set "
            "%s=true once every caller signs.", operation,
            REQUIRE_IDENTITY_ENV)
    return unverified_principal(), 200, "membership only (unverified)"


def require_service_identity(operation: str) -> Callable:
    """FastAPI dependency yielding a ``ServicePrincipal``.

    For the class-B endpoints. The handler receives the principal and
    must check ``usable_for_provenance`` before attributing anything to
    it — during the transition window it may legitimately be anonymous,
    and recording an anonymous caller by name is exactly the defect this
    replaces.
    """
    from fastapi import HTTPException, Request

    async def _dependency(request: "Request"):
        principal, status, detail = check_identity(
            request.headers, operation,
            destination=os.getenv("KAI_SERVICE_NAME", ""),
            method=request.method,
            path=request.url.path,
            body=await request.body(),
        )
        if principal is None:
            raise HTTPException(status_code=status, detail=detail)
        return principal

    return _dependency
