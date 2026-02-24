from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

_logger = logging.getLogger(__name__)

PRIMARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET"
SECONDARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET_PREV"
KEY_ID_ENV = "INTERSERVICE_HMAC_KEY_ID"
SECONDARY_KEY_ID_ENV = "INTERSERVICE_HMAC_KEY_ID_PREV"
REVOKED_KEY_IDS_ENV = "INTERSERVICE_HMAC_REVOKED_IDS"
STRICT_KEY_ID_ENV = "INTERSERVICE_HMAC_STRICT_KEY_ID"

_DEV_SECRET = "local-dev-shared-secret"
_WARNED_DEFAULT_SECRET = False


def _secret(env_name: str, default: str = "") -> bytes:
    global _WARNED_DEFAULT_SECRET
    value = os.getenv(env_name, default)
    if value == _DEV_SECRET and not _WARNED_DEFAULT_SECRET:
        _WARNED_DEFAULT_SECRET = True
        _logger.warning(
            "HMAC using default dev secret — set %s for production",
            env_name,
        )
    return value.encode("utf-8")


def _payload(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float) -> str:
    return f"{actor_did}|{session_id}|{tool}|{nonce}|{int(ts)}"


def sign_gate_request(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float, key_id: str = "") -> str:
    secret = _secret(PRIMARY_SECRET_ENV, "local-dev-shared-secret")
    resolved_key_id = key_id or os.getenv(KEY_ID_ENV, "v1")
    body = _payload(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)
    digest = hmac.new(secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{resolved_key_id}:{digest}"


def _strict_key_id() -> bool:
    return os.getenv(STRICT_KEY_ID_ENV, "false").lower() in {"1", "true", "yes"}


def sign_gate_request_bundle(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float) -> list[str]:
    signatures = [
        sign_gate_request(
            actor_did=actor_did,
            session_id=session_id,
            tool=tool,
            nonce=nonce,
            ts=ts,
            key_id=os.getenv(KEY_ID_ENV, "v1"),
        )
    ]
    secondary = _secret(SECONDARY_SECRET_ENV)
    if secondary:
        secondary_key_id = os.getenv(SECONDARY_KEY_ID_ENV, "v0")
        body = _payload(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)
        digest = hmac.new(secondary, body.encode("utf-8"), hashlib.sha256).hexdigest()
        signatures.append(f"{secondary_key_id}:{digest}")
    return signatures


def verify_gate_signature(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float, signature: Optional[str]) -> bool:
    if not signature or ":" not in signature:
        return False
    key_id, candidate = signature.split(":", 1)
    revoked = {x.strip() for x in os.getenv(REVOKED_KEY_IDS_ENV, "").split(",") if x.strip()}
    if key_id in revoked:
        return False
    body = _payload(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)

    primary_key_id = os.getenv(KEY_ID_ENV, "v1")
    secondary_key_id = os.getenv(SECONDARY_KEY_ID_ENV, "v0")
    strict = _strict_key_id()

    primary = _secret(PRIMARY_SECRET_ENV, "local-dev-shared-secret")
    expected_primary = hmac.new(primary, body.encode("utf-8"), hashlib.sha256).hexdigest()
    if (not strict or key_id == primary_key_id) and hmac.compare_digest(expected_primary, candidate):
        return True

    secondary = _secret(SECONDARY_SECRET_ENV)
    if secondary:
        expected_secondary = hmac.new(secondary, body.encode("utf-8"), hashlib.sha256).hexdigest()
        if (not strict or key_id == secondary_key_id) and hmac.compare_digest(expected_secondary, candidate):
            return True

    return False
