from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional


PRIMARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET"
SECONDARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET_PREV"


def _secret(env_name: str, default: str = "") -> bytes:
    value = os.getenv(env_name, default)
    return value.encode("utf-8")


def _payload(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float) -> str:
    return f"{actor_did}|{session_id}|{tool}|{nonce}|{int(ts)}"


def sign_gate_request(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float, key_id: str = "v1") -> str:
    secret = _secret(PRIMARY_SECRET_ENV, "local-dev-shared-secret")
    body = _payload(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)
    digest = hmac.new(secret, body.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{key_id}:{digest}"


def verify_gate_signature(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float, signature: Optional[str]) -> bool:
    if not signature or ":" not in signature:
        return False
    _, candidate = signature.split(":", 1)
    body = _payload(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)

    primary = _secret(PRIMARY_SECRET_ENV, "local-dev-shared-secret")
    expected_primary = hmac.new(primary, body.encode("utf-8"), hashlib.sha256).hexdigest()
    if hmac.compare_digest(expected_primary, candidate):
        return True

    secondary = _secret(SECONDARY_SECRET_ENV)
    if secondary:
        expected_secondary = hmac.new(secondary, body.encode("utf-8"), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected_secondary, candidate):
            return True

    return False
