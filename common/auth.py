from __future__ import annotations

import hashlib
import hmac
import os
from typing import Optional


SHARED_SECRET_ENV = "INTERSERVICE_HMAC_SECRET"


def _secret() -> bytes:
    value = os.getenv(SHARED_SECRET_ENV, "local-dev-shared-secret")
    return value.encode("utf-8")


def sign_gate_request(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float) -> str:
    payload = f"{actor_did}|{session_id}|{tool}|{nonce}|{int(ts)}"
    return hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_gate_signature(*, actor_did: str, session_id: str, tool: str, nonce: str, ts: float, signature: Optional[str]) -> bool:
    if not signature:
        return False
    expected = sign_gate_request(actor_did=actor_did, session_id=session_id, tool=tool, nonce=nonce, ts=ts)
    return hmac.compare_digest(expected, signature)
