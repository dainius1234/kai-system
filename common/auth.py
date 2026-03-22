from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)


def load_secret(env_name: str, default: str = "") -> str:
    """Load a secret from Docker secrets file or environment variable.

    Docker secrets are mounted at /run/secrets/<name>.  If the env var
    value looks like a Docker secret path (starts with /run/secrets/)
    and the file exists, read the value from the file.  Otherwise fall
    back to the raw env var value.
    """
    value = os.getenv(env_name, default)
    # Convention: if env value is a /run/secrets/ path, read from file
    if value.startswith("/run/secrets/"):
        secret_path = Path(value)
        if secret_path.is_file():
            return secret_path.read_text().strip()
        _logger.warning("Secret file %s not found, falling back to empty", value)
        return default
    # Also check if a Docker secret file exists by convention:
    # /run/secrets/<env_name_lowercase>
    secret_by_name = Path(f"/run/secrets/{env_name.lower()}")
    if secret_by_name.is_file():
        return secret_by_name.read_text().strip()
    return value


PRIMARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET"
SECONDARY_SECRET_ENV = "INTERSERVICE_HMAC_SECRET_PREV"
KEY_ID_ENV = "INTERSERVICE_HMAC_KEY_ID"
SECONDARY_KEY_ID_ENV = "INTERSERVICE_HMAC_KEY_ID_PREV"
REVOKED_KEY_IDS_ENV = "INTERSERVICE_HMAC_REVOKED_IDS"
STRICT_KEY_ID_ENV = "INTERSERVICE_HMAC_STRICT_KEY_ID"

_DEV_SECRET = "local-dev-shared-secret"
_WARNED_DEFAULT_SECRET = False


def _allow_dev_secret() -> bool:
    return os.getenv("HMAC_ALLOW_DEV_SECRET", "false").lower() in {"1", "true", "yes"}


def _secret(env_name: str, default: str = "") -> bytes:
    global _WARNED_DEFAULT_SECRET
    value = load_secret(env_name, default)
    if value == _DEV_SECRET:
        if not _allow_dev_secret():
            raise RuntimeError(
                f"HMAC dev secret in use but HMAC_ALLOW_DEV_SECRET is not set. "
                f"Set {env_name} to a real secret, or set HMAC_ALLOW_DEV_SECRET=true for local dev."
            )
        if not _WARNED_DEFAULT_SECRET:
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


def verify_gate_signature(
    *, actor_did: str, session_id: str, tool: str,
    nonce: str, ts: float, signature: Optional[str],
) -> bool:
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
