"""Structured error codes for sovereign AI services.

Every error returned to a client or logged internally should reference
one of these codes.  This gives operators a single lookup table when
triaging issues instead of grepping for ad-hoc message strings.

Usage:
    from common.errors import KaiError, ErrorCode

    raise KaiError(ErrorCode.CIRCUIT_OPEN, detail="memu-core unreachable")

    # In FastAPI:
    @app.exception_handler(KaiError)
    async def kai_error_handler(request, exc):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Enumerated error codes — one per failure mode."""

    # ── Client errors (4xx) ──────────────────────────────────────────
    VALIDATION_FAILED = "E1001"       # Bad input / schema mismatch
    INJECTION_DETECTED = "E1002"      # Prompt injection blocked
    RATE_LIMITED = "E1003"            # Too many requests
    UNAUTHORIZED = "E1004"            # Missing or invalid HMAC
    FORBIDDEN = "E1005"               # Policy-blocked action
    NOT_FOUND = "E1006"               # Resource not found
    CONFLICT = "E1007"                # Idempotency conflict / duplicate
    INPUT_TOO_LARGE = "E1008"         # Payload exceeds limit

    # ── Service errors (5xx) ─────────────────────────────────────────
    CIRCUIT_OPEN = "E2001"            # Dependency circuit breaker open
    DEPENDENCY_TIMEOUT = "E2002"      # Upstream call timed out
    DEPENDENCY_ERROR = "E2003"        # Upstream returned 5xx
    LLM_UNAVAILABLE = "E2004"        # No LLM backend reachable
    STORAGE_ERROR = "E2005"           # DB / Redis / disk write failed
    INTERNAL_ERROR = "E2006"          # Unclassified internal failure

    # ── Conviction / safety (5xx) ────────────────────────────────────
    CONVICTION_TOO_LOW = "E3001"      # Plan below conviction threshold
    SELF_DECEPTION_DETECTED = "E3002" # Self-deception guard triggered
    VERIFIER_FAIL_CLOSED = "E3003"    # Verifier rejected with FAIL_CLOSED
    ADVERSARY_BLOCKED = "E3004"       # Adversary challenge hard-block
    ERROR_BUDGET_EXHAUSTED = "E3005"  # Error budget breaker tripped

    # ── Operator / system (5xx) ──────────────────────────────────────
    CHECKPOINT_NOT_FOUND = "E4001"    # Invalid checkpoint ID
    RECOVERY_FAILED = "E4002"         # Self-heal attempt failed
    QUARANTINE_ACTIVE = "E4003"       # Service is quarantined
    FEATURE_DISABLED = "E4004"        # Feature flag is off


# ── Code → HTTP status mapping ───────────────────────────────────────
_STATUS_MAP: Dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_FAILED: 400,
    ErrorCode.INJECTION_DETECTED: 400,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.INPUT_TOO_LARGE: 413,
    ErrorCode.CIRCUIT_OPEN: 503,
    ErrorCode.DEPENDENCY_TIMEOUT: 504,
    ErrorCode.DEPENDENCY_ERROR: 502,
    ErrorCode.LLM_UNAVAILABLE: 503,
    ErrorCode.STORAGE_ERROR: 500,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.CONVICTION_TOO_LOW: 422,
    ErrorCode.SELF_DECEPTION_DETECTED: 422,
    ErrorCode.VERIFIER_FAIL_CLOSED: 422,
    ErrorCode.ADVERSARY_BLOCKED: 422,
    ErrorCode.ERROR_BUDGET_EXHAUSTED: 503,
    ErrorCode.CHECKPOINT_NOT_FOUND: 404,
    ErrorCode.RECOVERY_FAILED: 500,
    ErrorCode.QUARANTINE_ACTIVE: 503,
    ErrorCode.FEATURE_DISABLED: 501,
}


class KaiError(Exception):
    """Structured exception carrying an ErrorCode, HTTP status, and detail."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.detail = detail or code.name.replace("_", " ").title()
        self.context = context or {}
        self.http_status = _STATUS_MAP.get(code, 500)
        super().__init__(f"[{code.value}] {self.detail}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dict for API responses."""
        return {
            "error": True,
            "code": self.code.value,
            "name": self.code.name,
            "detail": self.detail,
            "status": self.http_status,
        }
