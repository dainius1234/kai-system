"""Degraded-state envelope — an outage must not look like an answer.

Closes the Wave 1 Track D findings (`KAI-DASH-016`, `061`, `063`–`067`,
`080`, `082`). They are nine symptoms of one disease: the dashboard could
not distinguish *"there is no data"* from *"I could not get the data"*,
and reported the second as the first.

Concretely, before this: a dead memU produced `{"nudges": []}` with HTTP
200 — indistinguishable from a healthy memU with nothing to say. An
unreachable backup service produced a fresh timestamp and the words
"service healthy". Absence of evidence was being rendered as evidence of
absence, and in a system whose whole purpose is deciding what is safe to
act on, that is the most expensive kind of bug.

Two channels carry the truth, deliberately:

  - **HTTP status.** A degraded read answers 503, so any machine consumer
    can tell without parsing.
  - **`degraded: true` in the body,** with a reason and the source that
    failed, so a human reading a panel can tell *why*.

The body also **keeps the shape the caller expected** (`{"nudges": []}`
stays `{"nudges": [], "degraded": true, ...}`). That is not politeness to
the UI: it means adopting this cannot silently break a panel into
throwing, which would have been a second outage dressed as a fix.

There is deliberately no `degraded_ok()` helper returning 200. Every use
site should have to think about it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

STATUS_UNAVAILABLE = "unavailable"
DEGRADED_STATUS_CODE = 503


def _now_iso() -> str:
    """Timezone-aware, because a naive stamp is its own finding (083/084)."""
    return datetime.now(timezone.utc).isoformat()


def degraded_body(
    source: str,
    reason: str,
    shape: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a body that cannot be mistaken for a successful read.

    ``shape`` is the empty form the caller expects, so existing consumers
    keep working; the markers are added on top and always win, since a
    backend that happens to return a key called ``degraded`` must not be
    able to talk its way out of being reported as degraded.
    """
    body: Dict[str, Any] = dict(shape or {})
    body.update({
        "status": STATUS_UNAVAILABLE,
        "degraded": True,
        "source": source,
        "reason": str(reason)[:300],
        "observed_at": _now_iso(),
    })
    return body


def degraded_response(
    source: str,
    reason: str,
    shape: Optional[Mapping[str, Any]] = None,
    status_code: int = DEGRADED_STATUS_CODE,
):
    """A JSON 503 carrying the degraded body.

    Imported lazily so this module stays usable without FastAPI.
    """
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content=degraded_body(source, reason, shape),
    )


def is_degraded(payload: Any) -> bool:
    """True when a payload is an explicit degraded envelope.

    Used to stop a degraded read being folded into an aggregate as if it
    were real data — the failure mode behind `KAI-DASH-062`, where zero
    counts from an unreachable backend were accepted as readiness
    evidence.
    """
    if not isinstance(payload, Mapping):
        return False
    if payload.get("degraded") is True:
        return True
    return payload.get("status") == STATUS_UNAVAILABLE


def unavailable_metric(name: str, reason: str) -> Dict[str, Any]:
    """A metric that is explicitly absent rather than wrongly substituted.

    For cases where the evidence genuinely cannot be obtained. The point
    is that a check reporting "I cannot measure this" is honest, whereas
    silently measuring something else and calling it the same name — a
    total count standing in for proven decisions (`063`), or the
    dashboard's own error ratio standing in for system reliability
    (`064`) — is not.
    """
    return {
        "metric": name,
        "available": False,
        "reason": str(reason)[:300],
        "observed_at": _now_iso(),
    }
