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

**Full versus partial.** The 503 rule above governs a read with *no
usable data*. A read that got four of five sources is a different
animal: throwing the four away because the fifth is down destroys real
value, and answering 503 would do exactly that.

So `degraded_partial()` returns **200** with `degraded: true` and an
explicit `degraded_sources` list. What keeps that from becoming the
`degraded_ok()` this module refuses to have is that it *cannot be called
without naming what is missing* — an empty list raises. There is no way
to spell "something went wrong but I would rather not say" with it.

  - no usable data          -> `degraded_response()`, 503
  - some sources answered   -> `degraded_partial()`, 200 + named sources

Added in H-6 after four endpoints in `memu-core` were found combining
several sources behind `except Exception: pass` and returning
`{"status": "ok", "nudge_count": 0}` with every one of them down —
character-identical to a healthy system with nothing to say.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

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


def degraded_partial(
    body: Mapping[str, Any],
    *,
    missing: Any,
    status: str = "degraded",
) -> Dict[str, Any]:
    """A 200-shaped body that admits which sources it could not read.

    For an aggregate whose remaining sources produced a real answer. The
    caller keeps everything that worked and learns, in a machine-readable
    field, what it is missing.

    `missing` is required and may not be empty. That is the entire reason
    this is safe to have: `degraded_ok()` was refused because a helper
    that lets you mark a response degraded without saying what failed is
    a way to stop thinking. This one makes you name it, or it raises.
    """
    names = [str(m) for m in missing]
    if not names:
        raise ValueError(
            "degraded_partial() requires the sources that failed. If you "
            "cannot name one, nothing is degraded — return the normal "
            "body. If nothing usable was read, use degraded_response()."
        )
    out: Dict[str, Any] = dict(body)
    out.update({
        "status": status,
        "degraded": True,
        "degraded_sources": names,
        "observed_at": _now_iso(),
    })
    return out


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


# ── Recording a dependency failure that is survived rather than raised ──
#
# Some failures genuinely should not propagate. `memu-core` keeps an
# in-process fallback for every Redis operation precisely so a Redis
# outage degrades the service instead of ending it, and that is a real
# design decision, not an oversight.
#
# What *was* an oversight is that it was spelled `except Exception: pass`
# — 33 times in one file. The fallback is fine; discarding the reason is
# not. A single-process fallback silently makes the service
# non-durable and non-shared: twelve workers reading twelve divergent
# in-memory lists, every health check green, and nothing anywhere saying
# the word "Redis".
#
# So a swallow is defensible only when all four hold:
#
#   1. the fallback is a genuine answer, not a placeholder,
#   2. the caller cannot act on the failure,
#   3. the failure is recorded, and
#   4. the record is *aggregatable* — an operator can tell "failing for
#      ten seconds" from "failing for ten days" without reading a log
#      line at a time.
#
# (3) and (4) are what this section provides. `record_degradation` is
# deliberately one call, shorter to type than the `pass` it replaces, so
# the correct handler is the path of least resistance:
#
#     except Exception as exc:
#         record_degradation("redis", "p20_put_value", exc)
#
# Logging is rate-limited per (source, operation) because these sit in
# hot loops — an unthrottled warning inside a per-request Redis write is
# its own outage. Every emitted line carries the running count and the
# age of the failure, so one line answers "how long, how often".

_DEGRADATION_LOG_EVERY = 300.0  # seconds between repeat log lines, per key
_MAX_TRACKED = 200  # a bound: an unbounded registry is a slow leak

_degradations: Dict[str, Dict[str, Any]] = {}
_degradation_lock = threading.Lock()
_degradation_logger = logging.getLogger("kai.degraded")


def record_degradation(
    source: str,
    operation: str,
    exc: BaseException | str,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Record a dependency failure the caller is deliberately surviving.

    Never raises: a failure in the failure recorder must not become the
    thing that takes the process down.
    """
    try:
        key = f"{source}:{operation}"
        reason = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        reason = reason[:300]
        now = time.time()
        with _degradation_lock:
            entry = _degradations.get(key)
            if entry is None:
                if len(_degradations) >= _MAX_TRACKED:
                    # Drop the least recently seen rather than refusing to
                    # track: the newest failure is the one being diagnosed.
                    oldest = min(_degradations, key=lambda k: _degradations[k]["last_seen"])
                    _degradations.pop(oldest, None)
                entry = {
                    "source": source,
                    "operation": operation,
                    "count": 0,
                    "first_seen": now,
                    "last_seen": now,
                    "_last_logged": 0.0,
                }
                _degradations[key] = entry
            entry["count"] += 1
            entry["last_seen"] = now
            entry["reason"] = reason
            should_log = (now - entry["_last_logged"]) >= _DEGRADATION_LOG_EVERY
            if should_log:
                entry["_last_logged"] = now
            count = entry["count"]
            age = now - entry["first_seen"]
        if should_log:
            (logger or _degradation_logger).warning(
                "degraded dependency source=%s operation=%s count=%d "
                "failing_for_seconds=%d reason=%s",
                source, operation, count, int(age), reason,
            )
    except Exception:  # pragma: no cover - defensive
        pass


def degradation_report() -> List[Dict[str, Any]]:
    """Every dependency failure survived since start, newest first.

    Intended for a health or introspection endpoint, so the answer to
    "is anything quietly broken" is a request rather than a log search.
    """
    now = time.time()
    with _degradation_lock:
        entries = list(_degradations.values())
    report = [
        {
            "source": e["source"],
            "operation": e["operation"],
            "count": e["count"],
            "reason": e.get("reason", ""),
            "first_seen": datetime.fromtimestamp(e["first_seen"], timezone.utc).isoformat(),
            "last_seen": datetime.fromtimestamp(e["last_seen"], timezone.utc).isoformat(),
            "failing_for_seconds": int(now - e["first_seen"]),
            "stale_seconds": int(now - e["last_seen"]),
        }
        for e in entries
    ]
    report.sort(key=lambda r: r["last_seen"], reverse=True)
    return report


def reset_degradations() -> None:
    """Clear the registry. For tests; nothing in production should call it."""
    with _degradation_lock:
        _degradations.clear()
