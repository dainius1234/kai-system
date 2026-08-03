"""HTTP hygiene shared by every service: bounded bodies, pooled connections.

Two defects the audit found in the dashboard are not dashboard defects.
They are repository-wide habits:

  - **Unbounded request bodies** (`KAI-DASH-017`). A service that reads
    ``await request.json()`` and forwards the result becomes an
    amplifier: one large or deeply nested body turns into work in
    whichever backend it is aimed at.
  - **A new connection pool per request** (`KAI-DASH-074`). Building an
    ``httpx.AsyncClient`` inside a handler opens and tears down a pool
    every time.

Both were first fixed inside `dashboard/app.py`. That was the wrong
altitude, and it produced a *second* implementation of payload bounds
while `common/perception_spine/ingress.py` already had one — the exact
duplication the fix's own comment claimed to be avoiding. The limits live
in one place now, and this module is the HTTP-layer adapter over them, so
the system has one answer to "how big is too big" rather than one per
service.

Usage::

    from common.http_hygiene import bounded_json, pooled_client

    @app.post("/thing")
    async def thing(request: Request):
        payload = await bounded_json(request)     # 413 if oversized
        async with pooled_client(timeout=5.0) as client:
            ...
"""
from __future__ import annotations

from typing import Any

import httpx

from common.perception_spine.ingress import (
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_DEPTH,
    MAX_PAYLOAD_KEYS,
    MAX_STRING_LENGTH,
    check_payload_bounds,
)

__all__ = [
    "MAX_PAYLOAD_BYTES", "MAX_PAYLOAD_DEPTH", "MAX_PAYLOAD_KEYS",
    "MAX_STRING_LENGTH", "bounded_json", "pooled_client", "shutdown_pool",
]


# ── Bounded request bodies ───────────────────────────────────────────

async def bounded_json(request: Any) -> Any:
    """Read a JSON body, refusing anything oversized or pathological.

    Raises ``413`` before the body reaches a backend, because the point
    is to refuse the work — checking after forwarding would protect
    nothing. A malformed body is ``400``: that is a client mistake, not
    a size limit, and conflating them makes both harder to diagnose.
    """
    import json

    from fastapi import HTTPException

    raw = await request.body()
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"request body exceeds {MAX_PAYLOAD_BYTES} bytes",
        )
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not valid JSON")

    violation = check_payload_bounds(payload)
    if violation:
        raise HTTPException(status_code=413, detail=violation)
    return payload


# ── Pooled connections ───────────────────────────────────────────────

class _SharedTransport(httpx.AsyncHTTPTransport):
    """A transport that outlives the clients borrowing it.

    ``AsyncClient.aclose()`` closes whatever transport it was handed, so
    a naively shared transport would be closed by the first request and
    found dead by the second. This ignores ``aclose()``; the process that
    actually owns the pool closes it on shutdown.

    A module-global *client* was tried first and rejected: it outlives
    the event loop it was created on, and silently survives test patching
    of ``httpx.AsyncClient``, which made suites order-dependent. Keeping
    the client per-request and sharing only the pool avoids both, and
    leaves every existing call site and test working unchanged.
    """

    async def aclose(self) -> None:  # noqa: D102 - deliberate no-op
        return None

    async def shutdown(self) -> None:
        await super().aclose()


_TRANSPORT = _SharedTransport(
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


def pooled_client(**kwargs: Any) -> httpx.AsyncClient:
    """Drop-in for ``httpx.AsyncClient(...)`` that reuses connections.

    An explicit ``transport=`` is honoured, so a caller that genuinely
    needs its own pool can still say so.
    """
    kwargs.setdefault("transport", _TRANSPORT)
    return httpx.AsyncClient(**kwargs)


async def shutdown_pool() -> None:
    """Close the shared pool. Call once, from an app shutdown hook."""
    await _TRANSPORT.shutdown()
