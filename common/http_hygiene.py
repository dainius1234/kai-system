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

# The payload limits live in `common.perception_spine.ingress`, which
# already owns them. They are resolved lazily rather than imported here,
# for the same reason the transport is built lazily: importing a shared
# utility must not drag in the perception spine. Several service tests
# replace the whole `common` package with a bare stub, and a module-level
# import made every service that adopted this module fail to load under
# them — surfacing as unrelated errors far from the cause.
_LIMIT_NAMES = frozenset({
    "MAX_PAYLOAD_BYTES", "MAX_PAYLOAD_DEPTH",
    "MAX_PAYLOAD_KEYS", "MAX_STRING_LENGTH",
})


def _ingress():
    from common.perception_spine import ingress
    return ingress


def __getattr__(name: str) -> Any:
    """Resolve the shared limits on first access (PEP 562)."""
    if name in _LIMIT_NAMES:
        return getattr(_ingress(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The first four resolve through the PEP 562 `__getattr__` above rather
# than existing as module-level assignments, which is deliberate: the
# limits live in one place and are read on first access. flake8 cannot
# see that and reports F822 "undefined name in __all__" for each — the
# one place in this file where the checker is wrong and the code is
# right. Suppressed narrowly, with the reason, rather than by widening
# the lint's ignore list.
__all__ = [  # noqa: F822 — resolved lazily by __getattr__ (PEP 562)
    "MAX_PAYLOAD_BYTES", "MAX_PAYLOAD_DEPTH", "MAX_PAYLOAD_KEYS",
    "MAX_STRING_LENGTH", "MAX_UPLOAD_BYTES", "bounded_json",
    "bounded_upload", "bounded_response", "pooled_client", "shutdown_pool",
]

# Uploads are a different order of magnitude from event payloads — a
# photo or a voice note is legitimately megabytes — so they get their own
# limit rather than being squeezed under MAX_PAYLOAD_BYTES.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


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

    ingress = _ingress()
    raw = await request.body()
    if len(raw) > ingress.MAX_PAYLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"request body exceeds {ingress.MAX_PAYLOAD_BYTES} bytes",
        )
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="body is not valid JSON")

    violation = ingress.check_payload_bounds(payload)
    if violation:
        raise HTTPException(status_code=413, detail=violation)
    return payload


async def bounded_upload(upload: Any, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file, refusing it *during* the read.

    The obvious version — ``data = await file.read()`` then check
    ``len(data)`` — is `KAI-DASH-045`: by the time the limit is enforced
    the whole body is already in memory, so the limit protects nothing
    that matters. A caller sending 2 GB gets 2 GB buffered and *then* a
    polite 413.

    This reads in chunks and gives up as soon as the total exceeds the
    limit, so the refusal costs one chunk beyond it rather than the whole
    body.
    """
    from fastapi import HTTPException

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413,
                detail=f"upload exceeds {limit} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def bounded_response(content: bytes, source: str,
                     limit: int = MAX_UPLOAD_BYTES) -> bytes:
    """Refuse a backend response that is too large to forward.

    A proxy that materialises whatever a backend sends inherits that
    backend's memory profile (`KAI-DASH-048`, `049`). The bound is here
    rather than in the caller so the answer is the same everywhere.

    Raises ``502``, not ``413``: the oversized thing came from upstream,
    and blaming the client for it would send whoever is debugging in
    precisely the wrong direction.
    """
    from fastapi import HTTPException

    if len(content) > limit:
        raise HTTPException(
            status_code=502,
            detail=f"{source} returned {len(content)} bytes, over the "
                   f"{limit} byte limit",
        )
    return content


# ── Pooled connections ───────────────────────────────────────────────

# The shared transport is built on first use, not at import.
#
# Subclassing `httpx.AsyncHTTPTransport` at module level made importing
# this module require a *complete* httpx. Several service tests stub
# `httpx` with a partial module — enough for their own use — and every
# service that adopted `pooled_client` then failed to import, surfacing
# as unrelated AttributeErrors elsewhere. A shared utility must not
# impose import-time requirements on everything that touches it.
_TRANSPORT: Any = None


def _shared_transport() -> Any:
    """Build (once) a transport that outlives the clients borrowing it.

    ``AsyncClient.aclose()`` closes whatever transport it was handed, so
    a naively shared transport would be closed by the first request and
    found dead by the second. This one ignores ``aclose()``; the process
    that actually owns the pool closes it on shutdown.

    A module-global *client* was tried first and rejected: it outlives
    the event loop it was created on, and silently survives test patching
    of ``httpx.AsyncClient``, which made suites order-dependent. Keeping
    the client per-request and sharing only the pool avoids both, and
    leaves every existing call site and test working unchanged.
    """
    global _TRANSPORT
    if _TRANSPORT is not None:
        return _TRANSPORT

    base = getattr(httpx, "AsyncHTTPTransport", None)
    if base is None:
        # A stubbed or minimal httpx. Pooling is an optimisation, not a
        # correctness property, so degrade to per-client transports
        # rather than refusing to import.
        return None

    class _SharedTransport(base):  # type: ignore[misc]
        async def aclose(self) -> None:  # noqa: D102 - deliberate no-op
            return None

        async def shutdown(self) -> None:
            await base.aclose(self)

    limits = getattr(httpx, "Limits", None)
    _TRANSPORT = _SharedTransport(
        limits=limits(max_connections=100, max_keepalive_connections=20)
    ) if limits else _SharedTransport()
    return _TRANSPORT


def pooled_client(**kwargs: Any) -> Any:
    """Drop-in for ``httpx.AsyncClient(...)`` that reuses connections.

    An explicit ``transport=`` is honoured, so a caller that genuinely
    needs its own pool can still say so.
    """
    if "transport" not in kwargs:
        transport = _shared_transport()
        if transport is not None:
            kwargs["transport"] = transport
    return httpx.AsyncClient(**kwargs)


async def shutdown_pool() -> None:
    """Close the shared pool. Call once, from an app shutdown hook."""
    global _TRANSPORT
    if _TRANSPORT is not None:
        await _TRANSPORT.shutdown()
        _TRANSPORT = None
