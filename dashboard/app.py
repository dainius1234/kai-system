

from __future__ import annotations
import asyncio
import hashlib
import time
import re
from functools import lru_cache
from datetime import datetime, timezone
import json as _json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import redis.asyncio as aioredis
from fastapi import (Body, Depends, FastAPI, File, HTTPException, Query,
                     Request, UploadFile)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (HTMLResponse, JSONResponse,
                               StreamingResponse)

from common.dashboard_auth import (DashboardPrincipal, Role, Scope,
                                   require_dashboard_auth)
from common.degraded import (degraded_response, is_degraded,
                             record_degradation, unavailable_metric)
from common.http_hygiene import (MAX_PAYLOAD_BYTES, bounded_json,
                                 bounded_response, bounded_upload,
                                 pooled_client, shutdown_pool)
from common.resilience import resilient_call
from common.runtime import AuditStream, ErrorBudget, detect_device, setup_json_logger

try:
    from common.policy import policy_hash, policy_version
except Exception:
    policy_hash = "unavailable"
    policy_version = "unknown"

logger = setup_json_logger("dashboard", os.getenv("LOG_PATH", "/tmp/dashboard.json.log"))

# ── Backend destinations (KAI-DASH-072) ──────────────────────────────
#
# Every backend address comes from the environment. Unvalidated, a typo or
# a tampered variable silently redirects the dashboard's whole authority
# at an attacker-chosen host — and because the dashboard proxies dozens of
# private services, that is the single highest-leverage variable in the
# deployment. Validated once, at import, so a bad value fails the
# container start rather than the first request that needs it.
_ALLOWED_SCHEMES = {"http", "https"}


def backend_url(name: str, default: str, optional: bool = False,
                schemes: Optional[set] = None) -> str:
    """Read a backend base URL, refusing anything that is not one.

    ``optional`` allows an empty value, for backends that are only wired
    up in some profiles — but an empty value is the *only* thing it
    allows. A half-written URL is still refused.
    """
    from urllib.parse import urlsplit

    allowed = schemes or _ALLOWED_SCHEMES
    raw = os.getenv(name, default).strip()
    if not raw:
        if optional:
            return ""
        raise RuntimeError(f"{name} is required but empty")
    parts = urlsplit(raw)
    if parts.scheme not in allowed:
        raise RuntimeError(
            f"{name}={raw!r} must use {' or '.join(sorted(allowed))}, "
            f"not {parts.scheme!r}"
        )
    if not parts.hostname:
        raise RuntimeError(f"{name}={raw!r} has no host")
    if parts.query or parts.fragment:
        raise RuntimeError(
            f"{name}={raw!r} must be a base URL, not a full request"
        )
    return raw.rstrip("/")


DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.4.0")

# mount static UI stub
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

TOOL_GATE_URL = backend_url("TOOL_GATE_URL", "http://tool-gate:8000")
VERIFIER_URL = backend_url("VERIFIER_URL", "http://verifier:8052")
# Store-maintenance reads (stats/search-by-category/quarantine listing) live on
# memu-core-introspect, split out from memu-core's hot path — see DECISIONS.md D21.
MEMU_INTROSPECT_URL = backend_url("MEMU_INTROSPECT_URL", "http://memu-core-introspect:8009")
budget = ErrorBudget(window_seconds=300)


# ── H1.7 + H2: Safe proxy helpers with retry + circuit breaker ──────
# A backend's content type is echoed only if it is one we expect. Trusting
# it outright would let a compromised backend pick the type the browser
# renders with; forcing a constant (KAI-DASH-089/090) mislabels everything
# that is not that constant. Neither is safe on its own.
ALLOWED_AUDIO_TYPES = frozenset({
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav",
    "audio/ogg", "audio/webm", "audio/aac", "audio/flac",
})
ALLOWED_IMAGE_TYPES = frozenset({
    "image/png", "image/jpeg", "image/webp", "image/gif",
})
ALLOWED_DOC_TYPES = frozenset({
    "application/pdf", "application/octet-stream", "text/plain", "text/csv",
    "application/zip", "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
})


def _safe_media_type(reported: str | None, allowed: frozenset,
                     fallback: str) -> str:
    """Echo the backend's type when recognised, else a neutral default."""
    if not reported:
        return fallback
    base = reported.split(";", 1)[0].strip().lower()
    return base if base in allowed else fallback


def _audit_actor(request: Request) -> str:
    """Identify the caller for the audit line (KAI-DASH-096).

    Derived from the credential presented, never from a caller-supplied
    name — an actor a client can choose is not an actor. The token itself
    is never logged; only a short digest, so two entries can be told
    apart without the log becoming a place credentials live.
    """
    header = request.headers.get("authorization", "")
    _, _, credential = header.partition(" ")
    credential = credential.strip()
    if not credential:
        return "actor=anonymous"
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
    session = request.headers.get("x-kai-session", "")
    suffix = f" session={session[:32]}" if session else ""
    return f"actor={digest}{suffix}"


# Connection pooling and payload bounds are repository-wide concerns, not
# dashboard ones — see `common/http_hygiene.py`. They were first written
# here, which produced a second copy of limits that
# `common/perception_spine/ingress.py` already owned.
@app.on_event("shutdown")
async def _close_shared_transport() -> None:
    await shutdown_pool()


# ── Outbound safety for uploads, errors and path parameters ──────────

_SYMBOL = re.compile(r"^[A-Za-z0-9._-]{1,20}$")


def safe_symbol(value: str, label: str = "symbol") -> str:
    """Validate a path parameter before it is interpolated into a URL.

    Symbols and pairs are pasted straight into backend URLs — including
    an outbound Binance query (KAI-DASH-094). Without a character class
    a caller controls part of the request the dashboard makes on its
    behalf, which is a request-forgery primitive, not a formatting bug.
    """
    candidate = (value or "").strip()
    if not _SYMBOL.match(candidate):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be 1-20 characters of A-Z, 0-9, dot, dash "
                   f"or underscore",
        )
    return candidate.upper()


# ── Outbound safety for uploads and errors ───────────────────────────

_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def safe_filename(name: str | None, fallback: str = "upload") -> str:
    """Canonicalise a caller-supplied filename before forwarding it.

    The name arrived from a browser and is passed to parser and OCR
    services that may write it to disk (`KAI-DASH-051`). Path separators,
    traversal segments and control characters are removed here rather
    than being every downstream service's problem.
    """
    if not name:
        return fallback
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = base.lstrip(".") or fallback
    cleaned = _SAFE_FILENAME.sub("_", base)[:120]
    return cleaned or fallback


def safe_content_type(reported: str | None, allowed: frozenset,
                      fallback: str) -> str:
    """Constrain a caller-declared content type to what we expect.

    The browser's declared type is a hint, not a fact (`KAI-DASH-091`).
    Forwarding it unchecked lets a caller tell the parser or vision
    service to treat a file as something it is not.
    """
    if not reported:
        return fallback
    base = reported.split(";", 1)[0].strip().lower()
    return base if base in allowed else fallback


def client_error(exc: Exception, message: str) -> str:
    """Log the real cause, return something safe to show a caller.

    Proxy errors were formatted straight into the response detail, which
    disclosed internal service URLs and transport diagnostics to anyone
    who could trigger a failure (`KAI-DASH-052`). The detail belongs in
    the log; the caller gets the shape of the problem, not its innards.
    """
    logger.warning("%s: %s", message, exc)
    return message


async def _proxy_get(url: str, params: dict | None = None,
                     fallback: Any = None, timeout: float = 10.0) -> Any:
    """GET from a backend with retry, circuit breaker, and fallback."""
    default = fallback if fallback is not None else {"status": "unavailable"}
    return await resilient_call(
        "GET", url, params=params, timeout=timeout,
        retries=2, backoff=0.3, fallback=default, logger=logger,
    )


async def _proxy_post(url: str, body: dict | None = None,
                      fallback: Any = None, timeout: float = 10.0) -> Any:
    """POST to a backend with retry, circuit breaker, and fallback."""
    default = fallback if fallback is not None else {"status": "unavailable"}
    return await resilient_call(
        "POST", url, json=body or {}, timeout=timeout,
        retries=2, backoff=0.3, fallback=default, logger=logger,
    )


# ── New API endpoints for dashboard UI extras ───────────────────────

@app.post("/api/mode",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_ROUTINE))])
async def api_set_mode(body: Dict[str, Any] = Body(...)):
    """Accept browser personality mode. Display-state only — does not mutate Tool Gate."""
    mode = str(body.get("mode", "")).upper()
    if mode not in ("WORK", "PUB"):
        raise HTTPException(status_code=400, detail="mode must be WORK or PUB")
    return {"status": "local_only", "mode": mode}


@app.get("/api/nudges",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_nudges():
    memu_url = backend_url("MEMU_URL", "http://memu-core:8001")
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{memu_url}/memory/proactive")
            resp.raise_for_status()
            payload = resp.json()
            return {"nudges": payload.get("nudges", [])}
    except Exception as exc:
        # An unreachable memU is not "no nudges" (KAI-DASH-067, 082).
        return degraded_response("memu", str(exc), {"nudges": []})


@app.get("/api/backup-status",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_backup_status():
    """Report the most recent backup that actually exists.

    Previously this probed ``/health`` and, if the service answered,
    printed the *current* time with the words "service healthy" — a fresh
    timestamp that proved only that a process was running, and read as
    proof that a backup had just been taken (KAI-DASH-065).
    """
    backup_url = backend_url("BACKUP_SERVICE_URL", "http://backup-service:8054")
    empty = {"status": "unknown", "latest_backup": None, "total_backups": 0}
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{backup_url}/backup/list")
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        return degraded_response("backup-service", str(exc), empty)

    backups = payload.get("backups") or []
    if not backups:
        # A reachable service with no backups is a real, reportable state —
        # and it is not healthy.
        return {
            "status": "no backups found",
            "latest_backup": None,
            "total_backups": 0,
            "verified": True,
        }
    latest = backups[0]
    return {
        "status": f"{latest.get('modified', 'unknown')} ({latest.get('filename', 'backup')})",
        "latest_backup": latest,
        "total_backups": payload.get("total", len(backups)),
        "verified": True,
    }


@app.get("/api/corrections",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_corrections():
    verifier_url = backend_url("VERIFIER_URL", "http://verifier:8052")
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{verifier_url}/metrics")
            resp.raise_for_status()
            payload = resp.json()
            verdicts = payload.get("verdicts", {})
            # These are running totals from the verifier's metrics
            # endpoint, not dated events. Stamping each with now() made
            # aggregates look like a chronology of corrections that had
            # just happened (KAI-DASH-066). They carry no timestamp
            # because none is known.
            corrections = [
                {
                    "verdict": verdict,
                    "count": count,
                    "kind": "aggregate",
                    "timestamp": None,
                    "summary": f"{verdict}: {count} total",
                }
                for verdict, count in verdicts.items()
                if verdict in ("REPAIR", "FAIL_CLOSED")
            ]
            return {"corrections": corrections, "kind": "aggregate_counts"}
    except Exception as exc:
        return degraded_response("verifier", str(exc), {"corrections": []})

# Audit defaults to *required* (KAI-DASH-096). An audit trail that is
# optional is not an audit trail: the one deployment that forgets to set
# the flag is exactly the one whose history you will later want.
audit = AuditStream(
    "dashboard",
    required=os.getenv("AUDIT_REQUIRED", "true").lower() == "true",
)

SUPERVISOR_URL = backend_url("SUPERVISOR_URL", "http://supervisor:8051")

NODES: Dict[str, str] = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": backend_url("MEMU_URL", "http://memu-core:8001") + "/health",
    "heartbeat": backend_url("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
    "supervisor": f"{SUPERVISOR_URL}/health",
    "verifier": backend_url("VERIFIER_URL", "http://verifier:8052") + "/health",
    "fusion-engine": backend_url("FUSION_URL", "http://fusion-engine:8053") + "/health",
    "memory-compressor": backend_url("MEMORY_COMPRESSOR_URL", "http://memory-compressor:8057") + "/health",
    "ledger-worker": backend_url("LEDGER_WORKER_URL", "http://ledger-worker:8056") + "/health",
    "metrics-gateway": backend_url("METRICS_GATEWAY_URL", "http://metrics-gateway:8058") + "/health",
}
_agentic_url = backend_url("LANGGRAPH_URL", "", optional=True)
if _agentic_url:
    NODES["agentic"] = _agentic_url + "/health"
_executor_url = backend_url("EXECUTOR_URL", "", optional=True)
if _executor_url:
    NODES["executor"] = _executor_url + "/health"
_wake_url = backend_url("WAKE_URL", "", optional=True)
if _wake_url:
    NODES["wake-service"] = _wake_url + "/health"


def _as_number(value: Any, cast=float, default: Any = None) -> Any:
    """Coerce a backend-supplied number, returning `default` if it is not.

    Backend fields are not trusted to be well-formed (KAI-DASH-079);
    a malformed one should degrade a single field, not the response.
    """
    if value is None:
        return default
    try:
        result = cast(value)
    except (TypeError, ValueError):
        return default
    if isinstance(result, float) and (result != result or result in (
            float("inf"), float("-inf"))):
        return default
    return result


def _bounded_setting(name: str, default: str, low: float, high: float,
                     cast=float):
    """Read a threshold, refusing values outside a safe range.

    These decide whether the system is judged fit to act. Parsed with a
    bare `int()`/`float()` (KAI-DASH-078), a typo could set the error
    tolerance to 5000% and the gate would agree with it forever. A
    misconfiguration must fail loudly at import, not silently at the
    moment it matters.
    """
    raw = os.getenv(name, default)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"{name}={raw!r} is not a number; expected {low}..{high}"
        ) from None
    if not (low <= value <= high):
        raise RuntimeError(f"{name}={value} is outside the safe range {low}..{high}")
    return value


NO_GO_GRACE_REQUESTS = _bounded_setting(
    "NO_GO_GRACE_REQUESTS", "20", 1, 100_000, int)
MAX_ERROR_RATIO = _bounded_setting("MAX_ERROR_RATIO", "0.05", 0.0, 1.0)


# ── Gateway workload controls (KAI-DASH-056) ─────────────────────────
#
# The gateway had no rate limit, concurrency cap or caller quota, so a
# single client could fan every request out across a dozen backends. The
# cap is on *concurrent in-flight requests* rather than a rolling rate:
# it bounds the work actually happening at once, which is what protects
# the backends, and it needs no per-caller bookkeeping to do it.
MAX_CONCURRENT_REQUESTS = int(os.getenv("DASHBOARD_MAX_CONCURRENCY", "64"))
_REQUEST_SLOTS = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Applied to HTML and static responses. There is no inline script in the
# shell, so 'unsafe-inline' is deliberately absent from script-src; the
# CDN origins the shell loads from are named explicitly rather than
# opened up with a wildcard.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Apply browser protections and bound concurrent work.

    Both live here because both must hold for every route, including any
    added later. Unlike authentication — which is declared per route so
    that a new unprotected route is *visible* — these are properties no
    route should be able to opt out of by omission.
    """
    try:
        await asyncio.wait_for(_REQUEST_SLOTS.acquire(), timeout=5.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "degraded": True,
                     "source": "dashboard",
                     "reason": "gateway is at its concurrency limit"},
            headers={"Retry-After": "1"},
        )
    try:
        response = await call_next(request)
    finally:
        _REQUEST_SLOTS.release()

    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/html"):
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        audit.log("info", f"{_audit_actor(request)} {request.method} "
                          f"{request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{_audit_actor(request)} {request.method} "
                           f"{request.url.path} -> 500")
        raise


# A backend that answers 200 while reporting its own trouble is not
# healthy. These are the self-reported values that mean "not ok".
UNHEALTHY_REPORTED = {"error", "unhealthy", "down", "fail", "failed",
                      "unavailable", "degraded", "critical"}


def _classify_node(payload: Any) -> Tuple[str, str]:
    """Judge a node by what it says about itself, not by its HTTP code.

    Treating any 2xx as healthy (KAI-DASH-061) meant a service reporting
    ``{"status": "degraded"}`` counted towards readiness.
    """
    if not isinstance(payload, dict):
        return "ok", ""
    reported = str(payload.get("status", "")).strip().lower()
    if not reported:
        return "ok", ""
    for bad in UNHEALTHY_REPORTED:
        if bad in reported:
            return "degraded", f"backend reports status={payload.get('status')!r}"
    return "ok", ""


# A single request can ask for node status more than once — `/` builds
# the go/no-go report *and* its own summary (KAI-DASH-058), and readiness
# used to call the whole of `/` (KAI-DASH-060). A short TTL collapses
# those into one fan-out without making the data meaningfully staler than
# the probes themselves.
STATUS_CACHE_TTL = float(os.getenv("DASHBOARD_STATUS_TTL", "2.0"))

# The nodes without which nothing else is meaningful. Named once so
# readiness and the root summary cannot drift apart.
CORE_NODES = ("tool-gate", "memu-core")

# Internal topology and policy state on the root payload (KAI-DASH-068).
# Off by default: useful when debugging a deployment, not something a
# status page should hand out as a matter of course.
EXPOSE_TOPOLOGY = os.getenv("DASHBOARD_EXPOSE_TOPOLOGY", "false").lower() in {
    "1", "true", "yes"}
_status_cache: Dict[str, Any] = {"at": 0.0, "value": None}


async def fetch_status(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Probe every node concurrently and honour its self-report.

    Sequential probing (KAI-DASH-057) meant the worst case was the sum of
    every timeout; ``asyncio.gather`` bounds it to the slowest single node.
    """
    now = time.monotonic()
    cached = _status_cache["value"]
    if (not force and cached is not None
            and now - _status_cache["at"] < STATUS_CACHE_TTL):
        return cached

    async def probe(client, name: str, url: str) -> Tuple[str, Dict[str, Any]]:
        try:
            resp = await client.get(url, timeout=2.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return name, {"status": "down", "error": str(exc)}
        status, note = _classify_node(payload)
        entry: Dict[str, Any] = {"status": status, "details": payload}
        if note:
            entry["error"] = note
        return name, entry

    async with pooled_client() as client:
        settled = await asyncio.gather(
            *(probe(client, name, url) for name, url in NODES.items()),
            return_exceptions=True,
        )
    results: Dict[str, Dict[str, Any]] = {}
    for item in settled:
        if isinstance(item, BaseException):
            continue
        name, entry = item
        results[name] = entry
    # A probe that vanished is not a healthy probe.
    for name in NODES:
        results.setdefault(name, {"status": "down", "error": "probe did not complete"})
    _status_cache["at"] = time.monotonic()
    _status_cache["value"] = results
    return results


async def build_go_no_go_report() -> Dict[str, Any]:
    reasons: List[str] = []
    statuses = await fetch_status()
    down_nodes = [name for name, payload in statuses.items() if payload.get("status") != "ok"]
    if down_nodes:
        reasons.append(f"Critical services are down: {', '.join(down_nodes)}")

    try:
        async with pooled_client() as client:
            tool_health_resp = await client.get(f"{TOOL_GATE_URL}/health", timeout=2.0)
            tool_health_resp.raise_for_status()
            tool_health = tool_health_resp.json()

            ledger_stats_resp = await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)
            ledger_stats_resp.raise_for_status()
            ledger_stats = ledger_stats_resp.json()
    except Exception:
        tool_health = {}
        ledger_stats = {}
        reasons.append("Unable to reach tool-gate for go/no-go checks.")

    metrics = budget.snapshot()

    mode = str(tool_health.get("mode", "PUB")).upper()
    if mode != "WORK":
        reasons.append("Tool Gate is not in WORK mode.")

    # KAI-DASH-063 — the proof metric.
    #
    # `/ledger/stats` returns only a total count. Proof of safe operation
    # is *recent, approved, successful* decisions, and that detail lives
    # in `/ledger/tail`, which requires a privileged Tool Gate token.
    # Giving the dashboard one would recreate exactly the confused deputy
    # that KAI-DASH-002 and 012 describe, so the metric is reported as
    # unavailable rather than substituted. A total count standing in for
    # proof is not a weaker measurement — it is a different one wearing
    # the same name.
    proof = unavailable_metric(
        "recent_approved_decisions",
        "requires privileged ledger access; the dashboard deliberately "
        "holds no Tool Gate credential (KAI-DASH-002)",
    )
    # KAI-DASH-079 — a backend returning "many" for a count should not
    # take the whole report down with a ValueError.
    proof["total_ledger_entries"] = _as_number(
        ledger_stats.get("count") if ledger_stats else None, int)
    unprovable = [
        "Recent approved decisions cannot be proven from here: the "
        "dashboard holds no ledger credential by design."
    ]

    # KAI-DASH-064 — the reliability metric.
    #
    # This used the dashboard's own HTTP error ratio, which measures how
    # often *callers* got errors from the dashboard, not whether the
    # system executes reliably. Fleet health is the closest honest signal
    # the dashboard can observe first-hand.
    total_nodes = len(statuses) or 1
    healthy_nodes = sum(1 for p in statuses.values() if p.get("status") == "ok")
    fleet_unhealthy_ratio = 1.0 - (healthy_nodes / total_nodes)
    if fleet_unhealthy_ratio > MAX_ERROR_RATIO:
        reasons.append(
            f"Fleet reliability is too low "
            f"({healthy_nodes}/{total_nodes} nodes healthy)."
        )

    # Three-valued on purpose. "I cannot tell" is not "no", and it is
    # certainly not "yes" — collapsing either way is how a dashboard ends
    # up asserting something it has not established. Only a clean GO is a
    # success; both other states fail closed (KAI-DASH-080).
    if reasons:
        decision, trust = "NO_GO", "prove-first"
        summary = "Hold execution until blockers are fixed."
    elif unprovable:
        decision, trust = "INDETERMINATE", "unproven"
        summary = "No blockers found, but readiness could not be proven."
    else:
        decision, trust = "GO", "trusted"
        summary = "System looks stable enough to proceed."

    return {
        "decision": decision,
        "trust_status": trust,
        "summary": summary,
        "unprovable": unprovable,
        "checks": {
            "required_mode": "WORK",
            "current_mode": mode,
            "proof_of_safe_operation": proof,
            "max_unhealthy_ratio": MAX_ERROR_RATIO,
            "fleet_unhealthy_ratio": round(fleet_unhealthy_ratio, 4),
            "healthy_nodes": healthy_nodes,
            "total_nodes": total_nodes,
            # Kept for the UI, explicitly labelled as what it is: a
            # measure of this process, not of the system.
            "dashboard_caller_error_ratio": _as_number(
                metrics.get("error_ratio"), float, 0.0),
            "down_nodes": down_nodes,
        },
        "reasons": reasons,
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness only.

    This is one of six routes that answer without a principal, so it must
    disclose nothing an anonymous caller should not have. It previously
    returned the Tool Gate URL, policy version and policy hash — internal
    topology and policy state, handed to anyone who could reach the port
    (KAI-DASH-069).
    """
    return {"status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)"}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def index() -> Dict[str, object]:
    statuses = await fetch_status()
    alive_nodes = [name for name, payload in statuses.items() if payload.get("status") == "ok"]
    # KAI-DASH-062 — these used to default to 0 and then be checked with
    # `>= 0`, which is true of the default. An unreachable Tool Gate and a
    # healthy empty one were indistinguishable, and both counted as ready.
    # None means "not observed"; readiness now requires observation.
    ledger_size = None
    memory_count = None
    try:
        async with pooled_client() as client:
            ledger_size = _as_number(
                (await client.get(f"{TOOL_GATE_URL}/ledger/stats",
                                  timeout=2.0)).json().get("count"), int)
            memory_count = _as_number(
                (await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats",
                                  timeout=2.0)).json().get("records"), int)
    except Exception as exc:
        logger.warning("Failed to fetch ledger/memory stats for index: %s", exc)

    go_no_go = await build_go_no_go_report()
    tool_gate_health = statuses.get("tool-gate", {}).get("details", {})
    policy_mode = str(tool_gate_health.get("mode", "PUB")).upper()

    # v7: fetch breaker states, quarantine count, verifier stats
    breaker_states: Dict[str, Any] = {}
    quarantine_count = 0
    verifier_stats: Dict[str, Any] = {}
    try:
        async with pooled_client(timeout=2.0) as client:
            # breakers from supervisor
            br_resp = await client.get(f"{SUPERVISOR_URL}/breakers")
            if br_resp.status_code == 200:
                breaker_states = br_resp.json()
    except Exception as _exc:
        record_degradation("supervisor", "read_breakers", _exc)
    try:
        async with pooled_client(timeout=2.0) as client:
            # quarantine count
            q_resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/quarantine/list")
            if q_resp.status_code == 200:
                quarantine_count = q_resp.json().get("count", 0)
    except Exception as _exc:
        record_degradation("memu", "read_quarantine", _exc)
    try:
        verifier_url = backend_url("VERIFIER_URL", "http://verifier:8052")
        async with pooled_client(timeout=2.0) as client:
            v_resp = await client.get(f"{verifier_url}/metrics")
            if v_resp.status_code == 200:
                verifier_stats = v_resp.json()
    except Exception as _exc:
        record_degradation("verifier", "read_metrics", _exc)
    core_nodes = CORE_NODES
    if _executor_url:
        core_nodes.append("executor")
    core_ready = (
        all(node in alive_nodes for node in core_nodes)
        and ledger_size is not None
        and memory_count is not None
    )
    return {
        "service": "dashboard",
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "core_ready": core_ready,
        "alive_nodes": alive_nodes,
        # KAI-DASH-077 — this used to carry each backend's full health
        # document. A status page needs to know *whether* a node is well,
        # not everything the node chose to say about itself.
        "node_status": {
            name: {k: v for k, v in payload.items() if k != "details"}
            for name, payload in statuses.items()
        },
        "ledger_size": ledger_size,
        "memory_count": memory_count,
        "policy_mode": policy_mode,
        "device_summary": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "go_no_go": go_no_go,
        # KAI-DASH-068 — `tool_gate_url`, `policy_version` and
        # `policy_hash` described the deployment's internal topology and
        # policy state. They are operator diagnostics, not status, so they
        # are behind an explicit flag rather than on by default.
        **(
            {
                "tool_gate_url": TOOL_GATE_URL,
                "policy_version": policy_version,
                "policy_hash": policy_hash,
            }
            if EXPOSE_TOPOLOGY else {}
        ),
        "breaker_states": breaker_states,
        "quarantine_count": quarantine_count,
        "verifier_stats": verifier_stats,
    }


@app.get("/go-no-go",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def go_no_go():
    """Return the go/no-go report, with a status a machine can enforce.

    This answered 200 for NO_GO (KAI-DASH-080), so anything consuming it
    programmatically saw a successful response and had to know to read
    the body to discover it had been told to stop.
    """
    report = await build_go_no_go_report()
    if report.get("decision") == "GO":
        return report
    return JSONResponse(status_code=503, content=report)


@app.get("/ui")
async def ui() -> HTMLResponse:
    # minimal single-page status dashboard
    html = """<!doctype html>
<html><head><title>Sovereign Dashboard</title>
<style>body{font-family:sans-serif;} .node{display:inline-block;padding:0.5em;margin:0.2em;border:1px solid #333;border-radius:4px;} .ok{background:#8f8;} .down{background:#f88;} </style>
</head><body>
<h1>Sovereign Core Status</h1>
<div id="nodes"></div>
<script>
async function refresh(){
  const r = await fetch('/');
  if(!r.ok){document.body.innerHTML='<p>unable to fetch status</p>';return;}
  const data = await r.json();
  const container=document.getElementById('nodes');
  container.innerHTML='';
  for(const [name,st] of Object.entries(data.node_status||{})){
    const div=document.createElement('div');div.className='node '+(st.status==='ok'?'ok':'down');
    div.textContent=name+' '+st.status;
    container.appendChild(div);
  }
}
setInterval(refresh,2000);
refresh();
</script>
</body></html>"""
    return HTMLResponse(html)


@app.get("/fleet",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def fleet() -> Dict[str, Any]:
    """Proxy the supervisor's fleet health view into the dashboard."""
    try:
        async with pooled_client(timeout=3.0) as client:
            resp = await client.get(f"{SUPERVISOR_URL}/status")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('supervisor', str(exc), {"fleet": "unknown", "error": "supervisor unreachable"})


@app.get("/readiness",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def readiness() -> Dict[str, Any]:
    """Bounded readiness check.

    This used to call `index()` — the entire root fan-out, including the
    go/no-go report and every widget (KAI-DASH-060). A readiness probe
    that costs as much as the dashboard's most expensive page is a probe
    that makes the thing it measures worse.
    """
    statuses = await fetch_status()
    down = sorted(name for name in CORE_NODES
                  if statuses.get(name, {}).get("status") != "ok")
    if down:
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "core_ready": False,
                    "down_nodes": down},
        )
    return {"status": "ready", "core_ready": True, "checked": sorted(CORE_NODES)}


# ── P8: Thinking Pathways — intelligence proxy endpoints ─────────────
MEMU_URL = backend_url("MEMU_URL", "http://memu-core:8001")
HEARTBEAT_URL = backend_url("HEARTBEAT_URL", "http://heartbeat:8010")
FINANCIAL_URL = backend_url("FINANCIAL_URL", "http://financial-awareness:8063")
WAKE_URL = backend_url("WAKE_URL", "http://wake-service:8022")


@app.get("/thinking")
async def thinking_page() -> HTMLResponse:
    """Serve the Thinking Pathways (legacy standalone, redirects to /app)."""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app">')


@app.get("/api/thinking")
async def api_thinking(
    principal: DashboardPrincipal = Depends(
        require_dashboard_auth(Scope.READ_SENSITIVE)),
):
    """Fetch latest episode data from agentic for thinking pathway visualization."""
    agentic_url = backend_url("LANGGRAPH_URL", "http://agentic:8007")
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.post(
                f"{agentic_url}/episodes/recall",
                json={"user_id": principal.identity, "days": 7},
            )
            resp.raise_for_status()
            data = resp.json()
            episodes = data.get("episodes", [])
            # Extract thinking pathway data from most recent episodes
            pathways = []
            for ep in episodes[-10:]:
                pathways.append({
                    "episode_id": ep.get("episode_id", ""),
                    "input": ep.get("input", "")[:200],
                    "output": ep.get("output", "")[:200],
                    "conviction_score": ep.get("conviction_score", 0),
                    "final_conviction": ep.get("final_conviction", 0),
                    "rethink_count": ep.get("rethink_count", 0),
                    "failure_class": ep.get("failure_class"),
                    "metacognitive_rule": ep.get("metacognitive_rule"),
                    "learning_value": ep.get("learning_value", 0),
                    "ts": ep.get("ts", 0),
                })
            return {
                "status": "ok",
                "total_episodes": data.get("count", 0),
                "pathways": pathways,
            }
    except Exception as exc:
        return degraded_response('agentic', str(exc), {"status": "unavailable", "total_episodes": 0, "pathways": []})


@app.get("/api/tempo",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_tempo():
    """Proxy operator tempo from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/tempo")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable", "tempo": "unknown"})


@app.get("/api/boundary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_boundary():
    """Proxy knowledge boundary map from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/boundary")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable", "zones": []})


@app.get("/api/silence",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_silence():
    """Proxy silence-as-signal data from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/silence")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable", "silence_topics": []})


@app.get("/api/self-assessment",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_self_assessment():
    """Proxy temporal self-model from heartbeat."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{HEARTBEAT_URL}/self-assessment")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('backend', str(exc), {"status": "unavailable"})


@app.post("/api/dream",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def api_dream():
    """Trigger a dream consolidation cycle via agentic-introspect."""
    introspect_url = backend_url("AGENTIC_INTROSPECT_URL", "http://agentic-introspect:8023")
    try:
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(f"{introspect_url}/dream")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('agentic-introspect', str(exc), {"status": "unavailable", "message": "Cannot reach agentic-introspect for dream cycle"})


@app.get("/api/ledger-stats",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_ledger_stats():
    """Proxy ledger statistics from ledger-worker."""
    ledger_url = backend_url("LEDGER_WORKER_URL", "http://ledger-worker:8056")
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{ledger_url}/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('backend', str(exc), {"status": "unavailable", "total_entries": 0})


# ── Redis pub/sub — real-time event streaming ────────────────────────
# An event stream commits its status line before anything is known to
# have failed, so a mid-stream failure cannot be signalled with an HTTP
# code. It is signalled in-band instead — as an explicit error event, not
# as content that reads like a normal answer (KAI-DASH-016).
def _sse_error(message: str) -> str:
    payload = _json.dumps({"event": "error", "degraded": True,
                           "error": message})
    return f"data: {payload}\n\ndata: [DONE]\n\n"


# Every subscriber holds a Redis connection for the life of the stream,
# so this is a connection cap as much as a client cap (KAI-DASH-042).
# A plain counter rather than a Semaphore: admission must be *refused*
# when full, not awaited, and asyncio gives us the single-threaded
# guarantee that makes the check-then-increment safe.
MAX_SSE_CLIENTS = int(os.getenv("DASHBOARD_MAX_SSE_CLIENTS", "32"))
_sse_clients = 0

REDIS_URL = backend_url("REDIS_URL", "redis://redis:6379/0",
                        schemes={"redis", "rediss"})

# Channels that the dashboard subscribes to
_EVENT_CHANNELS = [
    "kai:health",          # service up/down events
    "kai:episode",         # new episode recorded
    "kai:breaker",         # circuit breaker state change
    "kai:memory",          # memory store changes
]


# KAI-DASH-085 — a Redis client was constructed and closed per publish.
# One lazily-created client, closed on shutdown, like the HTTP pool.
_REDIS_CLIENT: Any = None


def _redis():
    global _REDIS_CLIENT
    if _REDIS_CLIENT is None:
        _REDIS_CLIENT = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _REDIS_CLIENT


@app.on_event("shutdown")
async def _close_redis() -> None:
    global _REDIS_CLIENT
    if _REDIS_CLIENT is not None:
        await _REDIS_CLIENT.aclose()
        _REDIS_CLIENT = None


def _event_visible_to(data: Any, principal: DashboardPrincipal) -> bool:
    """Whether a subscriber should receive an event (KAI-DASH-044).

    Events that name a subject belong to that subject. Events that do
    not are system-level and visible to any authenticated subscriber.
    Defaulting an unlabelled event to *visible* is deliberate: the event
    bus carries operational signal that a status page needs, and silently
    dropping it would trade a disclosure bug for a blindness bug. The
    keeper sees everything either way.
    """
    if principal.role is Role.KEEPER:
        return True
    if not isinstance(data, dict):
        return True
    subject = data.get("user_id") or data.get("subject") or data.get("principal")
    if not subject:
        return True
    return str(subject) == principal.identity


_event_publish_failures: Dict[str, int] = {}


async def _publish_event(channel: str, data: dict) -> bool:
    """Publish a JSON event to a Redis channel.

    Returns whether it was delivered. Previously the failure path logged
    at DEBUG and returned nothing (KAI-DASH-086), so an event bus that
    had stopped delivering looked exactly like one with nothing to say —
    the same confusion as KAI-DASH-067, one layer down. Failures are
    counted and surfaced on `/metrics`.
    """
    try:
        await _redis().publish(channel, _json.dumps(data))
        return True
    except Exception as exc:
        _event_publish_failures[channel] = (
            _event_publish_failures.get(channel, 0) + 1
        )
        logger.warning("event publish to %s failed (%d so far): %s",
                       channel, _event_publish_failures[channel], exc)
        return False


@app.get("/api/events",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def sse_events(
    request: Request,
    principal: DashboardPrincipal = Depends(
        require_dashboard_auth(Scope.READ_SENSITIVE)),
):
    """Server-Sent Events stream backed by Redis pub/sub.

    The dashboard JS streams this and receives real-time updates instead
    of polling.

    Each subscriber holds a dedicated Redis connection and pubsub for as
    long as it stays connected, so an unbounded client count is an
    unbounded Redis connection count (KAI-DASH-042). Admission is capped,
    and a refused client is told so rather than being quietly starved.
    """
    global _sse_clients
    if _sse_clients >= MAX_SSE_CLIENTS:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "degraded": True,
                     "source": "dashboard",
                     "reason": f"event stream is at its limit "
                               f"({MAX_SSE_CLIENTS} concurrent subscribers)"},
            headers={"Retry-After": "5"},
        )
    _sse_clients += 1

    async def event_generator():
        global _sse_clients
        try:
            r = aioredis.from_url(REDIS_URL, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe(*_EVENT_CHANNELS)
        except Exception as exc:
            _sse_clients = max(0, _sse_clients - 1)
            logger.warning("event stream failed: %s", exc)
            yield _sse_error("the event stream is unavailable")
            return

        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=15.0,
                )
                if msg and msg["type"] == "message":
                    # KAI-DASH-043 — a single unparseable payload used to
                    # raise out of the loop and terminate the client's
                    # stream. One bad publisher must not disconnect every
                    # subscriber.
                    raw = msg.get("data")
                    try:
                        data = _json.loads(raw) if isinstance(raw, str) else raw
                    except ValueError:
                        logger.warning("dropping malformed event on %s",
                                       msg.get("channel"))
                        continue
                    # KAI-DASH-044 — every subscriber received every
                    # event on every channel. An event carrying a subject
                    # is now delivered only to that subject; events with
                    # no subject are system-wide and go to everyone.
                    if not _event_visible_to(data, principal):
                        continue
                    payload = {"channel": msg.get("channel"), "data": data}
                    yield f"data: {_json.dumps(payload)}\n\n"
                else:
                    # keepalive heartbeat every 15s
                    heartbeat = {"channel": "heartbeat",
                                 "ts": datetime.now(timezone.utc).isoformat()}
                    yield f"data: {_json.dumps(heartbeat)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # The slot must come back however the stream ended, or the
            # cap becomes a slow leak towards permanent refusal.
            _sse_clients = max(0, _sse_clients - 1)
            await pubsub.unsubscribe(*_EVENT_CHANNELS)
            await pubsub.aclose()
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/security-audit",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_security_audit():
    """Proxy security self-hacking audit from agentic-introspect."""
    introspect_url = backend_url("AGENTIC_INTROSPECT_URL", "http://agentic-introspect:8023")
    try:
        async with pooled_client(timeout=15.0) as client:
            resp = await client.get(f"{introspect_url}/security/audit")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('agentic-introspect', str(exc), {"status": "unavailable", "findings": [], "risk_score": -1})


# ── P16 API proxies ─────────────────────────────────────────────────

@app.get("/api/goals",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_goals():
    """Proxy Ohana goals from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/goals")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable", "goals": []})


@app.post("/api/goals",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_ROUTINE))])
async def api_goals_create(request: Request):
    """Proxy create goal to memu-core."""
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/goals", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "error", "detail": "Cannot reach memu-core"})


@app.post("/api/goals/update",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_ROUTINE))])
async def api_goals_update(request: Request):
    """Proxy update goal progress to memu-core."""
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/goals/update", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "error", "detail": "Cannot reach memu-core"})


@app.get("/api/drift",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_drift():
    """Proxy drift detection from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/drift")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable"})


@app.get("/api/memories")
async def api_memories(
    query: str = "", category: str = "", top_k: int = Query(20, ge=1, le=200),
    principal: DashboardPrincipal = Depends(
        require_dashboard_auth(Scope.READ_SENSITIVE)),
):
    """Browse memories — search or list by category.

    Memory reads are scoped to the calling principal. ``user_id`` is
    required by ``/memory/retrieve``; omitting it returned 422, so search
    never worked from here.
    """
    try:
        async with pooled_client(timeout=5.0) as client:
            if query:
                resp = await client.get(
                    f"{MEMU_URL}/memory/retrieve",
                    params={"query": query, "user_id": principal.identity,
                            "top_k": top_k},
                )
            elif category:
                resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/search-by-category", params={"category": category, "top_k": top_k})
            else:
                resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu-introspect', str(exc), {"status": "unavailable", "memories": []})


@app.get("/api/memory/stats",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_memory_stats():
    """Proxy memory statistics from memu-core-introspect."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu-introspect', str(exc), {"status": "unavailable"})


@app.get("/api/memories/recent")
async def api_memories_recent(
    top_k: int = Query(30, ge=1, le=200),
    principal: DashboardPrincipal = Depends(
        require_dashboard_auth(Scope.READ_SENSITIVE)),
):
    """Browse recent memories for Diary tab (recency-weighted retrieve)."""
    raw = await _proxy_get(
        f"{MEMU_URL}/memory/retrieve",
        params={"query": "memories thoughts observations experiences",
                "user_id": principal.identity, "top_k": top_k},
        fallback=[],
    )
    records = raw if isinstance(raw, list) else raw.get("records", raw.get("memories", []))
    if not isinstance(records, list):
        records = []
    return {"records": records, "count": len(records)}


@app.get("/api/memory/graph-data")
async def api_memory_graph_data(
    top_k: int = Query(80, ge=1, le=200), query: str = "memories experiences observations",
    principal: DashboardPrincipal = Depends(
        require_dashboard_auth(Scope.READ_SENSITIVE)),
):
    """Return recent memories formatted as {nodes, links} for the D3 force-graph tab."""
    raw = await _proxy_get(
        f"{MEMU_URL}/memory/retrieve",
        params={"query": query, "user_id": principal.identity, "top_k": top_k},
        fallback=[],
    )
    records = raw if isinstance(raw, list) else raw.get("records", raw.get("memories", []))
    if not isinstance(records, list):
        records = []

    cat_counts: dict = {}
    mem_nodes = []
    links = []

    for r in records:
        cat = (r.get("category") or "general").lower().strip()
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        mem_id = f"mem:{r.get('id', str(id(r)))}"
        content = r.get("content") or {}
        if isinstance(content, dict):
            snippet = content.get("text", content.get("result_raw", r.get("event_type", "")))
        else:
            snippet = str(content)

        mem_nodes.append({
            "id": mem_id,
            "type": "memory",
            "label": r.get("event_type", "memory"),
            "snippet": str(snippet)[:100],
            "category": cat,
            "trust_tier": r.get("trust_tier", "unverified"),
            "importance": float(r.get("importance") or r.get("relevance") or 0.5),
            "timestamp": r.get("timestamp", ""),
            "pinned": bool(r.get("pinned", False)),
            "access_count": int(r.get("access_count", 0)),
        })
        links.append({"source": mem_id, "target": f"cat:{cat}"})

    cat_nodes = [
        {"id": f"cat:{cat}", "type": "category", "label": cat, "count": count}
        for cat, count in sorted(cat_counts.items())
    ]
    return {
        "nodes": cat_nodes + mem_nodes,
        "links": links,
        "categories": sorted(cat_counts.keys()),
        "count": len(records),
    }


@app.get("/api/finance/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_finance_summary():
    """Proxy CIS/VAT/tax financial summary from the financial-awareness service (P29)."""
    return await _proxy_get(f"{FINANCIAL_URL}/finance/summary", fallback={
        "status": "unavailable",
        "cis_summary": {},
        "vat_position": {},
        "tax_estimate": {},
        "invoices": [],
    })


@app.get("/api/finance/cis",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_finance_cis():
    """Proxy CIS YTD summary from the financial-awareness service."""
    return await _proxy_get(f"{FINANCIAL_URL}/finance/cis/summary", fallback={"status": "unavailable"})


@app.post("/api/finance/cis/record",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_finance_cis_record(request: Request):
    """Proxy CIS payment record creation to the financial-awareness service."""
    body = await bounded_json(request)
    return await _proxy_post(f"{FINANCIAL_URL}/finance/cis/record", body=body, fallback={"status": "unavailable"})


AGENTIC_URL = backend_url("LANGGRAPH_URL", "http://agentic:8007")


@app.get("/api/soul",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_soul_get():
    """Return current SOUL.md content from agentic."""
    return await _proxy_get(f"{AGENTIC_URL}/soul", fallback={"status": "unavailable", "content": ""})


@app.post("/api/soul",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def api_soul_post(request: Request):
    """Update SOUL.md content via agentic."""
    body = await bounded_json(request)
    return await _proxy_post(f"{AGENTIC_URL}/soul", body=body, fallback={"status": "unavailable"})


@app.get("/api/agents-registry",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_agents_registry_get():
    """Return current AGENTS.md content from agentic."""
    return await _proxy_get(f"{AGENTIC_URL}/agents-registry", fallback={"status": "unavailable", "content": ""})


@app.post("/api/agents-registry",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def api_agents_registry_post(request: Request):
    """Update AGENTS.md content via agentic."""
    body = await bounded_json(request)
    return await _proxy_post(f"{AGENTIC_URL}/agents-registry", body=body, fallback={"status": "unavailable"})


@app.post("/api/pii/scan",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_pii_scan(request: Request):
    """Scan text for PII (and optionally redact) via the verifier service."""
    body = await bounded_json(request)
    return await _proxy_post(
        f"{VERIFIER_URL}/redact",
        body={"text": body.get("text", ""), "auto_redact": body.get("auto_redact", True)},
        fallback={"status": "unavailable", "pii_found": {}, "total_pii": 0},
    )


@app.get("/api/struggle",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_struggle(session_id: str = "default"):
    """Proxy struggle detection from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/struggle", params={"session_id": session_id})
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable", "struggle_score": 0})


@app.post("/api/feedback",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def api_feedback(request: Request):
    """Proxy feedback rating to memu-core."""
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/feedback", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "error", "detail": "Cannot reach memu-core"})


@app.get("/api/feedback/stats",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_feedback_stats():
    """Proxy feedback stats from memu-core."""
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/feedback/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('memu', str(exc), {"status": "unavailable"})


@app.get("/api/logs",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_logs(limit: int = Query(100, ge=1, le=500), level: str = "", since: float = Query(0, ge=0)):
    """Aggregate logs from memu-core (and potentially other services)."""
    all_logs: list = []
    params: dict = {"limit": limit}
    if level:
        params["level"] = level
    if since:
        params["since"] = since

    # Collect from memu-core
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/logs", params=params)
            if resp.status_code == 200:
                data = resp.json()
                all_logs.extend(data.get("entries", []))
    except Exception as _exc:
        record_degradation("memu", "read_logs", _exc)

    # Collect from agentic
    agentic_url = backend_url("LANGGRAPH_URL", "http://agentic:8007")
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.get(f"{agentic_url}/logs", params=params)
            if resp.status_code == 200:
                data = resp.json()
                all_logs.extend(data.get("entries", []))
    except Exception as _exc:
        record_degradation("agentic", "read_logs", _exc)

    # Sort all by timestamp (most recent first)
    all_logs.sort(key=lambda x: x.get("time", 0), reverse=True)

    return {
        "status": "ok",
        "count": len(all_logs[:limit]),
        "entries": all_logs[:limit],
    }


# ── P17: Emotional Intelligence Proxies (H1.7: all wrapped) ──────────

@app.post("/api/emotion/record",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_emotion_record(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/emotion/record", body)


@app.get("/api/emotion/timeline",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_emotion_timeline(
    session_id: str | None = None,
    limit: int = Query(50, ge=1, le=500),
):
    params: dict = {"limit": limit}
    if session_id:
        params["session_id"] = session_id
    return await _proxy_get(f"{MEMU_URL}/memory/emotion/timeline", params=params,
                            fallback={"entries": [], "count": 0})


@app.post("/api/reflect",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_reflect(request: Request):
    body = await bounded_json(request) if (await request.body()) else {}
    return await _proxy_post(f"{MEMU_URL}/memory/self-reflect", body, timeout=15.0)


@app.get("/api/reflections",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_reflections(limit: int = Query(10, ge=1, le=500)):
    return await _proxy_get(f"{MEMU_URL}/memory/self-reflections", params={"limit": limit},
                            fallback={"entries": [], "count": 0})


@app.get("/api/relationship",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_relationship():
    return await _proxy_get(f"{MEMU_URL}/memory/relationship")


@app.post("/api/relationship/milestone",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_milestone(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/relationship/milestone", body)


@app.get("/api/confidence",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_confidence():
    return await _proxy_get(f"{MEMU_URL}/memory/confidence")


@app.get("/api/eq/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_eq_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/eq/summary")


@app.post("/api/confess",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_confess(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/confess", body)


# ── P18: Narrative Identity proxies (H1.7: all wrapped) ─────────────

@app.post("/api/autobiography/record",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_autobiography_record(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/autobiography/record", body)


@app.get("/api/autobiography",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_autobiography(request: Request):
    return await _proxy_get(f"{MEMU_URL}/memory/autobiography", params=dict(request.query_params),
                            fallback={"entries": [], "count": 0})


@app.get("/api/identity",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_identity():
    return await _proxy_get(f"{MEMU_URL}/memory/identity")


@app.get("/api/story-arcs",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_story_arcs():
    return await _proxy_get(f"{MEMU_URL}/memory/story-arcs", fallback={"arcs": []})


@app.get("/api/future-self",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_future_self():
    return await _proxy_get(f"{MEMU_URL}/memory/future-self")


@app.post("/api/legacy/write",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_legacy_write(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/legacy/write", body)


@app.get("/api/legacy",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_legacy(request: Request):
    return await _proxy_get(f"{MEMU_URL}/memory/legacy", params=dict(request.query_params),
                            fallback={"messages": []})


@app.get("/api/narrative/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_narrative_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/narrative/summary")


# ── P19 Imagination proxies (H1.7: all wrapped) ─────────────────────

@app.post("/api/imagine/counterfactual",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_counterfactual(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/counterfactual", body)


@app.get("/api/imagine/counterfactuals",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_counterfactuals():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/counterfactuals", fallback={"entries": []})


@app.post("/api/imagine/empathize",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_empathize(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/empathize", body)


@app.get("/api/imagine/empathy-map",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_empathy_map():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/empathy-map")


@app.post("/api/imagine/synthesize",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_synthesize(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/synthesize", body)


@app.get("/api/imagine/ideas",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_ideas():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/ideas", fallback={"ideas": []})


@app.post("/api/imagine/thought",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_thought(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/thought", body)


@app.get("/api/imagine/inner-monologue",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_inner_monologue():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/inner-monologue", fallback={"entries": []})


@app.post("/api/imagine/aspire",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_aspire(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/aspire", body)


@app.get("/api/imagine/aspirations",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_aspirations():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/aspirations", fallback={"entries": []})


@app.get("/api/imagine/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_imagination_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/summary")


# ── P20: Conscience & Values proxies (H1.7: all wrapped) ────────────

@app.post("/api/values/learn",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_values_learn(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/values/learn", body)


@app.get("/api/values",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_values():
    return await _proxy_get(f"{MEMU_URL}/memory/values", fallback={"values": []})


@app.post("/api/conscience/check",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_conscience_check(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/conscience/check", body)


@app.get("/api/conscience/audit",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_conscience_audit():
    return await _proxy_get(f"{MEMU_URL}/memory/conscience/audit")


@app.post("/api/loyalty/record",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_loyalty_record(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/loyalty/record", body)


@app.get("/api/loyalty",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_loyalty():
    return await _proxy_get(f"{MEMU_URL}/memory/loyalty", fallback={"entries": []})


@app.post("/api/gratitude/record",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def proxy_gratitude_record(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MEMU_URL}/memory/gratitude/record", body)


@app.get("/api/gratitude",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_gratitude():
    return await _proxy_get(f"{MEMU_URL}/memory/gratitude", fallback={"entries": []})


@app.get("/api/conscience/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_conscience_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/conscience/summary")


# ── P21: Proactive Agent Loop proxies (H1.7: all wrapped) ───────────

@app.get("/api/actions",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_actions():
    return await _proxy_get(f"{MEMU_URL}/memory/actions", fallback={"actions": []})


@app.post("/api/schedule/task",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_schedule_task(body: dict):
    return await _proxy_post(f"{MEMU_URL}/memory/schedule/task", body)


@app.get("/api/schedule/tasks",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_schedule_tasks():
    return await _proxy_get(f"{MEMU_URL}/memory/schedule/tasks", fallback={"tasks": []})


@app.post("/api/schedule/task/{task_id}/cancel",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_cancel_task(task_id: str):
    return await _proxy_post(f"{MEMU_URL}/memory/schedule/task/{task_id}/cancel")


@app.post("/api/reminders/set",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_set_reminder(body: dict):
    return await _proxy_post(f"{MEMU_URL}/memory/reminders/set", body)


@app.get("/api/reminders",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_reminders():
    return await _proxy_get(f"{MEMU_URL}/memory/reminders", fallback={"reminders": []})


@app.post("/api/reminders/{reminder_id}/cancel",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_cancel_reminder(reminder_id: str):
    return await _proxy_post(f"{MEMU_URL}/memory/reminders/{reminder_id}/cancel")


@app.post("/api/briefing/morning",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_morning_briefing():
    return await _proxy_post(f"{MEMU_URL}/memory/briefing/morning")


@app.post("/api/briefing/evening",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_evening_checkin():
    return await _proxy_post(f"{MEMU_URL}/memory/briefing/evening")


@app.get("/api/agent/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_agent_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/agent/summary")


# ── P22 Operator Model proxies (H1.7: all wrapped) ─────────────────

@app.post("/api/echo/analyse",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_echo_analyse(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/echo/analyse", body)


@app.get("/api/echo/history",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_echo_history():
    return await _proxy_get(f"{MEMU_URL}/memory/echo/history", fallback={"entries": []})


@app.post("/api/nudge/escalate",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_nudge_escalate(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/nudge/escalate", body)


@app.get("/api/nudge/ladder",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_nudge_ladder():
    return await _proxy_get(f"{MEMU_URL}/memory/nudge/ladder", fallback={"ladder": {}})


@app.post("/api/cross-mode/scan",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_cross_mode_scan(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/cross-mode/scan", body)


@app.get("/api/cross-mode",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_cross_mode():
    return await _proxy_get(f"{MEMU_URL}/memory/cross-mode", fallback={"insights": []})


@app.post("/api/oracle/predict",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_oracle_predict(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/oracle/predict", body)


@app.get("/api/oracle/chains",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_oracle_chains():
    return await _proxy_get(f"{MEMU_URL}/memory/oracle/chains", fallback={"chains": []})


@app.post("/api/shadow/branch",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_shadow_branch(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/shadow/branch", body)


@app.get("/api/shadow/branches",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_shadow_branches():
    return await _proxy_get(f"{MEMU_URL}/memory/shadow/branches", fallback={"branches": []})


@app.get("/api/operator-model",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def proxy_operator_model():
    return await _proxy_get(f"{MEMU_URL}/memory/operator-model")


# ── J2 Wake + Intent proxies ──────────────────────────────────────────

@app.post("/api/wake/detect",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_wake_detect(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/detect", body)


@app.post("/api/wake/intent",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_wake_intent(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/intent", body)


@app.post("/api/wake/process",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def proxy_wake_process(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/process", body)


# ── Unified App Shell ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _app_shell_html() -> str:
    """Read the shell once. It is a build artefact, not live data."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "app.html")
    with open(html_path, "r", encoding="utf-8") as handle:
        return handle.read()


@app.get("/app")
async def app_shell() -> HTMLResponse:
    """Serve the unified single-page app shell.

    Previously re-read from disk on every request (KAI-DASH-087) — a
    blocking read on the hot path of a page the UI polls.
    """
    return HTMLResponse(_app_shell_html())


# ── Chat proxy — Kai's face ─────────────────────────────────────────
LANGGRAPH_URL = backend_url("LANGGRAPH_URL", "http://agentic:8007")


@app.get("/chat")
async def chat_page() -> HTMLResponse:
    """Serve the chat UI (legacy standalone, redirects to /app)."""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app">')


@app.post("/api/chat",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_IDENTITY))])
async def api_chat_proxy(request: Request):
    """Proxy chat requests to agentic /chat with SSE streaming.

    This keeps the browser talking only to dashboard:8080.
    The agentic service does the actual LLM inference.
    """
    # KAI-DASH-053 — the request body was unbounded.
    # KAI-DASH-053 — the request body was unbounded.
    body = await bounded_json(request)

    async def stream_proxy():
        try:
            async with pooled_client(timeout=httpx.Timeout(180.0, connect=10.0, read=120.0)) as client:
                async with client.stream(
                    "POST",
                    f"{LANGGRAPH_URL}/chat",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    # KAI-DASH-054 — the backend's status was never
                    # checked, so a 500 body was streamed to the browser
                    # as if it were model output.
                    if resp.status_code >= 400:
                        await resp.aread()
                        logger.warning(
                            "chat backend returned %s", resp.status_code)
                        yield _sse_error(
                            f"the assistant service returned "
                            f"{resp.status_code}")
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as exc:
            # KAI-DASH-055 — the exception text went to the browser and
            # disclosed internal hosts and transport detail. It belongs
            # in the log, not the response.
            logger.warning("chat proxy connection failed: %s", exc)
            yield _sse_error("the assistant service is unavailable")

    return StreamingResponse(
        stream_proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


SCREEN_CAPTURE_URL = backend_url("SCREEN_CAPTURE_URL", "http://screen-capture:8059")
AUDIO_URL = backend_url("AUDIO_SERVICE_URL", "http://audio-service:8021")
TTS_URL = backend_url("TTS_SERVICE_URL", "http://tts-service:8030")
BROWSER_AGENT_URL = backend_url("BROWSER_AGENT_URL", "http://browser-agent:8040")
VISION_URL = backend_url("VISION_SERVICE_URL", "http://vision-service:8023")
CLIPBOARD_URL = backend_url("CLIPBOARD_SERVICE_URL", "http://clipboard-service:8024")
FILES_URL = backend_url("FILES_SERVICE_URL", "http://files-service:8025")
NOTIFY_URL = backend_url("NOTIFY_SERVICE_URL", "http://notify-service:8031")
DOC_PARSER_URL = backend_url("DOC_PARSER_URL", "http://document-parser:8032")
MONITOR_URL = backend_url("MONITOR_SERVICE_URL", "http://monitor-service:8033")
BROKER_URL = backend_url("BROKER_URL", "http://broker-bridge:8034")
SYSMETRICS_URL = backend_url("SYSMETRICS_URL", "http://sysmetrics:8035")
WEATHER_SERVICE_URL = backend_url("WEATHER_SERVICE_URL", "http://weather-service:8039")
DOCKER_WATCHER_URL = backend_url("DOCKER_WATCHER_URL", "http://docker-watcher:8041")
AIRQUALITY_URL = backend_url("AIRQUALITY_URL", "http://airquality-service:8042")
CALENDAR_SERVICE_URL = backend_url("CALENDAR_SERVICE_URL", "http://calendar-service:8043")
GIT_WATCHER_URL = backend_url("GIT_WATCHER_URL", "http://git-watcher:8044")
SCREEN_WATCHER_URL = backend_url("SCREEN_WATCHER_URL", "http://screen-watcher:8036")
EMAIL_READER_URL = backend_url("EMAIL_READER_URL", "http://email-reader:8037")
NEWS_FEED_URL = backend_url("NEWS_FEED_URL", "http://news-feed:8038")
_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"})
_DOC_EXTS = frozenset({"pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "dxf", "dwg", "zip"})


@app.post("/api/upload",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_upload(file: UploadFile = File(...)):
    """Route uploaded file to OCR (images) or document parser (PDF, Office, CAD, ZIP).

    Returns JSON with a 'text' field containing extracted content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # KAI-DASH-045 — the limit used to be checked after the whole body
    # was already in memory, which protected nothing.
    data = await bounded_upload(file, _UPLOAD_MAX_BYTES)

    ext = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "").lower()

    if ext in _IMAGE_EXTS:
        target_url = f"{SCREEN_CAPTURE_URL}/capture/file"
        service_name = "OCR"
        content_type = safe_content_type(file.content_type,
                                         ALLOWED_IMAGE_TYPES, "image/png")
    elif ext in _DOC_EXTS:
        target_url = f"{DOC_PARSER_URL}/parse"
        service_name = "document parser"
        content_type = safe_content_type(file.content_type, ALLOWED_DOC_TYPES,
                                         "application/octet-stream")
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext or '(none)'}")

    try:
        async with pooled_client(timeout=60.0) as client:
            resp = await client.post(
                target_url,
                files={"file": (safe_filename(file.filename), data, content_type)},
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status,
                                detail=client_error(exc, f"{service_name} service rejected the file"))
        raise HTTPException(status_code=502,
                            detail=client_error(exc, f"{service_name} service error"))
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=client_error(exc, f"{service_name} service unreachable"),
        )


@app.post("/api/tts/synthesize",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_tts_synthesize(request: Request):
    """Proxy text-to-speech synthesis to the TTS service.

    Accepts JSON: {text, voice?, rate?, volume?}
    Returns audio/mpeg when TTS service is available, 503 when offline.
    """
    from fastapi.responses import Response as FastAPIResponse
    body = await bounded_json(request)
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(f"{TTS_URL}/synthesize", json=body)
            resp.raise_for_status()
            # Forcing audio/mpeg regardless of what the backend actually
            # sent (KAI-DASH-089) mislabels any other format and hands the
            # browser a file its type does not describe.
            return FastAPIResponse(
                content=bounded_response(resp.content, "tts-service"),
                media_type=_safe_media_type(
                    resp.headers.get("content-type"), ALLOWED_AUDIO_TYPES,
                    "application/octet-stream"),
                headers={"X-Voice": resp.headers.get("X-Voice", "")},
            )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "TTS service rejected request:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "TTS service error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "TTS service unreachable:"))


@app.post("/api/audio/transcribe",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_audio_transcribe(file: UploadFile = File(...)):
    """Receive an audio blob from the browser MediaRecorder and return a transcript.

    Proxies to the audio-service Whisper backend. Degrades to 503 when unavailable.
    """
    data = await bounded_upload(file)  # KAI-DASH-046
    try:
        async with pooled_client(timeout=60.0) as client:
            resp = await client.post(
                f"{AUDIO_URL}/capture/file",
                files={"file": (safe_filename(file.filename, "audio.webm"), data,
                       safe_content_type(file.content_type,
                                         ALLOWED_AUDIO_TYPES, "audio/webm"))},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Audio service rejected file:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Audio service error:"))
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=client_error(exc, "Audio service unreachable:"),
        )


# ── Browser Agent proxies ────────────────────────────────────────────────────

@app.post("/api/browser/navigate",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_browser_navigate(request: Request):
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/navigate", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Browser agent rejected request:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Browser agent error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Browser agent unreachable:"))


@app.post("/api/browser/scrape",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_browser_scrape(request: Request):
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/scrape", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Browser agent rejected request:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Browser agent error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Browser agent unreachable:"))


@app.post("/api/browser/run",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_browser_run(request: Request):
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=60.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/run", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Browser agent rejected request:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Browser agent error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Browser agent unreachable:"))


@app.get("/api/browser/screenshot",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_browser_screenshot():
    from fastapi.responses import Response as FastAPIResponse
    try:
        async with pooled_client(timeout=15.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/screenshot")
            resp.raise_for_status()
            return FastAPIResponse(
                content=bounded_response(resp.content, "browser-agent"),
                media_type=_safe_media_type(
                    resp.headers.get("content-type"), ALLOWED_IMAGE_TYPES,
                    "application/octet-stream"),
            )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=client_error(exc, "Browser agent error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Browser agent unreachable:"))


# ── Vision / camera proxies ───────────────────────────────────────────────────

@app.post("/api/vision/analyze",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_vision_analyze(file: UploadFile = File(...)):
    data = await bounded_upload(file)  # KAI-DASH-047
    try:
        async with pooled_client(timeout=10.0) as client:
            resp = await client.post(
                f"{VISION_URL}/analyze/frame",
                files={"file": (safe_filename(file.filename, "frame.jpg"), data,
                       safe_content_type(file.content_type,
                                         ALLOWED_IMAGE_TYPES, "image/jpeg"))},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Vision service rejected frame:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Vision service error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Vision service unreachable:"))


@app.post("/api/vision/presence",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_vision_presence(file: UploadFile = File(...)):
    data = await bounded_upload(file)  # KAI-DASH-047
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.post(
                f"{VISION_URL}/analyze/presence",
                files={"file": (safe_filename(file.filename, "frame.jpg"), data,
                       safe_content_type(file.content_type,
                                         ALLOWED_IMAGE_TYPES, "image/jpeg"))},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Vision service rejected frame:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Vision service error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Vision service unreachable:"))


# ── Clipboard proxies ─────────────────────────────────────────────────────────

@app.post("/api/clipboard/push",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_clipboard_push(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{CLIPBOARD_URL}/push", body=body, fallback={"ok": False})


@app.get("/api/clipboard/latest",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_clipboard_latest():
    return await _proxy_get(f"{CLIPBOARD_URL}/latest", fallback={"content": "", "id": None})


@app.get("/api/clipboard/history",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_clipboard_history(limit: int = Query(20, ge=1, le=500)):
    return await _proxy_get(f"{CLIPBOARD_URL}/history", params={"limit": limit}, fallback={"entries": []})


@app.delete("/api/clipboard/history",
            dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_clipboard_clear():
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.delete(f"{CLIPBOARD_URL}/history")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('clipboard-service', str(exc), {"cleared": False})


# ── File Watcher proxies ───────────────────────────────────────────────────────

@app.get("/api/files/events",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_files_events(limit: int = Query(50, ge=1, le=500), event_type: str = ""):
    params: dict = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    return await _proxy_get(f"{FILES_URL}/events", params=params, fallback={"events": []})


@app.get("/api/files/watching",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_files_watching():
    return await _proxy_get(f"{FILES_URL}/watching", fallback={"directories": []})


@app.post("/api/files/watch",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_files_watch(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{FILES_URL}/watch", body=body, fallback={"ok": False})


# ── Notify proxies ─────────────────────────────────────────────────────────────

@app.post("/api/notify/send",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_notify_send(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{NOTIFY_URL}/notify", body=body, fallback={"ok": False})


@app.get("/api/notify/pending",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_notify_pending(unread_only: bool = True):
    return await _proxy_get(f"{NOTIFY_URL}/pending", params={"unread_only": unread_only},
                            fallback={"notifications": []})


@app.delete("/api/notify/pending/{notification_id}",
            dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_notify_dismiss(notification_id: int):
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.delete(f"{NOTIFY_URL}/pending/{notification_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('notify-service', str(exc), {"cleared": False})


@app.delete("/api/notify/pending",
            dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_notify_dismiss_all():
    try:
        async with pooled_client(timeout=5.0) as client:
            resp = await client.delete(f"{NOTIFY_URL}/pending")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('notify-service', str(exc), {"cleared": False})


# ── Monitor proxies ───────────────────────────────────────────────────────────

@app.get("/api/monitor/rules",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_monitor_rules():
    return await _proxy_get(f"{MONITOR_URL}/rules", fallback={"rules": [], "total": 0})


@app.post("/api/monitor/rules",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_add_rule(request: Request):
    body = await bounded_json(request)
    return await _proxy_post(f"{MONITOR_URL}/rules", body=body, fallback={"ok": False})


@app.delete("/api/monitor/rules/{rule_id}",
            dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_delete_rule(rule_id: str):
    try:
        async with pooled_client(timeout=10.0) as client:
            resp = await client.delete(f"{MONITOR_URL}/rules/{rule_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('monitor-service', str(exc), {"ok": False})


@app.post("/api/monitor/rules/{rule_id}/enable",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_enable_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/enable", fallback={"ok": False})


@app.post("/api/monitor/rules/{rule_id}/disable",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_disable_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/disable", fallback={"ok": False})


@app.post("/api/monitor/rules/{rule_id}/check",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_check_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/check", fallback={"ok": False})


@app.get("/api/monitor/alerts",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_monitor_alerts(limit: int = Query(50, ge=1, le=500)):
    return await _proxy_get(f"{MONITOR_URL}/alerts", params={"limit": limit}, fallback={"alerts": [], "total": 0})


@app.delete("/api/monitor/alerts",
            dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_monitor_clear_alerts():
    try:
        async with pooled_client(timeout=10.0) as client:
            resp = await client.delete(f"{MONITOR_URL}/alerts")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return degraded_response('monitor-service', str(exc), {"ok": False})


@app.get("/api/monitor/status",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_monitor_status():
    return await _proxy_get(f"{MONITOR_URL}/status", fallback={})


# ── Browser search proxy ───────────────────────────────────────────────────────

@app.post("/api/browser/search",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_browser_search(request: Request):
    body = await bounded_json(request)
    try:
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/search", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=client_error(exc, "Browser agent rejected request:"))
        raise HTTPException(status_code=502, detail=client_error(exc, "Browser agent error:"))
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=client_error(exc, "Browser agent unreachable:"))


# ── Broker bridge proxies ─────────────────────────────────────────────────────

@app.get("/api/broker/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_health():
    return await _proxy_get(f"{BROKER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/broker/ticker/{symbol}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_ticker(symbol: str):
    symbol = safe_symbol(symbol)
    return await _proxy_get(f"{BROKER_URL}/ticker/{symbol}", fallback={})


@app.get("/api/broker/ticker",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_tickers(symbols: str = ""):
    params = {"symbols": symbols} if symbols else {}
    return await _proxy_get(f"{BROKER_URL}/ticker", params=params, fallback={"tickers": []})


@app.get("/api/broker/balance",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_broker_balance():
    return await _proxy_get(f"{BROKER_URL}/balance", fallback={"assets": []})


@app.get("/api/broker/positions",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_broker_positions():
    return await _proxy_get(f"{BROKER_URL}/positions", fallback={"positions": []})


@app.get("/api/broker/orders",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_broker_orders(symbol: str = ""):
    params = {"symbol": symbol} if symbol else {}
    return await _proxy_get(f"{BROKER_URL}/orders", params=params, fallback={"orders": []})


@app.get("/api/broker/pnl",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_broker_pnl():
    return await _proxy_get(f"{BROKER_URL}/pnl/summary",
                            fallback={"total_unrealized_pnl": None, "positions": []})


@app.get("/api/broker/templates",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_templates():
    return await _proxy_get(f"{BROKER_URL}/templates", fallback={"templates": []})


@app.post("/api/broker/watch",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_broker_watch(request: Request):
    """Create a monitor rule for a position from the Broker tab Quick Watch button."""
    body = await bounded_json(request)
    # KAI-DASH-095 — the symbol reached an outbound URL unvalidated and
    # the threshold was never checked at all, so NaN or a string became a
    # monitor rule that could never fire meaningfully.
    symbol = safe_symbol(body.get("symbol", ""))
    threshold = _as_number(body.get("threshold"), float)
    if threshold is None:
        raise HTTPException(
            status_code=400,
            detail="threshold must be a finite number",
        )
    rule = {
        "source": {
            "type": "http",
            "url": f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT",
            "field": "price",
        },
        "condition": {"op": "changed"},
        "actions": [
            {"type": "notify", "message": f"{symbol} price changed"},
            {"type": "tts", "text": f"{symbol} price has changed"},
        ],
        "interval": 60,
        "cooldown": 300,
    }
    if threshold is not None:
        rule["condition"] = {"op": "lt", "threshold": float(threshold)}
    monitor_url = backend_url("MONITOR_SERVICE_URL", "http://monitor-service:8033")
    return await _proxy_post(f"{monitor_url}/rules", body=rule, fallback={"ok": False})


# ── Sysmetrics proxies ────────────────────────────────────────────────────────

@app.get("/api/sysmetrics/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_sysmetrics_health():
    return await _proxy_get(f"{SYSMETRICS_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/sysmetrics/snapshot",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_sysmetrics_snapshot():
    return await _proxy_get(f"{SYSMETRICS_URL}/snapshot", fallback={})


@app.get("/api/sysmetrics/processes",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_sysmetrics_processes():
    return await _proxy_get(f"{SYSMETRICS_URL}/processes", fallback={"processes": []})


# ── Screen-watcher proxies ────────────────────────────────────────────────────

@app.get("/api/screen-watcher/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_screen_watcher_health():
    return await _proxy_get(f"{SCREEN_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/screen-watcher/status",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_screen_watcher_status():
    return await _proxy_get(f"{SCREEN_WATCHER_URL}/status", fallback={})


@app.post("/api/screen-watcher/watch/start",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_screen_watcher_start(request: Request):
    body = await bounded_json(request) if request.headers.get("content-type") == "application/json" else {}
    return await _proxy_post(f"{SCREEN_WATCHER_URL}/watch/start", body=body, fallback={"ok": False})


@app.post("/api/screen-watcher/watch/stop",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_screen_watcher_stop():
    return await _proxy_post(f"{SCREEN_WATCHER_URL}/watch/stop", body={}, fallback={"ok": False})


# ── Email-reader proxies ──────────────────────────────────────────────────────

@app.get("/api/email/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_email_health():
    return await _proxy_get(f"{EMAIL_READER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/email/inbox",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_email_inbox(limit: int = Query(20, ge=1, le=500)):
    return await _proxy_get(f"{EMAIL_READER_URL}/inbox", params={"limit": limit}, fallback={"messages": []})


@app.get("/api/email/unread",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_SENSITIVE))])
async def api_email_unread():
    return await _proxy_get(f"{EMAIL_READER_URL}/unread", fallback={"unread_count": 0, "sample": []})


@app.post("/api/email/refresh",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_email_refresh():
    return await _proxy_post(f"{EMAIL_READER_URL}/refresh", body={}, fallback={"ok": False})


# ── News-feed proxies ─────────────────────────────────────────────────────────

@app.get("/api/news/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_news_health():
    return await _proxy_get(f"{NEWS_FEED_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/news/articles",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_news_articles(limit: int = Query(20, ge=1, le=500), tag: str = "", since_minutes: int = 0):
    params: dict = {"limit": limit}
    if tag:
        params["tag"] = tag
    if since_minutes > 0:
        params["since_minutes"] = since_minutes
    return await _proxy_get(f"{NEWS_FEED_URL}/articles", params=params, fallback={"articles": []})


@app.get("/api/news/search",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_news_search(q: str = "", limit: int = Query(10, ge=1, le=500)):
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    return await _proxy_get(f"{NEWS_FEED_URL}/search", params={"q": q, "limit": limit}, fallback={"results": []})


@app.post("/api/news/refresh",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_news_refresh():
    return await _proxy_post(f"{NEWS_FEED_URL}/refresh", body={}, fallback={"ok": False})


@app.get("/api/news/feeds",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_news_feeds():
    return await _proxy_get(f"{NEWS_FEED_URL}/feeds", fallback={"feeds": []})


# ── Broker market depth extensions ────────────────────────────────────────────

@app.get("/api/broker/depth/{symbol}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_depth(symbol: str, limit: int = Query(20, ge=1, le=500)):
    symbol = safe_symbol(symbol)
    return await _proxy_get(f"{BROKER_URL}/depth/{symbol}", params={"limit": limit}, fallback={})


@app.get("/api/broker/stats/{symbol}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_stats(symbol: str):
    symbol = safe_symbol(symbol)
    return await _proxy_get(f"{BROKER_URL}/stats/24hr/{symbol}", fallback={})


@app.get("/api/broker/trades/{symbol}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_broker_trades(symbol: str, limit: int = Query(20, ge=1, le=500)):
    symbol = safe_symbol(symbol)
    return await _proxy_get(f"{BROKER_URL}/trades/{symbol}", params={"limit": limit}, fallback={"trades": []})


@app.get("/api/broker/stocks/{symbol}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def broker_stocks(symbol: str):
    symbol = safe_symbol(symbol)
    async with pooled_client(timeout=15.0) as client:
        r = await client.get(f"{BROKER_URL}/stocks/{symbol}")
        r.raise_for_status()
        return r.json()


@app.get("/api/broker/forex/{pair}",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def broker_forex(pair: str):
    pair = safe_symbol(pair, "pair")
    async with pooled_client(timeout=15.0) as client:
        r = await client.get(f"{BROKER_URL}/forex/{pair}")
        r.raise_for_status()
        return r.json()


# ── Weather service proxies ───────────────────────────────────────────────────

@app.get("/api/weather/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_weather_health():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/weather/current",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_weather_current():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/current", fallback={})


@app.get("/api/weather/forecast",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_weather_forecast():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/forecast", fallback={"forecast": []})


@app.get("/api/weather/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_weather_summary():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/summary", fallback={"summary": "Weather unavailable."})


# ── Docker-watcher proxies ────────────────────────────────────────────────────

@app.get("/api/docker/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_docker_health():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/docker/containers",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_docker_containers():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/containers", fallback={"containers": [], "total": 0})


@app.get("/api/docker/unhealthy",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_docker_unhealthy():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/unhealthy", fallback={"unhealthy": [], "count": 0})


@app.get("/api/docker/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_docker_summary():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/summary", fallback={"summary": "Docker data unavailable."})


# ── Air quality proxies ───────────────────────────────────────────────────────

@app.get("/api/airquality/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_airquality_health():
    return await _proxy_get(f"{AIRQUALITY_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/airquality/current",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_airquality_current():
    return await _proxy_get(f"{AIRQUALITY_URL}/current", fallback={})


@app.get("/api/airquality/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_airquality_summary():
    return await _proxy_get(f"{AIRQUALITY_URL}/summary", fallback={"summary": "Air quality unavailable."})


# ── Calendar service proxies ──────────────────────────────────────────────────

@app.get("/api/calendar/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_calendar_health():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/calendar/events/today",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_calendar_today():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/events/today", fallback={"events": []})


@app.get("/api/calendar/events/upcoming",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_calendar_upcoming(days: int = Query(7, ge=1, le=365)):
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/events/upcoming", params={"days": days},
                            fallback={"events": []})


@app.get("/api/calendar/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_calendar_summary():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/summary", fallback={"summary": "Calendar not configured."})


@app.post("/api/calendar/refresh",
          dependencies=[Depends(require_dashboard_auth(Scope.WRITE_EXTERNAL))])
async def api_calendar_refresh():
    return await _proxy_post(f"{CALENDAR_SERVICE_URL}/refresh", body={}, fallback={"ok": False})


# ── Git-watcher proxies ───────────────────────────────────────────────

@app.get("/api/git/health",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_git_health():
    return await _proxy_get(f"{GIT_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/git/repos",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_git_repos():
    return await _proxy_get(f"{GIT_WATCHER_URL}/repos", fallback={"repos": [], "count": 0})


@app.get("/api/git/dirty",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_git_dirty():
    return await _proxy_get(f"{GIT_WATCHER_URL}/dirty", fallback={"repos": [], "count": 0})


@app.get("/api/git/summary",
         dependencies=[Depends(require_dashboard_auth(Scope.READ_OPERATIONAL))])
async def api_git_summary():
    return await _proxy_get(
        f"{GIT_WATCHER_URL}/summary", fallback={"summary": "Git data unavailable."}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
