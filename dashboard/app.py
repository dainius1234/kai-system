

from __future__ import annotations
import asyncio
from datetime import datetime
import json as _json
import os
from typing import Any, Dict, List

import httpx
import redis.asyncio as aioredis
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

from common.resilience import resilient_call
from common.runtime import AuditStream, ErrorBudget, detect_device, setup_json_logger

try:
    from common.policy import policy_hash, policy_version
except Exception:
    policy_hash = "unavailable"
    policy_version = "unknown"

logger = setup_json_logger("dashboard", os.getenv("LOG_PATH", "/tmp/dashboard.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.4.0")

# mount static UI stub
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
VERIFIER_URL   = os.getenv("VERIFIER_URL", "http://verifier:8052")
# Store-maintenance reads (stats/search-by-category/quarantine listing) live on
# memu-core-introspect, split out from memu-core's hot path — see DECISIONS.md D21.
MEMU_INTROSPECT_URL = os.getenv("MEMU_INTROSPECT_URL", "http://memu-core-introspect:8009")
budget = ErrorBudget(window_seconds=300)


# ── H1.7 + H2: Safe proxy helpers with retry + circuit breaker ──────
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

@app.get("/api/nudges")
async def api_nudges():
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{memu_url}/memory/proactive")
            resp.raise_for_status()
            payload = resp.json()
            return {"nudges": payload.get("nudges", [])}
    except Exception:
        return {"nudges": []}


@app.get("/api/backup-status")
async def api_backup_status():
    backup_url = os.getenv("BACKUP_SERVICE_URL", "http://backup-service:8054")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{backup_url}/health")
            resp.raise_for_status()
            # Optionally, fetch latest backup file info
            # For now, just return timestamp
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            return {"status": f"{now} (service healthy)"}
    except Exception:
        return {"status": "Backup service unreachable"}


@app.get("/api/corrections")
async def api_corrections():
    verifier_url = os.getenv("VERIFIER_URL", "http://verifier:8052")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{verifier_url}/metrics")
            resp.raise_for_status()
            payload = resp.json()
            verdicts = payload.get("verdicts", {})
            # Build correction history from REPAIR/FAIL_CLOSED
            corrections = []
            for verdict, count in verdicts.items():
                if verdict in ("REPAIR", "FAIL_CLOSED"):
                    corrections.append({
                        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "summary": f"{verdict}: {count} recent corrections"
                    })
            return {"corrections": corrections}
    except Exception:
        return {"corrections": []}

audit = AuditStream("dashboard", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")

SUPERVISOR_URL = os.getenv("SUPERVISOR_URL", "http://supervisor:8051")

NODES: Dict[str, str] = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": os.getenv("MEMU_URL", "http://memu-core:8001") + "/health",
    "heartbeat": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
    "supervisor": f"{SUPERVISOR_URL}/health",
    "verifier": os.getenv("VERIFIER_URL", "http://verifier:8052") + "/health",
    "fusion-engine": os.getenv("FUSION_URL", "http://fusion-engine:8053") + "/health",
    "memory-compressor": os.getenv("MEMORY_COMPRESSOR_URL", "http://memory-compressor:8057") + "/health",
    "ledger-worker": os.getenv("LEDGER_WORKER_URL", "http://ledger-worker:8056") + "/health",
    "metrics-gateway": os.getenv("METRICS_GATEWAY_URL", "http://metrics-gateway:8058") + "/health",
}
_agentic_url = os.getenv("LANGGRAPH_URL", "")
if _agentic_url:
    NODES["agentic"] = _agentic_url + "/health"
_executor_url = os.getenv("EXECUTOR_URL", "")
if _executor_url:
    NODES["executor"] = _executor_url + "/health"
_wake_url = os.getenv("WAKE_URL", "")
if _wake_url:
    NODES["wake-service"] = _wake_url + "/health"

NO_GO_GRACE_REQUESTS = int(os.getenv("NO_GO_GRACE_REQUESTS", "20"))
MAX_ERROR_RATIO = float(os.getenv("MAX_ERROR_RATIO", "0.05"))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise


async def fetch_status() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for name, url in NODES.items():
            try:
                resp = await client.get(url, timeout=2.0)
                resp.raise_for_status()
                results[name] = {"status": "ok", "details": resp.json()}
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "down", "error": str(exc)}
    return results


async def build_go_no_go_report() -> Dict[str, Any]:
    reasons: List[str] = []
    statuses = await fetch_status()
    down_nodes = [name for name, payload in statuses.items() if payload.get("status") != "ok"]
    if down_nodes:
        reasons.append(f"Critical services are down: {', '.join(down_nodes)}")

    try:
        async with httpx.AsyncClient() as client:
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

    ledger_count = int(ledger_stats.get("count", 0))
    if ledger_count < NO_GO_GRACE_REQUESTS:
        reasons.append(
            f"Not enough proof yet ({ledger_count}/{NO_GO_GRACE_REQUESTS} gate decisions observed)."
        )

    error_ratio = float(metrics.get("error_ratio", 0.0))
    if error_ratio > MAX_ERROR_RATIO:
        reasons.append(
            f"Recent API error ratio is too high ({error_ratio:.1%} > {MAX_ERROR_RATIO:.1%})."
        )

    go = len(reasons) == 0
    return {
        "decision": "GO" if go else "NO_GO",
        "trust_status": "trusted" if go else "prove-first",
        "summary": "System looks stable enough to proceed." if go else "Hold execution until blockers are fixed.",
        "checks": {
            "required_mode": "WORK",
            "current_mode": mode,
            "minimum_gate_decisions": NO_GO_GRACE_REQUESTS,
            "current_gate_decisions": ledger_count,
            "max_error_ratio": MAX_ERROR_RATIO,
            "current_error_ratio": error_ratio,
            "down_nodes": down_nodes,
        },
        "reasons": reasons,
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "tool_gate_url": TOOL_GATE_URL,
        "policy_version": policy_version,
        "policy_hash": policy_hash,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/")
async def index() -> Dict[str, object]:
    statuses = await fetch_status()
    alive_nodes = [name for name, payload in statuses.items() if payload.get("status") == "ok"]
    ledger_size = 0
    memory_count = 0
    try:
        async with httpx.AsyncClient() as client:
            ledger_size = int((await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)).json().get("count", 0))
            memory_count = int((await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats", timeout=2.0)).json().get("records", 0))
    except Exception:
        logger.warning("Failed to fetch ledger/memory stats for index")

    go_no_go = await build_go_no_go_report()
    tool_gate_health = statuses.get("tool-gate", {}).get("details", {})
    policy_mode = str(tool_gate_health.get("mode", "PUB")).upper()

    # v7: fetch breaker states, quarantine count, verifier stats
    breaker_states: Dict[str, Any] = {}
    quarantine_count = 0
    verifier_stats: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # breakers from supervisor
            br_resp = await client.get(f"{SUPERVISOR_URL}/breakers")
            if br_resp.status_code == 200:
                breaker_states = br_resp.json()
    except Exception:
        pass
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            # quarantine count
            q_resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/quarantine/list")
            if q_resp.status_code == 200:
                quarantine_count = q_resp.json().get("count", 0)
    except Exception:
        pass
    try:
        verifier_url = os.getenv("VERIFIER_URL", "http://verifier:8052")
        async with httpx.AsyncClient(timeout=2.0) as client:
            v_resp = await client.get(f"{verifier_url}/metrics")
            if v_resp.status_code == 200:
                verifier_stats = v_resp.json()
    except Exception:
        pass
    core_nodes = ["tool-gate", "memu-core"]
    if _executor_url:
        core_nodes.append("executor")
    core_ready = all(node in alive_nodes for node in core_nodes) and ledger_size >= 0 and memory_count >= 0
    return {
        "service": "dashboard",
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "tool_gate_url": TOOL_GATE_URL,
        "core_ready": core_ready,
        "alive_nodes": alive_nodes,
        "node_status": statuses,
        "ledger_size": ledger_size,
        "memory_count": memory_count,
        "policy_mode": policy_mode,
        "device_summary": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "go_no_go": go_no_go,
        "policy_version": policy_version,
        "policy_hash": policy_hash,
        "breaker_states": breaker_states,
        "quarantine_count": quarantine_count,
        "verifier_stats": verifier_stats,
    }


@app.get("/go-no-go")
async def go_no_go() -> Dict[str, Any]:
    return await build_go_no_go_report()


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


@app.get("/fleet")
async def fleet() -> Dict[str, Any]:
    """Proxy the supervisor's fleet health view into the dashboard."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{SUPERVISOR_URL}/status")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"fleet": "unknown", "error": "supervisor unreachable"}


@app.get("/readiness")
async def readiness() -> Dict[str, Any]:
    payload = await index()
    if not payload["core_ready"]:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "core_ready": False, "reasons": payload["go_no_go"]["reasons"]})
    return {"status": "ready", "core_ready": True}


# ── P8: Thinking Pathways — intelligence proxy endpoints ─────────────
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")
FINANCIAL_URL = os.getenv("FINANCIAL_URL", "http://financial-awareness:8063")
WAKE_URL = os.getenv("WAKE_URL", "http://wake-service:8022")


@app.get("/thinking")
async def thinking_page() -> HTMLResponse:
    """Serve the Thinking Pathways (legacy standalone, redirects to /app)."""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app">')


@app.get("/api/thinking")
async def api_thinking():
    """Fetch latest episode data from agentic for thinking pathway visualization."""
    agentic_url = os.getenv("LANGGRAPH_URL", "http://agentic:8007")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{agentic_url}/episodes/recall",
                json={"user_id": "keeper", "days": 7},
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
    except Exception:
        return {"status": "unavailable", "total_episodes": 0, "pathways": []}


@app.get("/api/tempo")
async def api_tempo():
    """Proxy operator tempo from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/tempo")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "tempo": "unknown"}


@app.get("/api/boundary")
async def api_boundary():
    """Proxy knowledge boundary map from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/boundary")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "zones": []}


@app.get("/api/silence")
async def api_silence():
    """Proxy silence-as-signal data from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/silence")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "silence_topics": []}


@app.get("/api/self-assessment")
async def api_self_assessment():
    """Proxy temporal self-model from heartbeat."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{HEARTBEAT_URL}/self-assessment")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable"}


@app.post("/api/dream")
async def api_dream():
    """Trigger a dream consolidation cycle via agentic-introspect."""
    introspect_url = os.getenv("AGENTIC_INTROSPECT_URL", "http://agentic-introspect:8023")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{introspect_url}/dream")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "message": "Cannot reach agentic-introspect for dream cycle"}


@app.get("/api/ledger-stats")
async def api_ledger_stats():
    """Proxy ledger statistics from ledger-worker."""
    ledger_url = os.getenv("LEDGER_WORKER_URL", "http://ledger-worker:8056")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ledger_url}/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "total_entries": 0}


# ── Redis pub/sub — real-time event streaming ────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Channels that the dashboard subscribes to
_EVENT_CHANNELS = [
    "kai:health",          # service up/down events
    "kai:episode",         # new episode recorded
    "kai:breaker",         # circuit breaker state change
    "kai:memory",          # memory store changes
]


async def _publish_event(channel: str, data: dict) -> None:
    """Publish a JSON event to a Redis channel (fire-and-forget)."""
    try:
        r = aioredis.from_url(REDIS_URL, decode_responses=True)
        await r.publish(channel, _json.dumps(data))
        await r.aclose()
    except Exception:
        logger.debug("Redis publish to %s failed (non-critical)", channel)


@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events stream backed by Redis pub/sub.

    The dashboard JS connects via EventSource('/api/events') and receives
    real-time updates instead of polling.
    """
    async def event_generator():
        try:
            r = aioredis.from_url(REDIS_URL, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe(*_EVENT_CHANNELS)
        except Exception:
            yield f"data: {_json.dumps({'channel': 'error', 'error': 'redis unavailable'})}\n\n"
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
                    payload = {
                        "channel": msg["channel"],
                        "data": _json.loads(msg["data"]) if isinstance(msg["data"], str) else msg["data"],
                    }
                    yield f"data: {_json.dumps(payload)}\n\n"
                else:
                    # keepalive heartbeat every 15s
                    yield f"data: {_json.dumps({'channel': 'heartbeat', 'ts': datetime.utcnow().isoformat()})}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(*_EVENT_CHANNELS)
            await pubsub.aclose()
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/security-audit")
async def api_security_audit():
    """Proxy security self-hacking audit from agentic-introspect."""
    introspect_url = os.getenv("AGENTIC_INTROSPECT_URL", "http://agentic-introspect:8023")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{introspect_url}/security/audit")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "findings": [], "risk_score": -1}


# ── P16 API proxies ─────────────────────────────────────────────────

@app.get("/api/goals")
async def api_goals():
    """Proxy Ohana goals from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/goals")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "goals": []}


@app.post("/api/goals")
async def api_goals_create(request: Request):
    """Proxy create goal to memu-core."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/goals", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "error", "detail": "Cannot reach memu-core"}


@app.post("/api/goals/update")
async def api_goals_update(request: Request):
    """Proxy update goal progress to memu-core."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/goals/update", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "error", "detail": "Cannot reach memu-core"}


@app.get("/api/drift")
async def api_drift():
    """Proxy drift detection from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/drift")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable"}


@app.get("/api/memories")
async def api_memories(query: str = "", category: str = "", top_k: int = 20):
    """Browse memories — search or list by category."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if query:
                resp = await client.get(f"{MEMU_URL}/memory/retrieve", params={"query": query, "top_k": top_k})
            elif category:
                resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/search-by-category", params={"category": category, "top_k": top_k})
            else:
                resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "memories": []}


@app.get("/api/memory/stats")
async def api_memory_stats():
    """Proxy memory statistics from memu-core-introspect."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_INTROSPECT_URL}/memory/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable"}


@app.get("/api/memories/recent")
async def api_memories_recent(top_k: int = 30):
    """Browse recent memories for Diary tab (recency-weighted retrieve)."""
    raw = await _proxy_get(
        f"{MEMU_URL}/memory/retrieve",
        params={"query": "memories thoughts observations experiences", "user_id": "keeper", "top_k": top_k},
        fallback=[],
    )
    records = raw if isinstance(raw, list) else raw.get("records", raw.get("memories", []))
    if not isinstance(records, list):
        records = []
    return {"records": records, "count": len(records)}


@app.get("/api/memory/graph-data")
async def api_memory_graph_data(top_k: int = 80, query: str = "memories experiences observations"):
    """Return recent memories formatted as {nodes, links} for the D3 force-graph tab."""
    raw = await _proxy_get(
        f"{MEMU_URL}/memory/retrieve",
        params={"query": query, "user_id": "keeper", "top_k": top_k},
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


@app.get("/api/finance/summary")
async def api_finance_summary():
    """Proxy CIS/VAT/tax financial summary from the financial-awareness service (P29)."""
    return await _proxy_get(f"{FINANCIAL_URL}/finance/summary", fallback={
        "status": "unavailable",
        "cis_summary": {},
        "vat_position": {},
        "tax_estimate": {},
        "invoices": [],
    })


@app.get("/api/finance/cis")
async def api_finance_cis():
    """Proxy CIS YTD summary from the financial-awareness service."""
    return await _proxy_get(f"{FINANCIAL_URL}/finance/cis/summary", fallback={"status": "unavailable"})


@app.post("/api/finance/cis/record")
async def api_finance_cis_record(request: Request):
    """Proxy CIS payment record creation to the financial-awareness service."""
    body = await request.json()
    return await _proxy_post(f"{FINANCIAL_URL}/finance/cis/record", body=body, fallback={"status": "unavailable"})


AGENTIC_URL = os.getenv("LANGGRAPH_URL", "http://agentic:8007")


@app.get("/api/soul")
async def api_soul_get():
    """Return current SOUL.md content from agentic."""
    return await _proxy_get(f"{AGENTIC_URL}/soul", fallback={"status": "unavailable", "content": ""})


@app.post("/api/soul")
async def api_soul_post(request: Request):
    """Update SOUL.md content via agentic."""
    body = await request.json()
    return await _proxy_post(f"{AGENTIC_URL}/soul", body=body, fallback={"status": "unavailable"})


@app.get("/api/agents-registry")
async def api_agents_registry_get():
    """Return current AGENTS.md content from agentic."""
    return await _proxy_get(f"{AGENTIC_URL}/agents-registry", fallback={"status": "unavailable", "content": ""})


@app.post("/api/agents-registry")
async def api_agents_registry_post(request: Request):
    """Update AGENTS.md content via agentic."""
    body = await request.json()
    return await _proxy_post(f"{AGENTIC_URL}/agents-registry", body=body, fallback={"status": "unavailable"})


@app.post("/api/pii/scan")
async def api_pii_scan(request: Request):
    """Scan text for PII (and optionally redact) via the verifier service."""
    body = await request.json()
    return await _proxy_post(
        f"{VERIFIER_URL}/redact",
        body={"text": body.get("text", ""), "auto_redact": body.get("auto_redact", True)},
        fallback={"status": "unavailable", "pii_found": {}, "total_pii": 0},
    )


@app.get("/api/struggle")
async def api_struggle(session_id: str = "default"):
    """Proxy struggle detection from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/struggle", params={"session_id": session_id})
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable", "struggle_score": 0}


@app.post("/api/feedback")
async def api_feedback(request: Request):
    """Proxy feedback rating to memu-core."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{MEMU_URL}/memory/feedback", json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "error", "detail": "Cannot reach memu-core"}


@app.get("/api/feedback/stats")
async def api_feedback_stats():
    """Proxy feedback stats from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/feedback/stats")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"status": "unavailable"}


@app.get("/api/logs")
async def api_logs(limit: int = 100, level: str = "", since: float = 0):
    """Aggregate logs from memu-core (and potentially other services)."""
    all_logs: list = []
    params: dict = {"limit": limit}
    if level:
        params["level"] = level
    if since:
        params["since"] = since

    # Collect from memu-core
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/logs", params=params)
            if resp.status_code == 200:
                data = resp.json()
                all_logs.extend(data.get("entries", []))
    except Exception:
        pass

    # Collect from agentic
    agentic_url = os.getenv("LANGGRAPH_URL", "http://agentic:8007")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{agentic_url}/logs", params=params)
            if resp.status_code == 200:
                data = resp.json()
                all_logs.extend(data.get("entries", []))
    except Exception:
        pass

    # Sort all by timestamp (most recent first)
    all_logs.sort(key=lambda x: x.get("time", 0), reverse=True)

    return {
        "status": "ok",
        "count": len(all_logs[:limit]),
        "entries": all_logs[:limit],
    }


# ── P17: Emotional Intelligence Proxies (H1.7: all wrapped) ──────────

@app.post("/api/emotion/record")
async def proxy_emotion_record(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/emotion/record", body)


@app.get("/api/emotion/timeline")
async def proxy_emotion_timeline(
    session_id: str | None = None,
    limit: int = 50,
):
    params: dict = {"limit": limit}
    if session_id:
        params["session_id"] = session_id
    return await _proxy_get(f"{MEMU_URL}/memory/emotion/timeline", params=params,
                            fallback={"entries": [], "count": 0})


@app.post("/api/reflect")
async def proxy_reflect(request: Request):
    body = await request.json() if (await request.body()) else {}
    return await _proxy_post(f"{MEMU_URL}/memory/self-reflect", body, timeout=15.0)


@app.get("/api/reflections")
async def proxy_reflections(limit: int = 10):
    return await _proxy_get(f"{MEMU_URL}/memory/self-reflections", params={"limit": limit},
                            fallback={"entries": [], "count": 0})


@app.get("/api/relationship")
async def proxy_relationship():
    return await _proxy_get(f"{MEMU_URL}/memory/relationship")


@app.post("/api/relationship/milestone")
async def proxy_milestone(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/relationship/milestone", body)


@app.get("/api/confidence")
async def proxy_confidence():
    return await _proxy_get(f"{MEMU_URL}/memory/confidence")


@app.get("/api/eq/summary")
async def proxy_eq_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/eq/summary")


@app.post("/api/confess")
async def proxy_confess(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/confess", body)


# ── P18: Narrative Identity proxies (H1.7: all wrapped) ─────────────

@app.post("/api/autobiography/record")
async def proxy_autobiography_record(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/autobiography/record", body)


@app.get("/api/autobiography")
async def proxy_autobiography(request: Request):
    return await _proxy_get(f"{MEMU_URL}/memory/autobiography", params=dict(request.query_params),
                            fallback={"entries": [], "count": 0})


@app.get("/api/identity")
async def proxy_identity():
    return await _proxy_get(f"{MEMU_URL}/memory/identity")


@app.get("/api/story-arcs")
async def proxy_story_arcs():
    return await _proxy_get(f"{MEMU_URL}/memory/story-arcs", fallback={"arcs": []})


@app.get("/api/future-self")
async def proxy_future_self():
    return await _proxy_get(f"{MEMU_URL}/memory/future-self")


@app.post("/api/legacy/write")
async def proxy_legacy_write(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/legacy/write", body)


@app.get("/api/legacy")
async def proxy_legacy(request: Request):
    return await _proxy_get(f"{MEMU_URL}/memory/legacy", params=dict(request.query_params),
                            fallback={"messages": []})


@app.get("/api/narrative/summary")
async def proxy_narrative_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/narrative/summary")


# ── P19 Imagination proxies (H1.7: all wrapped) ─────────────────────

@app.post("/api/imagine/counterfactual")
async def proxy_counterfactual(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/counterfactual", body)


@app.get("/api/imagine/counterfactuals")
async def proxy_counterfactuals():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/counterfactuals", fallback={"entries": []})


@app.post("/api/imagine/empathize")
async def proxy_empathize(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/empathize", body)


@app.get("/api/imagine/empathy-map")
async def proxy_empathy_map():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/empathy-map")


@app.post("/api/imagine/synthesize")
async def proxy_synthesize(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/synthesize", body)


@app.get("/api/imagine/ideas")
async def proxy_ideas():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/ideas", fallback={"ideas": []})


@app.post("/api/imagine/thought")
async def proxy_thought(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/thought", body)


@app.get("/api/imagine/inner-monologue")
async def proxy_inner_monologue():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/inner-monologue", fallback={"entries": []})


@app.post("/api/imagine/aspire")
async def proxy_aspire(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/imagine/aspire", body)


@app.get("/api/imagine/aspirations")
async def proxy_aspirations():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/aspirations", fallback={"entries": []})


@app.get("/api/imagine/summary")
async def proxy_imagination_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/imagine/summary")


# ── P20: Conscience & Values proxies (H1.7: all wrapped) ────────────

@app.post("/api/values/learn")
async def proxy_values_learn(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/values/learn", body)


@app.get("/api/values")
async def proxy_values():
    return await _proxy_get(f"{MEMU_URL}/memory/values", fallback={"values": []})


@app.post("/api/conscience/check")
async def proxy_conscience_check(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/conscience/check", body)


@app.get("/api/conscience/audit")
async def proxy_conscience_audit():
    return await _proxy_get(f"{MEMU_URL}/memory/conscience/audit")


@app.post("/api/loyalty/record")
async def proxy_loyalty_record(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/loyalty/record", body)


@app.get("/api/loyalty")
async def proxy_loyalty():
    return await _proxy_get(f"{MEMU_URL}/memory/loyalty", fallback={"entries": []})


@app.post("/api/gratitude/record")
async def proxy_gratitude_record(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MEMU_URL}/memory/gratitude/record", body)


@app.get("/api/gratitude")
async def proxy_gratitude():
    return await _proxy_get(f"{MEMU_URL}/memory/gratitude", fallback={"entries": []})


@app.get("/api/conscience/summary")
async def proxy_conscience_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/conscience/summary")


# ── P21: Proactive Agent Loop proxies (H1.7: all wrapped) ───────────

@app.get("/api/actions")
async def proxy_actions():
    return await _proxy_get(f"{MEMU_URL}/memory/actions", fallback={"actions": []})


@app.post("/api/schedule/task")
async def proxy_schedule_task(body: dict):
    return await _proxy_post(f"{MEMU_URL}/memory/schedule/task", body)


@app.get("/api/schedule/tasks")
async def proxy_schedule_tasks():
    return await _proxy_get(f"{MEMU_URL}/memory/schedule/tasks", fallback={"tasks": []})


@app.post("/api/schedule/task/{task_id}/cancel")
async def proxy_cancel_task(task_id: str):
    return await _proxy_post(f"{MEMU_URL}/memory/schedule/task/{task_id}/cancel")


@app.post("/api/reminders/set")
async def proxy_set_reminder(body: dict):
    return await _proxy_post(f"{MEMU_URL}/memory/reminders/set", body)


@app.get("/api/reminders")
async def proxy_reminders():
    return await _proxy_get(f"{MEMU_URL}/memory/reminders", fallback={"reminders": []})


@app.post("/api/reminders/{reminder_id}/cancel")
async def proxy_cancel_reminder(reminder_id: str):
    return await _proxy_post(f"{MEMU_URL}/memory/reminders/{reminder_id}/cancel")


@app.post("/api/briefing/morning")
async def proxy_morning_briefing():
    return await _proxy_post(f"{MEMU_URL}/memory/briefing/morning")


@app.post("/api/briefing/evening")
async def proxy_evening_checkin():
    return await _proxy_post(f"{MEMU_URL}/memory/briefing/evening")


@app.get("/api/agent/summary")
async def proxy_agent_summary():
    return await _proxy_get(f"{MEMU_URL}/memory/agent/summary")


# ── P22 Operator Model proxies (H1.7: all wrapped) ─────────────────

@app.post("/api/echo/analyse")
async def proxy_echo_analyse(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/echo/analyse", body)


@app.get("/api/echo/history")
async def proxy_echo_history():
    return await _proxy_get(f"{MEMU_URL}/memory/echo/history", fallback={"entries": []})


@app.post("/api/nudge/escalate")
async def proxy_nudge_escalate(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/nudge/escalate", body)


@app.get("/api/nudge/ladder")
async def proxy_nudge_ladder():
    return await _proxy_get(f"{MEMU_URL}/memory/nudge/ladder", fallback={"ladder": {}})


@app.post("/api/cross-mode/scan")
async def proxy_cross_mode_scan(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/cross-mode/scan", body)


@app.get("/api/cross-mode")
async def proxy_cross_mode():
    return await _proxy_get(f"{MEMU_URL}/memory/cross-mode", fallback={"insights": []})


@app.post("/api/oracle/predict")
async def proxy_oracle_predict(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/oracle/predict", body)


@app.get("/api/oracle/chains")
async def proxy_oracle_chains():
    return await _proxy_get(f"{MEMU_URL}/memory/oracle/chains", fallback={"chains": []})


@app.post("/api/shadow/branch")
async def proxy_shadow_branch(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{MEMU_URL}/memory/shadow/branch", body)


@app.get("/api/shadow/branches")
async def proxy_shadow_branches():
    return await _proxy_get(f"{MEMU_URL}/memory/shadow/branches", fallback={"branches": []})


@app.get("/api/operator-model")
async def proxy_operator_model():
    return await _proxy_get(f"{MEMU_URL}/memory/operator-model")


# ── J2 Wake + Intent proxies ──────────────────────────────────────────

@app.post("/api/wake/detect")
async def proxy_wake_detect(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/detect", body)


@app.post("/api/wake/intent")
async def proxy_wake_intent(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/intent", body)


@app.post("/api/wake/process")
async def proxy_wake_process(body: Dict[str, Any] = Body(...)):
    return await _proxy_post(f"{WAKE_URL}/wake/process", body)


# ── Unified App Shell ────────────────────────────────────────────────

@app.get("/app")
async def app_shell() -> HTMLResponse:
    """Serve the unified single-page app shell."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "app.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


# ── Chat proxy — Kai's face ─────────────────────────────────────────
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://agentic:8007")


@app.get("/chat")
async def chat_page() -> HTMLResponse:
    """Serve the chat UI (legacy standalone, redirects to /app)."""
    return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app">')


@app.post("/api/chat")
async def api_chat_proxy(request: Request):
    """Proxy chat requests to agentic /chat with SSE streaming.

    This keeps the browser talking only to dashboard:8080.
    The agentic service does the actual LLM inference.
    """
    body = await request.json()

    async def stream_proxy():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0, read=120.0)) as client:
                async with client.stream(
                    "POST",
                    f"{LANGGRAPH_URL}/chat",
                    json=body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception as exc:
            yield f"data: {_json.dumps({'token': f'[connection error: {str(exc)[:200]}]'})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_proxy(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


SCREEN_CAPTURE_URL = os.getenv("SCREEN_CAPTURE_URL", "http://screen-capture:8059")
AUDIO_URL = os.getenv("AUDIO_SERVICE_URL", "http://audio-service:8021")
TTS_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8030")
BROWSER_AGENT_URL = os.getenv("BROWSER_AGENT_URL", "http://browser-agent:8040")
VISION_URL = os.getenv("VISION_SERVICE_URL", "http://vision-service:8023")
CLIPBOARD_URL = os.getenv("CLIPBOARD_SERVICE_URL", "http://clipboard-service:8024")
FILES_URL = os.getenv("FILES_SERVICE_URL", "http://files-service:8025")
NOTIFY_URL = os.getenv("NOTIFY_SERVICE_URL", "http://notify-service:8031")
DOC_PARSER_URL = os.getenv("DOC_PARSER_URL", "http://document-parser:8032")
MONITOR_URL = os.getenv("MONITOR_SERVICE_URL", "http://monitor-service:8033")
BROKER_URL = os.getenv("BROKER_URL", "http://broker-bridge:8034")
SYSMETRICS_URL = os.getenv("SYSMETRICS_URL", "http://sysmetrics:8035")
WEATHER_SERVICE_URL = os.getenv("WEATHER_SERVICE_URL", "http://weather-service:8039")
DOCKER_WATCHER_URL = os.getenv("DOCKER_WATCHER_URL", "http://docker-watcher:8041")
AIRQUALITY_URL = os.getenv("AIRQUALITY_URL", "http://airquality-service:8042")
CALENDAR_SERVICE_URL = os.getenv("CALENDAR_SERVICE_URL", "http://calendar-service:8043")
GIT_WATCHER_URL = os.getenv("GIT_WATCHER_URL", "http://git-watcher:8044")
SCREEN_WATCHER_URL = os.getenv("SCREEN_WATCHER_URL", "http://screen-watcher:8036")
EMAIL_READER_URL = os.getenv("EMAIL_READER_URL", "http://email-reader:8037")
NEWS_FEED_URL = os.getenv("NEWS_FEED_URL", "http://news-feed:8038")
_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_IMAGE_EXTS = frozenset({"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff", "tif"})
_DOC_EXTS = frozenset({"pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt", "dxf", "dwg", "zip"})


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Route uploaded file to OCR (images) or document parser (PDF, Office, CAD, ZIP).

    Returns JSON with a 'text' field containing extracted content.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    data = await file.read()
    if len(data) > _UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    ext = (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "").lower()

    if ext in _IMAGE_EXTS:
        target_url = f"{SCREEN_CAPTURE_URL}/capture/file"
        service_name = "OCR"
        content_type = file.content_type or "image/png"
    elif ext in _DOC_EXTS:
        target_url = f"{DOC_PARSER_URL}/parse"
        service_name = "document parser"
        content_type = file.content_type or "application/octet-stream"
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: .{ext or '(none)'}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                target_url,
                files={"file": (file.filename, data, content_type)},
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"{service_name} service rejected the file: {exc}")
        raise HTTPException(status_code=502, detail=f"{service_name} service error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{service_name} service unreachable: {exc}",
        )


@app.post("/api/tts/synthesize")
async def api_tts_synthesize(request: Request):
    """Proxy text-to-speech synthesis to the TTS service.

    Accepts JSON: {text, voice?, rate?, volume?}
    Returns audio/mpeg when TTS service is available, 503 when offline.
    """
    from fastapi.responses import Response as FastAPIResponse
    body = await request.json()
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{TTS_URL}/synthesize", json=body)
            resp.raise_for_status()
            return FastAPIResponse(
                content=resp.content,
                media_type="audio/mpeg",
                headers={"X-Voice": resp.headers.get("X-Voice", "")},
            )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"TTS service rejected request: {exc}")
        raise HTTPException(status_code=502, detail=f"TTS service error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"TTS service unreachable: {exc}")


@app.post("/api/audio/transcribe")
async def api_audio_transcribe(file: UploadFile = File(...)):
    """Receive an audio blob from the browser MediaRecorder and return a transcript.

    Proxies to the audio-service Whisper backend. Degrades to 503 when unavailable.
    """
    data = await file.read()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{AUDIO_URL}/capture/file",
                files={"file": (file.filename or "audio.webm", data, file.content_type or "audio/webm")},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Audio service rejected file: {exc}")
        raise HTTPException(status_code=502, detail=f"Audio service error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Audio service unreachable: {exc}",
        )


# ── Browser Agent proxies ────────────────────────────────────────────────────

@app.post("/api/browser/navigate")
async def api_browser_navigate(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/navigate", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Browser agent rejected request: {exc}")
        raise HTTPException(status_code=502, detail=f"Browser agent error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Browser agent unreachable: {exc}")


@app.post("/api/browser/scrape")
async def api_browser_scrape(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/scrape", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Browser agent rejected request: {exc}")
        raise HTTPException(status_code=502, detail=f"Browser agent error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Browser agent unreachable: {exc}")


@app.post("/api/browser/run")
async def api_browser_run(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/run", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Browser agent rejected request: {exc}")
        raise HTTPException(status_code=502, detail=f"Browser agent error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Browser agent unreachable: {exc}")


@app.get("/api/browser/screenshot")
async def api_browser_screenshot():
    from fastapi.responses import Response as FastAPIResponse
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/screenshot")
            resp.raise_for_status()
            return FastAPIResponse(content=resp.content, media_type="image/png")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Browser agent error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Browser agent unreachable: {exc}")


# ── Vision / camera proxies ───────────────────────────────────────────────────

@app.post("/api/vision/analyze")
async def api_vision_analyze(file: UploadFile = File(...)):
    data = await file.read()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{VISION_URL}/analyze/frame",
                files={"file": (file.filename or "frame.jpg", data, file.content_type or "image/jpeg")},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Vision service rejected frame: {exc}")
        raise HTTPException(status_code=502, detail=f"Vision service error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Vision service unreachable: {exc}")


@app.post("/api/vision/presence")
async def api_vision_presence(file: UploadFile = File(...)):
    data = await file.read()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{VISION_URL}/analyze/presence",
                files={"file": (file.filename or "frame.jpg", data, file.content_type or "image/jpeg")},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Vision service rejected frame: {exc}")
        raise HTTPException(status_code=502, detail=f"Vision service error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Vision service unreachable: {exc}")


# ── Clipboard proxies ─────────────────────────────────────────────────────────

@app.post("/api/clipboard/push")
async def api_clipboard_push(request: Request):
    body = await request.json()
    return await _proxy_post(f"{CLIPBOARD_URL}/push", body=body, fallback={"ok": False})


@app.get("/api/clipboard/latest")
async def api_clipboard_latest():
    return await _proxy_get(f"{CLIPBOARD_URL}/latest", fallback={"content": "", "id": None})


@app.get("/api/clipboard/history")
async def api_clipboard_history(limit: int = 20):
    return await _proxy_get(f"{CLIPBOARD_URL}/history", params={"limit": limit}, fallback={"entries": []})


@app.delete("/api/clipboard/history")
async def api_clipboard_clear():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{CLIPBOARD_URL}/history")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"cleared": False}


# ── File Watcher proxies ───────────────────────────────────────────────────────

@app.get("/api/files/events")
async def api_files_events(limit: int = 50, event_type: str = ""):
    params: dict = {"limit": limit}
    if event_type:
        params["event_type"] = event_type
    return await _proxy_get(f"{FILES_URL}/events", params=params, fallback={"events": []})


@app.get("/api/files/watching")
async def api_files_watching():
    return await _proxy_get(f"{FILES_URL}/watching", fallback={"directories": []})


@app.post("/api/files/watch")
async def api_files_watch(request: Request):
    body = await request.json()
    return await _proxy_post(f"{FILES_URL}/watch", body=body, fallback={"ok": False})


# ── Notify proxies ─────────────────────────────────────────────────────────────

@app.post("/api/notify/send")
async def api_notify_send(request: Request):
    body = await request.json()
    return await _proxy_post(f"{NOTIFY_URL}/notify", body=body, fallback={"ok": False})


@app.get("/api/notify/pending")
async def api_notify_pending(unread_only: bool = True):
    return await _proxy_get(f"{NOTIFY_URL}/pending", params={"unread_only": unread_only},
                            fallback={"notifications": []})


@app.delete("/api/notify/pending/{notification_id}")
async def api_notify_dismiss(notification_id: int):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{NOTIFY_URL}/pending/{notification_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"cleared": False}


@app.delete("/api/notify/pending")
async def api_notify_dismiss_all():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.delete(f"{NOTIFY_URL}/pending")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"cleared": False}


# ── Monitor proxies ───────────────────────────────────────────────────────────

@app.get("/api/monitor/rules")
async def api_monitor_rules():
    return await _proxy_get(f"{MONITOR_URL}/rules", fallback={"rules": [], "total": 0})


@app.post("/api/monitor/rules")
async def api_monitor_add_rule(request: Request):
    body = await request.json()
    return await _proxy_post(f"{MONITOR_URL}/rules", body=body, fallback={"ok": False})


@app.delete("/api/monitor/rules/{rule_id}")
async def api_monitor_delete_rule(rule_id: str):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{MONITOR_URL}/rules/{rule_id}")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"ok": False}


@app.post("/api/monitor/rules/{rule_id}/enable")
async def api_monitor_enable_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/enable", fallback={"ok": False})


@app.post("/api/monitor/rules/{rule_id}/disable")
async def api_monitor_disable_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/disable", fallback={"ok": False})


@app.post("/api/monitor/rules/{rule_id}/check")
async def api_monitor_check_rule(rule_id: str):
    return await _proxy_post(f"{MONITOR_URL}/rules/{rule_id}/check", fallback={"ok": False})


@app.get("/api/monitor/alerts")
async def api_monitor_alerts(limit: int = 50):
    return await _proxy_get(f"{MONITOR_URL}/alerts", params={"limit": limit}, fallback={"alerts": [], "total": 0})


@app.delete("/api/monitor/alerts")
async def api_monitor_clear_alerts():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(f"{MONITOR_URL}/alerts")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return {"ok": False}


@app.get("/api/monitor/status")
async def api_monitor_status():
    return await _proxy_get(f"{MONITOR_URL}/status", fallback={})


# ── Browser search proxy ───────────────────────────────────────────────────────

@app.post("/api/browser/search")
async def api_browser_search(request: Request):
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{BROWSER_AGENT_URL}/search", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if 400 <= status < 500:
            raise HTTPException(status_code=status, detail=f"Browser agent rejected request: {exc}")
        raise HTTPException(status_code=502, detail=f"Browser agent error: {exc}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Browser agent unreachable: {exc}")


# ── Broker bridge proxies ─────────────────────────────────────────────────────

@app.get("/api/broker/health")
async def api_broker_health():
    return await _proxy_get(f"{BROKER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/broker/ticker/{symbol}")
async def api_broker_ticker(symbol: str):
    return await _proxy_get(f"{BROKER_URL}/ticker/{symbol}", fallback={})


@app.get("/api/broker/ticker")
async def api_broker_tickers(symbols: str = ""):
    params = {"symbols": symbols} if symbols else {}
    return await _proxy_get(f"{BROKER_URL}/ticker", params=params, fallback={"tickers": []})


@app.get("/api/broker/balance")
async def api_broker_balance():
    return await _proxy_get(f"{BROKER_URL}/balance", fallback={"assets": []})


@app.get("/api/broker/positions")
async def api_broker_positions():
    return await _proxy_get(f"{BROKER_URL}/positions", fallback={"positions": []})


@app.get("/api/broker/orders")
async def api_broker_orders(symbol: str = ""):
    params = {"symbol": symbol} if symbol else {}
    return await _proxy_get(f"{BROKER_URL}/orders", params=params, fallback={"orders": []})


@app.get("/api/broker/pnl")
async def api_broker_pnl():
    return await _proxy_get(f"{BROKER_URL}/pnl/summary",
                            fallback={"total_unrealized_pnl": None, "positions": []})


@app.get("/api/broker/templates")
async def api_broker_templates():
    return await _proxy_get(f"{BROKER_URL}/templates", fallback={"templates": []})


@app.post("/api/broker/watch")
async def api_broker_watch(request: Request):
    """Create a monitor rule for a position from the Broker tab Quick Watch button."""
    body = await request.json()
    symbol = body.get("symbol", "").upper()
    threshold = body.get("threshold")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol required")
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
    monitor_url = os.getenv("MONITOR_SERVICE_URL", "http://monitor-service:8033")
    return await _proxy_post(f"{monitor_url}/rules", body=rule, fallback={"ok": False})


# ── Sysmetrics proxies ────────────────────────────────────────────────────────

@app.get("/api/sysmetrics/health")
async def api_sysmetrics_health():
    return await _proxy_get(f"{SYSMETRICS_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/sysmetrics/snapshot")
async def api_sysmetrics_snapshot():
    return await _proxy_get(f"{SYSMETRICS_URL}/snapshot", fallback={})


@app.get("/api/sysmetrics/processes")
async def api_sysmetrics_processes():
    return await _proxy_get(f"{SYSMETRICS_URL}/processes", fallback={"processes": []})


# ── Screen-watcher proxies ────────────────────────────────────────────────────

@app.get("/api/screen-watcher/health")
async def api_screen_watcher_health():
    return await _proxy_get(f"{SCREEN_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/screen-watcher/status")
async def api_screen_watcher_status():
    return await _proxy_get(f"{SCREEN_WATCHER_URL}/status", fallback={})


@app.post("/api/screen-watcher/watch/start")
async def api_screen_watcher_start(request: Request):
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    return await _proxy_post(f"{SCREEN_WATCHER_URL}/watch/start", body=body, fallback={"ok": False})


@app.post("/api/screen-watcher/watch/stop")
async def api_screen_watcher_stop():
    return await _proxy_post(f"{SCREEN_WATCHER_URL}/watch/stop", body={}, fallback={"ok": False})


# ── Email-reader proxies ──────────────────────────────────────────────────────

@app.get("/api/email/health")
async def api_email_health():
    return await _proxy_get(f"{EMAIL_READER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/email/inbox")
async def api_email_inbox(limit: int = 20):
    return await _proxy_get(f"{EMAIL_READER_URL}/inbox", params={"limit": limit}, fallback={"messages": []})


@app.get("/api/email/unread")
async def api_email_unread():
    return await _proxy_get(f"{EMAIL_READER_URL}/unread", fallback={"unread_count": 0, "sample": []})


@app.post("/api/email/refresh")
async def api_email_refresh():
    return await _proxy_post(f"{EMAIL_READER_URL}/refresh", body={}, fallback={"ok": False})


# ── News-feed proxies ─────────────────────────────────────────────────────────

@app.get("/api/news/health")
async def api_news_health():
    return await _proxy_get(f"{NEWS_FEED_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/news/articles")
async def api_news_articles(limit: int = 20, tag: str = "", since_minutes: int = 0):
    params: dict = {"limit": limit}
    if tag:
        params["tag"] = tag
    if since_minutes > 0:
        params["since_minutes"] = since_minutes
    return await _proxy_get(f"{NEWS_FEED_URL}/articles", params=params, fallback={"articles": []})


@app.get("/api/news/search")
async def api_news_search(q: str = "", limit: int = 10):
    if not q:
        raise HTTPException(status_code=400, detail="q is required")
    return await _proxy_get(f"{NEWS_FEED_URL}/search", params={"q": q, "limit": limit}, fallback={"results": []})


@app.post("/api/news/refresh")
async def api_news_refresh():
    return await _proxy_post(f"{NEWS_FEED_URL}/refresh", body={}, fallback={"ok": False})


@app.get("/api/news/feeds")
async def api_news_feeds():
    return await _proxy_get(f"{NEWS_FEED_URL}/feeds", fallback={"feeds": []})


# ── Broker market depth extensions ────────────────────────────────────────────

@app.get("/api/broker/depth/{symbol}")
async def api_broker_depth(symbol: str, limit: int = 20):
    return await _proxy_get(f"{BROKER_URL}/depth/{symbol}", params={"limit": limit}, fallback={})


@app.get("/api/broker/stats/{symbol}")
async def api_broker_stats(symbol: str):
    return await _proxy_get(f"{BROKER_URL}/stats/24hr/{symbol}", fallback={})


@app.get("/api/broker/trades/{symbol}")
async def api_broker_trades(symbol: str, limit: int = 20):
    return await _proxy_get(f"{BROKER_URL}/trades/{symbol}", params={"limit": limit}, fallback={"trades": []})


@app.get("/api/broker/stocks/{symbol}")
async def broker_stocks(symbol: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{BROKER_URL}/stocks/{symbol}")
        r.raise_for_status()
        return r.json()


@app.get("/api/broker/forex/{pair}")
async def broker_forex(pair: str):
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(f"{BROKER_URL}/forex/{pair}")
        r.raise_for_status()
        return r.json()


# ── Weather service proxies ───────────────────────────────────────────────────

@app.get("/api/weather/health")
async def api_weather_health():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/weather/current")
async def api_weather_current():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/current", fallback={})


@app.get("/api/weather/forecast")
async def api_weather_forecast():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/forecast", fallback={"forecast": []})


@app.get("/api/weather/summary")
async def api_weather_summary():
    return await _proxy_get(f"{WEATHER_SERVICE_URL}/summary", fallback={"summary": "Weather unavailable."})


# ── Docker-watcher proxies ────────────────────────────────────────────────────

@app.get("/api/docker/health")
async def api_docker_health():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/docker/containers")
async def api_docker_containers():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/containers", fallback={"containers": [], "total": 0})


@app.get("/api/docker/unhealthy")
async def api_docker_unhealthy():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/unhealthy", fallback={"unhealthy": [], "count": 0})


@app.get("/api/docker/summary")
async def api_docker_summary():
    return await _proxy_get(f"{DOCKER_WATCHER_URL}/summary", fallback={"summary": "Docker data unavailable."})


# ── Air quality proxies ───────────────────────────────────────────────────────

@app.get("/api/airquality/health")
async def api_airquality_health():
    return await _proxy_get(f"{AIRQUALITY_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/airquality/current")
async def api_airquality_current():
    return await _proxy_get(f"{AIRQUALITY_URL}/current", fallback={})


@app.get("/api/airquality/summary")
async def api_airquality_summary():
    return await _proxy_get(f"{AIRQUALITY_URL}/summary", fallback={"summary": "Air quality unavailable."})


# ── Calendar service proxies ──────────────────────────────────────────────────

@app.get("/api/calendar/health")
async def api_calendar_health():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/calendar/events/today")
async def api_calendar_today():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/events/today", fallback={"events": []})


@app.get("/api/calendar/events/upcoming")
async def api_calendar_upcoming(days: int = 7):
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/events/upcoming", params={"days": days},
                            fallback={"events": []})


@app.get("/api/calendar/summary")
async def api_calendar_summary():
    return await _proxy_get(f"{CALENDAR_SERVICE_URL}/summary", fallback={"summary": "Calendar not configured."})


@app.post("/api/calendar/refresh")
async def api_calendar_refresh():
    return await _proxy_post(f"{CALENDAR_SERVICE_URL}/refresh", body={}, fallback={"ok": False})


# ── Git-watcher proxies ───────────────────────────────────────────────

@app.get("/api/git/health")
async def api_git_health():
    return await _proxy_get(f"{GIT_WATCHER_URL}/health", fallback={"status": "unavailable"})


@app.get("/api/git/repos")
async def api_git_repos():
    return await _proxy_get(f"{GIT_WATCHER_URL}/repos", fallback={"repos": [], "count": 0})


@app.get("/api/git/dirty")
async def api_git_dirty():
    return await _proxy_get(f"{GIT_WATCHER_URL}/dirty", fallback={"repos": [], "count": 0})


@app.get("/api/git/summary")
async def api_git_summary():
    return await _proxy_get(
        f"{GIT_WATCHER_URL}/summary", fallback={"summary": "Git data unavailable."}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
