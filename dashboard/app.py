from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from common.runtime import AuditStream, ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("dashboard", os.getenv("LOG_PATH", "/tmp/dashboard.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.4.0")

# mount static UI stub
app.mount("/static", StaticFiles(directory="static"), name="static")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("dashboard", required=os.getenv("AUDIT_REQUIRED", "false").lower()=="true")

SUPERVISOR_URL = os.getenv("SUPERVISOR_URL", "http://supervisor:8051")

NODES: Dict[str, str] = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": os.getenv("MEMU_URL", "http://memu-core:8001") + "/health",
    "heartbeat": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
    "supervisor": f"{SUPERVISOR_URL}/health",
    "verifier": os.getenv("VERIFIER_URL", "http://verifier:8052") + "/health",
}
_langgraph_url = os.getenv("LANGGRAPH_URL", "")
if _langgraph_url:
    NODES["langgraph"] = _langgraph_url + "/health"
_executor_url = os.getenv("EXECUTOR_URL", "")
if _executor_url:
    NODES["executor"] = _executor_url + "/health"

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
    return {"status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)", "tool_gate_url": TOOL_GATE_URL}


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
            memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
            memory_count = int((await client.get(f"{memu_url}/memory/stats", timeout=2.0)).json().get("records", 0))
    except Exception:
        logger.warning("Failed to fetch ledger/memory stats for index")

    go_no_go = await build_go_no_go_report()
    tool_gate_health = statuses.get("tool-gate", {}).get("details", {})
    policy_mode = str(tool_gate_health.get("mode", "PUB")).upper()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
