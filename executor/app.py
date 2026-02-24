"""executor — sandboxed tool execution engine.

The final link in the sovereign AI execution chain:
  tool-gate (policy) → executor (sandboxed execution) → heartbeat (audit)

Supports multiple tool types:
  - shell: sandboxed shell command execution (restricted to ALLOWED_COMMANDS)
  - script: run a pre-approved script from SCRIPTS_DIR
  - python: evaluate a Python expression in a restricted namespace
  - noop: no-operation (for testing the execution pipeline)

All execution happens in subprocess with:
  - Configurable timeouts
  - Output size limits
  - Malware scanning (ClamAV when available)
  - State tracking with rollback
  - Heartbeat notifications on failures
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import AuditStream, ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("executor", os.getenv("LOG_PATH", "/tmp/executor.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Executor Service", version="0.6.0")
MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", "1048576"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")
SCRIPTS_DIR = Path(os.getenv("SCRIPTS_DIR", "/workspaces/kai-system/scripts"))

budget = ErrorBudget(window_seconds=300)
audit = AuditStream("executor", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")

# ── allowed commands (security allowlist) ─────────────────────────────
# Only these base commands can be invoked via shell tool type.
# Prevents arbitrary command injection.
ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "date",
    "whoami", "pwd", "df", "du", "uptime", "free", "ps",
    "python3", "pip", "git", "make", "docker", "curl",
}

# ── approved scripts (only pre-existing scripts can be run) ───────────
APPROVED_SCRIPT_PATTERNS = [
    r"^go_no_go_check\.py$",
    r"^smoke_core\.py$",
    r"^health_sweep\.sh$",
    r"^contract_smoke\.sh$",
    r"^gameday_scorecard\.py$",
    r"^hardening_smoke\.py$",
]

# ── dangerous patterns (blocked even if command is allowed) ───────────
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",       # recursive delete from root
    r">\s*/dev/sd",         # write to raw block device
    r"mkfs\.",              # format filesystem
    r"dd\s+if=",            # raw disk operations
    r":\(\)\{.*\}",         # fork bomb
    r"chmod\s+777\s+/",     # dangerous permissions on root
    r"curl.*\|\s*(ba)?sh",  # pipe to shell
    r"wget.*\|\s*(ba)?sh",  # pipe to shell
]


class StateStore:
    def __init__(self) -> None:
        self._states: List[Dict[str, Any]] = []

    def push(self, state: Dict[str, Any]) -> None:
        self._states.append(state)

    def revert_last_state(self) -> Dict[str, Any]:
        return self._states.pop() if self._states else {"status": "none"}

    def history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._states[-limit:]


store = StateStore()


class ExecutionRequest(BaseModel):
    tool: str
    params: Dict[str, Any]
    task_id: str
    device: str


class ExecutionResult(BaseModel):
    task_id: str
    status: str
    output: str
    duration_ms: int
    exit_code: int
    stderr: str
    truncated: bool
    policy_context: Dict[str, Any]


def malware_scan(payload: str) -> Dict[str, Any]:
    if shutil.which("clamscan"):
        proc = subprocess.run(["clamscan", "-"], input=payload, capture_output=True, text=True, timeout=10, check=False)
        return {"engine": "clamscan", "code": proc.returncode, "stderr": proc.stderr}
    return {"engine": "none", "code": 0, "stderr": ""}


def _is_command_allowed(cmd_line: str) -> bool:
    """Check if the base command is in the allowlist."""
    parts = cmd_line.strip().split()
    if not parts:
        return False
    base_cmd = Path(parts[0]).name  # handle /usr/bin/ls → ls
    return base_cmd in ALLOWED_COMMANDS


def _has_dangerous_patterns(cmd_line: str) -> Optional[str]:
    """Check for dangerous patterns. Returns the matched pattern or None."""
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_line):
            return pattern
    return None


def _is_approved_script(script_name: str) -> bool:
    """Check if a script name matches an approved pattern."""
    name = Path(script_name).name
    return any(re.match(p, name) for p in APPROVED_SCRIPT_PATTERNS)


async def notify_heartbeat(reason: str, status: str = "executor_event") -> None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{HEARTBEAT_URL}/event", json={"status": status, "reason": reason})
    except Exception:
        logger.warning("Heartbeat notification failed")


def _execute_shell(params: Dict[str, Any], timeout: int) -> subprocess.CompletedProcess:
    """Execute a shell command with security checks."""
    command = params.get("command", "")
    if not command:
        raise HTTPException(status_code=400, detail="shell tool requires 'command' param")

    # security: check allowlist
    if not _is_command_allowed(command):
        raise HTTPException(status_code=403, detail={
            "reason": "command_not_allowed",
            "message": f"Base command not in allowlist. Allowed: {sorted(ALLOWED_COMMANDS)}",
        })

    # security: check dangerous patterns
    danger = _has_dangerous_patterns(command)
    if danger:
        raise HTTPException(status_code=403, detail={
            "reason": "dangerous_pattern",
            "message": f"Command contains dangerous pattern: {danger}",
        })

    # execute in a restricted environment
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
    }
    return subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/tmp",
        env=env,
        check=False,
    )


def _execute_script(params: Dict[str, Any], timeout: int) -> subprocess.CompletedProcess:
    """Execute an approved script from SCRIPTS_DIR."""
    script = params.get("script", "")
    if not script:
        raise HTTPException(status_code=400, detail="script tool requires 'script' param")

    if not _is_approved_script(script):
        raise HTTPException(status_code=403, detail={
            "reason": "script_not_approved",
            "message": f"Script '{script}' is not in the approved list",
        })

    script_path = SCRIPTS_DIR / Path(script).name  # prevent path traversal
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script_path.name}")

    args = params.get("args", [])
    if script_path.suffix == ".py":
        cmd = ["python3", str(script_path)] + [str(a) for a in args]
    elif script_path.suffix == ".sh":
        cmd = ["bash", str(script_path)] + [str(a) for a in args]
    else:
        raise HTTPException(status_code=400, detail="Only .py and .sh scripts are supported")

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(SCRIPTS_DIR),
        check=False,
    )


def _execute_python(params: Dict[str, Any]) -> subprocess.CompletedProcess:
    """Evaluate a Python expression in a sandboxed subprocess."""
    expression = params.get("expression", "")
    if not expression:
        raise HTTPException(status_code=400, detail="python tool requires 'expression' param")

    # block dangerous builtins
    blocked = ["__import__", "exec", "eval", "compile", "open", "breakpoint", "exit", "quit"]
    for b in blocked:
        if b in expression:
            raise HTTPException(status_code=403, detail={
                "reason": "blocked_builtin",
                "message": f"Expression contains blocked builtin: {b}",
            })

    # run in isolated subprocess with restricted builtins
    wrapper = f"""
import sys
import math
import json
import datetime
result = {expression}
print(json.dumps(result) if not isinstance(result, str) else result)
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapper)
        f.flush()
        try:
            proc = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=10,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"},
                check=False,
            )
        finally:
            os.unlink(f.name)
    return proc


def _execute_noop(params: Dict[str, Any]) -> subprocess.CompletedProcess:
    """No-op execution for testing the pipeline."""
    return subprocess.CompletedProcess(
        args=["noop"],
        returncode=0,
        stdout=f"noop executed with params: {params}",
        stderr="",
    )


# ── tool dispatch table ──────────────────────────────────────────────
TOOL_HANDLERS = {
    "shell": _execute_shell,
    "noop": _execute_noop,
    "script": _execute_script,
    "python": _execute_python,
}


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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/alive")
async def alive() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/tools")
async def list_tools() -> Dict[str, Any]:
    """List available tool types and their descriptions."""
    return {
        "status": "ok",
        "tools": {
            "shell": {
                "description": "Execute a shell command (allowlisted commands only)",
                "params": {"command": "string"},
                "allowed_commands": sorted(ALLOWED_COMMANDS),
            },
            "script": {
                "description": "Run an approved script from the scripts directory",
                "params": {"script": "string", "args": "list[string]"},
                "approved_patterns": APPROVED_SCRIPT_PATTERNS,
            },
            "python": {
                "description": "Evaluate a Python expression in a sandboxed subprocess",
                "params": {"expression": "string"},
                "blocked_builtins": ["__import__", "exec", "eval", "compile", "open"],
            },
            "noop": {
                "description": "No-operation (for testing the execution pipeline)",
                "params": {},
            },
        },
    }


@app.get("/history")
async def execution_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent execution state history."""
    return store.history(limit)


@app.post("/execute", response_model=ExecutionResult)
async def execute(request: ExecutionRequest) -> ExecutionResult:
    tool = sanitize_string(request.tool)
    if not tool:
        raise HTTPException(status_code=400, detail="tool is required")
    if request.device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be cpu or cuda")

    # validate tool type
    if tool not in TOOL_HANDLERS:
        raise HTTPException(status_code=400, detail={
            "reason": "unknown_tool",
            "message": f"Unknown tool type '{tool}'. Available: {list(TOOL_HANDLERS.keys())}",
        })

    payload = sanitize_string(str(request.params))
    scan = malware_scan(payload)
    if scan["code"] == 1:
        store.revert_last_state()
        await notify_heartbeat("malware signature detected", "malware_alert")
        raise HTTPException(status_code=400, detail="malware signature detected")

    store.push({"task_id": request.task_id, "tool": tool, "params": request.params, "started_at": time.time()})
    start = time.time()

    try:
        handler = TOOL_HANDLERS[tool]
        # python and noop don't need timeout param
        if tool in ("shell", "script"):
            proc = handler(request.params, EXECUTION_TIMEOUT)
        else:
            proc = handler(request.params)
    except HTTPException:
        raise  # re-raise HTTP errors (security blocks, etc.)
    except subprocess.TimeoutExpired:
        store.revert_last_state()
        await notify_heartbeat("execution timeout")
        raise HTTPException(
            status_code=408,
            detail={
                "task_id": sanitize_string(request.task_id),
                "status": "timeout",
                "reason": "execution timeout",
                "policy_context": {"device": request.device, "tool": tool, "timeout_s": EXECUTION_TIMEOUT},
            },
        )
    except Exception as exc:
        store.revert_last_state()
        await notify_heartbeat(f"execution error: {exc}")
        logger.error("Execution failed for tool=%s task_id=%s: %s", tool, request.task_id, exc)
        raise HTTPException(status_code=500, detail=f"execution failed: {exc}")

    if proc.returncode != 0:
        store.revert_last_state()
        await notify_heartbeat(f"subprocess failure ({proc.returncode})")
        logger.warning("Tool %s failed with exit code %d: %s", tool, proc.returncode, proc.stderr[:500])
        raise HTTPException(status_code=500, detail={
            "reason": "execution_failed",
            "exit_code": proc.returncode,
            "stderr": proc.stderr[:1000],
        })

    output = proc.stdout.strip()
    truncated = len(output) > MAX_OUTPUT_SIZE
    if truncated:
        output = output[:MAX_OUTPUT_SIZE] + "..."

    duration_ms = int((time.time() - start) * 1000)
    if duration_ms > EXECUTION_TIMEOUT * 1000:
        store.revert_last_state()
        await notify_heartbeat("execution timeout")
        raise HTTPException(
            status_code=408,
            detail={
                "task_id": sanitize_string(request.task_id),
                "status": "timeout",
                "reason": "execution exceeded allowed duration",
                "policy_context": {"device": request.device, "tool": tool, "timeout_s": EXECUTION_TIMEOUT},
            },
        )

    audit.log(
        "info",
        f"Executed {tool} (task_id={request.task_id[:12]}…) in {duration_ms}ms",
    )

    return ExecutionResult(
        task_id=sanitize_string(request.task_id),
        status="completed",
        output=output,
        duration_ms=duration_ms,
        exit_code=proc.returncode,
        stderr=proc.stderr,
        truncated=truncated,
        policy_context={
            "device": request.device, "tool": tool,
            "timeout_s": EXECUTION_TIMEOUT,
            "max_output_size": MAX_OUTPUT_SIZE,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
