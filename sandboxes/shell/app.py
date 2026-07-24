"""Shell Sandbox service — restricted subprocess execution.

Executes a limited allowlist of read-only shell commands with timeout and output
caps. Anything outside the allowlist is rejected with 403. No shell=True ever.

Endpoints:
  /health   - liveness
  /run      - execute one allowed command; returns stdout/stderr/returncode
  /allowlist - list permitted command names
"""
from __future__ import annotations

import os
import shlex
import subprocess
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Shell Sandbox", version="0.2.0")

EXECUTION_TIMEOUT = int(os.getenv("SANDBOX_TIMEOUT", "10"))
MAX_OUTPUT_BYTES = int(os.getenv("SANDBOX_MAX_OUTPUT", str(64 * 1024)))  # 64 KB

# Read-only, information-gathering commands only. No write/network/exec commands.
COMMAND_ALLOWLIST: frozenset[str] = frozenset({
    "cat", "date", "df", "du", "echo", "free",
    "head", "ls", "ps", "pwd", "tail", "uptime", "wc", "whoami",
})


def _sanitize(text: str, max_len: int = 4096) -> str:
    cleaned = text.replace("\x00", "").strip()
    if len(cleaned) > max_len:
        raise HTTPException(status_code=400, detail=f"Command too long (max {max_len} chars)")
    return cleaned


class ShellRequest(BaseModel):
    command: str


class ShellResult(BaseModel):
    status: str
    stdout: str
    stderr: str
    returncode: int
    command: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/allowlist")
async def allowlist() -> Dict[str, List[str]]:
    return {"commands": sorted(COMMAND_ALLOWLIST)}


@app.post("/run", response_model=ShellResult)
async def run(request: ShellRequest) -> ShellResult:
    raw = _sanitize(request.command)
    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid command syntax: {exc}")

    if not parts:
        raise HTTPException(status_code=400, detail="Empty command")

    binary = parts[0]
    if binary not in COMMAND_ALLOWLIST:
        raise HTTPException(
            status_code=403,
            detail=f"'{binary}' not in allowlist. Permitted: {sorted(COMMAND_ALLOWLIST)}",
        )

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail=f"Timed out after {EXECUTION_TIMEOUT}s")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"'{binary}' not found on PATH")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {exc}")

    return ShellResult(
        status="ok" if result.returncode == 0 else "error",
        stdout=result.stdout[:MAX_OUTPUT_BYTES],
        stderr=result.stderr[:MAX_OUTPUT_BYTES],
        returncode=result.returncode,
        command=raw,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8040")))
