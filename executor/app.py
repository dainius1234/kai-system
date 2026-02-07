from __future__ import annotations

import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Executor Service", version="0.1.0")

MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", "1048576"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))


class ExecutionRequest(BaseModel):
    tool: str
    params: Dict[str, Any]
    task_id: str


class ExecutionResult(BaseModel):
    task_id: str
    status: str
    output: str
    duration_ms: int


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/execute", response_model=ExecutionResult)
async def execute(request: ExecutionRequest) -> ExecutionResult:
    if not request.tool:
        raise HTTPException(status_code=400, detail="tool is required")

    start = time.time()
    output = f"Executed {request.tool} with params {request.params}"
    if len(output) > MAX_OUTPUT_SIZE:
        output = output[:MAX_OUTPUT_SIZE] + "..."

    duration_ms = int((time.time() - start) * 1000)
    if duration_ms > EXECUTION_TIMEOUT * 1000:
        raise HTTPException(status_code=408, detail="execution timeout")

    return ExecutionResult(
        task_id=request.task_id,
        status="completed",
        output=output,
        duration_ms=duration_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
