from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from common.runtime import sanitize_string


app = FastAPI(title="Shell Sandbox", version="0.1.0")


class ShellRequest(BaseModel):
    command: str


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
async def run(request: ShellRequest) -> Dict[str, str]:
    command = sanitize_string(request.command)
    return {"status": "blocked", "message": "sandbox execution disabled in stub", "command": command}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8040")))
