from __future__ import annotations

import os
import re
from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Shell Sandbox", version="0.1.0")


class ShellRequest(BaseModel):
    command: str


def sanitize_string(value: str) -> str:
    sanitized = re.sub(r"[;|&]", "", value)
    return sanitized[:1024]


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
