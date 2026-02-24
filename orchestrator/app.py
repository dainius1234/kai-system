"""Orchestrator stub — DEPRECATED.

Memu-core is the real orchestrator in this architecture.  This service
exists only as a placeholder for a potential future "final risk authority"
layer that sits between memu-core and executor.  It currently does nothing
beyond exposing a /health endpoint.
"""
from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="Orchestrator (stub)", version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "orchestrator",
        "note": "stub — memu-core is the active orchestrator",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8050")))
