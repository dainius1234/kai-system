from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="Sovereign Dashboard", version="0.1.0")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
LEDGER_URL = os.getenv("LEDGER_URL", "postgresql://keeper:***@postgres:5432/sovereign")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "tool_gate_url": TOOL_GATE_URL}


@app.get("/")
async def index() -> Dict[str, str]:
    return {
        "service": "dashboard",
        "tool_gate_url": TOOL_GATE_URL,
        "ledger_url": LEDGER_URL,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
