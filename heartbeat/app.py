from __future__ import annotations

import os
import time
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="Heartbeat Monitor", version="0.1.0")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ALERT_WINDOW = int(os.getenv("ALERT_WINDOW", "300"))

last_tick = time.time()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/tick")
async def tick() -> Dict[str, str]:
    global last_tick
    last_tick = time.time()
    return {"status": "ok", "message": "heartbeat received"}


@app.get("/status")
async def status() -> Dict[str, str]:
    elapsed = time.time() - last_tick
    state = "healthy" if elapsed <= ALERT_WINDOW else "stale"
    return {
        "status": state,
        "elapsed_seconds": f"{elapsed:.1f}",
        "check_interval": str(CHECK_INTERVAL),
        "alert_window": str(ALERT_WINDOW),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
