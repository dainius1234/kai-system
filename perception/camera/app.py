from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="Camera Service", version="0.1.0")

CAMERA_DEVICE = os.getenv("CAMERA_DEVICE", "/dev/video0")
VIRTUAL_DEVICE = os.getenv("VIRTUAL_DEVICE", "/dev/video10")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "camera_device": CAMERA_DEVICE,
        "virtual_device": VIRTUAL_DEVICE,
    }


@app.post("/process")
async def process_frame() -> Dict[str, str]:
    return {"status": "ok", "message": "frame processed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8020")))
