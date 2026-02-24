from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI


app = FastAPI(title=os.getenv("SERVICE_NAME", "service"), version="0.1.0")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": os.getenv("SERVICE_NAME", "service")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8053")))
