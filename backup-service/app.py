from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI, HTTPException
import subprocess
from datetime import datetime

app = FastAPI(title=os.getenv("SERVICE_NAME", "service"), version="0.1.0")
BACKUP_DIR = os.getenv("BACKUP_DIR", "/data/backup/")
PG_URI = os.getenv("PG_URI", "postgresql://postgres:postgres@postgres:5432/postgres")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": os.getenv("SERVICE_NAME", "service")}


@app.post("/backup")
async def backup() -> Dict[str, str]:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"kai-backup-{ts}.sql"
    path = os.path.join(BACKUP_DIR, fname)
    try:
        cmd = [
            "pg_dump",
            PG_URI,
            "-f", path,
            "--no-owner",
            "--no-acl",
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")
    return {"status": "ok", "path": path}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8054")))
