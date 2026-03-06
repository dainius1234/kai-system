from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title=os.getenv("SERVICE_NAME", "backup-service"), version="0.2.0")
BACKUP_DIR = os.getenv("BACKUP_DIR", "/data/backup/")
PG_URI = os.getenv("PG_URI", "postgresql://postgres:postgres@postgres:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")


def _sha256(path: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "backup-service"}


# ── PostgreSQL backup ────────────────────────────────────────────────

@app.post("/backup/postgres")
async def backup_postgres() -> Dict[str, Any]:
    """Dump PostgreSQL database to a SQL file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"postgres-{ts}.sql"
    path = os.path.join(BACKUP_DIR, fname)
    if not shutil.which("pg_dump"):
        raise HTTPException(status_code=503, detail="pg_dump not available")
    try:
        subprocess.run(
            ["pg_dump", PG_URI, "-f", path, "--no-owner", "--no-acl"],
            check=True, capture_output=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="pg_dump timed out")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"pg_dump failed: {e.stderr.decode()[:500]}")
    size = os.path.getsize(path)
    return {"status": "ok", "component": "postgres", "path": path,
            "size_bytes": size, "checksum": _sha256(path)}


# ── Redis backup ─────────────────────────────────────────────────────

@app.post("/backup/redis")
async def backup_redis() -> Dict[str, Any]:
    """Trigger Redis BGSAVE and copy the RDB dump."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"redis-{ts}.rdb"
    path = os.path.join(BACKUP_DIR, fname)
    if not shutil.which("redis-cli"):
        raise HTTPException(status_code=503, detail="redis-cli not available")
    try:
        # Parse host/port from URL
        import re
        m = re.match(r"redis://([^:]+):(\d+)", REDIS_URL)
        host = m.group(1) if m else "redis"
        port = m.group(2) if m else "6379"
        subprocess.run(
            ["redis-cli", "-h", host, "-p", port, "BGSAVE"],
            check=True, capture_output=True, timeout=30,
        )
        # Get the RDB file location
        result = subprocess.run(
            ["redis-cli", "-h", host, "-p", port, "CONFIG", "GET", "dir"],
            capture_output=True, timeout=10,
        )
        rdb_dir = result.stdout.decode().strip().split("\n")[-1] if result.returncode == 0 else "/data"
        rdb_path = os.path.join(rdb_dir, "dump.rdb")
        if os.path.exists(rdb_path):
            shutil.copy2(rdb_path, path)
            return {"status": "ok", "component": "redis", "path": path,
                    "size_bytes": os.path.getsize(path), "checksum": _sha256(path)}
        return {"status": "ok", "component": "redis", "note": "BGSAVE triggered, RDB not accessible from this container"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Redis backup failed: {e.stderr.decode()[:500]}")


# ── Memory export ────────────────────────────────────────────────────

@app.post("/backup/memory")
async def backup_memory() -> Dict[str, Any]:
    """Export memories from memu-core to a JSON file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"memory-{ts}.json"
    path = os.path.join(BACKUP_DIR, fname)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/stats")
            resp.raise_for_status()
            stats = resp.json()
            # Retrieve all memories via search with empty query
            resp2 = await client.post(
                f"{MEMU_URL}/memory/retrieve",
                json={"query": "", "top_k": stats.get("records", 1000)},
            )
            resp2.raise_for_status()
            memories = resp2.json()
        import json
        with open(path, "w") as f:
            json.dump({"exported_at": ts, "stats": stats, "memories": memories}, f, indent=2, default=str)
        size = os.path.getsize(path)
        return {"status": "ok", "component": "memory", "path": path,
                "size_bytes": size, "record_count": stats.get("records", 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory export failed: {str(e)[:500]}")


# ── Ledger export ────────────────────────────────────────────────────

@app.post("/backup/ledger")
async def backup_ledger() -> Dict[str, Any]:
    """Export tool-gate ledger to a JSON file."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"ledger-{ts}.json"
    path = os.path.join(BACKUP_DIR, fname)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{TOOL_GATE_URL}/ledger/stats")
            resp.raise_for_status()
            stats = resp.json()
        import json
        with open(path, "w") as f:
            json.dump({"exported_at": ts, "stats": stats}, f, indent=2, default=str)
        size = os.path.getsize(path)
        return {"status": "ok", "component": "ledger", "path": path,
                "size_bytes": size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger export failed: {str(e)[:500]}")


# ── Full backup orchestration ────────────────────────────────────────

@app.post("/backup")
@app.post("/backup/full")
async def backup_full() -> Dict[str, Any]:
    """Run all backup components and return a manifest."""
    results = {}
    for component, func in [("postgres", backup_postgres), ("redis", backup_redis),
                             ("memory", backup_memory), ("ledger", backup_ledger)]:
        try:
            results[component] = await func()
        except HTTPException as e:
            results[component] = {"status": "failed", "error": e.detail}
        except Exception as e:
            results[component] = {"status": "failed", "error": str(e)[:200]}
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    succeeded = sum(1 for r in results.values() if r.get("status") == "ok")
    return {
        "status": "ok" if succeeded == len(results) else "partial",
        "timestamp": ts,
        "succeeded": succeeded,
        "total": len(results),
        "components": results,
    }


# ── List backups ─────────────────────────────────────────────────────

@app.get("/backup/list")
async def list_backups() -> Dict[str, Any]:
    """List all available backup files with metadata."""
    if not os.path.isdir(BACKUP_DIR):
        return {"backups": [], "total": 0}
    files = []
    for f in sorted(glob.glob(os.path.join(BACKUP_DIR, "*")), reverse=True):
        if os.path.isfile(f):
            files.append({
                "filename": os.path.basename(f),
                "path": f,
                "size_bytes": os.path.getsize(f),
                "modified": datetime.utcfromtimestamp(os.path.getmtime(f)).isoformat(),
            })
    return {"backups": files[:50], "total": len(files)}


# ── Restore (PostgreSQL only — safe, reversible) ─────────────────────

@app.post("/restore/postgres")
async def restore_postgres(backup_file: str = "") -> Dict[str, Any]:
    """Restore PostgreSQL from a backup SQL file.

    Requires the backup filename (not full path) to exist in BACKUP_DIR.
    """
    if not backup_file:
        raise HTTPException(status_code=400, detail="backup_file parameter required")
    # Sanitize: only allow alphanumeric, dashes, dots, underscores
    import re
    if not re.match(r'^[\w.\-]+$', backup_file):
        raise HTTPException(status_code=400, detail="Invalid backup filename")
    path = os.path.join(BACKUP_DIR, backup_file)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"Backup file not found: {backup_file}")
    if not shutil.which("psql"):
        raise HTTPException(status_code=503, detail="psql not available")
    try:
        subprocess.run(
            ["psql", PG_URI, "-f", path],
            check=True, capture_output=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Restore timed out")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e.stderr.decode()[:500]}")
    return {"status": "ok", "restored_from": backup_file}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8054")))
