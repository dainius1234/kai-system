"""Git-watcher service — polls local git repositories for state.

Exposes repository status as JSON endpoints for Kai's situational awareness.

Endpoints:
  GET /health        → {status, repo_count, last_poll, uptime_seconds}
  GET /metrics       → error budget snapshot
  GET /repos         → {repos: [...], count, last_poll}
  GET /repos/{index} → single repo dict (404 if out of range)
  GET /dirty         → {repos: [...], count} — only repos with changes
  GET /summary       → one-sentence summary string
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("git-watcher", os.getenv("LOG_PATH", "/tmp/git-watcher.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("git-watcher")

    class ErrorBudget:
        def __init__(self, **_): pass

        def record(self, *_, **__): pass

        def snapshot(self): return {}

    budget = ErrorBudget()

PORT = int(os.getenv("PORT", "8044"))
GIT_WATCH_PATHS = os.getenv("GIT_WATCH_PATHS", "/workspace")
GIT_REFRESH_SECONDS = int(os.getenv("GIT_REFRESH_SECONDS", "60"))

_start = time.time()
_repos: List[dict] = []
_last_poll: float = 0.0
_refresh_task: Optional[asyncio.Task] = None


def _run_git(args: list, cwd: str) -> str:
    """Run a git subcommand and return stripped stdout, raise RuntimeError on failure."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True, text=True, cwd=cwd, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} exited {result.returncode}")
    return result.stdout.strip()


def _inspect_repo(path: str) -> dict:
    """Return a state snapshot dict for the git repository at *path*."""
    base: dict = {
        "path": path,
        "branch": "",
        "commit_hash": "",
        "commit_message": "",
        "commit_author": "",
        "commit_date": "",
        "uncommitted_changes": 0,
        "untracked_files": 0,
        "ahead": 0,
        "behind": 0,
        "stash_count": 0,
        "error": None,
    }

    if not os.path.isdir(path):
        base["error"] = f"path does not exist: {path}"
        return base

    try:
        _run_git(["rev-parse", "--git-dir"], path)
    except Exception:
        base["error"] = "not a git repo"
        return base

    try:
        base["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    except Exception as exc:
        base["error"] = str(exc)
        return base

    try:
        base["commit_hash"] = _run_git(["rev-parse", "--short", "HEAD"], path)
    except Exception:
        pass

    try:
        base["commit_message"] = _run_git(["log", "-1", "--format=%s"], path)
    except Exception:
        pass

    try:
        base["commit_author"] = _run_git(["log", "-1", "--format=%an"], path)
    except Exception:
        pass

    try:
        base["commit_date"] = _run_git(["log", "-1", "--format=%ci"], path)
    except Exception:
        pass

    try:
        status_out = _run_git(["status", "--short"], path)
        lines = status_out.splitlines() if status_out else []
        untracked = 0
        changed = 0
        for line in lines:
            if line.startswith("??"):
                untracked += 1
            elif line.strip():
                changed += 1
        base["uncommitted_changes"] = changed
        base["untracked_files"] = untracked
    except Exception:
        pass

    try:
        ahead_behind = _run_git(
            ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"], path,
        )
        parts = ahead_behind.split("\t")
        if len(parts) == 2:
            base["ahead"] = int(parts[0])
            base["behind"] = int(parts[1])
    except Exception:
        # No upstream configured — leave both at 0.
        pass

    try:
        stash_out = _run_git(["stash", "list"], path)
        base["stash_count"] = len(stash_out.splitlines()) if stash_out else 0
    except Exception:
        pass

    return base


async def _poll_loop():
    global _repos, _last_poll
    while True:
        paths = [p.strip() for p in GIT_WATCH_PATHS.split(":") if p.strip()]
        results = []
        for path in paths:
            loop = asyncio.get_running_loop()
            r = await loop.run_in_executor(None, _inspect_repo, path)
            results.append(r)
        _repos = results
        _last_poll = time.time()
        logger.info("git-watcher: polled %d repos", len(results))
        await asyncio.sleep(GIT_REFRESH_SECONDS)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _refresh_task
    _refresh_task = asyncio.create_task(_poll_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()


app = FastAPI(title="git-watcher", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "repo_count": len(_repos),
        "last_poll": _last_poll,
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/repos")
def repos_list():
    return {"repos": _repos, "count": len(_repos), "last_poll": _last_poll}


@app.get("/repos/{index}")
def repo_by_index(index: int):
    if index < 0 or index >= len(_repos):
        raise HTTPException(status_code=404, detail="repo index out of range")
    return _repos[index]


@app.get("/dirty")
def dirty():
    dirty_repos = [
        r for r in _repos
        if r.get("uncommitted_changes", 0) > 0 or r.get("untracked_files", 0) > 0
    ]
    return {"repos": dirty_repos, "count": len(dirty_repos)}


@app.get("/summary")
def summary():
    if not _repos and _last_poll == 0:
        return {"summary": "Git watcher not yet polled."}

    total = len(_repos)

    if total == 0:
        return {"summary": "No repositories configured."}

    dirty_repos = [
        r for r in _repos
        if r.get("uncommitted_changes", 0) > 0 or r.get("untracked_files", 0) > 0
    ]

    if not dirty_repos:
        if total == 1:
            branch = _repos[0].get("branch", "unknown")
            return {"summary": f"Watching 1 repo on branch {branch} — no uncommitted changes."}
        return {"summary": f"Watching {total} repos — all clean."}

    parts = []
    for r in _repos:
        path = r.get("path", "?")
        changes = r.get("uncommitted_changes", 0)
        untracked = r.get("untracked_files", 0)
        if changes > 0 or untracked > 0:
            detail = []
            if changes:
                detail.append(f"{changes} uncommitted change{'s' if changes != 1 else ''}")
            if untracked:
                detail.append(f"{untracked} untracked")
            parts.append(f"{path} has {', '.join(detail)}")
        else:
            parts.append(f"{path} is clean")

    label = f"Watching {total} repo{'s' if total != 1 else ''}"
    return {"summary": f"{label}: {'; '.join(parts)}."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
