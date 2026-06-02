from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

import prompts as prompt_catalog
from prompts import AGENTS_PATH, SOUL_PATH, _load_agents, _load_soul

router = APIRouter()


@router.get("/soul")
async def get_soul() -> Dict[str, Any]:
    """Return the current SOUL.md content."""
    return {"status": "ok", "content": prompt_catalog._soul_text, "path": str(SOUL_PATH)}


@router.post("/soul")
async def update_soul(request: Request) -> Dict[str, Any]:
    """Update SOUL.md content. Takes effect on next startup or reload."""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    for path in [SOUL_PATH, Path("data/SOUL.md")]:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            _load_soul()
            return {"status": "ok", "path": str(path), "chars": len(content)}
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Cannot write SOUL.md")


@router.get("/agents-registry")
async def get_agents_registry() -> Dict[str, Any]:
    """Return the current AGENTS.md content."""
    return {"status": "ok", "content": prompt_catalog._agents_text, "path": str(AGENTS_PATH)}


@router.post("/agents-registry")
async def update_agents_registry(request: Request) -> Dict[str, Any]:
    """Update AGENTS.md content."""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    for path in [AGENTS_PATH, Path("data/AGENTS.md")]:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            _load_agents()
            return {"status": "ok", "path": str(path), "chars": len(content)}
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Cannot write AGENTS.md")
