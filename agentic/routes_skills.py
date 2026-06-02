from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from router import (
    list_skills,
    load_skills,
    match_skill,
    prune_stale_skills,
    scan_skill_md,
    unload_skill,
)

router = APIRouter()


@router.get("/skills")
async def get_skills() -> Dict[str, Any]:
    """List all loaded skills from the skills directory."""
    skills = list_skills()
    return {"status": "ok", "skills": skills, "count": len(skills)}


@router.post("/skills/reload")
async def reload_skills() -> Dict[str, Any]:
    """Hot-reload skills from the skills directory."""
    loaded = load_skills()
    return {"status": "ok", "loaded": len(loaded), "skills": list_skills()}


@router.post("/skills/match")
async def test_skill_match(request: Request) -> Dict[str, Any]:
    """Test whether a message matches any loaded skill."""
    body = await request.json()
    text = body.get("text", "")
    skill = match_skill(text)
    if skill:
        return {
            "status": "matched",
            "skill_name": skill.name,
            "action": skill.action[:500],
            "response_template": skill.response_template[:500],
        }
    return {"status": "no_match", "skill_name": None}


@router.post("/skills/unload")
async def unload_skill_endpoint(request: Request) -> Dict[str, Any]:
    """Unload a skill by name."""
    body = await request.json()
    name = body.get("name", "")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    removed = unload_skill(name)
    return {"status": "ok" if removed else "not_found", "name": name}


@router.post("/skills/scan")
async def scan_skill_endpoint(request: Request) -> Dict[str, Any]:
    """Scan raw skill markdown text for security red flags."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return {"status": "ok", **scan_skill_md(text)}


@router.post("/skills/prune")
async def prune_skills_endpoint(request: Request) -> Dict[str, Any]:
    """Prune skills not used within max_age_days (default 30)."""
    body = await request.json()
    max_age = body.get("max_age_days", 30)
    pruned = prune_stale_skills(max_age)
    return {"status": "ok", "pruned": pruned, "pruned_count": len(pruned)}
