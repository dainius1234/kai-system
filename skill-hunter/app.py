"""skill-hunter — D88 M6 / D89 C2: autonomous capability growth service.

D89 additions:
    - Provenance YAML front-matter in every generated skill file
    - Sidecar .meta.json tracking error count + probationary status
    - POST /skill/{name}/error  — increment error count; disable at ≥3 errors
    - GET  /skill/{name}/health — return skill metadata
    - GET  /skills              — list skills with status
    - POST /hunt                — search + generate (unchanged interface)
    - GET  /health              — service health check
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from common.http_hygiene import pooled_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Skill Hunter", version="0.2.0")

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "/data/skills"))
PYPI_BASE = "https://pypi.org/pypi"
PORT = int(os.getenv("PORT", "8045"))
DISABLE_THRESHOLD = int(os.getenv("SKILL_DISABLE_THRESHOLD", "3"))
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")

# Heuristic keyword → candidate package map
_KW_PACKAGES: Dict[str, List[str]] = {
    "nlp": ["nltk", "spacy", "textblob"],
    "natural language": ["nltk", "spacy", "textblob"],
    "scrape": ["beautifulsoup4", "httpx", "playwright"],
    "web": ["httpx", "aiohttp", "requests"],
    "image": ["pillow", "opencv-python"],
    "pdf": ["pypdf2", "pdfminer.six", "reportlab"],
    "excel": ["openpyxl", "xlrd"],
    "data": ["pandas", "numpy"],
    "graph": ["networkx"],
    "ml": ["scikit-learn", "numpy"],
    "machine learning": ["scikit-learn", "xgboost"],
    "audio": ["pydub", "soundfile"],
    "crypto": ["cryptography", "pycryptodome"],
    "email": ["aiosmtplib"],
    "calendar": ["icalendar", "caldav"],
    "compress": ["py7zr"],
    "zip": ["py7zr"],
    "database": ["sqlalchemy", "aiosqlite"],
    "redis": ["redis"],
    "queue": ["celery", "rq"],
    "schedule": ["apscheduler"],
    "http": ["httpx", "aiohttp"],
    "markdown": ["markdown", "mistune"],
    "yaml": ["pyyaml"],
    "toml": ["toml"],
    "chart": ["matplotlib", "plotly"],
    "plot": ["matplotlib", "plotly"],
    "ocr": ["pytesseract", "easyocr"],
    "translate": ["deep-translator"],
    "weather": ["pyowm"],
    "time": ["arrow", "pendulum"],
    "date": ["arrow", "pendulum"],
    "qr": ["qrcode"],
    "barcode": ["python-barcode"],
    "ssh": ["paramiko"],
    "ftp": ["ftplib"],
    "serial": ["pyserial"],
    "bluetooth": ["bleak"],
    "usb": ["pyusb"],
    "jwt": ["pyjwt"],
    "oauth": ["authlib"],
    "llm": ["openai", "anthropic"],
    "embedding": ["sentence-transformers"],
    "vector": ["faiss-cpu"],
    "regex": ["regex"],
    "html": ["beautifulsoup4", "lxml"],
    "xml": ["lxml"],
    "json": ["orjson"],
    "csv": ["pandas"],
    "word": ["python-docx"],
    "docx": ["python-docx"],
    "pptx": ["python-pptx"],
    "powerpoint": ["python-pptx"],
    "video": ["moviepy"],
    "camera": ["opencv-python"],
    "gui": ["tkinter"],
    "notification": ["plyer"],
    "clipboard": ["pyperclip"],
    "browser": ["playwright", "selenium"],
    "automation": ["pyautogui"],
    "test": ["pytest"],
    "mock": ["pytest-mock"],
    "lint": ["flake8", "pylint"],
    "format": ["black", "autopep8"],
    "type": ["mypy"],
    "deploy": ["fabric"],
    "git": ["gitpython"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes"],
    "aws": ["boto3"],
    "gcp": ["google-cloud-storage"],
    "azure": ["azure-storage-blob"],
    "speech": ["speechrecognition", "pyttsx3"],
    "vision": ["opencv-python", "pillow"],
    "finance": ["yfinance", "pandas"],
    "stock": ["yfinance"],
    "chart": ["matplotlib", "plotly"],
    "statistics": ["scipy", "numpy"],
    "forecast": ["statsmodels", "prophet"],
    "anomaly": ["scikit-learn", "pyod"],
    "cluster": ["scikit-learn"],
}

_STOPWORDS = frozenset({
    "can", "you", "the", "a", "an", "and", "or", "to", "for", "of", "in",
    "is", "how", "do", "with", "help", "me", "i", "want", "need", "please",
    "like", "use", "make", "get", "set", "what", "when", "where", "why",
    "that", "this", "it", "my", "your", "his", "her", "its", "we", "they",
    "be", "are", "was", "were", "has", "have", "had", "will", "would",
    "could", "should", "may", "might", "shall", "must", "can", "cannot",
})


class HuntRequest(BaseModel):
    gap: str


def _skill_name(gap: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", gap.lower())[:30].strip("_")


def _meta_path(name: str) -> Path:
    return SKILLS_DIR / f"hunted_{name}.meta.json"


def _load_meta(name: str) -> Dict[str, Any]:
    mp = _meta_path(name)
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except Exception:
            pass
    return {}


def _save_meta(name: str, meta: Dict[str, Any]) -> None:
    _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _generate_skill_md(gap: str, package: str, name: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return (
        f"---\n"
        f"name: {name}\n"
        f"version: 1\n"
        f"source: skill-hunter\n"
        f"package: {package}\n"
        f"pypi_verified: true\n"
        f"hunted_at: {ts}\n"
        f"probationary: true\n"
        f"---\n\n"
        f"# Skill: {name}\n\n"
        f"## When to use\nWhen the operator asks about or needs: {gap}\n\n"
        f"## Action\nUse the `{package}` Python package.\n\n"
        f"Install if needed: `pip install {package}`\n\n"
        f"## Response template\n"
        f"I can help with that using the `{package}` package. Here is how to approach it:\n"
    )


def _extract_keywords(gap: str) -> List[str]:
    words = re.findall(r"[a-z]+", gap.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def _candidate_packages(keywords: List[str]) -> List[str]:
    seen: set = set()
    result: List[str] = []
    for kw in keywords:
        for key, pkgs in _KW_PACKAGES.items():
            if kw in key or key in kw:
                for p in pkgs:
                    if p not in seen:
                        seen.add(p)
                        result.append(p)
    return result[:8]


async def _pypi_exists(package: str) -> bool:
    try:
        async with pooled_client(timeout=5.0) as client:
            r = await client.get(f"{PYPI_BASE}/{package}/json")
            return r.status_code == 200
    except Exception:
        return False


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "skill-hunter", "version": "0.2.0", "port": PORT}


@app.post("/hunt")
async def hunt(req: HuntRequest) -> Dict[str, Any]:
    """Search for a package that fills the capability gap; generate a provenance-tracked skill file."""
    gap = req.gap.strip()
    if not gap:
        raise HTTPException(status_code=400, detail="gap is required")

    keywords = _extract_keywords(gap)
    candidates = _candidate_packages(keywords)

    found: Optional[str] = None
    for pkg in candidates:
        if await _pypi_exists(pkg):
            found = pkg
            break

    if not found:
        return {"skill_created": False, "gap": gap, "searched": candidates, "keywords": keywords}

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    name = _skill_name(gap)
    ts = datetime.now(timezone.utc).isoformat()
    skill_path = SKILLS_DIR / f"hunted_{name}.md"
    skill_path.write_text(_generate_skill_md(gap, found, name), encoding="utf-8")

    meta = {
        "name": name,
        "gap": gap,
        "package": found,
        "pypi_verified": True,
        "hunted_at": ts,
        "probationary": True,
        "error_count": 0,
        "disabled": False,
    }
    _save_meta(name, meta)

    # Log skill acquisition to memory so Kai knows what it learned and when
    async def _log_to_memory() -> None:
        try:
            async with pooled_client(timeout=5.0) as client:
                await client.post(
                    f"{MEMU_URL}/memory/memorize",
                    json={
                        "content": f"Acquired new skill '{name}' using package '{found}' to address gap: {gap}",
                        "metadata": meta,
                        "category": "skill_acquisition",
                        "user_id": "keeper",
                    },
                )
        except Exception:
            pass

    import asyncio as _asyncio
    _asyncio.create_task(_log_to_memory())

    return {
        "skill_created": True,
        "skill_name": name,
        "package": found,
        "skill_path": str(skill_path),
        "gap": gap,
        "searched": candidates,
        "provenance": meta,
    }


@app.get("/skills")
async def list_hunted_skills() -> Dict[str, Any]:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skills = []
    for f in sorted(SKILLS_DIR.glob("hunted_*.md")):
        name = f.stem.replace("hunted_", "", 1)
        meta = _load_meta(name)
        skills.append({
            "name": name,
            "probationary": meta.get("probationary", True),
            "disabled": meta.get("disabled", False),
            "error_count": meta.get("error_count", 0),
            "package": meta.get("package"),
            "hunted_at": meta.get("hunted_at"),
        })
    return {"status": "ok", "skills": skills, "count": len(skills)}


@app.get("/skill/{name}/health")
async def skill_health(name: str) -> Dict[str, Any]:
    skill_path = SKILLS_DIR / f"hunted_{name}.md"
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    meta = _load_meta(name)
    return {
        "name": name,
        "exists": True,
        "disabled": meta.get("disabled", False),
        "probationary": meta.get("probationary", True),
        "error_count": meta.get("error_count", 0),
        "package": meta.get("package"),
        "hunted_at": meta.get("hunted_at"),
        "disable_threshold": DISABLE_THRESHOLD,
    }


@app.post("/skill/{name}/error")
async def report_skill_error(name: str) -> Dict[str, Any]:
    """Increment error count for a probationary skill; disable it at DISABLE_THRESHOLD."""
    skill_path = SKILLS_DIR / f"hunted_{name}.md"
    if not skill_path.exists():
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    meta = _load_meta(name)
    meta.setdefault("error_count", 0)
    meta["error_count"] += 1
    if meta["error_count"] >= DISABLE_THRESHOLD:
        meta["disabled"] = True
    _save_meta(name, meta)
    return {
        "name": name,
        "error_count": meta["error_count"],
        "disabled": meta.get("disabled", False),
        "threshold": DISABLE_THRESHOLD,
    }
