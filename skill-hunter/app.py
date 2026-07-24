"""skill-hunter — D88 M6: autonomous capability growth service.

Watches for capability gaps (phrases Kai can't handle), searches PyPI for
relevant packages, generates .md skill files, and signals agentic to hot-reload.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Skill Hunter", version="0.1.0")

SKILLS_DIR = Path(os.getenv("SKILLS_DIR", "/data/skills"))
PYPI_BASE = "https://pypi.org/pypi"
PORT = int(os.getenv("PORT", "8045"))

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


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "skill-hunter", "port": PORT}


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
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{PYPI_BASE}/{package}/json")
            return r.status_code == 200
    except Exception:
        return False


def _skill_name(gap: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", gap.lower())[:30].strip("_")


def _generate_skill_md(gap: str, package: str) -> str:
    name = _skill_name(gap)
    return (
        f"---\nname: {name}\nversion: 1\nsource: skill-hunter\npackage: {package}\n---\n\n"
        f"# Skill: {name}\n\n"
        f"## When to use\nWhen the operator asks about or needs: {gap}\n\n"
        f"## Action\nUse the `{package}` Python package.\n\n"
        f"Install if needed: `pip install {package}`\n\n"
        f"## Response template\n"
        f"I can help with that using the `{package}` package. Here is how to approach it:\n"
    )


@app.post("/hunt")
async def hunt(req: HuntRequest) -> Dict[str, Any]:
    """Search for a package that fills the capability gap; generate a skill file."""
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
    skill_path = SKILLS_DIR / f"hunted_{name}.md"
    skill_path.write_text(_generate_skill_md(gap, found), encoding="utf-8")

    return {
        "skill_created": True,
        "skill_name": name,
        "package": found,
        "skill_path": str(skill_path),
        "gap": gap,
        "searched": candidates,
    }


@app.get("/skills")
async def list_hunted_skills() -> Dict[str, Any]:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skills = sorted(f.stem for f in SKILLS_DIR.glob("hunted_*.md"))
    return {"status": "ok", "skills": skills, "count": len(skills)}
