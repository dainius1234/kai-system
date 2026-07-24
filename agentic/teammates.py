"""D89: Persistent Teammates — named cognitive personas for Kai.

Each teammate is defined by a Markdown file in data/teammates/{slug}.md with:
    # Name
    **Specialty:** <domain>
    **Description:** <one-liner>
    ## System Prompt
    <persona instructions>

Teammates are loaded at startup and available via get_teammate() / list_teammates().
The POST /chat/teammate/{name} endpoint in app.py invokes them with full context.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("kai.teammates")

TEAMMATES_DIR = Path(__file__).parent.parent / "data" / "teammates"


@dataclass
class TeammateDef:
    slug: str
    name: str
    specialty: str
    description: str
    system_prompt: str


def _parse_teammate_md(slug: str, text: str) -> TeammateDef:
    lines = text.splitlines()
    name = slug.title()
    specialty = slug
    description = ""
    system_prompt = text

    for i, line in enumerate(lines):
        if line.startswith("# "):
            name = line[2:].strip()
        elif line.startswith("**Specialty:**"):
            specialty = line.split(":", 1)[-1].strip().lstrip("* ").strip()
        elif line.startswith("**Description:**"):
            description = line.split(":", 1)[-1].strip().lstrip("* ").strip()
        elif line.strip() == "## System Prompt":
            system_prompt = "\n".join(lines[i + 1:]).strip()
            break

    return TeammateDef(
        slug=slug,
        name=name,
        specialty=specialty,
        description=description,
        system_prompt=system_prompt,
    )


_registry: Dict[str, TeammateDef] = {}


def load_teammates() -> None:
    global _registry
    loaded: Dict[str, TeammateDef] = {}
    if not TEAMMATES_DIR.exists():
        logger.warning("Teammates directory not found: %s", TEAMMATES_DIR)
        _registry = loaded
        return
    for md_file in sorted(TEAMMATES_DIR.glob("*.md")):
        slug = md_file.stem
        try:
            text = md_file.read_text(encoding="utf-8")
            loaded[slug] = _parse_teammate_md(slug, text)
        except Exception as exc:
            logger.warning("Failed to load teammate %s: %s", slug, exc)
    _registry = loaded
    logger.info("Teammates loaded: %s", list(_registry.keys()))


def get_teammate(slug: str) -> Optional[TeammateDef]:
    return _registry.get(slug)


def list_teammates() -> List[Dict]:
    return [
        {
            "slug": t.slug,
            "name": t.name,
            "specialty": t.specialty,
            "description": t.description,
        }
        for t in _registry.values()
    ]


def build_teammate_context(slug: str) -> Optional[str]:
    """Return formatted system prompt block for the named teammate."""
    t = _registry.get(slug)
    if not t:
        return None
    return f"[Teammate: {t.name} | Specialty: {t.specialty}]\n\n{t.system_prompt}"
