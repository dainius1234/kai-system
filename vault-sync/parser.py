"""D91: Obsidian vault note parser.

Extracts frontmatter, wikilinks, tags, and body from .md files into NoteData.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_TAG_RE = re.compile(r"(?<!\w)#([\w/]+)")


@dataclass
class NoteData:
    filepath: str
    title: str
    frontmatter: Dict
    content: str
    wikilinks: List[Tuple[str, str]]   # (target, alias)
    tags: List[str]
    modified_at: float
    checksum: str


def parse_note(filepath: str) -> Optional[NoteData]:
    """Parse a vault .md file into NoteData. Returns None if file unreadable."""
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return None

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Split frontmatter
    fm: Dict = {}
    content = raw
    if raw.startswith("---"):
        try:
            end = raw.find("\n---", 3)
            if end > 0:
                import yaml
                fm = yaml.safe_load(raw[3:end]) or {}
                if not isinstance(fm, dict):
                    fm = {}
                content = raw[end + 4:].lstrip("\n")
        except Exception:
            pass

    # Extract wikilinks: [[target]] or [[target|alias]]
    wikilinks: List[Tuple[str, str]] = []
    for match in _WIKILINK_RE.findall(content):
        if "|" in match:
            target, alias = match.split("|", 1)
        else:
            target = alias = match
        wikilinks.append((target.strip(), alias.strip()))

    # Extract tags
    tags = _TAG_RE.findall(content)

    # Title from frontmatter or filename
    title = str(fm.get("title", path.stem))

    checksum = hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    return NoteData(
        filepath=str(path),
        title=title,
        frontmatter=fm,
        content=content,
        wikilinks=wikilinks,
        tags=tags,
        modified_at=path.stat().st_mtime,
        checksum=checksum,
    )
