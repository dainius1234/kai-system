"""D123: Web Scout — Kai's independent information gathering layer.

Phase 2: Self-Preservation. Kai can browse the web to gather context it
can't get from its local sensors — news, prices, documentation, any
publicly accessible information. This breaks the operator-as-information-
bottleneck dependency.

Trust gating (per trust ladder in trust_core.py):
    fetch/search (operator-directed, /web-scout endpoints): ASSISTANT (2)
    autonomous use (Kai initiates during pipeline/observer): PARTNER (4)

Feature-flagged: FF_WEB_SCOUT=true
Fail-open: all methods return error dicts on network/parse failures.
Never stores credentials. Never follows redirects to non-http(s) schemes.

Search backend: DuckDuckGo Instant Answers API (no key required).
Content extraction: stdlib html.parser — no extra dependencies.
"""
from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("kai.web_scout")

_DEFAULT_TIMEOUT_S = 8.0
_MAX_CONTENT_CHARS = 4000
_USER_AGENT = "KaiWebScout/1.0 (research assistant; contact operator)"
_DDG_API = "https://api.duckduckgo.com/"
_SAFE_SCHEMES = {"http", "https"}


# ── HTML text extractor ────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML tags; collect visible text only."""
    _SKIP = {"script", "style", "head", "meta", "link", "noscript", "svg", "iframe"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP:
            self._depth = max(0, self._depth - 1)

    def handle_data(self, data: str) -> None:
        if self._depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def text(self, max_chars: int = _MAX_CONTENT_CHARS) -> str:
        raw = " ".join(self._parts)
        # Collapse runs of whitespace
        clean = re.sub(r"\s{2,}", " ", raw)
        return clean[:max_chars]


def _extract_text(html: str, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    ex = _TextExtractor()
    try:
        ex.feed(html)
    except Exception:
        pass
    return ex.text(max_chars)


def _safe_url(url: str) -> str:
    """Raise ValueError if the URL scheme is not http/https."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _SAFE_SCHEMES:
        raise ValueError(f"Unsafe URL scheme: {parsed.scheme!r} — only http/https allowed")
    return url


# ── Result dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    url: str
    status_code: int
    content: str                # extracted visible text
    content_length: int         # raw response bytes
    elapsed_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    query: str
    abstract: str               # DuckDuckGo abstract text
    abstract_url: str
    topics: List[Dict[str, str]]   # [{title, url}, ...]
    answer: str                 # instant answer if present
    elapsed_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Trust gate helper ──────────────────────────────────────────────────────────

def _check_trust(capability: str, context: Dict[str, Any]) -> None:
    """Raise PermissionError if trust gate denies. Fail-open if infra is missing."""
    try:
        try:
            from trust_integration import gate_autonomous_action
        except ImportError:
            from agentic.trust_integration import gate_autonomous_action  # type: ignore
        allowed, reason = gate_autonomous_action(capability, context, conviction=6.0)
        if not allowed:
            raise PermissionError(f"Web scout trust gate denied {capability}: {reason}")
    except PermissionError:
        raise
    except Exception as exc:
        logger.debug("Trust gate unavailable (fail-open for web_scout): %s", exc)


# ── Core operations ────────────────────────────────────────────────────────────

def fetch(
    url: str,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_chars: int = _MAX_CONTENT_CHARS,
    autonomous: bool = False,
) -> FetchResult:
    """Fetch a URL and return extracted visible text.

    Args:
        url: must be http or https
        timeout_s: per-request timeout
        max_chars: max characters of extracted text to return
        autonomous: True = Kai initiated (needs PARTNER trust);
                    False = operator-directed (needs ASSISTANT trust)

    Never raises — returns FetchResult with error set on failure.
    """
    capability = "web_scout_autonomous_fetch" if autonomous else "web_scout_fetch"
    try:
        _safe_url(url)
        _check_trust(capability, {"url": url, "autonomous": autonomous})
    except PermissionError as exc:
        return FetchResult(url=url, status_code=0, content="", content_length=0,
                           elapsed_ms=0.0, error=str(exc))
    except ValueError as exc:
        return FetchResult(url=url, status_code=0, content="", content_length=0,
                           elapsed_ms=0.0, error=str(exc))

    t0 = time.monotonic()
    try:
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=timeout_s,
        ) as client:
            resp = client.get(url)
        elapsed = (time.monotonic() - t0) * 1000
        content_length = len(resp.content)
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or not content_type:
            text = _extract_text(resp.text, max_chars)
        else:
            # Non-HTML (JSON, plain text, etc.) — return as-is, truncated
            text = resp.text[:max_chars]
        logger.info("WebScout fetch: %s status=%d chars=%d", url, resp.status_code, len(text))
        return FetchResult(
            url=str(resp.url),
            status_code=resp.status_code,
            content=text,
            content_length=content_length,
            elapsed_ms=round(elapsed, 1),
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.debug("WebScout fetch failed for %s: %s", url, exc)
        return FetchResult(url=url, status_code=0, content="", content_length=0,
                           elapsed_ms=round(elapsed, 1), error=str(exc))


def search(
    query: str,
    max_results: int = 5,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    autonomous: bool = False,
) -> SearchResult:
    """Search using DuckDuckGo Instant Answers (no API key required).

    Returns an abstract summary and related topic links.
    Never raises — returns SearchResult with error set on failure.
    """
    capability = "web_scout_autonomous_search" if autonomous else "web_scout_search"
    try:
        _check_trust(capability, {"query": query[:200], "autonomous": autonomous})
    except PermissionError as exc:
        return SearchResult(query=query, abstract="", abstract_url="", topics=[],
                            answer="", elapsed_ms=0.0, error=str(exc))

    t0 = time.monotonic()
    try:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "no_redirect": "1",
        }
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            timeout=timeout_s,
        ) as client:
            resp = client.get(_DDG_API, params=params)
        elapsed = (time.monotonic() - t0) * 1000
        data = resp.json()

        abstract = data.get("AbstractText", "") or data.get("Abstract", "")
        abstract_url = data.get("AbstractURL", "")
        answer = data.get("Answer", "")

        topics: List[Dict[str, str]] = []
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                topics.append({
                    "title": topic.get("Text", "")[:200],
                    "url": topic.get("FirstURL", ""),
                })

        logger.info("WebScout search: %r → abstract=%d topics=%d", query, len(abstract), len(topics))
        return SearchResult(
            query=query,
            abstract=abstract[:_MAX_CONTENT_CHARS],
            abstract_url=abstract_url,
            topics=topics,
            answer=answer[:500],
            elapsed_ms=round(elapsed, 1),
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        logger.debug("WebScout search failed for %r: %s", query, exc)
        return SearchResult(query=query, abstract="", abstract_url="", topics=[],
                            answer="", elapsed_ms=round(elapsed, 1), error=str(exc))


def summarize(
    url: str,
    max_chars: int = 1500,
    autonomous: bool = False,
) -> Dict[str, Any]:
    """Fetch a URL and return a trimmed summary dict."""
    result = fetch(url, max_chars=max_chars, autonomous=autonomous)
    return {
        "url": result.url,
        "summary": result.content,
        "status_code": result.status_code,
        "content_length": result.content_length,
        "elapsed_ms": result.elapsed_ms,
        "error": result.error,
    }
