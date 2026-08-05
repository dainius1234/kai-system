"""Tests for D123: Web Scout — agentic/web_scout.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from web_scout import (
    _TextExtractor,
    _extract_text,
    _safe_url,
    fetch,
    search,
    summarize,
    FetchResult,
    SearchResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allow_trust(*a, **kw) -> None:
    """Patch _check_trust to always allow."""
    pass


def _deny_trust(*a, **kw) -> None:
    raise PermissionError("trust denied")


def _mock_httpx_get(html: str = "<p>Hello world</p>", status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.text = html
    resp.content = html.encode()
    resp.url = "https://example.com"
    resp.headers = {"content-type": "text/html"}
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


# ── _safe_url ─────────────────────────────────────────────────────────────────

def test_safe_url_accepts_http():
    assert _safe_url("http://example.com") == "http://example.com"


def test_safe_url_accepts_https():
    assert _safe_url("https://example.com/path?q=1") == "https://example.com/path?q=1"


def test_safe_url_rejects_ftp():
    with pytest.raises(ValueError, match="Unsafe URL scheme"):
        _safe_url("ftp://example.com/file")


def test_safe_url_rejects_file():
    with pytest.raises(ValueError, match="Unsafe URL scheme"):
        _safe_url("file:///etc/passwd")


def test_safe_url_rejects_javascript():
    with pytest.raises(ValueError, match="Unsafe URL scheme"):
        _safe_url("javascript:alert(1)")


# ── _TextExtractor ────────────────────────────────────────────────────────────

def test_text_extractor_strips_tags():
    ex = _TextExtractor()
    ex.feed("<p>Hello <b>world</b></p>")
    assert "Hello" in ex.text()
    assert "world" in ex.text()
    assert "<b>" not in ex.text()


def test_text_extractor_skips_script():
    ex = _TextExtractor()
    ex.feed("<p>Visible</p><script>alert('xss')</script>")
    assert "Visible" in ex.text()
    assert "alert" not in ex.text()


def test_text_extractor_skips_style():
    ex = _TextExtractor()
    ex.feed("<style>body{color:red}</style><p>Content</p>")
    assert "Content" in ex.text()
    assert "color" not in ex.text()


def test_text_extractor_converts_entities():
    ex = _TextExtractor()
    ex.feed("<p>Hello &amp; world</p>")
    assert "&" in ex.text()
    assert "&amp;" not in ex.text()


def test_text_extractor_caps_at_max_chars():
    ex = _TextExtractor()
    ex.feed("<p>" + "x" * 5000 + "</p>")
    assert len(ex.text(max_chars=100)) <= 100


def test_extract_text_from_real_html():
    html = """
    <html><head><title>Test</title></head>
    <body>
      <nav>nav content</nav>
      <p>Main paragraph text.</p>
      <script>var x = 1;</script>
    </body></html>
    """
    text = _extract_text(html)
    assert "Main paragraph text" in text
    assert "var x" not in text


def test_extract_text_empty_html():
    assert _extract_text("") == ""


def test_extract_text_no_tags():
    assert _extract_text("plain text") == "plain text"


# ── fetch() ───────────────────────────────────────────────────────────────────

def test_fetch_returns_content(tmp_path):
    html = "<html><body><p>Test content</p></body></html>"
    client = _mock_httpx_get(html)
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = fetch("https://example.com")
    assert isinstance(result, FetchResult)
    assert result.status_code == 200
    assert "Test content" in result.content
    assert result.error is None


def test_fetch_returns_error_on_network_failure():
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", side_effect=Exception("connection refused")):
        result = fetch("https://example.com")
    assert result.error is not None
    assert result.content == ""
    assert result.status_code == 0


def test_fetch_denied_on_trust_failure():
    with patch("web_scout._check_trust", _deny_trust):
        result = fetch("https://example.com")
    assert result.error is not None
    assert "trust" in result.error.lower() or "denied" in result.error.lower()


def test_fetch_rejects_unsafe_url():
    with patch("web_scout._check_trust", _allow_trust):
        result = fetch("ftp://example.com/file")
    assert result.error is not None
    assert result.status_code == 0


def test_fetch_caps_content_at_max_chars():
    html = "<p>" + "a" * 10000 + "</p>"
    client = _mock_httpx_get(html)
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = fetch("https://example.com", max_chars=200)
    assert len(result.content) <= 200


def test_fetch_non_html_content_returned_as_text():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"key": "value"}'
    resp.content = b'{"key": "value"}'
    resp.url = "https://api.example.com/data"
    resp.headers = {"content-type": "application/json"}
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = fetch("https://api.example.com/data")
    assert '"key"' in result.content


def test_fetch_records_elapsed_ms():
    client = _mock_httpx_get()
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = fetch("https://example.com")
    assert result.elapsed_ms >= 0.0


def test_fetch_result_to_dict():
    r = FetchResult(url="https://x.com", status_code=200, content="hello",
                    content_length=5, elapsed_ms=10.0)
    d = r.to_dict()
    assert d["url"] == "https://x.com"
    assert d["content"] == "hello"
    assert d["error"] is None


# ── search() ──────────────────────────────────────────────────────────────────

def _mock_ddg_response(abstract="Test abstract", topics=None):
    data = {
        "AbstractText": abstract,
        "Abstract": abstract,
        "AbstractURL": "https://en.wikipedia.org/wiki/Test",
        "Answer": "",
        "RelatedTopics": topics or [
            {"Text": "Topic 1", "FirstURL": "https://example.com/1"},
            {"Text": "Topic 2", "FirstURL": "https://example.com/2"},
        ],
    }
    resp = MagicMock()
    resp.status_code = 200
    resp.json = MagicMock(return_value=data)
    resp.headers = {"content-type": "application/json"}
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


def test_search_returns_abstract():
    client = _mock_ddg_response(abstract="Python is a programming language.")
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = search("python programming")
    assert isinstance(result, SearchResult)
    assert "Python" in result.abstract
    assert result.error is None


def test_search_returns_topics():
    client = _mock_ddg_response()
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = search("test query")
    assert len(result.topics) > 0
    assert "title" in result.topics[0]
    assert "url" in result.topics[0]


def test_search_max_results_caps_topics():
    topics = [{"Text": f"Topic {i}", "FirstURL": f"https://x.com/{i}"} for i in range(10)]
    client = _mock_ddg_response(topics=topics)
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = search("test", max_results=3)
    assert len(result.topics) <= 3


def test_search_error_on_network_failure():
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", side_effect=Exception("timeout")):
        result = search("test query")
    assert result.error is not None
    assert result.abstract == ""


def test_search_denied_on_trust_failure():
    with patch("web_scout._check_trust", _deny_trust):
        result = search("test query")
    assert result.error is not None


def test_search_result_to_dict():
    r = SearchResult(query="q", abstract="a", abstract_url="u", topics=[], answer="", elapsed_ms=5.0)
    d = r.to_dict()
    assert d["query"] == "q"
    assert d["abstract"] == "a"


# ── summarize() ───────────────────────────────────────────────────────────────

def test_summarize_returns_dict_with_summary():
    html = "<p>Summary content here</p>"
    client = _mock_httpx_get(html)
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", return_value=client):
        result = summarize("https://example.com", max_chars=500)
    assert "summary" in result
    assert "Summary content here" in result["summary"]


def test_summarize_returns_error_key_on_failure():
    with patch("web_scout._check_trust", _allow_trust), \
            patch("web_scout.httpx.Client", side_effect=Exception("err")):
        result = summarize("https://example.com")
    assert "error" in result
    assert result["error"] is not None
