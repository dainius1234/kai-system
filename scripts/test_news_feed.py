"""News-feed service tests — feedparser and httpx are stubbed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from scripts.module_stubs import stubbed  # noqa: E402
import time
import types
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Stubs ─────────────────────────────────────────────────────────────────────

def _make_feedparser_stub(entries=None):
    stub = types.ModuleType("feedparser")
    if entries is None:
        import time as _time
        entries = [
            MagicMock(
                title="Test Article",
                link="https://example.com/1",
                summary="Summary of test article",
                published_parsed=_time.gmtime(),
                updated_parsed=None,
                get=lambda k, d=None: {
                    "title": "Test Article",
                    "link": "https://example.com/1",
                    "summary": "Summary of test article",
                    "published_parsed": _time.gmtime(),
                    "updated_parsed": None,
                }.get(k, d),
            )
        ]

    parsed = MagicMock()
    parsed.entries = entries

    def feedparse(content):
        return parsed

    stub.parse = feedparse
    return stub


def _make_httpx_stub(content=b"<rss></rss>", status=200):
    stub = types.ModuleType("httpx")

    class FakeResp:
        def __init__(self):
            self.status_code = status
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def get(self, url, **kwargs):
            return FakeResp()

    stub.AsyncClient = FakeClient
    stub.HTTPStatusError = Exception
    stub.RequestError = Exception
    return stub


def _load_module(monkeypatch, with_feedparser=True):
    mod_name = "news_feed_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    stubs = {"httpx": _make_httpx_stub()}
    if with_feedparser:
        stubs["feedparser"] = _make_feedparser_stub()
    elif "feedparser" in sys.modules:
        del sys.modules["feedparser"]

    monkeypatch.setenv("PORT", "8038")
    monkeypatch.setenv("REFRESH_INTERVAL_SECONDS", "9999")  # disable auto-refresh in tests

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent.parent / "news-feed" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # `httpx` is a real, installed library. Replacing it permanently
    # here broke every TestClient in every suite collected after this
    # one. See scripts/module_stubs.py.
    with stubbed(stubs):
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(monkeypatch):
    mod = _load_module(monkeypatch)
    # Pre-seed feeds without starting the background refresh loop
    feed_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://example.com/feed.rss"))
    mod._feeds[feed_id] = {
        "url": "https://example.com/feed.rss",
        "name": "Example Feed",
        "tags": ["test"],
        "last_fetched": None,
        "error": None,
    }
    return TestClient(mod.app, raise_server_exceptions=True), mod


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "feeds" in body
    assert "articles" in body


def test_metrics(client):
    c, _ = client
    r = c.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_list_feeds(client):
    c, _ = client
    r = c.get("/feeds")
    assert r.status_code == 200
    body = r.json()
    assert "feeds" in body
    feeds = body["feeds"]
    assert any(f["name"] == "Example Feed" for f in feeds)


def test_add_feed(client):
    c, mod = client
    r = c.post("/feeds", json={"url": "https://new.example.com/rss", "name": "New Feed", "tags": ["news"]})
    assert r.status_code == 200
    body = r.json()
    assert "id" in body
    assert body["url"] == "https://new.example.com/rss"
    # Feed should be registered
    assert body["id"] in mod._feeds


def test_add_feed_no_name(client):
    c, mod = client
    r = c.post("/feeds", json={"url": "https://auto.example.com/rss"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"] in mod._feeds
    # Name defaults to URL when not given
    assert mod._feeds[body["id"]]["name"] == "https://auto.example.com/rss"


def test_remove_feed(client):
    c, mod = client
    # Add a feed then remove it
    add_r = c.post("/feeds", json={"url": "https://remove.me/rss"})
    fid = add_r.json()["id"]
    assert fid in mod._feeds
    del_r = c.delete(f"/feeds/{fid}")
    assert del_r.status_code == 200
    assert fid not in mod._feeds


def test_remove_nonexistent_feed(client):
    c, _ = client
    r = c.delete("/feeds/doesnotexist")
    assert r.status_code == 404


def test_articles_empty_initially(client):
    c, _ = client
    r = c.get("/articles")
    assert r.status_code == 200
    body = r.json()
    assert "articles" in body
    assert isinstance(body["articles"], list)


def test_articles_populated_after_refresh(client):
    c, mod = client
    # Manually inject an article
    mod._articles = [{
        "id": "art1",
        "feed_id": "f1",
        "feed_name": "Test Feed",
        "tags": ["test"],
        "title": "Test Article",
        "url": "https://example.com/1",
        "summary": "A summary",
        "published_ts": time.time(),
    }]
    r = c.get("/articles")
    assert r.status_code == 200
    assert len(r.json()["articles"]) == 1


def test_articles_limit(client):
    c, mod = client
    mod._articles = [
        {"id": f"a{i}", "feed_id": "f", "feed_name": "F", "tags": [], "title": f"A{i}",
         "url": f"https://x.com/{i}", "summary": "", "published_ts": float(i)}
        for i in range(30)
    ]
    r = c.get("/articles?limit=5")
    assert r.status_code == 200
    assert len(r.json()["articles"]) == 5


def test_articles_tag_filter(client):
    c, mod = client
    mod._articles = [
        {"id": "a1", "feed_id": "f", "feed_name": "F", "tags": ["tech"], "title": "Tech",
         "url": "https://x.com/1", "summary": "", "published_ts": 1.0},
        {"id": "a2", "feed_id": "f", "feed_name": "F", "tags": ["news"], "title": "News",
         "url": "https://x.com/2", "summary": "", "published_ts": 2.0},
    ]
    r = c.get("/articles?tag=tech")
    articles = r.json()["articles"]
    assert len(articles) == 1
    assert articles[0]["title"] == "Tech"


def test_search_hit(client):
    c, mod = client
    mod._articles = [{
        "id": "s1", "feed_id": "f", "feed_name": "F", "tags": [],
        "title": "Python release 3.14",
        "url": "https://python.org/news",
        "summary": "New Python version released",
        "published_ts": time.time(),
    }]
    r = c.get("/search?q=python")
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 1
    assert "Python" in results[0]["title"]


def test_search_no_hits(client):
    c, mod = client
    mod._articles = [{"id": "x", "feed_id": "f", "feed_name": "F", "tags": [],
                      "title": "Totally unrelated", "url": "https://x.com",
                      "summary": "", "published_ts": 0.0}]
    r = c.get("/search?q=xyzzy_notfound")
    assert r.json()["results"] == []


def test_search_missing_query(client):
    c, _ = client
    r = c.get("/search")
    assert r.status_code == 422  # q is required


def test_parse_feed_id_deterministic(client):
    c, mod = client
    fid1 = mod._parse_feed_id("https://example.com/rss")
    fid2 = mod._parse_feed_id("https://example.com/rss")
    assert fid1 == fid2


def test_parse_feed_id_different_urls(client):
    c, mod = client
    fid1 = mod._parse_feed_id("https://example.com/rss")
    fid2 = mod._parse_feed_id("https://other.com/rss")
    assert fid1 != fid2
