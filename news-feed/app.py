"""News-feed service — RSS aggregation and keyword alerting.

Endpoints:
  GET  /health                    → {status, uptime_seconds}
  GET  /metrics                   → error budget snapshot
  GET  /feeds                     → list of configured feeds
  POST /feeds                     {url, name?, tags?}  → add feed
  DELETE /feeds/{feed_id}         → remove feed
  GET  /articles                  ?limit=20&tag=&since_minutes=  → recent articles
  GET  /search                    ?q=&limit=10  → keyword search across cached articles
  POST /refresh                   → force-refresh all feeds now
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("news-feed", os.getenv("LOG_PATH", "/tmp/news-feed.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("news-feed")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

try:
    import feedparser
    _FEEDPARSER_OK = True
except ImportError:
    _FEEDPARSER_OK = False
    logger.warning("feedparser not available — news-feed in stub mode")

PORT = int(os.getenv("PORT", "8038"))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL_SECONDS", "300"))
MAX_ARTICLES_PER_FEED = int(os.getenv("MAX_ARTICLES_PER_FEED", "50"))
MAX_CACHED_ARTICLES = int(os.getenv("MAX_CACHED_ARTICLES", "500"))

# Default feeds (override with SEED_FEEDS env var, comma-separated URLs)
DEFAULT_FEEDS = [
    {"url": "https://feeds.bbci.co.uk/news/rss.xml", "name": "BBC News", "tags": ["news", "world"]},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml", "name": "NYT Tech", "tags": ["tech"]},
    {"url": "https://hnrss.org/frontpage", "name": "Hacker News", "tags": ["tech", "startup"]},
]

_feeds: Dict[str, dict] = {}   # feed_id → {url, name, tags, last_fetched, error}
_articles: List[dict] = []     # flat cache of all articles
_refresh_task: Optional[asyncio.Task] = None
_start = time.time()


def _parse_feed_id(url: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, url))


async def _fetch_feed(feed_id: str, feed: dict) -> List[dict]:
    if not _FEEDPARSER_OK:
        return []
    url = feed["url"]
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            content = resp.content
        parsed = feedparser.parse(content)
        articles = []
        for entry in parsed.entries[:MAX_ARTICLES_PER_FEED]:
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            articles.append({
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, entry.get("link", "") + entry.get("title", ""))),
                "feed_id": feed_id,
                "feed_name": feed.get("name", url),
                "tags": feed.get("tags", []),
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "summary": (entry.get("summary", "") or "")[:500],
                "published_ts": time.mktime(published) if published else 0.0,
            })
        feed["last_fetched"] = time.time()
        feed["error"] = None
        logger.info("feed %s: fetched %d articles", feed.get("name"), len(articles))
        return articles
    except Exception as exc:
        feed["error"] = str(exc)
        logger.warning("feed %s fetch error: %s", feed.get("name"), exc)
        return []


async def _refresh_all():
    global _articles
    new_articles = []
    for feed_id, feed in list(_feeds.items()):
        arts = await _fetch_feed(feed_id, feed)
        new_articles.extend(arts)
    seen_ids = set()
    deduped = []
    for art in new_articles:
        if art["id"] not in seen_ids:
            seen_ids.add(art["id"])
            deduped.append(art)
    deduped.sort(key=lambda a: a["published_ts"], reverse=True)
    _articles = deduped[:MAX_CACHED_ARTICLES]
    logger.info("refresh complete: %d unique articles", len(_articles))


async def _refresh_loop():
    while True:
        try:
            await _refresh_all()
        except Exception as exc:
            logger.error("refresh loop error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _refresh_task
    # Seed default feeds
    for feed in DEFAULT_FEEDS:
        fid = _parse_feed_id(feed["url"])
        _feeds[fid] = {**feed, "last_fetched": None, "error": None}
    seed_env = os.getenv("SEED_FEEDS", "")
    for url in filter(None, seed_env.split(",")):
        fid = _parse_feed_id(url.strip())
        _feeds[fid] = {"url": url.strip(), "name": url.strip(), "tags": [], "last_fetched": None, "error": None}
    _refresh_task = asyncio.create_task(_refresh_loop())
    yield
    if _refresh_task:
        _refresh_task.cancel()


app = FastAPI(title="news-feed", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "feedparser": _FEEDPARSER_OK, "feeds": len(_feeds), "articles": len(_articles)}


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/feeds")
def list_feeds():
    return {"feeds": [{"id": fid, **feed} for fid, feed in _feeds.items()]}


class AddFeedRequest(BaseModel):
    url: str
    name: Optional[str] = None
    tags: List[str] = []


@app.post("/feeds")
async def add_feed(req: AddFeedRequest):
    fid = _parse_feed_id(req.url)
    _feeds[fid] = {"url": req.url, "name": req.name or req.url, "tags": req.tags, "last_fetched": None, "error": None}
    asyncio.create_task(_fetch_feed(fid, _feeds[fid]))
    return {"id": fid, "url": req.url}


@app.delete("/feeds/{feed_id}")
def remove_feed(feed_id: str):
    if feed_id not in _feeds:
        raise HTTPException(404, "feed not found")
    del _feeds[feed_id]
    return {"ok": True}


@app.get("/articles")
def articles(
    limit: int = Query(20, ge=1, le=200),
    tag: Optional[str] = Query(None),
    since_minutes: Optional[int] = Query(None),
    feed_id: Optional[str] = Query(None),
):
    result = _articles
    if tag:
        result = [a for a in result if tag in a.get("tags", [])]
    if feed_id:
        result = [a for a in result if a["feed_id"] == feed_id]
    if since_minutes is not None:
        cutoff = time.time() - since_minutes * 60
        result = [a for a in result if a["published_ts"] >= cutoff]
    return {"articles": result[:limit], "total": len(result)}


@app.get("/search")
def search(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=100)):
    q_lower = q.lower()
    hits = [
        a for a in _articles
        if q_lower in a["title"].lower() or q_lower in a["summary"].lower()
    ]
    return {"query": q, "results": hits[:limit]}


@app.post("/refresh")
async def force_refresh():
    await _refresh_all()
    return {"ok": True, "articles": len(_articles)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
