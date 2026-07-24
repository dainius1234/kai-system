"""Browser Agent — Playwright-based web navigation for Kai.

Endpoints:
  POST /navigate    {url, wait_until?}              → {title, url, status}
  POST /click       {selector?, text?}              → {ok}
  POST /type        {selector, text}                → {ok}
  POST /scrape      {}                              → {title, url, text, links}
  POST /screenshot  {}                              → image/png
  POST /run         {task, url?}                    → {result, title, url, steps}
  POST /search      {query, max_results?}           → {query, results: [{title, url, snippet}]}
  GET  /health
  GET  /metrics
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("browser-agent", os.getenv("LOG_PATH", "/tmp/browser-agent.json.log"))
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("browser-agent")

    class ErrorBudget:  # type: ignore[no-redef]
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}

try:
    from playwright.async_api import async_playwright, Browser, Page
    _PLAYWRIGHT_OK = True
except ImportError:
    _PLAYWRIGHT_OK = False
    logger.info("playwright not available — browser-agent in stub mode")

PORT = int(os.getenv("PORT", "8040"))
NAV_TIMEOUT = int(os.getenv("NAV_TIMEOUT_MS", "20000"))
MAX_SCRAPE_CHARS = int(os.getenv("MAX_SCRAPE_CHARS", "50000"))
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "true").lower() != "false"

@asynccontextmanager
async def _lifespan(application: FastAPI):
    yield
    global _browser, _playwright_inst
    if _browser:
        await _browser.close()
    if _playwright_inst:
        await _playwright_inst.stop()


app = FastAPI(title="browser-agent", lifespan=_lifespan)
budget = ErrorBudget(window_seconds=300)

_playwright_inst = None
_browser: Optional["Browser"] = None
_page: Optional["Page"] = None
_lock = asyncio.Lock()


async def _get_page() -> "Page":
    global _playwright_inst, _browser, _page
    if _page and not _page.is_closed():
        return _page
    if not _PLAYWRIGHT_OK:
        raise RuntimeError("playwright not installed")
    if _playwright_inst is None:
        _playwright_inst = await async_playwright().start()
    if _browser is None or not _browser.is_connected():
        _browser = await _playwright_inst.chromium.launch(headless=BROWSER_HEADLESS)
    _page = await _browser.new_page()
    return _page


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    response = await call_next(request)
    budget.record(response.status_code >= 500)
    return response


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "playwright": _PLAYWRIGHT_OK,
        "browser_ready": _browser is not None and _browser.is_connected() if _browser else False,
    }


@app.get("/metrics")
async def metrics():
    return budget.snapshot()


class NavigateRequest(BaseModel):
    url: str
    wait_until: str = "domcontentloaded"


@app.post("/navigate")
async def navigate(req: NavigateRequest):
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    async with _lock:
        page = await _get_page()
        try:
            resp = await page.goto(req.url, timeout=NAV_TIMEOUT, wait_until=req.wait_until)
            return {"title": await page.title(), "url": page.url, "status": resp.status if resp else 0}
        except Exception as exc:
            raise HTTPException(502, f"Navigation failed: {exc}")


class ClickRequest(BaseModel):
    selector: Optional[str] = None
    text: Optional[str] = None


@app.post("/click")
async def click(req: ClickRequest):
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    if not req.selector and not req.text:
        raise HTTPException(400, "provide selector or text")
    async with _lock:
        page = await _get_page()
        try:
            if req.selector:
                await page.click(req.selector, timeout=5000)
            else:
                await page.get_by_text(req.text, exact=False).first.click(timeout=5000)
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(502, f"Click failed: {exc}")


class TypeRequest(BaseModel):
    selector: str
    text: str


@app.post("/type")
async def type_text(req: TypeRequest):
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    async with _lock:
        page = await _get_page()
        try:
            await page.fill(req.selector, req.text, timeout=5000)
            return {"ok": True}
        except Exception as exc:
            raise HTTPException(502, f"Type failed: {exc}")


@app.post("/scrape")
async def scrape():
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    async with _lock:
        page = await _get_page()
        try:
            title = await page.title()
            text = await page.evaluate("document.body.innerText")
            links = await page.evaluate(
                "Array.from(document.querySelectorAll('a[href]')).slice(0,50)"
                ".map(a=>({text:a.innerText.trim().slice(0,120),href:a.href}))"
            )
            return {"title": title, "url": page.url, "text": text[:MAX_SCRAPE_CHARS], "links": links}
        except Exception as exc:
            raise HTTPException(502, f"Scrape failed: {exc}")


@app.post("/screenshot")
async def screenshot():
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    async with _lock:
        page = await _get_page()
        try:
            data = await page.screenshot(type="png", full_page=False)
            return Response(content=data, media_type="image/png")
        except Exception as exc:
            raise HTTPException(502, f"Screenshot failed: {exc}")


class RunRequest(BaseModel):
    task: str
    url: Optional[str] = None


@app.post("/run")
async def run_task(req: RunRequest):
    """Navigate to URL (if given) then scrape — returns page text as task result."""
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    steps: list[str] = []
    async with _lock:
        page = await _get_page()
        try:
            if req.url:
                resp = await page.goto(req.url, timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
                steps.append(f"navigated → {page.url} (HTTP {resp.status if resp else '?'})")
            title = await page.title()
            text = await page.evaluate("document.body.innerText")
            steps.append(f"scraped '{title}' ({len(text):,} chars)")
            return {
                "result": text[:MAX_SCRAPE_CHARS],
                "title": title,
                "url": page.url,
                "steps": steps,
            }
        except Exception as exc:
            raise HTTPException(502, f"Task failed: {exc}")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 10


@app.post("/search")
async def search(req: SearchRequest):
    """Search DuckDuckGo and return structured results."""
    if not _PLAYWRIGHT_OK:
        raise HTTPException(503, "playwright not available")
    if not req.query.strip():
        raise HTTPException(400, "query is required")
    max_r = max(1, min(req.max_results, 30))
    async with _lock:
        page = await _get_page()
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={req.query.replace(' ', '+')}"
            await page.goto(ddg_url, timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
            results = await page.evaluate(f"""
                Array.from(document.querySelectorAll('.result')).slice(0, {max_r}).map(r => {{
                    const a = r.querySelector('.result__a');
                    const snip = r.querySelector('.result__snippet');
                    const url_el = r.querySelector('.result__url');
                    return {{
                        title: a ? a.innerText.trim() : '',
                        url: a ? a.href : (url_el ? url_el.innerText.trim() : ''),
                        snippet: snip ? snip.innerText.trim() : ''
                    }};
                }}).filter(r => r.title)
            """)
            logger.info("search '%s' → %d results", req.query, len(results))
            return {"query": req.query, "results": results}
        except Exception as exc:
            raise HTTPException(502, f"Search failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
