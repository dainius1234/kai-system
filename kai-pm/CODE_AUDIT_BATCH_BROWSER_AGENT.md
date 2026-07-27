# Kai Code Audit — Browser Agent Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BROWSER-001 | CRITICAL | Unauthenticated callers can navigate the browser to arbitrary URLs |
| KAI-BROWSER-002 | CRITICAL | Unauthenticated callers can click and type into the active page |
| KAI-BROWSER-003 | CRITICAL | Unauthenticated callers can scrape and screenshot the shared active page |
| KAI-BROWSER-004 | HIGH | All callers share one browser page and browsing state |
| KAI-BROWSER-005 | HIGH | Arbitrary navigation enables browser-mediated access to internal and local resources |
| KAI-BROWSER-006 | HIGH | Scrape text is fully materialised before truncation |
| KAI-BROWSER-007 | HIGH | Screenshot size and page-rendering cost are unbounded |
| KAI-BROWSER-008 | HIGH | One global lock allows slow navigation to block every browser operation |
| KAI-BROWSER-009 | MEDIUM | Navigation wait mode is caller-controlled and unvalidated |
| KAI-BROWSER-010 | MEDIUM | Error responses expose browser and network diagnostics |
| KAI-BROWSER-011 | MEDIUM | Search query construction does not perform proper URL encoding |
| KAI-BROWSER-012 | MEDIUM | Search queries are written to logs without redaction |
| KAI-BROWSER-013 | MEDIUM | Click and type fields are unbounded |
| KAI-BROWSER-014 | MEDIUM | Browser lifecycle is lazy and health reports ok in stub/unready states |
| KAI-BROWSER-015 | MEDIUM | Error-budget recording passes a Boolean and omits raised exceptions |
| KAI-BROWSER-016 | MEDIUM | Configuration values are not validated |

---

## Browser agent: `browser-agent/app.py`

### KAI-BROWSER-001 — CRITICAL — Arbitrary unauthenticated navigation
**Issue:** `POST /navigate` and `POST /run` require no authentication or authorisation. Caller-controlled URLs are passed directly to Playwright `page.goto`.  
**Risk:** Any reachable caller can direct the service browser to arbitrary internet, intranet, loopback or local-service destinations and retain the resulting page as the global active state.  
**Recommendation:** Require authenticated capability-scoped access and enforce canonical destination allowlists and network egress controls.  
**Status:** OPEN — immediate remediation required

### KAI-BROWSER-002 — CRITICAL — Remote interaction with active pages
**Issue:** `POST /click` and `POST /type` are unauthenticated. They operate on the current shared page using caller-controlled selectors, visible text and entered text.  
**Risk:** A caller can submit forms, alter settings, trigger purchases/actions, overwrite form fields or interact with any authenticated/internal page opened by another workflow.  
**Recommendation:** Isolate sessions per authorised principal and require explicit approval for consequential browser interactions.  
**Status:** OPEN — immediate remediation required

### KAI-BROWSER-003 — CRITICAL — Active-page content and screenshots are public
**Issue:** `POST /scrape`, `POST /screenshot` and `POST /run` return page text, links or image bytes without authentication.  
**Risk:** Any caller can inspect the page left open by another user or service, including confidential internal pages, authenticated content and sensitive on-screen data.  
**Recommendation:** Require session ownership checks and prevent cross-principal page access.  
**Status:** OPEN — immediate remediation required

### KAI-BROWSER-004 — HIGH — One global page is shared by every caller
**Issue:** `_page` is a single module-level Playwright page reused for every endpoint. There is no caller identity, context, tab or session separation.  
**Risk:** Navigation, clicks, typing, scraping and screenshots race through one mutable browser state. One caller can observe or alter another caller’s workflow.  
**Recommendation:** Create isolated browser contexts/pages per authenticated session with strict lifecycle and storage separation.  
**Status:** OPEN

### KAI-BROWSER-005 — HIGH — Browser-mediated SSRF and local-network reach
**Issue:** No scheme, host, DNS, resolved-IP, redirect or port restrictions are applied before navigation. Browser redirects are also unrestricted.  
**Risk:** The browser can reach network locations unavailable to external callers and render internal administrative or metadata endpoints. Subsequent scrape/screenshot endpoints exfiltrate the result.  
**Recommendation:** Enforce destination policy before and after redirects and block loopback, link-local, private and disallowed address ranges unless explicitly approved.  
**Status:** OPEN

### KAI-BROWSER-006 — HIGH — Scrape truncation occurs after full allocation
**Issue:** `/scrape` and `/run` evaluate `document.body.innerText`, receiving the complete page text in Python, and only then slice it to `MAX_SCRAPE_CHARS`.  
**Risk:** Extremely large DOM text can consume browser, transport and Python memory despite the configured output limit.  
**Recommendation:** Extract bounded text inside the page context or stream/chunk with strict limits.  
**Status:** OPEN

### KAI-BROWSER-007 — HIGH — Screenshot and rendering resources are unbounded
**Issue:** `/screenshot` returns `page.screenshot` bytes without size checks. Navigation imposes no page-weight, download, script, CPU or memory limits beyond timeout.  
**Risk:** Hostile pages can consume browser resources or produce large screenshot allocations, degrading or crashing the service.  
**Recommendation:** Run the browser in an isolated resource-limited container and enforce page, request and screenshot bounds.  
**Status:** OPEN

### KAI-BROWSER-008 — HIGH — Global lock creates a service-wide denial-of-service point
**Issue:** Every stateful browser endpoint holds one `_lock` across navigation, DOM evaluation or interaction. Navigation may hold it for `NAV_TIMEOUT`, while hostile pages can prolong operations and repeated callers queue behind it.  
**Risk:** One slow request blocks all browser users and dependent services.  
**Recommendation:** Use isolated per-session workers with bounded queues, global concurrency limits and cancellation.  
**Status:** OPEN

### KAI-BROWSER-009 — MEDIUM — Caller controls Playwright wait mode
**Issue:** `NavigateRequest.wait_until` is an arbitrary string passed directly to `page.goto`.  
**Risk:** Invalid values create avoidable errors; supported but unsuitable modes can extend blocking behaviour or change navigation semantics.  
**Recommendation:** Restrict the field to a validated enum.  
**Status:** OPEN

### KAI-BROWSER-010 — MEDIUM — Internal diagnostics are returned
**Issue:** Navigation, click, type, scrape, screenshot, run and search exceptions are interpolated directly into HTTP 502 details.  
**Risk:** Callers receive internal URLs, selectors, browser state, filesystem and network diagnostics useful for reconnaissance.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-BROWSER-011 — MEDIUM — Search query is not safely URL-encoded
**Issue:** DuckDuckGo search builds a URL using only `req.query.replace(' ', '+')`. Reserved characters such as `&`, `#`, `?`, `+` and `%` are not encoded.  
**Risk:** Query meaning can be altered, extra parameters injected or fragments introduced, producing incorrect or attacker-shaped navigation.  
**Recommendation:** Construct query parameters using a URL encoder or Playwright request parameters.  
**Status:** OPEN

### KAI-BROWSER-012 — MEDIUM — Search content is logged
**Issue:** The complete search query is logged at information level.  
**Risk:** Sensitive names, internal terms, health/legal queries or credentials accidentally pasted into search become persistent log content.  
**Recommendation:** Redact or hash query values and log only operational metadata.  
**Status:** OPEN

### KAI-BROWSER-013 — MEDIUM — Interaction fields lack bounds
**Issue:** URL, task, selectors, visible text, typed text and search query have no maximum lengths.  
**Risk:** Oversized values consume request, Playwright, logging and browser resources and can create long selector parsing operations.  
**Recommendation:** Enforce strict per-field and aggregate request limits.  
**Status:** OPEN

### KAI-BROWSER-014 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when Playwright is absent or the browser has never launched. `browser_ready` being false does not change the status.  
**Risk:** Orchestration treats a stubbed or non-started browser as ready. Browser launch failures are only discovered on first use.  
**Recommendation:** Separate liveness, dependency availability and verified browser readiness.  
**Status:** OPEN

### KAI-BROWSER-015 — MEDIUM — Reliability telemetry is incomplete
**Issue:** Middleware passes `response.status_code >= 500` to `budget.record`, rather than the actual status code, and does not record exceptions raised before a response.  
**Risk:** Error-budget reporting can misclassify or omit failures.  
**Recommendation:** Record actual status codes and exception outcomes consistently.  
**Status:** OPEN

### KAI-BROWSER-016 — MEDIUM — Startup configuration is unvalidated
**Issue:** Port, navigation timeout and scrape limit are parsed directly; headless mode accepts any value except exact `false` as true. Zero, negative or extreme limits can produce startup or runtime failure.  
**Risk:** Misconfiguration creates immediate errors, unlimited/invalid behaviour or unexpected visible browser operation.  
**Recommendation:** Validate typed configuration with explicit ranges and accepted Boolean values.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **3**
- High: **5**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **492**
- Critical: **54**
- High: **179**
- Medium: **256**
- Low: **3**

## Files materially reviewed in this batch

`browser-agent/app.py`.
