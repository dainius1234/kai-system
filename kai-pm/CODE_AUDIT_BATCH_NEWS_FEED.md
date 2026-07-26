# Kai Code Audit — News Feed Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NEWS-001 | CRITICAL | Unauthenticated feed registration creates an SSRF primitive |
| KAI-NEWS-002 | HIGH | Failed or partial refresh can erase the entire article cache |
| KAI-NEWS-003 | HIGH | Feed mutation and forced refresh endpoints are unauthenticated |
| KAI-NEWS-004 | HIGH | Feed and article state is process-local and lost on restart |
| KAI-NEWS-005 | MEDIUM | Per-feed fetches are sequential and one slow feed delays all others |
| KAI-NEWS-006 | MEDIUM | Fire-and-forget feed fetch tasks are untracked and unsupervised |
| KAI-NEWS-007 | MEDIUM | Raw fetch errors and internal feed state are exposed through `/feeds` |
| KAI-NEWS-008 | MEDIUM | Published timestamps are converted using local process timezone semantics |
| KAI-NEWS-009 | MEDIUM | Feed input fields lack scheme, host, length and tag bounds |
| KAI-NEWS-010 | MEDIUM | Health reports `ok` in stub mode and during total feed failure |
| KAI-NEWS-011 | MEDIUM | Error-budget metrics are exposed but refresh outcomes are never recorded |

---

## News feed: `news-feed/app.py`

### KAI-NEWS-001 — CRITICAL — Feed registration enables SSRF
**Issue:** `POST /feeds` accepts any caller-supplied URL. `_fetch_feed` sends a server-side HTTP request to that URL with `follow_redirects=True`. No scheme, hostname, IP range, DNS result or redirect destination validation is applied.  
**Risk:** A reachable caller can make the service probe loopback, internal services, cloud metadata endpoints or other privileged network locations, including through redirect chains.  
**Recommendation:** Restrict feeds to an approved registry or validate canonical HTTPS destinations, resolved IP ranges and every redirect hop. Require authenticated administrative access.  
**Status:** OPEN — immediate remediation required

### KAI-NEWS-002 — HIGH — Refresh can erase all cached articles
**Issue:** `_refresh_all` builds a fresh list only from articles returned during the current refresh and then replaces `_articles`. `_fetch_feed` returns an empty list on any feed error. If all feeds fail, the complete prior cache is replaced with `[]`; partial failure silently removes all articles from failed feeds.  
**Risk:** A temporary upstream outage destroys the service’s current working dataset and can make downstream context believe no news exists.  
**Recommendation:** Update each feed independently, preserve last-known-good records with freshness metadata and publish a new aggregate only after validated refresh results.  
**Status:** OPEN

### KAI-NEWS-003 — HIGH — Administrative endpoints are unauthenticated
**Issue:** Adding feeds, deleting feeds and forcing a full refresh require no authentication or authorisation.  
**Risk:** Network-reachable callers can alter the information sources used by agentic context, remove trusted feeds, inject hostile content or trigger repeated expensive refreshes.  
**Recommendation:** Require scoped operator identity and rate-limit all mutation and refresh operations.  
**Status:** OPEN

### KAI-NEWS-004 — HIGH — Feed state is non-durable and worker-local
**Issue:** `_feeds` and `_articles` are module-level in-memory structures. Added or removed feeds and all cached articles disappear on restart; multiple workers hold independent states.  
**Risk:** Behaviour changes across instances and restarts, administrative changes are lost, and downstream consumers receive inconsistent article sets.  
**Recommendation:** Use a shared durable feed registry and article store with explicit versioning.  
**Status:** OPEN

### KAI-NEWS-005 — MEDIUM — Refresh is fully sequential
**Issue:** `_refresh_all` awaits each feed one at a time. Each request can consume up to the configured HTTP timeout.  
**Risk:** A slow or adversarial feed delays every subsequent feed and the force-refresh request, increasing denial-of-service impact.  
**Recommendation:** Fetch with bounded concurrency, per-host limits and a global refresh deadline.  
**Status:** OPEN

### KAI-NEWS-006 — MEDIUM — Per-feed background tasks are not tracked
**Issue:** `add_feed` starts `_fetch_feed` with `asyncio.create_task` but does not retain, await, supervise or inspect the task. Its returned articles are also never merged into `_articles`.  
**Risk:** Task failure is invisible, shutdown can abandon work, and the apparent immediate fetch does not update the article cache.  
**Recommendation:** Route refresh through a supervised queue and publish results transactionally.  
**Status:** OPEN

### KAI-NEWS-007 — MEDIUM — Internal errors are exposed
**Issue:** `_fetch_feed` stores `str(exc)` in each feed dictionary, and `GET /feeds` returns the complete dictionary to any caller.  
**Risk:** Internal network, TLS, parser and upstream details are disclosed.  
**Recommendation:** Return stable public error states and keep detailed diagnostics in protected telemetry.  
**Status:** OPEN

### KAI-NEWS-008 — MEDIUM — Timestamp conversion depends on server local timezone
**Issue:** Feedparser returns a UTC-style `struct_time`, but the service converts it with `time.mktime`, which interprets the structure as local time.  
**Risk:** Published timestamps can be shifted by the host timezone or daylight-saving state, affecting ordering and `since_minutes` filtering.  
**Recommendation:** Convert with UTC semantics and preserve source timezone information where available.  
**Status:** OPEN

### KAI-NEWS-009 — MEDIUM — Feed input is weakly bounded
**Issue:** `url`, `name` and tags have no maximum lengths, tag counts or allowed schemes. Metadata from feeds is only partially truncated.  
**Risk:** Oversized requests and hostile metadata can consume memory, pollute logs and expand downstream context.  
**Recommendation:** Add strict schema limits and normalisation for all input and parsed fields.  
**Status:** OPEN

### KAI-NEWS-010 — MEDIUM — Health is not readiness-aware
**Issue:** `/health` always reports `status: ok`, including when `feedparser` is unavailable, every feed has failed or no successful refresh has occurred.  
**Risk:** Watchdogs treat a stub or non-functional service as healthy.  
**Recommendation:** Expose separate liveness, parser readiness and feed freshness status.  
**Status:** OPEN

### KAI-NEWS-011 — MEDIUM — Error-budget telemetry is inert
**Issue:** `ErrorBudget` is instantiated and exposed, but feed fetch and endpoint outcomes never call `budget.record`.  
**Risk:** Metrics provide no reliable signal of refresh health or failure rate.  
**Recommendation:** Record classified refresh and endpoint outcomes.  
**Status:** OPEN

---

## Batch totals

- Findings: **11**
- Critical: **1**
- High: **3**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **245**
- Critical: **29**
- High: **107**
- Medium: **107**
- Low: **2**

## Files materially reviewed in this batch

`news-feed/app.py`.
