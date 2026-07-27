# Kai Code Audit — News Feed Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NEWS-001 | CRITICAL | Unauthenticated feed registration creates an SSRF primitive |
| KAI-NEWS-002 | HIGH | Failed or partial refresh can erase valid cached article state |
| KAI-NEWS-003 | HIGH | Feed mutation and forced refresh endpoints are unauthenticated |
| KAI-NEWS-004 | HIGH | Feed and article state is process-local and lost on restart |
| KAI-NEWS-005 | HIGH | Sequential refresh duration scales with every registered feed |
| KAI-NEWS-006 | MEDIUM | Fire-and-forget feed fetch tasks are untracked and their results are discarded |
| KAI-NEWS-007 | MEDIUM | Raw fetch errors and internal feed state are exposed through `/feeds` |
| KAI-NEWS-008 | MEDIUM | Published timestamps are converted using local process timezone semantics |
| KAI-NEWS-009 | MEDIUM | Feed input fields lack scheme, host, length and tag bounds |
| KAI-NEWS-010 | MEDIUM | Health reports `ok` in stub mode and during total feed failure |
| KAI-NEWS-011 | MEDIUM | Error-budget metrics are exposed but refresh outcomes are never recorded |
| KAI-NEWS-012 | HIGH | Unbounded public feed registration creates persistent request and memory amplification |
| KAI-NEWS-013 | HIGH | Complete HTTP responses are buffered before parsing with no byte limit |
| KAI-NEWS-014 | HIGH | Background and manual refreshes can overlap and publish mixed generations |
| KAI-NEWS-015 | HIGH | Feed parsing executes synchronously on the event loop with unbounded input complexity |
| KAI-NEWS-016 | MEDIUM | Request model uses a mutable list default for tags |
| KAI-NEWS-017 | MEDIUM | Removing a feed does not remove its already cached articles |
| KAI-NEWS-018 | MEDIUM | A new HTTP client and connection pool are created for every feed request |
| KAI-NEWS-019 | MEDIUM | Refresh-task cancellation is not awaited during shutdown |
| KAI-NEWS-020 | MEDIUM | Search/filter inputs and parsed article metadata are incompletely bounded |

---

## News feed: `news-feed/app.py`

### KAI-NEWS-001 — CRITICAL — Feed registration enables SSRF
**Issue:** `POST /feeds` accepts any caller-supplied URL. `_fetch_feed` sends a server-side HTTP request to that URL with `follow_redirects=True`. No scheme, hostname, resolved IP range, port, embedded credentials or redirect destination validation is applied.  
**Risk:** A reachable caller can make the service probe loopback, Docker-internal services, private networks, link-local/cloud metadata endpoints or other privileged targets, including through redirect chains.  
**Recommendation:** Restrict feeds to an approved registry or validate canonical HTTPS destinations, DNS/IP results and every redirect hop. Require authenticated administrative access.  
**Status:** OPEN — immediate remediation required

### KAI-NEWS-002 — HIGH — Refresh can erase valid cached articles
**Issue:** `_refresh_all` builds a new list only from articles returned during the current refresh and replaces `_articles`. `_fetch_feed` returns `[]` on any error. Total failure clears the complete prior cache; partial failure silently removes all prior articles from failed feeds.  
**Risk:** A temporary outage or attacker-controlled failing feed destroys the working dataset and can make downstream consumers believe that no news or no matching stories exist.  
**Recommendation:** Maintain last-known-good state per feed with explicit freshness/error metadata and atomically publish only validated generations.  
**Status:** OPEN

### KAI-NEWS-003 — HIGH — Administrative endpoints are unauthenticated
**Issue:** Adding feeds, deleting feeds and forcing a full refresh require no authentication or authorisation.  
**Risk:** Network-reachable callers can alter information sources, remove trusted feeds, inject hostile content and repeatedly trigger expensive refresh work.  
**Recommendation:** Require scoped operator identity, immutable provenance and rate limits for every mutation/control endpoint.  
**Status:** OPEN

### KAI-NEWS-004 — HIGH — Feed state is non-durable and worker-local
**Issue:** `_feeds` and `_articles` are module-level in-memory structures. Added/removed feeds and all cached articles disappear on restart; multiple workers hold independent registries and caches.  
**Risk:** Behaviour changes across workers/restarts and downstream consumers receive inconsistent source sets and articles.  
**Recommendation:** Use a shared durable feed registry and versioned article store.  
**Status:** OPEN

### KAI-NEWS-005 — HIGH — Refresh time grows linearly without a global deadline
**Issue:** `_refresh_all` awaits every feed sequentially. Each feed request may consume the 15-second timeout, and the public registry has no feed-count limit.  
**Risk:** An attacker can register many slow endpoints so scheduled/manual refreshes occupy the event loop for minutes or hours and delay all legitimate feeds.  
**Recommendation:** Enforce a small registry limit, bounded concurrency, per-host limits and one global refresh deadline.  
**Status:** OPEN

### KAI-NEWS-006 — MEDIUM — Immediate fetch tasks are untracked and ineffective
**Issue:** `add_feed` launches `_fetch_feed` with `asyncio.create_task` but does not retain, await or supervise the task. Its returned article list is never merged into `_articles`.  
**Risk:** Failures are invisible, shutdown abandons work and the apparent immediate fetch does not make the new feed’s articles available.  
**Recommendation:** Route all fetches through a supervised refresh queue with transactional publication.  
**Status:** OPEN

### KAI-NEWS-007 — MEDIUM — Internal errors are exposed
**Issue:** `_fetch_feed` stores `str(exc)` in each feed dictionary, and `GET /feeds` returns the complete dictionary without authentication.  
**Risk:** Internal DNS, routing, TLS, parser and target-service details are disclosed.  
**Recommendation:** Return stable public error states and keep detailed diagnostics in protected telemetry.  
**Status:** OPEN

### KAI-NEWS-008 — MEDIUM — Timestamp conversion uses local timezone semantics
**Issue:** Feedparser provides a UTC-style `struct_time`, but the service converts it with `time.mktime`, which interprets the structure in the process local timezone.  
**Risk:** Published timestamps can be shifted by timezone/DST state, affecting ordering and `since_minutes` filtering.  
**Recommendation:** Convert using UTC semantics and preserve original timezone/date evidence.  
**Status:** OPEN

### KAI-NEWS-009 — MEDIUM — Feed input is weakly bounded
**Issue:** URL, name and tags have no maximum lengths, tag count, aggregate body bound or approved scheme/host policy.  
**Risk:** Oversized requests and hostile metadata consume memory, pollute logs and expand responses/downstream context.  
**Recommendation:** Add strict typed limits and normalisation.  
**Status:** OPEN

### KAI-NEWS-010 — MEDIUM — Health is not readiness-aware
**Issue:** `/health` always reports `status: ok`, including when feedparser is unavailable, every feed has failed or no successful refresh has occurred.  
**Risk:** Watchdogs treat a stub or non-functional source as healthy.  
**Recommendation:** Expose separate liveness, parser readiness, task state and per-feed aggregate freshness.  
**Status:** OPEN

### KAI-NEWS-011 — MEDIUM — Error-budget telemetry is inert
**Issue:** `ErrorBudget` is instantiated and exposed, but feed fetch and endpoint outcomes never call `budget.record`.  
**Risk:** Metrics provide no reliable refresh-health or failure-rate signal.  
**Recommendation:** Record classified refresh/request outcomes and latency.  
**Status:** OPEN

### KAI-NEWS-012 — HIGH — Public registry creates persistent amplification
**Issue:** There is no maximum number of feeds. Every unauthenticated `POST /feeds` adds/overwrites a process-global entry which is then contacted on every scheduled or forced refresh and returned by `/feeds`.  
**Risk:** Callers can create unbounded persistent memory growth and recurring outbound request amplification without sending further traffic.  
**Recommendation:** Enforce authenticated quotas, a small global registry limit and approved-source governance.  
**Status:** OPEN

### KAI-NEWS-013 — HIGH — HTTP response allocation is unbounded
**Issue:** `client.get` buffers the complete response and `resp.content` materialises it before feed parsing. No Content-Length policy, streamed byte cap or decompressed-size limit exists.  
**Risk:** A large, compressed or endless feed can consume excessive bandwidth and memory before article limits apply.  
**Recommendation:** Stream with strict wire/decompressed byte limits and reject oversized content before parsing.  
**Status:** OPEN

### KAI-NEWS-014 — HIGH — Refresh operations race
**Issue:** The background loop, `POST /refresh` and concurrent callers all invoke `_refresh_all` with no lock or generation ID. Feed additions/deletions can also occur during a refresh snapshot.  
**Risk:** Older/slower refreshes can overwrite newer results; deleted feeds can reappear in the published article cache and feed error metadata can be mixed across generations.  
**Recommendation:** Use one serialised refresh coordinator and atomically publish a versioned snapshot.  
**Status:** OPEN

### KAI-NEWS-015 — HIGH — Parsing blocks the event loop
**Issue:** After the async download, `feedparser.parse(content)` and all entry processing execute synchronously within `_fetch_feed` on the event loop. Input size and XML/feed complexity are not bounded.  
**Risk:** A malformed or computationally expensive feed blocks every request/background operation in that worker.  
**Recommendation:** Parse bounded inputs in a dedicated resource-limited worker and apply a total parse deadline.  
**Status:** OPEN

### KAI-NEWS-016 — MEDIUM — Mutable default tags
**Issue:** `AddFeedRequest.tags` is declared as `[]` rather than a default factory.  
**Risk:** Shared mutable defaults are unsafe and can permit cross-request contamination if later code mutates the list.  
**Recommendation:** Use `Field(default_factory=list)`.  
**Status:** OPEN

### KAI-NEWS-017 — MEDIUM — Feed deletion leaves stale articles
**Issue:** `DELETE /feeds/{feed_id}` removes only the registry entry. Existing `_articles` from that feed remain available until a later refresh replaces the cache.  
**Risk:** Content from a removed/untrusted source continues to be served after the operator believes it has been revoked.  
**Recommendation:** Atomically remove or quarantine all associated cached articles when deleting a feed.  
**Status:** OPEN

### KAI-NEWS-018 — MEDIUM — Connection pools are recreated per feed
**Issue:** Every `_fetch_feed` call creates a new `httpx.AsyncClient`, including each feed in every refresh cycle.  
**Risk:** Repeated DNS/TCP/TLS and connection-pool creation increases latency and socket pressure.  
**Recommendation:** Reuse a lifecycle-managed egress client with bounded pools and per-host controls.  
**Status:** OPEN

### KAI-NEWS-019 — MEDIUM — Shutdown does not await refresh termination
**Issue:** Lifespan shutdown cancels `_refresh_task` but does not await it or track per-feed tasks created by `add_feed`.  
**Risk:** Downloads/parsing can continue or be abandoned during shutdown and failures/resources are not observed.  
**Recommendation:** Retain all tasks and await bounded cancellation/cleanup.  
**Status:** OPEN

### KAI-NEWS-020 — MEDIUM — Query and parsed metadata limits are incomplete
**Issue:** Search query has a minimum but no maximum length; `since_minutes` accepts negative/extreme values. Parsed titles, links and feed names are not truncated, while summary truncation occurs only after parsing.  
**Risk:** Crafted queries/metadata consume CPU, memory and response capacity and produce misleading filtering semantics.  
**Recommendation:** Enforce strict query, time-window, field and aggregate response limits.  
**Status:** OPEN

---

## Batch totals

- Findings: **20**
- Critical: **1**
- High: **8**
- Medium: **11**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **822**
- Critical: **87**
- High: **293**
- Medium: **439**
- Low: **3**

## Files materially reviewed in this batch

`news-feed/app.py` and the relevant `news-feed` deployment definition in `docker-compose.minimal.yml`.
