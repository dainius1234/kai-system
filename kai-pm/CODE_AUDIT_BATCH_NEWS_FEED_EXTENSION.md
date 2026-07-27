# Kai Code Audit — News Feed Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_NEWS_FEED.md`. The existing 20 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NEWSX-001 | HIGH | Feed media type is not validated before XML/RSS parsing |
| KAI-NEWSX-002 | HIGH | Feed-controlled HTML in titles and summaries is retained and returned without sanitisation |
| KAI-NEWSX-003 | HIGH | Feed-controlled article URLs are returned without scheme, host or destination validation |
| KAI-NEWSX-004 | HIGH | Feed content is promoted to Dashboard and Agentic consumers without source-integrity provenance |
| KAI-NEWSX-005 | HIGH | One malformed entry can abort the complete source refresh |
| KAI-NEWSX-006 | HIGH | Feedparser error/bozo state is ignored and malformed feeds can be published as normal data |
| KAI-NEWSX-007 | HIGH | Article and search endpoints expose content from private or internal registered feeds without read authorisation |
| KAI-NEWSX-008 | MEDIUM | Missing publication times become Unix epoch zero and are silently deprioritised or filtered out |
| KAI-NEWSX-009 | MEDIUM | Article IDs built from empty link/title fields collapse unrelated entries to one deterministic ID |
| KAI-NEWSX-010 | MEDIUM | Cross-feed deduplication discards alternate source names, tags and corroborating provenance |
| KAI-NEWSX-011 | MEDIUM | Equivalent feed URLs receive different IDs because URLs are not canonicalised |
| KAI-NEWSX-012 | MEDIUM | The documented default-feed override actually appends environment feeds to the defaults |
| KAI-NEWSX-013 | MEDIUM | Search operates on raw HTML summaries rather than normalised visible text |
| KAI-NEWSX-014 | MEDIUM | Refresh interval and article-limit configuration values are not range validated at startup |
| KAI-NEWSX-015 | MEDIUM | Zero or negative refresh intervals can create tight-loop or invalid scheduling behaviour |
| KAI-NEWSX-016 | MEDIUM | Negative article/cache limits invoke surprising Python slicing semantics |
| KAI-NEWSX-017 | MEDIUM | No conditional GET, ETag or Last-Modified support preserves coherent source revisions |
| KAI-NEWSX-018 | MEDIUM | Logs retain full feed names and URLs that may contain credentials or sensitive paths |
| KAI-NEWSX-019 | MEDIUM | Feed publication dates are accepted without a plausibility or future-time check |
| KAI-NEWSX-020 | MEDIUM | No durable audit links feed configuration, resolved destination, response digest and published article IDs |

---

### KAI-NEWSX-001 — HIGH — Media type not enforced
**Issue:** Every successful HTTP response is passed to `feedparser.parse()` regardless of `Content-Type`.  
**Risk:** Binary payloads, HTML login/error pages or unrelated attacker-controlled data enter the parser and may consume resources or produce misleading articles.  
**Recommendation:** require approved RSS/Atom/XML media types and validate a bounded feed structure before publication.  
**Status:** OPEN

### KAI-NEWSX-002 — HIGH — Stored feed HTML
**Issue:** RSS title and summary fields are retained as feed-controlled strings. Only summary length is truncated; markup is not normalised or sanitised.  
**Risk:** Downstream HTML renderers receive stored hostile markup from persistent feeds. The Dashboard audit separately confirms unsafe rendering paths.  
**Recommendation:** store canonical plain text separately from the original signed/raw source bytes.  
**Status:** OPEN

### KAI-NEWSX-003 — HIGH — Unsafe article destinations
**Issue:** `entry.link` is returned directly as `url`. No approved scheme, public-address or redirect policy is enforced for links shown to users or passed to other agents.  
**Risk:** Articles can direct browsers/agents to `javascript:`-like, internal, malformed or attacker-controlled destinations.  
**Recommendation:** canonicalise and allowlist public HTTPS links, preserving untrusted originals only in protected evidence.  
**Status:** OPEN

### KAI-NEWSX-004 — HIGH — Missing source-integrity provenance
**Issue:** Published articles carry a friendly feed name/tags but no response digest, resolved host/IP, redirect chain, parser state, TLS/source identity or feed revision.  
**Risk:** Dashboard and Agentic consumers cannot distinguish authentic current reporting from redirected, poisoned or stale content.  
**Recommendation:** attach immutable retrieval and source provenance and require downstream trust-aware use.  
**Status:** OPEN

### KAI-NEWSX-005 — HIGH — One bad entry drops the feed
**Issue:** Timestamp conversion, string slicing and field assumptions for every entry run inside one feed-level `try`. One malformed entry raises and causes `_fetch_feed()` to return `[]`.  
**Risk:** A single crafted entry suppresses every valid article in that feed and triggers the cache-erasure behaviour already logged.  
**Recommendation:** validate/quarantine entries independently and publish a partial/degraded generation with explicit counts.  
**Status:** OPEN

### KAI-NEWSX-006 — HIGH — Parser integrity state ignored
**Issue:** The service does not inspect feedparser’s `bozo`/exception indicators.  
**Risk:** Malformed or partially parsed XML is represented as a healthy feed and may inject incomplete or misinterpreted fields.  
**Recommendation:** reject or quarantine malformed feeds under an explicit parser-integrity policy.  
**Status:** OPEN

### KAI-NEWSX-007 — HIGH — Private-feed content disclosure
**Issue:** `/articles` and `/search` require no read authorisation. If an administrator, environment value or SSRF caller registers an internal/private RSS source, its parsed content becomes publicly enumerable.  
**Risk:** Internal notices, repository feeds, monitoring alerts or credential-bearing article text can be exfiltrated.  
**Recommendation:** bind every feed and article to an authenticated visibility scope and enforce it on reads.  
**Status:** OPEN

### KAI-NEWSX-008 — MEDIUM — Epoch fallback
Entries without parsed publication data receive `published_ts=0.0`, making them appear decades old and excluding them from ordinary recent windows.

### KAI-NEWSX-009 — MEDIUM — Empty-field ID collapse
Article identity is UUID5 of `link + title`; entries with both fields absent all receive the same ID.

### KAI-NEWSX-010 — MEDIUM — Provenance-losing deduplication
The first matching article ID wins. Alternate feed names, tags and independent corroborating sources are discarded.

### KAI-NEWSX-011 — MEDIUM — Non-canonical feed identity
Case, default ports, trailing slash, fragments and equivalent URL spellings create distinct feed IDs and duplicate recurring fetches.

### KAI-NEWSX-012 — MEDIUM — Override/documentation mismatch
The source comment says `SEED_FEEDS` overrides defaults, but startup always seeds the three defaults and then appends environment URLs.

### KAI-NEWSX-013 — MEDIUM — Search over markup
Substring search operates on raw summary markup/entities rather than canonical visible text, producing misleading matches and snippets.

### KAI-NEWSX-014 — MEDIUM — Unsafe environment ranges
Refresh interval and article/cache limits are direct integer environment parses with no positive or safe upper bounds.

### KAI-NEWSX-015 — MEDIUM — Tight-loop refresh
A zero/negative refresh interval removes effective delay and can continuously fetch/parsing sources.

### KAI-NEWSX-016 — MEDIUM — Negative slicing semantics
Negative limits drop trailing entries rather than enforcing a safe maximum, and can make cache behaviour difficult to reason about.

### KAI-NEWSX-017 — MEDIUM — No conditional retrieval
Every cycle redownloads complete feeds. There is no ETag/Last-Modified revision, 304 handling or stable source-generation identity.

### KAI-NEWSX-018 — MEDIUM — Sensitive URL logging
Feed names and exceptions are logged directly; URLs may embed basic-auth credentials, tokens, internal hostnames or private paths.

### KAI-NEWSX-019 — MEDIUM — Implausible date acceptance
Successfully parsed dates are accepted even when far in the future or otherwise implausible, allowing sorting/recent-filter manipulation.

### KAI-NEWSX-020 — MEDIUM — Missing end-to-end audit lineage
No tamper-evident record links actor/feed revision, canonical URL, resolved destination, redirects, response digest, parser result and article IDs.

---

## Batch totals

- Findings: **20**
- Critical: **0**
- High: **7**
- Medium: **13**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,977**
- Critical: **179**
- High: **961**
- Medium: **834**
- Low: **3**

## Files materially reviewed

`news-feed/app.py`, `news-feed/Dockerfile`, existing News Feed audit findings, deployment in `docker-compose.minimal.yml`, and Dashboard/Agentic consumption paths.
