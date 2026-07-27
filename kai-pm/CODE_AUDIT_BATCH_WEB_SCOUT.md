# Kai Code Audit — Web Scout Egress Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

The unauthenticated Agentic routes exposing Web Scout are recorded in `CODE_AUDIT_BATCH_AGENTIC_API.md`. This batch records implementation defects inside `agentic/web_scout.py` only.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WEBSCOUT-001 | CRITICAL | URL validation permits direct requests to loopback, private, link-local and metadata networks |
| KAI-WEBSCOUT-002 | CRITICAL | Redirect destinations are followed without host/IP revalidation |
| KAI-WEBSCOUT-003 | CRITICAL | DNS resolution is neither pinned nor checked against rebinding/private-address changes |
| KAI-WEBSCOUT-004 | HIGH | Complete response bodies are materialised before output truncation |
| KAI-WEBSCOUT-005 | HIGH | HTTP status and content type do not gate body return |
| KAI-WEBSCOUT-006 | HIGH | Documented ASSISTANT/PARTNER trust capabilities are absent from the Trust Core capability map |
| KAI-WEBSCOUT-007 | HIGH | Trust requests use a fixed self-asserted conviction value |
| KAI-WEBSCOUT-008 | HIGH | The module does not enforce its own feature flag |
| KAI-WEBSCOUT-009 | HIGH | Untrusted search abstracts and instant answers are returned without provenance or freshness controls |
| KAI-WEBSCOUT-010 | HIGH | Error and authentication pages are extracted and returned as normal content |
| KAI-WEBSCOUT-011 | HIGH | Search query and result-count inputs are unbounded |
| KAI-WEBSCOUT-012 | HIGH | URL, timeout and output-limit parameters are weakly validated |
| KAI-WEBSCOUT-013 | HIGH | Full URLs and queries are logged and may contain secrets or personal data |
| KAI-WEBSCOUT-014 | HIGH | Raw exception strings are returned to callers |
| KAI-WEBSCOUT-015 | MEDIUM | Each request creates a synchronous HTTP client and connection pool |
| KAI-WEBSCOUT-016 | MEDIUM | No egress-specific rate limit, concurrency limit or circuit breaker exists |
| KAI-WEBSCOUT-017 | MEDIUM | Search attempts to parse every HTTP response as JSON without checking status |
| KAI-WEBSCOUT-018 | MEDIUM | Search response size, JSON depth and field types are not constrained |
| KAI-WEBSCOUT-019 | MEDIUM | Related-topic and abstract URLs are returned without safety validation |
| KAI-WEBSCOUT-020 | MEDIUM | `summarize()` performs truncation, not summarisation |
| KAI-WEBSCOUT-021 | MEDIUM | Arbitrary non-HTML content is decoded and returned as text |
| KAI-WEBSCOUT-022 | MEDIUM | Visible-text extraction does not account for many hidden/template/accessibility states |
| KAI-WEBSCOUT-023 | MEDIUM | Results have no cache, source timestamp or staleness contract |
| KAI-WEBSCOUT-024 | MEDIUM | Failures are represented inside success-shaped result objects |

---

### KAI-WEBSCOUT-001 — CRITICAL — Direct internal-network SSRF
**Issue:** `_safe_url()` checks only whether the scheme is `http` or `https`. Hostnames and resolved addresses are not checked.  
**Risk:** Callers can fetch `localhost`, Docker service names, RFC1918 networks, link-local addresses and cloud/container metadata endpoints.  
**Recommendation:** route all retrieval through a hardened egress proxy that rejects non-public destinations after canonical DNS resolution.  
**Status:** OPEN — immediate remediation required

### KAI-WEBSCOUT-002 — CRITICAL — Redirect target bypass
**Issue:** `httpx.Client(follow_redirects=True)` follows redirects, but only the original URL is passed to `_safe_url()`. Each redirect target is not independently approved.  
**Risk:** A public attacker-controlled URL can redirect to an internal service or metadata address.  
**Recommendation:** disable automatic redirects or revalidate every hop’s scheme, hostname and resolved IP with a strict hop/host policy.  
**Status:** OPEN — immediate remediation required

### KAI-WEBSCOUT-003 — CRITICAL — DNS rebinding exposure
**Issue:** The application performs no trusted DNS resolution, IP classification or address pinning before connection.  
**Risk:** A hostname can resolve publicly during validation and privately during connection or later redirect/retry, bypassing hostname-only controls added elsewhere.  
**Recommendation:** resolve through a controlled resolver, reject every non-public address and pin the approved address for the connection.  
**Status:** OPEN — immediate remediation required

### KAI-WEBSCOUT-004 — HIGH — Response limits apply after full download
**Issue:** `resp.content` and `resp.text` materialise the complete response before visible text is truncated to `max_chars`. Automatic decompression may also occur first.  
**Risk:** Large files or compression bombs consume memory, bandwidth and CPU despite a small requested output.  
**Recommendation:** stream with strict compressed/decompressed byte limits and abort before full materialisation.  
**Status:** OPEN

### KAI-WEBSCOUT-005 — HIGH — Unsafe body acceptance
**Issue:** Fetch returns extracted body content for any HTTP status. Content type determines only HTML stripping versus direct text; it is not an allowlist.  
**Risk:** Internal error pages, login pages, binary formats and unexpected content are exposed and may be treated as valid evidence.  
**Recommendation:** require approved success statuses, media types, encoding and content-disposition policy.  
**Status:** OPEN

### KAI-WEBSCOUT-006 — HIGH — Trust capability mismatch
**Issue:** Web Scout requests `web_scout_fetch`, `web_scout_search` and autonomous variants, while `TrustCore.CAPABILITY_GATES` contains none of those names. Unknown capabilities default to GUARDIAN, contradicting the documented ASSISTANT/PARTNER levels.  
**Risk:** Healthy governance blocks intended use until maximum trust, while governance outages fail open through the already-logged shared gate defect.  
**Recommendation:** define exact canonical capabilities in one policy authority and test every call site against it.  
**Status:** OPEN

### KAI-WEBSCOUT-007 — HIGH — Fixed conviction at the trust boundary
**Issue:** Every fetch/search trust check submits `conviction=6.0`, independent of request source, URL risk, evidence quality or autonomous status.  
**Risk:** A caller-selected operation receives a fabricated confidence value at governance evaluation.  
**Recommendation:** use an independently produced signed decision record and URL-risk classification.  
**Status:** OPEN

### KAI-WEBSCOUT-008 — HIGH — Feature flag is route-only
**Issue:** The module advertises `FF_WEB_SCOUT`, but `fetch`, `search` and `summarize` never check it.  
**Risk:** Direct internal callers bypass operational disablement.  
**Recommendation:** enforce the flag/policy at the egress boundary, not only in one HTTP router.  
**Status:** OPEN

### KAI-WEBSCOUT-009 — HIGH — Search text lacks evidence provenance
**Issue:** DuckDuckGo abstract, answer and related-topic text are returned without publication/event dates, source identity validation, corroboration or confidence.  
**Risk:** Stale, manipulated or irrelevant text is readily promoted into financial and agentic reasoning.  
**Recommendation:** preserve source URLs/dates and require downstream verification before using claims.  
**Status:** OPEN

### KAI-WEBSCOUT-010 — HIGH — Non-content pages become evidence
**Issue:** Fetch does not detect authentication challenges, bot blocks, consent pages, error templates or anti-automation responses. Their visible text is returned as the requested page content.  
**Risk:** The system may summarise and reason from a denial/error page as though it were the source.  
**Recommendation:** validate status, final URL, page markers and extraction quality.  
**Status:** OPEN

### KAI-WEBSCOUT-011 — HIGH — Unbounded search inputs
**Issue:** Query length and `max_results` are not bounded or validated. Negative values alter slicing; very large values permit large loops/results if upstream supplies them.  
**Risk:** Callers can amplify URL size, logs, parsing and response payloads.  
**Recommendation:** enforce strict query bytes and a small positive result cap.  
**Status:** OPEN

### KAI-WEBSCOUT-012 — HIGH — Weak fetch parameter validation
**Issue:** URL length, `timeout_s` and `max_chars` accept arbitrary caller values. Negative/huge/non-finite values can cause exceptions, long stalls or excessive output.  
**Risk:** Internal callers can defeat expected execution and memory limits.  
**Recommendation:** validate canonical URL length and finite bounded timeout/output limits.  
**Status:** OPEN

### KAI-WEBSCOUT-013 — HIGH — Sensitive request logging
**Issue:** Complete URLs and search queries are logged at info/debug level. URLs commonly carry tokens, signed parameters, document IDs and personal search terms.  
**Risk:** Egress secrets and private research enter application logs and the unauthenticated Agentic `/logs` buffer.  
**Recommendation:** log only redacted host/category, request ID and protected diagnostics.  
**Status:** OPEN

### KAI-WEBSCOUT-014 — HIGH — Exception disclosure
**Issue:** Network, TLS, DNS and parsing exception strings are copied into `FetchResult.error` or `SearchResult.error`.  
**Risk:** Callers learn internal hostnames, connection state, proxy/TLS details and library diagnostics.  
**Recommendation:** return stable public error codes and store protected traces separately.  
**Status:** OPEN

### KAI-WEBSCOUT-015 — MEDIUM — Blocking client churn
Every operation constructs a synchronous `httpx.Client`, performs blocking I/O and destroys its connection pool. Direct async callers must offload correctly or block their worker.

### KAI-WEBSCOUT-016 — MEDIUM — No egress capacity control
The module has no per-host/global rate limit, bounded concurrency, upstream quota, retry budget or circuit breaker.

### KAI-WEBSCOUT-017 — MEDIUM — Search ignores HTTP status
The DuckDuckGo path calls `resp.json()` for every response without first requiring a successful status or expected media type.

### KAI-WEBSCOUT-018 — MEDIUM — Unbounded search response parsing
The complete body is decoded as arbitrary JSON; nesting, bytes, list length and field types are not schema-limited before iteration.

### KAI-WEBSCOUT-019 — MEDIUM — Returned links are unvalidated
`AbstractURL` and `FirstURL` values are returned directly. They may contain unsafe, malformed or internal destinations later consumed by other modules/users.

### KAI-WEBSCOUT-020 — MEDIUM — Misleading summarisation contract
`summarize()` simply fetches and truncates extracted text. It performs no semantic summary, salience selection or indication that the source may be cut mid-sentence.

### KAI-WEBSCOUT-021 — MEDIUM — Arbitrary content decoding
Every non-HTML body is decoded through `resp.text` and returned, including binary or structured content that may be mis-decoded or contain control sequences.

### KAI-WEBSCOUT-022 — MEDIUM — Incomplete visibility model
The parser skips a short tag set but ignores CSS visibility, `<template>`, ARIA-hidden content, comments rendered through malformed markup and page structure. “Visible text” is therefore not a reliable contract.

### KAI-WEBSCOUT-023 — MEDIUM — No freshness state
Results contain local elapsed time but no source timestamp, cache status, retrieval revision or staleness policy. Repeated calls also generate avoidable upstream load.

### KAI-WEBSCOUT-024 — MEDIUM — Failure inside success-shaped values
Network and trust failures return ordinary dataclasses with status 0 and an error string. Callers must remember to inspect `error`; Agentic routes can still return HTTP 200 containing the failure object.

---

## Batch totals

- Findings: **24**
- Critical: **3**
- High: **11**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,138**
- Critical: **107**
- High: **468**
- Medium: **560**
- Low: **3**

## Files materially reviewed

`agentic/web_scout.py`, with policy integration confirmation against `agentic/trust_integration.py`, `agentic/trust_core.py` and route exposure confirmation against `agentic/app.py`.
