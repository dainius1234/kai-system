# Kai Code Audit — Browser Agent Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_BROWSER_AGENT.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BROWSERX-001 | HIGH | `FF_BROWSER_AGENT=false` is configured but never read or enforced |
| KAI-BROWSERX-002 | HIGH | Browser navigation, click, type and run actions bypass Tool Gate entirely |
| KAI-BROWSERX-003 | HIGH | `/run` ignores the caller-supplied task and performs only navigation plus scraping |
| KAI-BROWSERX-004 | HIGH | `/run` presents scrape steps/results as though the requested browser task was executed |
| KAI-BROWSERX-005 | HIGH | Cookies, localStorage, sessionStorage and authenticated web state persist across callers |
| KAI-BROWSERX-006 | HIGH | No endpoint clears browser context, cookies, permissions, cache or authenticated sessions |
| KAI-BROWSERX-007 | HIGH | Cross-caller actions inherit the previous caller’s authenticated page and origin state |
| KAI-BROWSERX-008 | HIGH | Popups, new tabs and auxiliary pages are not tracked, bounded or closed |
| KAI-BROWSERX-009 | HIGH | Service workers, WebSockets and background page activity persist across requests and callers |
| KAI-BROWSERX-010 | HIGH | Failed navigation can leave a partially changed global page with no rollback |
| KAI-BROWSERX-011 | HIGH | Click/type operations can cause partial page side effects before returning an error |
| KAI-BROWSERX-012 | HIGH | Fuzzy text clicking selects the first partial match without a unique-target or consequence check |
| KAI-BROWSERX-013 | HIGH | Click and type responses do not verify the resulting DOM, navigation, submitted value or action outcome |
| KAI-BROWSERX-014 | HIGH | Browser mutations have no idempotency key and retries can duplicate consequential interactions |
| KAI-BROWSERX-015 | HIGH | `_get_page()` returns a non-closed page before checking whether its browser connection is still alive |
| KAI-BROWSERX-016 | HIGH | A crashed/disconnected page can poison the singleton and cause repeated failures without recreation |
| KAI-BROWSERX-017 | HIGH | The Playwright browser installation directory is made world-writable inside the image |
| KAI-BROWSERX-018 | HIGH | Browser binaries can be modified or replaced by the runtime service user after compromise |
| KAI-BROWSERX-019 | HIGH | Chromium artefacts are installed from broad version ranges without immutable browser-binary digests |
| KAI-BROWSERX-020 | HIGH | Chromium launch does not explicitly enable a browser sandbox or verify sandbox readiness |
| KAI-BROWSERX-021 | HIGH | Downloads have no policy, size quota, filename control, storage limit or cleanup lifecycle |
| KAI-BROWSERX-022 | HIGH | Navigation has no total request-count, bandwidth, response-body or subresource budget |
| KAI-BROWSERX-023 | HIGH | Popups, downloads and background requests can continue after the initiating endpoint returns |
| KAI-BROWSERX-024 | HIGH | Scraped link destinations are returned without scheme or destination validation |
| KAI-BROWSERX-025 | HIGH | Page titles, link text, URLs and snippets have no safe display/control-character contract |
| KAI-BROWSERX-026 | HIGH | Scraped page text is not marked as untrusted web data before Dashboard inserts it into Agentic prompts |
| KAI-BROWSERX-027 | HIGH | Navigation with no HTTP response object is returned as successful status `0` |
| KAI-BROWSERX-028 | HIGH | DuckDuckGo bot challenges, consent pages or changed markup become empty successful search results |
| KAI-BROWSERX-029 | HIGH | Search results contain no publication date, retrieval timestamp, source trust or freshness evidence |
| KAI-BROWSERX-030 | HIGH | Screenshot and scraped-content responses lack `Cache-Control: no-store` |
| KAI-BROWSERX-031 | HIGH | Browser operations have no authenticated context ID, request ID, page revision or operation ledger |
| KAI-BROWSERX-032 | HIGH | No tamper-evident audit records navigation, page identity, selectors, entered text and resulting side effects |
| KAI-BROWSERX-033 | HIGH | The full Compose topology omits Browser Agent while other full-stack modules retain browser capability references |
| KAI-BROWSERX-034 | HIGH | Browser profile/cache state has no age-based retention or privacy deletion policy |
| KAI-BROWSERX-035 | MEDIUM | Main-frame text scraping omits iframe, shadow-DOM, canvas and accessibility-only content without completeness metadata |
| KAI-BROWSERX-036 | MEDIUM | Link extraction silently truncates at 50 links without reporting total or truncation |
| KAI-BROWSERX-037 | MEDIUM | Link URLs, page titles and search result fields have no individual response-size bounds |
| KAI-BROWSERX-038 | MEDIUM | Screenshot responses contain no page URL, capture timestamp, content digest or browser-context identity |
| KAI-BROWSERX-039 | MEDIUM | Navigate, scrape, run and search responses define no strict response models or schema version |
| KAI-BROWSERX-040 | MEDIUM | Health exposes no Chromium/Playwright version, executable digest or browser-context count |
| KAI-BROWSERX-041 | MEDIUM | Public metrics expose request telemetry without administrative authentication |
| KAI-BROWSERX-042 | MEDIUM | Browser launch and new-page creation have no explicit operation deadline |
| KAI-BROWSERX-043 | MEDIUM | Page evaluation and screenshot operations rely on implicit Playwright defaults rather than service-owned deadlines |
| KAI-BROWSERX-044 | MEDIUM | Search `max_results` accepts Boolean values through integer coercion |
| KAI-BROWSERX-045 | MEDIUM | Search result structure is assumed from page JavaScript without runtime schema validation |
| KAI-BROWSERX-046 | MEDIUM | The service has no browser-crash, popup, download, request-count or active-page metrics |
| KAI-BROWSERX-047 | MEDIUM | `sys.path` is mutated at import using a deployment-dependent parent path |
| KAI-BROWSERX-048 | MEDIUM | Missing shared-runtime imports silently replace structured telemetry with no-op fallbacks |
| KAI-BROWSERX-049 | MEDIUM | FastAPI, Playwright and the Python base image are not reproducibly digest-pinned |
| KAI-BROWSERX-050 | MEDIUM | The test suite mocks every Playwright call and never launches a real browser |
| KAI-BROWSERX-051 | MEDIUM | Tests do not assert that `/run` uses its task, that sessions are isolated or that destination policy exists |
| KAI-BROWSERX-052 | MEDIUM | Tests do not cover browser crash recovery, downloads, popups, cookies, redirects or internal-network destinations |
| KAI-BROWSERX-053 | MEDIUM | Shutdown closes the singleton without first rejecting new work or draining the active locked operation |
| KAI-BROWSERX-054 | MEDIUM | No permission policy governs clipboard, notifications, geolocation, camera, microphone or other browser capabilities |
| KAI-BROWSERX-055 | MEDIUM | Results omit the browser user agent, locale, timezone and viewport that materially affect page content |
| KAI-BROWSERX-056 | MEDIUM | Lazy launch failure is not persisted into readiness and is rediscovered independently by later requests |

---

## High-severity findings

### KAI-BROWSERX-001 — HIGH — Feature flag is unenforced
**Issue:** Minimal Compose sets `FF_BROWSER_AGENT=false`, but `browser-agent/app.py` never reads the variable.  
**Risk:** The service remains fully operational and host-published when deployment configuration claims it is disabled.  
**Recommendation:** Enforce a server-owned capability gate at every browser operation and fail readiness when disabled.  
**Status:** OPEN

### KAI-BROWSERX-002 — HIGH — Browser actions bypass Gate policy
No navigation or interaction endpoint obtains a Tool Gate decision for the exact URL, selector, text or consequence.

### KAI-BROWSERX-003 — HIGH — Task input is dead
`RunRequest.task` is never referenced after validation.

### KAI-BROWSERX-004 — HIGH — False agent execution claim
The endpoint returns `steps` and a `result` describing navigation/scraping even though it did not interpret or perform the requested task.

### KAI-BROWSERX-005 — HIGH — Persistent cross-caller web state
One browser context/page retains cookies and origin storage across every caller.

### KAI-BROWSERX-006 — HIGH — No session erasure
There is no logout/reset/new-context operation or automatic cleanup after a workflow.

### KAI-BROWSERX-007 — HIGH — Inherited authenticated authority
A later caller can interact with sites authenticated by an earlier caller because the same page/context is reused.

### KAI-BROWSERX-008 — HIGH — Unmanaged auxiliary pages
Sites can open popups/tabs that remain in the browser and consume resources/state; the service only tracks `_page`.

### KAI-BROWSERX-009 — HIGH — Persistent background execution
No context/page policy disables or terminates service workers, WebSockets or background traffic between requests.

### KAI-BROWSERX-010 — HIGH — Navigation has no transactional state
A timeout/error may occur after the page URL/DOM/cookies changed; no previous-page restoration occurs.

### KAI-BROWSERX-011 — HIGH — Interaction error does not mean no action
DOM events/network submissions may occur before Playwright raises or times out.

### KAI-BROWSERX-012 — HIGH — Ambiguous text action
`get_by_text(..., exact=False).first` can choose the wrong partial-match element on a consequential page.

### KAI-BROWSERX-013 — HIGH — No postcondition verification
Success means Playwright returned, not that the intended outcome occurred.

### KAI-BROWSERX-014 — HIGH — Duplicate action replay
Clicks, form fills and `/run` requests lack an operation/idempotency identity.

### KAI-BROWSERX-015 — HIGH — Disconnected-browser stale page
The page-close test precedes browser-connectivity validation, so a stale Page may be returned indefinitely.

### KAI-BROWSERX-016 — HIGH — No automatic crash recovery
Operation failures do not clear/rebuild `_page`, `_browser` or `_playwright_inst`.

### KAI-BROWSERX-017 — HIGH — World-writable browser installation
The Dockerfile applies `chmod 777 /ms-playwright`.

### KAI-BROWSERX-018 — HIGH — Runtime browser persistence primitive
The non-root service user can alter files under the browser installation directory, persisting modified executable/resources after a compromise within the container lifetime/volume layer.

### KAI-BROWSERX-019 — HIGH — Mutable browser artefact
`playwright>=1.47.0` and `playwright install chromium` do not bind an immutable reviewed browser digest.

### KAI-BROWSERX-020 — HIGH — Browser sandbox not asserted
Launch specifies only `headless`; no explicit Chromium sandbox configuration or startup verification is present.

### KAI-BROWSERX-021 — HIGH — Uncontrolled downloads
No listener/policy rejects downloads or limits their files, sizes and lifecycle.

### KAI-BROWSERX-022 — HIGH — Unbounded page-resource load
Navigation can load arbitrary scripts/media/subresources with no aggregate network/byte/request policy.

### KAI-BROWSERX-023 — HIGH — Post-request browser activity
Returning an HTTP result does not terminate site-created background activity.

### KAI-BROWSERX-024 — HIGH — Unsafe returned destinations
Scraped/search URLs can use `javascript:`, `data:`, internal or otherwise unsafe schemes and are returned as ordinary links.

### KAI-BROWSERX-025 — HIGH — Unsafe web presentation data
Web-controlled strings are propagated without a plain-text/control-character contract.

### KAI-BROWSERX-026 — HIGH — Missing prompt-evidence boundary
Dashboard can place scraped text/results into Agentic input; Browser Agent attaches no untrusted-source or injection metadata.

### KAI-BROWSERX-027 — HIGH — Non-HTTP success ambiguity
`page.goto` may return `None`; the endpoint reports status zero without an explicit non-HTTP/failed state.

### KAI-BROWSERX-028 — HIGH — Search challenge looks empty
Search DOM mismatch/consent/bot challenge returns an ordinary empty results list rather than a degraded source state.

### KAI-BROWSERX-029 — HIGH — Search lacks evidence chronology
Results have no event/publication dates or retrieval/source-confidence metadata.

### KAI-BROWSERX-030 — HIGH — Cacheable private page output
Screenshots and scraped authenticated content lack no-store controls.

### KAI-BROWSERX-031 — HIGH — Missing browser operation identity
Results cannot be tied to authenticated caller, browser context, exact before/after page or operation revision.

### KAI-BROWSERX-032 — HIGH — Missing browser audit
No immutable event captures requested action and observed postcondition.

### KAI-BROWSERX-033 — HIGH — Full-stack topology drift
Full Compose omits Browser Agent while full-stack Dashboard/Agentic/monitoring code still retains browser service assumptions.

### KAI-BROWSERX-034 — HIGH — No browsing-data retention policy
Cookies, cache and other context state live until browser/service shutdown without age/purpose deletion.

---

## Medium-severity findings

### KAI-BROWSERX-035 — MEDIUM — Incomplete scrape semantics
`document.body.innerText` is only one main-frame representation and does not declare omitted sources.

### KAI-BROWSERX-036 — MEDIUM — Hidden link truncation
Only the first 50 document-order links are returned.

### KAI-BROWSERX-037 — MEDIUM — Unbounded individual web fields
The configured text cap does not limit titles, hrefs or search fields individually.

### KAI-BROWSERX-038 — MEDIUM — Missing screenshot provenance
The binary response has only a media type.

### KAI-BROWSERX-039 — MEDIUM — Unmodelled responses
Dictionary output can drift silently across versions.

### KAI-BROWSERX-040 — MEDIUM — Incomplete browser readiness identity
Health provides only two Booleans.

### KAI-BROWSERX-041 — MEDIUM — Public telemetry
Metrics requires no administrative identity.

### KAI-BROWSERX-042 — MEDIUM — Launch deadline implicit
Playwright startup/new-page calls have no service-owned timeout.

### KAI-BROWSERX-043 — MEDIUM — Implicit operation deadlines
Evaluate/screenshot use library defaults rather than the service configuration.

### KAI-BROWSERX-044 — MEDIUM — Boolean result-count coercion
JSON `true`/`false` may be accepted as integers.

### KAI-BROWSERX-045 — MEDIUM — Search output unvalidated
The result of page JavaScript is returned without checking list/item types and lengths.

### KAI-BROWSERX-046 — MEDIUM — Missing operational metrics
Only HTTP error-budget data exists.

### KAI-BROWSERX-047 — MEDIUM — Import-path mutation
Global module resolution is altered with a layout-dependent path.

### KAI-BROWSERX-048 — MEDIUM — Silent telemetry downgrade
Missing common runtime gives no-op metrics while health remains normal.

### KAI-BROWSERX-049 — MEDIUM — Non-reproducible browser service
Critical packages/base artefacts are not fully locked.

### KAI-BROWSERX-050 — MEDIUM — Mock-only tests
No actual browser binary, renderer, network or storage context is exercised.

### KAI-BROWSERX-051 — MEDIUM — Core contract gaps untested
Task semantics, feature policy and session isolation are absent from assertions.

### KAI-BROWSERX-052 — MEDIUM — Security/runtime paths untested
No browser crash, redirect, internal-network, download or popup coverage exists.

### KAI-BROWSERX-053 — MEDIUM — Shutdown race
The lifespan does not establish an admission-stop/drain phase before closing the shared browser.

### KAI-BROWSERX-054 — MEDIUM — Browser permission policy absent
Capabilities are left to browser defaults rather than an explicit deny/allow contract.

### KAI-BROWSERX-055 — MEDIUM — Missing content-environment provenance
Result interpretation can change with locale/timezone/viewport/user agent, none of which is returned.

### KAI-BROWSERX-056 — MEDIUM — Repeated lazy failure
A browser launch failure is not stored as degraded readiness/cooldown and can be retried by every request.

---

## Batch totals

- Findings: **56**
- Critical: **0**
- High: **34**
- Medium: **22**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,704**
- Critical: **192**
- High: **1,367**
- Medium: **1,142**
- Low: **3**

## Files materially reviewed

`browser-agent/app.py`, `browser-agent/Dockerfile`, `browser-agent/requirements.txt`, `scripts/test_browser_agent.py`, minimal/full deployment topology, Dashboard/Monitor integrations and the existing Browser Agent audit.
