# Kai Code Audit — Dashboard Browser Client Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers `dashboard/static/app.html`. Backend gateway/exposure findings remain in `CODE_AUDIT_BATCH_DASHBOARD_GATEWAY.md` and are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DASHUI-001 | CRITICAL | A stray `/` before `function attachFile()` makes the complete unified inline script fail JavaScript parsing |
| KAI-DASHUI-002 | CRITICAL | Page initialisation automatically attempts to place Tool Gate into WORK mode from mutable browser storage |
| KAI-DASHUI-003 | CRITICAL | Financial records create a stored same-origin XSS path through unsanitised `innerHTML` |
| KAI-DASHUI-004 | CRITICAL | Email/calendar/news fields create stored or feed-driven same-origin XSS paths |
| KAI-DASHUI-005 | CRITICAL | Docker, Git and process fields create same-origin XSS paths in the System view |
| KAI-DASHUI-006 | CRITICAL | Broker symbols, assets, orders, templates and inline handlers create same-origin XSS and script-injection paths |
| KAI-DASHUI-007 | CRITICAL | Model output rendering fails open to raw HTML when Marked loads but DOMPurify does not |
| KAI-DASHUI-008 | CRITICAL | Browser-stored assistant messages become persistent XSS when the sanitiser is unavailable or compromised |
| KAI-DASHUI-009 | CRITICAL | Operator-model, escalation, oracle and shadow data is inserted into `innerHTML` without context-safe escaping |
| KAI-DASHUI-010 | HIGH | Marked and DOMPurify are loaded from mutable CDN version ranges without Subresource Integrity |
| KAI-DASHUI-011 | HIGH | The inline-script/inline-handler design prevents a strict script Content Security Policy |
| KAI-DASHUI-012 | HIGH | Up to 100 complete private chat messages are retained indefinitely in plaintext localStorage |
| KAI-DASHUI-013 | HIGH | Local chat history has no authenticated user partition, expiry, logout purge or consent state |
| KAI-DASHUI-014 | HIGH | LocalStorage history is loaded without schema, type, item-size or aggregate-size validation |
| KAI-DASHUI-015 | HIGH | Any non-`user` local history role is rendered as privileged assistant Markdown |
| KAI-DASHUI-016 | HIGH | LocalStorage mode accepts arbitrary strings and is sent to Agentic without enum validation |
| KAI-DASHUI-017 | HIGH | Mode synchronisation is fire-and-forget and the UI treats local state as authoritative |
| KAI-DASHUI-018 | HIGH | Merely opening a new browser profile defaults the enforcement request to WORK before onboarding |
| KAI-DASHUI-019 | HIGH | Global copy-event interception silently sends selected Dashboard text to the server clipboard service |
| KAI-DASHUI-020 | HIGH | Copied logs, email, finance, memory and identity text is relayed without sensitivity filtering or user indication |
| KAI-DASHUI-021 | HIGH | Pending notifications are automatically dismissed server-side merely because a browser polled them |
| KAI-DASHUI-022 | HIGH | Multiple Dashboard clients race to consume and globally dismiss the same notifications |
| KAI-DASHUI-023 | HIGH | Browser search results and scraped page text are injected directly into the next Agentic prompt |
| KAI-DASHUI-024 | HIGH | OCR and document-parser output is inserted directly into the Agentic prompt without provenance or injection treatment |
| KAI-DASHUI-025 | HIGH | Text attachments permit sensitive extensions such as `.env`, `.conf`, scripts and source files and insert them into chat |
| KAI-DASHUI-026 | HIGH | Image/document uploads have no browser-side byte limit before complete upload |
| KAI-DASHUI-027 | HIGH | Attachment routing trusts filename extensions rather than file content |
| KAI-DASHUI-028 | HIGH | WebSpeech transcripts are sent as Agentic requests immediately without operator review |
| KAI-DASHUI-029 | HIGH | Browser speech recognition may send audio/transcripts to the browser vendor without a data-use notice |
| KAI-DASHUI-030 | HIGH | MediaRecorder has no maximum recording duration or accumulated-byte bound |
| KAI-DASHUI-031 | HIGH | Stopping MediaRecorder automatically uploads the complete recording |
| KAI-DASHUI-032 | HIGH | Camera permission starts automatic frame upload every five seconds without per-frame confirmation or retention notice |
| KAI-DASHUI-033 | HIGH | Camera analysis requests can overlap indefinitely because no in-flight/backpressure guard exists |
| KAI-DASHUI-034 | HIGH | Camera and microphone collection are not stopped on page hide, navigation or unload |
| KAI-DASHUI-035 | HIGH | Chat streaming does not verify the HTTP response status before reading the body as SSE |
| KAI-DASHUI-036 | HIGH | Non-SSE and malformed backend text is appended to assistant output as trusted model content |
| KAI-DASHUI-037 | HIGH | The complete partial assistant response is reparsed as Markdown and rewritten into `innerHTML` for every token |
| KAI-DASHUI-038 | HIGH | Assistant links are not rewritten to a safe allowlist, warning interstitial or trusted provenance view |
| KAI-DASHUI-039 | HIGH | News/search URLs are inserted directly into `href` attributes without scheme validation |
| KAI-DASHUI-040 | HIGH | HTML escaping is incorrectly reused for JavaScript-string values inside inline `onclick` handlers |
| KAI-DASHUI-041 | HIGH | Memory-category values can break or inject the inline `searchByCategory('...')` handler |
| KAI-DASHUI-042 | HIGH | Broker symbols can inject the inline `brokerQuickWatch('...')` handler |
| KAI-DASHUI-043 | HIGH | Monitor rule IDs and backend-controlled identifiers are interpolated into inline handlers |
| KAI-DASHUI-044 | HIGH | Sanitised fragments are later mixed with unsanitised attribute/style/context data |
| KAI-DASHUI-045 | HIGH | Memory-graph tooltips directly require global DOMPurify and fail when the CDN object is absent |
| KAI-DASHUI-046 | HIGH | Critical SOUL and AGENTS rewrites have no preview, diff, revision, typed confirmation or rollback UI |
| KAI-DASHUI-047 | HIGH | Many state-changing actions report success without checking `response.ok` or the returned operation result |
| KAI-DASHUI-048 | HIGH | Goal, gratitude, shadow, screen-watch, feedback and monitoring actions can show success after backend fallback/failure |
| KAI-DASHUI-049 | HIGH | Feedback buttons use one hard-coded global session ID and client-derived message indexes |
| KAI-DASHUI-050 | HIGH | Local chat-history indexes do not reliably identify the corresponding memU session message |
| KAI-DASHUI-051 | HIGH | SSE connects automatically on every page load and continuously reconnects without authentication or connection admission |
| KAI-DASHUI-052 | HIGH | The page automatically starts expensive health, Dashboard and struggle-detection polling |
| KAI-DASHUI-053 | MEDIUM | The notification poll runs every ten seconds even when the notification UI is closed |
| KAI-DASHUI-054 | MEDIUM | Polling loops have no document-visibility pause or in-flight-request suppression |
| KAI-DASHUI-055 | MEDIUM | Async interval callbacks can overlap when a refresh takes longer than its period |
| KAI-DASHUI-056 | MEDIUM | EventSource retry continues indefinitely with a fixed five-second interval and no exponential backoff |
| KAI-DASHUI-057 | MEDIUM | Event parse errors and many refresh failures are silently ignored |
| KAI-DASHUI-058 | MEDIUM | A single malformed numeric backend field can throw and blank an entire view |
| KAI-DASHUI-059 | MEDIUM | `fmt_gbp()` displays `£NaN` or non-finite values rather than rejecting invalid finance data |
| KAI-DASHUI-060 | MEDIUM | Numerous view renderers build large HTML strings and replace complete subtrees on every poll |
| KAI-DASHUI-061 | MEDIUM | Calendar refresh temporarily assigns a Promise-derived value to `innerHTML` before overwriting it |
| KAI-DASHUI-062 | MEDIUM | TTS object URLs are not revoked when playback is manually stopped or fails before `onended` |
| KAI-DASHUI-063 | MEDIUM | TTS assumes returned bytes are playable MP3 and does not validate media type or size |
| KAI-DASHUI-064 | MEDIUM | Camera blobs are generated at source resolution despite only requesting a nominal 320×240 constraint |
| KAI-DASHUI-065 | MEDIUM | Browser camera-analysis errors are silently discarded, leaving stale presence/emotion status |
| KAI-DASHUI-066 | MEDIUM | Browser search/browse failures leave inserted or stale content without a source-state marker |
| KAI-DASHUI-067 | MEDIUM | Local settings and onboarding records are accepted without schema/version migration |
| KAI-DASHUI-068 | MEDIUM | PWA installation can persist the unauthenticated gateway and its local private history as an installed app |
| KAI-DASHUI-069 | MEDIUM | The page has no client-side inactivity lock, session expiry or sensitive-view reauthentication |
| KAI-DASHUI-070 | MEDIUM | The app has no unload cleanup for EventSource, timers, force simulations, audio URLs or media streams |
| KAI-DASHUI-071 | MEDIUM | Automatic Dashboard refresh starts before the first-run wizard establishes any user intent |
| KAI-DASHUI-072 | MEDIUM | Finance-record creation accepts non-finite materials and weakly validated free-text metadata |
| KAI-DASHUI-073 | MEDIUM | Backend response status is commonly ignored before JSON parsing, conflating error and business payloads |
| KAI-DASHUI-074 | MEDIUM | Many UI catches suppress the backend error and leave stale data displayed as current |
| KAI-DASHUI-075 | MEDIUM | Full backend objects are frequently stringified into toasts/notifications without a data-minimisation policy |
| KAI-DASHUI-076 | MEDIUM | Client-side actions have no correlation or idempotency identifier for tracing duplicate mutations |

---

## Critical findings

### KAI-DASHUI-001 — CRITICAL — Entire client script fails to parse
**Issue:** The source contains `/function attachFile() { ... }` inside the single unified inline `<script>`. The leading `/` is not a valid standalone token before the function declaration.  
**Risk:** JavaScript parsing fails for the complete script block; chat, mode synchronisation, health, navigation, uploads, monitoring, broker, feeds and every other client control are unavailable. This is a deterministic production UI outage, not a dormant code path.  
**Recommendation:** Fix the syntax and make parsing/linting/browser smoke tests mandatory in CI before deployment.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-002 — CRITICAL — Page-load transition to WORK mode
**Issue:** The immediately invoked `init()` calls `loadMode()`. With no prior value, `loadMode()` chooses `WORK` and calls `setMode()`, which POSTs `/api/mode`. This happens before first-run onboarding and without awaiting/confirming the server result.  
**Risk:** Opening the Dashboard can use the server’s privileged Tool Gate token to move the system into execution-capable WORK mode. A same-origin script or localStorage modification can choose another arbitrary mode.  
**Recommendation:** Never change server enforcement state during page rendering. Require authenticated, explicit, confirmed operator action and display the authoritative server mode.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-003 — CRITICAL — Stored finance XSS
**Issue:** VAT period and invoice `date`, `contractor` and `reference` are inserted directly into table `innerHTML`. Contractor/reference values can be created through the same Dashboard.  
**Risk:** A persisted finance record can execute JavaScript in the Dashboard origin, obtaining every privileged API and local chat-history capability. No CSP limits impact.  
**Recommendation:** Build DOM nodes with `textContent`; never interpolate backend text into HTML.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-004 — CRITICAL — Feed-driven XSS
**Issue:** Calendar summary/location, email subject/from/date/snippet, news title/feed/tags/summary and link URL are inserted into `innerHTML` without escaping or URL validation.  
**Risk:** An email, calendar event or feed article becomes a stored/external same-origin script source.  
**Recommendation:** Treat all feed fields as untrusted text and permit only canonical `https` links through safe DOM APIs.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-005 — CRITICAL — System-view XSS
**Issue:** Container names/images/status/health, Git paths/branches/commits and process names/status are directly interpolated into `innerHTML`.  
**Risk:** Attacker-controlled container labels, image names, repository paths/branches or process names can execute code in the privileged Dashboard origin.  
**Recommendation:** Use `textContent` for every operational field.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-006 — CRITICAL — Broker-view XSS and handler injection
**Issue:** Broker mode/status, symbols, assets, sides/types/status, template names/descriptions and symbol-bearing inline handlers are not context-safely escaped.  
**Risk:** Exchange/API metadata or a poisoned monitor template can execute arbitrary Dashboard-origin script and create actions.  
**Recommendation:** Render with DOM APIs and attach event listeners with closed-over validated IDs/symbols.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-007 — CRITICAL — Markdown sanitizer fail-open
**Issue:** `renderMarkdown()` returns raw `marked.parse(text)` output when `marked` exists but global `DOMPurify` does not. Both libraries are loaded independently from a CDN.  
**Risk:** A transient CDN failure/block of DOMPurify, or deliberate deletion/replacement of the global, converts model output into raw executable HTML.  
**Recommendation:** Fail closed to escaped text whenever the exact pinned sanitiser is unavailable; apply a strict URL/element/attribute policy.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-008 — CRITICAL — Persistent local chat XSS
**Issue:** Assistant Markdown is saved to `localStorage` and re-rendered on startup through the same fail-open Markdown path. Local history roles/content are not schema-validated.  
**Risk:** One malicious assistant response remains an executable payload across reloads whenever sanitisation is unavailable or compromised.  
**Recommendation:** store plain text or a safe canonical representation and always render through a mandatory pinned sanitizer.  
**Status:** OPEN — immediate remediation required

### KAI-DASHUI-009 — CRITICAL — Personal-model view XSS
**Issue:** Echo messages/types, nudge targets/names, oracle actions/risk/emotional forecast, shadow decisions/alternatives and several emotional/operator-model fields are directly interpolated into `innerHTML`. These values originate in globally writable memU state.  
**Risk:** Anonymous memU/Dashboard poisoning becomes stored XSS in the operator-model pages.  
**Recommendation:** render every derived field as text and validate numeric/style values separately.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-DASHUI-010 — HIGH — Mutable CDN supply chain
`marked@12` and `dompurify@3` are mutable major-version ranges loaded from jsDelivr without SRI, exact immutable asset digests or a restrictive CSP.

### KAI-DASHUI-011 — HIGH — CSP-hostile architecture
One large inline script and extensive inline `onclick` attributes require `unsafe-inline` or no CSP, removing an important XSS containment layer.

### KAI-DASHUI-012 — HIGH — Plaintext local chat retention
Up to 100 complete user/assistant messages remain in localStorage indefinitely and are readable by every same-origin script, browser extension with access, installed PWA instance or XSS payload.

### KAI-DASHUI-013 — HIGH — No local-history principal lifecycle
History is not partitioned by authenticated user/session, expired, cleared on logout or governed by retention/consent.

### KAI-DASHUI-014 — HIGH — Unvalidated local history
`JSON.parse()` output is assumed to be an array of bounded `{role,content}` objects; crafted/large storage can break or exhaust rendering and request construction.

### KAI-DASHUI-015 — HIGH — Role elevation from local storage
`addMessage()` treats every role other than exact `user` as assistant, enabling stored records to enter the Markdown/assistant rendering path.

### KAI-DASHUI-016 — HIGH — Arbitrary local mode
Any localStorage string is sent as the chat mode and to `/api/mode`; no client enum validation exists.

### KAI-DASHUI-017 — HIGH — Split-brain mode UI
`setMode()` updates localStorage/buttons first and ignores the asynchronous server response, so the UI can display a mode that was rejected or never applied.

### KAI-DASHUI-018 — HIGH — Pre-onboarding privileged mutation
The default WORK POST occurs before the wizard is shown; a visitor does not have to select or acknowledge a mode.

### KAI-DASHUI-019 — HIGH — Silent copy surveillance
A document-wide `copy` listener sends every selected text string to `/api/clipboard/push` after the browser copy action.

### KAI-DASHUI-020 — HIGH — Sensitive copied-data capture
The listener covers private logs, email, finance, memories, SOUL/AGENTS, broker data and chat. There is no warning, opt-in or sensitivity/source exclusion.

### KAI-DASHUI-021 — HIGH — Polling changes notification state
Every ten seconds the client fetches unread notifications and immediately issues DELETE dismissals simply because they were returned.

### KAI-DASHUI-022 — HIGH — Multi-client notification race
The first open browser can globally dismiss notifications before another operator/device sees them; failed deletes are ignored and cause duplicates.

### KAI-DASHUI-023 — HIGH — Web-to-prompt injection
Browser-agent search results and scraped page content are placed into the chat input as ordinary user instructions without source quoting, untrusted-data boundaries or confirmation.

### KAI-DASHUI-024 — HIGH — Document-to-prompt injection
OCR/parser output is inserted into the same prompt field automatically, allowing malicious documents/images to issue instructions to Agentic.

### KAI-DASHUI-025 — HIGH — Sensitive local-file ingestion
The text allowlist includes `.env`, shell, config, log and source-code files. Selected content is copied directly into persistent local chat and sent to Agentic.

### KAI-DASHUI-026 — HIGH — Unbounded non-text upload
Only text attachments have a 50 KB client limit. Images, office files, ZIP, CAD and audio can be arbitrarily large.

### KAI-DASHUI-027 — HIGH — Extension-only type routing
The final filename suffix selects text/OCR/document handling; magic bytes and container structure are not validated in the client.

### KAI-DASHUI-028 — HIGH — Speech-to-action without review
Both WebSpeech result handling and successful fallback transcription call `sendMessage()` automatically.

### KAI-DASHUI-029 — HIGH — Undisclosed browser speech provider
The preferred WebSpeech API may transmit audio to the browser vendor’s service, while the UI describes it simply as voice input.

### KAI-DASHUI-030 — HIGH — Unlimited microphone capture
MediaRecorder accumulates chunks until manually stopped; there is no time, byte or silence limit.

### KAI-DASHUI-031 — HIGH — Automatic recording upload
The `onstop` callback immediately builds and uploads the entire blob; there is no playback/review/discard step.

### KAI-DASHUI-032 — HIGH — Automatic camera surveillance loop
After browser permission and metadata load, a frame is captured and sent every five seconds until manually stopped, with no per-frame indication or data-retention notice.

### KAI-DASHUI-033 — HIGH — Camera request pile-up
`setInterval()` invokes `_sendFrame()` regardless of whether the prior canvas/blob/network analysis completed.

### KAI-DASHUI-034 — HIGH — Media survives UI lifecycle
No visibility/unload handler stops active camera tracks, microphone recording or pending media requests.

### KAI-DASHUI-035 — HIGH — Chat status not validated
`sendMessage()` immediately reads `resp.body` without checking `resp.ok` or requiring `text/event-stream`.

### KAI-DASHUI-036 — HIGH — Error body promoted to assistant content
Any line not parseable as JSON is appended verbatim to `fullResponse` and rendered as assistant Markdown.

### KAI-DASHUI-037 — HIGH — Per-token HTML rewrite
Every token reparses the full accumulated Markdown and replaces `innerHTML`, magnifying CPU/DOM cost and repeatedly exercising the HTML sanitisation boundary.

### KAI-DASHUI-038 — HIGH — Unsafe model links
Markdown-generated links are not forced through a safe external-link policy, scheme allowlist or warning/provenance interstitial.

### KAI-DASHUI-039 — HIGH — Unsafe feed href
News URLs are inserted directly into quoted `href` attributes. `javascript:`, malformed and attacker-controlled schemes are not rejected.

### KAI-DASHUI-040 — HIGH — Wrong escaping context
`escapeHtml()` protects HTML text/attributes, not JavaScript source embedded inside an inline handler. HTML entities are decoded into the handler value before JavaScript execution.

### KAI-DASHUI-041 — HIGH — Category handler injection
Memory categories are placed inside `onclick="searchByCategory('...')"`; a crafted category can break the JavaScript string or alter the handler.

### KAI-DASHUI-042 — HIGH — Broker-symbol handler injection
`brokerQuickWatch('${sym}')` uses an unescaped backend symbol in executable inline JavaScript.

### KAI-DASHUI-043 — HIGH — Identifier handler injection
Monitor/rule and other backend IDs are interpolated into inline event handlers rather than passed through safe listener closures.

### KAI-DASHUI-044 — HIGH — Mixed sanitisation contexts
Several renderers sanitise visible fragments but leave adjacent trust tier, counts, style values, attributes or handler data unsanitised, so sanitising one fragment does not secure the constructed HTML.

### KAI-DASHUI-045 — HIGH — Hard DOMPurify dependency in graph UI
Graph tooltip/detail code calls `DOMPurify.sanitize()` directly rather than the fail-closed wrapper. CDN unavailability throws and breaks the feature.

### KAI-DASHUI-046 — HIGH — Unsafe identity-edit workflow
SOUL and AGENTS editors save complete replacements with one click, no diff, version check, explicit risk confirmation, approval identity or tested rollback.

### KAI-DASHUI-047 — HIGH — Mutation status often ignored
Many POST/DELETE helpers await `fetch()` and show success without checking `response.ok`, status schema or committed operation ID.

### KAI-DASHUI-048 — HIGH — False UI success
Gratitude, goals, shadow branches, screen watcher, feedback, broker watch/templates and similar operations can display success after an HTTP-200 fallback or rejected backend operation.

### KAI-DASHUI-049 — HIGH — Hard-coded feedback session
Every rating submits `session_id: default`, irrespective of the actual Agentic/memU session.

### KAI-DASHUI-050 — HIGH — Wrong feedback message identity
The button index is derived from local `chatHistory.length`, which contains both roles and may differ from server session order, reloads, truncated history and streaming finalisation.

### KAI-DASHUI-051 — HIGH — Automatic public event subscription
`init()` opens the SSE event stream immediately and reconnects forever; no authenticated subscription or user choice occurs.

### KAI-DASHUI-052 — HIGH — Automatic workload amplification
Initialisation immediately runs mode sync, local-history render, SSE, health and full Dashboard refresh and schedules recurring checks before the user interacts.

---

## Medium-severity findings

### KAI-DASHUI-053 — MEDIUM — Permanent notification polling
The ten-second notification timer runs for the page lifetime, even when notifications are hidden or the document is backgrounded.

### KAI-DASHUI-054 — MEDIUM — No visibility/backpressure policy
Health, struggle, notifications and view refresh timers continue while hidden and do not suppress a new call when the previous one is pending.

### KAI-DASHUI-055 — MEDIUM — Overlapping async intervals
`setInterval()` does not await async callbacks, so slow backend responses create concurrent refreshes and out-of-order DOM state.

### KAI-DASHUI-056 — MEDIUM — Fixed infinite SSE retry
The client retries every five seconds without jitter, upper bound or offline/backpressure awareness.

### KAI-DASHUI-057 — MEDIUM — Silent client faults
Numerous `catch {}` blocks hide parse, permission, network and rendering failures while stale state remains visible.

### KAI-DASHUI-058 — MEDIUM — Numeric-view fragility
Direct `.toFixed()`, arithmetic and style interpolation on backend values can throw for missing strings/objects and abort a complete refresh.

### KAI-DASHUI-059 — MEDIUM — Invalid finance display
`fmt_gbp()` converts arbitrary data with `Number()` and can show `£NaN`, `£Infinity` or misleading locale output.

### KAI-DASHUI-060 — MEDIUM — Polling subtree churn
Large tables/cards are rebuilt as strings and assigned to `innerHTML` every refresh, increasing layout, garbage-collection and XSS surface.

### KAI-DASHUI-061 — MEDIUM — Promise assignment bug
Calendar refresh assigns an expression containing an unresolved `fetch(...).then(...)` Promise to `innerHTML` before immediately replacing it with today’s events.

### KAI-DASHUI-062 — MEDIUM — TTS object-URL leak
Manual stop clears `_ttsAudio` without retaining/revoking the object URL; failures before `onended` also skip revocation.

### KAI-DASHUI-063 — MEDIUM — Unvalidated TTS media
The browser trusts the response blob as playable audio, without content-type, byte or duration validation.

### KAI-DASHUI-064 — MEDIUM — Camera output-size assumption
Canvas dimensions use actual `videoWidth/videoHeight`; browser constraints are advisory and no explicit maximum pixel count is applied.

### KAI-DASHUI-065 — MEDIUM — Stale presence state
Vision failures are silently ignored, leaving the previous “present/emotion” result displayed as though current.

### KAI-DASHUI-066 — MEDIUM — Stale browse/search context
Failures do not clearly clear or provenance-mark previously inserted content; users can send stale material believing it reflects the requested source.

### KAI-DASHUI-067 — MEDIUM — Unversioned local settings
Settings/onboarding JSON has no schema version, type validation or migration path.

### KAI-DASHUI-068 — MEDIUM — PWA persistence of sensitive origin
Installing the UI preserves convenient access to the unauthenticated privileged origin and its local chat/settings state without an application lock.

### KAI-DASHUI-069 — MEDIUM — No sensitive-view lock
The client never expires, locks or reauthenticates before showing finance, email, logs, memories, SOUL or broker state.

### KAI-DASHUI-070 — MEDIUM — Missing unload cleanup
No central teardown closes EventSource, clears all timers, stops D3 simulations, revokes media URLs or stops camera/microphone streams.

### KAI-DASHUI-071 — MEDIUM — Data collection before onboarding
Initial Dashboard/SSE/health actions occur before the first-run wizard establishes any user preference or consent.

### KAI-DASHUI-072 — MEDIUM — Weak CIS client validation
Gross is checked only for truthiness/positive; materials can become non-finite, while contractor/reference lengths/content are unrestricted.

### KAI-DASHUI-073 — MEDIUM — Status-blind JSON parsing
Many helpers call `fetch(...).then(r => r.json())` without checking status, so error/fallback payloads flow into ordinary render/mutation logic.

### KAI-DASHUI-074 — MEDIUM — Stale state after silent failure
Refresh failures usually leave the prior successful view intact, with no “last updated/stale” marker.

### KAI-DASHUI-075 — MEDIUM — Excess object disclosure in UI messages
Backend objects are frequently JSON-stringified into notifications/toasts or copied into local client state without field minimisation.

### KAI-DASHUI-076 — MEDIUM — No client operation identity
Mutations contain no client-generated idempotency/correlation ID, making duplicate clicks/retries and backend committed-but-timed-out results difficult to reconcile.

---

## Batch totals

- Findings: **76**
- Critical: **9**
- High: **43**
- Medium: **24**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,724**
- Critical: **148**
- High: **825**
- Medium: **748**
- Low: **3**

## Files materially reviewed

`dashboard/static/app.html`, with same-origin backend and deployment context confirmed against `dashboard/app.py` and Dashboard service definitions.
