# Kai Code Audit — Dashboard Privileged Gateway Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers the Dashboard FastAPI backend. Findings in proxied services remain in their own batches and are not repeated unless the Dashboard creates a distinct gateway, confused-deputy, fan-out or failure-semantics defect.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DASH-001 | CRITICAL | The host-published Dashboard privileged gateway has no authentication or authorisation |
| KAI-DASH-002 | CRITICAL | Anonymous callers can use the Dashboard’s server-held bearer token to change Tool Gate mode |
| KAI-DASH-003 | CRITICAL | Anonymous callers can rewrite and immediately activate Agentic `SOUL.md` through the Dashboard |
| KAI-DASH-004 | CRITICAL | Anonymous callers can rewrite and reload the Agentic agent registry through the Dashboard |
| KAI-DASH-005 | CRITICAL | Anonymous Dashboard chat exposes the operator’s private Agentic memory/context and LLM execution |
| KAI-DASH-006 | CRITICAL | Anonymous callers can rewrite values, conscience, loyalty and gratitude state through P20 proxies |
| KAI-DASH-007 | CRITICAL | Anonymous callers can create/cancel reminders and scheduled tasks that Supervisor delivers externally |
| KAI-DASH-008 | CRITICAL | Anonymous callers can navigate, scrape and run browser-agent workflows |
| KAI-DASH-009 | CRITICAL | Anonymous callers can create monitor rules with external HTTP sources and notify/TTS actions |
| KAI-DASH-010 | CRITICAL | Anonymous callers can add file-watcher directories through the Dashboard |
| KAI-DASH-011 | HIGH | Incoming callers have no verified principal, role, session ownership or delegated service identity |
| KAI-DASH-012 | HIGH | The Dashboard bearer token is not bound to the initiating caller, reason or request context |
| KAI-DASH-013 | HIGH | Mode-sync failure returns HTTP 200 and permits browser/server security-mode divergence |
| KAI-DASH-014 | HIGH | Generic POST proxy retries can duplicate non-idempotent state-changing operations |
| KAI-DASH-015 | HIGH | Shared proxy resilience classifies every 4xx response as a successful dependency call |
| KAI-DASH-016 | HIGH | Backend failures are routinely returned as HTTP-200 fallback objects |
| KAI-DASH-017 | HIGH | Raw JSON proxy bodies have no aggregate byte, nesting or field-count limits |
| KAI-DASH-018 | HIGH | Dashboard exposes broad backend authority without route-specific scopes or least privilege |
| KAI-DASH-019 | HIGH | Anonymous callers can trigger dream consolidation and durable self-improvement writes |
| KAI-DASH-020 | HIGH | Anonymous callers can retrieve security-audit bypass payloads and defensive weaknesses |
| KAI-DASH-021 | HIGH | Thinking-pathway API exposes private episode input/output and learning metadata |
| KAI-DASH-022 | HIGH | Memory browsing proxies expose private memory records to arbitrary Dashboard callers |
| KAI-DASH-023 | HIGH | Recent-memory and graph endpoints hard-code the global `keeper` identity |
| KAI-DASH-024 | HIGH | Memory graph data exposes snippets, trust tier, importance, pin and access-count metadata |
| KAI-DASH-025 | HIGH | Emotional, identity, relationship and operator-model state is globally exposed |
| KAI-DASH-026 | HIGH | Goal creation and progress mutation are exposed without identity or ownership checks |
| KAI-DASH-027 | HIGH | Feedback proxy enables anonymous high-authority correction/positive-memory poisoning |
| KAI-DASH-028 | HIGH | Autobiography, legacy, inner-thought and aspiration proxies permit identity-state poisoning |
| KAI-DASH-029 | HIGH | Nudge escalation proxy enables anonymous escalation of externally delivered messages |
| KAI-DASH-030 | HIGH | Echo, cross-mode, oracle and shadow proxies expose or poison private operator-model state |
| KAI-DASH-031 | HIGH | Financial summary and CIS/tax information are exposed without authentication |
| KAI-DASH-032 | HIGH | CIS payment records can be created through the unauthenticated Dashboard |
| KAI-DASH-033 | HIGH | Broker balances, positions, orders and P&L are exposed without authentication |
| KAI-DASH-034 | HIGH | Email inbox/unread content is exposed and refresh can be triggered anonymously |
| KAI-DASH-035 | HIGH | Clipboard content/history can be read, pushed or cleared anonymously |
| KAI-DASH-036 | HIGH | Notifications can be sent or dismissed anonymously |
| KAI-DASH-037 | HIGH | Screen watching can be started or stopped anonymously |
| KAI-DASH-038 | HIGH | News-feed refresh can be triggered anonymously |
| KAI-DASH-039 | HIGH | PII-scanning proxy forwards caller text to Verifier without identity or data-purpose controls |
| KAI-DASH-040 | HIGH | Aggregated Agentic and memU logs are exposed without authentication |
| KAI-DASH-041 | HIGH | Redis health, episode, breaker and memory events are streamed publicly over SSE |
| KAI-DASH-042 | HIGH | Every SSE client consumes a dedicated Redis connection/pubsub without admission limits |
| KAI-DASH-043 | HIGH | One malformed Redis event payload terminates the client stream |
| KAI-DASH-044 | HIGH | Event channels have no user/tenant filtering or event-level authorisation |
| KAI-DASH-045 | HIGH | File upload reads the complete request before enforcing the 10 MB limit |
| KAI-DASH-046 | HIGH | Audio transcription uploads have no size limit |
| KAI-DASH-047 | HIGH | Vision/presence frame uploads have no size limit |
| KAI-DASH-048 | HIGH | TTS input and returned audio are unbounded |
| KAI-DASH-049 | HIGH | Browser screenshots are fully materialised without a response-size limit |
| KAI-DASH-050 | HIGH | Upload routing trusts filename extension and caller-provided content type |
| KAI-DASH-051 | HIGH | Uploaded filenames are forwarded to parser/OCR services without canonicalisation |
| KAI-DASH-052 | HIGH | Proxy error details disclose internal service URLs and transport diagnostics |
| KAI-DASH-053 | HIGH | Chat request bodies are unbounded |
| KAI-DASH-054 | HIGH | Chat proxy streams backend 4xx/5xx bodies without first validating response status |
| KAI-DASH-055 | HIGH | Chat connection exceptions are sent to the browser as internal diagnostic text |
| KAI-DASH-056 | HIGH | No global rate limit, concurrency cap or caller quota protects the privileged gateway |
| KAI-DASH-057 | HIGH | Node health checks run sequentially and can consume the sum of all service timeouts |
| KAI-DASH-058 | HIGH | Root status duplicates fleet probes inside `build_go_no_go_report()` and additional widgets |
| KAI-DASH-059 | HIGH | The built-in UI polls the expensive root fan-out every two seconds per browser |
| KAI-DASH-060 | HIGH | Readiness calls the full root fan-out instead of a bounded cached readiness check |
| KAI-DASH-061 | HIGH | Every successful HTTP response is classified as a healthy node regardless of semantic status |
| KAI-DASH-062 | HIGH | `core_ready` accepts fallback zero ledger/memory counts as valid readiness evidence |
| KAI-DASH-063 | HIGH | Go/no-go counts all ledger entries rather than recent approved, successful decisions |
| KAI-DASH-064 | HIGH | Go/no-go uses Dashboard caller error ratio rather than system execution/fleet reliability |
| KAI-DASH-065 | HIGH | Backup status reports a fresh healthy timestamp without verifying any backup exists or completed |
| KAI-DASH-066 | HIGH | Corrections API fabricates the current time for aggregate verdict counters |
| KAI-DASH-067 | HIGH | Many backend outages silently become empty lists or neutral data rather than degraded state |
| KAI-DASH-068 | HIGH | Root response exposes internal topology, policy, breakers, quarantine and verifier details |
| KAI-DASH-069 | MEDIUM | Health exposes Tool Gate URL, policy version and policy hash |
| KAI-DASH-070 | MEDIUM | Dashboard health always reports the process as running and is not a readiness contract |
| KAI-DASH-071 | MEDIUM | Node inventory is manually maintained and can drift from deployed services |
| KAI-DASH-072 | MEDIUM | Backend URLs are environment-controlled without scheme/host/identity validation |
| KAI-DASH-073 | MEDIUM | Backend service identity and deployment version are not verified |
| KAI-DASH-074 | MEDIUM | Many direct proxy routes create a new HTTP client and connection pool per request |
| KAI-DASH-075 | MEDIUM | Shared resilient proxy creates a new client for every retry attempt |
| KAI-DASH-076 | MEDIUM | Backend response bodies and nested JSON are not byte/depth bounded |
| KAI-DASH-077 | MEDIUM | Root status returns complete nested backend health details rather than minimised state |
| KAI-DASH-078 | MEDIUM | Go/no-go thresholds are parsed without complete safe-range validation |
| KAI-DASH-079 | MEDIUM | Malformed backend numeric fields can raise during go/no-go conversion |
| KAI-DASH-080 | MEDIUM | A `NO_GO` decision is returned with HTTP 200 rather than a machine-enforcing status |
| KAI-DASH-081 | MEDIUM | Missing Dashboard Gate token creates a knowingly divergent browser-local mode |
| KAI-DASH-082 | MEDIUM | Nudge and correction outages are silently represented as no data |
| KAI-DASH-083 | MEDIUM | Synthetic correction/backup timestamps use naive UTC strings |
| KAI-DASH-084 | MEDIUM | SSE keepalive timestamps are naive UTC strings |
| KAI-DASH-085 | MEDIUM | Redis clients are constructed per event publisher/stream rather than lifecycle-managed |
| KAI-DASH-086 | MEDIUM | `_publish_event()` is an unaudited fire-and-forget helper with silent delivery loss |
| KAI-DASH-087 | MEDIUM | Unified app shell reads the complete HTML file synchronously on each request |
| KAI-DASH-088 | MEDIUM | HTML/static responses set no Content-Security-Policy, frame or referrer protections |
| KAI-DASH-089 | MEDIUM | TTS proxy forces `audio/mpeg` regardless of the backend’s actual content type |
| KAI-DASH-090 | MEDIUM | Screenshot proxy forces `image/png` regardless of backend response type |
| KAI-DASH-091 | MEDIUM | Audio and vision proxies trust caller-supplied media types and filenames |
| KAI-DASH-092 | MEDIUM | Binary/audio/image responses are fully materialised before forwarding |
| KAI-DASH-093 | MEDIUM | Numerous list/search limits accept negative or extreme values |
| KAI-DASH-094 | MEDIUM | Path parameters and symbols are interpolated into backend URLs without canonical enums |
| KAI-DASH-095 | MEDIUM | Broker-watch symbol and threshold validation permits malformed or non-finite rules |
| KAI-DASH-096 | MEDIUM | Audit logging is optional and records only method/path/status, not actor or operation digest |

---

## Critical gateway findings

### KAI-DASH-001 — CRITICAL — Open privileged gateway
**Issue:** Compose publishes `8080:8080`. The Dashboard defines no inbound authentication, authorisation or principal/session ownership checks while proxying dozens of private and state-changing services.  
**Risk:** Any reachable caller obtains one unified control/data plane for the Sovereign stack.  
**Recommendation:** place the Dashboard behind strong user authentication, least-privilege backend credentials, CSRF/origin controls and endpoint-specific authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-DASH-002 — CRITICAL — Server-token confused deputy
**Issue:** When `DASHBOARD_GATE_TOKEN` is configured, any anonymous `POST /api/mode` causes Dashboard to authenticate to Tool Gate with its trusted bearer token and change WORK/PUB mode. The initiating caller is not represented.  
**Risk:** Anonymous callers borrow Dashboard’s internal privilege to change execution policy.  
**Recommendation:** authenticate the operator, use per-user delegated credentials and bind actor/reason/request ID to the Tool Gate event.  
**Status:** OPEN — immediate remediation required

### KAI-DASH-003 — CRITICAL — SOUL identity rewrite proxy
`POST /api/soul` forwards arbitrary JSON to Agentic’s live SOUL rewrite endpoint without caller identity, schema or review.

### KAI-DASH-004 — CRITICAL — Agent-registry rewrite proxy
`POST /api/agents-registry` exposes live AGENTS.md mutation/reload through port 8080.

### KAI-DASH-005 — CRITICAL — Open private-context chat
`POST /api/chat` forwards arbitrary Agentic chat bodies. Agentic reads/writes global `keeper` memory, sensory, financial, identity and conscience context and performs LLM execution.

### KAI-DASH-006 — CRITICAL — Moral-model mutation gateway
P20 proxies expose value learning, conscience checks, loyalty and gratitude creation; these states are later injected into Agentic prompts and trust/alignment logic.

### KAI-DASH-007 — CRITICAL — External-message scheduling gateway
Task/reminder proxies permit anonymous creation/cancellation. Supervisor polls memU and sends due content to Telegram.

### KAI-DASH-008 — CRITICAL — Browser automation gateway
Navigate, scrape, search, run and screenshot routes expose browser-agent actions to unauthenticated callers.

### KAI-DASH-009 — CRITICAL — Monitoring/action-rule gateway
Anonymous callers can create monitor rules whose sources perform HTTP retrieval and whose actions send notifications/TTS, then enable/disable/check/delete them.

### KAI-DASH-010 — CRITICAL — File-watcher path gateway
`POST /api/files/watch` forwards caller-selected directory configuration to the filesystem watcher.

---

## High-severity authority and disclosure findings

### KAI-DASH-011 — HIGH — No principal model
Requests carry no authenticated person/service, role, tenant, session owner or delegated backend scope.

### KAI-DASH-012 — HIGH — Token use lacks delegation evidence
Dashboard’s Tool Gate token is static and does not identify the browser caller, requested reason, client session or approval event.

### KAI-DASH-013 — HIGH — Mode failure appears as normal success
Transport/backend failures return `{status:"sync_failed"}` with HTTP 200; the browser may keep a local mode that disagrees with the enforcement authority.

### KAI-DASH-014 — HIGH — Retried mutation duplication
`_proxy_post` uses `resilient_call(... retries=2)`. A timed-out first request may have committed before the second goal, feedback, schedule, notification, value or other mutation is sent.

### KAI-DASH-015 — HIGH — 4xx treated as dependency success
Shared resilience returns parsed JSON and resets its circuit for every status below 500, including authentication, validation and safety rejections.

### KAI-DASH-016 — HIGH — Success-shaped fallbacks
Most proxy failures become ordinary HTTP-200 dictionaries such as `unavailable`, empty arrays or `ok:false`; browser automation cannot reliably distinguish transport, policy and empty-state outcomes.

### KAI-DASH-017 — HIGH — Unbounded JSON fan-in
Most mutation proxies call `await request.json()` and forward the entire nested body without a Dashboard schema or byte/depth limit.

### KAI-DASH-018 — HIGH — No backend least privilege
One process can call identity, finance, browser, file, email, monitoring, notification, memory and security endpoints; credentials/network access are not separated by route or role.

### KAI-DASH-019 — HIGH — Dream mutation exposure
`POST /api/dream` triggers Agentic Introspection consolidation, high-importance memory writes and checkpoint creation.

### KAI-DASH-020 — HIGH — Security exploit disclosure
`GET /api/security-audit` returns self-audit findings and bypass payloads from Agentic Introspection.

### KAI-DASH-021 — HIGH — Episode disclosure
`/api/thinking` recalls global keeper episodes and returns input, output, conviction, failure class, metacognitive rule and learning value.

### KAI-DASH-022 — HIGH — Memory browsing disclosure
Query/category/recent-memory routes expose memory content and metadata without user authorisation.

### KAI-DASH-023 — HIGH — Hard-coded keeper scope
Recent-memory and graph routes explicitly request `user_id=keeper`, converting every Dashboard caller into the operator identity.

### KAI-DASH-024 — HIGH — Graph view leaks security/ranking metadata
Returned nodes include memory snippets, trust tier, importance/relevance, pin and access count.

### KAI-DASH-025 — HIGH — Personal-model disclosure
Emotion, relationship, identity, story arcs, future self, empathy, inner monologue, aspirations, conscience, loyalty, gratitude, echo, cross-mode and operator-model endpoints are public.

### KAI-DASH-026 — HIGH — Goal mutation exposure
Anonymous clients can create goals and update their progress through raw body forwarding.

### KAI-DASH-027 — HIGH — Feedback poisoning gateway
Feedback body is forwarded to memU’s global feedback/correction engine, creating high-authority memories and calibration effects.

### KAI-DASH-028 — HIGH — Narrative-state poisoning gateway
Autobiography, legacy, counterfactual, empathy, thought and aspiration mutation routes accept anonymous content.

### KAI-DASH-029 — HIGH — Escalation poisoning gateway
Nudge escalation requests feed Supervisor’s externally delivered escalation ladder.

### KAI-DASH-030 — HIGH — Operator-model manipulation/disclosure
Echo analysis, cross-mode scan, oracle prediction and shadow branching use/return private global memories and derived state.

### KAI-DASH-031 — HIGH — Financial disclosure
CIS/VAT/tax estimates, invoices and summary data are public.

### KAI-DASH-032 — HIGH — Financial record mutation
CIS payment creation body is forwarded directly to Financial Awareness.

### KAI-DASH-033 — HIGH — Broker disclosure
Balances, positions, orders, P&L and market/account state are public.

### KAI-DASH-034 — HIGH — Email disclosure/control
Inbox and unread samples are public and refresh can be triggered repeatedly.

### KAI-DASH-035 — HIGH — Clipboard disclosure/control
Latest/history reads, push and clear expose or alter copied private data.

### KAI-DASH-036 — HIGH — Notification control
Anonymous callers can create, dismiss individual or dismiss all notifications.

### KAI-DASH-037 — HIGH — Screen-watcher control
Watch start/stop is publicly proxied.

### KAI-DASH-038 — HIGH — News refresh control
Anonymous requests trigger feed/network refresh work.

### KAI-DASH-039 — HIGH — Sensitive-text forwarding
PII scan sends arbitrary caller text to Verifier without authentication, declared purpose or retention controls.

### KAI-DASH-040 — HIGH — Log aggregation disclosure
Dashboard combines memU and Agentic log entries and returns them to every caller.

### KAI-DASH-041 — HIGH — Public internal event bus
SSE subscribes to shared health, episode, breaker and memory Redis channels and streams event data without authorisation.

### KAI-DASH-042 — HIGH — SSE connection exhaustion
Each browser opens its own Redis connection and pubsub subscription; there is no connection limit or authenticated quota.

### KAI-DASH-043 — HIGH — Malformed event denial
`json.loads(msg["data"])` is not protected per message. One malformed publisher event terminates the stream generator.

### KAI-DASH-044 — HIGH — No event-level isolation
All subscribers receive the same channels; user/session/source fields are not used for filtering.

### KAI-DASH-045 — HIGH — Post-read upload limit
`await file.read()` materialises the complete request before checking 10 MB, so the limit does not protect memory/network ingestion.

### KAI-DASH-046 — HIGH — Unlimited audio upload
Audio capture reads the complete file and forwards it with no size or duration bound.

### KAI-DASH-047 — HIGH — Unlimited vision upload
Both vision routes read complete frames without size/dimension/decompression limits.

### KAI-DASH-048 — HIGH — Unlimited TTS work/response
Text/body size is not bounded and complete backend audio is materialised before returning.

### KAI-DASH-049 — HIGH — Unlimited screenshot response
Complete browser screenshot bytes are materialised without a maximum.

### KAI-DASH-050 — HIGH — Extension/MIME trust
Upload routing is based on filename suffix; caller content type is forwarded unchanged. No magic-byte or safe-container validation occurs here.

### KAI-DASH-051 — HIGH — Filename propagation
The raw uploaded filename is forwarded into multipart requests, potentially entering downstream logs, parsers and temporary-file handling.

### KAI-DASH-052 — HIGH — Internal error disclosure
HTTP exception details contain backend service names, URLs/status and transport exception text.

### KAI-DASH-053 — HIGH — Unbounded chat body
The complete JSON body is loaded and streamed to Agentic without a Dashboard schema or context/message limit.

### KAI-DASH-054 — HIGH — Chat status not validated
The streaming proxy does not call `raise_for_status()` before yielding bytes, so backend rejection/error bodies are presented as chat SSE.

### KAI-DASH-055 — HIGH — Chat diagnostics leak
Connection errors are embedded in SSE token text and sent to the browser.

### KAI-DASH-056 — HIGH — No gateway workload controls
LLM chat, browser, upload, dream, security audit, status fan-out, SSE and all mutation routes lack global/per-route admission control.

### KAI-DASH-057 — HIGH — Sequential health fan-out
`fetch_status()` loops over nodes sequentially with a two-second timeout each. One request can occupy the worker for the cumulative node timeout.

### KAI-DASH-058 — HIGH — Duplicate root fan-out
Root first calls `fetch_status()`, then `build_go_no_go_report()` calls it again, then separate Supervisor/memU/Verifier probes run.

### KAI-DASH-059 — HIGH — UI amplification loop
Every `/ui` browser calls the expensive root endpoint every two seconds.

### KAI-DASH-060 — HIGH — Readiness amplification
`/readiness` calls `index()`, performing the full status/ledger/memory/breaker/quarantine/verifier fan-out instead of reading bounded cached readiness.

### KAI-DASH-061 — HIGH — HTTP success equals node health
`fetch_status()` uses `raise_for_status()` and then assigns `status: ok`, ignoring backend `degraded`, `disabled`, `stub` or failed nested checks.

### KAI-DASH-062 — HIGH — False core readiness
Ledger/memory retrieval failures leave counts at zero, and `ledger_size >= 0 and memory_count >= 0` still passes. Only HTTP-alive node names matter.

### KAI-DASH-063 — HIGH — Invalid proof metric
Go/no-go treats total ledger entry count as proof, regardless of denials, failures, age, actor, policy version or successful execution outcomes.

### KAI-DASH-064 — HIGH — Wrong reliability metric
The error ratio is Dashboard’s own incoming HTTP response history, not fleet health, Gate decisions, executor outcomes or dependency errors.

### KAI-DASH-065 — HIGH — False backup status
A successful backup-service `/health` response is converted into “current timestamp (service healthy)” without checking backup inventory, recency, integrity or restoreability.

### KAI-DASH-066 — HIGH — Fabricated correction chronology
Aggregate Verifier counters are converted into correction entries stamped with the current request time, not actual verdict times.

### KAI-DASH-067 — HIGH — Evidence outage becomes absence
Numerous endpoints catch all exceptions and return empty lists/neutral values, making unavailable private/safety data look genuinely empty.

### KAI-DASH-068 — HIGH — Root operational disclosure
The root response exposes node details/errors, internal URLs, policy digest/mode, counts, breaker snapshots, quarantine count and full Verifier metrics.

---

## Medium-severity health, proxy and response findings

### KAI-DASH-069 — MEDIUM — Health topology disclosure
Health returns Tool Gate URL plus policy version/hash.

### KAI-DASH-070 — MEDIUM — Liveness mislabeled as health
Health always reports running CPU/CUDA and never checks Redis, static files, Tool Gate, memU or required proxy dependencies.

### KAI-DASH-071 — MEDIUM — Inventory drift
NODES is manually assembled and optional services appear only when selected environment variables are set.

### KAI-DASH-072 — MEDIUM — Unvalidated backend destinations
Environment strings become outbound URLs without required schemes, approved hosts, service names or TLS policy.

### KAI-DASH-073 — MEDIUM — No backend identity proof
Any process responding at the configured URL is accepted as that service; version/deployment identity is not checked.

### KAI-DASH-074 — MEDIUM — Direct-client churn
Most individual routes construct/destroy an `AsyncClient` rather than using lifecycle-managed pools.

### KAI-DASH-075 — MEDIUM — Retry-client churn
`resilient_call()` constructs a new client for each retry attempt.

### KAI-DASH-076 — MEDIUM — Unbounded backend responses
JSON/binary response bodies are fully parsed/materialised without an aggregate byte/depth limit.

### KAI-DASH-077 — MEDIUM — Excess health detail
Root returns complete nested service health documents instead of a versioned minimal readiness schema.

### KAI-DASH-078 — MEDIUM — Unsafe go/no-go configuration
Grace-count and error-ratio values are parsed without finite, positive and relationship validation.

### KAI-DASH-079 — MEDIUM — Malformed numeric backend data
Direct `int()`/`float()` conversion of ledger/error fields can raise and fail the report.

### KAI-DASH-080 — MEDIUM — Advisory-only NO_GO
The endpoint always returns 200 and no downstream execution authority is bound to its result.

### KAI-DASH-081 — MEDIUM — Deliberate mode split
Without a configured token, `/api/mode` returns `local_only`; the UI can display a mode that the server does not enforce.

### KAI-DASH-082 — MEDIUM — Silent empty-state fallbacks
Nudges and corrections return empty arrays on any failure with no degraded/source state.

### KAI-DASH-083 — MEDIUM — Naive synthetic times
Backup/correction display times use naive UTC strings with no event/source identity.

### KAI-DASH-084 — MEDIUM — Naive SSE heartbeat time
Keepalive events use `datetime.utcnow().isoformat()` without timezone marker.

### KAI-DASH-085 — MEDIUM — Redis lifecycle churn
Publisher and each SSE stream create separate Redis clients rather than shared managed resources.

### KAI-DASH-086 — MEDIUM — Silent event loss
`_publish_event()` suppresses every Redis failure and provides no durable delivery state.

### KAI-DASH-087 — MEDIUM — Blocking app-shell read
`/app` synchronously opens and reads the complete HTML file on every request.

### KAI-DASH-088 — MEDIUM — Missing browser security headers
HTML/static/SSE responses set no CSP, `frame-ancestors`/X-Frame-Options, Referrer-Policy or Permissions-Policy.

### KAI-DASH-089 — MEDIUM — Forced TTS media type
Dashboard labels every successful TTS response `audio/mpeg` instead of validating/forwarding backend content type.

### KAI-DASH-090 — MEDIUM — Forced screenshot media type
Every screenshot response is labelled PNG regardless of actual bytes/content type.

### KAI-DASH-091 — MEDIUM — Caller media metadata trusted
Audio/vision multipart requests use caller filenames and content types without canonical validation.

### KAI-DASH-092 — MEDIUM — Binary response buffering
Audio, screenshot and other media are buffered completely rather than streamed with limits.

### KAI-DASH-093 — MEDIUM — Weak query limits
Log, memory, graph, email, news, trade, event and alert limits are not consistently constrained to positive safe maxima.

### KAI-DASH-094 — MEDIUM — Path interpolation
Symbols, rule IDs, notification IDs, session IDs and other caller fields are interpolated into backend paths without canonical enums/encoding policy.

### KAI-DASH-095 — MEDIUM — Weak broker-watch rule validation
Symbol is only uppercased; threshold accepts direct float conversion including non-finite values, creating malformed monitoring rules.

### KAI-DASH-096 — MEDIUM — Weak optional audit
`AUDIT_REQUIRED` defaults false. Events contain only method/path/status and no caller identity, request/body digest, target backend or mutation outcome.

---

## Batch totals

- Findings: **96**
- Critical: **10**
- High: **58**
- Medium: **28**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,648**
- Critical: **139**
- High: **782**
- Medium: **724**
- Low: **3**

## Files materially reviewed

`dashboard/app.py`, Dashboard deployment in `docker-compose.minimal.yml`/`docker-compose.full.yml`, and direct proxy integration against Agentic, memU, Supervisor, Tool Gate, Financial Awareness, Browser Agent, Monitor, Files, Notify, Email, Broker and related service APIs.
