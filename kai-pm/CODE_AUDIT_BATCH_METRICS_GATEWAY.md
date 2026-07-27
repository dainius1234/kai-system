# Kai Code Audit — Metrics Gateway Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-METRICS-001 | CRITICAL | Default deployment exposes aggregated health and raw metrics from security-sensitive services without authentication |
| KAI-METRICS-002 | HIGH | Prometheus metrics, manual scrape and service registry endpoints bypass configured authentication |
| KAI-METRICS-003 | HIGH | Unauthenticated callers can force parallel scraping of every registered service |
| KAI-METRICS-004 | HIGH | HTTP 200 reachability is treated as service health regardless of readiness semantics |
| KAI-METRICS-005 | HIGH | Failed scrapes leave stale metrics cached and presented as current |
| KAI-METRICS-006 | HIGH | Default registry omits multiple actively deployed services |
| KAI-METRICS-007 | HIGH | Custom registry permits arbitrary destinations, duplicate names and unvalidated schemes |
| KAI-METRICS-008 | HIGH | Manual and background scrapes can overlap without locking or generation control |
| KAI-METRICS-009 | MEDIUM | Authentication is optional, defaults off and uses direct string comparison |
| KAI-METRICS-010 | MEDIUM | Health and metrics response sizes and JSON complexity are unbounded |
| KAI-METRICS-011 | MEDIUM | Scrape task exceptions can disappear without updating service downtime history |
| KAI-METRICS-012 | MEDIUM | Prometheus label values and metric values are emitted without escaping or type validation |
| KAI-METRICS-013 | MEDIUM | Error strings from internal service access are exposed through aggregate output |
| KAI-METRICS-014 | MEDIUM | Background scraper has no shutdown lifecycle |
| KAI-METRICS-015 | MEDIUM | Metrics, health, uptime and task state are process-local and lost on restart |
| KAI-METRICS-016 | MEDIUM | Uptime measures only recent gateway reachability samples, not actual service availability |
| KAI-METRICS-017 | MEDIUM | Health reports ok when the scraper is inactive or all dependencies are unavailable |
| KAI-METRICS-018 | MEDIUM | Error-budget telemetry, intervals, timeouts, URLs and port configuration are not properly validated |

---

## Metrics gateway: `metrics-gateway/app.py`

### KAI-METRICS-001 — CRITICAL — Central unauthenticated telemetry disclosure
**Issue:** Authentication is enforced only when `METRICS_AUTH_TOKEN` is non-empty. The full Compose deployment does not configure that variable. `/metrics` therefore returns cached raw health and metrics payloads from Tool Gate, memu-core, executor, agentic, heartbeat, supervisor, verifier, fusion-engine, memory-compressor and ledger-worker without authentication.  
**Risk:** One public endpoint aggregates security-sensitive operational state, errors, memory-maintenance history, ledger verification results and other downstream telemetry that would otherwise require probing multiple services.  
**Recommendation:** Require authentication by default, fail startup when a protected deployment lacks credentials and minimise/redact downstream payloads before aggregation.  
**Status:** OPEN — immediate remediation required

### KAI-METRICS-002 — HIGH — Auth coverage is incomplete
**Issue:** `_check_auth` is called only by `/metrics` and `/fleet`. `/metrics/text`, `POST /scrape` and `/registry` never call it, despite the comment claiming all data endpoints require auth when configured.  
**Risk:** A configured token does not protect service topology, health/error-rate exposition or active scrape control.  
**Recommendation:** Apply one authentication/authorisation middleware to every non-liveness endpoint.  
**Status:** OPEN

### KAI-METRICS-003 — HIGH — Public fleet-wide scrape trigger
**Issue:** `POST /scrape` is unauthenticated and immediately performs health and metrics requests to every registered service in parallel.  
**Risk:** Repeated callers can amplify internal network traffic and expensive metrics generation across the whole stack, including services whose metrics endpoints perform non-trivial serialisation.  
**Recommendation:** Restrict manual scraping, rate-limit requests and enforce one bounded in-flight scrape.  
**Status:** OPEN

### KAI-METRICS-004 — HIGH — Reachability is misreported as health
**Issue:** Any health endpoint returning HTTP 200 is marked `_reachable: true` and counted up. The response body’s `status`, readiness, disabled/stub state, scheduler state and freshness do not affect the up metric.  
**Risk:** Non-functional stubs and services that always report `status: ok` despite failed dependencies are shown healthy, compounding false readiness across the stack.  
**Recommendation:** Validate service-specific readiness contracts and distinguish transport reachability from operational health.  
**Status:** OPEN

### KAI-METRICS-005 — HIGH — Stale successful metrics survive failures
**Issue:** `_latest_metrics[name]` is updated only when a new metrics response succeeds. A later failed scrape sets no replacement or stale marker, so the previous metrics remain indefinitely while `/metrics` presents them without per-payload age.  
**Risk:** Operators and automation can act on obsolete error rates and state after the source service became unavailable.  
**Recommendation:** Publish timestamped snapshots and explicitly expire/clear metrics on failed or overdue scrapes.  
**Status:** OPEN

### KAI-METRICS-006 — HIGH — Registry coverage is incomplete
**Issue:** `DEFAULT_SERVICES` is manually maintained and omits multiple services active in `docker-compose.full.yml`, including memu-core-introspect, audio, camera, wake, TTS, avatar, screen-capture, backup, calendar-sync, telegram, workspace-manager, skill-hunter, house-doctor, Letta and financial-awareness.  
**Risk:** Fleet status can appear complete and green while significant active services are unobserved.  
**Recommendation:** Generate the registry from a versioned deployment manifest and fail coverage checks when deployed services are missing.  
**Status:** OPEN

### KAI-METRICS-007 — HIGH — Custom registry is an unvalidated network authority
**Issue:** `METRICS_SERVICES` accepts arbitrary `name=url` pairs. Schemes, hosts, ports, paths, duplicates and network ranges are not validated. Duplicate names overwrite shared caches and display entries.  
**Risk:** Compromised configuration can turn the gateway into a periodic requester of arbitrary internal/external destinations, hide legitimate services under duplicate names and expose returned data.  
**Recommendation:** Restrict registry entries to an approved signed service manifest with unique identities and network policy.  
**Status:** OPEN

### KAI-METRICS-008 — HIGH — Scrape publication races
**Issue:** The background loop and manual endpoint both call `scrape_all` with no lock. Each concurrent run updates shared dictionaries, uptime histories and `_last_scrape` item by item.  
**Risk:** Older/slower runs can overwrite newer health/metrics, append duplicate samples and publish mixed-generation fleet snapshots.  
**Recommendation:** Use one scrape coordinator and atomically publish a complete versioned generation.  
**Status:** OPEN

### KAI-METRICS-009 — MEDIUM — Authentication defaults fail open
**Issue:** An empty token disables auth entirely. Token comparison uses normal string equality and the code does not validate token strength, source or rotation.  
**Risk:** Omitted/mistyped configuration silently creates public telemetry; comparison is not hardened for secret verification.  
**Recommendation:** Fail closed, use secret-managed high-entropy tokens or mutual service identity, and compare secrets in constant time.  
**Status:** OPEN

### KAI-METRICS-010 — MEDIUM — Downstream payload allocation is unbounded
**Issue:** Health and metrics responses are fully materialised and parsed as JSON without response-byte, nesting, key-count or schema limits. Cached payloads are then returned wholesale.  
**Risk:** A compromised/malformed service can exhaust gateway memory and inject oversized aggregate responses.  
**Recommendation:** Enforce strict response limits and per-service typed schemas.  
**Status:** OPEN

### KAI-METRICS-011 — MEDIUM — Task-level failures are not recorded as downtime
**Issue:** `asyncio.gather(..., return_exceptions=True)` results that are exceptions are simply skipped. No service name is recovered, no health cache is updated and no false sample is appended.  
**Risk:** Catastrophic task failures preserve old health state and inflate uptime by omitting failed observations.  
**Recommendation:** Bind each task to an identity and record every scrape attempt as success or failure.  
**Status:** OPEN

### KAI-METRICS-012 — MEDIUM — Prometheus output is not safely encoded
**Issue:** Service names are inserted directly into label values without Prometheus escaping. `error_rate` values from arbitrary JSON are emitted without numeric type/finite-value validation.  
**Risk:** Configured names or malformed metrics can break exposition format, inject labels/lines or poison downstream scraping.  
**Recommendation:** Escape labels and validate all metric names/values against the exposition specification.  
**Status:** OPEN

### KAI-METRICS-013 — MEDIUM — Internal diagnostics are aggregated and exposed
**Issue:** Health exceptions are stored as raw `_error` strings and returned by `/metrics` when auth is disabled by default.  
**Risk:** Internal DNS, connection, routing and parser diagnostics are centralised for reconnaissance.  
**Recommendation:** Store stable error codes publicly and retain detailed traces only in protected logs.  
**Status:** OPEN

### KAI-METRICS-014 — MEDIUM — Scraper task lacks shutdown ownership
**Issue:** Startup creates `_scraper_loop` but no shutdown handler cancels and awaits it.  
**Risk:** Reloads/tests can create duplicate scrapers, and shutdown abandons in-flight fleet requests.  
**Recommendation:** Own the scraper in FastAPI lifespan with explicit cancellation and awaited completion.  
**Status:** OPEN

### KAI-METRICS-015 — MEDIUM — State is volatile and worker-local
**Issue:** Metrics, health, uptime history, scrape timestamp and task reference are module-level memory. Multiple workers run separate scrapers and expose inconsistent snapshots; restart erases uptime.  
**Risk:** Fleet state and uptime ratios are non-authoritative and vary by request routing.  
**Recommendation:** Run one scraper authority and store timestamped generations in shared telemetry storage.  
**Status:** OPEN

### KAI-METRICS-016 — MEDIUM — Uptime semantics are misleading
**Issue:** Uptime is the fraction of at most 100 gateway scrape samples where HTTP 200 was received. It resets on restart, excludes skipped task failures and does not account for time intervals or service readiness.  
**Risk:** The displayed ratio is not actual uptime and can materially overstate availability.  
**Recommendation:** Name it recent reachability-sample ratio or derive real time-based availability from durable observations.  
**Status:** OPEN

### KAI-METRICS-017 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when the scraper is disabled, crashed or no service has ever been scraped successfully.  
**Risk:** Orchestration treats a non-observing gateway as ready.  
**Recommendation:** Separate liveness, scraper activity, snapshot freshness and minimum-fleet readiness.  
**Status:** OPEN

### KAI-METRICS-018 — MEDIUM — Configuration and gateway telemetry are weak
**Issue:** `budget` is exposed but never records requests. Scrape interval, timeout, registry URLs and port are parsed directly; zero/negative intervals can create tight loops or invalid runtime behaviour.  
**Risk:** Reliability data is empty and misconfiguration causes uncontrolled scraping or startup failure.  
**Recommendation:** Record real request/scrape outcomes and validate typed configuration with strict ranges and approved destinations.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **1**
- High: **7**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **744**
- Critical: **82**
- High: **261**
- Medium: **398**
- Low: **3**

## Files materially reviewed in this batch

`metrics-gateway/app.py` and the relevant service definitions in `docker-compose.full.yml`.
