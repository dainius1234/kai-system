# Kai Code Audit — Weather Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WEATHER-001 | HIGH | Exact configured latitude and longitude are exposed without authentication |
| KAI-WEATHER-002 | HIGH | Failed refreshes preserve and serve stale weather as live situational context |
| KAI-WEATHER-003 | HIGH | Upstream response bytes and JSON complexity are unbounded |
| KAI-WEATHER-004 | HIGH | Unverified weather text is promoted into agent context without provenance or freshness enforcement |
| KAI-WEATHER-005 | MEDIUM | Health reports `ok` before first fetch and after fetch failure |
| KAI-WEATHER-006 | MEDIUM | Documented `feels_like_c` output is never returned |
| KAI-WEATHER-007 | MEDIUM | Open-Meteo response structure and numeric values are not schema-validated |
| KAI-WEATHER-008 | MEDIUM | Cache, error and task state are process-local and unsynchronised |
| KAI-WEATHER-009 | MEDIUM | Refresh-task cancellation is not awaited during shutdown |
| KAI-WEATHER-010 | MEDIUM | A new HTTP client and connection pool are created for every refresh |
| KAI-WEATHER-011 | MEDIUM | Raw upstream/network errors are exposed through health |
| KAI-WEATHER-012 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-WEATHER-013 | MEDIUM | Coordinates, intervals, port and location configuration are not validated |
| KAI-WEATHER-014 | MEDIUM | Location and upstream string fields are not length-bounded before responses/context assembly |

---

## Weather service: `weather-service/app.py`

### KAI-WEATHER-001 — HIGH — Exact location disclosure
**Issue:** `/health` requires no authentication and returns the configured `LOCATION_NAME`, latitude and longitude. The service is published on host port 8039 in the minimal Compose deployment.  
**Risk:** When configured for the operator’s home, workplace or current location, any reachable caller can obtain precise location coordinates and correlate them with weather/context activity.  
**Recommendation:** Keep exact coordinates internal, require authenticated owner access and expose only a coarse locality where necessary.  
**Status:** OPEN

### KAI-WEATHER-002 — HIGH — Stale weather is served as current
**Issue:** On fetch failure, `_fetch_error` changes but `_cache` and `_last_fetch` remain. `/current`, `/forecast` and `/summary` continue returning the previous data; only `/current` includes `fetched_at`, and none of the data endpoints exposes the active error or a maximum-age decision.  
**Risk:** Agentic workflows and users can act on obsolete severe-weather, precipitation, temperature or wind information while the service appears operational.  
**Recommendation:** Enforce a strict freshness TTL, mark all stale responses degraded and suppress agent-context use beyond the TTL.  
**Status:** OPEN

### KAI-WEATHER-003 — HIGH — Upstream payload allocation is unbounded
**Issue:** `_fetch_weather` materialises the complete HTTP response and calls `resp.json()` without response-byte, decompressed-size, nesting or field-count limits.  
**Risk:** A compromised/malformed upstream response can consume excessive memory and CPU before the fixed seven-day output is assembled.  
**Recommendation:** Stream with a strict byte cap and validate a bounded endpoint-specific schema.  
**Status:** OPEN

### KAI-WEATHER-004 — HIGH — Weak external data becomes privileged context
**Issue:** `/summary` is explicitly designed for agentic context injection. It combines environment-controlled location text and unverified provider values into a natural-language statement without source identity, confidence, freshness or instruction/data separation.  
**Risk:** Stale, malformed or configuration-poisoned text is promoted into the assistant’s situational model and can influence recommendations as trusted fact.  
**Recommendation:** Pass typed signed observations with source, sample time, expiry and confidence; never inject raw location/provider strings into privileged prompts.  
**Status:** OPEN

### KAI-WEATHER-005 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including before any successful fetch and while `_fetch_error` is populated.  
**Risk:** Compose and monitoring treat an empty or failed weather source as ready.  
**Recommendation:** Separate liveness, upstream connectivity and fresh-snapshot readiness.  
**Status:** OPEN

### KAI-WEATHER-006 — MEDIUM — Public contract is incomplete
**Issue:** The module endpoint documentation promises `feels_like_c`, and `_fetch_weather` requests hourly `apparent_temperature`, but `/current` never returns or aligns an apparent-temperature value.  
**Risk:** Consumers relying on the documented schema receive silently incomplete data and may substitute actual temperature for perceived temperature.  
**Recommendation:** Implement a time-aligned field or remove it from the contract and unused upstream query.  
**Status:** OPEN

### KAI-WEATHER-007 — MEDIUM — Upstream schema is trusted implicitly
**Issue:** The code assumes dictionaries, lists and numeric values at several paths. It converts weather codes with `int`, multiplies wind speed and indexes daily arrays without validating types, finite values, units or aligned lengths. Missing values are often converted to benign defaults such as weather code zero.  
**Risk:** Malformed data can produce server errors or false “clear sky”/zero values rather than an explicit invalid-snapshot state.  
**Recommendation:** Validate a strict typed response and reject incomplete/non-finite/misaligned data.  
**Status:** OPEN

### KAI-WEATHER-008 — MEDIUM — State is volatile and worker-local
**Issue:** Cache, timestamps, errors and refresh task are module-level process memory.  
**Risk:** Multiple workers poll independently and expose different forecasts; restart erases freshness/error history.  
**Recommendation:** Use one fetch authority and shared immutable timestamped snapshots.  
**Status:** OPEN

### KAI-WEATHER-009 — MEDIUM — Shutdown does not await refresh termination
**Issue:** Lifespan shutdown calls `_refresh_task.cancel()` but does not await it or close an in-flight client operation explicitly.  
**Risk:** Refresh work can continue or be abandoned during shutdown and task errors are not observed.  
**Recommendation:** Await cancellation and lifecycle cleanup.  
**Status:** OPEN

### KAI-WEATHER-010 — MEDIUM — HTTP connection pools are recreated
**Issue:** Every refresh creates a new `httpx.AsyncClient`.  
**Risk:** Periodic polling repeatedly creates DNS/TCP/TLS state and connection pools.  
**Recommendation:** Reuse one lifecycle-managed egress client with bounded pools.  
**Status:** OPEN

### KAI-WEATHER-011 — MEDIUM — Internal diagnostics are public
**Issue:** Raw exception strings are retained in `_fetch_error` and returned by `/health`.  
**Risk:** Callers receive DNS, TLS, routing, provider and parser diagnostics.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-WEATHER-012 — MEDIUM — Error-budget telemetry is inert
**Issue:** `budget` is exposed through `/metrics`, but no request or refresh path records outcomes.  
**Risk:** Reliability metrics provide no evidence of provider availability or stale-data frequency.  
**Recommendation:** Record endpoint and fetch outcomes, latency and freshness violations.  
**Status:** OPEN

### KAI-WEATHER-013 — MEDIUM — Configuration lacks validation
**Issue:** Latitude, longitude, refresh interval and port are parsed directly. Non-finite/out-of-range coordinates and zero/negative/extreme intervals are not rejected.  
**Risk:** Misconfiguration causes invalid provider requests, tight loops, misleading location output or startup failure.  
**Recommendation:** Validate finite geographic ranges and safe numeric intervals at startup.  
**Status:** OPEN

### KAI-WEATHER-014 — MEDIUM — Text fields are unbounded
**Issue:** `LOCATION_NAME`, provider dates and other upstream string values have no length limit before being returned or inserted into the summary.  
**Risk:** Configuration or malformed provider data can inflate responses and agent context or inject misleading natural-language content.  
**Recommendation:** Apply strict field lengths and structured encoding before publication.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **0**
- High: **4**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **836**
- Critical: **87**
- High: **297**
- Medium: **449**
- Low: **3**

## Files materially reviewed in this batch

`weather-service/app.py` and the relevant `weather-service` deployment definition in `docker-compose.minimal.yml`.
