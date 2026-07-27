# Kai Code Audit — Air Quality Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AQ-001 | HIGH | “Current” readings select the final non-null hourly forecast value rather than the current hour |
| KAI-AQ-002 | HIGH | Raw PM2.5 concentration bands are labelled as an AQI category without calculating an AQI or averaging standard |
| KAI-AQ-003 | HIGH | Failed refreshes preserve and serve stale air-quality data as live health context |
| KAI-AQ-004 | HIGH | Upstream response bytes and JSON complexity are unbounded |
| KAI-AQ-005 | HIGH | Unverified environmental values are promoted into agent health/wellbeing context |
| KAI-AQ-006 | MEDIUM | Hourly timestamps are ignored, preventing time alignment and freshness verification |
| KAI-AQ-007 | MEDIUM | UV classification has no `extreme` category and labels every value at or above 8 as `very high` |
| KAI-AQ-008 | MEDIUM | Open-Meteo response structure, units and numeric values are not schema-validated |
| KAI-AQ-009 | MEDIUM | Health reports `ok` before first fetch and after fetch failure |
| KAI-AQ-010 | MEDIUM | Cache, error and task state are process-local and unsynchronised |
| KAI-AQ-011 | MEDIUM | Refresh-task cancellation is not awaited during shutdown |
| KAI-AQ-012 | MEDIUM | A new HTTP client and connection pool are created for every refresh |
| KAI-AQ-013 | MEDIUM | Raw upstream/network errors are exposed through health |
| KAI-AQ-014 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-AQ-015 | MEDIUM | Coordinates, intervals, port and location text are not safely validated or bounded |

---

## Air quality service: `airquality-service/app.py`

### KAI-AQ-001 — HIGH — Future/end-of-day values are reported as current
**Issue:** `_fetch_aq` requests a full day of hourly values. `_latest` iterates each array in reverse and returns the final non-null value, without reading the associated hourly `time` array or comparing it with the current time. `/current` then presents those values as current readings.  
**Risk:** The service commonly reports the last forecast hour of the day rather than present PM2.5, PM10, ozone, NO₂ and UV conditions. Agentic recommendations can therefore be based on future values mislabelled as current.  
**Recommendation:** Parse timezone-aware timestamps, select the nearest completed/current observation and return its exact sample time and forecast/observation status.  
**Status:** OPEN

### KAI-AQ-002 — HIGH — Concentration thresholds are misrepresented as AQI
**Issue:** `_aqi_category` directly compares one PM2.5 concentration value with hard-coded bands and returns `good` through `hazardous`. It does not calculate an air-quality index, identify a jurisdiction/standard, apply an averaging period or incorporate other pollutants. The output field is named `aqi_category`.  
**Risk:** Users and downstream agents may interpret the label as an official/current AQI health category when it is only an undocumented concentration-band heuristic.  
**Recommendation:** Name the result accurately, specify the standard and averaging period, or implement a validated jurisdiction-specific AQI calculation with current guidance/versioning.  
**Status:** OPEN

### KAI-AQ-003 — HIGH — Stale environmental health data remains active
**Issue:** On fetch failure, `_fetch_error` changes but `_cache` and `_last_fetch` remain. `/current` and `/summary` continue serving the prior values; summary contains no fetched time, stale marker or active error.  
**Risk:** Users and autonomous workflows can receive obsolete pollution and UV guidance as current during prolonged upstream failure.  
**Recommendation:** Enforce a strict freshness TTL, mark stale output degraded and block health/wellbeing recommendations when current data is unavailable.  
**Status:** OPEN

### KAI-AQ-004 — HIGH — Upstream payload allocation is unbounded
**Issue:** `_fetch_aq` materialises the complete HTTP response and calls `resp.json()` without response-byte, decompressed-size, nesting or field-count limits.  
**Risk:** A compromised/malformed provider response can consume excessive memory and CPU before hourly values are processed.  
**Recommendation:** Stream with a strict byte cap and validate a bounded endpoint-specific schema.  
**Status:** OPEN

### KAI-AQ-005 — HIGH — Weak data becomes health/wellbeing authority
**Issue:** `/summary` is explicitly designed for agentic context injection. It converts the flawed “latest” values and heuristic category into natural-language health context; Cortex/House Doctor can use air-quality text to recommend actions such as changing ventilation. No provenance, confidence, current-hour evidence or standard is included.  
**Risk:** Incorrect, stale or malformed values can generate personal/environmental advice with an authority level unsupported by the data.  
**Recommendation:** Use typed signed observations with sample time, standard, source, confidence and expiry; require corroboration before health-related recommendations.  
**Status:** OPEN

### KAI-AQ-006 — MEDIUM — Hourly time axis is discarded
**Issue:** The response requests hourly values but never reads the hourly `time` array. Each pollutant is selected independently by its last non-null position.  
**Risk:** Pollutants can be drawn from different hours, the service cannot prove temporal coherence and `fetched_at` is incorrectly used as a substitute for measurement time.  
**Recommendation:** Align all fields to one validated timestamp and return both measurement and retrieval times.  
**Status:** OPEN

### KAI-AQ-007 — MEDIUM — UV categories are incomplete
**Issue:** Summary labels UV as low below 3, moderate below 6, high below 8 and `very high` for every value at or above 8. No separate extreme state exists for higher values.  
**Risk:** Very high and extreme UV conditions are collapsed, reducing the severity signal supplied to the user/agent.  
**Recommendation:** Use a versioned recognised UV classification and include explicit boundary/source metadata.  
**Status:** OPEN

### KAI-AQ-008 — MEDIUM — Upstream schema and units are trusted implicitly
**Issue:** The code assumes dictionary/list/numeric structures and rounds values without validating types, finite ranges, units, array lengths or provider errors.  
**Risk:** Malformed/non-finite data can raise unstructured errors or be represented as plausible environmental readings.  
**Recommendation:** Validate a strict typed schema with finite physical ranges and unit checks.  
**Status:** OPEN

### KAI-AQ-009 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including before any successful fetch and while `_fetch_error` is populated.  
**Risk:** Monitoring treats an empty or failed environmental source as ready.  
**Recommendation:** Separate liveness, provider connectivity and fresh-snapshot readiness.  
**Status:** OPEN

### KAI-AQ-010 — MEDIUM — State is volatile and worker-local
**Issue:** Cache, fetch time, error and task reference are module-level process memory.  
**Risk:** Multiple workers poll independently and expose different values; restart erases freshness/error history.  
**Recommendation:** Use one fetch authority and shared immutable timestamped snapshots.  
**Status:** OPEN

### KAI-AQ-011 — MEDIUM — Shutdown does not await refresh termination
**Issue:** Lifespan shutdown cancels `_refresh_task` without awaiting it or explicitly closing an in-flight client.  
**Risk:** Refresh work can continue or be abandoned during shutdown and task errors/resources are not observed.  
**Recommendation:** Await cancellation and lifecycle cleanup.  
**Status:** OPEN

### KAI-AQ-012 — MEDIUM — HTTP pools are recreated
**Issue:** Every refresh creates a new `httpx.AsyncClient`.  
**Risk:** Periodic polling repeatedly creates DNS/TCP/TLS state and connection pools.  
**Recommendation:** Reuse one lifecycle-managed egress client.  
**Status:** OPEN

### KAI-AQ-013 — MEDIUM — Internal diagnostics are public
**Issue:** Raw exception strings are retained in `_fetch_error` and returned by `/health`.  
**Risk:** Callers receive DNS, TLS, routing, provider and parser diagnostics.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-AQ-014 — MEDIUM — Error-budget telemetry is inert
**Issue:** `budget` is exposed through `/metrics`, but no endpoint or refresh result is recorded.  
**Risk:** Reliability metrics provide no evidence of provider availability or stale-data frequency.  
**Recommendation:** Record fetch/request outcomes, latency and freshness violations.  
**Status:** OPEN

### KAI-AQ-015 — MEDIUM — Configuration and context strings lack validation
**Issue:** Latitude, longitude, refresh interval and port are parsed directly; coordinates are not checked for finite geographic ranges. `LOCATION_NAME` has no length/control-character bound and is inserted into summary text.  
**Risk:** Misconfiguration causes invalid requests, tight loops, startup failure or misleading/oversized agent context.  
**Recommendation:** Validate typed geographic/numeric configuration and strictly bound structured location labels.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **0**
- High: **5**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **851**
- Critical: **87**
- High: **302**
- Medium: **459**
- Low: **3**

## Files materially reviewed in this batch

`airquality-service/app.py` and the relevant `airquality-service` deployment definition in `docker-compose.minimal.yml`.
