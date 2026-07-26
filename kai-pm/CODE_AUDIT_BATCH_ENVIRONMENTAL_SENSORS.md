# Kai Code Audit — Environmental Sensor Services Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SENSOR-001 | HIGH | Weather data is served indefinitely after refresh failure without stale-state enforcement |
| KAI-SENSOR-002 | HIGH | Air-quality “current” readings select the last forecast-hour value rather than the current hour |
| KAI-SENSOR-003 | HIGH | Air-quality data is served indefinitely after refresh failure without stale-state enforcement |
| KAI-SENSOR-004 | MEDIUM | Both health endpoints report `ok` despite failed or never-successful upstream refresh |
| KAI-SENSOR-005 | MEDIUM | Environmental refresh workers are not supervised for unexpected termination |
| KAI-SENSOR-006 | MEDIUM | Raw upstream exception details are exposed through health responses |
| KAI-SENSOR-007 | MEDIUM | Refresh intervals and coordinates lack startup validation and safety bounds |
| KAI-SENSOR-008 | MEDIUM | Air-quality category thresholds are unlabeled and may not match the deployment’s UK health standard |
| KAI-SENSOR-009 | MEDIUM | Metrics/error-budget objects are instantiated but refresh outcomes are never recorded |
| KAI-SENSOR-010 | LOW | Weather endpoint contract advertises apparent temperature but does not return it |

---

## Weather service: `weather-service/app.py`

### KAI-SENSOR-001 — HIGH — Stale weather is served indefinitely
**Issue:** On refresh failure, `_fetch_error` is updated but the previous `_cache` and `_last_fetch` remain active. `/current`, `/forecast` and `/summary` continue returning the old data with no maximum-age rejection or explicit stale flag.  
**Risk:** Agentic planning can treat hours- or days-old weather as current during a prolonged upstream outage, potentially affecting travel, site-work or safety decisions.  
**Recommendation:** Enforce a maximum data age, expose freshness explicitly and return a degraded/unavailable state once the age threshold is exceeded.  
**Status:** OPEN

### KAI-SENSOR-004 — MEDIUM — Health reports success during upstream failure
**Issue:** `/health` always returns `status: ok`, including before the first successful fetch and while `fetch_error` is populated.  
**Risk:** Watchdogs and context gatherers treat a non-functional or stale service as healthy.  
**Recommendation:** Separate process liveness from data readiness and freshness.  
**Status:** OPEN

### KAI-SENSOR-005 — MEDIUM — Refresh worker is not supervised
**Issue:** The lifespan retains the task for cancellation but does not check whether `_refresh_loop` exits unexpectedly, restart it or reflect task death in readiness.  
**Risk:** A non-caught termination can permanently stop refresh while endpoints continue serving cached data.  
**Recommendation:** Supervise the task and fail readiness or restart with bounded backoff.  
**Status:** OPEN

### KAI-SENSOR-006 — MEDIUM — Upstream errors are exposed
**Issue:** `_fetch_error = str(exc)` is returned directly from `/health`.  
**Risk:** Callers can learn networking, TLS, proxy and upstream implementation details.  
**Recommendation:** Expose a stable error code and retain detailed exceptions only in protected logs.  
**Status:** OPEN

### KAI-SENSOR-007 — MEDIUM — Configuration is not validated
**Issue:** Latitude, longitude and refresh interval are converted directly from environment values. There are no geographic bounds or positive minimum/maximum interval checks.  
**Risk:** Invalid coordinates crash startup; zero/negative intervals can cause tight refresh loops; extreme intervals can leave data effectively unrefreshed.  
**Recommendation:** Validate all configuration against explicit ranges at startup.  
**Status:** OPEN

### KAI-SENSOR-009 — MEDIUM — Error-budget telemetry is inert
**Issue:** An `ErrorBudget` instance is exposed through `/metrics`, but fetch successes and failures never call `budget.record`.  
**Risk:** Operational metrics can appear empty or healthy while upstream refresh is failing.  
**Recommendation:** Record every refresh and endpoint outcome with classified failure reasons.  
**Status:** OPEN

### KAI-SENSOR-010 — LOW — Documented field is missing
**Issue:** The service requests hourly `apparent_temperature`, and its endpoint documentation advertises `feels_like_c`, but `/current` does not return that field.  
**Risk:** Consumers relying on the published contract receive incomplete data.  
**Recommendation:** Return the current apparent temperature or correct the contract.  
**Status:** OPEN

---

## Air-quality service: `airquality-service/app.py`

### KAI-SENSOR-002 — HIGH — “Current” values use the final forecast hour
**Issue:** `_latest` iterates the hourly forecast array in reverse and returns the last non-null value. With a one-day hourly forecast, this normally selects the furthest future hour rather than the present hour.  
**Risk:** `/current` and `/summary` can label a future PM2.5, PM10, ozone, NO₂ or UV forecast as the current environmental condition.  
**Recommendation:** Match the hourly timestamp to current local time or request a dedicated current-value API field.  
**Status:** OPEN

### KAI-SENSOR-003 — HIGH — Stale air-quality data is served indefinitely
**Issue:** Refresh failures preserve the previous cache, and read endpoints impose no maximum age or stale-state flag.  
**Risk:** Health-related context can rely on obsolete pollution and UV values during a long outage.  
**Recommendation:** Apply freshness thresholds and fail or clearly degrade stale outputs.  
**Status:** OPEN

### KAI-SENSOR-008 — MEDIUM — AQ classification standard is not identified
**Issue:** `_aqi_category` applies PM2.5 breakpoint values associated with a particular AQI convention but does not identify the standard, averaging period or jurisdiction. The deployment defaults to London.  
**Risk:** UK users may interpret categories as UK DAQI guidance when the thresholds and terminology are from another framework.  
**Recommendation:** Name the standard, use the correct averaging period and select a jurisdiction-appropriate classification.  
**Status:** OPEN

The shared findings KAI-SENSOR-004 through KAI-SENSOR-007 and KAI-SENSOR-009 also apply materially to this service.

---

## Batch totals

- Findings: **10**
- Critical: **0**
- High: **3**
- Medium: **6**
- Low: **1**

## Provisional repository totals after all logged batches

- Findings: **234**
- Critical: **28**
- High: **104**
- Medium: **100**
- Low: **2**

## Files materially reviewed in this batch

`weather-service/app.py`, `airquality-service/app.py`.
