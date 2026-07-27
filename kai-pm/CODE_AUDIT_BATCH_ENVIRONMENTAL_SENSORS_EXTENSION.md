# Kai Code Audit — Environmental Sensor Services Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_ENVIRONMENTAL_SENSORS.md`. The existing 10 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SENSORX-001 | HIGH | Weather missing fields default to clear sky, zero wind and daytime |
| KAI-SENSORX-002 | HIGH | Weather current/summary responses do not preserve the upstream observation timestamp or timezone |
| KAI-SENSORX-003 | HIGH | Weather summary can report `None°C` or malformed numeric data as natural-language context |
| KAI-SENSORX-004 | HIGH | Weather output omits units metadata and assumes upstream defaults |
| KAI-SENSORX-005 | HIGH | Returned upstream coordinates/timezone are not checked against the configured location |
| KAI-SENSORX-006 | HIGH | Daily rain probability index zero is treated as “today” without validating its date |
| KAI-SENSORX-007 | HIGH | Air-quality pollutants can be selected from different forecast hours, producing a non-existent composite condition |
| KAI-SENSORX-008 | HIGH | Air-quality responses omit the actual hourly timestamp used for every pollutant |
| KAI-SENSORX-009 | HIGH | AQ category uses PM2.5 only and ignores hazardous PM10, ozone and nitrogen dioxide values |
| KAI-SENSORX-010 | HIGH | Negative and non-finite PM2.5 values can be categorised as good or break JSON/logic |
| KAI-SENSORX-011 | HIGH | Air-quality units and averaging periods are assumed rather than validated or returned |
| KAI-SENSORX-012 | HIGH | UV classification has no extreme category and treats every value at least eight as merely very high |
| KAI-SENSORX-013 | HIGH | Exact configured location names and coordinates are publicly disclosed |
| KAI-SENSORX-014 | HIGH | Environment-controlled location text is inserted into Agentic/Cortex context without a safe provenance boundary |
| KAI-SENSORX-015 | HIGH | Complete unvalidated upstream JSON is cached and used as an authoritative state source |
| KAI-SENSORX-016 | HIGH | Environmental summaries carry no source URL, provider revision, observation time, freshness or confidence metadata |
| KAI-SENSORX-017 | HIGH | Multiple workers independently poll the same upstreams and expose divergent caches |
| KAI-SENSORX-018 | HIGH | Read endpoints have no authenticated user/location partition and expose one global configured context |
| KAI-SENSORX-019 | MEDIUM | Weather code conversion can raise on null or malformed current values |
| KAI-SENSORX-020 | MEDIUM | Weather wind conversion can raise when `windspeed` is null or non-numeric |
| KAI-SENSORX-021 | MEDIUM | Forecast array lengths and types are independently trusted and can produce partial mismatched rows |
| KAI-SENSORX-022 | MEDIUM | Forecast response length has no explicit application maximum beyond trust in the upstream parameter |
| KAI-SENSORX-023 | MEDIUM | Current weather does not indicate whether values are observations, interpolation or model output |
| KAI-SENSORX-024 | MEDIUM | Loading summaries return HTTP 200 and can be injected as normal context rather than unavailable data |
| KAI-SENSORX-025 | MEDIUM | Upstream response bytes and JSON complexity are not bounded or schema-validated |
| KAI-SENSORX-026 | MEDIUM | A new HTTP client and connection pool is created for every refresh |
| KAI-SENSORX-027 | MEDIUM | Refresh failures use the normal interval with no jitter or transient/permanent backoff |
| KAI-SENSORX-028 | MEDIUM | Cached environmental state has no immutable generation, digest or compare-and-swap publication contract |
| KAI-SENSORX-029 | MEDIUM | Caches and timestamps are process-local and disappear on restart |
| KAI-SENSORX-030 | MEDIUM | Public metrics expose telemetry without administrative authentication |
| KAI-SENSORX-031 | MEDIUM | Missing shared-runtime imports silently replace telemetry with no-op fallbacks |
| KAI-SENSORX-032 | MEDIUM | Fetch and uptime timestamps use wall-clock floats without a trusted source sequence |
| KAI-SENSORX-033 | MEDIUM | Shutdown cancels refresh tasks without awaiting completion or active HTTP cleanup |
| KAI-SENSORX-034 | MEDIUM | No immutable audit links provider response, location configuration, published summary and downstream context use |

---

## High-severity findings

### KAI-SENSORX-001 — HIGH — Missing data becomes safe weather
**Issue:** `weathercode` defaults to zero, wind speed defaults to zero and `is_day` defaults to true. WMO code zero is rendered as clear sky.  
**Risk:** Partial/malformed upstream responses become reassuring current conditions rather than unavailable data.  
**Recommendation:** require a validated complete current observation and represent every missing field explicitly.  
**Status:** OPEN

### KAI-SENSORX-002 — HIGH — Current time provenance is discarded
The upstream `current_weather.time`, UTC offset and timezone are not returned; only local fetch time is exposed.

### KAI-SENSORX-003 — HIGH — Invalid natural-language weather
A missing/non-numeric temperature can be formatted directly into the context sentence.

### KAI-SENSORX-004 — HIGH — Unit assumptions
The service labels values °C, km/h, millimetres and percentages but does not validate/return upstream unit metadata.

### KAI-SENSORX-005 — HIGH — Location response not verified
The provider’s returned latitude, longitude, elevation and timezone are ignored, so a proxy/cache/wrong response cannot be detected against configuration.

### KAI-SENSORX-006 — HIGH — “Today” is positional
The first precipitation-probability value is used without checking the corresponding daily date against the configured local current date.

### KAI-SENSORX-007 — HIGH — Impossible mixed-hour AQ snapshot
`_latest()` independently finds the last non-null value for each pollutant. Missing values can make PM2.5, PM10, O3, NO2 and UV come from different hours.

### KAI-SENSORX-008 — HIGH — AQ timestamp removed
No hourly time/index accompanies any current value.

### KAI-SENSORX-009 — HIGH — Single-pollutant health classification
The overall category is based only on PM2.5; high ozone/NO2/PM10 cannot worsen it.

### KAI-SENSORX-010 — HIGH — Invalid PM2.5 categorisation
Negative values satisfy `<=12` and become good. NaN/infinity are not explicitly rejected and can yield misleading or non-standard JSON.

### KAI-SENSORX-011 — HIGH — AQ units/averaging omitted
Values are labelled µg/m³ without validating provider unit fields, and no averaging period/source standard is attached.

### KAI-SENSORX-012 — HIGH — Incomplete UV risk categories
Every UV value of 8 or above is “very high”; extreme risk is never represented.

### KAI-SENSORX-013 — HIGH — Public location disclosure
Weather health/current/forecast and AQ health/current/summary reveal the configured location; Weather health also returns precise latitude/longitude.

### KAI-SENSORX-014 — HIGH — Configuration text becomes privileged context
`LOCATION_NAME` is inserted into summary strings that Agentic/Cortex consume, without validation, quoting or configuration provenance.

### KAI-SENSORX-015 — HIGH — Upstream JSON is authority without schema
The services cache complete arbitrary JSON dictionaries and later make typed assumptions at read time.

### KAI-SENSORX-016 — HIGH — Evidence metadata absent
Summaries do not identify Open-Meteo, endpoint revision, source time, age, forecast/observation status or uncertainty.

### KAI-SENSORX-017 — HIGH — Replica divergence and upstream amplification
Every worker launches its own refresh task and maintains its own cache/error state.

### KAI-SENSORX-018 — HIGH — Global one-location context
Any caller receives the same location-specific data; there is no authenticated user/site/location access model.

---

## Medium-severity findings

### KAI-SENSORX-019 — MEDIUM — Null weather-code crash
`int(cw.get("weathercode", 0))` fails when the key exists with null or malformed content.

### KAI-SENSORX-020 — MEDIUM — Null wind crash
Multiplication/rounding fails for null or non-numeric wind speed.

### KAI-SENSORX-021 — MEDIUM — Independent array trust
Each daily array is indexed independently; mismatched lengths/types silently create partial rows or exceptions.

### KAI-SENSORX-022 — MEDIUM — No response cardinality bound
The application iterates every returned daily time entry rather than enforcing seven validated rows.

### KAI-SENSORX-023 — MEDIUM — Observation/model ambiguity
The contract calls data current conditions but does not state whether it is measured, interpolated or modelled.

### KAI-SENSORX-024 — MEDIUM — Loading is success-shaped
Summary endpoints return normal HTTP-200 text while data is absent.

### KAI-SENSORX-025 — MEDIUM — Unbounded upstream parse
Complete bodies are decoded and cached without byte, depth or array limits.

### KAI-SENSORX-026 — MEDIUM — Connection churn
Every interval creates and closes an `AsyncClient`.

### KAI-SENSORX-027 — MEDIUM — No failure backoff/jitter
Repeated provider/network failures continue on the fixed normal cadence and replicas synchronise requests.

### KAI-SENSORX-028 — MEDIUM — No snapshot identity
Published cache updates contain no generation/digest and readers cannot request a consistent revision.

### KAI-SENSORX-029 — MEDIUM — Volatile environmental history
Restart loses last successful data/time/error and creates an evidence gap.

### KAI-SENSORX-030 — MEDIUM — Public telemetry
Metrics endpoints are unauthenticated.

### KAI-SENSORX-031 — MEDIUM — Silent runtime downgrade
If common runtime imports fail, logging/metrics fall back without readiness impact.

### KAI-SENSORX-032 — MEDIUM — Weak chronology
Fetch/uptime use `time.time()` and no provider event sequence.

### KAI-SENSORX-033 — MEDIUM — Incomplete shutdown
Tasks are cancelled but not awaited; clients/updates may be interrupted without a final state.

### KAI-SENSORX-034 — MEDIUM — Missing context audit
No record binds exact provider payload/revision and configured location to the context string consumed downstream.

---

## Batch totals

- Findings: **34**
- Critical: **0**
- High: **18**
- Medium: **16**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,281**
- Critical: **189**
- High: **1,140**
- Medium: **949**
- Low: **3**

## Files materially reviewed

`weather-service/app.py`, `airquality-service/app.py`, the existing environmental-sensor audit, deployment configuration and Agentic/Cortex/Dashboard integrations.
