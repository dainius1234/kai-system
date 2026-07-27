# Kai Code Audit — Metrics Gateway Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_METRICS_GATEWAY.md`. The existing 18 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-METRICSX-001 | HIGH | A partially valid custom registry silently replaces the complete default fleet |
| KAI-METRICSX-002 | HIGH | The gateway has no per-service credential model and cannot scrape services once their metrics endpoints are secured |
| KAI-METRICSX-003 | HIGH | Service identity and metrics traffic use unauthenticated plaintext HTTP |
| KAI-METRICSX-004 | HIGH | A complete fleet outage still returns scrape `status: ok` |
| KAI-METRICSX-005 | HIGH | A successful empty metrics response cannot clear previously cached metrics |
| KAI-METRICSX-006 | HIGH | Anonymous manual scrapes can replace the entire rolling uptime window with attacker-timed observations |
| KAI-METRICSX-007 | HIGH | Custom registry cardinality is unbounded and creates one concurrent task per service |
| KAI-METRICSX-008 | HIGH | Metrics Gateway’s regular Heartbeat probes perpetually reset Heartbeat’s activity timer and prevent auto-sleep |
| KAI-METRICSX-009 | HIGH | Missing or stale service error-rate data is exported as zero |
| KAI-METRICSX-010 | HIGH | No rate limit, caller quota or global scrape admission policy protects manual scraping |
| KAI-METRICSX-011 | MEDIUM | Health and metrics are fetched sequentially for each service |
| KAI-METRICSX-012 | MEDIUM | Scrapes have no retry, circuit-breaker or service-specific backoff policy |
| KAI-METRICSX-013 | MEDIUM | Dashboard contributes no metrics because its metrics path is configured equal to its health path |
| KAI-METRICSX-014 | MEDIUM | The Prometheus endpoint exports only synthesised up/uptime/error-rate series rather than aggregating downstream metrics |
| KAI-METRICSX-015 | MEDIUM | One global last-scrape timestamp is reported for every service regardless of its individual outcome |
| KAI-METRICSX-016 | MEDIUM | Per-service latency, last-success time and timeout reason are not retained |
| KAI-METRICSX-017 | MEDIUM | Scrape timestamps mix naive UTC and host-local time representations |
| KAI-METRICSX-018 | MEDIUM | Manual scrape and telemetry reads have no immutable actor/job audit trail |

---

### KAI-METRICSX-001 — HIGH — Partial custom registry removes default coverage
**Issue:** If `METRICS_SERVICES` contains at least one syntactically valid `name=url` pair, `_build_registry()` returns only those entries.  
**Risk:** Adding one experimental service can silently stop monitoring Tool Gate, memU, Executor and the rest of the default fleet.  
**Recommendation:** require a complete signed manifest or merge explicitly with unique defaults under a validated override policy.  
**Status:** OPEN

### KAI-METRICSX-002 — HIGH — No authenticated downstream scrape support
**Issue:** Health and metrics requests send no bearer token, HMAC, mTLS identity or per-service credentials.  
**Risk:** Remediating downstream public metrics endpoints will cause the Gateway to mark them unreachable; the architecture depends on services remaining unauthenticated.  
**Recommendation:** provision least-privilege per-service scraper identities and validate endpoint-specific scopes.  
**Status:** OPEN

### KAI-METRICSX-003 — HIGH — Unauthenticated plaintext service traffic
All default URLs are `http://`; the Gateway neither authenticates the responder nor verifies a response signature/version.

### KAI-METRICSX-004 — HIGH — Total outage reports success
`scrape_all()` unconditionally returns `status: ok` after gathering, even when `reachable == 0`.

### KAI-METRICSX-005 — HIGH — Empty response preserves stale state
`_latest_metrics[name]` updates only when `metrics_data` is truthy. A valid `{}` response leaves the older metrics active.

### KAI-METRICSX-006 — HIGH — Caller-controlled uptime sample window
Each public `/scrape` appends one observation and the list keeps only 100. Repeated calls can evict the scheduled history in seconds.

### KAI-METRICSX-007 — HIGH — Unbounded custom fan-out
The custom registry has no entry cap; `scrape_all()` creates and gathers one task per entry simultaneously.

### KAI-METRICSX-008 — HIGH — Cross-service sleep suppression
The background scraper repeatedly calls Heartbeat `/health`. Heartbeat middleware treats every request as activity, so the default 30-second Gateway cadence prevents Heartbeat’s 1,800-second idle threshold from ever becoming true.

### KAI-METRICSX-009 — HIGH — Unknown error rate becomes healthy zero
Prometheus output uses `m.get("error_rate", 0.0)`. Missing, failed or stale metrics therefore emit zero error rate rather than unknown/stale.

### KAI-METRICSX-010 — HIGH — No manual scrape admission control
There is no lock, queue, rate limit or caller quota beyond the already-logged overlap race.

### KAI-METRICSX-011 — MEDIUM — Per-service requests are serial
Within each service task, health completes before metrics begins, doubling timeout exposure and delaying snapshot completion.

### KAI-METRICSX-012 — MEDIUM — No service-specific resilience
One failed request immediately records reachability failure; repeated failures continue at full interval with no retry classification or breaker.

### KAI-METRICSX-013 — MEDIUM — Dashboard metrics are never collected
The Dashboard registry entry uses `/health` for both paths. `_scrape_one()` skips metrics when both paths match.

### KAI-METRICSX-014 — MEDIUM — Misleading aggregation contract
`/metrics/text` does not merge downstream Prometheus metrics; it synthesises only three Gateway-owned series.

### KAI-METRICSX-015 — MEDIUM — Global timestamp misattribution
Every fleet row receives `_last_scrape`, even if its own task failed, was skipped or retained older cached data.

### KAI-METRICSX-016 — MEDIUM — Missing per-service diagnostics
The cache retains no request duration, last successful sample, response status or timeout class.

### KAI-METRICSX-017 — MEDIUM — Timestamp representation drift
`scrape_all()` returns naive `datetime.utcnow()`, while `/metrics` and `/fleet` use host-local `datetime.fromtimestamp()`.

### KAI-METRICSX-018 — MEDIUM — No scrape audit identity
No durable event ties the initiating actor, registry revision, service set, sample generation and publication result.

---

## Batch totals

- Findings: **18**
- Critical: **0**
- High: **10**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,071**
- Critical: **184**
- High: **1,021**
- Medium: **863**
- Low: **3**

## Files materially reviewed

`metrics-gateway/app.py`, the existing Metrics Gateway audit, fleet deployment definitions and cross-service Heartbeat behaviour.
