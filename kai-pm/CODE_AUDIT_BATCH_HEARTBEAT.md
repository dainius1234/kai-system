# Kai Code Audit — Heartbeat Monitor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HB-001 | CRITICAL | The host-published heartbeat, recovery and alert authority has no authentication or authorisation |
| KAI-HB-002 | CRITICAL | Anonymous `/tick` and `/recover` calls can indefinitely erase stale-heartbeat evidence |
| KAI-HB-003 | HIGH | Every incoming request is treated as operator activity, including health probes and hostile traffic |
| KAI-HB-004 | HIGH | Docker and Supervisor health polling permanently prevents the configured auto-sleep condition |
| KAI-HB-005 | HIGH | Auto-sleep has no background scheduler and is evaluated only inside requests after activity was refreshed |
| KAI-HB-006 | HIGH | `CHECK_INTERVAL` is reported/configured but never drives a heartbeat loop |
| KAI-HB-007 | HIGH | Heartbeat freshness measures external `/tick` calls rather than an internal liveness task |
| KAI-HB-008 | HIGH | Read-only `/status` performs notifications, log scanning and attempted maintenance side effects |
| KAI-HB-009 | HIGH | Anonymous callers can send arbitrary notification events through `/event` |
| KAI-HB-010 | HIGH | Event status/reason fields are unbounded and enter logs and external notification text |
| KAI-HB-011 | HIGH | Notification delivery uses synchronous HTTP inside async request paths |
| KAI-HB-012 | HIGH | Notification HTTP status is ignored and delivery is not acknowledged |
| KAI-HB-013 | HIGH | Alerting is silently disabled when `NOTIFY_URL` is unset |
| KAI-HB-014 | HIGH | Every status request reads the complete Executor log synchronously |
| KAI-HB-015 | HIGH | Historical keyword hits generate repeated alerts forever because no read offset or event window exists |
| KAI-HB-016 | HIGH | Executor-log monitoring has no rotation, inode, size or truncation handling |
| KAI-HB-017 | HIGH | The deployed Heartbeat container does not mount the configured Executor log path |
| KAI-HB-018 | HIGH | Intrusion detection is a weak count of three generic substrings |
| KAI-HB-019 | HIGH | CPU pressure uses host load divided by visible CPU count rather than cgroup quota/usage |
| KAI-HB-020 | HIGH | Missing or unreadable CPU sensors are represented as a safe 0°C |
| KAI-HB-021 | HIGH | Missing, failed or malformed GPU telemetry is represented as a safe 0°C |
| KAI-HB-022 | HIGH | `nvidia-smi` executes synchronously in async health, status and self-assessment requests |
| KAI-HB-023 | HIGH | Resource-pressure alerts claim auto-sleep is being triggered although the activity guard immediately suppresses it |
| KAI-HB-024 | HIGH | Memory compression, focus compression and decay HTTP statuses are never checked |
| KAI-HB-025 | HIGH | 4xx/5xx maintenance responses can be logged as successful sleep and start the cooldown |
| KAI-HB-026 | HIGH | Compression, focus compression and decay are separate non-transactional mutations |
| KAI-HB-027 | HIGH | Sleep cooldown and activity state are process-local and inconsistent across workers/restarts |
| KAI-HB-028 | HIGH | Self-assessment window and offset arguments are ignored by the memU data fetches |
| KAI-HB-029 | HIGH | “Previous week” is actually the previous API call, regardless of when it occurred |
| KAI-HB-030 | HIGH | GET `/self-assessment` mutates the future comparison baseline |
| KAI-HB-031 | HIGH | Anonymous callers can manipulate trend baselines by repeatedly invoking self-assessment |
| KAI-HB-032 | HIGH | Diagnostics are fetched during self-assessment and then completely discarded |
| KAI-HB-033 | HIGH | Self-assessment error rate measures Heartbeat API HTTP responses, not system execution reliability |
| KAI-HB-034 | HIGH | “Uptime ratio” measures time since the last caller-supplied tick, not process or fleet uptime |
| KAI-HB-035 | HIGH | Growth in raw memory count is classified as improvement without quality or correctness evidence |
| KAI-HB-036 | HIGH | Instantaneous CPU/temperature samples are presented as temporal peaks/trends without historical measurements |
| KAI-HB-037 | HIGH | Assessment state is an unsigned, restart-volatile file in `/tmp` |
| KAI-HB-038 | HIGH | Corrupt assessment state silently resets the baseline without quarantine or degraded status |
| KAI-HB-039 | HIGH | Concurrent assessment reads and complete-file writes can lose or corrupt baseline state |
| KAI-HB-040 | HIGH | World/context information is exposed without authentication |
| KAI-HB-041 | MEDIUM | World-context cache is unsigned mutable local JSON |
| KAI-HB-042 | MEDIUM | Calendar-sync response fields can overwrite local date, time and fetch metadata |
| KAI-HB-043 | MEDIUM | Calendar-sync HTTP errors/non-200 responses still produce overall world status `ok` |
| KAI-HB-044 | MEDIUM | Cache-write failure still produces world status `ok` |
| KAI-HB-045 | MEDIUM | Failed world fetches set the daily cooldown and suppress retries |
| KAI-HB-046 | MEDIUM | Cached responses can claim `stale: false` even when no valid cache was returned |
| KAI-HB-047 | MEDIUM | World date/time and schedule semantics use the service host timezone |
| KAI-HB-048 | MEDIUM | Calendar response bytes, JSON depth and field sizes are unbounded |
| KAI-HB-049 | MEDIUM | World-fetch cooldown/cache state is worker-local and causes duplicate or divergent fetches |
| KAI-HB-050 | MEDIUM | Every notification, world, stats and diagnostics operation creates a new HTTP client/pool |
| KAI-HB-051 | MEDIUM | Health ignores memU, notification, world-cache and Executor-log readiness |
| KAI-HB-052 | MEDIUM | Health omits CPU/GPU temperature checks even though status/watchdog use them |
| KAI-HB-053 | MEDIUM | Intervals, windows, cooldowns and assessment periods are not range-validated |
| KAI-HB-054 | MEDIUM | Resource thresholds accept unsafe, non-finite or internally inconsistent values |
| KAI-HB-055 | MEDIUM | Public metrics expose Heartbeat request-error behaviour without authentication |
| KAI-HB-056 | MEDIUM | ErrorBudget measures served HTTP statuses rather than heartbeat freshness or alert delivery |
| KAI-HB-057 | MEDIUM | Tick, activity, sleep and world-fetch state is volatile and worker-local |
| KAI-HB-058 | MEDIUM | Log/cache/assessment filesystem operations run synchronously inside async handlers |
| KAI-HB-059 | MEDIUM | No rate limit, caller quota or workload-admission control protects status, alerts or self-assessment |
| KAI-HB-060 | MEDIUM | The service has no lifespan-owned clients, scheduler, shutdown drain or task supervision |
| KAI-HB-061 | MEDIUM | Downstream failures are reduced to warning strings with no durable incident state |
| KAI-HB-062 | MEDIUM | World-cache writes are non-atomic and can leave truncated JSON |
| KAI-HB-063 | MEDIUM | Assessment writes are non-atomic and do not fsync or retain a previous generation |
| KAI-HB-064 | MEDIUM | Alerts and maintenance actions lack immutable actor, incident, source offset and operation identifiers |

---

## Critical findings

### KAI-HB-001 — CRITICAL — Open heartbeat/recovery/alert authority
**Issue:** `docker-compose.full.yml` publishes `8010:8010`. `heartbeat/app.py` defines no authentication or authorisation while exposing heartbeat reset, recovery, alert relay, status and self-assessment endpoints.  
**Risk:** Any reachable caller can falsify health evidence, create external alerts and consume monitoring resources.  
**Recommendation:** remove host publication and require authenticated service heartbeats plus separately authorised recovery/alert operations.  
**Status:** OPEN — immediate remediation required

### KAI-HB-002 — CRITICAL — Stale evidence can be erased remotely
**Issue:** `/tick` and `/recover` directly assign `last_tick = time.time()` without proving the expected service executed, recovered or is healthy.  
**Risk:** An attacker or faulty service can keep Heartbeat and Supervisor green indefinitely while the real executor/control loop is dead.  
**Recommendation:** accept signed heartbeats bound to an authenticated service instance and never let recovery overwrite observation evidence without a verified postcondition.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-HB-003 — HIGH — Monitoring traffic counts as operator activity
Middleware updates `last_activity` before every endpoint, including `/health`, `/metrics`, `/status`, `/world`, hostile calls and automated probes.

### KAI-HB-004 — HIGH — Polling permanently defeats auto-sleep
Compose health checks and Supervisor call `/health` repeatedly, resetting `last_activity` more frequently than the default 1,800-second idle threshold.

### KAI-HB-005 — HIGH — Auto-sleep is operationally unreachable
There is no background loop. `_auto_sleep_check()` is called only from `/status` or `_watchdog_check()` after middleware has just refreshed activity, so it immediately returns.

### KAI-HB-006 — HIGH — Dead scheduler configuration
`CHECK_INTERVAL` is never used to schedule or sleep a task; it is only returned as status text.

### KAI-HB-007 — HIGH — External-input liveness model
`last_tick` changes only at process import, `/tick` and `/recover`; it does not measure an internally supervised heartbeat task.

### KAI-HB-008 — HIGH — GET status has side effects
A read triggers auto-sleep evaluation, resource-pressure notification, synchronous GPU command execution, complete log scan and possible intrusion alert.

### KAI-HB-009 — HIGH — Anonymous notification relay
Any caller can POST arbitrary event status/reason and cause a notification-gateway request.

### KAI-HB-010 — HIGH — Unbounded alert/log content
The request model has no string length/control-character limits; values are written to logs and notifications.

### KAI-HB-011 — HIGH — Blocking notification transport
`_send_notification()` uses synchronous `httpx.Client` in async request paths.

### KAI-HB-012 — HIGH — Delivery result ignored
Any HTTP response is treated as delivery success; only transport exceptions are logged.

### KAI-HB-013 — HIGH — Default alert silence
When no URL is configured, alerts disappear at debug level and callers still receive success.

### KAI-HB-014 — HIGH — Full-log read amplification
Every `/status` call reads the complete Executor log into memory.

### KAI-HB-015 — HIGH — Historical alert repetition
Counts cover the entire file on every scan. One old occurrence causes alerts on every later status request.

### KAI-HB-016 — HIGH — No log lifecycle handling
Rotation, truncation, maximum bytes, inode changes and a durable consumed offset are absent.

### KAI-HB-017 — HIGH — Deployed log source absent
The Heartbeat Docker/Compose service does not mount `/var/log/sovereign/executor.log`; the normal scan therefore returns zero regardless of Executor behaviour.

### KAI-HB-018 — HIGH — Weak intrusion detector
The detector counts `timeout`, `blocked` and `injection` anywhere in text, producing false positives while missing most attacks/failures.

### KAI-HB-019 — HIGH — Incorrect container CPU pressure
Host load average and visible CPU count do not represent container cgroup CPU quota or actual process consumption.

### KAI-HB-020 — HIGH — Missing CPU telemetry is safe zero
Absent/unreadable thermal sensors return 0°C and cannot degrade health.

### KAI-HB-021 — HIGH — Missing GPU telemetry is safe zero
No `nvidia-smi`, command failure or malformed output all return 0°C.

### KAI-HB-022 — HIGH — Blocking GPU probe
`subprocess.check_output()` runs synchronously during async requests.

### KAI-HB-023 — HIGH — False auto-sleep alert
Resource-pressure text states “Triggering auto-sleep”, but the immediately called guard sees fresh request activity and returns without maintenance.

### KAI-HB-024 — HIGH — Maintenance status ignored
None of the three memU POST responses calls `raise_for_status()` or validates a result schema.

### KAI-HB-025 — HIGH — Failure starts sleep cooldown
HTTP 4xx/5xx are not exceptions, so the service logs successful compression and sets `last_sleep_action`.

### KAI-HB-026 — HIGH — Partial destructive maintenance
Compress, focus-compress and decay are separate calls with no transaction or rollback; focus failure alone is explicitly suppressed.

### KAI-HB-027 — HIGH — Inconsistent sleep state
Activity/cooldown are ordinary module floats and reset/diverge by process.

### KAI-HB-028 — HIGH — Window parameters are fiction
`_fetch_memu_stats(days, offset_days)` and `_fetch_recent_episodes(days)` never use those arguments in requests or filtering.

### KAI-HB-029 — HIGH — Previous call is labelled previous week
The file contains only the last assessment snapshot and no timestamp/window boundaries.

### KAI-HB-030 — HIGH — Read mutates trend evidence
Every GET writes the current snapshot as the next baseline.

### KAI-HB-031 — HIGH — Anonymous trend manipulation
Rapid repeated calls can turn current values into the baseline and suppress or fabricate trends.

### KAI-HB-032 — HIGH — Diagnostics are discarded
The diagnostics request result is awaited and ignored, creating load without contributing evidence.

### KAI-HB-033 — HIGH — Wrong error metric
`budget` records Heartbeat endpoint response statuses, not Executor, memU, notification or fleet failures.

### KAI-HB-034 — HIGH — Wrong uptime metric
The calculation is one minus elapsed time since a mutable externally supplied tick divided by alert window.

### KAI-HB-035 — HIGH — Memory volume treated as quality
More stored records are automatically labelled improving even when duplicated, poisoned or synthetic.

### KAI-HB-036 — HIGH — Snapshot presented as temporal trend
CPU usage and temperatures are one current sample; no prior/peak history is retained.

### KAI-HB-037 — HIGH — Weak baseline storage
Assessment state is plaintext `/tmp` data without integrity, retention or durable volume.

### KAI-HB-038 — HIGH — Corruption disappears
Parse errors return `None`, causing a first-run/new baseline without health degradation or quarantine.

### KAI-HB-039 — HIGH — Baseline race
Concurrent GETs read the same prior file and replace it with complete non-atomic writes.

### KAI-HB-040 — HIGH — Public world context
Any caller can retrieve date/time and calendar-sync context, potentially including operational events/headlines/weather data.

---

## Medium-severity findings

### KAI-HB-041 — MEDIUM — Tamperable world cache
The cache is ordinary local JSON without signature, ownership checks or revision.

### KAI-HB-042 — MEDIUM — Remote field overwrite
`anchor.update(resp.json())` permits Calendar Sync to replace local fetched-at/date/time/day fields.

### KAI-HB-043 — MEDIUM — Downstream failure stays `ok`
Non-200 responses are ignored and the final status is unconditionally set to `ok`.

### KAI-HB-044 — MEDIUM — Persistence failure stays `ok`
Cache write errors do not alter the response status.

### KAI-HB-045 — MEDIUM — Failed-fetch retry suppression
`_last_world_fetch` is advanced even when Calendar Sync or disk persistence failed.

### KAI-HB-046 — MEDIUM — False non-stale cache response
Inside the cooldown, a missing/unreadable cache yields `{status:"cached", stale:false}` with no context.

### KAI-HB-047 — MEDIUM — Host timezone authority
Date, time and day use `time.strftime()` from the container timezone.

### KAI-HB-048 — MEDIUM — Unbounded context response
Complete Calendar Sync bytes/JSON are parsed and merged without schema/size limits.

### KAI-HB-049 — MEDIUM — Worker-local cache cadence
Each worker has its own `_last_world_fetch`; restarts and replicas duplicate requests and disagree on cache state.

### KAI-HB-050 — MEDIUM — HTTP connection churn
New clients are created for every notification, maintenance, world, stats and diagnostics operation.

### KAI-HB-051 — MEDIUM — Readiness-blind health
Health tests only elapsed tick and CPU load.

### KAI-HB-052 — MEDIUM — Temperature omitted from health
CPU/GPU temperature thresholds are not evaluated in `/health`.

### KAI-HB-053 — MEDIUM — Unsafe interval configuration
Negative/zero/extreme check, alert, idle, cooldown, assessment and world-fetch intervals are accepted.

### KAI-HB-054 — MEDIUM — Unsafe threshold configuration
NaN/infinite/negative CPU and temperature thresholds produce misleading comparisons.

### KAI-HB-055 — MEDIUM — Public request metrics
The error-budget snapshot is exposed without authentication.

### KAI-HB-056 — MEDIUM — Mislabelled ErrorBudget
The metric concerns served HTTP statuses, not heartbeat health or alert success.

### KAI-HB-057 — MEDIUM — Volatile process state
All tick/activity/sleep/world state resets on restart and differs across workers.

### KAI-HB-058 — MEDIUM — Blocking filesystem work
Executor log, cache and assessment files are read/written synchronously inside async endpoints.

### KAI-HB-059 — MEDIUM — No admission control
Status/self-assessment/world/event operations can be invoked concurrently without quotas.

### KAI-HB-060 — MEDIUM — Missing lifecycle management
No lifespan owns shared clients, a real scheduler, cancellation or shutdown drain.

### KAI-HB-061 — MEDIUM — No durable incidents
Failures are warning messages only; there is no incident ID/state/retry history.

### KAI-HB-062 — MEDIUM — Non-atomic world writes
`write_text()` can leave a partial cache if interrupted.

### KAI-HB-063 — MEDIUM — Non-atomic assessment writes
The prior generation is overwritten directly without fsync or backup.

### KAI-HB-064 — MEDIUM — Missing causal audit
Alerts and maintenance lack authenticated actor, source-log offset, observed evidence and downstream operation IDs.

---

## Batch totals

- Findings: **64**
- Critical: **2**
- High: **38**
- Medium: **24**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,021**
- Critical: **181**
- High: **992**
- Medium: **845**
- Low: **3**

## Files materially reviewed

`heartbeat/app.py`, `heartbeat/Dockerfile`, Heartbeat deployment in `docker-compose.full.yml`, and integrations with Supervisor, memU Introspection, Executor logging and notification gateways.
