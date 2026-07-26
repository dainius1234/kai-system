# Kai Code Audit — Heartbeat and Temporal Self-Model Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HB-001 | CRITICAL | Unauthenticated callers can reset heartbeat state and conceal a stale system |
| KAI-HB-002 | HIGH | Unauthenticated event endpoint can generate trusted alerts and log entries |
| KAI-HB-003 | HIGH | GET `/status` can invoke memory compression and decay side effects |
| KAI-HB-004 | HIGH | Auto-sleep inactivity logic is defeated by request middleware |
| KAI-HB-005 | HIGH | Memory maintenance success is logged without checking HTTP outcomes |
| KAI-HB-006 | HIGH | Heartbeat/world/self-assessment operational state is exposed without authentication |
| KAI-HB-007 | MEDIUM | `/status` rereads the complete executor log on every request |
| KAI-HB-008 | MEDIUM | Executor alert counting repeatedly notifies for historical hits |
| KAI-HB-009 | MEDIUM | Notification delivery ignores non-success HTTP responses |
| KAI-HB-010 | MEDIUM | Missing temperature sensors are represented as 0°C |
| KAI-HB-011 | MEDIUM | “Weekly” self-assessment does not query a previous time window |
| KAI-HB-012 | MEDIUM | Recent episode diagnostics are fetched and discarded |
| KAI-HB-013 | MEDIUM | Self-assessment GET mutates non-atomic ephemeral state |
| KAI-HB-014 | MEDIUM | World cache can report `stale: false` when no valid cached data exists |
| KAI-HB-015 | MEDIUM | World fetch reports `status: ok` when calendar sync fails |
| KAI-HB-016 | MEDIUM | Configuration thresholds and intervals are not validated |

---

## Heartbeat monitor: `heartbeat/app.py`

### KAI-HB-001 — CRITICAL — Liveness can be falsified remotely
**Issue:** `POST /recover` and `POST /tick` require no authentication. Both directly set `last_tick = time.time()`. `/recover` is explicitly described as resetting the timer to prevent stale alerts.  
**Risk:** Any network-reachable caller can conceal a failed or disconnected executor by continuously refreshing heartbeat state, causing `/health` and `/status` to report the system as healthy.  
**Recommendation:** Bind heartbeat updates to authenticated service identity, signed freshness/nonces and the actual monitored process lifecycle. Recovery must not fabricate evidence of a real heartbeat.  
**Status:** OPEN — immediate remediation required

### KAI-HB-002 — HIGH — Alert and audit channel injection
**Issue:** `POST /event` accepts arbitrary status and reason strings without authentication, logs them as executor events and forwards them to the notification gateway.  
**Risk:** Callers can create false security/operational alerts, poison logs and trigger notification spam under the trusted heartbeat identity.  
**Recommendation:** Authenticate event producers, validate event schemas and preserve signed source identity.  
**Status:** OPEN

### KAI-HB-003 — HIGH — GET endpoint can trigger destructive maintenance
**Issue:** `GET /status` calls `_auto_sleep_check` and `_watchdog_check`; those paths can POST to memory compression, focus-compression and decay endpoints.  
**Risk:** A nominally read-only GET can initiate state-changing memory maintenance, making crawlers, probes or repeated reads operationally consequential and violating HTTP safety/idempotency expectations.  
**Recommendation:** Separate observation from authorised maintenance commands and use explicit authenticated POST operations.  
**Status:** OPEN

### KAI-HB-004 — HIGH — Auto-sleep inactivity detection is effectively disabled
**Issue:** HTTP middleware sets `last_activity = time.time()` before invoking every handler. `/status` then immediately calls `_auto_sleep_check`, which sees near-zero inactivity and returns. Resource-pressure checks invoked from the same request call the same inactivity-gated function and likewise return. No independent background scheduler is present in this module.  
**Risk:** The advertised inactivity-triggered compression and decay path does not execute through its exposed check path, while operators may believe it is active.  
**Recommendation:** Track genuine operator/executor activity separately from monitoring requests and run maintenance from a supervised scheduler.  
**Status:** OPEN

### KAI-HB-005 — HIGH — Maintenance is acknowledged without response validation
**Issue:** `_auto_sleep_check` awaits POST requests but never calls `raise_for_status` or checks response status/body. It logs “memory compressed + focus-compressed + decay applied” and advances `last_sleep_action` even if endpoints return 4xx/5xx.  
**Risk:** Failed maintenance is recorded as completed and suppressed until cooldown expires, producing false operational assurance.  
**Recommendation:** Validate every result and commit completion state only after all required operations succeed or are transactionally reconciled.  
**Status:** OPEN

### KAI-HB-006 — HIGH — Sensitive operational state is unauthenticated
**Issue:** `/health`, `/metrics`, `/status`, `/self-assessment` and `/world` expose device classification, heartbeat freshness, CPU/temperature state, error rates, memory counts, trend labels, intrusion-pattern counts and world context without access control.  
**Risk:** Callers can fingerprint system health, activity and internal memory state and use it for reconnaissance or behavioural inference.  
**Recommendation:** Require scoped operational read access and redact unnecessary internal state.  
**Status:** OPEN

### KAI-HB-007 — MEDIUM — Full executor log is read per status request
**Issue:** `_scan_executor_log` calls `read_text` on the complete executor log every time `/status` is requested.  
**Risk:** Log growth makes each request increasingly expensive; unauthenticated repeated requests can amplify disk I/O, memory use and latency.  
**Recommendation:** Tail incrementally with bounded offsets and persist scan state.  
**Status:** OPEN

### KAI-HB-008 — MEDIUM — Historical hits repeatedly trigger alerts
**Issue:** Every scan counts all occurrences of `timeout`, `blocked` and `injection` in the entire file and sends a notification whenever the total is non-zero. No offset, deduplication or event identity is stored.  
**Risk:** One old log entry causes repeated alerts on every `/status` call, creating alert fatigue and notification amplification.  
**Recommendation:** Track immutable event offsets/IDs and alert only on newly observed classified events.  
**Status:** OPEN

### KAI-HB-009 — MEDIUM — Notification HTTP failures are treated as success
**Issue:** `_send_notification` catches transport exceptions but does not inspect HTTP response status.  
**Risk:** 4xx/5xx notification failures remain silent and alerts are lost.  
**Recommendation:** Validate response status, record delivery outcome and retry through a durable outbox.  
**Status:** OPEN

### KAI-HB-010 — MEDIUM — Sensor absence is represented as a safe temperature
**Issue:** CPU and GPU temperature helpers return `0.0` when sensors/tools are unavailable or fail.  
**Risk:** “Unknown” is interpreted as a valid cold reading, suppressing resource-pressure alerts and creating false safety.  
**Recommendation:** Represent unavailable measurements explicitly and degrade readiness/alert confidence.  
**Status:** OPEN

### KAI-HB-011 — MEDIUM — Temporal comparison is not a previous-week comparison
**Issue:** `_fetch_memu_stats(days, offset_days)` ignores both arguments and always calls the same current `/memory/stats` endpoint. The assessment compares the current snapshot with the previous invocation stored in a file, not a defined prior seven-day period.  
**Risk:** Output labelled as weekly improvement/decline can instead compare arbitrary request times and misrepresent trends.  
**Recommendation:** Query immutable time-bounded metrics for both explicit windows.  
**Status:** OPEN

### KAI-HB-012 — MEDIUM — Episode diagnostics have no effect
**Issue:** `_fetch_recent_episodes` is awaited, but its returned data is discarded.  
**Risk:** Documentation claims episode-derived self-assessment while the calculation ignores that evidence.  
**Recommendation:** Either incorporate validated diagnostics or remove the claim and call.  
**Status:** OPEN

### KAI-HB-013 — MEDIUM — Self-assessment read mutates fragile state
**Issue:** `GET /self-assessment` overwrites the previous-assessment file. The file defaults to `/tmp`, is written directly without locking or atomic replacement and corruption is silently treated as no previous assessment.  
**Risk:** Concurrent reads race, history disappears on restart and a nominal GET changes future results.  
**Recommendation:** Use durable versioned snapshots and an explicit scheduled/write operation.  
**Status:** OPEN

### KAI-HB-014 — MEDIUM — Invalid/missing cache is declared non-stale
**Issue:** During the rate-limit window, if the cache is absent or unreadable, `fetch_world` returns `{"status":"cached","stale":false}`.  
**Risk:** Consumers are told valid fresh cached context exists when none was returned.  
**Recommendation:** Return an unavailable/integrity-error state and only assert freshness for a validated snapshot.  
**Status:** OPEN

### KAI-HB-015 — MEDIUM — Failed world fetch returns ok
**Issue:** Calendar-sync transport failure only adds `calendar_sync: unavailable`; the function then writes the partial anchor and sets `status: ok`. Non-200 responses are similarly ignored.  
**Risk:** Downstream consumers treat incomplete world context as successful and current.  
**Recommendation:** Validate dependency response and expose explicit degraded provenance/freshness.  
**Status:** OPEN

### KAI-HB-016 — MEDIUM — Operational configuration lacks validation
**Issue:** Check intervals, alert windows, sleep/cooldown intervals, assessment window, world-fetch interval and resource thresholds are parsed directly without positive bounds or cross-field validation.  
**Risk:** Invalid settings can crash startup, disable checks, create tight loops or invert expected alert behaviour.  
**Recommendation:** Validate a typed configuration model at startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **1**
- High: **5**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **302**
- Critical: **32**
- High: **125**
- Medium: **142**
- Low: **3**

## Files materially reviewed in this batch

`heartbeat/app.py`.
