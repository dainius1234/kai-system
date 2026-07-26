# Kai Code Audit — Sysmetrics and World Anchor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HOST-001 | HIGH | Host process and resource inventory is exposed without authentication |
| KAI-HOST-002 | MEDIUM | Sysmetrics health reports `ok` in stub mode |
| KAI-HOST-003 | MEDIUM | Synchronous CPU sampling blocks request workers |
| KAI-HOST-004 | MEDIUM | Process CPU ranking is based on unprimed instantaneous counters |
| KAI-HOST-005 | MEDIUM | System metrics collection has no structured failure handling |
| KAI-HOST-006 | MEDIUM | Sysmetrics limits are not validated |
| KAI-HOST-007 | MEDIUM | Error-budget telemetry is exposed but never recorded |
| KAI-WORLD-001 | CRITICAL | Unauthenticated callers can inject news and calendar events into agentic context |
| KAI-WORLD-002 | HIGH | Calendar and local-news content is exposed without authentication |
| KAI-WORLD-003 | HIGH | Corrupt JSON is interpreted as an empty store and can be overwritten, erasing prior data |
| KAI-WORLD-004 | HIGH | File updates are non-atomic and lose concurrent writes |
| KAI-WORLD-005 | HIGH | World-anchor state defaults to ephemeral `/tmp` storage |
| KAI-WORLD-006 | MEDIUM | Event date handling uses naive server-local time |
| KAI-WORLD-007 | MEDIUM | Health ignores file integrity and write readiness |
| KAI-WORLD-008 | MEDIUM | Retention silently discards older news and events |
| KAI-WORLD-009 | MEDIUM | Input schemas are unstructured and weakly validated |

---

## Sysmetrics: `sysmetrics/app.py`

### KAI-HOST-001 — HIGH — Unauthenticated host reconnaissance
**Issue:** `/snapshot`, `/processes`, `/temperature` and `/battery` require no authentication. Responses expose CPU topology and frequency, memory usage, mount points and disk capacity, network counters, load averages, process IDs and names, process memory/status, temperature sensors and battery state.  
**Risk:** Reachable callers can fingerprint the host, identify valuable processes and infer workload, hardware and resource pressure.  
**Recommendation:** Restrict access to authenticated operational consumers and minimise process- and filesystem-level detail.  
**Status:** OPEN

### KAI-HOST-002 — MEDIUM — Health reports success in stub mode
**Issue:** `/health` always returns `status: ok`, including when psutil is unavailable and primary endpoints return stub/error payloads.  
**Risk:** Watchdogs treat a non-functional metrics service as ready.  
**Recommendation:** Separate liveness and collector readiness.  
**Status:** OPEN

### KAI-HOST-003 — MEDIUM — CPU sampling blocks request execution
**Issue:** `psutil.cpu_percent(interval=0.2)` performs a blocking sleep inside a synchronous request handler.  
**Risk:** Concurrent snapshot requests consume worker capacity and can be amplified into avoidable latency or denial of service.  
**Recommendation:** Sample in a background collector or use non-blocking cached measurements.  
**Status:** OPEN

### KAI-HOST-004 — MEDIUM — Process CPU ranking is unreliable
**Issue:** `process_iter(..., "cpu_percent")` reads CPU percentage values without a prior measurement interval. psutil commonly returns zero or stale values on first observation.  
**Risk:** The claimed “top N by CPU” list may not identify actual high-CPU processes.  
**Recommendation:** Maintain interval-based process samples across collection cycles.  
**Status:** OPEN

### KAI-HOST-005 — MEDIUM — Collection exceptions are not normalised
**Issue:** Snapshot calls such as CPU frequency, network counters, load average and sensor access are not wrapped in a service-level error boundary. Only disk permission and process disappearance cases are selectively caught.  
**Risk:** Platform-specific psutil failures become unhandled 500 responses and may disclose framework diagnostics through surrounding infrastructure.  
**Recommendation:** Classify unavailable metrics and return explicit partial/degraded snapshots.  
**Status:** OPEN

### KAI-HOST-006 — MEDIUM — Limits are not validated
**Issue:** `TOP_PROCESSES` and port are parsed directly. Negative or extreme process limits are accepted.  
**Risk:** Misconfiguration can produce misleading slicing behaviour or excessive response and collection cost.  
**Recommendation:** Validate bounded startup configuration.  
**Status:** OPEN

### KAI-HOST-007 — MEDIUM — Error-budget telemetry is inert
**Issue:** `ErrorBudget` is exposed but no endpoint or collection outcome is recorded.  
**Risk:** Metrics do not describe actual service reliability.  
**Recommendation:** Record classified collection and endpoint outcomes.  
**Status:** OPEN

---

## World anchor / calendar sync: `calendar-sync/app.py`

### KAI-WORLD-001 — CRITICAL — Unauthenticated context poisoning
**Issue:** `POST /news` and `POST /events` accept arbitrary caller-supplied entries without authentication or authorisation. These records are then returned through `/context`, which is explicitly designed for nudge and agentic world-context consumption.  
**Risk:** Any reachable caller can inject false news, fabricated deadlines, appointments or contextual instructions that influence autonomous reasoning and user nudges.  
**Recommendation:** Require authenticated provenance, trusted-source scopes, approval and integrity metadata before content enters world context.  
**Status:** OPEN — immediate remediation required

### KAI-WORLD-002 — HIGH — Unauthenticated personal-context disclosure
**Issue:** `/context`, `/news` and `/events` expose local notes and calendar entries without access control.  
**Risk:** Private schedules, descriptions and manually curated information are disclosed to any reachable caller.  
**Recommendation:** Apply user-scoped authentication and minimise returned fields.  
**Status:** OPEN

### KAI-WORLD-003 — HIGH — Corruption can be converted into permanent data loss
**Issue:** `_load_json` catches every error and returns an empty list. A subsequent add operation appends to that empty list and writes it back to the same file.  
**Risk:** A transient read error, malformed JSON or partial write causes the service to overwrite and permanently discard the previous news or event store.  
**Recommendation:** Fail closed on integrity errors, quarantine the damaged generation and recover from a validated prior version.  
**Status:** OPEN

### KAI-WORLD-004 — HIGH — Read-modify-write updates are non-atomic
**Issue:** Add operations load the complete file, append, then overwrite it directly without locking, temporary replacement, fsync or version checks.  
**Risk:** Concurrent requests lose updates; interruption can truncate files; multiple workers overwrite one another.  
**Recommendation:** Use a transactional shared datastore or locked atomic generation files.  
**Status:** OPEN

### KAI-WORLD-005 — HIGH — Default persistence is ephemeral
**Issue:** `WORLD_ANCHOR_DATA_DIR` defaults to `/tmp/world-anchor`.  
**Risk:** Calendar and context state can disappear on restart, container replacement or temporary-directory cleanup.  
**Recommendation:** Require an explicitly mounted durable location with protected permissions.  
**Status:** OPEN

### KAI-WORLD-006 — MEDIUM — Time handling is server-local and naive
**Issue:** `datetime.now()` and `datetime.fromisoformat()` are used without timezone requirements. Offset-aware input can also raise when compared to naive `now`, causing the event to be skipped.  
**Risk:** Upcoming-event filtering changes with host timezone and can omit valid events or misclassify deadlines.  
**Recommendation:** Store timezone-aware UTC timestamps and apply an explicit user timezone for presentation.  
**Status:** OPEN

### KAI-WORLD-007 — MEDIUM — Health ignores datastore integrity
**Issue:** `/health` always returns ok and does not verify that data files are readable, valid JSON, writable or durable.  
**Risk:** A corrupted or read-only store remains advertised as healthy.  
**Recommendation:** Separate liveness from file integrity and write readiness.  
**Status:** OPEN

### KAI-WORLD-008 — MEDIUM — Retention deletes history silently
**Issue:** News is truncated to 200 entries and events to 500 by retaining only the newest list tail. No archive, tombstone or retention evidence is created.  
**Risk:** Historical context disappears silently and ordering depends on append order rather than validated timestamps.  
**Recommendation:** Define explicit retention and immutable archival/deletion records.  
**Status:** OPEN

### KAI-WORLD-009 — MEDIUM — Inputs use arbitrary dictionaries
**Issue:** Mutation endpoints accept `Dict[str, Any]` rather than validated models. Dates are truncated strings, blank titles are accepted and no semantic or provenance checks are applied.  
**Risk:** Invalid or misleading records enter trusted context and malformed dates silently disappear from event queries.  
**Recommendation:** Use strict schemas, required fields, timezone-aware dates and provenance validation.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **1**
- High: **5**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **286**
- Critical: **31**
- High: **120**
- Medium: **132**
- Low: **3**

## Files materially reviewed in this batch

`sysmetrics/app.py`, `calendar-sync/app.py`.
