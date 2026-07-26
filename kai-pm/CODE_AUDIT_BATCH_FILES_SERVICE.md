# Kai Code Audit — File Watcher Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FILES-001 | CRITICAL | Unauthenticated callers can recursively monitor arbitrary accessible directories |
| KAI-FILES-002 | HIGH | File paths and activity history are exposed without authentication |
| KAI-FILES-003 | HIGH | Removing a watch does not unschedule the underlying observer |
| KAI-FILES-004 | HIGH | Directory validation permits symlinked and unrestricted host paths |
| KAI-FILES-005 | MEDIUM | Event pagination returns the wrong end of the buffer |
| KAI-FILES-006 | MEDIUM | Negative event limits produce unintended slicing behaviour |
| KAI-FILES-007 | MEDIUM | Event buffer overflow silently discards older activity |
| KAI-FILES-008 | MEDIUM | Observer thread has no application lifecycle shutdown |
| KAI-FILES-009 | MEDIUM | Watches are started at module import time |
| KAI-FILES-010 | MEDIUM | Health reports ok in stub mode and does not verify observer liveness |
| KAI-FILES-011 | MEDIUM | Watch state and events are process-local and inconsistent across workers |
| KAI-FILES-012 | MEDIUM | Error-budget recording passes a Boolean and omits exceptions |
| KAI-FILES-013 | MEDIUM | Configuration and watch-request lengths are not validated |

---

## File watcher service: `perception/files/app.py`

### KAI-FILES-001 — CRITICAL — Arbitrary recursive filesystem monitoring
**Issue:** `POST /watch` requires no authentication or authorisation. Any caller can supply any existing path accessible to the service process, and `_observer.schedule(..., recursive=True)` begins recursive monitoring.  
**Risk:** A network caller can turn the service into a host/container filesystem surveillance mechanism, observing activity in configuration, secrets, mounted repositories, user directories or other sensitive trees.  
**Recommendation:** Restrict watches to an immutable allowlist of approved roots and require authenticated administrative control.  
**Status:** OPEN — immediate remediation required

### KAI-FILES-002 — HIGH — Filesystem activity is publicly disclosed
**Issue:** `/events`, `/watching` and `/health` expose full source paths, event types, timestamps and watched-directory lists without authentication.  
**Risk:** Callers can infer filenames, project structure, user activity, secret locations and operational timing.  
**Recommendation:** Require scoped access and redact paths to approved logical roots.  
**Status:** OPEN

### KAI-FILES-003 — HIGH — Watch removal is functionally false
**Issue:** `_stop_watching` only removes the directory string from `_watching`. It never retains the `ObservedWatch` object returned by `schedule`, never calls `unschedule`, and does not stop/restart the observer.  
**Risk:** `DELETE /watch` reports a directory removed while events from that directory continue to be captured and exposed. Operators receive false assurance that surveillance stopped.  
**Recommendation:** Track observer watch handles and unschedule them atomically before reporting success.  
**Status:** OPEN

### KAI-FILES-004 — HIGH — Path controls do not constrain filesystem scope
**Issue:** `_start_watching` checks only `Path(directory).exists()`. It does not resolve canonical paths, reject symlinks, require directories, prevent traversal to mounted host paths or constrain paths to configured roots.  
**Risk:** Callers can select sensitive paths through direct, relative or symlinked references and potentially schedule invalid non-directory objects.  
**Recommendation:** Canonicalise and verify paths beneath explicit approved directories with symlink policy enforcement.  
**Status:** OPEN

### KAI-FILES-005 — MEDIUM — Event limit selects the wrong records
**Issue:** `/events` reverses the complete event list and then applies `[-limit:]`. For a positive limit smaller than the buffer, this selects the tail of the reversed list—older records—rather than the newest requested records.  
**Risk:** Consumers believe they are receiving recent activity while stale events are returned, corrupting agent context about what the operator is currently doing.  
**Recommendation:** Slice the chronological buffer first or use a correctly ordered bounded query.  
**Status:** OPEN

### KAI-FILES-006 — MEDIUM — Negative limits are accepted
**Issue:** `limit = min(limit, MAX_EVENTS)` imposes no lower bound. Negative values are used in Python slicing with unexpected results.  
**Risk:** API responses differ from documented bounded behaviour and can disclose unintended portions of the buffer.  
**Recommendation:** Validate limits with explicit positive bounds.  
**Status:** OPEN

### KAI-FILES-007 — MEDIUM — Event overflow is silent
**Issue:** `_events` is a `deque(maxlen=MAX_EVENTS)`. New events automatically evict older entries without an overflow signal, dropped-event counter or persistence.  
**Risk:** High filesystem activity or deliberate churn erases evidence and makes the event stream incomplete while appearing normal.  
**Recommendation:** Record dropped-event metrics and use durable bounded ingestion where event completeness matters.  
**Status:** OPEN

### KAI-FILES-008 — MEDIUM — Observer thread is not shut down cleanly
**Issue:** The watchdog `Observer` is started but no FastAPI lifespan/shutdown hook calls `stop()` and `join()`.  
**Risk:** Tests, reloads and process shutdown can leak or abruptly terminate observer threads, leaving inconsistent lifecycle behaviour.  
**Recommendation:** Manage the observer through application lifespan hooks.  
**Status:** OPEN

### KAI-FILES-009 — MEDIUM — Import has active filesystem side effects
**Issue:** Paths in `WATCH_DIRS` are scheduled during module import, before FastAPI startup and readiness handling.  
**Risk:** Importing the module for tests, tooling or multi-worker preload starts background threads and filesystem surveillance unexpectedly; failures cannot be represented cleanly as startup errors.  
**Recommendation:** Move side effects into explicit application startup.  
**Status:** OPEN

### KAI-FILES-010 — MEDIUM — Health is capability- and thread-blind
**Issue:** `/health` always returns `status: ok`, including when watchdog is unavailable. It does not check whether `_observer` exists, its thread is alive, scheduled watches remain active or configured paths failed.  
**Risk:** Orchestration treats a stubbed or dead watcher as functional and may interpret absence of events as absence of file activity.  
**Recommendation:** Separate liveness from watcher readiness and surface failed/dead schedules.  
**Status:** OPEN

### KAI-FILES-011 — MEDIUM — State is process-local
**Issue:** `_events`, `_watching` and `_observer` are module-level process state.  
**Risk:** Multiple workers independently schedule watchers and expose different event histories; restart loses all history and watch mutations.  
**Recommendation:** Enforce a single watcher process and shared API state, or use a dedicated event broker.  
**Status:** OPEN

### KAI-FILES-012 — MEDIUM — Reliability metrics are recorded incorrectly
**Issue:** Middleware passes `response.status_code >= 500` to `budget.record` and does not record exceptions raised before a response.  
**Risk:** Error-budget data can misclassify or omit failures.  
**Recommendation:** Record actual status codes and explicit exception outcomes.  
**Status:** OPEN

### KAI-FILES-013 — MEDIUM — Configuration and request fields are weakly bounded
**Issue:** Port and maximum-event values are parsed directly. `WatchRequest.directory` has no maximum length. Zero/negative values can alter deque construction, and oversized path strings consume request/log capacity.  
**Risk:** Misconfiguration or hostile requests can make startup fail or degrade memory and logging behaviour.  
**Recommendation:** Validate typed configuration and path field lengths.  
**Status:** OPEN

---

## Batch totals

- Findings: **13**
- Critical: **1**
- High: **3**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **411**
- Critical: **42**
- High: **158**
- Medium: **208**
- Low: **3**

## Files materially reviewed in this batch

`perception/files/app.py`.
