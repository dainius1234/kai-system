# Kai Code Audit — File Watcher Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_FILES_SERVICE.md`. The existing 13 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FILESX-001 | HIGH | The advertised no-watchdog stub mode crashes during module import because `FileSystemEventHandler` is undefined |
| KAI-FILESX-002 | HIGH | Watch count and recursive inotify descriptor use are unbounded |
| KAI-FILESX-003 | HIGH | Large-tree recursive registration can block requests and exhaust kernel watch limits |
| KAI-FILESX-004 | HIGH | Watchdog/kernel queue overflow is not detected, counted or surfaced |
| KAI-FILESX-005 | HIGH | Modify storms are neither debounced nor coalesced and can erase meaningful history |
| KAI-FILESX-006 | HIGH | Move events omit the destination path and therefore misrepresent filesystem changes |
| KAI-FILESX-007 | HIGH | Canonical aliases and symlinks can schedule the same tree repeatedly under different strings |
| KAI-FILESX-008 | HIGH | Watch-list checks, observer scheduling and list updates are unsynchronised across concurrent requests |
| KAI-FILESX-009 | HIGH | Observer schedule/start exceptions propagate as untyped 500 responses and may leave a running empty observer |
| KAI-FILESX-010 | HIGH | Dynamic watch changes are not persisted and vanish on restart |
| KAI-FILESX-011 | HIGH | File events carry no watch-root, principal, device/inode or immutable event identity |
| KAI-FILESX-012 | HIGH | No audit trail records who added/removed watches or which root exposed each event |
| KAI-FILESX-013 | HIGH | Deployment read-only mounts do not prevent metadata surveillance of every accessible container path |
| KAI-FILESX-014 | MEDIUM | Event records use wall-clock timestamps without an inotify sequence or monotonic ordering |
| KAI-FILESX-015 | MEDIUM | Relative watch paths depend on the container working directory and are returned without canonicalisation |
| KAI-FILESX-016 | MEDIUM | Removing a canonical alias does not affect a watch registered under another spelling |
| KAI-FILESX-017 | MEDIUM | `event_type` filtering accepts arbitrary strings and has no documented enum |
| KAI-FILESX-018 | MEDIUM | Event response `total` counts the complete buffer rather than the filtered result set |
| KAI-FILESX-019 | MEDIUM | Event responses have no snapshot generation and can be inconsistent with concurrent observer writes |
| KAI-FILESX-020 | MEDIUM | Event paths are not mapped to logical/redacted roots before storage and output |
| KAI-FILESX-021 | MEDIUM | No ignore rules exclude temporary, cache, secret, dependency or high-churn filesystem paths |
| KAI-FILESX-022 | MEDIUM | Directory-deletion and watch-invalidated conditions are not reflected in `_watching` or health |
| KAI-FILESX-023 | MEDIUM | Observer thread exceptions and liveness transitions have no durable incident state |
| KAI-FILESX-024 | MEDIUM | Public event reads have no rate limit or per-caller response budget |
| KAI-FILESX-025 | MEDIUM | Missing shared runtime imports silently replace structured telemetry with no-op fallbacks |
| KAI-FILESX-026 | MEDIUM | The service has no authoritative single-watcher lease, event broker or graceful watch-reconciliation lifecycle |

---

## High-severity findings

### KAI-FILESX-001 — HIGH — Stub fallback cannot import
**Issue:** The watchdog import exception sets `_WATCHDOG_OK=False`, but `FileSystemEventHandler` is not defined. The later `_Handler(FileSystemEventHandler)` class declaration raises `NameError`.  
**Risk:** A missing optional dependency crashes service startup instead of producing the documented stub mode and health response.  
**Recommendation:** define a safe fallback base or conditionally define/start the handler only when the dependency is available.  
**Status:** OPEN

### KAI-FILESX-002 — HIGH — Unlimited watches
No maximum watch roots, recursive descriptors, total paths or per-caller quota exists.

### KAI-FILESX-003 — HIGH — Recursive registration exhaustion
Scheduling a very large tree can synchronously consume inotify descriptors, memory and request time and affect host/container watch limits.

### KAI-FILESX-004 — HIGH — Kernel event loss is invisible
The application records only delivered watchdog events; inotify queue overflow/watch invalidation has no dropped-event counter or degraded status.

### KAI-FILESX-005 — HIGH — Event-storm evidence erasure
Every modify event is retained independently. Rapid writes can fill the 200-entry deque and evict unrelated security/operational events.

### KAI-FILESX-006 — HIGH — Move semantics are incomplete
`on_any_event()` stores only `event.src_path`; watchdog move events also carry `dest_path`, which is discarded.

### KAI-FILESX-007 — HIGH — Duplicate physical watches
Duplicate detection uses the raw directory string, not a resolved canonical/inode identity.

### KAI-FILESX-008 — HIGH — Watch-management races
Concurrent add/remove requests can both pass membership checks, start/schedule duplicate observers or lose list changes.

### KAI-FILESX-009 — HIGH — Partial observer startup
`Observer.start()` occurs before `schedule()`. If scheduling raises, the request returns an unhandled error while the background observer remains running.

### KAI-FILESX-010 — HIGH — Dynamic configuration loss
API-created watches live only in `_watching`; restart returns to environment paths and silently stops later additions.

### KAI-FILESX-011 — HIGH — Weak event provenance
A record cannot prove which configured watch produced it, the file identity, rename destination, process/user or a unique sequence.

### KAI-FILESX-012 — HIGH — Missing watch-control audit
No tamper-evident actor, requested/canonical path, watch handle, result or removal event is stored.

### KAI-FILESX-013 — HIGH — Read-only is not privacy isolation
The service needs no write permission to expose filenames and activity timing from mounted/container-sensitive paths.

---

## Medium-severity findings

### KAI-FILESX-014 — MEDIUM — Weak event chronology
`time.time()` can move backwards/forwards and has no kernel event sequence.

### KAI-FILESX-015 — MEDIUM — Relative-path ambiguity
Relative strings are scheduled against the process working directory but returned exactly as supplied.

### KAI-FILESX-016 — MEDIUM — Alias-removal mismatch
`DELETE` removes only an exact string in `_watching`; alternate paths to the same watch cannot manage it reliably.

### KAI-FILESX-017 — MEDIUM — Unvalidated event filter
Unknown event types return an empty set rather than a typed validation error/list of supported types.

### KAI-FILESX-018 — MEDIUM — Misleading total
The response’s total ignores both `event_type` filtering and limit.

### KAI-FILESX-019 — MEDIUM — No consistent event snapshot
The observer thread can append while the async route copies/filters/returns state, with no revision or snapshot lock.

### KAI-FILESX-020 — MEDIUM — Raw path storage
Absolute filesystem structure is stored instead of a bounded logical root plus relative path.

### KAI-FILESX-021 — MEDIUM — No noise/sensitivity policy
Dependency directories, caches, temporary files, credentials and logs are all treated identically.

### KAI-FILESX-022 — MEDIUM — Stale watch registry
Deleted/unmounted roots and invalidated OS watches remain listed as active.

### KAI-FILESX-023 — MEDIUM — Unsupervised observer failures
No watcher error callback, restart policy or readiness transition exists.

### KAI-FILESX-024 — MEDIUM — Unmetered reads
Repeated public event queries can copy/filter/serialise the complete buffer without caller quotas.

### KAI-FILESX-025 — MEDIUM — Silent telemetry downgrade
If common runtime import fails, basic logging/no-op metrics are used with normal health.

### KAI-FILESX-026 — MEDIUM — Missing authoritative watcher lifecycle
No distributed/single-process lease, shared durable event broker, replay position or shutdown reconciliation exists.

---

## Batch totals

- Findings: **26**
- Critical: **0**
- High: **13**
- Medium: **13**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,307**
- Critical: **189**
- High: **1,153**
- Medium: **962**
- Low: **3**

## Files materially reviewed

`perception/files/app.py`, the existing Files Service audit, deployment mounts and Dashboard/Agentic file-event integrations.
