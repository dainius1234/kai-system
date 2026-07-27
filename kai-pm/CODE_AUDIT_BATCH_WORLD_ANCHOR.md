# Kai Code Audit — World Anchor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WORLD-001 | CRITICAL | Unauthenticated callers can inject persistent events and news into agent/nudge world context |
| KAI-WORLD-002 | HIGH | Local event and news context is exposed without authentication |
| KAI-WORLD-003 | HIGH | All callers share one unpartitioned world-context namespace |
| KAI-WORLD-004 | HIGH | JSON persistence uses non-atomic read-modify-write operations without locking |
| KAI-WORLD-005 | MEDIUM | Corrupt or unreadable context files are silently treated as empty |
| KAI-WORLD-006 | MEDIUM | Synchronous file reads/writes execute inside async handlers |
| KAI-WORLD-007 | MEDIUM | Event dates are accepted as arbitrary strings and invalid records are silently skipped |
| KAI-WORLD-008 | MEDIUM | Date/time context depends on the container’s local timezone rather than an explicit user timezone |
| KAI-WORLD-009 | MEDIUM | Request bodies and dictionary complexity are unbounded before field truncation |
| KAI-WORLD-010 | MEDIUM | Context entries have no identity, provenance, update or deletion controls |
| KAI-WORLD-011 | MEDIUM | Health reports ok without validating file integrity or directory writability |
| KAI-WORLD-012 | MEDIUM | Data-directory and port configuration are not validated |

---

## World anchor / calendar sync: `calendar-sync/app.py`

### KAI-WORLD-001 — CRITICAL — Persistent world-context poisoning
**Issue:** `POST /news` and `POST /events` require no authentication or authorisation. Caller-controlled titles, summaries, sources, dates, descriptions and categories are persisted and returned by `/context`, which the module explicitly describes as context for the nudge engine.  
**Risk:** Any reachable caller can inject false deadlines, events, news or situational claims into the agent’s real-world grounding layer, influencing later advice and proactive nudges as if the operator had supplied them.  
**Recommendation:** Accept only authenticated, provenance-signed updates from approved sources and clearly separate untrusted notes from trusted world state.  
**Status:** OPEN — immediate remediation required

### KAI-WORLD-002 — HIGH — World context is publicly readable
**Issue:** `/context`, `/news`, `/events` and `/date` require no authentication and return stored notes, event titles/descriptions/categories and adaptive suggestions.  
**Risk:** Callers can inspect the operator’s locally maintained schedule, notes and contextual signals.  
**Recommendation:** Require owner-scoped access and minimise sensitive fields.  
**Status:** OPEN

### KAI-WORLD-003 — HIGH — Global cross-user context contamination
**Issue:** All data is stored in two global JSON files. No user, session, source or tenant partition exists.  
**Risk:** Different users, services, tests and attackers read and modify the same contextual memory, causing persistent cross-session contamination.  
**Recommendation:** Partition storage by authenticated principal and context source.  
**Status:** OPEN

### KAI-WORLD-004 — HIGH — Persistence is race-prone and non-atomic
**Issue:** Each write loads the entire JSON list, appends, truncates and rewrites the target file directly. No lock, temporary file, atomic rename or version check is used.  
**Risk:** Concurrent requests can lose updates; crashes can truncate or corrupt world-context files.  
**Recommendation:** Use transactional storage or locked atomic write-rename with fsync.  
**Status:** OPEN

### KAI-WORLD-005 — MEDIUM — Corruption becomes empty context
**Issue:** `_load_json` catches every exception and returns an empty list.  
**Risk:** File corruption or permission failure is presented as “no news/events,” allowing false normality and silently discarding evidence of a storage failure.  
**Recommendation:** Expose explicit storage-error state and preserve/quarantine damaged files.  
**Status:** OPEN

### KAI-WORLD-006 — MEDIUM — File I/O blocks async endpoints
**Issue:** All JSON reads, parsing, sorting, serialisation and writes are synchronous operations performed directly inside async handlers.  
**Risk:** Larger files or slow storage block the event-loop worker.  
**Recommendation:** Use asynchronous transactional storage or bounded worker execution.  
**Status:** OPEN

### KAI-WORLD-007 — MEDIUM — Event date validation is deferred and lossy
**Issue:** `/events` stores the caller’s date as a truncated string without parsing it. `_upcoming_events` later attempts `datetime.fromisoformat` and silently skips invalid values.  
**Risk:** The API confirms successful persistence for events that can never appear in context; malformed or timezone-inconsistent dates vanish without warning.  
**Recommendation:** Parse and normalise timezone-aware dates before accepting the event.  
**Status:** OPEN

### KAI-WORLD-008 — MEDIUM — Container-local time drives behavioural suggestions
**Issue:** `_date_context` and event filtering use naive `datetime.now()`. No configured user timezone is used.  
**Risk:** Day, time-of-day, weekend state and suggestions such as “It’s late” can be wrong when the container timezone differs from the operator’s timezone.  
**Recommendation:** Use an explicit authenticated-user timezone and timezone-aware datetimes.  
**Status:** OPEN

### KAI-WORLD-009 — MEDIUM — Body complexity is unbounded
**Issue:** Endpoints accept arbitrary dictionaries. Individual selected fields are truncated only after the full JSON body and nested values are parsed/stringified.  
**Risk:** Oversized or deeply nested bodies consume request, conversion and memory resources before limits apply.  
**Recommendation:** Use typed schemas with body, field and nesting limits.  
**Status:** OPEN

### KAI-WORLD-010 — MEDIUM — Context records are unauditable and unmanageable
**Issue:** News and event entries receive no stable ID, authenticated author, source signature or revision metadata. No update/delete endpoint exists.  
**Risk:** Poisoned or mistaken entries cannot be reliably identified, corrected or removed through the service, and provenance cannot be assessed.  
**Recommendation:** Add immutable IDs, source provenance, revision history and authorised correction/deletion controls.  
**Status:** OPEN

### KAI-WORLD-011 — MEDIUM — Health is storage-blind
**Issue:** `/health` always returns ok and does not verify that JSON files are valid, readable/writable or that the data directory has capacity.  
**Risk:** Orchestration treats corrupted or unwritable context storage as ready.  
**Recommendation:** Separate liveness from verified storage integrity/readiness.  
**Status:** OPEN

### KAI-WORLD-012 — MEDIUM — Configuration lacks validation
**Issue:** The data directory and port are accepted directly from environment values; directory creation occurs at import time.  
**Risk:** Invalid or unsafe paths cause import/startup failure or write context into unintended locations.  
**Recommendation:** Validate approved absolute paths and numeric ranges during controlled startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **1**
- High: **3**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **624**
- Critical: **69**
- High: **217**
- Medium: **335**
- Low: **3**

## Files materially reviewed in this batch

`calendar-sync/app.py`.
