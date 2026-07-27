# Kai Code Audit — Calendar Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CALENDAR-001 | CRITICAL | Private calendar events, descriptions and locations are exposed without authentication |
| KAI-CALENDAR-002 | HIGH | Unauthenticated callers can force repeated polling of the configured CalDAV account |
| KAI-CALENDAR-003 | HIGH | Configurable CalDAV URL receives configured credentials without destination validation |
| KAI-CALENDAR-004 | HIGH | Calendar/event result volume and parsed iCalendar complexity are unbounded |
| KAI-CALENDAR-005 | HIGH | Calendar summaries expose schedule titles to downstream unauthenticated context consumers |
| KAI-CALENDAR-006 | MEDIUM | Naive event datetimes are silently reinterpreted as UTC |
| KAI-CALENDAR-007 | MEDIUM | UTC polling is combined with server-local date filtering |
| KAI-CALENDAR-008 | MEDIUM | ISO strings are used for date-window comparisons rather than parsed instants |
| KAI-CALENDAR-009 | MEDIUM | Per-calendar and per-event parsing failures are silently discarded |
| KAI-CALENDAR-010 | MEDIUM | Manual refresh and background polling can race on shared state |
| KAI-CALENDAR-011 | MEDIUM | Shutdown cancels the polling task without awaiting completion |
| KAI-CALENDAR-012 | MEDIUM | Poll errors and event counts are exposed publicly |
| KAI-CALENDAR-013 | MEDIUM | Health reports ok in stub, failed and never-polled states |
| KAI-CALENDAR-014 | MEDIUM | Error-budget telemetry is instantiated but never populated |
| KAI-CALENDAR-015 | MEDIUM | Configuration values are not validated at startup |
| KAI-CALENDAR-016 | MEDIUM | Calendar names, titles, locations and UIDs are not length-bounded |

---

## Calendar service: `calendar-service/app.py`

### KAI-CALENDAR-001 — CRITICAL — Unauthenticated private schedule disclosure
**Issue:** `GET /events/today` and `GET /events/upcoming` require no authentication or authorisation. Cached events include UID, title, start/end, location, description and calendar name.  
**Risk:** Any reachable caller can reconstruct meetings, movements, workplaces, personal appointments and associated notes from the configured private calendar account.  
**Recommendation:** Require authenticated owner-scoped access, minimise returned fields and redact descriptions/locations unless explicitly requested.  
**Status:** OPEN — immediate remediation required

### KAI-CALENDAR-002 — HIGH — Public forced CalDAV polling
**Issue:** `POST /refresh` is unauthenticated and triggers a complete CalDAV poll across all calendars and the configured look-ahead period.  
**Risk:** Repeated callers can force network, authentication, calendar-search and iCalendar parsing work, consuming upstream quotas and local CPU while racing the background poller.  
**Recommendation:** Restrict refresh to authorised operators/schedulers and rate-limit with one in-flight poll.  
**Status:** OPEN

### KAI-CALENDAR-003 — HIGH — Credential-bearing destination is configuration-controlled
**Issue:** `CALDAV_URL`, username and password are accepted directly from environment configuration and passed to `caldav.DAVClient` without host or scheme validation.  
**Risk:** Compromised or mistaken deployment configuration can direct account credentials to an unintended or attacker-controlled server.  
**Recommendation:** Pin approved HTTPS hosts, require valid TLS and retrieve credentials from secret-managed configuration.  
**Status:** OPEN

### KAI-CALENDAR-004 — HIGH — Unbounded calendar and iCalendar processing
**Issue:** The service iterates every calendar and every `date_search` result within the look-ahead period, parses complete iCalendar payloads and stores all VEVENT components in memory. No result count, response-byte, component-count or field-size limits are enforced.  
**Risk:** A large or hostile CalDAV account can consume excessive memory and CPU during polling.  
**Recommendation:** Apply page/result limits, bounded payload parsing and aggregate cache limits.  
**Status:** OPEN

### KAI-CALENDAR-005 — HIGH — Schedule titles enter unauthenticated context feeds
**Issue:** `/summary` exposes up to three event titles for today and the next event title/date. Cortex polls this endpoint and incorporates the summary into agent context.  
**Risk:** Private schedule content is both publicly readable and promoted into broader agentic context without source authentication or user consent.  
**Recommendation:** Produce a privacy-minimised summary through an authenticated internal channel with explicit provenance.  
**Status:** OPEN

### KAI-CALENDAR-006 — MEDIUM — Naive datetimes are reinterpreted as UTC
**Issue:** `_dt_to_iso` assigns `timezone.utc` to any naive `datetime` rather than preserving the calendar’s intended local/floating-time semantics.  
**Risk:** Events can be shifted by the local UTC offset, producing incorrect schedule times and date assignment.  
**Recommendation:** Resolve floating times using the calendar/event timezone or configured user timezone.  
**Status:** OPEN

### KAI-CALENDAR-007 — MEDIUM — UTC and server-local dates are mixed
**Issue:** CalDAV polling uses `datetime.now(timezone.utc)`, while `_today_events` and `_upcoming_events` use `date.today()` in the server’s local timezone.  
**Risk:** Near midnight and across timezone boundaries, events can be omitted, duplicated or assigned to the wrong day.  
**Recommendation:** Use one explicit user timezone and compare timezone-aware instants throughout.  
**Status:** OPEN

### KAI-CALENDAR-008 — MEDIUM — Date windows use string comparison
**Issue:** Upcoming filtering compares ISO date strings against event `start` strings. Date-only and datetime values, offsets and lexical forms are mixed.  
**Risk:** Events can be incorrectly included/excluded because string ordering is not a reliable substitute for normalised datetime comparison.  
**Recommendation:** Parse and normalise all starts to typed timezone-aware values before filtering.  
**Status:** OPEN

### KAI-CALENDAR-009 — MEDIUM — Parsing failures disappear
**Issue:** Every malformed event payload is ignored with `except Exception: pass`; calendar-level failures are only logged.  
**Risk:** The service presents an apparently complete schedule while silently dropping events, with no completeness indicator to callers.  
**Recommendation:** Return per-calendar freshness/error state and explicit partial-data status.  
**Status:** OPEN

### KAI-CALENDAR-010 — MEDIUM — Poll operations race
**Issue:** Background polling and `/refresh` can execute concurrently and both replace `_events`, `_last_poll` and `_poll_error` without locking or generation IDs.  
**Risk:** An older/slower poll can overwrite newer results or clear a more recent error state.  
**Recommendation:** Enforce one in-flight poll and atomically publish versioned snapshots.  
**Status:** OPEN

### KAI-CALENDAR-011 — MEDIUM — Poll task cancellation is not awaited
**Issue:** Lifespan shutdown calls `_poll_task.cancel()` but does not await task termination.  
**Risk:** In-flight executor/CalDAV work can continue after shutdown begins and cancellation exceptions/resources are not observed.  
**Recommendation:** Await cancellation and executor completion within bounded shutdown time.  
**Status:** OPEN

### KAI-CALENDAR-012 — MEDIUM — Operational diagnostics are public
**Issue:** `/health` returns raw `_poll_error`, configured status, dependency availability, poll time and event count without authentication.  
**Risk:** Callers learn account/dependency state and potentially sensitive network or authentication diagnostics.  
**Recommendation:** Restrict detailed readiness information and expose stable public status codes only.  
**Status:** OPEN

### KAI-CALENDAR-013 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when unconfigured, the CalDAV library is absent, polling failed or no successful poll has occurred.  
**Risk:** Orchestration treats a stubbed or failed schedule source as ready.  
**Recommendation:** Separate liveness, configured state and fresh-calendar readiness.  
**Status:** OPEN

### KAI-CALENDAR-014 — MEDIUM — Error budget is never recorded
**Issue:** `budget` is created and `/metrics` returns its snapshot, but no request or poll path calls `budget.record`.  
**Risk:** Reliability metrics appear available while containing no meaningful outcome data.  
**Recommendation:** Record HTTP and polling outcomes consistently.  
**Status:** OPEN

### KAI-CALENDAR-015 — MEDIUM — Startup configuration lacks validation
**Issue:** Port, refresh interval and look-ahead days are parsed directly. Zero, negative or extreme intervals/durations are not rejected; credential combinations are only checked for non-empty strings.  
**Risk:** Misconfiguration can create tight polling loops, excessive queries or startup failure.  
**Recommendation:** Validate typed configuration with safe ranges and complete credential requirements.  
**Status:** OPEN

### KAI-CALENDAR-016 — MEDIUM — Event metadata is not bounded
**Issue:** UID, summary, location and calendar name are stored without length limits; only description is sliced to 300 characters after parsing.  
**Risk:** Oversized fields consume cache, response and downstream Cortex context capacity.  
**Recommendation:** Apply strict per-field and aggregate event limits before caching.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **1**
- High: **4**
- Medium: **11**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **612**
- Critical: **68**
- High: **214**
- Medium: **327**
- Low: **3**

## Files materially reviewed in this batch

`calendar-service/app.py`.
