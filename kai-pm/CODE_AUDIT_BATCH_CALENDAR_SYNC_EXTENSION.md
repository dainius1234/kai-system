# Kai Code Audit — Calendar Sync / World Anchor Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_PERSISTENCE_1.md`, `CODE_AUDIT_BATCH_WORLD_ANCHOR.md` or the Heartbeat world-context batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CALSYNCX-001 | HIGH | The service named `calendar-sync` performs no calendar synchronisation or external-calendar reconciliation |
| KAI-CALSYNCX-002 | HIGH | `MEMU_URL` is configured but never used, so world-context writes have no memory/graph integration or acknowledgement |
| KAI-CALSYNCX-003 | HIGH | All news, events and date context share one global namespace with no authenticated user/calendar partition |
| KAI-CALSYNCX-004 | HIGH | World-context reads expose all global news and events without principal or purpose checks |
| KAI-CALSYNCX-005 | HIGH | Service time and suggestions use the container host timezone, not the operator’s configured timezone |
| KAI-CALSYNCX-006 | HIGH | Date responses omit timezone name and UTC offset, making displayed time ambiguous |
| KAI-CALSYNCX-007 | HIGH | Timezone-aware event timestamps are silently dropped when compared with naive server time |
| KAI-CALSYNCX-008 | HIGH | Same-day all-day ISO dates are interpreted at midnight and disappear after midnight has passed |
| KAI-CALSYNCX-009 | HIGH | Invalid event timestamps are persisted successfully and then silently omitted from every upcoming-event response |
| KAI-CALSYNCX-010 | HIGH | Event date strings are truncated to 30 characters and can turn valid timestamps into invalid persisted values |
| KAI-CALSYNCX-011 | HIGH | Missing event dates default to “now” and create immediate events rather than failing validation |
| KAI-CALSYNCX-012 | HIGH | Events have no immutable ID, source calendar, organiser, attendee, timezone, duration or recurrence identity |
| KAI-CALSYNCX-013 | HIGH | No update, delete, cancel, supersede or deduplicate operation exists for incorrect events |
| KAI-CALSYNCX-014 | HIGH | Duplicate submissions create indistinguishable repeated events and repeated downstream context |
| KAI-CALSYNCX-015 | HIGH | News items have no immutable ID, publication/event time, URL, author or verified source identity |
| KAI-CALSYNCX-016 | HIGH | Caller-supplied source text is stored as provenance without authentication or source verification |
| KAI-CALSYNCX-017 | HIGH | News ingestion replaces source chronology with the service receipt time |
| KAI-CALSYNCX-018 | HIGH | Stored news/event text is returned without an untrusted-data or prompt-injection boundary |
| KAI-CALSYNCX-019 | HIGH | Poisoned world-context text can be consumed by Heartbeat/Agentic as grounding without claim verification |
| KAI-CALSYNCX-020 | HIGH | News/event responses lack `Cache-Control: no-store` despite containing operational/private context |
| KAI-CALSYNCX-021 | HIGH | The complete JSON files are read and parsed for every request with no byte or nesting limit |
| KAI-CALSYNCX-022 | HIGH | A list containing any non-dictionary news item can crash news/context requests |
| KAI-CALSYNCX-023 | HIGH | A non-dictionary event item can crash event/context requests rather than being quarantined |
| KAI-CALSYNCX-024 | HIGH | The last-200/last-500 retention policy silently deletes earlier records without an audit or retention reason |
| KAI-CALSYNCX-025 | HIGH | Retention is based on insertion order rather than event/publication chronology or protected status |
| KAI-CALSYNCX-026 | HIGH | Context contains no source freshness, last successful ingestion or stale/unavailable state |
| KAI-CALSYNCX-027 | HIGH | Generic Monday/Friday/weekend suggestions are presented without operator schedule, locale or working-pattern evidence |
| KAI-CALSYNCX-028 | HIGH | Service/Compose identity says Calendar Sync while health and FastAPI identity say World Anchor |
| KAI-CALSYNCX-029 | MEDIUM | Date-only, local-time and offset-aware event formats have inconsistent inclusion semantics |
| KAI-CALSYNCX-030 | MEDIUM | DST transitions and ambiguous/nonexistent local times are not modelled |
| KAI-CALSYNCX-031 | MEDIUM | Events have no explicit all-day field and cannot distinguish date-only from midnight-timed events |
| KAI-CALSYNCX-032 | MEDIUM | Events have no end time or duration and cannot detect overlaps or current ongoing events |
| KAI-CALSYNCX-033 | MEDIUM | Event sorting is lexical on raw date strings rather than canonical instants |
| KAI-CALSYNCX-034 | MEDIUM | News sorting is lexical on stored timestamp strings and can be manipulated by persisted/tampered data |
| KAI-CALSYNCX-035 | MEDIUM | News and event text truncation is silent and provides no original-length or truncation marker |
| KAI-CALSYNCX-036 | MEDIUM | Response dictionaries define no strict response models or API-schema revision |
| KAI-CALSYNCX-037 | MEDIUM | Health does not test file readability, writeability, JSON schema, clock, timezone or data freshness |
| KAI-CALSYNCX-038 | MEDIUM | Data directory and seed files are created during module import rather than controlled startup |
| KAI-CALSYNCX-039 | MEDIUM | Synchronous filesystem parsing and writes run directly in async request handlers |
| KAI-CALSYNCX-040 | MEDIUM | No rate limit, caller quota or ingestion/read workload control exists |
| KAI-CALSYNCX-041 | MEDIUM | The service exposes no metrics for ingestion failures, corruption, dropped events or retained records |
| KAI-CALSYNCX-042 | MEDIUM | No structured/tamper-evident audit records who added context or when records were evicted |
| KAI-CALSYNCX-043 | MEDIUM | FastAPI dependencies and the Python base image are not reproducibly digest-pinned |
| KAI-CALSYNCX-044 | MEDIUM | No dedicated Calendar Sync tests were found for timezone, all-day, malformed or concurrent records |
| KAI-CALSYNCX-045 | MEDIUM | The service has no lifespan-owned storage validation, graceful write drain or persistence reconciliation |
| KAI-CALSYNCX-046 | MEDIUM | Responses contain no dataset revision, item version or consistent snapshot identifier |
| KAI-CALSYNCX-047 | MEDIUM | Date/time responses use wall-clock time without clock-health or monotonic sequence evidence |
| KAI-CALSYNCX-048 | MEDIUM | Calendar/world-context data has no consent, retention or privacy-deletion partition |

---

## High-severity findings

### KAI-CALSYNCX-001 — HIGH — Calendar-sync contract is false
**Issue:** The implementation reads and writes two local JSON files. It performs no Google/ICS/CalDAV/calendar-provider synchronisation, conflict resolution or source reconciliation.  
**Risk:** Other services and operators may treat this service as an authoritative current calendar when it is only a manually populated local context file.  
**Recommendation:** Rename the capability accurately or implement a source-authenticated synchronisation contract with freshness and conflict state.  
**Status:** OPEN

### KAI-CALSYNCX-002 — HIGH — Dead memU integration
`MEMU_URL` is declared but never referenced; event/news writes receive no downstream memory operation, source linkage or durable acknowledgement.

### KAI-CALSYNCX-003 — HIGH — Global world-context namespace
No user, tenant, calendar or session identity exists in storage or endpoints.

### KAI-CALSYNCX-004 — HIGH — Global context disclosure
Every caller receives the same news/events and adaptive context.

### KAI-CALSYNCX-005 — HIGH — Host-time personal advice
`datetime.now()` uses the container timezone; Compose supplies no operator timezone configuration.

### KAI-CALSYNCX-006 — HIGH — Ambiguous time response
Only date and clock strings are returned; timezone/offset are absent.

### KAI-CALSYNCX-007 — HIGH — Aware events silently disappear
`datetime.fromisoformat()` may create an offset-aware datetime. Comparing it with naive `now` raises `TypeError`, which the loop catches and treats as an invalid/omitted event.

### KAI-CALSYNCX-008 — HIGH — Current all-day event omitted
A date-only string becomes midnight. At any later time that day, `now <= ev_date` is false.

### KAI-CALSYNCX-009 — HIGH — Invalid persisted event succeeds
`POST /events` never parses/validates the date before writing and returning `status=ok`.

### KAI-CALSYNCX-010 — HIGH — Valid timestamp truncation
Thirty characters may cut a long fractional-second/offset value into invalid syntax.

### KAI-CALSYNCX-011 — HIGH — Missing date becomes fabricated now
The absence of a required event time is not rejected.

### KAI-CALSYNCX-012 — HIGH — Event identity/provenance absent
Events cannot be tied to an authoritative calendar object or update sequence.

### KAI-CALSYNCX-013 — HIGH — No correction lifecycle
An incorrect event cannot be amended or cancelled through the service.

### KAI-CALSYNCX-014 — HIGH — Duplicate event amplification
No idempotency key or semantic duplicate check exists.

### KAI-CALSYNCX-015 — HIGH — News evidence identity absent
The record lacks the fields necessary to verify origin and chronology.

### KAI-CALSYNCX-016 — HIGH — Caller labels itself as source
The `source` string has no relationship to authenticated ingestion identity.

### KAI-CALSYNCX-017 — HIGH — Source date is discarded
Receipt time replaces publication/event time.

### KAI-CALSYNCX-018 — HIGH — Untrusted context returned as ordinary grounding
No provenance/trust classification accompanies text.

### KAI-CALSYNCX-019 — HIGH — Poisoning reaches downstream reasoning
Heartbeat’s world context and Agentic integrations can consume these records as environmental facts without independent verification.

### KAI-CALSYNCX-020 — HIGH — Cacheable contextual data
World/news/event responses lack privacy-oriented cache headers.

### KAI-CALSYNCX-021 — HIGH — Unbounded whole-file parsing
The only record-count caps are applied during writes; a tampered/oversized file is fully read/decoded/parsed first.

### KAI-CALSYNCX-022 — HIGH — Non-dictionary news denial
`items.sort(key=lambda x: x.get(...))` assumes every list item is a dictionary.

### KAI-CALSYNCX-023 — HIGH — Non-dictionary event denial
`ev.get(...)` can raise `AttributeError`, which is not caught by the date-parsing exception handler.

### KAI-CALSYNCX-024 — HIGH — Silent historical deletion
Overflow truncation destroys records without deletion event or operator review.

### KAI-CALSYNCX-025 — HIGH — Wrong retention ordering
A back-dated important event added recently survives while an older inserted future/important item may be removed.

### KAI-CALSYNCX-026 — HIGH — No freshness contract
Consumers cannot determine whether data was manually updated today, months ago or partially failed.

### KAI-CALSYNCX-027 — HIGH — Uncalibrated lifestyle suggestions
The service infers work/weekend recommendations from weekday/hour alone.

### KAI-CALSYNCX-028 — HIGH — Service identity drift
Different names imply different authority and complicate monitoring/dependency contracts.

---

## Medium-severity findings

### KAI-CALSYNCX-029 — MEDIUM — Inconsistent date formats
Equivalent instants encoded differently may sort/filter differently or be omitted.

### KAI-CALSYNCX-030 — MEDIUM — DST unsupported
No named timezone or fold/nonexistent-time handling exists.

### KAI-CALSYNCX-031 — MEDIUM — All-day semantics absent
Date-only events are not represented as date ranges.

### KAI-CALSYNCX-032 — MEDIUM — Duration absent
Ongoing/conflicting appointments cannot be determined.

### KAI-CALSYNCX-033 — MEDIUM — Lexical event ordering
Raw strings rather than parsed canonical instants control display order.

### KAI-CALSYNCX-034 — MEDIUM — Lexical news ordering
Tampered timestamps can dominate the “recent” feed.

### KAI-CALSYNCX-035 — MEDIUM — Silent content truncation
No metadata tells consumers that title/summary/description/category were cut.

### KAI-CALSYNCX-036 — MEDIUM — Unversioned API shape
Free dictionaries can drift without validation.

### KAI-CALSYNCX-037 — MEDIUM — Readiness blind
Health is unconditional.

### KAI-CALSYNCX-038 — MEDIUM — Import-time storage mutation
Failures happen before a controlled readiness state and tests/imports mutate the filesystem.

### KAI-CALSYNCX-039 — MEDIUM — Blocking async handlers
File reads, JSON parsing, sorting and complete-file writes are synchronous.

### KAI-CALSYNCX-040 — MEDIUM — No workload governance
Public reads/writes can be spammed without quotas.

### KAI-CALSYNCX-041 — MEDIUM — No operational telemetry
Dropped/invalid records and persistence failures are invisible.

### KAI-CALSYNCX-042 — MEDIUM — Missing ingestion audit
There is no actor/source/revision/eviction chain.

### KAI-CALSYNCX-043 — MEDIUM — Mutable build inputs
Dependencies/base use ranges/tags.

### KAI-CALSYNCX-044 — MEDIUM — Missing temporal tests
Repository search found no dedicated Calendar Sync service test suite.

### KAI-CALSYNCX-045 — MEDIUM — Missing storage lifecycle
No startup schema validation/task drain/reconciliation exists.

### KAI-CALSYNCX-046 — MEDIUM — No snapshot revision
Multiple file reads can produce an unlabelled mixed/current dataset.

### KAI-CALSYNCX-047 — MEDIUM — Weak time provenance
Clock adjustments can change suggestions/order without a health indication.

### KAI-CALSYNCX-048 — MEDIUM — Missing privacy lifecycle
Global operator context cannot be selectively exported/deleted by principal or purpose.

---

## Batch totals

- Findings: **48**
- Critical: **0**
- High: **28**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,955**
- Critical: **195**
- High: **1,516**
- Medium: **1,241**
- Low: **3**

## Files materially reviewed

`calendar-sync/app.py`, `calendar-sync/Dockerfile`, `calendar-sync/requirements.txt`, full-stack deployment, Heartbeat/Agentic world-context consumption and existing persistence/world-anchor audits.
