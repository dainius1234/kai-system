# Kai Code Audit — Introspection and Narrative Identity Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-IDENT-001 | CRITICAL | Introspection exposes destructive memory maintenance without authentication |
| KAI-IDENT-002 | HIGH | Narrative and governance state silently falls back to process-local memory when Redis fails |
| KAI-IDENT-003 | HIGH | Redis append-with-cap is not atomic as a compound operation |
| KAI-IDENT-004 | HIGH | Corrupt Redis entries can make complete personal-data reads fall back to stale local state |
| KAI-IDENT-005 | HIGH | Introspection health ignores datastore and maintenance readiness |
| KAI-IDENT-006 | HIGH | Weekly compaction worker is not retained, supervised or restarted |
| KAI-IDENT-007 | MEDIUM | Legacy-message updates use non-transactional list scan and index mutation |
| KAI-IDENT-008 | MEDIUM | Redis failures are suppressed without durable reconciliation state |
| KAI-IDENT-009 | MEDIUM | Personal-state caps silently discard historical records |
| KAI-IDENT-010 | MEDIUM | Audit enforcement defaults to optional for privileged maintenance service |

---

## Introspection service: `memu-core/introspect_app.py`

### KAI-IDENT-001 — CRITICAL — Destructive memory maintenance is unauthenticated
**Issue:** The separate introspection service directly registers compression, cleanup, decay, revert, quarantine and quarantine-clear handlers without adding authentication or operator authorisation. Both `/memory/revert` and the alias `/revert` are exposed.  
**Risk:** Any caller with network reachability can alter, remove, revert, quarantine or release persistent memory and governance evidence.  
**Recommendation:** Place the service on a restricted management plane and require strong operator identity, scoped approvals and immutable audit linkage for every mutation.  
**Status:** OPEN — immediate remediation required

### KAI-IDENT-005 — HIGH — Health ignores maintenance readiness
**Issue:** `/health` always returns `status: ok` and only reports the detected device. It does not validate Postgres, Redis, vector-store integrity, audit availability or compaction-worker state.  
**Risk:** Orchestration can route privileged maintenance operations to an instance unable to access or safely modify the authoritative store.  
**Recommendation:** Separate liveness from dependency-aware readiness and integrity checks.  
**Status:** OPEN

### KAI-IDENT-006 — HIGH — Weekly compaction worker is unsupervised
**Issue:** Startup creates the weekly compaction loop with `asyncio.create_task` but does not retain the task, observe termination or restart it.  
**Risk:** Unexpected task failure permanently disables maintenance while the service remains healthy.  
**Recommendation:** Retain lifecycle tasks, expose worker state and fail readiness or restart when mandatory workers exit.  
**Status:** OPEN

### KAI-IDENT-010 — MEDIUM — Privileged audit enforcement is optional by default
**Issue:** `AuditStream` is created with `required` controlled by `AUDIT_REQUIRED`, defaulting to `false`.  
**Risk:** Destructive memory operations can run when durable audit logging is unavailable, weakening forensic accountability.  
**Recommendation:** Require audit availability for privileged mutation and provide a separately authorised emergency procedure.  
**Status:** OPEN

---

## Narrative, emotional and governance state: `memu-core/app.py`

### KAI-IDENT-002 — HIGH — Redis failure creates process-local split-brain state
**Issue:** P17–P22 helpers silently fall back to module-level lists and dictionaries whenever Redis is absent or an operation throws. Different workers then read and mutate independent copies of autobiography, emotional timeline, relationship milestones, values, conscience, loyalty, gratitude and other identity state.  
**Risk:** The same system can hold conflicting identities, boundaries and relationship histories simultaneously, with behaviour depending on which worker handles a request. State created during degradation is not automatically reconciled when Redis returns.  
**Recommendation:** Treat Redis as mandatory for shared identity state, fail readiness closed or use a durable outbox/reconciliation protocol.  
**Status:** OPEN

### KAI-IDENT-003 — HIGH — Append and cap are not one atomic transaction
**Issue:** Helpers describe append-with-cap as atomic but execute separate `RPUSH` and `LTRIM` commands. Each command is individually atomic; the pair is not atomic without a transaction or Lua script.  
**Risk:** Concurrent readers can observe over-cap intermediate state, and process failure between commands leaves unbounded lists until a later successful write.  
**Recommendation:** Execute append and trim in a Redis transaction or Lua script.  
**Status:** OPEN

### KAI-IDENT-004 — HIGH — One corrupt Redis item can redirect an entire read to stale fallback state
**Issue:** List readers deserialize all entries inside one broad `try`. A single malformed JSON element raises and causes the helper to return the process-local fallback list instead of the valid Redis records.  
**Risk:** Most or all authoritative emotional, relationship or identity history can disappear from a response because of one damaged record, while stale local state is presented as valid.  
**Recommendation:** Validate entries individually, quarantine corruption and expose a degraded integrity state without substituting stale data.  
**Status:** OPEN

### KAI-IDENT-007 — MEDIUM — Legacy updates are non-transactional
**Issue:** `_p18_update_entry` reads the complete Redis list, locates an item by index, then applies `LSET`. Concurrent trimming or list modification can change indexes between the read and write.  
**Risk:** The wrong legacy message can be updated, or the intended update can be lost.  
**Recommendation:** Store entries in hashes keyed by immutable ID or perform compare-and-update atomically in Lua.  
**Status:** OPEN

### KAI-IDENT-008 — MEDIUM — Redis errors have no reconciliation record
**Issue:** Most Redis exceptions are swallowed with `pass`; writes continue into local fallback state without recording a durable pending operation.  
**Risk:** Data accepted during an outage is lost on restart or remains permanently absent from the shared state after recovery.  
**Recommendation:** Use an explicit degraded state and durable replay queue, or reject writes until the authoritative store is restored.  
**Status:** OPEN

### KAI-IDENT-009 — MEDIUM — History caps silently destroy older personal records
**Issue:** Emotional timelines, reflections, relationship milestones, conscience records, loyalty and gratitude logs are trimmed to fixed caps on every append without archival, deletion evidence or user-facing retention policy.  
**Risk:** Sensitive personal and governance history disappears silently, undermining continuity, accountability and any claim of durable autobiography.  
**Recommendation:** Define retention classes, archive immutably where appropriate and record explicit deletion/tombstone events.  
**Status:** OPEN

---

## Batch totals

- Findings: **10**
- Critical: **1**
- High: **5**
- Medium: **4**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **193**
- Critical: **25**
- High: **88**
- Medium: **79**
- Low: **1**

## Files materially reviewed in this batch

`memu-core/introspect_app.py`, `memu-core/app.py` narrative-state and Redis helper paths.
