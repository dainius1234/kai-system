# Kai Code Audit — Ledger Worker Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-LEDGERWORK-001 | CRITICAL | Unauthenticated archive flooding can rotate out and delete older audit snapshots |
| KAI-LEDGERWORK-002 | CRITICAL | Configurable Tool Gate URL receives the configured bearer token without destination validation |
| KAI-LEDGERWORK-003 | HIGH | Unauthenticated callers can trigger ledger archival, verification and statistics collection |
| KAI-LEDGERWORK-004 | HIGH | Raw ledger snapshots are stored unencrypted with default filesystem permissions |
| KAI-LEDGERWORK-005 | HIGH | Archives are not signed or verified after writing |
| KAI-LEDGERWORK-006 | HIGH | Concurrent archive requests can collide, overwrite and race cleanup |
| KAI-LEDGERWORK-007 | HIGH | Ledger and verification metadata are exposed without authentication |
| KAI-LEDGERWORK-008 | HIGH | Tool Gate payload sizes and JSON structures are unbounded and weakly validated |
| KAI-LEDGERWORK-009 | MEDIUM | Snapshot filenames use second-resolution timestamps and can collide |
| KAI-LEDGERWORK-010 | MEDIUM | Synchronous archive writes, hashing metadata reads and deletions run in async handlers |
| KAI-LEDGERWORK-011 | MEDIUM | Heartbeat alert delivery status is ignored |
| KAI-LEDGERWORK-012 | MEDIUM | Verification trusts an unauthenticated remote Boolean without independent local validation |
| KAI-LEDGERWORK-013 | MEDIUM | A new HTTP client is created for every Tool Gate retry and heartbeat alert |
| KAI-LEDGERWORK-014 | MEDIUM | Raw downstream errors are logged, retained and returned to callers |
| KAI-LEDGERWORK-015 | MEDIUM | Scheduler task has no shutdown lifecycle |
| KAI-LEDGERWORK-016 | MEDIUM | History, statistics and scheduler state are process-local and unsynchronised |
| KAI-LEDGERWORK-017 | MEDIUM | Health and metrics are readiness-blind and error-budget telemetry is not populated |
| KAI-LEDGERWORK-018 | MEDIUM | Pagination, intervals, retry values, paths and service URLs are not validated |

---

## Ledger worker: `ledger-worker/app.py`

### KAI-LEDGERWORK-001 — CRITICAL — Public archive flooding destroys older evidence
**Issue:** `POST /archive` requires no authentication or authorisation. Every successful call creates a snapshot and then deletes all snapshots beyond the newest 30.  
**Risk:** Any reachable caller can repeatedly invoke the endpoint to force legitimate older audit snapshots out of retention and delete historical evidence. This is a direct integrity/forensics attack against the audit trail.  
**Recommendation:** Restrict archive creation and retention deletion to an authenticated scheduler, use immutable/WORM storage and require retention-policy approval independent of request volume.  
**Status:** OPEN — immediate remediation required

### KAI-LEDGERWORK-002 — CRITICAL — Bearer token can be exfiltrated through configured destination
**Issue:** `TOOL_GATE_URL` is accepted directly from environment configuration. `_call_tool_gate` attaches `Authorization: Bearer <BEARER_TOKEN>` to every request without validating the URL scheme, hostname or TLS identity.  
**Risk:** Compromised or mistaken deployment configuration can redirect authenticated ledger requests to an attacker-controlled server and disclose the Tool Gate bearer token.  
**Recommendation:** Pin approved HTTPS/internal hosts, validate TLS and load narrowly scoped credentials from secret-managed configuration.  
**Status:** OPEN — immediate remediation required

### KAI-LEDGERWORK-003 — HIGH — Public control of audit operations
**Issue:** `/verify`, `/stats/refresh` and `/archive` are unauthenticated. `/stats` also triggers collection when no cache exists.  
**Risk:** Callers can force repeated full-chain verification, large ledger-tail retrieval and filesystem archival, consuming Tool Gate capacity and local I/O.  
**Recommendation:** Require administrative authentication, quotas and one-in-flight jobs.  
**Status:** OPEN

### KAI-LEDGERWORK-004 — HIGH — Raw audit records are stored unencrypted
**Issue:** `archive_snapshot` writes up to 10,000 complete ledger entries as JSONL under `ARCHIVE_DIR` with process-default permissions. No encryption, permission hardening or secure deletion is implemented.  
**Risk:** Tool arguments, decisions, identities and other ledger payload data are readable by any process/user with directory access.  
**Recommendation:** Encrypt archives, enforce restrictive ownership/mode and store them in dedicated protected audit storage.  
**Status:** OPEN

### KAI-LEDGERWORK-005 — HIGH — Archive integrity is not established
**Issue:** The archive header copies a remote Merkle root and validity field, but the written JSONL file is not hashed, signed or re-read/verified against that root after writing.  
**Risk:** Partial writes, later tampering or mismatched tail/root responses are not detectable from the archive itself.  
**Recommendation:** Produce a canonical archive manifest, compute a local digest, verify record/root consistency and sign it with a protected audit key.  
**Status:** OPEN

### KAI-LEDGERWORK-006 — HIGH — Archive operations race
**Issue:** No lock or job registry prevents concurrent `/archive` calls. Same-second calls use the same filename, and every caller independently enumerates and deletes old archives.  
**Risk:** Concurrent calls can overwrite snapshots, interleave writes, delete files another request is using and produce inconsistent retention.  
**Recommendation:** Use exclusive collision-resistant files and one serialised archive/retention worker.  
**Status:** OPEN

### KAI-LEDGERWORK-007 — HIGH — Audit activity is publicly disclosed
**Issue:** `/metrics`, `/stats`, `/history` and `/archive/list` expose verification results, failed request IDs, Merkle roots, entry counts, approval rates, tool-usage names, archive filenames/sizes and timestamps without authentication.  
**Risk:** Callers can map sensitive tool activity, audit growth, policy outcomes and integrity incidents.  
**Recommendation:** Require scoped audit-reader access and minimise identifiers and operational detail.  
**Status:** OPEN

### KAI-LEDGERWORK-008 — HIGH — Remote payload processing is unbounded
**Issue:** Tool Gate responses are fully materialised and parsed without response-byte, list-length, nesting or field limits. The archive path accepts up to 10,000 entries and serialises every payload directly.  
**Risk:** A compromised or malformed Tool Gate response can exhaust memory, CPU or disk and inject oversized content into archives/history.  
**Recommendation:** Stream bounded records and validate strict endpoint-specific schemas and aggregate sizes.  
**Status:** OPEN

### KAI-LEDGERWORK-009 — MEDIUM — Snapshot name collisions
**Issue:** Archive filenames use UTC timestamps with one-second resolution.  
**Risk:** Multiple archives in one second target the same path and silently overwrite or corrupt one another.  
**Recommendation:** Use immutable UUID/job IDs and exclusive file creation.  
**Status:** OPEN

### KAI-LEDGERWORK-010 — MEDIUM — Filesystem work blocks the event loop
**Issue:** JSON serialisation/writes, file stat/glob and deletion run synchronously in async request/job functions.  
**Risk:** Large snapshots and cleanup operations block health, verification and other requests.  
**Recommendation:** Use a bounded filesystem worker or asynchronous streaming writer.  
**Status:** OPEN

### KAI-LEDGERWORK-011 — MEDIUM — Integrity alert delivery is unverified
**Issue:** `_notify_heartbeat` does not check response status/body and suppresses all exceptions.  
**Risk:** A detected ledger integrity failure can be recorded locally while the operator alert was rejected or lost.  
**Recommendation:** Validate durable alert acknowledgement and escalate delivery failure through an independent channel.  
**Status:** OPEN

### KAI-LEDGERWORK-012 — MEDIUM — Verification is delegated without independent evidence
**Issue:** `verify_chain` accepts Tool Gate’s returned `valid`, count and failed request ID; this worker does not fetch records and independently recompute the hash chain.  
**Risk:** A compromised Tool Gate can report its own ledger valid, defeating the stated independent integrity-monitor role.  
**Recommendation:** Verify exported signed ledger data independently with separately protected trust material.  
**Status:** OPEN

### KAI-LEDGERWORK-013 — MEDIUM — HTTP connection churn
**Issue:** Every retry attempt and heartbeat event creates a new `httpx.AsyncClient`.  
**Risk:** Scheduled/manual activity repeatedly creates sockets and connection pools.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-LEDGERWORK-014 — MEDIUM — Internal errors are broadly exposed
**Issue:** Raw exceptions are logged, stored in verification/stat/archive results and returned through metrics/history or HTTP 502 detail.  
**Risk:** Callers receive internal network, authentication, filesystem and Tool Gate diagnostics.  
**Recommendation:** Use stable error codes and protected redacted traces.  
**Status:** OPEN

### KAI-LEDGERWORK-015 — MEDIUM — Scheduler has no shutdown owner
**Issue:** Startup creates `_scheduled_loop` but no shutdown handler cancels and awaits it.  
**Risk:** Reloads/tests can duplicate verification loops, and shutdown can abandon in-flight audit operations.  
**Recommendation:** Manage the scheduler in FastAPI lifespan with explicit cancellation and awaited completion.  
**Status:** OPEN

### KAI-LEDGERWORK-016 — MEDIUM — State is volatile and worker-local
**Issue:** Verification history, statistics cache and scheduler task are module-level memory. Multiple workers run separate schedulers and expose inconsistent data; restart erases history.  
**Risk:** Audit monitoring is duplicated and historical verification evidence is incomplete/non-authoritative.  
**Recommendation:** Use one scheduler authority and durable shared verification/job storage.  
**Status:** OPEN

### KAI-LEDGERWORK-017 — MEDIUM — Readiness and telemetry are misleading
**Issue:** `/health` always reports ok even when the scheduler is inactive or Tool Gate/heartbeat are unavailable. `budget` is exposed but no code calls `budget.record`.  
**Risk:** Orchestration and operators receive false readiness and empty reliability metrics.  
**Recommendation:** Separate liveness, scheduler, Tool Gate and alert readiness; populate telemetry with real outcomes.  
**Status:** OPEN

### KAI-LEDGERWORK-018 — MEDIUM — Configuration and query inputs lack validation
**Issue:** Archive offsets/limits and history offsets/limits accept negative/extreme values. Verification interval, retry count, archive path, token-bearing URLs and port are parsed directly; zero intervals can create tight loops and zero retries can raise `None`.  
**Risk:** Misconfiguration or crafted queries create broken pagination, uncontrolled scheduling and unsafe routing/storage.  
**Recommendation:** Validate typed startup and request parameters with strict ranges and approved paths/destinations.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **2**
- High: **6**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **714**
- Critical: **81**
- High: **248**
- Medium: **382**
- Low: **3**

## Files materially reviewed in this batch

`ledger-worker/app.py`.
