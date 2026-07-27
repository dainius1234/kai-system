# Kai Code Audit — Ledger Worker Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_LEDGER_WORKER.md`. The existing 18 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-LEDGERX-001 | CRITICAL | The deployed Ledger Worker has no bearer token and therefore cannot collect authenticated ledger tails or create archives |
| KAI-LEDGERX-002 | CRITICAL | Archives silently contain only the newest 10,000 entries rather than the complete ledger |
| KAI-LEDGERX-003 | CRITICAL | Anonymous verification flooding can evict integrity-failure evidence from the 100-entry history |
| KAI-LEDGERX-004 | HIGH | Archives are stored under `/tmp` with no persistent volume and disappear on restart/redeployment |
| KAI-LEDGERX-005 | HIGH | The documented replay-nonce cleanup responsibility is not implemented |
| KAI-LEDGERX-006 | HIGH | The scheduler never performs archival despite archival being a stated background responsibility |
| KAI-LEDGERX-007 | HIGH | The scheduler sleeps for the full interval before the first integrity verification |
| KAI-LEDGERX-008 | HIGH | Health reports ok before any verification has completed |
| KAI-LEDGERX-009 | HIGH | Health reports ok when the scheduler is disabled, inactive or has crashed |
| KAI-LEDGERX-010 | HIGH | Tool Gate verification errors/unavailability do not generate a Heartbeat integrity alert |
| KAI-LEDGERX-011 | HIGH | Heartbeat’s request schema discards the failed ledger request ID sent by Ledger Worker |
| KAI-LEDGERX-012 | HIGH | Permanent authentication and policy failures are retried as though transient |
| KAI-LEDGERX-013 | HIGH | Verification, tail retrieval and Merkle root are fetched in separate non-snapshot requests |
| KAI-LEDGERX-014 | HIGH | Verification accepts loosely typed remote fields and can treat a truthy non-Boolean `valid` value as success |
| KAI-LEDGERX-015 | HIGH | Statistics classify every non-approved ledger entry as a denied tool decision |
| KAI-LEDGERX-016 | HIGH | Tool-use statistics miss tools nested inside co-sign and other wrapped ledger payloads |
| KAI-LEDGERX-017 | HIGH | Cached statistics have no freshness/TTL contract and may be served indefinitely |
| KAI-LEDGERX-018 | HIGH | Failed refreshes leave prior statistics active without a stale/degraded marker |
| KAI-LEDGERX-019 | HIGH | Archival duplicates trusted tokens, signatures and sensitive Gate parameters into another plaintext file |
| KAI-LEDGERX-020 | HIGH | The Tool Gate bearer credential is a plain process-environment value rather than a mounted secret identity |
| KAI-LEDGERX-021 | HIGH | The bearer token and ledger data are sent over ordinary internal HTTP without transport identity/confidentiality |
| KAI-LEDGERX-022 | HIGH | Manual and scheduled verification/stat/archive jobs have no in-flight lock or idempotency identity |
| KAI-LEDGERX-023 | MEDIUM | Scheduler cadence drifts by adding verification/statistics runtime to every configured interval |
| KAI-LEDGERX-024 | MEDIUM | Verification and statistics timestamps are naive UTC strings |
| KAI-LEDGERX-025 | MEDIUM | Verification duration uses wall-clock time rather than a monotonic clock |
| KAI-LEDGERX-026 | MEDIUM | Archive files are written directly without atomic replacement, fsync or a retained previous generation |
| KAI-LEDGERX-027 | MEDIUM | Archive retention is count-based only and has no legal-hold, incident or minimum-age protection |
| KAI-LEDGERX-028 | MEDIUM | Cleanup failure can return an archive error after the new archive was already committed |
| KAI-LEDGERX-029 | MEDIUM | Successful archive responses disclose the absolute server filesystem path |
| KAI-LEDGERX-030 | MEDIUM | AuditStream is optional and omits authenticated actor, request digest and complete job outcome |
| KAI-LEDGERX-031 | MEDIUM | Multiple workers each launch their own scheduler because no leader election exists |
| KAI-LEDGERX-032 | MEDIUM | No operation record binds verification, Merkle root, statistics and archive to one ledger revision |

---

## Critical findings

### KAI-LEDGERX-001 — CRITICAL — Deployed worker cannot archive or collect stats
**Issue:** Tool Gate’s `/ledger/tail` requires a trusted bearer token. `docker-compose.full.yml` configures no `BEARER_TOKEN` for Ledger Worker, and repository search shows no deployment source for it. `_auth_headers()` therefore returns an empty dictionary.  
**Risk:** Scheduled `collect_stats()` and every archive attempt receive 401 from the tail endpoint. The service can report health ok while its core audit-maintenance functions are non-operational.  
**Recommendation:** provision a narrowly scoped audit-reader identity from secret storage and make missing/invalid credentials fail startup readiness.  
**Status:** OPEN — immediate remediation required

### KAI-LEDGERX-002 — CRITICAL — “Current ledger” archive silently truncates history
**Issue:** `archive_snapshot()` requests `/ledger/tail?limit=10000`. Tool Gate returns only the newest requested tail, not the complete ledger. No comparison verifies that returned entries equal `/ledger/stats` count.  
**Risk:** Once the ledger exceeds 10,000 entries, every archive silently omits the oldest evidence while the header describes the result as a ledger snapshot.  
**Recommendation:** archive by immutable paginated sequence/checkpoint and require completeness against a fixed ledger revision/count.  
**Status:** OPEN — immediate remediation required

### KAI-LEDGERX-003 — CRITICAL — Public verification can erase incident history
**Issue:** `/verify` is unauthenticated and appends one result to `_verification_history`; the list automatically removes its oldest entry above 100.  
**Risk:** An attacker can call `/verify` repeatedly after an integrity incident to evict the failure record from `/history` and `/metrics`.  
**Recommendation:** persist append-only verification evidence and rate-limit/authenticate manual jobs; request volume must never control incident retention.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-LEDGERX-004 — HIGH — Archives are restart-volatile
The default archive path is `/tmp/ledger-archives`; the Docker service mounts no persistent archive volume. Container recreation deletes every snapshot.

### KAI-LEDGERX-005 — HIGH — Replay-nonce cleanup is absent
The module documentation lists nonce cleanup as responsibility 4, but no nonce endpoint, file or cleanup logic exists.

### KAI-LEDGERX-006 — HIGH — No scheduled archival
The background loop performs only `verify_chain()` and `collect_stats()`. Archives exist only when an unauthenticated HTTP caller invokes them.

### KAI-LEDGERX-007 — HIGH — Delayed first verification
The loop sleeps before its first verification. With the default value interpreted as minutes, a newly started service provides no integrity evidence for the first hour.

### KAI-LEDGERX-008 — HIGH — Evidence-free health
Health reports `status: ok` immediately after startup even when `_verification_history` is empty.

### KAI-LEDGERX-009 — HIGH — Dead scheduler remains healthy
The response exposes `scheduler: inactive`, but top-level status remains ok when scheduling is disabled or the task is done after an exception.

### KAI-LEDGERX-010 — HIGH — Verification outage is not alerted
Only a returned `valid: false` sends Heartbeat an integrity event. Authentication failure, Tool Gate outage, malformed JSON or timeout produces local status `error` without an external alert.

### KAI-LEDGERX-011 — HIGH — Alert evidence is discarded
Ledger Worker sends `failed_request_id` as an extra top-level JSON field. Heartbeat’s `EventPayload` defines only `status` and `reason`; default Pydantic handling discards the identifier before notification/logging.

### KAI-LEDGERX-012 — HIGH — Permanent failures are retried
401, 403, 404 and schema errors are retried up to `MAX_RETRIES`, amplifying load and delaying the scheduler with no chance of transient recovery.

### KAI-LEDGERX-013 — HIGH — No fixed ledger snapshot
`/ledger/verify`, `/ledger/tail` and `/ledger/merkle` are independent calls. Concurrent Gate appends can make the verification result, archived entries and Merkle root refer to different histories.

### KAI-LEDGERX-014 — HIGH — Loose verification schema
`verify.get("valid")` is used by truthiness, not strict Boolean validation. A malformed string such as `"false"` is truthy and is recorded as status ok.

### KAI-LEDGERX-015 — HIGH — Invalid approval-rate denominator
Every entry without a truthy `approved` value counts as denied, including autonomy requests, mode changes, recovery/audit events and malformed records.

### KAI-LEDGERX-016 — HIGH — Incomplete tool statistics
The code reads only `entry.payload.tool`. Co-sign records store the tool under `original_request`, and other wrapped event types are omitted.

### KAI-LEDGERX-017 — HIGH — Indefinitely stale statistics
`GET /stats` returns `_last_stats` forever once populated; it has no maximum age, ledger revision or background-task freshness requirement.

### KAI-LEDGERX-018 — HIGH — Failed refresh preserves stale success
`collect_stats()` returns an error object but does not replace `_last_stats`; subsequent `/stats` requests continue serving the prior snapshot as current.

### KAI-LEDGERX-019 — HIGH — Credential duplication into archives
Tool Gate ledger payloads contain trusted tokens/signatures and action parameters. Ledger Worker copies those complete entries into a second plaintext archive location.

### KAI-LEDGERX-020 — HIGH — Bearer token in process environment
`BEARER_TOKEN` is read directly from environment rather than a mounted secret/file descriptor with ownership and rotation controls.

### KAI-LEDGERX-021 — HIGH — Plaintext credential transport
The default Tool Gate URL is `http://tool-gate:8000`; the bearer token and sensitive ledger entries traverse the bridge network without TLS/mTLS identity.

### KAI-LEDGERX-022 — HIGH — Overlapping audit jobs
Manual endpoints and the scheduler can run verify/stats/archive concurrently. No job lock, operation ID or immutable target revision prevents duplicate/conflicting work.

---

## Medium-severity findings

### KAI-LEDGERX-023 — MEDIUM — Scheduler drift
The loop sleeps for the configured interval only after the previous work has completed, so actual cadence equals interval plus request/retry runtime.

### KAI-LEDGERX-024 — MEDIUM — Naive timestamps
`datetime.utcnow().isoformat()` omits timezone and a ledger sequence/revision.

### KAI-LEDGERX-025 — MEDIUM — Non-monotonic duration
Verification timing uses `time.time()` and can be distorted by clock adjustments.

### KAI-LEDGERX-026 — MEDIUM — Non-atomic archive commit
The final path is opened and written directly. Interruption can leave a truncated file that is still included in listings/retention.

### KAI-LEDGERX-027 — MEDIUM — Weak retention semantics
The newest 30 filenames are retained regardless of age, incident status, legal hold, integrity failure or whether a later snapshot is complete.

### KAI-LEDGERX-028 — MEDIUM — Partial success becomes whole error
If an old-file `unlink()` fails, the outer exception handler returns archive status error although the new archive was already written and logged.

### KAI-LEDGERX-029 — MEDIUM — Absolute path disclosure
The unauthenticated archive response returns `str(archive_path)` rather than a protected archive/job identifier.

### KAI-LEDGERX-030 — MEDIUM — Sparse optional audit
Audit defaults optional; successful verify/stats refreshes and manual actor identity/body digest are not durably logged.

### KAI-LEDGERX-031 — MEDIUM — Duplicate schedulers
Each worker process starts `_scheduled_loop()` independently and operates on the same Tool Gate/archive directory without leader election.

### KAI-LEDGERX-032 — MEDIUM — Missing revision correlation
Verification result, Merkle call, tail sample, statistics and archive header have no common immutable ledger sequence/hash proving they describe one state.

---

## Batch totals

- Findings: **32**
- Critical: **3**
- High: **19**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,053**
- Critical: **184**
- High: **1,011**
- Medium: **855**
- Low: **3**

## Files materially reviewed

`ledger-worker/app.py`, `ledger-worker/Dockerfile`, Tool Gate ledger endpoints and Ledger Worker deployment in `docker-compose.full.yml`, plus Heartbeat event integration.
