# Kai Code Audit — Memory Compressor Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_MEMORY_COMPRESSOR.md`. The existing 18 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MEMCOMPX-001 | CRITICAL | A sustained high watermark can repeatedly run destructive consolidation and reflection because deployed TurboVec compression does not reduce the count |
| KAI-MEMCOMPX-002 | CRITICAL | Memory Compressor, Heartbeat and memU Introspection are independent autonomous maintenance authorities with no shared lease |
| KAI-MEMCOMPX-003 | CRITICAL | Anonymous manual-run flooding can evict destructive incident evidence from the 50-entry history |
| KAI-MEMCOMPX-004 | HIGH | The deployed TurboVec `/memory/compress` implementation is a no-op but is counted as successful archival |
| KAI-MEMCOMPX-005 | HIGH | HTTP-200 downstream payloads with `status: error`, `skipped` or malformed results are accepted as successful steps |
| KAI-MEMCOMPX-006 | HIGH | Maintenance calls use no service authentication, signed operation or delegated operator identity |
| KAI-MEMCOMPX-007 | HIGH | Maintenance commands and responses traverse plaintext internal HTTP without backend identity verification |
| KAI-MEMCOMPX-008 | HIGH | Every compression cycle operates across the global memory store without a user, tenant or source scope |
| KAI-MEMCOMPX-009 | HIGH | The scheduled cycle runs only after a full interval; repeated restarts can prevent it from ever running |
| KAI-MEMCOMPX-010 | HIGH | “Quiet period/nightly” scheduling is actually a fixed interval from process startup and ignores operator activity/timezone |
| KAI-MEMCOMPX-011 | HIGH | The cycle performs no verified postcondition against expected record, graph, vector or archive changes |
| KAI-MEMCOMPX-012 | HIGH | A cycle can return `completed` when focus compression was skipped and stale-memory compression changed nothing |
| KAI-MEMCOMPX-013 | HIGH | Public run history retains full reflection insights and detailed pre/post maintenance responses |
| KAI-MEMCOMPX-014 | HIGH | Maintenance runs have no durable job ID, idempotent operation record or resumable step state |
| KAI-MEMCOMPX-015 | HIGH | Reflection writes new high-importance memories that can increase the count and retrigger watermark maintenance |
| KAI-MEMCOMPX-016 | MEDIUM | Pre- and post-cycle statistics are not bound to one immutable store revision |
| KAI-MEMCOMPX-017 | MEDIUM | `tokens_saved` stores a percentage rather than a token count |
| KAI-MEMCOMPX-018 | MEDIUM | Run timestamps are naive UTC strings without timezone or store revision |
| KAI-MEMCOMPX-019 | MEDIUM | Cycle duration uses wall-clock time rather than a monotonic clock |
| KAI-MEMCOMPX-020 | MEDIUM | Scheduled cadence drifts by adding the full cycle runtime to each interval |
| KAI-MEMCOMPX-021 | MEDIUM | Watermark count parsing assumes a directly convertible `records` field without a strict stats schema |
| KAI-MEMCOMPX-022 | MEDIUM | Audit records omit authenticated actor, maintenance lease, input/output revision and per-step operation IDs |

---

## Critical findings

### KAI-MEMCOMPX-001 — CRITICAL — Repeating destructive watermark loop
**Issue:** The watermark loop runs a full cycle whenever `records >= WATERMARK_HIGH`. The deployed TurboVec store inherits a `/memory/compress` implementation that returns zero archival work. There is no low-water hysteresis or completed-cycle cooldown beyond the five-minute check interval.  
**Risk:** If consolidate/focus-compress do not reduce the count below 4,500—or reflection adds replacement records—the service repeatedly consolidates, merges and generates new insights every five minutes.  
**Recommendation:** require a distributed maintenance lease, high/low watermarks, a verified reduction postcondition and a long failure/no-progress backoff.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCOMPX-002 — CRITICAL — Multiple autonomous maintenance authorities
**Issue:** Memory Compressor runs scheduled and watermark cycles; Heartbeat independently calls compress/focus-compress/decay; memU Introspection launches its own weekly compression loop. They coordinate through no shared job/lease authority.  
**Risk:** Separate processes can concurrently mutate the same Postgres/TurboVec memory store, causing duplicate reflection, conflicting deletion/insert cycles and index inconsistency.  
**Recommendation:** consolidate all maintenance into one durable serialised scheduler with a distributed lease and immutable operation IDs.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCOMPX-003 — CRITICAL — Public runs erase maintenance history
**Issue:** Every unauthenticated `/compress/run` appends a result and removes the oldest record above 50.  
**Risk:** After an incident or partial destructive cycle, a caller can trigger enough runs to evict that evidence from metrics/history.  
**Recommendation:** persist append-only job evidence independently of request volume and authenticate/rate-limit manual runs.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-MEMCOMPX-004 — HIGH — Compression success is false in deployment
The TurboVec store uses the inherited compression stub returning zero values; Memory Compressor logs and reports it as the stale-memory archival step.

### KAI-MEMCOMPX-005 — HIGH — Business-status failures are ignored
`_call_memu()` checks only HTTP status. An HTTP-200 JSON body declaring error, skipped, partial or no-op is treated as a successful typed result.

### KAI-MEMCOMPX-006 — HIGH — Unauthenticated downstream mutation
No HMAC, mTLS, bearer credential, nonce, job grant or operator delegation is attached to memU maintenance requests.

### KAI-MEMCOMPX-007 — HIGH — Unverified plaintext transport
Default URLs use `http://`; the worker does not verify service identity/version or a signed response.

### KAI-MEMCOMPX-008 — HIGH — Global maintenance scope
Requests contain no authenticated user/tenant/source filter; consolidate, focus-compress, compress and reflect act on the entire store.

### KAI-MEMCOMPX-009 — HIGH — Restart starvation
The scheduled loop sleeps 24 hours before the first cycle. Restarting more frequently than the interval prevents scheduled maintenance indefinitely.

### KAI-MEMCOMPX-010 — HIGH — Not a quiet-period scheduler
The schedule is elapsed hours from startup and does not inspect operator activity, local timezone, maintenance windows or system load.

### KAI-MEMCOMPX-011 — HIGH — Missing postcondition
The worker does not verify that expected records were archived, graph nodes reconciled, vector IDs consistent, backups restorable or counts changed as claimed.

### KAI-MEMCOMPX-012 — HIGH — False completed status
Focus failure is explicitly non-fatal, and compress can be an acknowledged no-op. The cycle still sets `status="completed"` after reflection/post-stats.

### KAI-MEMCOMPX-013 — HIGH — Sensitive derived-data retention
Run history stores complete step dictionaries, including reflection insights, memory counts, failures and downstream metadata, and exposes them publicly.

### KAI-MEMCOMPX-014 — HIGH — No durable operation identity
A cycle exists only as an awaited request and volatile history object; no job ID, per-step idempotency key, resume token or authoritative status store exists.

### KAI-MEMCOMPX-015 — HIGH — Reflection can feed the trigger
Reflection writes new high-importance summaries into the same memory store. Those records increase count and become source material for later reflection cycles.

---

## Medium-severity findings

### KAI-MEMCOMPX-016 — MEDIUM — Inconsistent pre/post snapshots
Stats calls occur before and after multiple independent mutations and concurrent writers; no store generation/revision proves the difference belongs to this cycle.

### KAI-MEMCOMPX-017 — MEDIUM — Incorrect result unit
`tokens_saved` is assigned `focus_result.savings_pct`, so consumers receive a percentage under a token-count name.

### KAI-MEMCOMPX-018 — MEDIUM — Ambiguous timestamps
`datetime.utcnow().isoformat()` omits timezone, sequence and data-store revision.

### KAI-MEMCOMPX-019 — MEDIUM — Non-monotonic timing
Duration uses `time.time()` and can be distorted by wall-clock adjustment.

### KAI-MEMCOMPX-020 — MEDIUM — Schedule drift
The loop sleeps only after each completed cycle, so cadence equals interval plus cycle/retry runtime.

### KAI-MEMCOMPX-021 — MEDIUM — Fragile stats contract
`int(stats.get("records", 0))` accepts no explicit stats schema/version and fails the whole check on malformed/non-finite values.

### KAI-MEMCOMPX-022 — MEDIUM — Incomplete audit evidence
Audit messages contain aggregate counts/error text but no actor, source revision, distributed lease, exact downstream operation IDs or postcondition digest.

---

## Batch totals

- Findings: **22**
- Critical: **3**
- High: **12**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,093**
- Critical: **187**
- High: **1,033**
- Medium: **870**
- Low: **3**

## Files materially reviewed

`memory-compressor/app.py`, the existing Memory Compressor audit, memU Core/Introspection maintenance implementations, Heartbeat maintenance triggers and deployment configuration.
