# Kai Code Audit — Memory Compressor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MEMCOMP-001 | CRITICAL | Unauthenticated callers can trigger destructive memory consolidation, compression and archival |
| KAI-MEMCOMP-002 | CRITICAL | Automatic retries can repeat non-idempotent memory mutations after ambiguous failures |
| KAI-MEMCOMP-003 | HIGH | Manual, scheduled and watermark cycles can overlap without mutual exclusion |
| KAI-MEMCOMP-004 | HIGH | Multi-step destructive cycles have no transaction, checkpoint or rollback |
| KAI-MEMCOMP-005 | HIGH | A focus-compress failure causes an unbound-variable failure after later destructive steps complete |
| KAI-MEMCOMP-006 | HIGH | Untrusted memory statistics can automatically trigger destructive compression |
| KAI-MEMCOMP-007 | HIGH | Run history and metrics expose raw memory-maintenance results and errors without authentication |
| KAI-MEMCOMP-008 | HIGH | Scheduler and watermark background tasks have no shutdown ownership |
| KAI-MEMCOMP-009 | HIGH | Compression requests can occupy a worker for many minutes with repeated sequential calls and backoff |
| KAI-MEMCOMP-010 | MEDIUM | A new HTTP client is created for every retry attempt |
| KAI-MEMCOMP-011 | MEDIUM | Final retry still sleeps before reporting failure |
| KAI-MEMCOMP-012 | MEDIUM | HTTP response size and JSON complexity are not bounded |
| KAI-MEMCOMP-013 | MEDIUM | Downstream exception text is stored, logged and returned to callers |
| KAI-MEMCOMP-014 | MEDIUM | Run history and task state are process-local and unsynchronised |
| KAI-MEMCOMP-015 | MEDIUM | History limit is unvalidated and has misleading negative-value behaviour |
| KAI-MEMCOMP-016 | MEDIUM | Health reports ok when maintenance tasks are inactive or dependencies are unavailable |
| KAI-MEMCOMP-017 | MEDIUM | Error-budget telemetry is exposed but never records request outcomes |
| KAI-MEMCOMP-018 | MEDIUM | Intervals, retry values, watermarks and service URLs are not validated |

---

## Memory compressor: `memory-compressor/app.py`

### KAI-MEMCOMP-001 — CRITICAL — Public destructive memory maintenance
**Issue:** `POST /compress/run` requires no authentication, authorisation, approval or maintenance lock. It invokes `/memory/consolidate`, `/memory/focus-compress`, `/memory/compress` and `/memory/reflect` against the live stores. The documented operations prune, merge and archive memories.  
**Risk:** Any reachable caller can alter or remove live memory records, change active context and generate reflected insights, affecting all later agent behaviour.  
**Recommendation:** Restrict execution to a protected scheduler/administrator, require a maintenance lease and create a verified restorable snapshot before mutation.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCOMP-002 — CRITICAL — Retries repeat non-idempotent mutations
**Issue:** `_call_memu` retries every failed POST up to `MAX_RETRIES`. A timeout or connection loss after the downstream service committed a consolidation/compression/reflection is indistinguishable from a pre-commit failure, so the mutation is submitted again.  
**Risk:** Ambiguous network failures can duplicate pruning, merging, archival or insight generation and amplify irreversible memory changes.  
**Recommendation:** Use idempotency keys and durable operation status; never blindly retry destructive POSTs without confirmed non-commit.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCOMP-003 — HIGH — Compression cycles can overlap
**Issue:** Manual requests, the scheduled loop and the watermark loop all call `run_compression_cycle` directly. No lock, job registry or one-in-flight guard exists.  
**Risk:** Concurrent cycles can prune/archive the same records, race statistics, produce duplicate reflections and publish inconsistent histories.  
**Recommendation:** Use one authoritative maintenance queue with a distributed lease and idempotent job IDs.  
**Status:** OPEN

### KAI-MEMCOMP-004 — HIGH — No transaction or rollback across destructive steps
**Issue:** The cycle commits consolidation, focus compression, archival and reflection sequentially. A later failure marks the whole cycle failed but does not undo earlier mutations.  
**Risk:** Callers see a failed maintenance run even though substantial irreversible changes already occurred, encouraging retries and leaving an unknown partial state.  
**Recommendation:** Use explicit checkpoints, durable per-step status and rollback/restore semantics, or make each step independently idempotent and report partial completion accurately.  
**Status:** OPEN

### KAI-MEMCOMP-005 — HIGH — Focus failure produces a post-mutation `NameError`
**Issue:** If focus-compress raises, the exception branch stores a `{"status":"skipped"...}` dictionary but never defines `focus_result`. Later, `tokens_saved` evaluates `focus_result.get(...)` because the stored skip value is still a dictionary. This raises after consolidate, compress, reflect and post-stats may already have completed.  
**Risk:** A non-fatal focus failure turns a materially completed destructive cycle into a reported failure, promoting duplicate reruns and concealing completed changes.  
**Recommendation:** Initialise a typed focus result and derive tokens only from the stored successful response.  
**Status:** OPEN

### KAI-MEMCOMP-006 — HIGH — Automatic destructive trigger trusts one remote count
**Issue:** The watermark loop accepts `stats.get("records", 0)` from `memu-core-introspect` and starts a full destructive cycle whenever it reaches the configured threshold. No signature, freshness, corroboration or hysteresis is required.  
**Risk:** A compromised/misconfigured statistics service or transient false count can automatically prune and archive live memory.  
**Recommendation:** Authenticate the source, require fresh signed metrics and use sustained threshold/hysteresis plus a maintenance lease.  
**Status:** OPEN

### KAI-MEMCOMP-007 — HIGH — Sensitive maintenance data is public
**Issue:** `/metrics`, `/compress/history`, `/compress/status` and the manual result expose raw pre/post memory stats, consolidation/compression/reflection responses, errors, counts, timings and generated insight metadata without authentication.  
**Risk:** Callers can infer memory-store size, maintenance effects, failure details and possibly sensitive downstream content.  
**Recommendation:** Require scoped administrative access and return minimised summaries with protected diagnostics.  
**Status:** OPEN

### KAI-MEMCOMP-008 — HIGH — Background tasks are not shut down safely
**Issue:** Startup creates scheduler and watermark tasks, but there is no shutdown handler to cancel and await them.  
**Risk:** Reloads/tests can create duplicate maintenance loops, and shutdown can interrupt destructive operations without a known final state.  
**Recommendation:** Own both tasks in a lifespan context, prevent duplicate startup and await graceful termination/checkpointing.  
**Status:** OPEN

### KAI-MEMCOMP-009 — HIGH — Long unauthenticated worker occupation
**Issue:** A cycle performs up to six sequential service operations; each can retry three times with 60-second timeouts and exponential sleeps. The manual endpoint awaits the whole cycle in the request worker.  
**Risk:** Repeated callers can monopolise workers and generate sustained downstream load for extended periods.  
**Recommendation:** Return a durable job ID, process through a bounded queue and enforce caller quotas and global concurrency limits.  
**Status:** OPEN

### KAI-MEMCOMP-010 — MEDIUM — HTTP client churn
**Issue:** Every retry attempt creates a new `httpx.AsyncClient`.  
**Risk:** Scheduled and repeated maintenance causes unnecessary connection and socket churn.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-MEMCOMP-011 — MEDIUM — Unnecessary final backoff
**Issue:** The retry loop calculates and awaits exponential backoff after every exception, including the final attempt when no retry remains.  
**Risk:** Failures are delayed by an additional avoidable sleep, increasing worker occupation and timeout ambiguity.  
**Recommendation:** Sleep only when another attempt will occur.  
**Status:** OPEN

### KAI-MEMCOMP-012 — MEDIUM — Downstream payloads are unbounded
**Issue:** `_call_memu` materialises complete responses and calls `resp.json()` without byte, nesting or schema limits. Reflection insights and step results are retained in history.  
**Risk:** Large or malformed downstream responses can exhaust memory and inflate public history output.  
**Recommendation:** Enforce strict response sizes and typed per-endpoint schemas.  
**Status:** OPEN

### KAI-MEMCOMP-013 — MEDIUM — Internal errors are propagated broadly
**Issue:** Raw exception strings are logged, stored in cycle results, written to the audit stream and returned through HTTP 502 details/history.  
**Risk:** Network, filesystem and downstream-service diagnostics are exposed to unauthenticated callers and retained in multiple stores.  
**Recommendation:** Use stable error codes and protected trace IDs with redacted logs.  
**Status:** OPEN

### KAI-MEMCOMP-014 — MEDIUM — State is volatile and worker-local
**Issue:** Run history and task references are module-level process memory without locks. Multiple workers run independent schedulers/watermarks and expose different histories.  
**Risk:** Maintenance can be duplicated and operational records disappear on restart or vary by worker.  
**Recommendation:** Use one scheduler authority and durable shared job/history storage.  
**Status:** OPEN

### KAI-MEMCOMP-015 — MEDIUM — History limit is not validated
**Issue:** `/compress/history?limit=` accepts any integer. Negative values use Python negative slicing and return all but the last records rather than rejecting the value.  
**Risk:** API semantics are misleading and callers can retrieve more history than intended.  
**Recommendation:** Validate a bounded positive query parameter.  
**Status:** OPEN

### KAI-MEMCOMP-016 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, even when both maintenance tasks are inactive, have crashed or memu services are unavailable.  
**Risk:** Orchestration treats a non-operational maintenance service as ready.  
**Recommendation:** Separate liveness, scheduler state and verified dependency readiness.  
**Status:** OPEN

### KAI-MEMCOMP-017 — MEDIUM — Error budget is never populated
**Issue:** `budget` is created and exposed, but no middleware or operation calls `budget.record`.  
**Risk:** Metrics appear authoritative while containing no request outcome data.  
**Recommendation:** Record HTTP and maintenance-job outcomes consistently.  
**Status:** OPEN

### KAI-MEMCOMP-018 — MEDIUM — Configuration lacks validation
**Issue:** Intervals, retry counts/backoff, watermark, service URLs and port are accepted directly. Zero/negative intervals can create tight loops; zero retries can lead to raising `None`; unsafe URLs are not rejected.  
**Risk:** Misconfiguration causes uncontrolled polling, broken failure paths or routing to unintended services.  
**Recommendation:** Validate typed startup configuration with safe ranges and approved internal destinations.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **2**
- High: **7**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **696**
- Critical: **79**
- High: **242**
- Medium: **372**
- Low: **3**

## Files materially reviewed in this batch

`memory-compressor/app.py`.
