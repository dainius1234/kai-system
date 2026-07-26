# Kai Code Audit — Autonomous State and Paper Trading Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-ASTATE-001 | HIGH | Paper-trading trust enforcement fails open when governance is unavailable |
| KAI-ASTATE-002 | HIGH | Paper-trading operations report success after persistence failure |
| KAI-ASTATE-003 | HIGH | Position close is not transactionally consistent with trade append |
| KAI-ASTATE-004 | HIGH | Paper-trading state is process-local and concurrency-unsafe |
| KAI-ASTATE-005 | MEDIUM | Corrupt paper-trading records are silently skipped |
| KAI-ASTATE-006 | MEDIUM | Position persistence lacks fsync and multi-process coordination |
| KAI-ASTATE-007 | HIGH | Wisdom graph persistence is non-atomic and concurrency-unsafe |
| KAI-ASTATE-008 | HIGH | Corrupt wisdom graph silently starts fresh while retaining partial in-memory state risk |
| KAI-ASTATE-009 | HIGH | Wisdom confidence and edge weights are accepted without bounds |
| KAI-ASTATE-010 | HIGH | Boundary enforcement uses substring word matching with fail-neutral default alignment |
| KAI-ASTATE-011 | MEDIUM | Hypothesis verdicts are derived from untrusted memory text without provenance validation |
| KAI-ASTATE-012 | MEDIUM | Hypothesis audit logging is non-atomic, unsynchronised and failure-silent |
| KAI-ASTATE-013 | MEDIUM | Hypothesis timestamps use process-monotonic time and are not durable event timestamps |
| KAI-ASTATE-014 | MEDIUM | Curiosity and hypothesis orchestration suppresses all execution failures |

---

## Paper trading: `agentic/paper_trader.py`

### KAI-ASTATE-001 — HIGH — Paper-trading trust enforcement fails open
**Issue:** `_check_trust` catches every governance exception other than an explicit `PermissionError`, logs at debug level and continues. The module documentation explicitly labels trust-infrastructure absence as fail-open.  
**Risk:** Financial-autonomy training actions proceed precisely when the trust authority is unavailable or malfunctioning, allowing an ungoverned track record to influence later autonomy decisions.  
**Recommendation:** Fail closed for state-changing actions and expose governance dependency readiness separately.  
**Status:** OPEN

### KAI-ASTATE-002 — HIGH — Operations report success after persistence failure
**Issue:** `_save_positions` and `_append_trade` swallow all exceptions. `open_position` and `close_position` return successful objects regardless of whether durable writes completed.  
**Risk:** Callers and downstream scoring can believe trades were recorded when state will disappear or change after restart.  
**Recommendation:** Propagate durable-write failure and return success only after verified persistence.  
**Status:** OPEN

### KAI-ASTATE-003 — HIGH — Position close is not transactionally consistent with trade append
**Issue:** Closing a trade deletes the in-memory position, saves the positions file, then separately appends the closed trade. Either operation can silently fail independently.  
**Risk:** A position can disappear without a trade record, remain open while a close is reported, or be duplicated during retry. P&L and performance evidence become unreliable.  
**Recommendation:** Commit close and trade history in one ACID transaction with idempotency keys.  
**Status:** OPEN

### KAI-ASTATE-004 — HIGH — Paper-trading state is concurrency-unsafe
**Issue:** The singleton mutable dictionary is read and modified without locks, transactions or compare-and-swap controls. File updates are not coordinated across workers or processes.  
**Risk:** Concurrent opens, closes and marks can race, lose updates or calculate status from inconsistent state.  
**Recommendation:** Use a transactional shared store and serialize updates per position.  
**Status:** OPEN

### KAI-ASTATE-005 — MEDIUM — Corrupt trading records are silently skipped
**Issue:** Invalid position entries and individual JSONL trade lines are ignored without preserving or surfacing them. File-level failures are only debug logged.  
**Risk:** Losses, positions or trades can vanish from reported statistics while the service continues normally.  
**Recommendation:** Quarantine malformed records, fail integrity checks and require reconciliation before producing performance metrics.  
**Status:** OPEN

### KAI-ASTATE-006 — MEDIUM — Position persistence is not fully durable
**Issue:** The implementation uses temporary-file replacement but performs no file or directory fsync and provides no multi-process locking. Trade append similarly has no flush/fsync boundary.  
**Risk:** A crash or host failure can lose acknowledged state; multiple workers can overwrite one another.  
**Recommendation:** Use a database or implement locked, fsynced atomic persistence with recovery journals.  
**Status:** OPEN

---

## Wisdom graph: `agentic/wisdom_graph.py`

### KAI-ASTATE-007 — HIGH — Wisdom graph persistence is non-atomic and concurrency-unsafe
**Issue:** Every node or edge mutation rewrites the complete graph with `Path.write_text`, without locking, atomic replacement or transaction boundaries. Auto-edge creation can trigger repeated nested saves.  
**Risk:** Concurrent governance updates can be lost, and interruption can corrupt the value graph used in alignment decisions.  
**Recommendation:** Use transactional graph/database storage or a single-writer event log with atomic snapshots.  
**Status:** OPEN

### KAI-ASTATE-008 — HIGH — Corrupt graph silently starts fresh
**Issue:** Graph loading catches any exception, logs a warning and continues. The graph is then usable as empty or potentially partially populated depending on where deserialisation failed.  
**Risk:** Boundaries and values can disappear without blocking governance decisions, while partial state can produce non-deterministic alignment.  
**Recommendation:** Load into temporary structures, validate completely and fail readiness closed on corruption while preserving the original file.  
**Status:** OPEN

### KAI-ASTATE-009 — HIGH — Governance weights are unbounded
**Issue:** Node confidence and edge weight are ordinary floats with no finite range validation. Alignment directly uses confidence values in scoring.  
**Risk:** Negative, non-finite or excessively large caller-derived values can distort alignment, override intended scoring and produce invalid results.  
**Recommendation:** Validate finite values within explicit policy ranges and bind provenance to authorised policy updates.  
**Status:** OPEN

### KAI-ASTATE-010 — HIGH — Boundary enforcement is linguistically unsound and defaults to neutral approval
**Issue:** A boundary blocks when any single word from its content occurs as a substring in the action text. Actions with no relevant nodes return alignment `0.5` rather than an unknown or denied state.  
**Risk:** Common words can cause false blocks, while paraphrased prohibited actions can evade boundaries and receive a neutral score.  
**Recommendation:** Use structured policy predicates and treat unmatched consequential actions as requiring explicit review.  
**Status:** OPEN

---

## Hypothesis and curiosity: `agentic/hypothesis.py`, `agentic/curiosity.py`

### KAI-ASTATE-011 — MEDIUM — Hypothesis verdicts lack evidence provenance controls
**Issue:** Memory strings are concatenated directly into an LLM adjudication prompt. The engine does not verify source identity, independence, integrity or contradiction metadata before assigning fixed confidence scores.  
**Risk:** Poisoned or duplicated memory can produce a `SUPPORTED` hypothesis and an apparently authoritative confidence value.  
**Recommendation:** Require provenance-scored independent evidence and use the verifier authority rather than free-form model adjudication.  
**Status:** OPEN

### KAI-ASTATE-012 — MEDIUM — Hypothesis logging is non-atomic and failure-silent
**Issue:** Multiple cycles append to a shared Markdown file without locking or fsync. Initial creation uses a separate existence check and write. All failures are debug-only and do not affect cycle success.  
**Risk:** Concurrent entries can interleave or disappear, while callers receive tested hypotheses with no durable audit trail.  
**Recommendation:** Use an append-only transactional event store and fail or explicitly mark unaudited results.  
**Status:** OPEN

### KAI-ASTATE-013 — MEDIUM — Hypothesis formation time is not an event timestamp
**Issue:** `formed_at` uses `time.monotonic`, which is meaningful only relative to the current process lifetime and is not written in the Markdown entry.  
**Risk:** Hypotheses cannot be reliably ordered or correlated across restarts, hosts or forensic records.  
**Recommendation:** Store timezone-aware wall-clock time plus a monotonic sequence or durable event ID.  
**Status:** OPEN

### KAI-ASTATE-014 — MEDIUM — Curiosity orchestration suppresses all failures
**Issue:** The entire hypothesis cycle is wrapped in a broad exception handler that logs only at debug level and returns normal idle behaviour.  
**Risk:** Persistent engine, import, feature-flag and memory failures remain operationally invisible, making the feature appear merely inactive.  
**Recommendation:** Emit structured failure metrics and expose component readiness/state.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **0**
- High: **8**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **159**
- Critical: **21**
- High: **72**
- Medium: **65**
- Low: **1**

## Files materially reviewed in this batch

`agentic/paper_trader.py`, `agentic/wisdom_graph.py`, `agentic/hypothesis.py`, `agentic/curiosity.py`.
