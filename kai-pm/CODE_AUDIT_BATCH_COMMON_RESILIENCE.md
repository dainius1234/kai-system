# Kai Code Audit — Shared Resilience and Healing Primitives Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records shared-library defects not already counted as endpoint-specific findings in earlier service batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-RESILIENCE-001 | CRITICAL | HealingEngine records `auto_recovery` as a successful known fix without executing or verifying any recovery action |
| KAI-RESILIENCE-002 | HIGH | Caller-supplied `fix_applied` is immediately stored as a proven fix without postcondition evidence |
| KAI-RESILIENCE-003 | HIGH | Known fixes are discovered but never applied before the state machine advances toward healthy |
| KAI-RESILIENCE-004 | HIGH | HTTP methods other than GET are silently converted to POST |
| KAI-RESILIENCE-005 | HIGH | Derived breaker names collapse different ports and services on the same hostname into one circuit |
| KAI-RESILIENCE-006 | HIGH | Caller-controlled `service_name` can intentionally collide with or bypass existing circuit state |
| KAI-RESILIENCE-007 | HIGH | Breaker creation is unsynchronised despite a declared but unused lock |
| KAI-RESILIENCE-008 | HIGH | Retry logic has no idempotency, request digest or committed-but-timed-out reconciliation |
| KAI-RESILIENCE-009 | HIGH | Retry logic ignores `Retry-After`, method safety and permanent/transient failure classes |
| KAI-RESILIENCE-010 | HIGH | Fallback values are indistinguishable from genuine successful response data |
| KAI-RESILIENCE-011 | HIGH | Deep health checks run sequentially with no per-check or whole-probe deadline |
| KAI-RESILIENCE-012 | HIGH | A service with zero registered health checks reports healthy |
| KAI-RESILIENCE-013 | HIGH | Duplicate health-check names overwrite earlier results and hide failed dependencies |
| KAI-RESILIENCE-014 | HIGH | Health-check exception text is copied into returned status data |
| KAI-RESILIENCE-015 | HIGH | TaskWatchdog cannot detect required tasks that never produced their first heartbeat |
| KAI-RESILIENCE-016 | HIGH | Healing phase progression depends on call count rather than verified diagnosis, action or outcome |
| KAI-RESILIENCE-017 | HIGH | Healing containment force-opens shared breakers by hard-coding exactly three failure records |
| KAI-RESILIENCE-018 | HIGH | Healing knowledge and phase state are process-local, volatile and inconsistent across workers |
| KAI-RESILIENCE-019 | MEDIUM | New HTTP clients are constructed for every retry attempt |
| KAI-RESILIENCE-020 | MEDIUM | Retry backoff has no jitter and can synchronise fleet-wide retry storms |
| KAI-RESILIENCE-021 | MEDIUM | Zero or negative retry counts record a failure without making any request |
| KAI-RESILIENCE-022 | MEDIUM | Raw URLs and exception strings can enter resilience logs |
| KAI-RESILIENCE-023 | MEDIUM | ServiceHealth stores one mutable last result without a snapshot revision or lock |
| KAI-RESILIENCE-024 | MEDIUM | Health and watchdog timestamps use wall-clock time rather than monotonic time |
| KAI-RESILIENCE-025 | MEDIUM | Watchdog names and stale thresholds are unvalidated and process-local |
| KAI-RESILIENCE-026 | MEDIUM | Healing history, errors, fixes and service names are unbounded caller-controlled strings |
| KAI-RESILIENCE-027 | MEDIUM | `history_limit=0` retains the complete failure history because `-0` slicing means no truncation |
| KAI-RESILIENCE-028 | MEDIUM | Healing state updates are unsynchronised and concurrent calls can skip or duplicate phases |
| KAI-RESILIENCE-029 | MEDIUM | `knowledge_base()` returns shallow copies whose nested dictionaries remain externally mutable |
| KAI-RESILIENCE-030 | MEDIUM | Healing reset marks a service healthy without clearing or reconciling its failure history and learned fixes |

---

## Critical finding

### KAI-RESILIENCE-001 — CRITICAL — Fabricated successful auto-recovery
**Issue:** A typical failure progresses from containment to diagnosis and then to `PHASE_KNOWLEDGE`. On the next `heal()` call, the code invokes `_record_knowledge(service, error, "auto_recovery")`, stores that string as the working fix and sets the phase healthy. No recovery command, dependency check or postcondition occurs.  
**Risk:** The shared self-healing authority can claim recovery, close incident reasoning and teach future calls a fix that never happened. Downstream automation may suppress escalation based on fabricated success.  
**Recommendation:** separate observation, proposed action, executed action and verified outcome; never record knowledge or healthy state without authenticated execution evidence and a service-specific postcondition.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-RESILIENCE-002 — HIGH — Caller assertion becomes proven fix
Any non-empty `fix_applied` skips diagnosis and is immediately persisted in `_knowledge`, with no authorised actor or successful health check.

### KAI-RESILIENCE-003 — HIGH — Known fix is never applied
Diagnosis returns `known_fix_found` and moves the phase to knowledge, but performs no action. A later call can record auto-recovery instead.

### KAI-RESILIENCE-004 — HIGH — Method coercion
`resilient_call()` implements GET explicitly and sends POST for every other method string, including PUT, PATCH and DELETE.

### KAI-RESILIENCE-005 — HIGH — Breaker identity collision by hostname
When `service_name` is absent, the key removes scheme, port and path. `http://localhost:8001` and `http://localhost:8002` both become `localhost` and share failures.

### KAI-RESILIENCE-006 — HIGH — Caller-selected breaker namespace
Callers may supply arbitrary service names, allowing one dependency to poison another’s breaker or evade existing failures by changing spelling.

### KAI-RESILIENCE-007 — HIGH — Breaker creation race
`_breaker_lock` is declared but never used. Concurrent first calls can construct/replace separate breakers for one name.

### KAI-RESILIENCE-008 — HIGH — Mutation retries lack operation identity
A timeout or JSON error after a backend committed a POST causes another POST with no idempotency key or reconciliation.

### KAI-RESILIENCE-009 — HIGH — Retry policy ignores semantics
All exceptions and 5xx responses retry identically; response retry guidance, method idempotency and permanent failures are not modelled.

### KAI-RESILIENCE-010 — HIGH — Fallback ambiguity
A fallback may be `None`, `{}`, `[]` or a normal-looking business object. The return contract carries no source, error or circuit-open state.

### KAI-RESILIENCE-011 — HIGH — Deep health can hang indefinitely
Each check is awaited serially without an application timeout. One blocked dependency prevents every later check and the health response.

### KAI-RESILIENCE-012 — HIGH — No-check health is green
An empty check registry yields `degraded=False` and status ok.

### KAI-RESILIENCE-013 — HIGH — Duplicate check names hide evidence
Results are a dictionary keyed by name; a later check with the same name overwrites the earlier state.

### KAI-RESILIENCE-014 — HIGH — Health diagnostic disclosure
Up to 80 characters of raw dependency exception text is returned to callers.

### KAI-RESILIENCE-015 — HIGH — Missing task is invisible
TaskWatchdog tracks only names that called `heartbeat()`. A required loop that never started or died before its first beat is absent rather than frozen.

### KAI-RESILIENCE-016 — HIGH — Calls, not outcomes, drive healing
Each invocation advances the current phase regardless of whether the system changed, recovered, worsened or was rechecked.

### KAI-RESILIENCE-017 — HIGH — Hard-coded containment threshold
Containment calls `record_failure()` exactly three times, assuming every shared breaker opens at three and disregarding its prior/configured threshold.

### KAI-RESILIENCE-018 — HIGH — Worker-local recovery knowledge
Histories, known fixes, phases and breakers exist only in one process and disappear on restart.

---

## Medium-severity findings

### KAI-RESILIENCE-019 — MEDIUM — Connection-pool churn
Every attempt constructs and closes a new `AsyncClient`.

### KAI-RESILIENCE-020 — MEDIUM — Synchronised backoff
Deterministic exponential sleeps cause replicas/callers failing together to retry together.

### KAI-RESILIENCE-021 — MEDIUM — Zero-attempt failure
`retries<=0` skips the loop, records one breaker failure and returns fallback with no network attempt.

### KAI-RESILIENCE-022 — MEDIUM — Sensitive logging
Failure logs include the complete URL and exception string, which may contain credentials, query data or internal topology.

### KAI-RESILIENCE-023 — MEDIUM — Mutable one-result health cache
`_last_result` has no generation, lock, immutable copy or expiry semantics.

### KAI-RESILIENCE-024 — MEDIUM — Non-monotonic liveness clocks
Service health timestamps and watchdog age calculations use `time.time()`.

### KAI-RESILIENCE-025 — MEDIUM — Weak watchdog configuration
Negative/NaN stale thresholds and arbitrary/high-cardinality task names are accepted.

### KAI-RESILIENCE-026 — MEDIUM — Unbounded failure metadata
Error/fix/service strings enter memory and returned status/details without length or sensitivity controls.

### KAI-RESILIENCE-027 — MEDIUM — Zero history cap bug
When `history_limit=0`, slicing `[-0:]` returns the complete list rather than clearing it.

### KAI-RESILIENCE-028 — MEDIUM — Healing races
History append, phase read/write and knowledge mutation have no lock or transaction.

### KAI-RESILIENCE-029 — MEDIUM — Mutable nested knowledge escapes
`dict(self._knowledge)` copies only the outer mapping; consumers can mutate each service’s inner dictionary.

### KAI-RESILIENCE-030 — MEDIUM — Reset does not reconcile evidence
`reset()` changes only the phase. Failure counts and possibly false learned fixes remain and can influence later diagnosis.

---

## Batch totals

- Findings: **30**
- Critical: **1**
- High: **17**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,530**
- Critical: **192**
- High: **1,268**
- Medium: **1,067**
- Low: **3**

## Files materially reviewed

`common/resilience.py`, with duplicate endpoint consequences reconciled against existing Dashboard, Supervisor, Agentic and Executor batches.
