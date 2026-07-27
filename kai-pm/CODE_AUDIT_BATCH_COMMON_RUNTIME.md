# Kai Code Audit — Shared Runtime, Audit and Breaker Primitives Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records shared implementation defects not already counted as endpoint-specific misuse in earlier service batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-RUNTIME-001 | CRITICAL | AuditStream hash-chain appends are non-transactional and concurrent writers can create sibling entries |
| KAI-RUNTIME-002 | HIGH | A crash between Redis `XADD` and `SET audit:last_hash` leaves the next audit entry on the wrong predecessor |
| KAI-RUNTIME-003 | HIGH | The audit chain is an unauthenticated hash, so any Redis writer can rewrite entries and recompute a valid chain |
| KAI-RUNTIME-004 | HIGH | Every service shares one global audit stream and last-hash key, coupling all audit availability and integrity |
| KAI-RUNTIME-005 | HIGH | AuditStream verifies the complete unbounded Redis stream during construction |
| KAI-RUNTIME-006 | HIGH | AuditStream uses no Redis stream maximum length, retention or archival policy |
| KAI-RUNTIME-007 | HIGH | Optional audit permanently disables itself after a transient Redis failure and never reconnects |
| KAI-RUNTIME-008 | HIGH | JSON log formatting does not JSON-escape message content and can emit invalid or forged log records |
| KAI-RUNTIME-009 | HIGH | Multiple processes rotate and write the same log file without inter-process coordination |
| KAI-RUNTIME-010 | HIGH | Reinitialising a named logger replaces handlers without closing the previous file handlers |
| KAI-RUNTIME-011 | HIGH | Log files are opened without ownership, mode, regular-file or symlink validation |
| KAI-RUNTIME-012 | HIGH | ErrorBudget treats only 429, 408 and exactly 500 as errors and ignores most 5xx failures |
| KAI-RUNTIME-013 | HIGH | ErrorBudget can retain an unbounded number of request samples within the time window |
| KAI-RUNTIME-014 | HIGH | An empty ErrorBudget reports a perfect zero error ratio without an insufficient-evidence state |
| KAI-RUNTIME-015 | HIGH | PII detection omits many credential, payment-card, identity and encoded-secret formats |
| KAI-RUNTIME-016 | HIGH | Credit-card detection does not validate issuer length or checksum and can both miss and falsely redact values |
| KAI-RUNTIME-017 | HIGH | PII scans allocate complete match lists for every pattern before counting |
| KAI-RUNTIME-018 | HIGH | CircuitBreaker half-open state permits unlimited concurrent trial requests |
| KAI-RUNTIME-019 | HIGH | CircuitBreaker and ErrorBudget state is unsynchronised, process-local and inconsistent across workers |
| KAI-RUNTIME-020 | HIGH | ErrorBudgetCircuitBreaker can open from one sample and has no minimum-evidence threshold |
| KAI-RUNTIME-021 | HIGH | ErrorBudgetCircuitBreaker inherits the incomplete HTTP error classification and can remain closed through repeated 502/503/504 failures |
| KAI-RUNTIME-022 | MEDIUM | Logger timestamps omit an explicit timezone and immutable event sequence |
| KAI-RUNTIME-023 | MEDIUM | Log messages permit newline, carriage-return, bidi and other control characters |
| KAI-RUNTIME-024 | MEDIUM | Audit entries contain unbounded service, level and message strings with no sensitivity classification |
| KAI-RUNTIME-025 | MEDIUM | Audit timestamps use wall-clock strings without source event time or a monotonic sequence |
| KAI-RUNTIME-026 | MEDIUM | `AuditStream.enabled()` reports client presence but not current chain integrity or writeability |
| KAI-RUNTIME-027 | MEDIUM | ErrorBudget window values are not validated and use wall-clock pruning |
| KAI-RUNTIME-028 | MEDIUM | PII redaction does not return source spans or a deterministic mapping to original fields |
| KAI-RUNTIME-029 | MEDIUM | Broad phone and postcode regexes can redact ordinary engineering and reference numbers |
| KAI-RUNTIME-030 | MEDIUM | CircuitBreaker invalid thresholds are silently clamped rather than rejected |
| KAI-RUNTIME-031 | MEDIUM | CircuitBreaker recovery and cooldown use wall-clock time |
| KAI-RUNTIME-032 | MEDIUM | Circuit-breaker snapshots contain no generation, last-success or last-failure identity |
| KAI-RUNTIME-033 | MEDIUM | Device detection imports Torch on demand and silently converts every failure into CPU capability |
| KAI-RUNTIME-034 | MEDIUM | Runtime primitives provide no shared lifecycle, distributed state or graceful persistence contract |

---

## Critical finding

### KAI-RUNTIME-001 — CRITICAL — Audit chain forks under concurrency
**Issue:** `AuditStream.log()` performs `GET audit:last_hash`, calculates a hash, `XADD`s the event and then separately `SET`s the new last hash. Two services can read the same predecessor and append sibling events.  
**Risk:** Normal concurrent logging breaks the claimed linear append-only chain, causing later integrity validation to disable or halt services and making ordering/evidence ambiguous.  
**Recommendation:** use one atomic Redis Lua/transaction operation with a monotonically increasing sequence and predecessor compare-and-swap, or a dedicated single-writer audit authority.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-RUNTIME-002 — HIGH — Partial audit commit
If the process fails after `XADD` but before updating `audit:last_hash`, the next writer hashes against the prior predecessor and creates a fork.

### KAI-RUNTIME-003 — HIGH — No cryptographic trust anchor
The chain uses ordinary SHA-256 with no secret/signature/external checkpoint. Any client with Redis write access can replace the stream and recompute hashes.

### KAI-RUNTIME-004 — HIGH — Global audit failure domain
All service instances use the same stream and key without service-specific partition or writer identity; one faulty writer can invalidate every service’s audit startup.

### KAI-RUNTIME-005 — HIGH — Unbounded startup verification
`xrange(min="-", max="+")` loads every audit event into memory and verifies synchronously during object construction.

### KAI-RUNTIME-006 — HIGH — No audit retention lifecycle
`xadd` supplies no `maxlen`; the stream grows indefinitely and no archive/checkpoint policy exists.

### KAI-RUNTIME-007 — HIGH — No audit recovery
A transient logging exception sets `_client=None`; optional streams never retry or surface a degraded transition.

### KAI-RUNTIME-008 — HIGH — Invalid JSON logs
The formatter injects `%(message)s` inside quoted JSON without `json.dumps`. Quotes, backslashes and line breaks can terminate fields or forge additional records.

### KAI-RUNTIME-009 — HIGH — Multi-process rotation race
TimedRotatingFileHandler is not safe for several workers rotating the same path; records/backups can be overwritten, interleaved or lost.

### KAI-RUNTIME-010 — HIGH — Handler/file-descriptor leakage
`logger.handlers = [...]` abandons existing handlers without closing them when setup is called again for the same name.

### KAI-RUNTIME-011 — HIGH — Untrusted log target
The runtime does not verify parent ownership, file permissions, symlinks or whether the target is a regular file before opening/rotating it.

### KAI-RUNTIME-012 — HIGH — Most server errors are counted healthy
The error set excludes 501, 502, 503, 504 and all other 5xx values except 500.

### KAI-RUNTIME-013 — HIGH — Request-rate memory growth
Every call appends one tuple; high traffic can create an arbitrarily large deque until samples age out.

### KAI-RUNTIME-014 — HIGH — No-evidence perfection
Before any sample, callers receive `error_ratio:0.0`, which is indistinguishable from verified flawless reliability.

### KAI-RUNTIME-015 — HIGH — Incomplete sensitive-data detection
Patterns omit common cloud/provider tokens, JWTs, private keys, bank accounts, passports, addresses, non-UK identities, encoded secrets and many payment formats.

### KAI-RUNTIME-016 — HIGH — Weak payment-card semantics
Only a simple 16-digit layout is matched; no Luhn validation or issuer/length handling exists.

### KAI-RUNTIME-017 — HIGH — PII allocation amplification
`pattern.findall(text)` materialises all matches for every expression even though only a count is needed.

### KAI-RUNTIME-018 — HIGH — Unlimited half-open probes
Once recovery time passes, every concurrent caller sees `half_open` as allowed; there is no single trial lease.

### KAI-RUNTIME-019 — HIGH — Local unsynchronised breaker truth
Failure counters/state/samples have no locks or distributed backing and differ by process.

### KAI-RUNTIME-020 — HIGH — One-sample circuit opening
One 500 response gives an error ratio of 1.0 and immediately opens the ErrorBudget circuit, regardless of configured evidence volume.

### KAI-RUNTIME-021 — HIGH — Shared incomplete classification
Because ErrorBudget ignores most 5xx responses, an ErrorBudgetCircuitBreaker can record repeated gateway/service-unavailable failures while calculating zero errors.

---

## Medium-severity findings

### KAI-RUNTIME-022 — MEDIUM — Ambiguous log time
`%(asctime)s` has no timezone/UTC marker or immutable sequence.

### KAI-RUNTIME-023 — MEDIUM — Control-character log injection
Messages are not normalised to one safe line or escaped control representation.

### KAI-RUNTIME-024 — MEDIUM — Unbounded audit metadata
Audit fields can retain secrets, PII and large operational payloads indefinitely.

### KAI-RUNTIME-025 — MEDIUM — Weak audit chronology
Timestamps are local wall-clock strings and do not bind the source event or request.

### KAI-RUNTIME-026 — MEDIUM — Misleading enabled state
A non-null Redis client may point to a stale/broken connection or an already-corrupt chain.

### KAI-RUNTIME-027 — MEDIUM — Unsafe budget window
Negative, zero, NaN or extreme windows are not rejected, and pruning depends on adjustable wall clock.

### KAI-RUNTIME-028 — MEDIUM — Redaction provenance lost
Only replacement text and counts remain; consumers cannot verify which source fields/spans were changed.

### KAI-RUNTIME-029 — MEDIUM — Engineering/reference false positives
Broad numeric address/phone patterns can classify coordinates, chainages, permits or other identifiers as PII.

### KAI-RUNTIME-030 — MEDIUM — Invalid breaker configuration hidden
Thresholds/recovery values are coerced to at least one rather than producing a startup policy error.

### KAI-RUNTIME-031 — MEDIUM — Non-monotonic breaker timing
Clock changes can reopen early or keep circuits open longer.

### KAI-RUNTIME-032 — MEDIUM — Incomplete breaker evidence
Snapshots omit timestamps/IDs of the failures and success that established the state.

### KAI-RUNTIME-033 — MEDIUM — Silent device downgrade
Any Torch/import/driver exception returns `cpu` with no error or readiness distinction.

### KAI-RUNTIME-034 — MEDIUM — Missing runtime lifecycle authority
Audit clients, handlers, samples and breakers have no shared startup validation, close/flush, persistence or distributed coordination.

---

## Batch totals

- Findings: **34**
- Critical: **1**
- High: **20**
- Medium: **13**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,564**
- Critical: **193**
- High: **1,288**
- Medium: **1,080**
- Low: **3**

## Files materially reviewed

`common/runtime.py`, with endpoint-specific Boolean ErrorBudget misuse and sanitizer consequences excluded where already logged elsewhere.
