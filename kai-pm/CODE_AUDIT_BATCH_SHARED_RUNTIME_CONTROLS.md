# Kai Code Audit — Shared Runtime Controls

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation and duplicate reconciliation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-RUNTIME-001 | HIGH | JSON logger permits message-based JSON/log injection |
| KAI-RUNTIME-002 | MEDIUM | Logger path is arbitrary and its parent is not created safely |
| KAI-RUNTIME-003 | MEDIUM | Repeated logger setup leaks/replaces handlers without closing them |
| KAI-RUNTIME-004 | MEDIUM | Log files are not permission-hardened |
| KAI-RUNTIME-005 | MEDIUM | Log timestamps use implicit local time |
| KAI-RUNTIME-006 | MEDIUM | Every record is duplicated to file and stdout without sensitivity policy |
| KAI-RUNTIME-007 | HIGH | `sanitize_string` is not a safe command, path, SQL or prompt sanitiser |
| KAI-RUNTIME-008 | MEDIUM | Sanitisation deletes characters and can merge tokens into new meanings |
| KAI-RUNTIME-009 | HIGH | Prompt-injection detection is a small bypassable phrase blacklist |
| KAI-RUNTIME-010 | MEDIUM | Injection regex creates false positives and has no provenance context |
| KAI-RUNTIME-011 | HIGH | PII detection misses broad classes and common representations of sensitive data |
| KAI-RUNTIME-012 | HIGH | PII patterns generate material false positives and can destroy valid data |
| KAI-RUNTIME-013 | HIGH | Unicode, encoding and obfuscation bypass PII and injection patterns |
| KAI-RUNTIME-014 | HIGH | Regex scans process unbounded text and match collections |
| KAI-RUNTIME-015 | MEDIUM | Redaction output has no protected mapping or field-level provenance |
| KAI-RUNTIME-016 | MEDIUM | Credit-card and identifier patterns lack validity checks |
| KAI-RUNTIME-017 | HIGH | ErrorBudget ignores most HTTP failure statuses |
| KAI-RUNTIME-018 | HIGH | Transport, parsing and timeout exceptions have no ErrorBudget representation |
| KAI-RUNTIME-019 | HIGH | A single error can open ratio-based controls without a minimum sample size |
| KAI-RUNTIME-020 | HIGH | ErrorBudget sample storage is unbounded within the time window |
| KAI-RUNTIME-021 | MEDIUM | ErrorBudget windows use non-monotonic wall-clock time |
| KAI-RUNTIME-022 | HIGH | ErrorBudget state is process-local and concurrency-unsafe |
| KAI-RUNTIME-023 | CRITICAL | Audit verification returns true when no audit backend exists |
| KAI-RUNTIME-024 | HIGH | Audit persistence defaults to optional fail-open operation |
| KAI-RUNTIME-025 | CRITICAL | Concurrent audit writers can fork the global hash chain |
| KAI-RUNTIME-026 | HIGH | Audit append and last-hash update are not atomic |
| KAI-RUNTIME-027 | HIGH | Audit hashes are unkeyed and forgeable by a Redis writer |
| KAI-RUNTIME-028 | HIGH | Stored `audit:last_hash` is not verified against the stream tail |
| KAI-RUNTIME-029 | HIGH | Audit Redis stream has no retention or size cap |
| KAI-RUNTIME-030 | HIGH | Startup verification scans the complete audit stream |
| KAI-RUNTIME-031 | HIGH | Audit Redis destination is unvalidated and may lack TLS/authentication |
| KAI-RUNTIME-032 | HIGH | Runtime audit failures silently disable future logging |
| KAI-RUNTIME-033 | MEDIUM | Audit messages have no byte limit or redaction contract |
| KAI-RUNTIME-034 | MEDIUM | Audit entries lack event IDs, actor identity and operation correlation |
| KAI-RUNTIME-035 | HIGH | CircuitBreaker half-open state permits unlimited concurrent probes |
| KAI-RUNTIME-036 | HIGH | One success closes the breaker despite concurrent or recent failures |
| KAI-RUNTIME-037 | MEDIUM | Failures while open reset the recovery timer indefinitely |
| KAI-RUNTIME-038 | HIGH | CircuitBreaker state is process-local and concurrency-unsafe |
| KAI-RUNTIME-039 | HIGH | Unknown/restored breaker states are treated as permissive |
| KAI-RUNTIME-040 | HIGH | ErrorBudgetCircuitBreaker inherits incomplete failure classification |
| KAI-RUNTIME-041 | HIGH | Ratio circuit can open on the first recorded failure |
| KAI-RUNTIME-042 | HIGH | A later low ratio can close an open guard before recovery validation |
| KAI-RUNTIME-043 | MEDIUM | Error-ratio thresholds accept invalid, non-finite and ineffective values |
| KAI-RUNTIME-044 | HIGH | Ratio-breaker half-open mode also permits unlimited probes |

---

## Logging and sanitisation: `common/runtime.py`

### KAI-RUNTIME-001 — HIGH — Logger output is not validly JSON-encoded
**Issue:** the formatter inserts `%(message)s` directly inside quoted JSON. Quotes, backslashes, control characters and newlines in a message are not JSON-escaped.  
**Risk:** untrusted text can break log records, inject apparent fields/records and mislead parsers or incident analysis.  
**Recommendation:** serialize a structured dictionary with a JSON encoder.  
**Status:** OPEN

### KAI-RUNTIME-002 — MEDIUM — Unsafe log destination
The supplied path is used directly; parent creation, canonicalisation, ownership and symlink policy are absent.

### KAI-RUNTIME-003 — MEDIUM — Handler lifecycle leak
`logger.handlers` is replaced on every setup call without closing existing file handlers, leaking descriptors and rotation owners.

### KAI-RUNTIME-004 — MEDIUM — Weak log permissions
TimedRotatingFileHandler creates ordinary files with process umask and no sensitive-data classification.

### KAI-RUNTIME-005 — MEDIUM — Ambiguous chronology
Formatter timestamps use logging’s local-time default and contain no explicit UTC offset.

### KAI-RUNTIME-006 — MEDIUM — Unconditional duplicate exposure
All records go to both persistent file and stdout; no policy distinguishes sensitive audit/context messages from safe operational logs.

### KAI-RUNTIME-007 — HIGH — Misleading sanitisation boundary
**Issue:** `sanitize_string` removes only `;`, `|` and `&`. Newlines, command substitutions, quotes, redirections, path traversal, SQL syntax and prompt instructions remain.  
**Risk:** callers can treat the function as security sanitisation while dangerous constructs pass.  
**Recommendation:** use typed allowlisted operations rather than string deletion.  
**Status:** OPEN

### KAI-RUNTIME-008 — MEDIUM — Character deletion changes semantics
Removing separators can join tokens/commands and produce a different value rather than safely rejecting the input.

### KAI-RUNTIME-009 — HIGH — Injection detector is trivially bypassed
The regex covers a few English phrases only. Spacing, punctuation, Unicode, translation, synonyms, indirect instructions and encoded content bypass it.

### KAI-RUNTIME-010 — MEDIUM — Context-free false positives
Benign discussions quoting “system prompt” or “ignore previous” are flagged, while source role/provenance is ignored.

### KAI-RUNTIME-011 — HIGH — Incomplete PII scope
Patterns omit names, addresses beyond UK postcodes, dates of birth, bank details, UTRs, passport/driving-licence numbers, crypto keys and many token formats.

### KAI-RUNTIME-012 — HIGH — Destructive false positives
The broad phone pattern can match ordinary numerical identifiers; credit-card pattern accepts any 16 digits. Redaction can corrupt non-PII operational data.

### KAI-RUNTIME-013 — HIGH — Normalisation bypass
No Unicode normalisation, de-obfuscation or encoded-text handling occurs before PII/injection matching.

### KAI-RUNTIME-014 — HIGH — Unbounded regex workload
Complete text is scanned by every pattern and `findall` materialises all matches with no input or match limit.

### KAI-RUNTIME-015 — MEDIUM — Redaction loses structural provenance
All values become generic tags; no protected record identifies the original field, policy reason or authorised recovery process.

### KAI-RUNTIME-016 — MEDIUM — No identifier validity
Credit-card matches have no Luhn test; NI/postcode/phone matches do not enforce full domain validity.

---

## Error budget: `common/runtime.py`

### KAI-RUNTIME-017 — HIGH — Most failures count as success
**Issue:** only 429, 500 and 408 are errors. 400/401/403/404/409/422 and 501–599 except 500 do not increase the error ratio.  
**Risk:** authentication failures, safety blocks, dependency outages and gateway errors make reliability look healthy.  
**Recommendation:** use operation-specific typed success/failure classification.  
**Status:** OPEN

### KAI-RUNTIME-018 — HIGH — Exceptions are absent from the model
The API accepts only an integer status, so transport exceptions and malformed responses cannot be represented unless callers invent a code.

### KAI-RUNTIME-019 — HIGH — No statistical floor
Ratio-based breakers may open immediately from one failure because no minimum sample count/confidence interval exists.

### KAI-RUNTIME-020 — HIGH — High-rate memory growth
All samples inside the window remain in a deque with no maximum; unauthenticated high request rates consume memory.

### KAI-RUNTIME-021 — MEDIUM — Wall-clock window
Clock rollback retains samples too long; forward jumps discard history.

### KAI-RUNTIME-022 — HIGH — Worker-local unsynchronised reliability
Deque mutation/snapshots are unlocked and independent per thread/process, so fleet error ratios differ and race.

---

## Audit stream: `common/runtime.py`

### KAI-RUNTIME-023 — CRITICAL — Absence verifies successfully
**Issue:** `verify_or_halt()` returns `True` when `_client` is absent, including no configured Redis or a previous optional connection failure.  
**Risk:** hardening/readiness code can report audit integrity verified when no audit data exists.  
**Recommendation:** return `unavailable` and fail closed whenever audit is required.  
**Status:** OPEN — immediate remediation required

### KAI-RUNTIME-024 — HIGH — Fail-open is the default
`required=False` is the constructor default and many services instantiate it without enforcing durable audit availability.

### KAI-RUNTIME-025 — CRITICAL — Global chain forks under concurrency
**Issue:** every writer reads shared `audit:last_hash`, separately appends to the stream and separately updates the key. Two writers can use the same predecessor, creating two children; one last-hash update wins.  
**Risk:** normal concurrent service logging breaks the claimed single append-only chain and enables event omission/reordering.  
**Recommendation:** perform append and predecessor update atomically through one sequencer/Lua transaction.  
**Status:** OPEN — immediate remediation required

### KAI-RUNTIME-026 — HIGH — Append is not transactional
A failure between XADD and SET leaves a stream event not reflected by `last_hash`; the reverse state can also arise through external mutation.

### KAI-RUNTIME-027 — HIGH — Chain is not authenticated
SHA-256 has no secret/signature. A Redis writer can rewrite events and recompute the entire chain.

### KAI-RUNTIME-028 — HIGH — Tail authority is unchecked
Verification recomputes the stream but never compares the computed final hash to `audit:last_hash`, so the coordination key can be stale or malicious.

### KAI-RUNTIME-029 — HIGH — Audit storage grows indefinitely
XADD has no MAXLEN/retention/archive policy.

### KAI-RUNTIME-030 — HIGH — Full-stream startup scan
`xrange("-", "+")` retrieves and verifies the complete history on construction, with no pagination, checkpoint or byte bound.

### KAI-RUNTIME-031 — HIGH — Redis trust zone is unvalidated
Arbitrary Redis URLs are accepted without requiring TLS, server identity, restricted credentials or an approved host.

### KAI-RUNTIME-032 — HIGH — Failure permanently disables logging
For optional streams, any log/verify exception sets `_client=None`; later calls silently return without reconnecting or reporting a persistent readiness failure.

### KAI-RUNTIME-033 — MEDIUM — Unbounded audit message
Complete caller messages are written with no size, PII or secret redaction limit.

### KAI-RUNTIME-034 — MEDIUM — Weak event schema
Entries contain timestamp, service, level and message only—no event ID, authenticated actor, trace, operation or schema/key version.

---

## Circuit breakers: `common/runtime.py`

### KAI-RUNTIME-035 — HIGH — Half-open flood
Once recovery time elapses, the first `allow()` sets half_open; every subsequent call returns true because only `open` is restricted. There is no single trial request.

### KAI-RUNTIME-036 — HIGH — One success erases concurrent failure evidence
Any successful call resets failures and closes the breaker, even if other in-flight calls are failing or the success belongs to a different dependency operation.

### KAI-RUNTIME-037 — MEDIUM — Open timer can be extended forever
Every `record_failure()` after threshold resets `opened_at`, delaying recovery indefinitely under continued probe traffic.

### KAI-RUNTIME-038 — HIGH — No shared/atomic state
State, counters and timestamps are mutable process fields without locks or distributed coordination.

### KAI-RUNTIME-039 — HIGH — Invalid states fail permissively
`allow()` blocks only exact `state == "open"`; corrupted/restored unknown values allow traffic.

### KAI-RUNTIME-040 — HIGH — Ratio guard misclassifies failures
ErrorBudgetCircuitBreaker uses the same three-code error set and ignores most operational failures.

### KAI-RUNTIME-041 — HIGH — First-error opening
With no minimum samples, one recorded qualifying failure produces ratio 1.0 and opens at default thresholds.

### KAI-RUNTIME-042 — HIGH — Recovery can bypass cooldown
`record()` sets state closed whenever current ratio falls below warn, even if the guard was open and recovery time has not elapsed.

### KAI-RUNTIME-043 — MEDIUM — Unsafe threshold domains
NaN, infinity and values above one are accepted by `max`; impossible thresholds can permanently disable or distort opening.

### KAI-RUNTIME-044 — HIGH — Ratio half-open is unbounded
After cooldown, `allow()` changes to half_open and permits every concurrent call with no probe lease or re-open-on-probe semantics.

---

## Batch totals

- Findings: **44**
- Critical: **2**
- High: **28**
- Medium: **14**
- Low: **0**

Repository-wide cumulative totals are intentionally omitted until duplicate reconciliation is completed.

## Files materially reviewed in this batch

`common/runtime.py`.
