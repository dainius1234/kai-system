# Kai Code Audit — Operational State, Episode Persistence and Recovery Gates

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Public Agentic checkpoint endpoint findings already recorded in `CODE_AUDIT_BATCH_AGENTIC_API.md` are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-STATE-001 | HIGH | Operational FSM state is process-local and lost on restart |
| KAI-STATE-002 | HIGH | `HEAL_COMPLETE` returns to IDLE without verifying recovery success |
| KAI-STATE-003 | HIGH | Any `SERVICE_RESTORED` event can clear DEGRADED state without a fleet-health check |
| KAI-STATE-004 | HIGH | State-transition events carry no authenticated source or supporting evidence |
| KAI-STATE-005 | MEDIUM | Restoring a service loses the prior FOCUSED state |
| KAI-STATE-006 | MEDIUM | Restoring a service loses the prior ACTIVE session state |
| KAI-STATE-007 | HIGH | Critical anomalies are silently ignored in FOCUSED, DEGRADED and RECOVERING states |
| KAI-STATE-008 | HIGH | Additional service-down events are silently ignored while DEGRADED or RECOVERING |
| KAI-STATE-009 | MEDIUM | User activity is silently ignored while DEGRADED or RECOVERING |
| KAI-STATE-010 | MEDIUM | Session-end events are silently ignored outside ACTIVE |
| KAI-STATE-011 | MEDIUM | FOCUS_EXIT always returns to IDLE rather than the pre-focus state |
| KAI-STATE-012 | HIGH | Undefined transitions are neither rejected nor audited |
| KAI-STATE-013 | HIGH | FSM history lacks event IDs, actor identity, reasons and evidence references |
| KAI-STATE-014 | HIGH | DEGRADED and RECOVERING states have no dwell timeout or escalation policy |
| KAI-STATE-015 | HIGH | State transitions are not coordinated with the side effect they claim occurred |
| KAI-STATE-016 | MEDIUM | The “thread-safe” claim applies only to one asyncio event loop |
| KAI-STATE-017 | MEDIUM | State and snapshot reads occur without the transition lock |
| KAI-STATE-018 | MEDIUM | Transition history has no timestamps and excludes ignored events |
| KAI-STATE-019 | HIGH | Every new worker starts IDLE even when the system is actively degraded |
| KAI-STATE-020 | HIGH | Operational state is not linked to dependency freshness or health generations |
| KAI-STATE-021 | CRITICAL | Redis episode decay permanently deletes all records beyond the first 1,001 entries |
| KAI-STATE-022 | HIGH | Redis decay is a non-atomic destructive read-modify-write operation |
| KAI-STATE-023 | HIGH | Archive writes can be duplicated or partially committed before active-list replacement |
| KAI-STATE-024 | HIGH | Redis decay reverses the retained episode ordering |
| KAI-STATE-025 | HIGH | Redis episode lists and archives have no retention or size cap |
| KAI-STATE-026 | MEDIUM | Recall inspects only the first 201 records and silently omits older in-window episodes |
| KAI-STATE-027 | HIGH | One malformed Redis episode aborts recall or decay |
| KAI-STATE-028 | HIGH | Redis failure silently changes the system to a local spool persistence model |
| KAI-STATE-029 | HIGH | Fallback episode data is written as plaintext to a shared `/tmp` path |
| KAI-STATE-030 | HIGH | Spool checksums are unkeyed and do not prevent malicious tampering |
| KAI-STATE-031 | HIGH | Corrupt or invalid spool lines are silently discarded during replay |
| KAI-STATE-032 | HIGH | Spool rotation is non-atomic and races concurrent writers |
| KAI-STATE-033 | MEDIUM | Same-second spool rotations can overwrite an existing archive |
| KAI-STATE-034 | MEDIUM | Spool loading and rotation read the complete file into memory |
| KAI-STATE-035 | MEDIUM | A single oversized episode can exceed the spool limit until a later write |
| KAI-STATE-036 | MEDIUM | Spool size and storage-path configuration are not safely validated |
| KAI-STATE-037 | MEDIUM | In-memory episode and archive collections are unbounded |
| KAI-STATE-038 | MEDIUM | In-memory saving performs only a shallow copy of caller-owned episode data |
| KAI-STATE-039 | HIGH | Fallback and in-memory episode state diverges across workers |
| KAI-STATE-040 | MEDIUM | Caller-controlled user IDs create unrestricted Redis/storage namespaces |
| KAI-STATE-041 | HIGH | The recursive self-improvement gate returns advice but enforces no rollback or block |
| KAI-STATE-042 | HIGH | Empty before/after datasets are approved as a neutral self-modification |
| KAI-STATE-043 | HIGH | Performance snapshots trust self-reported episode conviction and outcomes |
| KAI-STATE-044 | HIGH | Higher average conviction is treated as improvement independently of correctness |
| KAI-STATE-045 | HIGH | Improvement approval requires no minimum sample size or matched evaluation cohort |
| KAI-STATE-046 | HIGH | Performance snapshots are not bound to a code/configuration change identity |
| KAI-STATE-047 | HIGH | Improvement evidence is stored unsigned in mutable `/tmp` JSON |
| KAI-STATE-048 | HIGH | Snapshot persistence and loading failures are silently suppressed |
| KAI-STATE-049 | MEDIUM | Snapshot read-modify-write persistence is non-atomic and concurrency-unsafe |
| KAI-STATE-050 | MEDIUM | Snapshot metrics and tolerance accept invalid, non-finite and out-of-range values |

---

## Operational FSM: `agentic/system_fsm.py`

### KAI-STATE-001 — HIGH — Operational state is volatile and worker-local
**Issue:** the singleton starts in IDLE and stores state/history only in process memory. No persistence, shared authority or recovery replay exists.  
**Risk:** restart erases DEGRADED/RECOVERING/FOCUSED state, and multiple workers can simultaneously report different operational states.  
**Recommendation:** use one durable versioned state authority with explicit startup reconciliation.  
**Status:** OPEN

### KAI-STATE-002 — HIGH — Healing completion is not verified
**Issue:** `RECOVERING + HEAL_COMPLETE` transitions directly to IDLE. The event contains no health result, operation ID or proof that recovery succeeded.  
**Risk:** any caller can declare healing complete and clear degraded containment while dependencies remain failed.  
**Recommendation:** accept only an authenticated recovery result tied to fresh required-service checks.  
**Status:** OPEN

### KAI-STATE-003 — HIGH — One restoration event clears the global outage state
**Issue:** `DEGRADED + SERVICE_RESTORED` and `RECOVERING + SERVICE_RESTORED` both transition to IDLE without identifying the service or checking whether other critical services remain down.  
**Risk:** restoration of one component—or a fabricated event—clears system-wide degradation.  
**Recommendation:** derive state from a complete versioned critical-dependency set.  
**Status:** OPEN

### KAI-STATE-004 — HIGH — Event authority is absent
**Issue:** `fire()` accepts only a `KaiEvent`. It has no actor, service identity, signature, request ID, reason or observation payload.  
**Risk:** any imported code can generate authoritative state changes with no attribution or evidence.  
**Recommendation:** require authenticated typed events from approved producers.  
**Status:** OPEN

### KAI-STATE-005 — MEDIUM — Focus context is lost after degradation
FOCUSED transitions to DEGRADED on service failure, then restoration returns IDLE rather than FOCUSED.

### KAI-STATE-006 — MEDIUM — Active session context is lost after degradation
ACTIVE transitions to DEGRADED, and restoration returns IDLE even if the user session remains active.

### KAI-STATE-007 — HIGH — Critical anomalies disappear in several states
**Issue:** `ANOMALY_CRITICAL` is defined only from IDLE and ACTIVE. In FOCUSED, DEGRADED and RECOVERING it is silently ignored.  
**Risk:** a new critical anomaly can be omitted during precisely the states requiring heightened attention.  
**Recommendation:** record and escalate every critical event regardless of current state.  
**Status:** OPEN

### KAI-STATE-008 — HIGH — Concurrent outages are not represented
**Issue:** `SERVICE_DOWN` has no transition from DEGRADED or RECOVERING and is silently ignored.  
**Risk:** additional failed services are not captured, and one later restoration can falsely clear the state.  
**Recommendation:** maintain a set of active incidents rather than one scalar state.  
**Status:** OPEN

### KAI-STATE-009 — MEDIUM — User activity vanishes during incidents
`USER_MESSAGE` has no transition/history entry from DEGRADED or RECOVERING, so active interaction is not represented.

### KAI-STATE-010 — MEDIUM — Session termination is state-specific
`SESSION_END` is recognised only from ACTIVE; it is ignored from FOCUSED, DEGRADED and RECOVERING.

### KAI-STATE-011 — MEDIUM — Focus exit has no state restoration
The FSM does not remember whether focus was entered from IDLE or ACTIVE; every exit returns IDLE.

### KAI-STATE-012 — HIGH — Invalid transitions fail silently
**Issue:** undefined `(state,event)` pairs return `None` without warning, metric, history record or caller-visible rejection reason.  
**Risk:** control/event delivery defects are indistinguishable from successful no-op handling.  
**Recommendation:** return a typed rejected transition and audit it.  
**Status:** OPEN

### KAI-STATE-013 — HIGH — Transition evidence is unauditable
**Issue:** history contains only `(event, previous, next)` strings and is capped at 100.  
**Risk:** state changes cannot be attributed, correlated with recovery work or reconstructed after incidents.  
**Recommendation:** append immutable timestamped event records with actor, cause and correlation ID.  
**Status:** OPEN

### KAI-STATE-014 — HIGH — Incident states can persist forever
No recovery deadline, stale-event timeout, retry counter or mandatory operator escalation exists for DEGRADED/RECOVERING.

### KAI-STATE-015 — HIGH — State and operational action are decoupled
**Issue:** the FSM performs no healing, health verification, focus-mode application or acknowledgement. A transition can succeed even when the represented side effect never occurred.  
**Risk:** state labels become assertions rather than verified operational truth.  
**Recommendation:** transition only after durable side-effect acknowledgement or enter an intermediate pending state.  
**Status:** OPEN

### KAI-STATE-016 — MEDIUM — Concurrency guarantee is overstated
The asyncio lock protects tasks sharing one event loop; it does not protect threads, processes or replicas.

### KAI-STATE-017 — MEDIUM — Reads are not synchronised
The `state` property and `snapshot()` access mutable state/history without acquiring the lock.

### KAI-STATE-018 — MEDIUM — History omits essential chronology
Transition entries have no timestamps/duration and ignored/failed events are absent.

### KAI-STATE-019 — HIGH — Worker startup creates false IDLE state
**Issue:** every imported singleton starts IDLE regardless of persisted dependency incidents or another worker’s state.  
**Risk:** load balancing can return a healthy-looking IDLE snapshot while another worker is DEGRADED.  
**Recommendation:** initialise from one authoritative incident generation.  
**Status:** OPEN

### KAI-STATE-020 — HIGH — State is not freshness-bound
Operational state is not linked to the timestamp/generation of watchdog, health or recovery evidence, so it can remain IDLE after observations expire.

---

## Episode savers: `agentic/kai_config.py`

### KAI-STATE-021 — CRITICAL — Decay truncates large Redis histories
**Issue:** `RedisSaver.decay()` reads indices 0–1000, then deletes the complete active key and rewrites only retained records from that limited slice. Every record beyond index 1000 is permanently discarded, regardless of age or outcome.  
**Risk:** running maintenance against a user with more than 1,001 episodes destroys unexamined history.  
**Recommendation:** use an atomic paginated migration that never deletes records outside the evaluated set.  
**Status:** OPEN — immediate remediation required

### KAI-STATE-022 — HIGH — Destructive decay races writers
**Issue:** decay separately reads, archives, deletes and rewrites the list without a lock, transaction or revision check.  
**Risk:** episodes saved during decay can be deleted, and overlapping decays lose/duplicate records.  
**Recommendation:** perform one server-side atomic operation or transactional job with immutable IDs.  
**Status:** OPEN

### KAI-STATE-023 — HIGH — Archive mutation is partially committed
**Issue:** qualifying records are pushed to the archive before the active-list replacement pipeline executes. A failure or retry can leave duplicates in both stores or repeatedly archive the same episode.  
**Risk:** retention and learning counts become incorrect while the active list remains unchanged or partially replaced.  
**Recommendation:** atomically move uniquely identified records.  
**Status:** OPEN

### KAI-STATE-024 — HIGH — Retained episode order is reversed
**Issue:** active records are read newest-first, but `rpush(*reversed(kept))` rewrites them oldest-first. Subsequent `lpush` creates mixed ordering.  
**Risk:** recall recency and planning history are corrupted.  
**Recommendation:** preserve a defined timestamp/index ordering.  
**Status:** OPEN

### KAI-STATE-025 — HIGH — Redis growth is unbounded
`save_episode` performs unlimited `LPUSH`; active and archive keys have no max length, TTL, byte quota or per-principal retention.

### KAI-STATE-026 — MEDIUM — Recall is silently incomplete
Only indices 0–200 are read, so valid episodes within the requested number of days can be omitted without a truncation indicator.

### KAI-STATE-027 — HIGH — One bad record disables history operations
**Issue:** JSON and numeric conversions in Redis recall/decay are not isolated per record.  
**Risk:** one malformed episode aborts the whole request/maintenance operation.  
**Recommendation:** validate on ingestion and quarantine corrupt records with explicit partial-state reporting.  
**Status:** OPEN

### KAI-STATE-028 — HIGH — Storage semantics silently change on Redis failure
**Issue:** any Redis construction, authentication or ping exception causes `build_saver()` to return a local ChecksummedSpoolSaver with no degraded status.  
**Risk:** a distributed durable store silently becomes node-local `/tmp` state; workers diverge and restart may erase data while the application reports normal operation.  
**Recommendation:** fail readiness or expose an explicit approved degraded storage mode.  
**Status:** OPEN

### KAI-STATE-029 — HIGH — Sensitive episode data is plaintext in `/tmp`
The fallback spool stores complete payloads—including prompts, outputs and metadata—under a shared temporary path with ordinary filesystem permissions and no encryption.

### KAI-STATE-030 — HIGH — Checksum is not authenticity protection
SHA-256 is stored beside the payload with no secret/signature. Anyone able to edit the file can recompute a valid checksum for poisoned episodes.

### KAI-STATE-031 — HIGH — Replay hides corruption
Checksum mismatches, malformed JSON and invalid structures are silently skipped, producing a shortened apparently valid history with no integrity/freshness state.

### KAI-STATE-032 — HIGH — Rotation is unsafe
Rotation reads, archives and rewrites the spool directly without locking, temporary files, atomic replacement or fsync for rotated outputs. Concurrent appends can be lost or interleaved.

### KAI-STATE-033 — MEDIUM — Archive names collide
Archive filenames use whole-second timestamps; multiple rotations in one second target the same path and overwrite prior content.

### KAI-STATE-034 — MEDIUM — Whole-file memory allocation
Spool startup and rotation call `read_text().splitlines()` on the complete file.

### KAI-STATE-035 — MEDIUM — Size limit is delayed
Rotation occurs before the new line is appended. A single very large episode can exceed the limit and remains until a later save triggers rotation.

### KAI-STATE-036 — MEDIUM — Unsafe spool configuration
Negative/extreme size values and arbitrary storage paths are accepted; a negative cap causes repeated rotation behaviour.

### KAI-STATE-037 — MEDIUM — In-memory stores grow indefinitely
Both active and archive dictionaries retain unlimited episode dictionaries with no byte/count policy.

### KAI-STATE-038 — MEDIUM — Nested caller data remains shared
`dict(payload)` copies only the top-level dictionary; nested structures can be mutated later by the caller and silently change stored in-memory episodes.

### KAI-STATE-039 — HIGH — Worker histories diverge
Each process has its own in-memory buckets. Multiple fallback workers also concurrently access the same spool without coordination while serving different loaded snapshots.

### KAI-STATE-040 — MEDIUM — Namespace creation is unrestricted
Arbitrary user IDs become Redis key suffixes and in-memory dictionary keys with no authentication, length, character or namespace quota.

---

## Recursive self-improvement gate: `agentic/kai_config.py`

### KAI-STATE-041 — HIGH — Verdict is non-enforcing
**Issue:** `evaluate_improvement()` only returns `approved` and recommendation text. It does not control a change, restore a checkpoint or require callers to comply.  
**Risk:** architecture can claim automatic rollback protection while degraded changes remain active.  
**Recommendation:** place the comparison in an authoritative transactional deployment gate.  
**Status:** OPEN

### KAI-STATE-042 — HIGH — No-data changes are approved
**Issue:** empty episode sets produce all-zero snapshots; comparing them yields no degraded metrics and `approved=True` with a neutral recommendation.  
**Risk:** a self-modification can pass without any before/after evaluation evidence.  
**Recommendation:** require a minimum representative test corpus and return insufficient-evidence.  
**Status:** OPEN

### KAI-STATE-043 — HIGH — Evaluation data is self-reported
**Issue:** snapshots average caller-supplied episode conviction, outcome and rethink values without authenticated evaluator provenance.  
**Risk:** poisoned/generated episodes make a harmful change appear improved.  
**Recommendation:** use immutable independently scored benchmark outcomes.  
**Status:** OPEN

### KAI-STATE-044 — HIGH — Confidence inflation is rewarded
Higher average conviction is classified as improvement even if objective outcomes are unchanged or worse within tolerance.  
**Risk:** changes that make the model more confident earn approval without becoming more correct.  
**Recommendation:** remove self-confidence as a positive quality metric unless calibrated against verified accuracy.  
**Status:** OPEN

### KAI-STATE-045 — HIGH — Cohorts are incomparable
No minimum episode count, task matching, held-out set, statistical significance or before/after sample-size equality is required.

### KAI-STATE-046 — HIGH — Snapshot is not tied to the modification
Snapshots contain no code digest, configuration revision, model identity, test-set ID or change operation ID.

### KAI-STATE-047 — HIGH — Improvement evidence is unauthenticated temporary data
The snapshot file defaults to `/tmp`, is plaintext/unsigned, and can be edited or replaced before comparison.

### KAI-STATE-048 — HIGH — Evidence failure disappears
Snapshot save errors are suppressed; load errors return `None`. No health/readiness or audit event records that the claimed protection lacked evidence.

### KAI-STATE-049 — MEDIUM — Snapshot writes race
The complete JSON list is read-modify-written directly with no lock, atomic rename or version check.

### KAI-STATE-050 — MEDIUM — Numerical policy is unvalidated
Episode metrics, counts and tolerance accept NaN, infinity, negative or out-of-range values. NaN comparisons can leave `degraded` empty and approve the change.

---

## Batch totals

- Findings: **50**
- Critical: **1**
- High: **31**
- Medium: **18**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,223**
- Critical: **100**
- High: **514**
- Medium: **606**
- Low: **3**

## Files materially reviewed in this batch

`agentic/system_fsm.py` and the episode-storage/self-improvement sections of `agentic/kai_config.py`, with checkpoint endpoint overlap reconciled against `CODE_AUDIT_BATCH_AGENTIC_API.md`.
