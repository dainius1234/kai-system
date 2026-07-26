# Kai Code Audit Master Register

Repository: `dainius1234/kai-system`  
Status: ACTIVE — SINGLE SOURCE OF TRUTH  
Last updated: 26 July 2026  
Audit method: file-by-file review from core execution paths outward

This is the definitive audit register. Earlier registers are retained only as historical working records:

- `kai-pm/CODE_AUDIT_REGISTER.md`
- `kai-pm/CODE_AUDIT_REGISTER_CONTINUED.md`
- `kai-pm/CODE_AUDIT_REGISTER_CONTINUED_2.md`

All future findings, status changes and remediation notes must be recorded here.

## Severity scale

- **CRITICAL** — credible compromise, destructive operation or major integrity risk
- **HIGH** — serious correctness, reliability, security or production-readiness risk
- **MEDIUM** — material defect, scalability issue or maintainability risk
- **LOW** — limited defect or standards issue

---

## Consolidated findings index — 1 to 91

| ID | Severity | Finding |
|---|---|---|
| KAI-CORE-001 | CRITICAL | Privileged mutation endpoints appear unauthenticated |
| KAI-CORE-002 | CRITICAL | SSRF through Web Scout |
| KAI-CORE-003 | HIGH | Context-budget enforcement can exceed its own limit |
| KAI-CORE-004 | HIGH | Non-atomic writes to identity files |
| KAI-CORE-005 | HIGH | Repeated `httpx.AsyncClient` creation |
| KAI-CORE-006 | HIGH | Silent broad exception handling |
| KAI-CORE-007 | MEDIUM | Process-local mutable state is not multi-worker safe |
| KAI-CORE-008 | MEDIUM | Missing request size and value constraints |
| KAI-CORE-009 | MEDIUM | Internal exception details may leak to API callers |
| KAI-CORE-010 | LOW | Timezone-naive UTC timestamps |
| KAI-WEB-001 | CRITICAL | Trust control fails open |
| KAI-WEB-002 | HIGH | Response body is downloaded without a byte limit |
| KAI-WEB-003 | MEDIUM | Error responses expose raw network exception text |
| KAI-AUTH-001 | CRITICAL | HMAC does not bind the full request |
| KAI-AUTH-002 | CRITICAL | Idempotency lookup occurs before authentication and validation |
| KAI-AUTH-003 | HIGH | Nonce persistence is non-atomic and concurrency-unsafe |
| KAI-AUTH-004 | HIGH | Co-sign is represented as an untrusted request boolean |
| KAI-RUN-001 | HIGH | Redis audit hash-chain append is non-atomic |
| KAI-RUN-002 | MEDIUM | Structured logger does not emit reliably valid JSON |
| KAI-RUN-003 | MEDIUM | Audit verification performs an unbounded full-stream startup scan |
| KAI-RUN-004 | MEDIUM | Error-budget calculation omits most failures |
| KAI-RES-001 | HIGH | All HTTP responses below 500 are treated as successful |
| KAI-RES-002 | MEDIUM | Deep health checks run sequentially without deadlines |
| KAI-RES-003 | MEDIUM | Circuit-breaker state is concurrency-unsafe |
| KAI-RES-004 | HIGH | Healing engine records unverified `auto_recovery` as a known fix |
| KAI-LLM-001 | MEDIUM | Streaming and non-streaming model availability behaviour diverges |
| KAI-LLM-002 | HIGH | Transport failures are converted into model-like text |
| KAI-SWARM-001 | HIGH | Reputation persistence is non-atomic and multi-worker unsafe |
| KAI-SWARM-002 | HIGH | Conviction score rewards evidence quantity rather than quality |
| KAI-SWARM-003 | MEDIUM | Reputation is trained from self-reported confidence |
| KAI-STAGE-001 | HIGH | Untrusted retrieved content is inserted directly into agent prompts |
| KAI-STAGE-002 | HIGH | Adversary failure becomes successful conviction-gate completion |
| KAI-STAGE-003 | HIGH | Moral-imagination safety stage fails open silently |
| KAI-STAGE-004 | MEDIUM | JSON parse failure can be recorded as successful stage execution |
| KAI-FSM-001 | HIGH | Failed fact-check reruns gathering but skips revalidation |
| KAI-FSM-002 | HIGH | Unexpected stage exceptions escape the FSM |
| KAI-FSM-003 | MEDIUM | `AgentHandoff` claims schema is not runtime validated |
| KAI-TRUST-001 | CRITICAL | Shared autonomy gate fails open when governance is unavailable |
| KAI-TRUST-002 | HIGH | Low moral alignment is warning-only rather than enforcement |
| KAI-TRUST-003 | HIGH | Trust increases from self-declared model confidence |
| KAI-TRUST-004 | MEDIUM | Trust-ledger recording is synchronous despite nonblocking claims |
| KAI-TRUST-005 | CRITICAL | Trust level and evidence persistence are tamperable local files |
| KAI-TRUST-006 | HIGH | Trust mutations and capability checks are concurrency-unsafe |
| KAI-TRUST-007 | HIGH | Auto-promotion accepts unbounded caller-supplied evidence |
| KAI-TRUST-008 | MEDIUM | Corrupt trust state resets without preserving forensic evidence |
| KAI-ROUTE-001 | HIGH | Route priority can misroute consequential actions |
| KAI-ROUTE-002 | HIGH | Semantic model may download and initialise on a live request |
| KAI-ROUTE-003 | MEDIUM | Route-anchor embeddings are recomputed on every request |
| KAI-ROUTE-004 | MEDIUM | Blocking embedding inference executes synchronously |
| KAI-ROUTE-005 | MEDIUM | Silent semantic-classifier failures hide degradation |
| KAI-SUP-001 | CRITICAL | Manual recovery endpoint is unauthenticated |
| KAI-SUP-002 | HIGH | Due items may be marked fired without confirmed delivery |
| KAI-SUP-003 | HIGH | Reminder and task delivery is not transactionally idempotent |
| KAI-SUP-004 | HIGH | Background loop task is not retained, supervised or restarted |
| KAI-SUP-005 | MEDIUM | Watchdog can report healthy before first heartbeat |
| KAI-SUP-006 | MEDIUM | Watchdog conflates long work with a frozen loop |
| KAI-SUP-007 | MEDIUM | Recovery success is accepted without post-recovery verification |
| KAI-SUP-008 | MEDIUM | Recovery cooldown is consumed before outcome is known |
| KAI-SUP-009 | MEDIUM | Predictive trend uses current breaker state as historical data |
| KAI-SUP-010 | MEDIUM | On-demand sweep can race background sweep and recovery |
| KAI-SUP-011 | MEDIUM | Proactive mode lookup fails open to `PUB` |
| KAI-MEMU-001 | HIGH | Memory verification policy fails open when policy loading fails |
| KAI-MEMU-002 | HIGH | Missing LakeFS dependency silently installs a non-durable stub |
| KAI-MEMU-003 | MEDIUM | Graph fan-out ignores unsuccessful HTTP responses |
| KAI-MEMU-004 | MEDIUM | Core memory request models lack bounded validation |
| KAI-VER-001 | CRITICAL | Caller-supplied evidence can forge a PASS verdict |
| KAI-VER-002 | HIGH | Duplicate evidence can inflate verification confidence |
| KAI-VER-003 | MEDIUM | Any sufficiently long context improves plausibility |
| KAI-VER-004 | MEDIUM | Health reports healthy without checking dependencies |
| KAI-VER-005 | MEDIUM | Verdict counters are process-local and concurrency-unsafe |
| KAI-EXEC-001 | CRITICAL | Shell allowlist permits arbitrary code and host-control operations |
| KAI-EXEC-002 | CRITICAL | Python expression sandbox can reach imported module state |
| KAI-EXEC-003 | CRITICAL | Execution endpoint lacks authentication and Tool Gate proof |
| KAI-EXEC-004 | HIGH | Subprocess output is fully buffered before truncation |
| KAI-EXEC-005 | HIGH | Malware scanning fails open when unavailable or errored |
| KAI-EXEC-006 | HIGH | State rollback does not roll back execution effects |
| KAI-EXEC-007 | MEDIUM | Execution history exposes raw parameters and is process-local |
| KAI-EXEC-008 | MEDIUM | Internal execution errors and stderr leak to callers |
| KAI-EXEC-009 | MEDIUM | Executor inputs and history limits lack bounded validation |
| KAI-TLED-001 | CRITICAL | Trust-ledger mutation and acknowledgement endpoints are unauthenticated |
| KAI-TLED-002 | CRITICAL | Trust ledger uses a predictable default HMAC secret |
| KAI-TLED-003 | HIGH | Trust-critical event fields are excluded from the signature |
| KAI-TLED-004 | CRITICAL | Replay skips corrupt events and can report the filtered chain as intact |
| KAI-TLED-005 | HIGH | Operator acknowledgements are neither persisted nor cryptographically bound |
| KAI-TLED-006 | HIGH | File-ledger append is non-atomic and concurrency-unsafe |
| KAI-TLED-007 | MEDIUM | Merkle publication is mutable, local and non-atomic |
| KAI-LWORK-001 | HIGH | Ledger archives silently contain only the latest 10,000 entries |
| KAI-LWORK-002 | MEDIUM | Ledger-worker operational endpoints lack visible authentication |
| KAI-LWORK-003 | MEDIUM | Heartbeat notifications ignore unsuccessful HTTP responses |

The detailed Issue / Risk / Recommendation evidence for findings 1–63 remains in the three historical registers and will be migrated into this master during the final editorial pass. Findings 64 onward are detailed below.

---

## Memory core: `memu-core/app.py`

### KAI-MEMU-001 — HIGH — Memory verification policy fails open when policy loading fails

**Issue:** Importing or reading `common.policy` is wrapped in a broad exception handler that sets `REQUIRE_VERDICT_PASS = False` and `LOG_ONLY_MODE = False`.

**Risk:** A missing module, malformed policy, configuration error or startup defect silently disables mandatory verifier gating. Unverified or poisoned information can enter persistent memory precisely when the control is unavailable.

**Recommendation:** Fail closed when verification policy cannot be loaded. Expose a not-ready state, emit a critical configuration event and require a separately controlled emergency override.

**Status:** OPEN

### KAI-MEMU-002 — HIGH — Missing LakeFS dependency silently replaces durable versioning with a non-durable stub

**Issue:** Any exception importing the LakeFS client installs an in-memory replacement. The stub stores commits only in a process list and implements `revert()` as a no-op.

**Risk:** The service may report commit identifiers and rollback behaviour while providing no durable history or actual rollback.

**Recommendation:** Treat the durable version-store dependency as required whenever versioning is enabled. Fail readiness or explicitly expose versioning as disabled; never emulate successful rollback with a no-op.

**Status:** OPEN

### KAI-MEMU-003 — MEDIUM — Graph ingest and forget fan-out ignore unsuccessful HTTP responses

**Issue:** Graph fan-out awaits POST requests but does not inspect status codes or call `raise_for_status()`.

**Risk:** HTTP 4xx and 5xx responses are treated as completed operations, allowing vector and graph memory to diverge silently.

**Recommendation:** Validate accepted status codes, emit structured failure metrics and use a durable idempotent retry/outbox mechanism.

**Status:** OPEN

### KAI-MEMU-004 — MEDIUM — Core memory request models lack bounded validation

**Issue:** Queries, session IDs, notes, event data, scores and embeddings use unconstrained fields.

**Risk:** Oversized or malformed payloads can consume resources or corrupt ranking and persistence logic.

**Recommendation:** Add strict lengths, finite numeric ranges, timestamp parsing, nested-object limits and an application-wide body cap.

**Status:** OPEN

---

## Verifier: `verifier/app.py`

### KAI-VER-001 — CRITICAL — Caller-supplied evidence can forge a PASS verdict

**Issue:** `/verify` accepts an arbitrary `evidence_pack` and uses caller-controlled rank, relevance and importance scores directly.

**Risk:** A caller can fabricate high-scoring evidence and obtain a PASS verdict from the service intended to authorise memory promotion and tool execution.

**Recommendation:** Resolve immutable evidence IDs server-side or require signed evidence packs with issuer, digest, freshness and chain-of-custody validation.

**Status:** OPEN — immediate remediation required

### KAI-VER-002 — HIGH — Evidence quantity and duplicates can inflate verification confidence

**Issue:** Support and strong-chunk counts do not check semantic duplication, source independence or circular provenance.

**Risk:** Repeated copies of one assertion can satisfy corroboration thresholds without independent evidence.

**Recommendation:** Deduplicate semantically, group by source lineage and count corroboration only across independent trusted sources.

**Status:** OPEN

### KAI-VER-003 — MEDIUM — Any sufficiently long context automatically improves plausibility

**Issue:** Plausibility increases whenever context exceeds 20 characters, regardless of relevance or contradiction.

**Risk:** Irrelevant filler can increase the aggregate score.

**Recommendation:** Remove unconditional bonuses and score context only through evidence-grounded entailment and contradiction analysis.

**Status:** OPEN

### KAI-VER-004 — MEDIUM — Health endpoint reports healthy without checking verification dependencies

**Issue:** `/health` does not test Memu availability or policy coherence.

**Risk:** Safety-critical traffic may be routed to an instance unable to verify evidence.

**Recommendation:** Separate liveness and readiness and validate required dependencies with bounded deadlines.

**Status:** OPEN

### KAI-VER-005 — MEDIUM — Verdict counters are process-local and concurrency-unsafe

**Issue:** Verdict telemetry uses an unsynchronised process dictionary.

**Risk:** Multi-worker metrics are incomplete and divergent.

**Recommendation:** Use atomic counters in a proper metrics backend.

**Status:** OPEN

---

## Executor: `executor/app.py`

### KAI-EXEC-001 — CRITICAL — Shell allowlist permits arbitrary code and host-control operations

**Issue:** The allowlist includes `python3`, `pip`, `git`, `make`, `docker` and `curl`. Checking only the first executable and using `shell=False` does not constrain these programs.

**Risk:** A caller can execute arbitrary code, install packages, control containers or exfiltrate data. Docker socket or sensitive mounts could turn this into host compromise.

**Recommendation:** Remove general-purpose interpreters and control clients. Implement typed operations in disposable hardened sandboxes with read-only filesystems and explicit egress controls.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-002 — CRITICAL — Python expression sandbox can reach imported module state

**Issue:** The wrapper imports modules and evaluates the caller expression in its global namespace. AST checks permit ordinary attribute and subscript access to imported module state.

**Risk:** Crafted expressions can escape the intended restricted namespace. This is not a security sandbox.

**Recommendation:** Remove arbitrary Python evaluation. Interpret a strict pure-expression AST or use a disposable hardened code sandbox.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-003 — CRITICAL — Execution endpoint lacks authentication and proof of Tool Gate approval

**Issue:** `/execute` accepts execution requests without validating caller identity, a signed gate decision, request digest, nonce, expiry or policy verdict.

**Risk:** Any reachable caller can bypass Tool Gate and invoke execution directly.

**Recommendation:** Require mutual service authentication and a short-lived replay-protected capability signed over the complete canonical request.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-004 — HIGH — Subprocess output is fully buffered before truncation

**Issue:** `capture_output=True` buffers stdout and stderr completely; the configured cap is applied only after process completion and only to stdout.

**Risk:** A subprocess can exhaust executor memory.

**Recommendation:** Stream through bounded pipes and enforce OS-level CPU, memory, process and file-size limits.

**Status:** OPEN

### KAI-EXEC-005 — HIGH — Malware scanning fails open when unavailable or errored

**Issue:** Missing ClamAV is represented as clean, while only scanner return code 1 blocks execution.

**Risk:** Production may claim scanning while executing with no scanner or after scanner failure.

**Recommendation:** Fail closed where scanning is required and expose scanner state, version and signature age in readiness and policy context.

**Status:** OPEN

### KAI-EXEC-006 — HIGH — State rollback does not roll back execution effects

**Issue:** Rollback merely pops request metadata from a process list and does not reverse filesystem, network, container or repository effects.

**Risk:** Failed actions can leave persistent changes while the system implies rollback occurred.

**Recommendation:** Rename this as history bookkeeping. Use transactional adapters, snapshots or disposable environments for genuinely reversible actions.

**Status:** OPEN

### KAI-EXEC-007 — MEDIUM — Execution history exposes raw parameters and is process-local

**Issue:** `/history` returns raw commands, arguments and expressions without visible authentication.

**Risk:** Secrets may be disclosed and history is incomplete across workers or restarts.

**Recommendation:** Restrict access, redact structurally and store immutable records centrally.

**Status:** OPEN

### KAI-EXEC-008 — MEDIUM — Internal execution errors and subprocess stderr leak to callers

**Issue:** Generic exceptions and raw stderr are included in API responses.

**Risk:** Filesystem paths, versions and internal behaviour may be exposed.

**Recommendation:** Return stable error codes and trace IDs; retain details only in access-controlled logs.

**Status:** OPEN

### KAI-EXEC-009 — MEDIUM — Executor inputs and history limits lack bounded validation

**Issue:** Tool fields, nested parameters, commands, expressions and history limits are not consistently bounded.

**Risk:** Oversized payloads can increase parsing, memory and logging load.

**Recommendation:** Add strict schema constraints and application-wide body limits.

**Status:** OPEN

---

## Trust ledger: `trust-ledger/app.py` and `trust-ledger/ledger.py`

### KAI-TLED-001 — CRITICAL — Trust-ledger mutation and acknowledgement endpoints are unauthenticated

**Issue:** Despite the module header describing HMAC-authenticated writes and operator reads, no authentication middleware or dependency protects `POST /trust/event`, `POST /trust/alignment-audit` or `PATCH /trust/events/{event_id}/ack`.

**Risk:** Any reachable caller can create GRANT, REVOKE, OVERRIDE, autonomous-action or alignment events and can falsely acknowledge them as the operator. Trust scores and governance decisions can be manipulated directly.

**Recommendation:** Require mutually authenticated service identities and role-scoped authorisation. Bind each write to a canonical signed request with timestamp, nonce, expiry and idempotency key. Require a separate operator credential for acknowledgements.

**Status:** OPEN — immediate remediation required

### KAI-TLED-002 — CRITICAL — Trust ledger uses a predictable default HMAC secret

**Issue:** `TRUST_LEDGER_HMAC_SECRET` defaults to the literal `trust-dev-secret` when unset.

**Risk:** Any deployment missing the environment variable uses a publicly knowable signing key. An attacker with file access can forge events and recompute a valid chain.

**Recommendation:** Refuse startup without a strong secret supplied by a secret manager. Support key identifiers and rotation, and alert if a development key is detected.

**Status:** OPEN — immediate remediation required

### KAI-TLED-003 — HIGH — Trust-critical event fields are excluded from the signature

**Issue:** Event HMACs cover only event ID, timestamp, event type, initiator and `event_data`. They exclude `capability`, `trust_tier`, `previous_hash`, `operator_ack` and `operator_note`.

**Risk:** Capability and trust-tier values can be altered in the JSONL file without invalidating the event signature. A record can therefore retain a valid cryptographic appearance while changing its governance meaning.

**Recommendation:** Sign a canonical serialisation of every immutable field, including chain predecessor and schema version. Represent acknowledgements as separate signed events rather than mutable fields.

**Status:** OPEN

### KAI-TLED-004 — CRITICAL — Replay skips corrupt events and can report the filtered chain as intact

**Issue:** `_replay()` skips malformed or integrity-mismatched lines and continues from the last accepted signature. `verify_chain()` then checks only the filtered in-memory event list.

**Risk:** Deleted, altered or corrupt records can disappear from the reconstructed ledger, after which health and integrity endpoints may report the remaining chain as intact. This defeats detection of omission and truncation attacks.

**Recommendation:** Halt replay at the first malformed or mismatched record, preserve forensic details and mark readiness failed. Verify the physical record sequence, line count and an externally anchored checkpoint before accepting new writes.

**Status:** OPEN — immediate remediation required

### KAI-TLED-005 — HIGH — Operator acknowledgements are neither persisted nor cryptographically bound

**Issue:** `ack()` mutates only the in-memory event object. It does not append a new record, rewrite durable storage or update a signature.

**Risk:** Acknowledgements disappear after restart and cannot be independently verified. Runtime responses and trust-score calculations may differ from the durable ledger.

**Recommendation:** Model acknowledgement as a new append-only signed event referencing the original event ID, authenticated operator and note digest.

**Status:** OPEN

### KAI-TLED-006 — HIGH — File-ledger append is non-atomic and concurrency-unsafe

**Issue:** The service derives the predecessor from a mutable list, appends to that list and writes one JSON line without a lock, file lock, transaction, flush or `fsync`.

**Risk:** Concurrent requests or multiple workers can create sibling chain entries, interleave writes, lose records or acknowledge success before durable storage.

**Recommendation:** Use a transactional database with a unique predecessor/sequence constraint. For temporary file mode, enforce one writer, lock the append, flush and `fsync`, and verify the committed record before returning success.

**Status:** OPEN

### KAI-TLED-007 — MEDIUM — Merkle publication is mutable, local and non-atomic

**Issue:** Merkle manifests are read, extended and rewritten as an ordinary JSON array without locking, atomic replacement, signature or external timestamping.

**Risk:** The publication history can be edited or lost together with the ledger and concurrent publications can overwrite one another. It does not provide an independent integrity anchor.

**Recommendation:** Publish signed checkpoints to append-only external storage or transparency infrastructure. Use atomic writes and include ledger sequence, key ID and prior checkpoint.

**Status:** OPEN

---

## Ledger worker: `ledger-worker/app.py`

### KAI-LWORK-001 — HIGH — Ledger archives silently contain only the latest 10,000 entries

**Issue:** `archive_snapshot()` calls `/ledger/tail?limit=10000` and writes the result as the current ledger snapshot without recording that earlier entries may be omitted.

**Risk:** Once the ledger exceeds 10,000 records, archives are incomplete while appearing to be full backups. Recovery and forensic reconstruction may permanently lose older events.

**Recommendation:** Export by immutable sequence ranges with pagination and checkpoint verification. Record first/last sequence, total source count, completeness and content digest in the archive manifest.

**Status:** OPEN

### KAI-LWORK-002 — MEDIUM — Ledger-worker operational endpoints lack visible authentication

**Issue:** Manual verification, stats refresh, archive creation, archive listing and verification history are exposed without route-level authentication.

**Risk:** Reachable callers can trigger expensive full-chain operations, force archive creation and inspect internal integrity status, paths and errors.

**Recommendation:** Require operator authentication and scopes for mutation endpoints; restrict read endpoints and rate-limit expensive operations.

**Status:** OPEN

### KAI-LWORK-003 — MEDIUM — Heartbeat notifications ignore unsuccessful HTTP responses

**Issue:** `_notify_heartbeat()` awaits the POST but does not inspect status codes.

**Risk:** Ledger integrity alerts can be rejected with 4xx or 5xx responses while the worker treats notification as completed.

**Recommendation:** Validate accepted status codes, record delivery failure metrics and place critical integrity alerts in a durable retry queue.

**Status:** OPEN

---

## Current totals

- Findings logged: **91**
- Critical: **16**
- High: **38**
- Medium: **36**
- Low: **1**
- Current security posture: **HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE**
- Audit state: **IN PROGRESS**

## Files materially reviewed

`agentic/app.py`, `agentic/web_scout.py`, `common/auth.py`, `tool-gate/app.py`, `common/runtime.py`, `common/resilience.py`, `common/llm.py`, `agentic/swarm.py`, `agentic/swarm_stages.py`, `agentic/cognitive_fsm.py`, `agentic/trust_integration.py`, `agentic/trust_core.py`, `agentic/router.py`, `supervisor/app.py`, `memu-core/app.py`, `verifier/app.py`, `executor/app.py`, `trust-ledger/app.py`, `trust-ledger/ledger.py`, `ledger-worker/app.py`.
