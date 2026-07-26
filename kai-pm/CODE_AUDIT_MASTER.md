# Kai Code Audit Master Register

Repository: `dainius1234/kai-system`  
Status: ACTIVE — SINGLE SOURCE OF TRUTH  
Last updated: 26 July 2026  
Audit method: file-by-file review from core execution paths outward

This file is now the definitive audit register. Earlier files remain as historical working records only:

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

## Consolidated findings index — 1 to 63

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

The complete Issue / Risk / Recommendation evidence for findings 1–63 is retained in the three historical registers listed above and will be migrated into this master during the final editorial pass. The index above ensures the full numbered list is already available in one file and prevents any finding from being lost.

---

## Memory core: `memu-core/app.py`

### KAI-MEMU-001 — HIGH — Memory verification policy fails open when policy loading fails

**Issue:** Importing or reading `common.policy` is wrapped in a broad exception handler that sets `REQUIRE_VERDICT_PASS = False` and `LOG_ONLY_MODE = False`.

**Risk:** A missing module, malformed policy, configuration error or startup defect silently disables mandatory verifier gating. Unverified or poisoned information can then enter persistent memory precisely when the policy control is unavailable.

**Recommendation:** Fail closed when verification policy cannot be loaded. Expose a not-ready state, emit a critical configuration event and require an explicit, separately controlled emergency override.

**Status:** OPEN

### KAI-MEMU-002 — HIGH — Missing LakeFS dependency silently replaces durable versioning with a non-durable stub

**Issue:** Any exception importing the LakeFS client installs an in-memory replacement. The stub stores commits only in a process list and implements `revert()` as a no-op.

**Risk:** The service may report commit identifiers and versioning behaviour while providing no durable history or actual rollback. Restarting loses all stub commits, and recovery operations can appear successful without changing state.

**Recommendation:** Treat the durable version-store dependency as required whenever versioning or rollback is enabled. Fail readiness or explicitly expose `versioning_mode: disabled`; never emulate successful rollback with a no-op.

**Status:** OPEN

### KAI-MEMU-003 — MEDIUM — Graph ingest and forget fan-out ignore unsuccessful HTTP responses

**Issue:** Graph fan-out awaits `POST` requests but does not inspect status codes or call `raise_for_status()`.

**Risk:** HTTP 4xx and 5xx responses are treated as completed fan-out operations. Vector memory and graph memory can diverge silently, leaving missing relationships or orphaned graph nodes.

**Recommendation:** Validate accepted status codes, emit structured failure metrics and use a durable retry/outbox mechanism with source IDs and idempotency.

**Status:** OPEN

### KAI-MEMU-004 — MEDIUM — Core memory request models lack bounded validation

**Issue:** Memory queries, session IDs, timestamps, free-text notes, event types, result text, metrics, state deltas, relevance, importance, user IDs and embeddings are represented by unconstrained Pydantic fields.

**Risk:** Oversized payloads, malformed timestamps, non-finite scores, excessive nested objects and unexpectedly large embeddings can consume memory and CPU or corrupt ranking and persistence logic.

**Recommendation:** Add strict lengths, finite numeric ranges, timestamp parsing, nested-object depth/size limits and an application-wide body cap. Reject unknown or malformed roles and event types where applicable.

**Status:** OPEN

---

## Verifier: `verifier/app.py`

### KAI-VER-001 — CRITICAL — Caller-supplied evidence can forge a PASS verdict

**Issue:** `/verify` accepts an arbitrary `evidence_pack` from the caller and `_memory_cross_ref()` uses it directly instead of retrieving trusted evidence. Caller-controlled `rank_score`, `relevance`, `importance` and content are converted into support scores with no provenance, signature or source validation.

**Risk:** A caller can fabricate duplicate high-scoring evidence records and obtain strong chunks and a PASS verdict from the service described as the single authority for memory promotion and tool execution.

**Recommendation:** Never trust evidence scores supplied by ordinary callers. Accept only immutable evidence IDs resolved server-side from an authenticated store, or require signed evidence packs with issuer, digest, freshness and chain-of-custody validation.

**Status:** OPEN — immediate remediation required

### KAI-VER-002 — HIGH — Evidence quantity and duplicates can inflate verification confidence

**Issue:** Memory support is calculated from the ratio of records above a threshold, and strong chunks are counted independently. There is no semantic deduplication, source independence check or cap per originating memory/source.

**Risk:** Repeated copies of the same claim can create apparent corroboration and satisfy the minimum strong-chunk requirement without independent evidence.

**Recommendation:** Deduplicate semantically, group by source lineage and count corroboration only across independent trusted sources. Penalise circular or self-referential evidence.

**Status:** OPEN

### KAI-VER-003 — MEDIUM — Any sufficiently long context automatically improves plausibility

**Issue:** `_keyword_plausibility()` adds `0.1` whenever context exists and is longer than 20 characters, regardless of whether the context supports, contradicts or is unrelated to the claim.

**Risk:** Irrelevant or adversarial filler can increase the aggregate verification score and shift a verdict from FAIL_CLOSED toward REPAIR or PASS.

**Recommendation:** Remove unconditional context bonuses. Score context only through evidence-grounded entailment and contradiction analysis with explicit provenance.

**Status:** OPEN

### KAI-VER-004 — MEDIUM — Health endpoint reports healthy without checking verification dependencies

**Issue:** `/health` always returns `status: ok` and policy metadata. It does not test Memu availability, policy validity or whether required thresholds are coherent.

**Risk:** Orchestration can route safety-critical verification traffic to an instance unable to obtain evidence, while dashboards report it as healthy.

**Recommendation:** Separate liveness and readiness. Readiness should validate policy configuration and required dependency connectivity with bounded deadlines.

**Status:** OPEN

### KAI-VER-005 — MEDIUM — Verdict counters are process-local and concurrency-unsafe

**Issue:** `_verdict_counts` is a mutable process dictionary incremented without synchronisation or shared persistence.

**Risk:** Multi-worker deployments expose incomplete and divergent metrics; concurrent increments may be lost, weakening alerting and audit reconstruction.

**Recommendation:** Use a proper metrics backend with atomic counters and labels for verdict, policy version and source. Do not treat process-local dictionaries as fleet-wide telemetry.

**Status:** OPEN

---

## Executor: `executor/app.py`

### KAI-EXEC-001 — CRITICAL — Shell allowlist permits arbitrary code and host-control operations

**Issue:** The shell allowlist includes general-purpose execution and control tools: `python3`, `pip`, `git`, `make`, `docker` and `curl`. Checking only the first executable and using `shell=False` does not constrain what those programs can do.

**Risk:** A caller can execute arbitrary Python through `python3 -c`, install or run packages, invoke Makefile targets, use Git helper/config features, control Docker workloads or exfiltrate data over HTTP. If the container has a Docker socket or sensitive mounts, this can become host compromise.

**Recommendation:** Remove interpreters, package managers, build systems, container clients and unrestricted network clients from the generic command allowlist. Implement per-tool typed operations with fixed argument schemas, isolated containers, read-only filesystems, seccomp/AppArmor, no Docker socket and explicit egress policy.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-002 — CRITICAL — Python expression sandbox can reach imported module state

**Issue:** The generated wrapper imports `sys`, `math`, `json` and `datetime`, then evaluates the caller expression in the module global namespace. AST checks block selected names and private attributes but permit ordinary attribute and subscript access such as `sys.modules[...]` and calls on retrieved module objects.

**Risk:** A crafted expression can access already-loaded modules and invoke dangerous functionality outside the intended restricted namespace. The mechanism is not a security sandbox.

**Recommendation:** Remove arbitrary Python evaluation. For mathematical expressions, parse and interpret a strict AST allowlist without `eval`, names, attributes, subscripts or calls except explicitly implemented pure functions. Run any broader code in a disposable hardened sandbox with no secrets, network or host mounts.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-003 — CRITICAL — Execution endpoint lacks visible authentication and proof of Tool Gate approval

**Issue:** `/execute` accepts `tool`, `params`, `task_id` and `device` directly. It does not validate an authenticated caller, signed gate decision, immutable request digest, nonce, expiry or policy verdict.

**Risk:** Any network caller able to reach Executor can bypass Tool Gate and invoke shell, script or Python execution directly.

**Recommendation:** Require mutually authenticated service identity and a short-lived, replay-protected execution capability signed by Tool Gate over the complete canonical request. Bind tool, parameters, task, device, policy version and expiry.

**Status:** OPEN — immediate remediation required

### KAI-EXEC-004 — HIGH — Subprocess output is fully buffered before truncation

**Issue:** Shell, script and Python handlers use `subprocess.run(..., capture_output=True, text=True)`. `MAX_OUTPUT_SIZE` is applied only after the process exits and only to stdout.

**Risk:** A process can emit unbounded stdout or stderr and exhaust executor memory despite the configured output limit.

**Recommendation:** Stream stdout and stderr through bounded pipes, terminate the process when either or combined output exceeds the cap, and enforce OS-level memory, CPU, process and file-size limits.

**Status:** OPEN

### KAI-EXEC-005 — HIGH — Malware scanning fails open when unavailable or errored

**Issue:** If ClamAV is absent, `malware_scan()` returns code `0`. The execution path blocks only scan code `1`; scanner errors such as code `2`, timeouts or malformed scanner output are not treated as unsafe.

**Risk:** Production may claim malware scanning while executing payloads with no active scanner or after scan failure.

**Recommendation:** Expose scanner state in readiness and policy context. For tools requiring scanning, fail closed on unavailable, timeout or error outcomes; record scanner version and signature age.

**Status:** OPEN

### KAI-EXEC-006 — HIGH — State rollback does not roll back execution effects

**Issue:** `StateStore` records only request metadata in a process-local list. `revert_last_state()` merely pops that metadata and does not reverse filesystem, network, container, repository or script side effects.

**Risk:** Documentation and error paths imply rollback protection that does not exist. Failed or malicious commands can leave persistent changes while the system reports that state was reverted.

**Recommendation:** Rename this to execution-history bookkeeping and remove rollback claims. For reversible operations, use explicit transactional adapters, snapshots or disposable environments, with independently verified rollback results.

**Status:** OPEN

### KAI-EXEC-007 — MEDIUM — Execution history exposes raw parameters and is process-local

**Issue:** `/history` returns stored request parameters, including arbitrary command lines, script arguments and expressions, without visible authentication. History exists only in process memory and has no concurrency control.

**Risk:** Sensitive values embedded in commands or arguments may be disclosed, while restarts and multiple workers produce incomplete or divergent records.

**Recommendation:** Restrict history to authorised operators, redact secrets structurally, store immutable audit records centrally and return references rather than raw payloads by default.

**Status:** OPEN

### KAI-EXEC-008 — MEDIUM — Internal execution errors and subprocess stderr leak to callers

**Issue:** Generic exceptions are returned through `detail=f"execution failed: {exc}"`, and non-zero subprocess responses expose up to 1,000 characters of raw stderr.

**Risk:** Callers may learn filesystem paths, dependency versions, repository state, environment details and command behaviour useful for further exploitation.

**Recommendation:** Return stable public error codes and a trace ID. Keep full exceptions and stderr in access-controlled internal logs with redaction.

**Status:** OPEN

### KAI-EXEC-009 — MEDIUM — Request models and history limits lack bounded validation

**Issue:** `tool`, `params`, `task_id`, `device`, script arguments, shell commands, Python expressions and `/history?limit=` do not have consistent size, depth or numerical bounds.

**Risk:** Oversized nested parameters, extremely long commands or expressions and pathological history limits can increase memory, parsing and logging load.

**Recommendation:** Add strict Pydantic constraints, nested payload limits, finite positive configuration validation and an application-wide request-body cap.

**Status:** OPEN

---

## Current totals

- Findings logged: **81**
- Critical: **13**
- High: **34**
- Medium: **33**
- Low: **1**
- Current security posture: **HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE**
- Audit state: **IN PROGRESS**

## Files materially reviewed

`agentic/app.py`, `agentic/web_scout.py`, `common/auth.py`, `tool-gate/app.py`, `common/runtime.py`, `common/resilience.py`, `common/llm.py`, `agentic/swarm.py`, `agentic/swarm_stages.py`, `agentic/cognitive_fsm.py`, `agentic/trust_integration.py`, `agentic/trust_core.py`, `agentic/router.py`, `supervisor/app.py`, `memu-core/app.py`, `verifier/app.py`, `executor/app.py`.
