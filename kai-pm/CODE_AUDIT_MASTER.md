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

## Consolidated findings index — 1 to 104

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
| KAI-DASH-001 | CRITICAL | Dashboard is an unauthenticated privileged mutation proxy |
| KAI-DASH-002 | HIGH | Dashboard exposes sensitive personal, financial and operational data |
| KAI-DASH-003 | HIGH | Dashboard permits unauthenticated identity and memory mutations |
| KAI-DASH-004 | HIGH | Readiness can report ready while go/no-go is NO_GO |
| KAI-DASH-005 | HIGH | Internal Redis event stream is exposed without access control |
| KAI-DASH-006 | MEDIUM | Proxy inputs and pagination lack consistent bounds |
| KAI-DASH-007 | MEDIUM | Sequential fleet polling creates avoidable latency and false cascades |
| KAI-DEP-001 | CRITICAL | Network service receives Docker socket access |
| KAI-DEP-002 | HIGH | Internal services are broadly published on host interfaces |
| KAI-DEP-003 | HIGH | Development HMAC mode is enabled by default |
| KAI-DEP-004 | HIGH | Database credentials have a predictable deployment default |
| KAI-DEP-005 | HIGH | One flat bridge network collapses service trust boundaries |
| KAI-DEP-006 | MEDIUM | Runtime images are not pinned by immutable digest |

The detailed evidence for findings 1–63 remains in the historical registers and will be migrated during the final editorial pass. Findings 64 onward are summarised below with Issue / Risk / Recommendation evidence.

---

## Memory core: `memu-core/app.py`

### KAI-MEMU-001 — HIGH — Memory verification policy fails open when policy loading fails
**Issue:** Policy import/read failures disable mandatory verifier gating.  
**Risk:** Unverified information can enter persistent memory when governance is unavailable.  
**Recommendation:** Fail readiness closed and require a separately controlled emergency override.  
**Status:** OPEN

### KAI-MEMU-002 — HIGH — Missing LakeFS dependency silently replaces durable versioning with a non-durable stub
**Issue:** Import failure installs an in-memory implementation whose rollback is a no-op.  
**Risk:** Commit and rollback behaviour may be reported without durable history or recovery.  
**Recommendation:** Require the durable store when versioning is enabled and expose an explicit disabled state otherwise.  
**Status:** OPEN

### KAI-MEMU-003 — MEDIUM — Graph ingest and forget fan-out ignore unsuccessful HTTP responses
**Issue:** Fan-out requests do not validate status codes.  
**Risk:** Vector and graph stores can silently diverge.  
**Recommendation:** Validate responses and use an idempotent durable outbox.  
**Status:** OPEN

### KAI-MEMU-004 — MEDIUM — Core memory request models lack bounded validation
**Issue:** Text, identifiers, nested objects, scores and embeddings are substantially unconstrained.  
**Risk:** Oversized or malformed payloads can consume resources or corrupt persistence and ranking.  
**Recommendation:** Add strict lengths, finite ranges, depth limits and a request-body cap.  
**Status:** OPEN

---

## Verifier: `verifier/app.py`

### KAI-VER-001 — CRITICAL — Caller-supplied evidence can forge a PASS verdict
**Issue:** `/verify` accepts caller-controlled evidence and scores.  
**Risk:** Fabricated evidence can produce a PASS from the authority used for promotion and execution.  
**Recommendation:** Resolve immutable evidence server-side or require signed provenance-checked packs.  
**Status:** OPEN — immediate remediation required

### KAI-VER-002 — HIGH — Evidence quantity and duplicates can inflate confidence
**Issue:** Strong chunks are counted without semantic deduplication or source-independence checks.  
**Risk:** Repeated copies can masquerade as corroboration.  
**Recommendation:** Deduplicate by lineage and count only independent trusted sources.  
**Status:** OPEN

### KAI-VER-003 — MEDIUM — Long context automatically improves plausibility
**Issue:** Context over 20 characters receives an unconditional score bonus.  
**Risk:** Irrelevant filler can shift verdicts upward.  
**Recommendation:** Replace with grounded entailment and contradiction analysis.  
**Status:** OPEN

### KAI-VER-004 — MEDIUM — Health does not check verification dependencies
**Issue:** Health omits Memu availability and policy coherence.  
**Risk:** Traffic can be routed to an incapable verifier.  
**Recommendation:** Separate liveness and dependency-aware readiness.  
**Status:** OPEN

### KAI-VER-005 — MEDIUM — Verdict counters are process-local and concurrency-unsafe
**Issue:** Metrics use an unsynchronised process dictionary.  
**Risk:** Multi-worker telemetry is incomplete.  
**Recommendation:** Use atomic shared metrics.  
**Status:** OPEN

---

## Executor: `executor/app.py`

### KAI-EXEC-001 — CRITICAL — Shell allowlist permits arbitrary code and host-control operations
**Issue:** Allowed commands include interpreters, package managers, Docker and unrestricted HTTP clients.  
**Risk:** Callers can run code, control containers or exfiltrate data.  
**Recommendation:** Replace generic commands with fixed-schema operations in hardened disposable sandboxes.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-002 — CRITICAL — Python expression sandbox can reach imported module state
**Issue:** Caller expressions are evaluated in a wrapper with imported modules; AST filtering is not a security boundary.  
**Risk:** Crafted expressions can escape the intended namespace.  
**Recommendation:** Remove arbitrary `eval`; interpret a strict pure-expression AST or use hardened isolation.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-003 — CRITICAL — Execution endpoint lacks authentication and Tool Gate proof
**Issue:** `/execute` accepts requests without an authenticated immutable approval.  
**Risk:** Reachable callers can bypass Tool Gate.  
**Recommendation:** Require mTLS and a replay-protected capability signed over the canonical request.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-004 — HIGH — Subprocess output is fully buffered before truncation
**Issue:** stdout and stderr are buffered before output limits are applied.  
**Risk:** Output can exhaust memory.  
**Recommendation:** Stream through bounded pipes and apply OS resource limits.  
**Status:** OPEN

### KAI-EXEC-005 — HIGH — Malware scanning fails open
**Issue:** Missing or errored scanning does not block execution.  
**Risk:** Payloads execute without a functioning scanner.  
**Recommendation:** Fail closed where scanning is required and expose scanner readiness.  
**Status:** OPEN

### KAI-EXEC-006 — HIGH — State rollback does not reverse execution effects
**Issue:** Rollback only removes request metadata.  
**Risk:** Persistent side effects remain while recovery is implied.  
**Recommendation:** Remove rollback claims or implement transactional adapters and verified snapshots.  
**Status:** OPEN

### KAI-EXEC-007 — MEDIUM — Execution history exposes raw parameters and is process-local
**Issue:** Raw commands and arguments are returned without visible authentication.  
**Risk:** Secrets may leak and history is incomplete.  
**Recommendation:** Restrict access, redact structurally and centralise immutable records.  
**Status:** OPEN

### KAI-EXEC-008 — MEDIUM — Internal errors and stderr leak to callers
**Issue:** Raw exceptions and stderr are returned in API responses.  
**Risk:** Internal paths, versions and behaviour are disclosed.  
**Recommendation:** Return stable codes and trace IDs; keep details in protected logs.  
**Status:** OPEN

### KAI-EXEC-009 — MEDIUM — Executor inputs lack bounded validation
**Issue:** Commands, expressions, arguments, nested params and history limits lack consistent bounds.  
**Risk:** Pathological inputs can increase memory and parsing load.  
**Recommendation:** Add strict schemas and application-wide body limits.  
**Status:** OPEN

---

## Trust ledger: `trust-ledger/app.py`, `trust-ledger/ledger.py`

### KAI-TLED-001 — CRITICAL — Mutation and acknowledgement endpoints are unauthenticated
**Issue:** Trust writes and operator acknowledgements have no effective authentication.  
**Risk:** Reachable callers can fabricate governance records and endorsements.  
**Recommendation:** Require authenticated service identity and separate operator authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-TLED-002 — CRITICAL — Predictable default HMAC secret
**Issue:** The signing key defaults to `trust-dev-secret`.  
**Risk:** Misconfigured deployments produce forgeable records.  
**Recommendation:** Refuse startup without a strong managed secret and support rotation.  
**Status:** OPEN — immediate remediation required

### KAI-TLED-003 — HIGH — Trust-critical fields are excluded from signatures
**Issue:** Capability, trust tier, predecessor and acknowledgement fields are outside the HMAC.  
**Risk:** Governance meaning can be changed without invalidating signatures.  
**Recommendation:** Sign a canonical encoding of every immutable field and append acknowledgements as new events.  
**Status:** OPEN

### KAI-TLED-004 — CRITICAL — Replay skips corruption and verifies the filtered subset
**Issue:** Invalid physical records are skipped during replay.  
**Risk:** Removed or changed events can disappear while the remaining list reports intact.  
**Recommendation:** Halt at the first invalid record and enter forensic read-only mode.  
**Status:** OPEN — immediate remediation required

### KAI-TLED-005 — HIGH — Acknowledgements are not durable or cryptographically bound
**Issue:** Acknowledgement mutates only in-memory fields.  
**Risk:** It disappears after restart and cannot be independently verified.  
**Recommendation:** Append a signed acknowledgement event.  
**Status:** OPEN

### KAI-TLED-006 — HIGH — File append is non-atomic and concurrency-unsafe
**Issue:** Chain-head selection and JSONL append have no transactional lock or fsync.  
**Risk:** Concurrent writers can create sibling records or data loss.  
**Recommendation:** Use a serialised database transaction or a rigorously locked single-writer mode.  
**Status:** OPEN

### KAI-TLED-007 — MEDIUM — Merkle publication is mutable, local and non-atomic
**Issue:** Checkpoints are ordinary rewritten local JSON without signature or external anchoring.  
**Risk:** Ledger and proof history can be rewritten together.  
**Recommendation:** Sign and publish checkpoints to independent append-only storage.  
**Status:** OPEN

---

## Ledger worker: `ledger-worker/app.py`

### KAI-LWORK-001 — HIGH — Archives silently contain only the latest 10,000 entries
**Issue:** A tail query is written as though it were a complete snapshot.  
**Risk:** Older records disappear from backups once the ledger grows.  
**Recommendation:** Export immutable ranges with completeness metadata and checkpoint verification.  
**Status:** OPEN

### KAI-LWORK-002 — MEDIUM — Operational endpoints lack authentication
**Issue:** Verification, refresh, archive and history routes are unprotected.  
**Risk:** Reachable callers can trigger expensive work and inspect integrity metadata.  
**Recommendation:** Require operator scopes and rate limits.  
**Status:** OPEN

### KAI-LWORK-003 — MEDIUM — Heartbeat notifications ignore unsuccessful responses
**Issue:** Alert POST status codes are not validated.  
**Risk:** Critical alerts can be rejected while delivery appears complete.  
**Recommendation:** Validate delivery and use a durable retry queue.  
**Status:** OPEN

---

## Dashboard and API exposure: `dashboard/app.py`

### KAI-DASH-001 — CRITICAL — Dashboard is an unauthenticated privileged mutation proxy
**Issue:** Dashboard routes establish no browser-user identity or authorisation before forwarding state-changing calls. `/api/mode` uses a server-held Tool Gate token, allowing an unauthenticated browser request to exercise the dashboard's trusted identity. Other mutation routes similarly forward directly to internal services.  
**Risk:** Any caller reaching port 8080 can exercise privileged internal capabilities through the dashboard and bypass intended service-boundary controls.  
**Recommendation:** Put the dashboard behind strong operator authentication, enforce CSRF protection, apply route-level scopes and propagate a verified end-user identity rather than a universal server credential.  
**Status:** OPEN — immediate remediation required

### KAI-DASH-002 — HIGH — Sensitive personal, financial and operational data is exposed
**Issue:** Unauthenticated routes return memories, thinking episodes, autobiographical and relationship data, identity state, emotional records, financial summaries, email/calendar-adjacent feeds, logs, fleet details, policy hashes and service errors.  
**Risk:** A reachable caller can obtain highly sensitive operator information and internal architecture intelligence.  
**Recommendation:** Classify data, require least-privilege read scopes, redact by default and expose only the minimum UI projection.  
**Status:** OPEN

### KAI-DASH-003 — HIGH — Unauthenticated identity and memory mutations are available
**Issue:** Routes can create/update goals, edit SOUL and AGENTS content, record finance/CIS entries, submit feedback, emotional records, reflections, relationship milestones, confessions, autobiography and legacy content, and trigger dream/introspection actions.  
**Risk:** Attackers can poison durable memory, alter identity/governance material and initiate consequential backend work.  
**Recommendation:** Require explicit operator authentication, per-operation authorisation, immutable audit linkage and confirmation for high-impact identity changes.  
**Status:** OPEN

### KAI-DASH-004 — HIGH — Readiness can report ready while go/no-go is NO_GO
**Issue:** `core_ready` checks only selected node liveness and tests `ledger_size >= 0` and `memory_count >= 0`. Failed dependency reads default both values to zero, which satisfies the test. `/readiness` ignores the separate go/no-go decision.  
**Risk:** Orchestration can mark the dashboard ready despite failed evidence stores, insufficient gate proof or other explicit NO_GO blockers.  
**Recommendation:** Derive readiness from mandatory dependency checks and require the go/no-go decision to be GO; distinguish unknown from zero.  
**Status:** OPEN

### KAI-DASH-005 — HIGH — Internal Redis event stream is exposed without access control
**Issue:** `/api/events` subscribes to health, episode, breaker and memory channels and streams their payloads to any connected client.  
**Risk:** Internal events and potentially sensitive memory/episode metadata can be monitored continuously; connection floods can also consume Redis and application resources.  
**Recommendation:** Authenticate SSE clients, authorise channels, redact event payloads, cap concurrent streams and enforce idle/lifetime limits.  
**Status:** OPEN

### KAI-DASH-006 — MEDIUM — Proxy inputs and pagination lack consistent bounds
**Issue:** Raw JSON bodies and query parameters such as `top_k`, `limit`, query text, category and session IDs are forwarded without strict size/range schemas.  
**Risk:** Oversized requests and extreme fan-out/result sizes can pressure the dashboard and backend services.  
**Recommendation:** Use typed Pydantic models, global body limits, finite pagination caps and allowlisted fields.  
**Status:** OPEN

### KAI-DASH-007 — MEDIUM — Sequential fleet polling creates latency and false cascades
**Issue:** `fetch_status()` requests every node sequentially with a per-node timeout. Index and go/no-go paths repeat additional backend calls.  
**Risk:** A few slow services can multiply response time and make healthy systems appear unavailable under load.  
**Recommendation:** Poll concurrently with a total deadline, cache briefly and avoid duplicate dependency calls within one report.  
**Status:** OPEN

---

## Deployment and runtime: `docker-compose.minimal.yml`, representative Dockerfiles

### KAI-DEP-001 — CRITICAL — Network service receives Docker socket access
**Issue:** `docker-watcher` mounts `/var/run/docker.sock` while publishing port 8041. Marking the socket mount read-only does not make Docker API operations read-only; the Unix socket remains a control channel.  
**Risk:** A compromise of the service or its dependencies can generally create privileged containers, mount the host filesystem and obtain host-level control.  
**Recommendation:** Remove direct socket access. Use a narrowly scoped authenticated proxy exposing only required read operations, or collect metrics through a separate hardened host agent.  
**Status:** OPEN — immediate remediation required

### KAI-DEP-002 — HIGH — Internal services are broadly published on host interfaces
**Issue:** Tool Gate, Memu, introspection, agentic, heartbeat, dashboard, sensory services, broker, Docker watcher, calendar, supervisor, verifier and numerous other internal components use host mappings such as `"8000:8000"`, which bind to all host interfaces by default.  
**Risk:** Controls designed as internal service boundaries become directly reachable from the host network and potentially the wider LAN, magnifying every unauthenticated endpoint finding.  
**Recommendation:** Publish only the intended ingress, bind local-only development ports to `127.0.0.1`, and keep internal services on non-published segmented networks.  
**Status:** OPEN

### KAI-DEP-003 — HIGH — Development HMAC mode is enabled by default
**Issue:** Tool Gate and agentic are launched with `HMAC_ALLOW_DEV_SECRET: "true"`.  
**Risk:** Development signing behaviour can remain active in real deployments, weakening service authentication and making predictable credentials operationally acceptable.  
**Recommendation:** Default development-secret support to false, refuse it outside an explicit development profile and fail startup when production key material is absent.  
**Status:** OPEN

### KAI-DEP-004 — HIGH — Database credentials have a predictable deployment default
**Issue:** PostgreSQL and service connection strings fall back to password `localdev`.  
**Risk:** Any deployment that omits `DB_PASSWORD` runs with a source-known credential. The flat internal network and exposed services increase the chance of lateral access.  
**Recommendation:** Require a generated secret, reject known development values and use a secret manager or Compose secrets rather than ordinary environment interpolation.  
**Status:** OPEN

### KAI-DEP-005 — HIGH — One flat bridge network collapses service trust boundaries
**Issue:** Database, Redis, LLM, governance, execution-adjacent, dashboard, personal-data and sensory services share the same `/16` bridge network with no network-level segmentation.  
**Risk:** Compromise of a low-trust feed or UI service provides direct network reachability to high-trust stores and control services.  
**Recommendation:** Segment ingress/UI, data, governance/execution, observability and external-fetch services; permit only explicit service-to-service flows.  
**Status:** OPEN

### KAI-DEP-006 — MEDIUM — Runtime images are not pinned by immutable digest
**Issue:** Images include mutable references such as `ollama/ollama:latest`, `python:3.11-slim`, `redis:7-alpine` and `pgvector/pgvector:pg15` rather than immutable digests.  
**Risk:** Rebuilds can silently consume different upstream content, weakening reproducibility, rollback and supply-chain review.  
**Recommendation:** Pin production images by digest, automate reviewed updates and generate an SBOM/provenance record for each release.  
**Status:** OPEN

---

## Current totals

- Findings logged: **104**
- Critical: **18**
- High: **46**
- Medium: **39**
- Low: **1**
- Current security posture: **HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE**
- Audit state: **IN PROGRESS**

## Files materially reviewed

`agentic/app.py`, `agentic/web_scout.py`, `common/auth.py`, `tool-gate/app.py`, `common/runtime.py`, `common/resilience.py`, `common/llm.py`, `agentic/swarm.py`, `agentic/swarm_stages.py`, `agentic/cognitive_fsm.py`, `agentic/trust_integration.py`, `agentic/trust_core.py`, `agentic/router.py`, `supervisor/app.py`, `memu-core/app.py`, `verifier/app.py`, `executor/app.py`, `trust-ledger/app.py`, `trust-ledger/ledger.py`, `ledger-worker/app.py`, `dashboard/app.py`, `docker-compose.minimal.yml`, `dashboard/Dockerfile`, `docker-watcher/Dockerfile`.
