# Kai Code Audit Register

Repository: `dainius1234/kai-system`  
Audit method: file-by-file review from the core execution path outward  
Status: ACTIVE  
Last updated: 26 July 2026

## Severity scale

- CRITICAL — credible compromise, destructive operation, or major integrity risk
- HIGH — serious correctness, reliability, security, or production-readiness risk
- MEDIUM — material defect, scalability issue, or maintainability risk
- LOW — limited defect or standards issue
- IMPROVEMENT — architectural or implementation enhancement

---

## Core file: `agentic/app.py`

### KAI-CORE-001 — CRITICAL — Privileged mutation endpoints appear unauthenticated

**Issue:** Endpoints appear able to reset recovery state, replace `SOUL.md` and `AGENTS.md`, modify skills, alter trust, and invoke trading operations without visible route-level authentication.

**Risk:** A reachable unauthorised caller could change Kai's operating identity, instructions, trust state, skills, recovery state, or trading behaviour.

**Recommendation:** Add central authentication and route-level scopes: `read`, `operate`, `admin`, `trust_admin`, and `trading`. Sign internal service calls with timestamp, nonce, and replay protection.

**Status:** OPEN — confirmed that `common.auth.verify_gate_signature` is used by Tool Gate only, not as general Agentic API middleware.

### KAI-CORE-002 — CRITICAL — SSRF through Web Scout

**Issue:** `agentic/web_scout.py` validates only that the URL scheme is HTTP or HTTPS. It does not block loopback, private, link-local, reserved, multicast, metadata, Docker-internal or credential-bearing targets. Redirects are followed automatically without target revalidation.

**Risk:** A caller may use Kai to query internal services, localhost, host-network resources, cloud metadata endpoints or other non-public systems.

**Recommendation:** Resolve and validate every destination IP before connection and after every redirect. Block private, loopback, link-local, multicast, unspecified and reserved ranges; reject embedded credentials; constrain ports and redirect count; use a dedicated egress proxy where possible.

**Status:** OPEN — CONFIRMED

### KAI-CORE-003 — HIGH — Context-budget enforcement can exceed its own limit

**Issue:** `_trim_context()` always preserves the first system message and last user message, even when those messages alone exceed the configured budget.

**Risk:** Model context overflow, downstream failures, or uncontrolled provider-side truncation.

**Recommendation:** Reserve output capacity, cap user input, section-trim or compress the system prompt, and assert the final token count before invoking the model.

**Status:** OPEN

### KAI-CORE-004 — HIGH — Non-atomic writes to identity files

**Issue:** `SOUL.md` and `AGENTS.md` are written directly without temporary files, atomic replacement, revision checks, locks, backups, validation, or size limits.

**Risk:** Partial writes, corruption, concurrent lost updates, uncontrolled prompt growth, and accidental or malicious identity replacement.

**Recommendation:** Use same-filesystem temporary writes, flush, `fsync`, and `os.replace`; add hash/revision checks, size limits, backup, audit identity, and rollback.

**Status:** OPEN

### KAI-CORE-005 — HIGH — Repeated `httpx.AsyncClient` creation

**Issue:** Context gathering and several integration paths create new HTTP clients per request or per dependency call.

**Risk:** Lost connection pooling, repeated DNS/TCP setup, socket churn, higher latency, and less predictable shutdown.

**Recommendation:** Create lifecycle-managed shared clients for internal and external traffic with separate transports, connection limits, timeout policies, and security controls.

**Status:** OPEN

### KAI-CORE-006 — HIGH — Silent broad exception handling

**Issue:** Multiple central paths use broad `except Exception` handling and silently continue.

**Risk:** Invisible degradation, false confidence in completed context gathering, weak observability, and difficult root-cause analysis.

**Recommendation:** Record structured, rate-limited service failure events with operation, elapsed time, error class, correlation ID, failure counters, stale-data state, and explicit degraded provenance.

**Status:** OPEN

### KAI-CORE-007 — MEDIUM — Process-local mutable state is not multi-worker safe

**Issue:** Alerts, sensor baselines, observations, gap counters, ritual state, snapshots, and background-task references are held in process globals.

**Risk:** Divergent state between workers and state loss after worker restart.

**Recommendation:** Either enforce and document a single orchestrator worker or move shared state into Redis/Postgres with atomic operations and TTLs.

**Status:** OPEN

### KAI-CORE-008 — MEDIUM — Missing request size and value constraints

**Issue:** Price arrays, URL fetch limits, result limits, messages, quantities, prices, and other request fields lack consistent upper bounds and finite/positive validation.

**Risk:** CPU and memory pressure, malformed numerical input, and denial-of-service exposure.

**Recommendation:** Use constrained Pydantic fields and apply an application-wide request-body limit.

**Status:** OPEN

### KAI-CORE-009 — MEDIUM — Internal exception details may leak to API callers

**Issue:** Some endpoint responses include raw exception strings.

**Risk:** Disclosure of internal URLs, filesystem paths, model identifiers, dependency errors, or configuration details.

**Recommendation:** Log full details internally and return a stable public error code plus trace identifier.

**Status:** OPEN

### KAI-CORE-010 — LOW — Timezone-naive UTC timestamps

**Issue:** Conversation memory and world-state persistence use `datetime.utcnow().isoformat()`.

**Risk:** Ambiguous timestamps during exchange, ordering, migration, or cross-system analysis.

**Recommendation:** Use `datetime.now(timezone.utc).isoformat()`.

**Status:** OPEN

---

## Web access: `agentic/web_scout.py`

### KAI-WEB-001 — CRITICAL — Trust control fails open

**Issue:** `_check_trust()` permits Web Scout operations when trust infrastructure raises any unexpected exception.

**Risk:** A broken import, corrupted trust state, programming error or unavailable trust dependency silently removes the intended trust boundary.

**Recommendation:** Fail closed for autonomous and externally directed network access. Return a distinct `trust_unavailable` result and emit a high-severity audit event.

**Status:** OPEN

### KAI-WEB-002 — HIGH — Response body is downloaded without a byte limit

**Issue:** `httpx.Client.get()` buffers the complete response before text is truncated to `max_chars`. `content_length` is measured only after the full body has been received.

**Risk:** A large or endless response can consume excessive memory, bandwidth and worker time despite a small configured output limit.

**Recommendation:** Stream responses, enforce a strict maximum byte count, reject excessive declared content lengths, and stop reading once the cap is reached.

**Status:** OPEN

### KAI-WEB-003 — MEDIUM — Error responses expose raw network exception text

**Issue:** Web Scout returns `str(exc)` in public result objects.

**Risk:** DNS details, proxy configuration, internal addresses, certificate information or connection behaviour may be disclosed.

**Recommendation:** Return stable error codes externally and retain full exception details only in structured internal logs.

**Status:** OPEN

---

## Authentication: `common/auth.py` and `tool-gate/app.py`

### KAI-AUTH-001 — CRITICAL — HMAC does not bind the full request

**Issue:** The signature payload contains only `actor_did`, `session_id`, `tool`, `nonce` and integer timestamp. It excludes `params`, `conviction`, `cosign`, `rationale`, `device`, `request_source`, `trace_id` and `idempotency_key`.

**Risk:** A valid signed request can be modified after signing. In particular, conviction can be raised, `cosign` can be changed to `true`, and tool parameters can be replaced while the signature remains valid.

**Recommendation:** Sign a canonical serialization of every security-relevant field, including method, route, body digest, actor, session, nonce, timestamp and key ID. Reject duplicate JSON keys and non-canonical numerical forms.

**Status:** OPEN — immediate remediation required

### KAI-AUTH-002 — CRITICAL — Idempotency lookup occurs before authentication and request validation

**Issue:** `/gate/request` checks the caller-supplied idempotency key and returns a cached `GateDecision` before token validation, signature verification, nonce validation, tool allowlisting or comparison with the current request body.

**Risk:** Anyone who obtains or guesses a key may retrieve a previous decision. More seriously, an approved decision can be replayed for a different request carrying the same idempotency key, because the cache is not bound to a request digest or identity.

**Recommendation:** Authenticate and validate first. Namespace idempotency keys by authenticated principal and route, store a canonical request hash with the decision, and reject reuse with non-identical content.

**Status:** OPEN — immediate remediation required

### KAI-AUTH-003 — HIGH — Nonce persistence is non-atomic and concurrency-unsafe

**Issue:** The nonce cache is a process-global dictionary written wholesale with `Path.write_text()` on every accepted request, without a lock, atomic replacement or shared store.

**Risk:** Concurrent requests may race; multi-worker instances maintain different replay caches; a crash during write can corrupt persistence; replay protection can be bypassed across workers or restarts.

**Recommendation:** Use Redis `SET NX EX` or a transactional database uniqueness constraint for nonce consumption. Do not use a JSON file as the primary replay-control mechanism.

**Status:** OPEN

### KAI-AUTH-004 — HIGH — Co-sign is represented as an untrusted boolean in the request

**Issue:** `cosign: bool` is accepted in `GateRequest`, and policy evaluation treats `cosign=True` as operator approval. The field is not cryptographically bound by the current HMAC.

**Risk:** A caller able to submit a validly signed base request can elevate it to operator-approved status by changing one boolean.

**Recommendation:** Remove direct co-sign assertion from ordinary gate requests. Represent approval as a separate, authenticated operator action referencing an immutable request hash and one-time challenge.

**Status:** OPEN

---

## Runtime primitives: `common/runtime.py`

### KAI-RUN-001 — HIGH — Redis audit hash-chain append is non-atomic

**Issue:** `AuditStream.log()` reads `audit:last_hash`, calculates the next hash, appends the stream entry and updates `audit:last_hash` as separate Redis operations.

**Risk:** Concurrent writers can read the same prior hash and create sibling entries. The stored linear chain then becomes invalid, causing later integrity verification to halt or disable auditing even though no malicious tampering occurred.

**Recommendation:** Perform compare-and-append atomically with a Lua script or Redis transaction using optimistic locking. Include the predecessor hash in each entry and define deterministic multi-writer ordering.

**Status:** OPEN

### KAI-RUN-002 — MEDIUM — Structured logger does not emit reliably valid JSON

**Issue:** The formatter interpolates `%(message)s` directly inside a JSON string without JSON escaping.

**Risk:** Quotes, backslashes, control characters or multiline exception messages can corrupt log records, break ingestion and undermine incident investigation.

**Recommendation:** Use a real JSON logging formatter that serialises a dictionary with `json.dumps`; include exception, correlation and trace fields separately.

**Status:** OPEN

### KAI-RUN-003 — MEDIUM — Audit verification performs an unbounded full-stream scan at startup

**Issue:** `verify_or_halt()` calls `XRANGE` across the complete `audit:logs` stream and recomputes every historical hash during construction.

**Risk:** Startup latency and memory consumption grow indefinitely with audit history; large streams may cause readiness failures or restart loops.

**Recommendation:** Use periodic signed checkpoints, bounded incremental verification and retention/archival policy. Persist the last verified stream ID and checkpoint hash.

**Status:** OPEN

### KAI-RUN-004 — MEDIUM — Error-budget calculation omits most server and client failures

**Issue:** `ErrorBudget.snapshot()` counts only status codes 429, 500 and 408 as errors.

**Risk:** Failures such as 502, 503, 504, 401 and 403 can be recorded as successful samples, materially understating the error ratio and delaying circuit opening.

**Recommendation:** Make success/error classification explicit per dependency. Default to 2xx/3xx success, classify all 5xx and selected 4xx as failures, and track categories separately.

**Status:** OPEN

---

## Resilience layer: `common/resilience.py`

### KAI-RES-001 — HIGH — All HTTP responses below 500 are treated as successful

**Issue:** `resilient_call()` resets the circuit breaker and returns `resp.json()` for every response with status `<500`, including 400, 401, 403, 404, 408 and 429.

**Risk:** Authentication failures, rate limits, malformed requests and missing resources are represented as successful dependency calls. Error payloads can flow into business logic while the circuit breaker is incorrectly marked healthy.

**Recommendation:** Call `raise_for_status()` or implement an explicit accepted-status set. Treat 408, 425, 429 and appropriate 5xx responses as retryable; treat other 4xx responses as non-retryable failures without resetting the breaker.

**Status:** OPEN

### KAI-RES-002 — MEDIUM — Deep health checks run sequentially without per-check deadlines

**Issue:** `ServiceHealth.probe()` awaits each registered dependency check one after another and applies no timeout.

**Risk:** One hanging dependency can block the entire health endpoint; total probe latency grows as the sum of all dependency latencies, potentially causing orchestrator timeouts and cascading restarts.

**Recommendation:** Execute checks concurrently with individual deadlines and a bounded overall deadline. Distinguish timeout, failure and degraded results.

**Status:** OPEN

### KAI-RES-003 — MEDIUM — Circuit-breaker state is concurrency-unsafe

**Issue:** Shared breaker objects mutate counters and state without synchronisation. An `_breaker_lock` exists but is unused, and half-open state allows any number of concurrent probe calls.

**Risk:** Concurrent requests can lose failure increments, prematurely close a breaker, or create a thundering herd against a recovering dependency.

**Recommendation:** Guard state transitions, permit only one or a bounded number of half-open probes, and make breaker storage worker-shared when multiple processes are supported.

**Status:** OPEN

### KAI-RES-004 — HIGH — Healing engine records unverified `auto_recovery` as a known fix

**Issue:** Reaching the knowledge phase without a caller-supplied confirmed fix automatically records `auto_recovery` for the current error. No health check or successful remediation is required.

**Risk:** The knowledge base can learn fictitious remedies and later claim that a known fix exists, creating false assurance and potentially suppressing appropriate escalation.

**Recommendation:** Record knowledge only after an explicit remediation action and independent post-fix verification. Store evidence, outcome, version, expiry and failure recurrence data.

**Status:** OPEN

---

## LLM routing: `common/llm.py`

### KAI-LLM-001 — MEDIUM — Streaming and non-streaming model availability behaviour diverges

**Issue:** Non-streaming `_live_query()` checks Ollama model availability and falls back to the default model, while `stream()` sends the requested model directly without the same pre-flight and fallback logic.

**Risk:** The same specialist can succeed in ordinary mode but fail in streaming mode, creating inconsistent user-visible behaviour and operational diagnostics.

**Recommendation:** Centralise backend/model resolution and availability checks so query and stream paths use identical routing, fallback and timeout policy.

**Status:** OPEN

### KAI-LLM-002 — HIGH — Transport failures are converted into model-like text

**Issue:** Failed non-streaming calls return an `LLMResponse` whose `text` contains `[error: ...]`; streaming failures yield `[LLM error: ...]` as text tokens rather than raising or emitting a typed failure event.

**Risk:** Downstream agents may treat infrastructure errors as genuine model output, store them in memory, score them as evidence or include them in final reasoning. This contaminates AI state and obscures failure provenance.

**Recommendation:** Use a typed result/error channel and require callers to branch on failure before consuming content. Never mix diagnostic messages into the model token stream or prompt-visible response body.

**Status:** OPEN

---

## Swarm state and scoring: `agentic/swarm.py`

### KAI-SWARM-001 — HIGH — Reputation persistence is non-atomic and multi-worker unsafe

**Issue:** Teammate reputation is maintained in a process-global dictionary and saved by replacing the JSON file directly with `Path.write_text()`, without locking, revision control, atomic replacement or a shared transactional store.

**Risk:** Concurrent swarm requests or multiple workers can lose updates, overwrite each other, partially corrupt the file, or maintain divergent reputations. A corrupted load silently resets all reputation to an empty state.

**Recommendation:** Store reputation in Redis or a transactional database with atomic increments and versioning. If file persistence remains temporarily, use a lock, temporary file, `fsync`, atomic replacement and explicit corruption alerts.

**Status:** OPEN

### KAI-SWARM-002 — HIGH — Conviction score rewards evidence quantity rather than evidence quality

**Issue:** Conflict resolution assigns evidence score solely from the number of evidence items and causal score solely from the number of generated causal chains. It does not assess source independence, provenance, duplication, recency, contradiction, verification strength or chain validity.

**Risk:** Duplicate, low-quality, poisoned or model-generated material can increase conviction merely by increasing item count. The system can become more confident without becoming more correct.

**Recommendation:** Score evidence by provenance, independence, source reliability, freshness, relevance and corroboration. Deduplicate semantically and require causal chains to be linked to supported evidence before contributing to conviction.

**Status:** OPEN

### KAI-SWARM-003 — MEDIUM — Reputation is trained from self-reported confidence rather than verified outcomes

**Issue:** Successful handoffs add the stage's own confidence to reputation, while success is defined by pipeline completion rather than external correctness or later validation.

**Risk:** An overconfident but inaccurate teammate can gain influence, creating a positive feedback loop in which self-confidence increases future voting weight.

**Recommendation:** Separate operational reliability from epistemic accuracy. Update accuracy reputation only from verified outcomes, operator review, benchmark results or later contradiction resolution; apply decay and minimum-sample confidence intervals.

**Status:** OPEN

---

## Swarm stages: `agentic/swarm_stages.py`

### KAI-STAGE-001 — HIGH — Untrusted retrieved content is inserted directly into agent prompts

**Issue:** Memory and world-state content are concatenated directly into LLM prompts without provenance boundaries, quoting, instruction stripping or a rule that retrieved text is data rather than instructions.

**Risk:** Prompt injection stored in memory or obtained from external sources can manipulate claim extraction, debate, fact-checking and causal analysis across the entire swarm.

**Recommendation:** Wrap retrieved content in strongly delimited data blocks, add explicit non-execution instructions, label provenance, run injection detection/classification and isolate external content from privileged system instructions.

**Status:** OPEN

### KAI-STAGE-002 — HIGH — Adversary failure is converted into successful conviction-gate completion

**Issue:** Any exception in `conviction_gate()` returns `HandoffStatus.COMPLETE` with the incoming confidence unchanged. The FSM may therefore present the result when prior confidence already exceeds the threshold, despite the adversary stage not running successfully.

**Risk:** A safety-critical challenge step can disappear without lowering conviction or blocking presentation, creating false assurance precisely when the adversary dependency is unavailable or malformed.

**Recommendation:** Return a typed degraded or failed status, apply a defined confidence penalty and require operator escalation for high-impact swarm types when the adversary stage is unavailable.

**Status:** OPEN

### KAI-STAGE-003 — HIGH — Moral-imagination safety stage fails open without signalling degradation

**Issue:** Any import or execution failure in the moral-imagination stage returns the original handoff unchanged. No degraded status, penalty or user-visible provenance is added.

**Risk:** The pipeline can silently bypass an intended ethical/safety analysis while producing an output indistinguishable from one that passed the stage.

**Recommendation:** Mark the handoff degraded, record the missing control in structured output, reduce conviction and fail closed or escalate for actions with financial, physical, legal or autonomy impact.

**Status:** OPEN

### KAI-STAGE-004 — MEDIUM — JSON parse failure can be recorded as a successful stage

**Issue:** Gather and causal stages convert malformed model JSON to empty lists but still call `record_success()` and return `COMPLETE`; causal analysis receives a baseline confidence of 5.0 even when no chain was parsed.

**Risk:** Model-format failures improve teammate reliability statistics and permit weak or absent reasoning to appear operationally successful.

**Recommendation:** Distinguish valid-empty results from parse failures. Return degraded status for malformed structured output, record a format error and exclude failed parses from successful reputation updates.

**Status:** OPEN

---

## Cognitive FSM: `agentic/cognitive_fsm.py`

### KAI-FSM-001 — HIGH — Failed fact-check reruns gathering but skips fact-check revalidation

**Issue:** When fact-check returns `FAIL`, the FSM reruns the gather stage once and then proceeds directly to causal checking. The newly gathered claims and evidence are not passed through fact-check again.

**Risk:** Unverified or previously failed claims can advance to causal analysis and conviction scoring. The retry path does not actually satisfy the failed validation gate.

**Recommendation:** Loop back through `GATHER → DEBATE or FACT_CHECK` with a bounded retry counter, and only proceed when the new evidence has been explicitly revalidated.

**Status:** OPEN

### KAI-FSM-002 — HIGH — Unexpected stage exceptions escape the FSM

**Issue:** `_run_stage()` catches only `asyncio.TimeoutError`. Any other exception raised outside a stage's own local handler propagates out of `run()` and bypasses the FSM's HALT result and transition logging.

**Risk:** A malformed dependency, programmer error or cancellation edge case can crash the request path without a structured halt reason, violating the stated guarantee that failures transition to HALT.

**Recommendation:** Catch expected operational exceptions at the FSM boundary, preserve cancellation semantics, emit a failed handoff and transition to HALT with a stable error code and trace ID.

**Status:** OPEN

### KAI-FSM-003 — MEDIUM — AgentHandoff claims schema is not runtime validated

**Issue:** `AgentHandoff` is a plain dataclass; confidence ranges, stage names, payload shape and claim dictionary contents are not validated despite documentation calling the handoffs schema-validated.

**Risk:** Invalid confidence values, malformed claims or inconsistent stage transitions can enter the pipeline and affect threshold logic or cause downstream errors.

**Recommendation:** Use Pydantic or explicit `__post_init__` validation, clamp or reject non-finite confidence values and validate legal state transitions centrally.

**Status:** OPEN

---

## Audit summary

- Findings logged: 39
- Critical: 6
- High: 19
- Medium: 13
- Low: 1
- Files materially reviewed: `agentic/app.py`, `agentic/web_scout.py`, `common/auth.py`, `tool-gate/app.py`, `common/runtime.py`, `common/resilience.py`, `common/llm.py`, `agentic/swarm.py`, `agentic/swarm_stages.py`, `agentic/cognitive_fsm.py`
- Current security posture: HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE
- Audit state: IN PROGRESS
