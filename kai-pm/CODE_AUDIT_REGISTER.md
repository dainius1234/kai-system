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

## Audit summary

- Findings logged: 17
- Critical: 6
- High: 7
- Medium: 3
- Low: 1
- Files materially reviewed: `agentic/app.py`, `agentic/web_scout.py`, `common/auth.py`, `tool-gate/app.py`
- Current security posture: HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE
- Audit state: IN PROGRESS
