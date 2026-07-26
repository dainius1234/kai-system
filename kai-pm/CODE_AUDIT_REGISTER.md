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

**Status:** OPEN

### KAI-CORE-002 — CRITICAL — Possible SSRF through Web Scout

**Issue:** Caller-supplied URLs reach Web Scout fetch and summarise operations without visible validation at the API boundary.

**Risk:** Requests may reach loopback, Docker-internal services, private networks, link-local addresses, metadata services, or credential-bearing endpoints.

**Recommendation:** Validate HTTP/HTTPS URLs after DNS resolution; block private, loopback, link-local, multicast, reserved and credential-bearing targets; revalidate every redirect.

**Status:** OPEN — requires inspection of `agentic/web_scout.py`

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

**Issue:** Conversation memory uses `datetime.utcnow().isoformat()`.

**Risk:** Ambiguous timestamps during exchange, ordering, migration, or cross-system analysis.

**Recommendation:** Use `datetime.now(timezone.utc).isoformat()`.

**Status:** OPEN

---

## Audit summary

- Findings logged: 10
- Critical: 2
- High: 4
- Medium: 3
- Low: 1
- Current reviewed file risk: HIGH
- Audit state: IN PROGRESS
