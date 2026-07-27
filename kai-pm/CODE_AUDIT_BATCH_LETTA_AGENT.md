# Kai Code Audit — Letta Agent Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-LETTA-001 | CRITICAL | Unauthenticated callers can submit arbitrary tasks to a persistent shared memory agent |
| KAI-LETTA-002 | CRITICAL | Archival memories are exposed without authentication |
| KAI-LETTA-003 | HIGH | Caller-controlled context is interpolated directly into the agent prompt |
| KAI-LETTA-004 | HIGH | All callers share one global Letta agent and memory namespace |
| KAI-LETTA-005 | HIGH | Lazy initialisation can create duplicate agents across workers and restarts |
| KAI-LETTA-006 | HIGH | Synchronous Letta/Ollama operations run directly inside async handlers |
| KAI-LETTA-007 | MEDIUM | Task and context sizes and nesting are unbounded |
| KAI-LETTA-008 | MEDIUM | Mutable dictionary default is used in the request model |
| KAI-LETTA-009 | MEDIUM | Memory-write detection is heuristic and can misreport persistence |
| KAI-LETTA-010 | MEDIUM | Error details are returned directly to callers |
| KAI-LETTA-011 | MEDIUM | Health reports ok before client, model or storage readiness is verified |
| KAI-LETTA-012 | MEDIUM | Agent identifier and model metadata are disclosed publicly |
| KAI-LETTA-013 | MEDIUM | Configuration values and storage paths are not validated |
| KAI-LETTA-014 | MEDIUM | No shutdown or client lifecycle cleanup is implemented |

---

## Letta agent: `letta-agent/app.py`

### KAI-LETTA-001 — CRITICAL — Unauthenticated persistent-agent execution
**Issue:** `POST /agent/run` requires no authentication or authorisation. Any caller can submit arbitrary tasks and context to the single Letta agent. The code explicitly detects archival/memory tool calls, confirming the agent may persist information.  
**Risk:** Reachable callers can inject instructions and false facts into a durable memory-bearing agent, alter its future behaviour and consume local model resources.  
**Recommendation:** Require authenticated user/session authority, strict tool policy and human-approved memory writes with signed provenance.  
**Status:** OPEN — immediate remediation required

### KAI-LETTA-002 — CRITICAL — Archival memory disclosure
**Issue:** `GET /agent/memory/export` requires no authentication and returns up to 200 archival passages as plaintext.  
**Risk:** Callers can exfiltrate stored personal, operational or conversation-derived memory from the shared agent.  
**Recommendation:** Remove public export or require tightly scoped owner/admin access with redaction and audit logging.  
**Status:** OPEN — immediate remediation required

### KAI-LETTA-003 — HIGH — Context prompt injection
**Issue:** Caller-controlled dictionary entries are converted with `f"{k}={v}"` and inserted into a plaintext `[context: ...]` prefix before the task. No typed provenance, escaping, delimiting or instruction/data separation exists.  
**Risk:** Context values can inject instructions, impersonate trusted system context or manipulate memory/tool behaviour.  
**Recommendation:** Use structured tool/context channels with strict schemas and mark all caller data as untrusted.  
**Status:** OPEN

### KAI-LETTA-004 — HIGH — One shared agent for every caller
**Issue:** `_agent_id` and `_letta_client` are module-level globals and no user/session identifier is accepted.  
**Risk:** Different users and services contaminate one another’s conversation and archival memory; one attacker can influence every later caller.  
**Recommendation:** Isolate agents and memory stores per authenticated principal and purpose.  
**Status:** OPEN

### KAI-LETTA-005 — HIGH — Duplicate agent creation
**Issue:** `_agent_id` is not restored from persistent configuration. On first call in each process/restart, `_client()` unconditionally creates a new agent. Multiple workers can race because there is no lock.  
**Risk:** Restarts and concurrent workers create duplicate agents and fragmented memory stores, while callers receive inconsistent identities and history.  
**Recommendation:** Persist and atomically resolve a stable agent ID; enforce single initialisation with locking or an external registry.  
**Status:** OPEN

### KAI-LETTA-006 — HIGH — Blocking model/client calls on the event loop
**Issue:** `create_client`, `create_agent`, `send_message`, and `get_archival_memory` are synchronous calls executed directly inside async endpoints.  
**Risk:** Model inference, storage access or initialisation blocks the event-loop worker, enabling straightforward denial of service through unauthenticated requests.  
**Recommendation:** Run synchronous work in bounded worker threads/processes or use an asynchronous client with concurrency limits and timeouts.  
**Status:** OPEN

### KAI-LETTA-007 — MEDIUM — Request complexity is unbounded
**Issue:** `task` has no length limit; `context` accepts arbitrary keys, values, nesting and aggregate size. String conversion may traverse large structures.  
**Risk:** Oversized requests consume memory, prompt context, serialization time and model tokens.  
**Recommendation:** Enforce strict body, field, depth and token limits.  
**Status:** OPEN

### KAI-LETTA-008 — MEDIUM — Mutable default context
**Issue:** `RunRequest.context` is declared as `{}` rather than using a default factory.  
**Risk:** Shared mutable defaults are unsafe and can permit cross-instance contamination if later code mutates the dictionary.  
**Recommendation:** Use `Field(default_factory=dict)`.  
**Status:** OPEN

### KAI-LETTA-009 — MEDIUM — Memory persistence reporting is heuristic
**Issue:** `memories_updated` is set true whenever a returned function-call name contains `archival` or `memory`; the code does not verify that a write succeeded or distinguish reads from writes.  
**Risk:** Callers may be told memory changed when no durable write occurred, or a write may be missed under a different tool name.  
**Recommendation:** Use explicit structured write acknowledgements and durable operation IDs.  
**Status:** OPEN

### KAI-LETTA-010 — MEDIUM — Internal diagnostics are exposed
**Issue:** Complete exception strings from agent execution and memory export are returned in HTTP 502 details.  
**Risk:** Callers receive model, storage, filesystem, agent and dependency diagnostics.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-LETTA-011 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, even before the client/agent has been created and without testing Ollama, embedding model or storage.  
**Risk:** Orchestration treats an uninitialised or unusable agent as ready.  
**Recommendation:** Separate liveness, storage readiness, model readiness and agent-initialised state.  
**Status:** OPEN

### KAI-LETTA-012 — MEDIUM — Agent metadata is public
**Issue:** Health and run responses disclose the agent ID and configured model without authentication.  
**Risk:** Callers learn internal identity and deployment details useful for reconnaissance and cross-request correlation.  
**Recommendation:** Minimise public metadata and restrict operational details.  
**Status:** OPEN

### KAI-LETTA-013 — MEDIUM — Configuration is not validated
**Issue:** Model names, Ollama URL, embedding dimension, context window, port and data directory are parsed directly. Negative/extreme numeric values or unsafe paths fail only during use.  
**Risk:** Misconfiguration creates startup/runtime failure, excessive resource use or unintended storage locations.  
**Recommendation:** Validate typed configuration with safe ranges and approved paths/URLs.  
**Status:** OPEN

### KAI-LETTA-014 — MEDIUM — Client lifecycle is unmanaged
**Issue:** The lifespan handler logs startup and yields but does not close the Letta client, storage handles or any model/network resources.  
**Risk:** Reloads and shutdowns can leave resources unflushed or inconsistently closed.  
**Recommendation:** Implement explicit client/storage shutdown and await completion.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **2**
- High: **4**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **578**
- Critical: **65**
- High: **204**
- Medium: **306**
- Low: **3**

## Files materially reviewed in this batch

`letta-agent/app.py`.
