# Kai Code Audit — Questioner, Teammates and Service Watchdog Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AWARE-001 | HIGH | The Socratic stage is not wired into the live `/chat/swarm` pipeline |
| KAI-AWARE-002 | HIGH | `decompose()` ignores the Socratic feature gate |
| KAI-AWARE-003 | HIGH | Generic fixed fallback questions are represented as query decomposition |
| KAI-AWARE-004 | HIGH | One parsed question is accepted as a successful 3–5-question decomposition |
| KAI-AWARE-005 | HIGH | Generated questions are concatenated into trusted downstream query text |
| KAI-AWARE-006 | MEDIUM | Socratic query and output sizes are unbounded |
| KAI-AWARE-007 | MEDIUM | Injected LLM calls have no local timeout or cancellation contract |
| KAI-AWARE-008 | MEDIUM | Feature-flag import failure enables questioning |
| KAI-AWARE-009 | HIGH | Unsigned Markdown files define persistent teammate instructions |
| KAI-AWARE-010 | HIGH | Teammate system prompts are sent inside a user-role message rather than as system instructions |
| KAI-AWARE-011 | HIGH | World state, trust status and caller query share one undelimited prompt channel |
| KAI-AWARE-012 | HIGH | Missing `## System Prompt` makes the complete Markdown file the prompt |
| KAI-AWARE-013 | HIGH | Teammate prompts and files have no size, provenance or content validation |
| KAI-AWARE-014 | MEDIUM | Teammate registry is process-local, mutable and concurrency-unsafe |
| KAI-AWARE-015 | MEDIUM | Teammate loading is synchronous and partial failures still publish a reduced registry |
| KAI-AWARE-016 | MEDIUM | Teammate metadata is interpolated into prompts without escaping |
| KAI-AWARE-017 | MEDIUM | Teammate chat input is unbounded and reflected in responses |
| KAI-AWARE-018 | MEDIUM | Teammate invocation exposes raw LLM/internal exception text |
| KAI-AWARE-019 | HIGH | Watchdog persistence loader discards all saved service state |
| KAI-AWARE-020 | HIGH | Watchdog registry omits core execution and governance dependencies |
| KAI-AWARE-021 | HIGH | Any HTTP status below 400 is treated as healthy |
| KAI-AWARE-022 | HIGH | Health response bodies and readiness semantics are ignored |
| KAI-AWARE-023 | HIGH | Watchdog URLs are unvalidated network destinations |
| KAI-AWARE-024 | HIGH | Custom service lists permit arbitrary fan-out and duplicate identities |
| KAI-AWARE-025 | HIGH | Repeated checks create new thread pools, clients and unbounded queued futures |
| KAI-AWARE-026 | HIGH | Watchdog implements no TrustCore OBSERVER gate despite its documentation |
| KAI-AWARE-027 | HIGH | Public manual checks can drive service-down/restored FSM events |
| KAI-AWARE-028 | MEDIUM | Service checks are not tied to authenticated service identities |
| KAI-AWARE-029 | MEDIUM | Failure counters and restoration history reset on restart |
| KAI-AWARE-030 | MEDIUM | Failed worker futures can disappear without an unhealthy result |
| KAI-AWARE-031 | MEDIUM | Status and FSM logic use inconsistent definitions of critical-down |
| KAI-AWARE-032 | MEDIUM | Watchdog state saves fail silently |
| KAI-AWARE-033 | MEDIUM | Status can remain stale indefinitely without a readiness/freshness failure |
| KAI-AWARE-034 | MEDIUM | Internal network errors are persisted and exposed in status output |
| KAI-AWARE-035 | MEDIUM | Timeout values and custom service fields lack safe-range/schema validation |
| KAI-AWARE-036 | MEDIUM | Watchdog state is worker-local and concurrent checks race shared results |
| KAI-AWARE-037 | MEDIUM | First singleton construction fixes the watchdog storage directory |
| KAI-AWARE-038 | MEDIUM | FSM event delivery failures are silently suppressed by the caller |

---

## Socratic Questioner: `agentic/questioner.py`, `/chat/swarm` integration

### KAI-AWARE-001 — HIGH — Socratic stage is absent from live swarm
**Issue:** `/chat/swarm` calls `build_swarm_pipeline` without a `questioner` argument. The factory therefore does not add `questioner_fn`; `CognitiveFSM.run` also has no questioner stage parameter.  
**Risk:** the default-enabled feature and documentation claim every query is decomposed, but no live request runs it.  
**Recommendation:** report the capability disabled/unimplemented until it is explicitly integrated and tested.  
**Status:** OPEN

### KAI-AWARE-002 — HIGH — Feature gate is not enforced at the operation boundary
**Issue:** `decompose()` never calls `can_question()`. Any direct caller runs the LLM/fallback even when `FF_SOCRATIC` is disabled.  
**Risk:** internal callers bypass operational disablement.  
**Recommendation:** enforce the gate within `decompose` and return a typed disabled state.  
**Status:** OPEN

### KAI-AWARE-003 — HIGH — Fixed text masquerades as decomposition
**Issue:** absent/failed LLM calls always return the first three generic fallback questions, independent of the query, with a normal `SocraticResult`.  
**Risk:** downstream/UI can treat canned prompts as query-specific analytical improvement.  
**Recommendation:** label fallback as generic scaffolding and do not claim decomposition quality.  
**Status:** OPEN

### KAI-AWARE-004 — HIGH — Output contract is not enforced
**Issue:** any non-empty parsed question list is accepted and `used_llm=True`, even though the system requires exactly 3–5 questions. Questions are not checked for uniqueness, relevance or length.  
**Risk:** one malicious/irrelevant question becomes a successful Socratic stage.  
**Recommendation:** enforce a strict bounded schema and relevance validation.  
**Status:** OPEN

### KAI-AWARE-005 — HIGH — Model output becomes downstream instruction text
**Issue:** `_build_enriched_query` appends generated questions directly to the original query. No provenance or untrusted-data boundary exists.  
**Risk:** prompt-injected/model-generated instructions gain authority in every later stage when integration is enabled.  
**Recommendation:** preserve questions as typed untrusted analytical suggestions.  
**Status:** OPEN

### KAI-AWARE-006 — MEDIUM — Unbounded decomposition data
Complete queries and model outputs are processed/concatenated without character, token, question or aggregate limits.

### KAI-AWARE-007 — MEDIUM — No model deadline
The injected LLM callable is awaited directly without a local timeout, so a hung backend blocks decomposition indefinitely.

### KAI-AWARE-008 — MEDIUM — Missing flag infrastructure fails open
`can_question()` returns true when importing feature flags fails.

---

## Persistent teammates: `agentic/teammates.py`, `/chat/teammate/{name}`

### KAI-AWARE-009 — HIGH — Filesystem text becomes persistent persona policy
**Issue:** every Markdown file in `data/teammates` is loaded without signature, approved revision, ownership/permission or content review.  
**Risk:** a filesystem/repository change modifies privileged persona instructions used across teammate and swarm calls.  
**Recommendation:** load signed immutable prompt manifests from a governed registry.  
**Status:** OPEN

### KAI-AWARE-010 — HIGH — Persona prompt has the wrong role
**Issue:** `/chat/teammate` builds one string containing teammate context and sends it as `{"role":"user"}` to `_llm.chat`; it is not a system message despite documentation.  
**Risk:** persona/safety instructions have no privileged role and can be overridden by the caller/world text in the same message.  
**Recommendation:** keep server-owned policy in a validated system channel and untrusted text separate.  
**Status:** OPEN

### KAI-AWARE-011 — HIGH — Trust/world/query prompt injection channel
**Issue:** teammate prompt, trust status or world snapshot and the caller query are concatenated without boundaries into one message.  
**Risk:** poisoned world/trust content or the query can impersonate teammate instructions and manipulate output.  
**Recommendation:** pass structured bounded provenance-labelled data.  
**Status:** OPEN

### KAI-AWARE-012 — HIGH — Malformed teammate file becomes full prompt
**Issue:** absent `## System Prompt` leaves `system_prompt=text`, including headings, metadata and any arbitrary file content.  
**Risk:** formatting errors or injected metadata become model instructions without validation.  
**Recommendation:** reject files lacking the exact required schema.  
**Status:** OPEN

### KAI-AWARE-013 — HIGH — Prompt artefacts are unbounded and unauthenticated
**Issue:** file count/size, names, specialties, descriptions and prompts have no bounds or trusted-source checks.  
**Risk:** oversized/malicious prompts consume context and alter behaviour.  
**Recommendation:** enforce strict signed artefact limits.  
**Status:** OPEN

### KAI-AWARE-014 — MEDIUM — Registry races and diverges
The complete registry is a mutable process-global dictionary without locks/shared authority; workers can load different versions.

### KAI-AWARE-015 — MEDIUM — Partial synchronous loading appears successful
File reads happen synchronously; individual failures are warned and omitted while the reduced registry is published without degraded status.

### KAI-AWARE-016 — MEDIUM — Metadata can inject prompt syntax
Name and specialty are interpolated into the prompt header without escaping or length validation.

### KAI-AWARE-017 — MEDIUM — Unbounded/reflected user messages
Teammate requests accept unrestricted message/session strings and return the original message, duplicating sensitive content into responses/logs.

### KAI-AWARE-018 — MEDIUM — Raw exception disclosure
Teammate invocation returns `detail=f"Teammate invocation failed: {exc}"`, exposing model/network internals.

---

## Service Watchdog: `agentic/service_watchdog.py`, `agentic/app.py`

### KAI-AWARE-019 — HIGH — Persistence restoration is a no-op
**Issue:** `_load_state` iterates saved service entries but executes only `pass`; failure counters, previous down state and results are never restored.  
**Risk:** the persistent-history and consecutive-failure claims are false; restart erases the evidence needed for down/restored decisions.  
**Recommendation:** validate and restore a versioned state snapshot or remove the persistence claim.  
**Status:** OPEN

### KAI-AWARE-020 — HIGH — Critical dependency coverage is incomplete
**Issue:** default registry omits memu-core, Tool Gate, executor, verifier, trust ledger, LLM/Ollama, Redis, Postgres and the Agentic service itself. Broker and Skill Hunter are marked critical instead.  
**Risk:** core governance/execution can fail while watchdog reports no critical outage.  
**Recommendation:** derive required dependencies from a signed deployment/service graph.  
**Status:** OPEN

### KAI-AWARE-021 — HIGH — Redirects and error-like success codes are healthy
**Issue:** `healthy = resp.status_code < 400`, so every 1xx/2xx/3xx response is healthy.  
**Risk:** redirects, unauthorised alternate endpoints and non-readiness responses count as service availability.  
**Recommendation:** require exact expected status and readiness schema.  
**Status:** OPEN

### KAI-AWARE-022 — HIGH — Health semantics are ignored
**Issue:** response body is never parsed. Services returning `status: degraded`, stub/disabled state or failed dependencies with HTTP 200 are marked healthy.  
**Risk:** false readiness propagates into self-preservation/FSM decisions.  
**Recommendation:** validate service-specific signed readiness contracts.  
**Status:** OPEN

### KAI-AWARE-023 — HIGH — Watchdog is a configurable network requester
**Issue:** environment/custom service URLs are used directly without scheme, host, port or network-range policy.  
**Risk:** compromised configuration/internal calls probe arbitrary internal/external destinations and expose timing/error data.  
**Recommendation:** restrict to an approved service registry and network policy.  
**Status:** OPEN

### KAI-AWARE-024 — HIGH — Custom inventory fan-out is unbounded
**Issue:** `check_all(services=...)` accepts arbitrary-length dictionaries, duplicate names and URLs. All futures are submitted immediately.  
**Risk:** internal callers create arbitrary network/queue load and duplicate names overwrite state.  
**Recommendation:** enforce a small unique approved inventory.  
**Status:** OPEN

### KAI-AWARE-025 — HIGH — Check amplification
**Issue:** each call creates a new ten-thread executor and each ping creates a new synchronous HTTP client. There is no one-in-flight lock/rate limit; public manual and background checks may overlap.  
**Risk:** repeated callers exhaust threads, sockets and downstream health endpoints.  
**Recommendation:** use one lifecycle-managed bounded scheduler/client.  
**Status:** OPEN

### KAI-AWARE-026 — HIGH — Documented trust control is absent
**Issue:** status/check operations claim OBSERVER trust but never call TrustCore/gate integration. Feature flags are the only route-level condition.  
**Risk:** internal and unauthenticated API callers execute monitoring/FSM-influencing operations regardless of trust level.  
**Recommendation:** enforce authenticated capability policy at the watchdog boundary.  
**Status:** OPEN

### KAI-AWARE-027 — HIGH — Public calls influence the system FSM
**Issue:** `/watchdog/check` runs checks and fires returned `service_down`/`service_restored` events. No endpoint authentication exists in Agentic.  
**Risk:** callers repeatedly trigger state-machine transitions based on weak/spoofable health evidence.  
**Recommendation:** restrict event emission to one authenticated scheduler and corroborated signed health observations.  
**Status:** OPEN

### KAI-AWARE-028 — MEDIUM — No service identity proof
A process on the configured host/port is accepted without expected service name/version/instance verification.

### KAI-AWARE-029 — MEDIUM — Counters reset after restart
Because restoration is not implemented, consecutive failures and `was_down` history start over, delaying outage detection and preventing correct restoration events.

### KAI-AWARE-030 — MEDIUM — Missing result is not unhealthy
Thread-future exceptions are debug logged and no `CheckResult` is appended, so a service can vanish from the result set rather than count as failed.

### KAI-AWARE-031 — MEDIUM — Critical-down semantics conflict
FSM events require two failures, while `status()` lists any currently unhealthy critical result under `critical_down` after the first failure.

### KAI-AWARE-032 — MEDIUM — Persistence failure is invisible
`_save_state` suppresses all errors and check results still return successfully.

### KAI-AWARE-033 — MEDIUM — Stale status has no failure threshold
`status()` returns indefinitely old data and age but never marks the watchdog unavailable/stale.

### KAI-AWARE-034 — MEDIUM — Network diagnostics are exposed
Raw exception strings are persisted in status JSON and returned through watchdog APIs.

### KAI-AWARE-035 — MEDIUM — Inputs/configuration are unvalidated
Timeouts, names, URLs, paths and critical flags lack strict types/ranges; zero/negative/extreme timeout values are accepted.

### KAI-AWARE-036 — MEDIUM — Shared result state races
Overlapping checks read and write `_last_results` and `_last_checked_at` without locks; workers maintain independent watchdogs.

### KAI-AWARE-037 — MEDIUM — Singleton storage is call-order dependent
The first `get_watchdog(data_dir)` call fixes the process storage path and later calls silently ignore alternatives.

### KAI-AWARE-038 — MEDIUM — FSM delivery failure disappears
The Agentic caller catches every `fsm_fire` exception and continues, reporting the recommended event list as fired without durable acknowledgement.

---

## Batch totals

- Findings: **38**
- Critical: **0**
- High: **18**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,173**
- Critical: **99**
- High: **483**
- Medium: **588**
- Low: **3**

## Files materially reviewed in this batch

`agentic/questioner.py`, `agentic/teammates.py`, `agentic/service_watchdog.py`, and their live integrations in `agentic/app.py`.
