# Kai Code Audit — Agentic API Control Plane Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records defects in the main `agentic` FastAPI application and its direct control-plane integrations. Previously logged implementation defects in Trust Core, Trust Integration, conviction scoring, planning, model selection, market modules and cognitive foundations are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AGAPI-001 | CRITICAL | The host-published Agentic control plane has no inbound authentication or authorisation |
| KAI-AGAPI-002 | HIGH | A mounted inter-service HMAC secret is not used to authenticate Agentic API callers |
| KAI-AGAPI-003 | CRITICAL | Unauthenticated callers can rewrite and immediately activate `SOUL.md` system identity |
| KAI-AGAPI-004 | CRITICAL | Unauthenticated callers can rewrite and reload the agent registry |
| KAI-AGAPI-005 | CRITICAL | Unauthenticated callers can grant any Trust Level, including GUARDIAN |
| KAI-AGAPI-006 | CRITICAL | Unauthenticated callers can revoke or arbitrarily reset Trust Level |
| KAI-AGAPI-007 | HIGH | Trust mutations falsely attribute every remote caller as `dainius` |
| KAI-AGAPI-008 | CRITICAL | Unauthenticated callers can reset failure-containment circuit breakers |
| KAI-AGAPI-009 | CRITICAL | Unauthenticated callers can restore live breaker state from arbitrary checkpoints |
| KAI-AGAPI-010 | HIGH | Unauthenticated callers can delete recovery checkpoints |
| KAI-AGAPI-011 | HIGH | Unauthenticated checkpoint creation permits storage churn and eviction of older evidence |
| KAI-AGAPI-012 | CRITICAL | Unauthenticated callers can write arbitrary vault content and paths through the proxy |
| KAI-AGAPI-013 | CRITICAL | Unauthenticated callers can search private vault data |
| KAI-AGAPI-014 | CRITICAL | Episode recall accepts an arbitrary user ID and returns full raw history |
| KAI-AGAPI-015 | CRITICAL | Unauthenticated chat accesses and mutates private operator memory, profile and sensory context |
| KAI-AGAPI-016 | CRITICAL | Unauthenticated `/run` can obtain server-signed Tool Gate requests as trusted actor `langgraph` |
| KAI-AGAPI-017 | CRITICAL | Web Scout proxy enables unauthenticated SSRF to HTTP/HTTPS internal and metadata targets |
| KAI-AGAPI-018 | HIGH | Paper-position open and close routes expose unauthenticated financial-state mutation |
| KAI-AGAPI-019 | HIGH | Strategy auto-trade exposes unauthenticated autonomous financial-state mutation |
| KAI-AGAPI-020 | HIGH | Model Council benchmark exposes unauthenticated persistent model-profile mutation and compute use |
| KAI-AGAPI-021 | HIGH | Skill reload, unload and prune routes expose unauthenticated runtime behaviour mutation |
| KAI-AGAPI-022 | HIGH | `SOUL.md` content and storage path are disclosed without authentication |
| KAI-AGAPI-023 | HIGH | Agent registry content and storage path are disclosed without authentication |
| KAI-AGAPI-024 | HIGH | Skill matching discloses internal skill actions and response templates |
| KAI-AGAPI-025 | HIGH | Capability introspection exposes service topology, flags, trust and operational state |
| KAI-AGAPI-026 | HIGH | Recent application logs are exposed without authentication |
| KAI-AGAPI-027 | HIGH | Trust status, readiness and audit history are exposed without authentication |
| KAI-AGAPI-028 | HIGH | Model identities, capabilities, ranking and availability are exposed without authentication |
| KAI-AGAPI-029 | HIGH | Trading, market, opportunity and cache state are exposed without authentication |
| KAI-AGAPI-030 | MEDIUM | Health exposes device, dependency breaker and error-guard state |
| KAI-AGAPI-031 | HIGH | No global rate limiting, admission control or principal quota protects expensive routes |
| KAI-AGAPI-032 | HIGH | Request bodies, lists, query strings and nested payloads lack aggregate bounds |
| KAI-AGAPI-033 | HIGH | Identity-file writes are synchronous and unbounded inside async request handlers |
| KAI-AGAPI-034 | HIGH | Agent-registry writes are synchronous and unbounded inside async request handlers |
| KAI-AGAPI-035 | HIGH | Caller-selected session IDs have no ownership or isolation check |
| KAI-AGAPI-036 | HIGH | Any caller can select the relaxed PUB system-prompt mode |
| KAI-AGAPI-037 | HIGH | Direct-dispatch answers are memorised with fixed conviction 9.0 regardless of correctness |
| KAI-AGAPI-038 | HIGH | Streamed LLM answers are memorised with fixed conviction 8.0 and the default specialist |
| KAI-AGAPI-039 | HIGH | Partial streamed responses are persisted after client disconnect |
| KAI-AGAPI-040 | HIGH | Every chat turn is written into global `keeper` learning stores rather than an authenticated principal |
| KAI-AGAPI-041 | HIGH | Every chat message is submitted to value learning with outcome hard-coded as `positive` |
| KAI-AGAPI-042 | HIGH | Numerous unauthenticated downstream records are promoted into privileged system messages |
| KAI-AGAPI-043 | HIGH | Context-source outages are converted to empty context without a degraded decision state |
| KAI-AGAPI-044 | CRITICAL | Low conviction does not prevent `/run` from requesting Tool Gate approval |
| KAI-AGAPI-045 | HIGH | An adversary `block` recommendation does not directly stop execution gating |
| KAI-AGAPI-046 | CRITICAL | Caller-controlled `task_hint` is signed by the server as a trusted Tool Gate tool identity |
| KAI-AGAPI-047 | CRITICAL | Tool Gate HMAC does not bind plan parameters or conviction to the signature |
| KAI-AGAPI-048 | HIGH | Caller-selected session IDs are included in trusted signatures without caller authentication |
| KAI-AGAPI-049 | CRITICAL | Blocked or unavailable Gate decisions are recorded as perfect successful outcomes |
| KAI-AGAPI-050 | HIGH | Runs with no Gate decision are still recorded as successful outcomes |
| KAI-AGAPI-051 | HIGH | All `/run` episodes are stored under global user `keeper` |
| KAI-AGAPI-052 | HIGH | `/run` returns internal plan, history, adversary and operational metadata to the caller |
| KAI-AGAPI-053 | HIGH | Self-generated failure classifications become durable correction rules without external validation |
| KAI-AGAPI-054 | HIGH | Local substring conviction overrides bypass normal low-conviction refusal logic |
| KAI-AGAPI-055 | HIGH | Teammate chat exposes unauthenticated LLM execution with trust or live world context |
| KAI-AGAPI-056 | HIGH | Swarm chat exposes unauthenticated multi-stage LLM work and reputation mutation |
| KAI-AGAPI-057 | HIGH | Watchdog check exposes unauthenticated active probes and FSM event injection |
| KAI-AGAPI-058 | HIGH | Vault proxy forwards caller-supplied conviction and requester identity as authoritative fields |
| KAI-AGAPI-059 | MEDIUM | Log-query limits and filters are weakly validated |
| KAI-AGAPI-060 | MEDIUM | Trust audit reads and parses the complete audit file for each request |
| KAI-AGAPI-061 | MEDIUM | Checkpoint list/detail/diff parameters are unbounded and expose operational history |
| KAI-AGAPI-062 | HIGH | Checkpoint files are unsigned and may be tampered before restore |
| KAI-AGAPI-063 | HIGH | Checkpoint restore reports rollback success while restoring only breaker state |
| KAI-AGAPI-064 | HIGH | Restored breaker fields are not validated against legal states or numeric ranges |
| KAI-AGAPI-065 | MEDIUM | Checkpoint operations perform synchronous non-transactional filesystem work in async routes |
| KAI-AGAPI-066 | HIGH | Startup warm-up and proactive-observer tasks are untracked |
| KAI-AGAPI-067 | HIGH | Shutdown does not cancel or await the startup background tasks |
| KAI-AGAPI-068 | HIGH | Cleanup task admission is unbounded and can retain unlimited fire-and-forget work |
| KAI-AGAPI-069 | MEDIUM | Background task exceptions are not consumed or surfaced by the cleanup manager |
| KAI-AGAPI-070 | MEDIUM | Deprecated startup/shutdown event hooks weaken lifecycle control |
| KAI-AGAPI-071 | MEDIUM | Health can report `ok` while required services, models and background loops are unavailable |
| KAI-AGAPI-072 | MEDIUM | The application repeatedly creates short-lived HTTP clients across hot paths |

---

## Exposure and authentication

### KAI-AGAPI-001 — CRITICAL — Unauthenticated host-published control plane
**Issue:** `docker-compose.full.yml` publishes `8007:8007`. `agentic/app.py` defines no authentication dependency, API-key validation, mTLS identity check or authorisation middleware. The only middleware records metrics.  
**Risk:** Any host/network caller that can reach port 8007 can invoke identity, trust, checkpoint, vault, LLM, financial and runtime-control operations.  
**Recommendation:** Remove direct host exposure and require authenticated, authorised, replay-protected access at both network and application layers.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-002 — HIGH — Available HMAC secret is unused for inbound requests
**Issue:** The compose service mounts `INTERSERVICE_HMAC_SECRET`, but Agentic uses it only to sign outbound Tool Gate requests. Incoming API calls are not authenticated with it.  
**Risk:** The deployment appears to possess inter-service authentication while its primary control plane remains open.  
**Recommendation:** Use distinct inbound service identities with request-body binding, nonce/timestamp validation and endpoint-specific scopes.  
**Status:** OPEN

### KAI-AGAPI-003 — CRITICAL — Remote system-identity rewrite
**Issue:** `POST /soul` accepts arbitrary JSON content, writes it to the persistent SOUL path and calls `_load_soul()`, which rebuilds system prompts immediately.  
**Risk:** A remote caller can persist prompt injection, redefine identity and alter every later chat response.  
**Recommendation:** Make identity artefacts immutable at runtime or require separately authenticated, reviewed, signed revisions.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-004 — CRITICAL — Remote agent-registry rewrite
**Issue:** `POST /agents-registry` writes arbitrary content into AGENTS.md and reloads the registry without authentication, signature, schema or review.  
**Risk:** A caller can poison agent instructions/capabilities and future routing behaviour.  
**Recommendation:** Load a signed immutable registry and restrict changes to a controlled deployment workflow.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-005 — CRITICAL — Remote GUARDIAN promotion
**Issue:** `POST /trust/promote` accepts any integer TrustLevel and directly calls `TrustCore.grant`. No caller identity or current-level approval is checked.  
**Risk:** One request can grant maximum autonomy and unlock downstream capabilities.  
**Recommendation:** Remove network self-promotion; require strong operator authentication, out-of-band confirmation and monotonic signed governance events.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-006 — CRITICAL — Remote trust revocation/reset
**Issue:** `POST /trust/demote` similarly accepts any level and directly calls `revoke`.  
**Risk:** A remote caller can deny service, rewrite governance state or race a legitimate promotion/revocation.  
**Recommendation:** Apply the same protected operator-control path and revocation-precedence semantics.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-007 — HIGH — Caller impersonation in trust audit
**Issue:** Both trust routes hard-code the actor as `dainius`; the promote request’s `reason` is ignored.  
**Risk:** Audit records falsely assert operator authorisation for anonymous remote actions.  
**Recommendation:** derive actor identity from verified credentials and bind supplied reason/approval evidence to the signed event.  
**Status:** OPEN

### KAI-AGAPI-008 — CRITICAL — Remote circuit-breaker reset
**Issue:** `POST /recover` resets MEMU and Tool Gate breaker failures/state to closed. It is unauthenticated and proceeds even if the pre-recovery checkpoint fails.  
**Risk:** An attacker can repeatedly defeat failure containment and force traffic toward unhealthy dependencies.  
**Recommendation:** restrict recovery to authenticated operators/automation with dependency-health verification and immutable audit.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-009 — CRITICAL — Remote operational rollback
**Issue:** `POST /checkpoint/{id}/restore` loads a caller-selected local checkpoint and directly replaces live breaker state.  
**Risk:** An attacker can reopen failed dependencies, restore stale containment state or disrupt incident response.  
**Recommendation:** require signed checkpoints, strong operator authorisation, validation and compare-and-swap against a monotonic configuration revision.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-010 — HIGH — Remote checkpoint deletion
**Issue:** `DELETE /checkpoint/{id}` removes checkpoint evidence without authentication or retention policy.  
**Risk:** Recovery and forensic history can be selectively erased.  
**Recommendation:** make checkpoints append-only/retention-governed and require protected deletion approval.  
**Status:** OPEN

### KAI-AGAPI-011 — HIGH — Checkpoint churn and eviction
**Issue:** Any caller can create checkpoints. `_enforce_cap()` deletes the oldest files once `CHECKPOINT_MAX` is exceeded.  
**Risk:** Repeated requests can evict legitimate recovery evidence.  
**Recommendation:** authenticate creation, enforce quotas and preserve protected incident checkpoints independently.  
**Status:** OPEN

### KAI-AGAPI-012 — CRITICAL — Remote vault write
**Issue:** `POST /vault/export` forwards caller-controlled `filepath`, `content`, `conviction` and `requester` to vault-sync. Agentic performs no authentication, path policy or independent conviction verification.  
**Risk:** A caller can create/overwrite private knowledge files and poison later context.  
**Recommendation:** authenticate the principal, bind a server-generated authorised operation and enforce canonical vault paths/content limits.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-013 — CRITICAL — Remote vault search
**Issue:** `GET /vault/search` proxies arbitrary queries and folder filters to the private vault.  
**Risk:** Sensitive notes and operational data can be enumerated remotely.  
**Recommendation:** require principal-scoped search authorisation and result-level access control.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-014 — CRITICAL — Cross-user episode exfiltration
**Issue:** `/episodes/recall` accepts caller-selected `user_id` and returns raw episodes plus a concatenated context string.  
**Risk:** Any stored user namespace can be read, including full inputs, outputs, confidence and failure metadata.  
**Recommendation:** derive user identity from authentication and return only authorised, minimised fields.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-015 — CRITICAL — Chat is an open private-context gateway
**Issue:** `/chat` retrieves session history, global memories, goals, topics, emotion, identity, conscience, tasks, operator model, graph/Letta data, finance and live sensory state, then writes learning records. No caller authentication exists.  
**Risk:** Remote users can extract or poison deeply personal operator context and consume LLM resources.  
**Recommendation:** require authenticated principal/session ownership and explicit source-specific consent/authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-016 — CRITICAL — Anonymous caller receives trusted Tool Gate signature
**Issue:** `/run` accepts caller input and `task_hint`, then uses the server HMAC secret to sign a Gate request as `actor_did="langgraph"`.  
**Risk:** Anonymous input is transformed into a request carrying trusted internal service identity.  
**Recommendation:** authenticate the initiating principal and authorise the exact action before signing; preserve end-user identity/delegation in the signed request.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-017 — CRITICAL — SSRF through Web Scout routes
**Issue:** `/web-scout/fetch` and `/web-scout/summarize` accept arbitrary URLs. `web_scout._safe_url()` checks only `http`/`https`; the client follows redirects and does not reject loopback, private, link-local, Unix-proxy, DNS-rebinding or cloud metadata destinations.  
**Risk:** A remote caller can probe internal services and retrieve sensitive metadata through Agentic.  
**Recommendation:** use a hardened egress proxy with DNS/IP validation on every redirect, strict allowlists and response limits.  
**Status:** OPEN — immediate remediation required

---

## Exposed mutation and disclosure routes

### KAI-AGAPI-018 — HIGH — Open paper-trading mutation
`/paper-trading/open` and `/paper-trading/close` expose position mutation to anonymous callers. The underlying fail-open trust defects are already logged separately; this finding concerns the public route exposure.

### KAI-AGAPI-019 — HIGH — Open auto-trade mutation
`/strategy/auto-trade` lets anonymous callers supply the symbol, full price history, quantity and tag that drive automatic position changes.

### KAI-AGAPI-020 — HIGH — Open Model Council benchmark
`/model-council/benchmark` allows anonymous persistent model availability/score mutation and compute consumption.

### KAI-AGAPI-021 — HIGH — Open skill lifecycle controls
`/skills/reload`, `/skills/unload` and `/skills/prune` change live runtime skills without authentication, change approval or rollback transaction.

### KAI-AGAPI-022 — HIGH — Identity disclosure
`GET /soul` returns the complete system-identity file and filesystem path.

### KAI-AGAPI-023 — HIGH — Agent-registry disclosure
`GET /agents-registry` returns complete internal agent instructions and filesystem path.

### KAI-AGAPI-024 — HIGH — Skill-instruction disclosure
`/skills/match` returns up to 500 characters of a matched skill action and response template, allowing systematic extraction through crafted messages.

### KAI-AGAPI-025 — HIGH — Capability/topology disclosure
`/introspect/capabilities` actively probes internal service URLs and returns reachability, HTTP status, feature flags, skills, baselines, observation depth, FSM, teammate, trust, model, trading and market state.

### KAI-AGAPI-026 — HIGH — Log disclosure
`/logs` exposes root-logger messages, including downstream errors, model/service names and potentially sensitive content captured from other modules.

### KAI-AGAPI-027 — HIGH — Governance disclosure
Trust status, readiness and raw audit events are accessible without authorisation.

### KAI-AGAPI-028 — HIGH — Model-topology disclosure
`/models` and Model Council routes reveal model IDs, live availability, strengths, quality/speed tiers, benchmark data and primary/failover state.

### KAI-AGAPI-029 — HIGH — Financial/market disclosure
Paper positions/trades, P&L, market caches, alpha signals, opportunity evidence and recommendations are accessible without principal checks.

### KAI-AGAPI-030 — MEDIUM — Operational health disclosure
`/health` reveals detected device and detailed breaker/error-guard snapshots, aiding targeting and incident-state inference.

### KAI-AGAPI-031 — HIGH — No admission control
No global rate limiter or authenticated quota covers LLM chat/swarm, Web Scout, model benchmarking, watchdog probes, market scans, checkpoint churn or mutation routes.

### KAI-AGAPI-032 — HIGH — Weak aggregate input limits
Most Pydantic models specify types only. Chat/query text, price arrays, vault content, SOUL/AGENTS content, topics, categories, checkpoint labels and nested plan/context payloads lack strict byte/item/depth limits.

### KAI-AGAPI-033 — HIGH — Blocking unbounded SOUL write
The async route reads arbitrary JSON, writes potentially large text synchronously and rebuilds prompts inline. No maximum size, atomic replace or schema validation exists.

### KAI-AGAPI-034 — HIGH — Blocking unbounded AGENTS write
The agent-registry route repeats the same unbounded synchronous complete-file write/reload pattern.

---

## Chat, memory and planning integration

### KAI-AGAPI-035 — HIGH — Session ownership absent
Caller-supplied session IDs select working-memory append/recall. There is no authenticated owner, unpredictable server-generated session token or access-control check.

### KAI-AGAPI-036 — HIGH — Caller chooses relaxed risk mode
A caller may set `mode="PUB"`; PUB’s system prompt explicitly uses relaxed risk tolerance and unrestricted-topic framing. Invalid modes also become PUB.

### KAI-AGAPI-037 — HIGH — Direct answers receive fabricated 9.0 conviction
When zero-LLM dispatch succeeds, `_auto_memorize` stores conviction 9.0 regardless of downstream evidence, errors or user outcome.

### KAI-AGAPI-038 — HIGH — Stream answers receive fabricated 8.0 conviction/model identity
Finalisation memorises every non-empty streamed response using `_DEFAULT_SPECIALIST` and conviction 8.0 rather than the selected model, route, completion state or verified quality.

### KAI-AGAPI-039 — HIGH — Disconnected partial output becomes durable memory
The generator’s `finally` schedules finalisation after disconnect whenever any tokens were produced. Truncated/cancelled output is stored as a completed assistant turn and learning record.

### KAI-AGAPI-040 — HIGH — Global operator-learning namespace
Memory recall, auto-memorisation, planning episodes and predicted prefetches repeatedly hard-code `keeper`, regardless of caller/session identity.

### KAI-AGAPI-041 — HIGH — Every user message is labelled a positive value outcome
`_learn_from_exchange` posts `{"experience": ..., "outcome": "positive"}` for every chat turn, including hostile, erroneous or rejected input.

### KAI-AGAPI-042 — HIGH — Untrusted data receives system-message privilege
Retrieved memories, graph results, Letta response, financial fields, wake reasoning, goals, topics, emotional warnings, identity narrative, empathy, conscience, tasks, operator-model output, live world data and skill text are each inserted as `system` messages. Their provenance/integrity is not enforced at this boundary.

### KAI-AGAPI-043 — HIGH — Missing context is treated as no context
Parallel `_safe` wrappers and many helpers suppress all failures and return empty objects/lists. The model is not told that required sources failed, were stale or were unauthorised.

### KAI-AGAPI-044 — CRITICAL — Low conviction does not stop Gate submission
After all rethink/tree-search attempts, low conviction merely changes `plan["summary"]`. If `task_hint` exists, the code still signs and sends the Gate request with that low conviction.  
**Risk:** The stated execution threshold is advisory rather than an enforcement boundary.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-045 — HIGH — Adversary block is non-enforcing
The adversary modifier changes conviction and its recommendation is stored in metadata, but `recommendation="block"` or critical warnings do not independently prevent the Tool Gate request.

### KAI-AGAPI-046 — CRITICAL — Caller chooses the signed tool
`request.task_hint` directly becomes the signed `tool` field sent by trusted actor `langgraph`. There is no server-owned action derivation or allowlist at Agentic.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-047 — CRITICAL — Signature excludes consequential request fields
`common.auth._payload()` signs only actor, session, tool, nonce and integer timestamp. The plan in `params`, conviction, device and other request fields are outside the HMAC.  
**Risk:** A valid signature does not attest to the exact action parameters or conviction evaluated by Agentic.  
**Recommendation:** sign a canonical digest of every security-relevant field and bind it to policy/version/body hash.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-048 — HIGH — Unauthenticated session data enters trusted signature
The caller supplies `session_id`; Agentic sanitises only a few characters/length and then signs it as part of an internal trusted request.

### KAI-AGAPI-049 — CRITICAL — Rejected Gate response becomes outcome 1.0
Episode creation uses `1.0 if gate_decision else 0.7`. Any non-empty dictionary—including `{"approved": false, "status": "blocked"}` or `unavailable`—is truthy and recorded as perfect success.  
**Risk:** Failed/blocked actions poison future planning, calibration, trust and auto-promotion evidence.  
**Status:** OPEN — immediate remediation required

### KAI-AGAPI-050 — HIGH — Unevaluated run becomes success 0.7
When no task hint/Gate decision exists, the episode still receives outcome 0.7, enough to be treated as a successful past outcome by planner/adversary logic.

### KAI-AGAPI-051 — HIGH — Cross-caller episode contamination
Every run is persisted with `user_id="keeper"` and history is recalled from the same namespace.

### KAI-AGAPI-052 — HIGH — Internal decision evidence returned to caller
`GraphResponse.plan` can contain session-context counts, reused history, correction data, adversary findings, strategy/financial information, predictions and operational warnings.

### KAI-AGAPI-053 — HIGH — Self-generated corrections become durable rules
Failure classification and metacognitive rules derive from the system’s own fabricated outcome/conviction fields, then are memorised as high-relevance correction-like records without operator or verifier confirmation.

### KAI-AGAPI-054 — HIGH — Substring override bypass
`is_conviction_override` returns true if any line from a mutable local file is a substring of input. A match suppresses the final low-conviction refusal/tree-search condition and marks the plan as operator-overridden without authenticated operator action.

### KAI-AGAPI-055 — HIGH — Teammate context exposure and compute
`/chat/teammate/{name}` invokes the LLM with internal teammate instructions and either live world state or detailed trust state; no caller identity or quota exists.

### KAI-AGAPI-056 — HIGH — Swarm compute and reputation mutation
`/chat/swarm` runs the full multi-stage CognitiveFSM pipeline and saves teammate reputation, allowing anonymous cost amplification and persistent behavioural-state changes.

### KAI-AGAPI-057 — HIGH — Watchdog/FSM control exposure
`/watchdog/check` triggers active service checks and converts returned event names into FSM events. Anonymous callers can repeatedly alter system state and probe internal availability.

### KAI-AGAPI-058 — HIGH — Caller assertions forwarded to vault authority
Vault export forwards `conviction` and `requester` exactly as supplied, allowing the downstream service to mistake anonymous assertions for trusted decision metadata.

---

## Checkpoint, logging and lifecycle defects

### KAI-AGAPI-059 — MEDIUM — Weak log-query controls
`limit`, `level` and `since` are not range/schema constrained. Negative slicing produces surprising results and arbitrary level strings create inconsistent query behaviour.

### KAI-AGAPI-060 — MEDIUM — Audit-tail request is full-file work
`TrustCore.audit_tail()` reads and parses the complete audit file before slicing the requested tail. Repeated anonymous calls amplify I/O and malformed lines can fail the endpoint.

### KAI-AGAPI-061 — MEDIUM — Checkpoint enumeration/detail exposure
List limits, checkpoint IDs and diff IDs are caller-controlled; metadata and full checkpoint content reveal breaker, guard, budget and conviction-override history.

### KAI-AGAPI-062 — HIGH — Unsigned checkpoint state
Checkpoint JSON files have no MAC/signature, trusted ownership verification or monotonic revision. Local tampering becomes live state during restore.

### KAI-AGAPI-063 — HIGH — Misleading partial restore
Checkpoints capture breakers, error guards, error budget and conviction overrides, but the restore route restores only breaker fields and still returns `status="ok"` as a time-travel rollback.

### KAI-AGAPI-064 — HIGH — Restored fields are unvalidated
`state`, `failures` and `opened_at` are assigned directly from JSON. Unknown states, negative/extreme failures and non-finite timestamps can enter live circuit breakers.

### KAI-AGAPI-065 — MEDIUM — Non-transactional blocking checkpoint I/O
Create, load, list, diff, restore and delete perform synchronous filesystem operations inside async handlers. Complete files are written directly without locks/fsync/atomic replacement.

### KAI-AGAPI-066 — HIGH — Startup tasks are outside lifecycle management
LLM warm-up and the infinite proactive observer are created with raw `asyncio.create_task()` and no retained reference.

### KAI-AGAPI-067 — HIGH — Shutdown does not stop startup loops
Shutdown drains only `_cleanup_mgr`. It does not cancel or await warm-up/proactive-observer tasks, so termination may abandon work or leave writes in progress.

### KAI-AGAPI-068 — HIGH — Unbounded cleanup-task admission
`_CleanupTaskManager.submit()` creates every requested task and stores it in an unbounded set. Chat finalisers, skill hunts, ritual proposals, snapshots and other paths can exhaust memory/worker resources.

### KAI-AGAPI-069 — MEDIUM — Background exceptions are unobserved
The done callback only discards the task; it does not call `task.exception()` or emit structured failure state. Many submitted coroutines also suppress errors internally.

### KAI-AGAPI-070 — MEDIUM — Deprecated lifecycle mechanism
The application uses `@app.on_event("startup"/"shutdown")` rather than a lifespan context that owns resources and background tasks.

### KAI-AGAPI-071 — MEDIUM — Readiness-blind health
Health considers only whether two circuit breakers are open. It can return `ok` while LLMs, memory, Tool Gate, trust storage, proactive observer or startup tasks are absent/failed.

### KAI-AGAPI-072 — MEDIUM — Connection churn across hot paths
The application repeatedly creates new `httpx.AsyncClient` instances for context sources, alerts, routing, Gate requests, vault operations and sensory probes instead of using lifecycle-managed bounded pools.

---

## Batch totals

- Findings: **72**
- Critical: **17**
- High: **46**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,114**
- Critical: **104**
- High: **457**
- Medium: **550**
- Low: **3**

## Files materially reviewed

`agentic/app.py`, `docker-compose.full.yml`, `common/auth.py`, `common/runtime.py`, `agentic/kai_config.py`, and direct integration confirmation against `agentic/web_scout.py`.
