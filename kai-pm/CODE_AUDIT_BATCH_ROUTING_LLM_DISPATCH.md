# Kai Code Audit — Routing and LLM Dispatch Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records findings beyond `KAI-ROUTE-001` through `KAI-ROUTE-005` already present in `CODE_AUDIT_REGISTER_CONTINUED.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DISPATCH-001 | HIGH | Semantic routing selects the maximum single-anchor similarity without an ambiguity margin |
| KAI-DISPATCH-002 | HIGH | Router input size and embedding work are unbounded |
| KAI-DISPATCH-003 | HIGH | Caller-supplied prior route increases the selected route confidence |
| KAI-DISPATCH-004 | HIGH | Memory recall is hard-coded to the global `keeper` identity |
| KAI-DISPATCH-005 | HIGH | `dispatch_route` accepts but ignores the session ID |
| KAI-DISPATCH-006 | HIGH | Tax advisory ignores the user’s actual question |
| KAI-DISPATCH-007 | HIGH | Tax routing performs synchronous local financial-file processing inside an async path |
| KAI-DISPATCH-008 | HIGH | Reflection is classified as a bypass route despite triggering state-changing memory reflection |
| KAI-DISPATCH-009 | HIGH | Unknown or malformed verifier verdicts are formatted as authoritative fact-check results |
| KAI-DISPATCH-010 | MEDIUM | Raw memory text is returned without provenance, trust or sensitivity labelling |
| KAI-DISPATCH-011 | MEDIUM | Proactive review ignores request/session context and returns global nudges |
| KAI-DISPATCH-012 | MEDIUM | Dependency failures collapse to generic prose without machine-readable degraded state |
| KAI-DISPATCH-013 | HIGH | Unsigned Markdown files become active skill-routing definitions |
| KAI-DISPATCH-014 | HIGH | Skill security scanning is a bypassable literal blacklist |
| KAI-DISPATCH-015 | HIGH | Skill disablement metadata is ignored by the router loader |
| KAI-DISPATCH-016 | HIGH | Unloaded or pruned skills return on the next reload because files remain active |
| KAI-DISPATCH-017 | HIGH | Duplicate skill triggers and names are resolved by directory/file order |
| KAI-DISPATCH-018 | MEDIUM | Skill file count and content size are unbounded during import/reload |
| KAI-DISPATCH-019 | MEDIUM | Skill loading performs synchronous filesystem work in live async endpoints |
| KAI-DISPATCH-020 | MEDIUM | Loaded skills and last-used state are worker-local and concurrency-unsafe |
| KAI-DISPATCH-021 | MEDIUM | Negative or extreme stale-age values can unload every skill or disable pruning |
| KAI-DISPATCH-022 | MEDIUM | Skill inventory exposes filesystem paths, triggers and action instructions |
| KAI-DISPATCH-023 | MEDIUM | Literal word-boundary triggers produce unreliable phrase and punctuation matching |
| KAI-DISPATCH-024 | CRITICAL | Cloud API credentials are sent to arbitrary configured backend URLs |
| KAI-DISPATCH-025 | HIGH | Backend URLs and model identifiers are not constrained to approved destinations/artefacts |
| KAI-DISPATCH-026 | HIGH | Ollama availability checks fail open on every network or parsing failure |
| KAI-DISPATCH-027 | HIGH | Base-name matching treats different model tags/variants as the requested model |
| KAI-DISPATCH-028 | HIGH | Model-tag cache is global and not keyed by Ollama endpoint |
| KAI-DISPATCH-029 | HIGH | Fallback model availability is never checked |
| KAI-DISPATCH-030 | HIGH | Unknown or unavailable specialists return deterministic stubs as normal responses |
| KAI-DISPATCH-031 | HIGH | Empty/model-error output can be labelled as a live successful response |
| KAI-DISPATCH-032 | HIGH | Multi-specialist fan-out is unbounded and accepts duplicates |
| KAI-DISPATCH-033 | HIGH | Prompt, system message, temperature and token limits are weakly bounded or unvalidated |
| KAI-DISPATCH-034 | HIGH | LLM response bytes and JSON structures are unbounded and weakly schema-checked |
| KAI-DISPATCH-035 | HIGH | Automatic retries can duplicate billed/non-idempotent model requests after ambiguous failures |
| KAI-DISPATCH-036 | HIGH | Every retry creates a new HTTP client and connection pool |
| KAI-DISPATCH-037 | MEDIUM | Retry policy ignores common transient statuses and server `Retry-After` guidance |
| KAI-DISPATCH-038 | MEDIUM | Backend exception details are returned as model output text |
| KAI-DISPATCH-039 | MEDIUM | Global router configuration is captured at import time and does not track deployment changes |
| KAI-DISPATCH-040 | MEDIUM | Cloud API keys are captured at import time without rotation/version support |
| KAI-DISPATCH-041 | HIGH | Streaming bypasses model availability checks and fallback logic |
| KAI-DISPATCH-042 | HIGH | Streaming accepts arbitrary role/message structures without policy validation |
| KAI-DISPATCH-043 | HIGH | Stream stalls and backend failures are emitted as ordinary assistant content |
| KAI-DISPATCH-044 | MEDIUM | Malformed streaming chunks are silently discarded |
| KAI-DISPATCH-045 | MEDIUM | Streaming output has no local aggregate byte/token enforcement |
| KAI-DISPATCH-046 | MEDIUM | Streaming uses generic timeouts rather than the selected model contract |
| KAI-DISPATCH-047 | HIGH | Warm-up reports completion even when the result is stub or error output |
| KAI-DISPATCH-048 | HIGH | Auto-pull can retrieve an arbitrary configured model from an arbitrary configured Ollama endpoint |
| KAI-DISPATCH-049 | MEDIUM | Model-pull status and completion are not validated |
| KAI-DISPATCH-050 | MEDIUM | Warm-up defaults on and performs external model traffic during startup |
| KAI-DISPATCH-051 | MEDIUM | Timeout, retry, backoff, cache and heartbeat configuration lacks safe-range validation |
| KAI-DISPATCH-052 | MEDIUM | Stub and error responses are written into A/B quality logs as model observations |

---

## Specialist router and skill routing: `agentic/router.py`, `agentic/app.py`

### KAI-DISPATCH-001 — HIGH — No semantic ambiguity margin
**Issue:** semantic routing uses the maximum similarity to any single anchor and accepts the winner at 0.45. It does not compare the top two route scores or require route-level consensus.  
**Risk:** ambiguous/mixed consequential requests can be confidently assigned to a bypass route from one superficially similar anchor.  
**Recommendation:** require a safety-priority action detector, top-two margin and calibrated route thresholds.  
**Status:** OPEN

### KAI-DISPATCH-002 — HIGH — Unbounded classification cost
**Issue:** complete user input is passed to regexes and the embedding model without body, character or token limits.  
**Risk:** large requests consume CPU/memory and block routing workers.  
**Recommendation:** cap and normalise input before every classifier.  
**Status:** OPEN

### KAI-DISPATCH-003 — HIGH — Route confidence trusts caller context
**Issue:** `session_context["last_route"]` adds confidence, but the context is not authenticated or derived internally by this module.  
**Risk:** callers can increase a desired route’s score and resolve close classifications in its favour.  
**Recommendation:** accept only server-owned signed session state.  
**Status:** OPEN

### KAI-DISPATCH-004 — HIGH — Global memory identity
**Issue:** memory recall defaults to and is always called with `user_id="keeper"`.  
**Risk:** all sessions retrieve the same memory namespace, allowing cross-user/session disclosure and contamination.  
**Recommendation:** derive the principal from authenticated request context.  
**Status:** OPEN

### KAI-DISPATCH-005 — HIGH — Session isolation parameter is unused
**Issue:** `dispatch_route(..., session_id)` never reads `session_id`; route helpers do not receive it.  
**Risk:** callers may believe dispatch is session-scoped while memory, nudges and reflection are global.  
**Recommendation:** propagate an authenticated principal/session to every dependency.  
**Status:** OPEN

### KAI-DISPATCH-006 — HIGH — Tax question is ignored
**Issue:** `dispatch_tax_advisory(query)` never uses `query`; it always returns the same income, thresholds and generic suggestions.  
**Risk:** specific tax questions receive unrelated directive-looking output, creating false confidence in an answer that was never evaluated.  
**Recommendation:** explicitly classify supported questions or return a bounded data summary labelled as such.  
**Status:** OPEN

### KAI-DISPATCH-007 — HIGH — Financial file work blocks async serving
**Issue:** CSV/log loading and advisory calculation execute synchronously in the async dispatch path.  
**Risk:** large/slow files block the event loop and all requests.  
**Recommendation:** use validated transactional financial storage and bounded worker execution.  
**Status:** OPEN

### KAI-DISPATCH-008 — HIGH — State-changing reflection bypasses action governance
**Issue:** REFLECT is marked `bypass_llm=True` and dispatches an unauthenticated POST to `/memory/reflect`. Reflection can create insights/state rather than merely read data, but no plan, conviction or trust gate is applied.  
**Risk:** a request classified as summarisation can mutate long-term cognitive state outside autonomous-action controls.  
**Recommendation:** distinguish read-only summary from state mutation and gate the latter.  
**Status:** OPEN

### KAI-DISPATCH-009 — HIGH — Verifier output is not validated
**Issue:** fact-check formatting accepts any `verdict` string and evidence value from a 200 response.  
**Risk:** malformed/compromised verifier responses are displayed as a Fact Check Result without a strict PASS/REPAIR/FAIL/UNKNOWN contract.  
**Recommendation:** validate a signed typed verifier schema and fail closed on unknown values.  
**Status:** OPEN

### KAI-DISPATCH-010 — MEDIUM — Memory output lacks provenance
**Issue:** stored text/category is returned directly as numbered memory content without source identity, date, confidence, integrity or privacy classification.  
**Risk:** stale/poisoned/private records appear as equally trustworthy recalled facts.  
**Recommendation:** include bounded provenance and trust labels and enforce access policy.  
**Status:** OPEN

### KAI-DISPATCH-011 — MEDIUM — Global proactive output
**Issue:** proactive review takes no query, user or session and returns all current nudges from the shared service.  
**Risk:** unrelated/private reminders can be disclosed and the requested review scope is ignored.  
**Recommendation:** scope nudges to the authenticated principal and request window.  
**Status:** OPEN

### KAI-DISPATCH-012 — MEDIUM — Failure state is prose-only
**Issue:** dispatch helpers suppress exceptions and return natural-language failure strings indistinguishable from ordinary assistant output.  
**Risk:** callers cannot reliably detect degraded dependencies or prevent the text entering memory/evaluation as a valid answer.  
**Recommendation:** return typed success/degraded/error results.  
**Status:** OPEN

### KAI-DISPATCH-013 — HIGH — Unsigned files become routing capabilities
**Issue:** every Markdown file in `/skills` and `data/skills` is parsed and loaded at import/reload without signature, trusted source, review identity or immutable digest.  
**Risk:** filesystem writes, including Skill Hunter output, alter live request matching and action instructions.  
**Recommendation:** load only signed approved skill manifests through an activation registry.  
**Status:** OPEN

### KAI-DISPATCH-014 — HIGH — Blacklist scanner is not a security boundary
**Issue:** scanning detects a short set of literal strings such as `curl`, `exec(` and `subprocess.`. Equivalent instructions, encoded text, URLs, alternate libraries, spacing or natural-language harmful actions pass.  
**Risk:** unsafe skill instructions are labelled safe and loaded.  
**Recommendation:** use allowlisted typed capabilities and sandboxed review, not content blacklisting.  
**Status:** OPEN

### KAI-DISPATCH-015 — HIGH — Disabled skills remain loadable
**Issue:** the router reads Markdown only and never checks Skill Hunter sidecar metadata containing `disabled`, probation, package or error state.  
**Risk:** a skill marked disabled by the safety system is reactivated in routing.  
**Recommendation:** make one authoritative loader enforce signed status/revocation.  
**Status:** OPEN

### KAI-DISPATCH-016 — HIGH — Unload/prune is temporary
**Issue:** unload and stale pruning remove only in-memory objects. The source files remain and `load_skills()` restores them.  
**Risk:** operational controls report success but do not durably disable the capability.  
**Recommendation:** revoke/quarantine the authoritative artefact atomically.  
**Status:** OPEN

### KAI-DISPATCH-017 — HIGH — Trigger shadowing is order-dependent
**Issue:** duplicate names/triggers are accepted. Directories and sorted filenames determine which matching skill is returned first.  
**Risk:** a new file can shadow a legitimate capability with attacker-controlled action/response instructions.  
**Recommendation:** require unique immutable names/triggers and reject ambiguity.  
**Status:** OPEN

### KAI-DISPATCH-018 — MEDIUM — Import/reload allocation is unbounded
**Issue:** all `*.md` files are read fully and parsed; file count, per-file size, trigger count and aggregate action/response size are unrestricted.  
**Risk:** large skill directories delay startup/reload and exhaust memory.  
**Recommendation:** enforce strict artefact and registry limits.  
**Status:** OPEN

### KAI-DISPATCH-019 — MEDIUM — Reload blocks the event loop
**Issue:** the async reload endpoint performs directory scans and full file reads synchronously.  
**Risk:** slow storage blocks all Agentic requests.  
**Recommendation:** activate a prepared registry atomically through a bounded worker.  
**Status:** OPEN

### KAI-DISPATCH-020 — MEDIUM — Skill state is volatile and races
**Issue:** `_loaded_skills` and `_skill_last_used` are process-local mutable globals with no locks. Reload, match, unload and prune can race; workers disagree.  
**Risk:** routing and safety status vary by request/worker.  
**Recommendation:** use one immutable versioned shared registry.  
**Status:** OPEN

### KAI-DISPATCH-021 — MEDIUM — Invalid pruning policy
**Issue:** `max_age_days` accepts any numeric-like value; negative values make the cutoff future and prune every skill. Extreme values produce nonsensical retention.  
**Risk:** crafted/mistaken calls cause capability denial.  
**Recommendation:** validate a bounded positive policy.  
**Status:** OPEN

### KAI-DISPATCH-022 — MEDIUM — Capability internals are exposed
**Issue:** inventory/match APIs return source paths, trigger phrases, action text and response templates.  
**Risk:** callers can map and deliberately trigger/shadow capabilities.  
**Recommendation:** restrict detailed skill metadata to authorised administrators.  
**Status:** OPEN

### KAI-DISPATCH-023 — MEDIUM — Trigger semantics are unreliable
**Issue:** every literal trigger is wrapped with `\b`; phrases ending/starting in punctuation or non-word characters do not have predictable boundaries, and no normalisation beyond case occurs.  
**Risk:** expected requests miss skills while unintended text matches.  
**Recommendation:** use a tested typed intent matcher.  
**Status:** OPEN

---

## Unified LLM router: `common/llm.py`

### KAI-DISPATCH-024 — CRITICAL — API-key exfiltration through backend configuration
**Issue:** Groq/OpenRouter keys are attached based solely on the specialist name. The corresponding backend URL is environment-controlled and is not restricted to the provider host or even validated beyond string use.  
**Risk:** compromised/mistaken configuration sends bearer credentials and prompts to an attacker-controlled URL.  
**Recommendation:** pin provider HTTPS hosts/TLS identity and bind each secret to one approved destination.  
**Status:** OPEN — immediate remediation required

### KAI-DISPATCH-025 — HIGH — Untrusted routing destinations and model identities
**Issue:** arbitrary backend URLs and model strings are accepted without scheme, host, port, TLS, artefact digest or registry validation.  
**Risk:** prompts/system context are routed to unintended services and unapproved models.  
**Recommendation:** activate a signed allowlisted backend/artefact manifest.  
**Status:** OPEN

### KAI-DISPATCH-026 — HIGH — Availability check fails open
**Issue:** any Ollama connection, status or JSON error returns `True`.  
**Risk:** routing proceeds as though the model is available when the authority cannot be queried.  
**Recommendation:** return unavailable/unknown and let a typed failover policy decide.  
**Status:** OPEN

### KAI-DISPATCH-027 — HIGH — Model variants are conflated
**Issue:** `_check_model_in_tags` accepts any tag sharing the model base before `:`.  
**Risk:** a different size, quantisation, fine-tune or incompatible variant is treated as the requested model.  
**Recommendation:** require exact immutable model digest/tag.  
**Status:** OPEN

### KAI-DISPATCH-028 — HIGH — Cache crosses endpoints
**Issue:** one module-global tags cache is not keyed by Ollama base URL. Environment/test endpoint changes reuse the previous server’s model list until expiry.  
**Risk:** availability decisions refer to the wrong backend.  
**Recommendation:** key cache by validated endpoint and generation.  
**Status:** OPEN

### KAI-DISPATCH-029 — HIGH — Fallback is not validated
**Issue:** when the requested local model is absent, the router assigns `_OLLAMA_MODEL` without checking that the fallback is present or compatible.  
**Risk:** requests fail later or silently use a different capability than promised.  
**Recommendation:** select only a fresh verified compatible fallback.  
**Status:** OPEN

### KAI-DISPATCH-030 — HIGH — Missing specialists become successful-looking stubs
**Issue:** any name without a configured URL returns a deterministic `LLMResponse(source="stub")` rather than an unavailable result.  
**Risk:** reasoning/consensus pipelines can consume canned text as model evidence.  
**Recommendation:** prohibit stubs from production decisions and return typed unavailable.  
**Status:** OPEN

### KAI-DISPATCH-031 — HIGH — Failed output can retain live status
**Issue:** empty text and JSON-looking model errors are converted to bracketed text by `_validate_llm_response`, but the response is returned with `source="live"`.  
**Risk:** callers count an empty/error response as successful live inference.  
**Recommendation:** validate schema/content and return a typed error source.  
**Status:** OPEN

### KAI-DISPATCH-032 — HIGH — Unbounded parallel model fan-out
**Issue:** `query_multi` creates one task per list element with no item limit, deduplication, quota or concurrency semaphore.  
**Risk:** one call creates arbitrary external/model traffic and memory use.  
**Recommendation:** use a small approved distinct set with bounded concurrency.  
**Status:** OPEN

### KAI-DISPATCH-033 — HIGH — Request generation controls are unvalidated
**Issue:** prompt/system strings have no size/token bounds; temperature and max tokens accept invalid, negative or extreme values.  
**Risk:** context overflow, excessive cost/load and backend-specific undefined behaviour.  
**Recommendation:** enforce model-specific bounded schemas before dispatch.  
**Status:** OPEN

### KAI-DISPATCH-034 — HIGH — Response parsing is unbounded and permissive
**Issue:** complete response bodies/JSON are materialised without byte/depth limits. Choices, message content, usage and model fields are not validated against a strict schema.  
**Risk:** malformed/oversized backends exhaust memory or inject invalid output/usage data.  
**Recommendation:** enforce response limits and typed provider contracts.  
**Status:** OPEN

### KAI-DISPATCH-035 — HIGH — Ambiguous retries duplicate inference
**Issue:** connection/timeout failures are retried even though the provider may have accepted/generated/billed the preceding request. No request ID/idempotency key is supplied.  
**Risk:** costs and side effects such as provider logging/tool execution can be repeated.  
**Recommendation:** use provider-supported idempotency and durable operation status; avoid blind retries after ambiguous commit.  
**Status:** OPEN

### KAI-DISPATCH-036 — HIGH — Retry connection churn
**Issue:** every attempt creates a new `AsyncClient` and pool.  
**Risk:** retries amplify DNS/TCP/TLS/socket pressure.  
**Recommendation:** reuse lifecycle-managed clients per trust zone/provider.  
**Status:** OPEN

### KAI-DISPATCH-037 — MEDIUM — Incomplete retry policy
**Issue:** only 429 and 503 are status-retried; 502/504 and other transient failures are not. `Retry-After` is ignored.  
**Risk:** unnecessary failures or abusive retry timing.  
**Recommendation:** use provider-specific typed retry rules and jitter.  
**Status:** OPEN

### KAI-DISPATCH-038 — MEDIUM — Backend diagnostics become assistant text
**Issue:** final exceptions are embedded in `[error: ...]`; streaming yields `[LLM error: ...]`.  
**Risk:** internal URLs, certificates and provider diagnostics leak and can enter memory.  
**Recommendation:** return stable error codes and protected traces.  
**Status:** OPEN

### KAI-DISPATCH-039 — MEDIUM — Import-time backend snapshot
**Issue:** the global router and default maps are constructed at import; later environment/secret/config changes are ignored.  
**Risk:** rotation/reconfiguration appears applied but traffic continues using stale destinations/models.  
**Recommendation:** use one versioned configuration lifecycle.  
**Status:** OPEN

### KAI-DISPATCH-040 — MEDIUM — Secret rotation is unsupported
**Issue:** API keys are captured into `_API_KEY_MAP` at import with no key ID, expiry or refresh.  
**Risk:** revoked keys remain in use until process restart and audit cannot identify the key version.  
**Recommendation:** obtain short-lived credentials through a secret provider.  
**Status:** OPEN

### KAI-DISPATCH-041 — HIGH — Streaming has divergent model safety
**Issue:** `stream` does not call `ensure_model_available`, perform fallback or verify the exact model.  
**Risk:** non-streaming and streaming routes use different readiness/identity behaviour.  
**Recommendation:** share one validated dispatch preparation path.  
**Status:** OPEN

### KAI-DISPATCH-042 — HIGH — Arbitrary role messages reach the model
**Issue:** streaming accepts a list of ordinary dictionaries and forwards all roles/content unchanged.  
**Risk:** untrusted history can inject system/developer/tool roles and override policy.  
**Recommendation:** validate server-owned role provenance and normalise untrusted data.  
**Status:** OPEN

### KAI-DISPATCH-043 — HIGH — Transport failures are normal content
**Issue:** stalls and exceptions are yielded as plain text chunks rather than terminating with a typed error.  
**Risk:** callers concatenate failure markers into assistant answers, memory and trust evidence.  
**Recommendation:** separate stream data from terminal status/error channels.  
**Status:** OPEN

### KAI-DISPATCH-044 — MEDIUM — Malformed chunks vanish
**Issue:** JSON/delta parsing exceptions are silently ignored.  
**Risk:** truncated/corrupt streams appear complete without a data-integrity warning.  
**Recommendation:** fail the stream or mark it incomplete.  
**Status:** OPEN

### KAI-DISPATCH-045 — MEDIUM — No local output cap
**Issue:** the client trusts the backend to respect `max_tokens`; it does not cap cumulative bytes/tokens received.  
**Risk:** a faulty/hostile endpoint streams indefinitely until heartbeat/read timeout while continuously sending data.  
**Recommendation:** enforce a local aggregate cap and terminate.  
**Status:** OPEN

### KAI-DISPATCH-046 — MEDIUM — Stream timeout ignores model profile
**Issue:** streaming uses generic global timeouts while non-streaming uses `_model_timeout`.  
**Risk:** slow valid models are cut or fast models occupy capacity too long depending on path.  
**Recommendation:** use one validated model-specific deadline contract.  
**Status:** OPEN

### KAI-DISPATCH-047 — HIGH — Warm-up false success
**Issue:** warm-up logs completion for any returned response, including `source="error"` or `source="stub"`.  
**Risk:** startup/readiness logs imply the model loaded when inference failed or never occurred.  
**Recommendation:** require verified live output from the exact model.  
**Status:** OPEN

### KAI-DISPATCH-048 — HIGH — Unreviewed model acquisition
**Issue:** when enabled, warm-up posts an arbitrary configured model name to `/api/pull` on an arbitrary configured Ollama endpoint. No allowlist, digest, signature or storage/resource approval exists.  
**Risk:** deployment configuration triggers supply-chain downloads and substantial disk/network consumption.  
**Recommendation:** pre-provision signed pinned model artefacts through a reviewed deployment process.  
**Status:** OPEN

### KAI-DISPATCH-049 — MEDIUM — Pull completion is not established
**Issue:** `_pull_model` does not call `raise_for_status`, require a terminal success message or return a result. All exceptions are swallowed.  
**Risk:** warm-up proceeds without knowing whether acquisition succeeded.  
**Recommendation:** validate status, digest and final installation.  
**Status:** OPEN

### KAI-DISPATCH-050 — MEDIUM — Startup generates model traffic by default
**Issue:** `LLM_WARMUP_ENABLED` defaults true and causes tag lookup plus a model prompt during startup.  
**Risk:** restarts consume model/provider capacity and may send traffic before service policy/readiness is established.  
**Recommendation:** opt in through controlled deployment and coordinate one warm-up owner.  
**Status:** OPEN

### KAI-DISPATCH-051 — MEDIUM — Numeric configuration is unsafe
**Issue:** timeouts, retries, backoff, tag TTL and heartbeat values are parsed directly and accept zero, negative, non-finite or extreme values.  
**Risk:** tight retry loops, disabled timeouts, immediate stalls or startup exceptions.  
**Recommendation:** validate strict safe ranges.  
**Status:** OPEN

### KAI-DISPATCH-052 — MEDIUM — Invalid outputs contaminate A/B evidence
**Issue:** every query result, including stub/error/live-empty markers, is passed to `log_ab_entry`.  
**Risk:** model comparison datasets treat infrastructure failure and canned text as model-quality observations.  
**Recommendation:** log typed outcome separately and exclude invalid inference from quality scoring.  
**Status:** OPEN

---

## Batch totals

- Findings: **52**
- Critical: **1**
- High: **29**
- Medium: **22**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,090**
- Critical: **96**
- High: **438**
- Medium: **553**
- Low: **3**

## Files materially reviewed in this batch

`agentic/router.py`, `common/llm.py`, and the active skill/routing integration paths in `agentic/app.py`. Existing `KAI-ROUTE-001` through `KAI-ROUTE-005` were not duplicated.
