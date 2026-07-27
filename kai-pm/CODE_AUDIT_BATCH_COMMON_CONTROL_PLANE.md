# Kai Code Audit — Common Control Plane Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CONTROL-001 | HIGH | Runtime policy is loaded from an unsigned, environment-selectable local file |
| KAI-CONTROL-002 | HIGH | Missing or corrupt policy does not reliably fail closed |
| KAI-CONTROL-003 | HIGH | Unknown tools default to MEDIUM risk instead of deny/high risk |
| KAI-CONTROL-004 | HIGH | Policy structure, ranges and cross-field invariants are not validated |
| KAI-CONTROL-005 | MEDIUM | The optional YAML fallback cannot parse the documented policy format |
| KAI-CONTROL-006 | MEDIUM | `policy_hash` does not attest that the policy parsed or became effective |
| KAI-CONTROL-007 | MEDIUM | Policy bytes are re-read outside the protected load path |
| KAI-CONTROL-008 | HIGH | Rate limits are global per endpoint rather than scoped to an authenticated actor |
| KAI-CONTROL-009 | HIGH | Per-process rate limits are bypassable across workers and replicas |
| KAI-CONTROL-010 | HIGH | Rate-limit check and append operations are concurrency-unsafe |
| KAI-CONTROL-011 | HIGH | The burst multiplier expands the full one-minute allowance rather than a short burst |
| KAI-CONTROL-012 | MEDIUM | Wall-clock use and process restart allow inconsistent/reset rate windows |
| KAI-CONTROL-013 | MEDIUM | Invalid policy numbers can crash or invert rate-limit behaviour at request time |
| KAI-CONTROL-014 | MEDIUM | Rate-limit responses omit retry metadata and disclose internal counters |
| KAI-CONTROL-015 | HIGH | Feature flags claim safe-by-default while consequential capabilities default ON |
| KAI-CONTROL-016 | HIGH | Misspelled environment values silently disable safety-critical flags |
| KAI-CONTROL-017 | HIGH | Public mutable registry objects allow runtime flag replacement |
| KAI-CONTROL-018 | HIGH | Flag decisions can change at runtime without policy revision or audit evidence |
| KAI-CONTROL-019 | MEDIUM | Flag dependencies and required preconditions are not enforced |
| KAI-CONTROL-020 | MEDIUM | Pending or no-op mechanisms are represented as enabled capabilities |
| KAI-CONTROL-021 | HIGH | Token counting uses one GPT tokenizer for every registered model |
| KAI-CONTROL-022 | HIGH | Character-count fallback is not conservatively safe for all languages and code |
| KAI-CONTROL-023 | HIGH | Static model capability cards are not verified against deployed artefacts |
| KAI-CONTROL-024 | HIGH | Arbitrary environment-selected model names are accepted without readiness checks |
| KAI-CONTROL-025 | MEDIUM | Prefix matching assigns capabilities to unverified lookalike model names |
| KAI-CONTROL-026 | MEDIUM | Message token counting assumes string-only content and fixed OpenAI overhead |
| KAI-CONTROL-027 | MEDIUM | Tokenizer failure and heuristic fallback are invisible to callers and readiness |
| KAI-CONTROL-028 | MEDIUM | Specialist routing uses manipulable substring voting without safety precedence |
| KAI-CONTROL-029 | HIGH | Tier-1 models lose WORK-mode and other richer system instructions entirely |
| KAI-CONTROL-030 | HIGH | Untrusted context and evidence are inserted directly into the system prompt |
| KAI-CONTROL-031 | HIGH | Conversation history may inject additional arbitrary system-role messages |
| KAI-CONTROL-032 | HIGH | Fact-check, planning, reflection and intent prompts do not delimit untrusted text |
| KAI-CONTROL-033 | MEDIUM | Prompt builders apply no field, aggregate or context-window bounds |
| KAI-CONTROL-034 | MEDIUM | Unknown modes silently receive PUB behaviour |
| KAI-CONTROL-035 | MEDIUM | Custom prompt templates are mutable and unaudited; configured template directory is unused |
| KAI-CONTROL-036 | HIGH | Safety-control rejections use HTTP 422, which existing resilience logic treats as success |
| KAI-CONTROL-037 | MEDIUM | Structured error details remain unbounded while diagnostic context is discarded |
| KAI-CONTROL-038 | HIGH | A/B quality scoring rewards vocabulary diversity and penalises honest uncertainty |
| KAI-CONTROL-039 | HIGH | A/B logging is enabled by default and stores session/model metadata in plaintext |
| KAI-CONTROL-040 | MEDIUM | Prompt identity uses only a collision-prone 32-bit hash of the first 200 characters |
| KAI-CONTROL-041 | MEDIUM | A/B file writes are synchronous and protected only within one process |
| KAI-CONTROL-042 | MEDIUM | A/B logging failures are silently suppressed and create undetectable evidence gaps |

---

## Policy loader: `common/policy.py`, `security/policy.yml`

### KAI-CONTROL-001 — HIGH — Unsigned environment-selectable policy source
**Issue:** `SOVEREIGN_POLICY_PATH` selects any local path. The file is trusted after ordinary parsing; ownership, permissions, symlinks, signature, trusted revision and deployment identity are not verified.  
**Risk:** A compromised process, volume or deployment setting can replace the system’s claimed single source of truth for verifier thresholds, tool risk, limits and governance.  
**Recommendation:** Load a signed, schema-validated immutable policy artefact from an approved root and verify ownership, mode, digest and revision before readiness.  
**Status:** OPEN

### KAI-CONTROL-002 — HIGH — “Fail closed” defaults are not consistently restrictive
**Issue:** On missing/corrupt policy, `POLICY` remains empty. Accessors then supply operational defaults: rate limits default to 60, circuit breakers remain active with normal thresholds, evidence weights remain usable, and unknown tools are MEDIUM. Mode and verifier dictionaries simply become empty.  
**Risk:** Callers receive a mixture of permissive assumptions, missing controls and runtime-specific fallbacks rather than one locked governance state.  
**Recommendation:** Expose policy-unavailable readiness failure and deny consequential operations until an explicitly verified emergency policy is loaded.  
**Status:** OPEN

### KAI-CONTROL-003 — HIGH — Unknown tools are not denied
**Issue:** `risk_tier_for_tool` returns `MEDIUM` when a tool is absent from policy.  
**Risk:** Newly introduced, misspelled or attacker-influenced tool names avoid explicit HIGH classification and co-sign requirements.  
**Recommendation:** Default unknown tools to denied or the most restrictive tier.  
**Status:** OPEN

### KAI-CONTROL-004 — HIGH — Policy semantics are unvalidated
**Issue:** No schema checks threshold ranges/order, evidence weights, duplicate tool membership, required keys, types, mode completeness or consistency between comments and runtime.  
**Risk:** A syntactically valid but unsafe policy can silently weaken gates or fail later on live requests.  
**Recommendation:** Validate a versioned strict schema and all cross-field invariants before publishing policy.  
**Status:** OPEN

### KAI-CONTROL-005 — MEDIUM — Optional loader cannot parse normal YAML
**Issue:** Without PyYAML, the fallback removes comments and attempts `json.loads`; the repository’s ordinary YAML is not JSON, so the loader returns `{}`.  
**Risk:** Dependency drift converts a valid policy into policy-unavailable defaults rather than preserving equivalent behaviour.  
**Recommendation:** Make a real YAML parser mandatory or compile the policy to a verified canonical format during build.  
**Status:** OPEN

### KAI-CONTROL-006 — MEDIUM — Hash can legitimise an ineffective policy
**Issue:** `policy_hash` hashes raw file bytes whether parsing succeeded or not. It does not include parsed canonical content, schema result or runtime defaults applied.  
**Risk:** Dashboards/logs can display a policy hash even while the service is operating on empty/default policy.  
**Recommendation:** Publish separate raw digest, canonical effective-policy digest and verified-load state.  
**Status:** OPEN

### KAI-CONTROL-007 — MEDIUM — Unprotected second read
**Issue:** After the load attempt, `_POLICY_PATH.read_bytes()` is performed separately and outside the exception handler. The file may change/disappear between existence checks and reads.  
**Risk:** Import can crash or the reported hash can refer to different bytes than the parsed policy.  
**Recommendation:** Read once into bytes, hash those exact bytes and parse from the same immutable buffer.  
**Status:** OPEN

---

## Rate limiting: `common/rate_limit.py`

### KAI-CONTROL-008 — HIGH — One caller can consume everyone’s allowance
**Issue:** Counters are keyed only by endpoint name; no authenticated principal, API key, IP, session or tenant participates in the key.  
**Risk:** An attacker can exhaust the shared endpoint window and deny legitimate users while remaining indistinguishable from them.  
**Recommendation:** Scope limits by authenticated actor and add separately governed global capacity limits.  
**Status:** OPEN

### KAI-CONTROL-009 — HIGH — Multi-worker bypass
**Issue:** Every process owns an independent `_windows` dictionary.  
**Risk:** Effective allowance multiplies by worker/replica count and resets whenever traffic is routed to a fresh process.  
**Recommendation:** Use atomic Redis/database counters with consistent principal and route keys.  
**Status:** OPEN

### KAI-CONTROL-010 — HIGH — Check-and-record race
**Issue:** Pruning, count comparison and timestamp append are separate unsynchronised operations.  
**Risk:** Concurrent requests can all observe capacity and pass beyond the intended cap; snapshots can race with mutations.  
**Recommendation:** Use one atomic distributed sliding-window/token-bucket operation.  
**Status:** OPEN

### KAI-CONTROL-011 — HIGH — Burst logic is not a short burst
**Issue:** `burst_limit = limit * BURST_MULTIPLIER` is compared against the same 60-second timestamp list. There is no shorter burst window or sustained base-rate enforcement.  
**Risk:** The configured per-minute limit is effectively doubled for all traffic, contradicting policy wording and weakening protection.  
**Recommendation:** Implement separate burst capacity and refill rate using a token bucket or dual windows.  
**Status:** OPEN

### KAI-CONTROL-012 — MEDIUM — Non-monotonic and resettable timing
**Issue:** Windows use `time.time()` and process memory. Clock corrections can retain or prematurely expire entries; restart erases the entire history.  
**Risk:** Limits behave inconsistently around clock changes and deployment churn.  
**Recommendation:** Use server-side monotonic/Redis time and durable bounded counters.  
**Status:** OPEN

### KAI-CONTROL-013 — MEDIUM — Unsafe numeric configuration
**Issue:** Policy values and burst multiplier are converted directly. Negative, non-finite, non-numeric or extreme values can disable limiting, deny all calls or raise during live requests.  
**Risk:** A malformed policy causes availability or enforcement failure after startup.  
**Recommendation:** Validate numeric ranges during policy activation.  
**Status:** OPEN

### KAI-CONTROL-014 — MEDIUM — Weak 429 contract
**Issue:** The error exposes internal count/base/burst values and omits `Retry-After`, reset time, structured error code and caller scope.  
**Risk:** Clients cannot back off reliably and receive unnecessary internal policy detail.  
**Recommendation:** Return stable rate-limit headers and a minimal typed response.  
**Status:** OPEN

---

## Feature flags: `common/feature_flags.py`

### KAI-CONTROL-015 — HIGH — Consequential capabilities default enabled
**Issue:** The module states every flag defaults OFF, but numerous flags default `True`, including proactive agent activity, memory consolidation, security self-audit, world-model persistence, sensory learning, skill hunting, scheduling, House Doctor, swarm, vault sync, hypothesis generation and forecasting.  
**Risk:** New deployments activate broad autonomous and state-changing behaviour without explicit operator opt-in.  
**Recommendation:** Default every consequential or experimental capability OFF and activate through signed deployment policy.  
**Status:** OPEN

### KAI-CONTROL-016 — HIGH — Invalid values disable safety flags
**Issue:** Any present value outside `1/true/yes/on` becomes false. A typo such as `FF_CONSCIENCE_FILTER=ture` silently disables the control.  
**Risk:** Misconfiguration fails open for safety mechanisms while looking explicitly configured.  
**Recommendation:** Accept only strict Boolean values and fail startup on invalid input.  
**Status:** OPEN

### KAI-CONTROL-017 — HIGH — Registry can be overwritten in-process
**Issue:** `FLAGS` exposes the underlying dictionary and `register_flag` replaces any existing name without authorisation, duplicate rejection or locking.  
**Risk:** Any imported plugin/module can alter defaults or descriptions of safety and autonomy controls for the whole process.  
**Recommendation:** Freeze a validated registry after startup and require signed versioned extension manifests.  
**Status:** OPEN

### KAI-CONTROL-018 — HIGH — Decisions mutate without governance evidence
**Issue:** `is_enabled` reads `os.environ` every call and runtime code can mutate both environment and registry. No change event, policy revision, author or timestamp is recorded.  
**Risk:** Behaviour can change mid-request/mid-session without an auditable configuration transition.  
**Recommendation:** Resolve one immutable flag snapshot at startup and activate changes through controlled restart/revision.  
**Status:** OPEN

### KAI-CONTROL-019 — MEDIUM — Dependencies are comments only
**Issue:** Flags describing prerequisites (for example causal surprise requiring causal world model) are independently evaluated; no dependency graph or mutual-exclusion validation exists.  
**Risk:** Unsupported combinations produce partial behaviour and misleading health/capability state.  
**Recommendation:** Validate a formal dependency graph before readiness.  
**Status:** OPEN

### KAI-CONTROL-020 — MEDIUM — Enabled does not mean implemented
**Issue:** Several descriptions explicitly state pending data/hardware/full implementation, while related flags are enabled or represented alongside operational capabilities.  
**Risk:** Dashboards and callers can treat interfaces/no-ops as functioning cognitive controls.  
**Recommendation:** Track `implemented`, `configured`, `ready` and `enabled` separately.  
**Status:** OPEN

---

## Model registry and token accounting: `common/model_registry.py`

### KAI-CONTROL-021 — HIGH — Wrong tokenizer for all models
**Issue:** One `cl100k_base` encoder is initialised globally. The `model` parameter and each spec’s `tiktoken_encoding` are ignored. Qwen, Llama, Mistral, Gemma, Yi and other model-native tokenisers are never used.  
**Risk:** Context budgets described as accurate can undercount and overflow actual model windows or over-trim useful context.  
**Recommendation:** Use the exact deployed model tokenizer/version and verify accounting against backend token usage.  
**Status:** OPEN

### KAI-CONTROL-022 — HIGH — Character heuristic is not universally conservative
**Issue:** Fallback assumes about 3.5 characters per token and claims safety. CJK text, unusual Unicode, source code and structured data can tokenize at far more tokens per character.  
**Risk:** Fallback mode can materially undercount and defeat context-limit enforcement.  
**Recommendation:** Apply a demonstrably worst-case bound or refuse large-context operations without a verified tokenizer.  
**Status:** OPEN

### KAI-CONTROL-023 — HIGH — Capability cards are assertions, not probes
**Issue:** Context windows, JSON/vision support, quality tiers and timeouts are hard-coded and not tied to model manifests, backend metadata, quantisation or runtime tests.  
**Risk:** Prompt construction and budgets can rely on capabilities the deployed artefact does not possess.  
**Recommendation:** Bind specs to immutable model digests and verify capabilities during controlled startup.  
**Status:** OPEN

### KAI-CONTROL-024 — HIGH — Arbitrary model selection
**Issue:** Active and specialist model names come directly from environment variables. Availability, provider, digest and registry membership are not required before return/use.  
**Risk:** Typoed or attacker-controlled configuration routes requests to unavailable/unapproved models while metadata falls back independently.  
**Recommendation:** Require approved model IDs and ready backend bindings.  
**Status:** OPEN

### KAI-CONTROL-025 — MEDIUM — Prefix capability inheritance
**Issue:** Any unknown name beginning with a registry key inherits that key’s capabilities.  
**Risk:** Lookalike, custom or materially different derivatives can receive unsupported context/JSON/quality assumptions.  
**Recommendation:** Match exact immutable artefact identities or explicitly registered aliases.  
**Status:** OPEN

### KAI-CONTROL-026 — MEDIUM — Message accounting is format-incomplete
**Issue:** `count_messages_tokens` assumes `content` is a string and adds fixed OpenAI-style overhead. Tool calls, names, multimodal lists/images and backend-specific framing are not handled.  
**Risk:** Vision/tool messages fail or are substantially miscounted.  
**Recommendation:** Implement provider/schema-specific accounting with strict supported content types.  
**Status:** OPEN

### KAI-CONTROL-027 — MEDIUM — Silent tokenizer downgrade
**Issue:** Import/encoding failures silently switch to the heuristic. No readiness, metric or result provenance indicates degraded accounting.  
**Risk:** Services continue under weaker limits while operators believe counting remains accurate.  
**Recommendation:** Expose tokenizer identity/degraded state and fail high-risk long-context operations closed.  
**Status:** OPEN

### KAI-CONTROL-028 — MEDIUM — Manipulable specialist detection
**Issue:** Routing counts raw substrings. Terms can match inside unrelated words (`api`, `sum`, etc.); ties and mixed consequential intents have no safety-priority rule.  
**Risk:** Crafted or ordinary phrasing can select an inappropriate specialist and its associated prompt/model.  
**Recommendation:** Use validated intent classification with action-risk precedence and token/word boundaries.  
**Status:** OPEN

---

## Prompt construction: `common/prompt_templates.py`

### KAI-CONTROL-029 — HIGH — Tier-1 path discards mode controls
**Issue:** If `quality_tier <= 1`, `build_system_prompt` always uses the same minimal persona before checking WORK/PUB. WORK requirements about precision, UK financial sources, uncertainty and numerical/deadline caution disappear.  
**Risk:** The smallest/default models receive the least safety/domain instruction exactly when they need stronger scaffolding.  
**Recommendation:** Preserve mandatory governance and domain constraints at every model tier; trim style, not controls.  
**Status:** OPEN

### KAI-CONTROL-030 — HIGH — Untrusted material becomes system instruction
**Issue:** `extra_context`, `evidence` and `personality_note` are concatenated directly into the system message with no provenance, quoting or instruction/data separation.  
**Risk:** Poisoned memory, sensors, documents or caller data can override privileged behaviour through stored/context prompt injection.  
**Recommendation:** Keep system policy immutable and provide retrieved content through typed, strongly delimited untrusted data channels.  
**Status:** OPEN

### KAI-CONTROL-031 — HIGH — History may add arbitrary system messages
**Issue:** `build_chat_messages` accepts historical roles of `system` and appends them after the primary system prompt. No trusted-origin check exists.  
**Risk:** Caller/persisted history can insert new high-priority instructions and override current policy.  
**Recommendation:** Reject system-role history unless it is signed server-owned policy; normalise all prior content to user/assistant data.  
**Status:** OPEN

### KAI-CONTROL-032 — HIGH — Task prompts are injection-prone
**Issue:** Claim, evidence, goal, constraints, topic, context and wake text are interpolated into instruction text without explicit boundaries or escaping.  
**Risk:** Inputs can instruct the model to ignore output schemas, forge fact-check verdicts or alter task intent.  
**Recommendation:** Use structured messages/fields, delimit untrusted text and independently validate outputs.  
**Status:** OPEN

### KAI-CONTROL-033 — MEDIUM — No prompt budget enforcement
**Issue:** Builders apply no length, token, history-count or aggregate bounds and do not call the registry’s context-budget functions.  
**Risk:** Prompt assembly can exceed context windows before the backend call.  
**Recommendation:** Enforce section-specific limits and assert final provider-specific token count.  
**Status:** OPEN

### KAI-CONTROL-034 — MEDIUM — Unknown mode becomes PUB
**Issue:** Every non-WORK mode receives PUB instructions for tier 2+.  
**Risk:** Typoed or new restricted modes silently become casual/permissive rather than rejected.  
**Recommendation:** Validate an enum and fail closed on unknown modes.  
**Status:** OPEN

### KAI-CONTROL-035 — MEDIUM — Template governance is unfinished
**Issue:** Any code can replace `_CUSTOM_TEMPLATES` entries through `register_template`; no locking, versioning or provenance exists. `PROMPT_TEMPLATE_DIR` is read but never used.  
**Risk:** Runtime prompt behaviour is mutable and configuration suggests a capability that does not exist.  
**Recommendation:** Load signed immutable templates during startup and remove/implement unused configuration.  
**Status:** OPEN

---

## Structured errors: `common/errors.py`

### KAI-CONTROL-036 — HIGH — Safety rejection status collides with success-classifying resilience
**Issue:** Conviction, self-deception, verifier and adversary blocks map to HTTP 422. Existing `common.resilience.resilient_call` treats every status below 500 as a successful dependency call.  
**Risk:** A critical safety refusal can reset a circuit and flow into downstream business logic as a valid successful response.  
**Recommendation:** Use a typed internal result contract and make resilience classify safety blocks explicitly as failures/blocked outcomes.  
**Status:** OPEN

### KAI-CONTROL-037 — MEDIUM — Error detail/provenance contract is incomplete
**Issue:** `detail` is unbounded and may contain exception text, while the supplied `context` is omitted entirely from `to_dict` and no trace/correlation identifier is standardised.  
**Risk:** APIs can leak diagnostics while operators lose structured evidence needed to investigate the same event.  
**Recommendation:** Redact/bound public detail and store structured protected context under a trace ID.  
**Status:** OPEN

---

## A/B logging: `common/ab_log.py`

### KAI-CONTROL-038 — HIGH — Confidence style is mistaken for response quality
**Issue:** `net_quality_signal` rewards type-token diversity and subtracts points for hedging words such as “might”, “approximately” and “uncertain”. No correctness, evidence, user outcome or calibration is measured.  
**Risk:** Evaluation/model-selection processes can favour confident varied prose over accurate cautious answers, directly discouraging epistemic honesty.  
**Recommendation:** Score verified correctness, task success, calibration and human review; never treat uncertainty language as intrinsic error.  
**Status:** OPEN

### KAI-CONTROL-039 — HIGH — Plaintext logging defaults on
**Issue:** A/B logging is enabled unless explicitly disabled and writes timestamp, specialist/model, source, latency, session ID and usage/quality metadata to a normal JSONL file. No permission hardening, encryption, retention or deletion policy exists.  
**Risk:** Session-linked behavioural and model-use telemetry accumulates by default and is readable to processes/users with filesystem access.  
**Recommendation:** Default off, minimise/pseudonymise identifiers and use protected governed telemetry storage.  
**Status:** OPEN

### KAI-CONTROL-040 — MEDIUM — Weak prompt identifier
**Issue:** `prompt_hash` is only the first eight hex characters of SHA-256 over `prompt[:200]`. Prompts sharing the first 200 characters collide deterministically; 32-bit identifiers also collide at moderate volumes.  
**Risk:** Entries can be incorrectly grouped or attributed and the value cannot prove which complete prompt was evaluated.  
**Recommendation:** Hash canonical complete prompt metadata with a full cryptographic digest or privacy-preserving keyed identifier.  
**Status:** OPEN

### KAI-CONTROL-041 — MEDIUM — File append is not multi-process safe
**Issue:** Logging performs synchronous mkdir/open/write on the caller thread. The lock protects only threads in one process; multiple workers append independently without file locking or durable flush.  
**Risk:** Request latency increases and JSONL records can interleave, disappear on crash or vary across mounted filesystems.  
**Recommendation:** Send events to a bounded asynchronous durable telemetry pipeline.  
**Status:** OPEN

### KAI-CONTROL-042 — MEDIUM — Silent evidence loss
**Issue:** Every exception is reduced to a debug message and never propagated, counted or reflected in health.  
**Risk:** A/B datasets can have systematic missing periods while appearing complete, biasing later comparisons.  
**Recommendation:** Track durable delivery status, dropped-event counters and explicit dataset completeness windows.  
**Status:** OPEN

---

## Batch totals

- Findings: **42**
- Critical: **0**
- High: **23**
- Medium: **19**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **823**
- Critical: **86**
- High: **299**
- Medium: **435**
- Low: **3**

## Files materially reviewed in this batch

`common/policy.py`, `security/policy.yml`, `common/rate_limit.py`, `common/feature_flags.py`, `common/model_registry.py`, `common/prompt_templates.py`, `common/errors.py`, `common/ab_log.py`, with interaction confirmation against `common/resilience.py` and `agentic/conviction.py`.
