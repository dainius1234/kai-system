# Kai Code Audit — Common Model, Prompt, Rate-Limit and Telemetry Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `common/errors.py`, `common/model_registry.py`, `common/prompt_templates.py`, `common/rate_limit.py` and `common/ab_log.py`. Existing `common/auth.py`, policy, runtime and resilience findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-COMMONX-001 | HIGH | The documented structured-error standard is unused by active service code |
| KAI-COMMONX-002 | HIGH | Arbitrary error detail strings are intended for direct API return without redaction or size limits |
| KAI-COMMONX-003 | MEDIUM | Structured error context is accepted but discarded by serialisation |
| KAI-COMMONX-004 | MEDIUM | Error responses have no event ID, timestamp, retryability or protected trace reference |
| KAI-COMMONX-005 | MEDIUM | `KaiError` does not runtime-validate that `code` is an `ErrorCode` |
| KAI-COMMONX-006 | MEDIUM | Error context is retained by mutable reference rather than copied or frozen |
| KAI-COMMONX-007 | MEDIUM | `FEATURE_DISABLED` is mapped to HTTP 501 rather than a deployment/readiness policy status |
| KAI-COMMONX-008 | MEDIUM | Safety and verifier blocks are represented as generic 422 client-validation errors |
| KAI-COMMONX-009 | HIGH | Every registered non-GPT model is token-counted with the GPT `cl100k_base` tokenizer |
| KAI-COMMONX-010 | HIGH | Each model’s declared `tiktoken_encoding` field is never used |
| KAI-COMMONX-011 | HIGH | The fallback character heuristic is not safely conservative for code, Unicode and CJK text |
| KAI-COMMONX-012 | HIGH | Message overhead is hard-coded to OpenAI-style 4+2 tokens for every model/backend |
| KAI-COMMONX-013 | HIGH | Prefix matching assigns trusted capabilities to arbitrary model names beginning with a known identifier |
| KAI-COMMONX-014 | HIGH | Static context, JSON, vision, quality and timeout claims are not verified against the loaded artefact |
| KAI-COMMONX-015 | HIGH | Unknown models receive a generic 4K specification that can still exceed the real model limit |
| KAI-COMMONX-016 | MEDIUM | `OLLAMA_MODEL` is accepted as an unvalidated arbitrary identifier |
| KAI-COMMONX-017 | MEDIUM | Specialist model environment values can reference unknown/unapproved artefacts |
| KAI-COMMONX-018 | MEDIUM | Specialist detection uses raw substring matching inside unrelated words |
| KAI-COMMONX-019 | MEDIUM | Keyword stuffing can force specialist routing |
| KAI-COMMONX-020 | MEDIUM | Overlapping broad specialist vocabularies produce unstable domain selection |
| KAI-COMMONX-021 | MEDIUM | Registry lookups do not prove backend/model readiness or exact digest |
| KAI-COMMONX-022 | MEDIUM | Registry entries have no version, effective date, source or verification digest |
| KAI-COMMONX-023 | MEDIUM | Specialist configuration exposes internal routing keywords and model selections to callers that publish it |
| KAI-COMMONX-024 | HIGH | Tier-1 models ignore WORK mode and receive only the minimal casual assistant prompt |
| KAI-COMMONX-025 | HIGH | Untrusted evidence is promoted directly into the system prompt |
| KAI-COMMONX-026 | HIGH | Untrusted personality notes are promoted directly into the system prompt |
| KAI-COMMONX-027 | HIGH | Arbitrary `extra_context` is appended as privileged system instructions |
| KAI-COMMONX-028 | HIGH | Conversation history accepts externally supplied `system` messages |
| KAI-COMMONX-029 | HIGH | System prompts, history, evidence, personality and user input have no aggregate bounds |
| KAI-COMMONX-030 | HIGH | Prompt construction performs no model-specific token-budget enforcement |
| KAI-COMMONX-031 | HIGH | Every unknown mode silently receives PUB behaviour |
| KAI-COMMONX-032 | HIGH | Fact-check claims and evidence are concatenated without robust data delimiters or provenance |
| KAI-COMMONX-033 | HIGH | Planning goals and constraints can inject instructions into the plan prompt |
| KAI-COMMONX-034 | HIGH | Reflection topics and context can inject instructions into the reflection prompt |
| KAI-COMMONX-035 | HIGH | Wake-intent user text is appended directly after JSON-format instructions |
| KAI-COMMONX-036 | MEDIUM | Tier-3 JSON support adds a global JSON-only hint even when structured output was not requested |
| KAI-COMMONX-037 | MEDIUM | Specialist prompt richness is based on the global active model rather than the actual selected backend |
| KAI-COMMONX-038 | MEDIUM | Any imported code can overwrite custom templates at runtime |
| KAI-COMMONX-039 | MEDIUM | Custom-template state is process-local, unlocked and unversioned |
| KAI-COMMONX-040 | MEDIUM | `PROMPT_TEMPLATE_DIR` is declared but never loaded or used |
| KAI-COMMONX-041 | MEDIUM | Prompt-only JSON instructions are not backed by a schema-enforced output contract |
| KAI-COMMONX-042 | HIGH | The shared rate limiter is used only by Tool Gate despite documenting several protected endpoints |
| KAI-COMMONX-043 | HIGH | Per-process counters let traffic bypass limits across workers and replicas |
| KAI-COMMONX-044 | HIGH | Limits are global per endpoint name rather than per caller, token, IP or principal |
| KAI-COMMONX-045 | HIGH | The “burst” multiplier raises the sustained one-minute limit rather than controlling a short burst window |
| KAI-COMMONX-046 | HIGH | Counter lists are mutated without a lock or atomic operation |
| KAI-COMMONX-047 | HIGH | Missing, zero or negative policy limits silently disable protection |
| KAI-COMMONX-048 | MEDIUM | Rate windows use wall-clock time and are vulnerable to clock changes |
| KAI-COMMONX-049 | MEDIUM | Arbitrary endpoint names can create unbounded counter keys |
| KAI-COMMONX-050 | MEDIUM | Empty endpoint keys are never removed from the global dictionary |
| KAI-COMMONX-051 | MEDIUM | 429 responses omit `Retry-After` and reset-window metadata |
| KAI-COMMONX-052 | MEDIUM | Rate-limit errors disclose internal endpoint names, counts and policy values |
| KAI-COMMONX-053 | MEDIUM | Timestamp lists and slicing create linear pruning/memory overhead under load |
| KAI-COMMONX-054 | MEDIUM | Burst-multiplier values are not validated for finiteness or safe range |
| KAI-COMMONX-055 | MEDIUM | Metrics snapshots can race concurrent counter mutation |
| KAI-COMMONX-056 | HIGH | A/B query telemetry is enabled by default for every shared LLM request |
| KAI-COMMONX-057 | HIGH | Session identifiers are written to plaintext JSONL telemetry |
| KAI-COMMONX-058 | HIGH | Prompt identity is only a 32-bit displayed hash of the first 200 characters |
| KAI-COMMONX-059 | HIGH | Distinct prompts sharing the first 200 characters receive the same logged identity |
| KAI-COMMONX-060 | HIGH | Honest uncertainty language is treated as a response-quality penalty |
| KAI-COMMONX-061 | HIGH | Lexical diversity is treated as quality despite no correctness or outcome evidence |
| KAI-COMMONX-062 | HIGH | Full response analysis and synchronous file I/O run on the LLM request path |
| KAI-COMMONX-063 | HIGH | The lock protects threads only and does not prevent multi-process JSONL interleaving |
| KAI-COMMONX-064 | HIGH | The telemetry file has no rotation, retention or maximum-size policy |
| KAI-COMMONX-065 | MEDIUM | The log path is a relative or arbitrary environment-controlled filesystem path |
| KAI-COMMONX-066 | MEDIUM | Telemetry write failures are swallowed at debug level |
| KAI-COMMONX-067 | MEDIUM | Quality analysis scans the complete unbounded response text |
| KAI-COMMONX-068 | MEDIUM | The unsalted short prompt hash is vulnerable to dictionary guessing for known short prompts |
| KAI-COMMONX-069 | MEDIUM | Logs omit a response digest, final outcome and independent quality review |
| KAI-COMMONX-070 | MEDIUM | Telemetry has no authenticated user/tenant partition or request correlation ID |
| KAI-COMMONX-071 | MEDIUM | File ownership, permissions, fsync and atomic durability are not enforced |
| KAI-COMMONX-072 | MEDIUM | Model, source, latency and usage fields are not validated before JSON serialisation |

---

## Structured errors — `common/errors.py`

### KAI-COMMONX-001 — HIGH — Declared standard is not adopted
The module says every client/log error should use `ErrorCode`, but repository search finds `KaiError` only in this module and tests. Active services continue to return ad-hoc exceptions and raw strings, so the promised uniform control does not exist.

### KAI-COMMONX-002 — HIGH — Detail is a direct disclosure channel
`detail` is arbitrary text and `to_dict()` returns it verbatim with no redaction, truncation or audience classification.

### KAI-COMMONX-003 — MEDIUM — Context is silently lost
`context` is accepted and stored, but omitted from `to_dict()` and the exception string, defeating structured diagnostic use.

### KAI-COMMONX-004 — MEDIUM — No traceable error event
Responses lack a stable event ID, timestamp, retryability, dependency/source code and protected trace reference.

### KAI-COMMONX-005 — MEDIUM — Code type is unenforced
Passing a string or foreign enum fails while accessing `.name`/`.value` instead of producing a controlled internal error.

### KAI-COMMONX-006 — MEDIUM — Mutable context alias
The caller’s dictionary is retained directly and may be altered after exception creation.

### KAI-COMMONX-007 — MEDIUM — Misleading disabled-feature status
HTTP 501 means the server does not implement the method; a configured feature-off state is ordinarily a policy/readiness condition.

### KAI-COMMONX-008 — MEDIUM — Safety blocks look like input-schema failures
Conviction, verifier and adversary decisions all map to 422, preventing clients/metrics from distinguishing validation from safety/governance refusal.

---

## Model registry and token accounting — `common/model_registry.py`

### KAI-COMMONX-009 — HIGH — Wrong tokenizer for all models
One global `cl100k_base` encoder is used for Qwen, Llama, DeepSeek, Kimi, Mistral, Gemma, Phi and Yi models.

### KAI-COMMONX-010 — HIGH — Per-model encoding is dead metadata
`ModelSpec.tiktoken_encoding` is never read.

### KAI-COMMONX-011 — HIGH — Unsafe fallback claim
Character count divided by 3.5 can materially undercount code, symbols and many languages, despite being described as conservative.

### KAI-COMMONX-012 — HIGH — Incorrect chat framing overhead
The fixed four tokens per message and two reply tokens are OpenAI assumptions, not validated Ollama/model-template costs.

### KAI-COMMONX-013 — HIGH — Prefix capability confusion
Names such as `qwen2.5:7b-untrusted-custom` inherit the known 7B capability card solely because the string starts with the key.

### KAI-COMMONX-014 — HIGH — Static claims are not runtime facts
Context window, JSON/vision support and timeout are source constants, not verified against the exact downloaded model/Modelfile/backend.

### KAI-COMMONX-015 — HIGH — Unknown-model overflow risk
A real model with a smaller context than 4,096 tokens still receives the generic 4K budget.

### KAI-COMMONX-016 — MEDIUM — Unvalidated active identity
`active_model()` returns arbitrary environment text without canonicalisation or approved-registry enforcement.

### KAI-COMMONX-017 — MEDIUM — Unapproved specialist artefacts
Specialist environment variables bypass the registry and may name any model string.

### KAI-COMMONX-018 — MEDIUM — Substring routing
A keyword counts anywhere inside the lowercased input, including unrelated larger words.

### KAI-COMMONX-019 — MEDIUM — Keyword-forceable routing
Two injected domain words are sufficient to select a specialist.

### KAI-COMMONX-020 — MEDIUM — Overlapping broad vocabulary
Terms such as plan, rate, number, explain and build make domain assignment sensitive to wording rather than required capability.

### KAI-COMMONX-021 — MEDIUM — Registry does not establish availability
Looking up a spec proves neither endpoint reachability nor that the artefact is loaded.

### KAI-COMMONX-022 — MEDIUM — No metadata provenance
Capability records have no source, revision, verification date or digest.

### KAI-COMMONX-023 — MEDIUM — Routing internals are exportable
`specialist_config()` returns keyword lists, model IDs and environment-variable names; several health/introspection paths publish similar configuration.

---

## Prompt construction — `common/prompt_templates.py`

### KAI-COMMONX-024 — HIGH — Default model loses WORK safeguards
For quality tier 1, `build_system_prompt()` always returns `_SYSTEM_BASE_MINIMAL` before checking mode. The default `qwen2.5:0.5b` therefore receives none of WORK’s professional, HMRC, uncertainty or “never guess numbers/deadlines” instructions.

### KAI-COMMONX-025 — HIGH — Evidence gains system authority
Arbitrary evidence text is appended to the system message without provenance or instruction/data separation.

### KAI-COMMONX-026 — HIGH — Personality note gains system authority
Any personality note is appended as system instruction text.

### KAI-COMMONX-027 — HIGH — Extra context gains system authority
`extra_context` is placed directly under `Context:` in the system prompt.

### KAI-COMMONX-028 — HIGH — History can add system messages
Any history item whose role equals `system` is accepted and appended after the primary system message.

### KAI-COMMONX-029 — HIGH — Unbounded prompt assembly
No item, byte, message-count or aggregate-depth limits protect prompt construction.

### KAI-COMMONX-030 — HIGH — No context-window enforcement
The helper accepts a model argument but never calls `context_budget()` or trims messages.

### KAI-COMMONX-031 — HIGH — Unknown mode becomes PUB
Only exact WORK receives the professional prompt; every other value receives PUB.

### KAI-COMMONX-032 — HIGH — Fact-check prompt injection
Claim/evidence are concatenated as ordinary instruction text with no quoting, source identity or deterministic verification.

### KAI-COMMONX-033 — HIGH — Planning prompt injection
Goal and constraints can redefine the requested output or safety rules.

### KAI-COMMONX-034 — HIGH — Reflection prompt injection
Topic/context can insert new instructions into a self-reflection operation.

### KAI-COMMONX-035 — HIGH — Wake JSON instruction injection
User text follows immediately after the strict JSON directions and can request a competing output/label.

### KAI-COMMONX-036 — MEDIUM — Global JSON hint
Tier-3 JSON-capable models are told to output JSON whenever “asked for structured data”, regardless of the calling task’s actual output contract.

### KAI-COMMONX-037 — MEDIUM — Wrong model drives specialist prompts
Fact-check/planning/reflection call `get_model_spec()` without the selected specialist model.

### KAI-COMMONX-038 — MEDIUM — Runtime template overwrite
`register_template()` replaces a name without identity, approval, duplicate rejection or audit.

### KAI-COMMONX-039 — MEDIUM — Worker-local template state
The dictionary is unlocked, process-local and restart-volatile.

### KAI-COMMONX-040 — MEDIUM — Misleading template-directory configuration
`PROMPT_TEMPLATE_DIR` is read but no directory or file is loaded.

### KAI-COMMONX-041 — MEDIUM — Instructions are not validation
JSON-only prompt text does not constrain model output or guarantee schema compliance.

---

## Rate limiting — `common/rate_limit.py`

### KAI-COMMONX-042 — HIGH — Protection adoption gap
Repository search finds active `check_rate_limit()` use only in Tool Gate, despite documented limits for memory and execution.

### KAI-COMMONX-043 — HIGH — Replica bypass
Each worker maintains independent timestamp lists.

### KAI-COMMONX-044 — HIGH — Cross-user starvation
All callers share one endpoint counter; one caller can consume the allowance for everyone.

### KAI-COMMONX-045 — HIGH — Burst is sustained capacity
The code permits `limit * BURST_MULTIPLIER` requests across the entire 60-second window, effectively changing the minute limit.

### KAI-COMMONX-046 — HIGH — Counter races
List pruning, length checking and append are separate unlocked operations.

### KAI-COMMONX-047 — HIGH — Fail-open limit semantics
Any policy result at or below zero disables the limiter.

### KAI-COMMONX-048 — MEDIUM — Wall-clock window
Clock movement changes request ages and pruning.

### KAI-COMMONX-049 — MEDIUM — Unbounded endpoint cardinality
The public helper accepts arbitrary endpoint strings that create dictionary/list entries.

### KAI-COMMONX-050 — MEDIUM — Key retention
Pruning replaces lists but never removes empty endpoint keys.

### KAI-COMMONX-051 — MEDIUM — Incomplete 429 contract
No Retry-After or reset timestamp is included.

### KAI-COMMONX-052 — MEDIUM — Internal policy disclosure
The response reveals endpoint identifiers, current count, configured limit and burst cap.

### KAI-COMMONX-053 — MEDIUM — Linear sliding-window structure
Every request scans old entries and may copy the remaining list.

### KAI-COMMONX-054 — MEDIUM — Invalid multiplier behaviour
Negative, zero, NaN or extreme multipliers are not validated before multiplication/integer conversion.

### KAI-COMMONX-055 — MEDIUM — Snapshot race
Metrics iterate keys and prune lists while requests may mutate them.

---

## A/B telemetry — `common/ab_log.py`

### KAI-COMMONX-056 — HIGH — Default-on query telemetry
`AB_LOG_ENABLED` defaults true and `common.llm` invokes the logger for every query.

### KAI-COMMONX-057 — HIGH — Plaintext session tracking
Session IDs are stored directly.

### KAI-COMMONX-058 — HIGH — Weak prompt identity
Only eight hexadecimal characters of SHA-256 are retained.

### KAI-COMMONX-059 — HIGH — Prefix collision by design
Only `prompt[:200]` is hashed.

### KAI-COMMONX-060 — HIGH — Uncertainty is penalised
Words expressing appropriate uncertainty reduce the quality signal.

### KAI-COMMONX-061 — HIGH — Style replaces correctness
Unique-word ratio increases quality even when the answer is wrong, fabricated or unsafe.

### KAI-COMMONX-062 — HIGH — Hot-path synchronous work
Regex analysis, directory creation, file open and append occur before the LLM call returns to its caller.

### KAI-COMMONX-063 — HIGH — Multi-process log corruption
`threading.Lock` does not coordinate workers/processes writing the same file.

### KAI-COMMONX-064 — HIGH — Unlimited telemetry growth
No rotation, maximum bytes, age retention or archival exists.

### KAI-COMMONX-065 — MEDIUM — Ambiguous storage location
The default relative path depends on service working directory; environment can redirect to arbitrary writable paths.

### KAI-COMMONX-066 — MEDIUM — Silent loss
Every exception is swallowed at debug level and no health/readiness metric changes.

### KAI-COMMONX-067 — MEDIUM — Unbounded quality scan
The complete response is tokenised into words/sets.

### KAI-COMMONX-068 — MEDIUM — Guessable prompt fingerprints
An unsalted 32-bit digest of a short/common prompt can be precomputed.

### KAI-COMMONX-069 — MEDIUM — Missing result evidence
The log has no response digest, verifier outcome, user feedback or ground-truth result.

### KAI-COMMONX-070 — MEDIUM — Missing principal/correlation boundary
There is no authenticated user/tenant or request/trace ID.

### KAI-COMMONX-071 — MEDIUM — Weak file durability/security
No explicit mode, owner verification, fsync, append transaction or integrity chain exists.

### KAI-COMMONX-072 — MEDIUM — Unvalidated telemetry fields
Model/source/latency/usage values are accepted from response objects and serialised without finite/type/range checks.

---

## Batch totals

- Findings: **72**
- Critical: **0**
- High: **36**
- Medium: **36**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,077**
- Critical: **181**
- High: **1,022**
- Medium: **871**
- Low: **3**

## Files materially reviewed

`common/errors.py`, `common/model_registry.py`, `common/prompt_templates.py`, `common/rate_limit.py`, `common/ab_log.py`, plus usage confirmation against `common/llm.py`, Tool Gate and repository call-site searches.
