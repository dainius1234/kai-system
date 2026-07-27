# Kai Code Audit — Wake Intent Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WAKE-001 | CRITICAL | The host-published wake/intent service has no authentication or authorisation |
| KAI-WAKE-002 | CRITICAL | An unauthenticated high-confidence `command` classification can override Agentic routing to `EXECUTE_ACTION` when wake-intent routing is enabled |
| KAI-WAKE-003 | HIGH | Incoming requests have no verified user, device, microphone, session or service identity |
| KAI-WAKE-004 | HIGH | Text and base64-audio request bodies have no byte-size limits |
| KAI-WAKE-005 | HIGH | Base64 audio is fully materialised and decoded, multiplying peak memory consumption |
| KAI-WAKE-006 | HIGH | Whisper transcription runs synchronously inside async FastAPI request handlers |
| KAI-WAKE-007 | HIGH | The Whisper model is lazily constructed during the first user request |
| KAI-WAKE-008 | HIGH | First-use model resolution may perform heavyweight artefact loading or download work in the request path |
| KAI-WAKE-009 | HIGH | Transcription has no deadline, cancellation or maximum processing-time contract |
| KAI-WAKE-010 | HIGH | No rate limit, transcription semaphore, queue bound or caller quota protects CPU and memory |
| KAI-WAKE-011 | HIGH | Arbitrary decoded bytes are written as a `.wav` file without validating container, codec or magic bytes |
| KAI-WAKE-012 | HIGH | Audio duration, sample rate, channel count, decompressed size and segment count are unbounded |
| KAI-WAKE-013 | HIGH | The complete transcription is joined in memory before sanitisation/truncation |
| KAI-WAKE-014 | HIGH | Audio-only `/wake/process` detects the wake word but always classifies an empty text string as `unknown` |
| KAI-WAKE-015 | HIGH | One process-global cooldown allows one caller/device to suppress every other caller |
| KAI-WAKE-016 | HIGH | Cooldown state is worker-local, so load-balanced requests can bypass debounce or receive inconsistent outcomes |
| KAI-WAKE-017 | HIGH | Cooldown is consumed before the configured confidence threshold is evaluated |
| KAI-WAKE-018 | HIGH | A threshold at or below 0.49 turns cooldown-suppressed events back into detected wake events |
| KAI-WAKE-019 | HIGH | Cooldown and confidence configuration accept negative, non-finite and unsafe values |
| KAI-WAKE-020 | HIGH | Cooldown uses wall-clock time and is vulnerable to clock adjustment |
| KAI-WAKE-021 | HIGH | Wake confidence is a hard-coded constant, not acoustic or classification confidence |
| KAI-WAKE-022 | HIGH | Multi-word wake phrases automatically receive higher confidence regardless of evidence quality |
| KAI-WAKE-023 | HIGH | Caller text is inserted into the intent-classification prompt without an instruction/data boundary |
| KAI-WAKE-024 | HIGH | Keyword heuristic classification ignores negation, quotation, attribution and context |
| KAI-WAKE-025 | HIGH | Command keywords are evaluated before question/task/emotional intent and can force the command route |
| KAI-WAKE-026 | HIGH | LLM failure or invalid output silently falls back to heuristics without a degraded/source marker |
| KAI-WAKE-027 | HIGH | `WAKE_INTENT_MODEL` is cosmetic; `query_specialist("Ollama")` actually uses the shared `OLLAMA_MODEL` mapping |
| KAI-WAKE-028 | HIGH | Health probes Ollama reachability but does not verify the model actually used by classification is loaded |
| KAI-WAKE-029 | HIGH | Ollama HTTP 4xx responses are classified as available because only 5xx degrades health |
| KAI-WAKE-030 | HIGH | Health reports `ok` when Whisper is unavailable even though audio requests silently cannot work |
| KAI-WAKE-031 | HIGH | Missing transcription capability returns an ordinary non-detection instead of an unavailable/error result |
| KAI-WAKE-032 | HIGH | The routing feature flag is enforced only by Agentic callers; direct wake endpoints remain fully active |
| KAI-WAKE-033 | HIGH | `/wake/intent` is an unauthenticated open LLM/heuristic workload endpoint |
| KAI-WAKE-034 | HIGH | NaN intent confidence passes validation and can enter non-standard JSON/downstream comparisons |
| KAI-WAKE-035 | MEDIUM | Boolean confidence values are accepted because Python booleans are integers |
| KAI-WAKE-036 | MEDIUM | Wake-word count and individual phrase lengths are not bounded or validated |
| KAI-WAKE-037 | MEDIUM | Wake regular expressions are recompiled for every request instead of once at startup |
| KAI-WAKE-038 | MEDIUM | Unicode canonicalisation is absent, producing inconsistent matching for visually equivalent text |
| KAI-WAKE-039 | MEDIUM | Data-URI MIME metadata is ignored and any encoded bytes are accepted |
| KAI-WAKE-040 | MEDIUM | Invalid-base64 parser diagnostics are returned directly to callers |
| KAI-WAKE-041 | MEDIUM | When both text and audio are supplied, audio is silently ignored |
| KAI-WAKE-042 | MEDIUM | Intent and wake responses omit inference source, model identity, transcript confidence and degradation state |
| KAI-WAKE-043 | MEDIUM | Health publicly discloses configured wake words, thresholds, model name and transcription availability |
| KAI-WAKE-044 | MEDIUM | Every health check creates a new HTTP client and connection pool |
| KAI-WAKE-045 | MEDIUM | Compose declares a healthy memU dependency although the wake service never uses memU |
| KAI-WAKE-046 | MEDIUM | Several Python dependencies use broad lower-bound version ranges rather than reproducible pins |
| KAI-WAKE-047 | MEDIUM | The service has no structured request/decision audit trail, correlation ID or security event record |
| KAI-WAKE-048 | MEDIUM | No lifespan-owned warm-up, readiness transition, inference lock or graceful model shutdown exists |

---

## Critical findings

### KAI-WAKE-001 — CRITICAL — Open wake and audio-ingestion authority
**Issue:** `docker-compose.full.yml` publishes `8022:8022`. `perception/wake/app.py` defines no authentication, authorisation, service identity or principal/session ownership checks.  
**Risk:** Any reachable caller can submit large audio, consume transcription/LLM resources, manipulate global cooldown state and obtain route-influencing intent classifications.  
**Recommendation:** Remove direct host publication and require authenticated device/service identity, principal-bound sessions and endpoint-specific authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-WAKE-002 — CRITICAL — Untrusted classification changes Agentic action routing
**Issue:** The wake service accepts arbitrary text and returns caller-influenceable intent/confidence. The Agentic integration tests confirm that, when `FF_WAKE_INTENT_ROUTING` is enabled, `intent="command"` with confidence 0.95 overrides ordinary routing to `EXECUTE_ACTION`. No signed source or wake-device identity is carried with the result.  
**Risk:** Prompt/keyword manipulation at an unauthenticated classifier becomes an execution-route decision signal.  
**Recommendation:** Treat wake output only as untrusted input metadata; independently authenticate the speaker/device and reclassify/authorise the exact requested action at the execution boundary.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-WAKE-003 — HIGH — Missing caller and sensor identity
Requests do not identify an authenticated operator, microphone/device, session or calling service. The same endpoint handles browser, service and arbitrary network callers identically.

### KAI-WAKE-004 — HIGH — Unbounded request bodies
`text` and `audio_b64` are unconstrained strings. FastAPI/Pydantic must first receive and parse the complete JSON body.

### KAI-WAKE-005 — HIGH — Base64 memory amplification
The encoded string and decoded bytes coexist, and later the temporary file/transcription introduce further copies.

### KAI-WAKE-006 — HIGH — Event-loop blocking transcription
`_transcribe_audio()` performs model inference synchronously inside `wake_detect()`.

### KAI-WAKE-007 — HIGH — First-request model construction
`_get_whisper_model()` creates `WhisperModel` lazily during the first audio request rather than startup readiness.

### KAI-WAKE-008 — HIGH — Request-time artefact loading
The configured Faster Whisper model is not copied into the image or explicitly preloaded. Model resolution/cache population can therefore become first-request work.

### KAI-WAKE-009 — HIGH — No transcription deadline
No timeout or cooperative cancellation surrounds model loading/transcription; client disconnect does not stop the synchronous work.

### KAI-WAKE-010 — HIGH — Missing workload admission
Unlimited concurrent callers can submit audio or LLM classification work without a semaphore, queue, rate limit or per-device quota.

### KAI-WAKE-011 — HIGH — Unvalidated audio container
Any decoded bytes are written to a temporary file with suffix `.wav`; MIME declaration, file signature and decoder format are not checked.

### KAI-WAKE-012 — HIGH — Unbounded decoded workload
Duration, sample rate, channels, audio-frame count and decoder expansion are unrestricted.

### KAI-WAKE-013 — HIGH — Post-processing limit only
Every segment’s text is concatenated before `sanitize_string()` applies its later truncation.

### KAI-WAKE-014 — HIGH — Audio-only combined processing is broken
`wake_process()` calls `wake_detect()`, which may transcribe audio internally, but then reconstructs text only from `req.text`. For audio-only input, intent is always `unknown` even after successful wake detection.

### KAI-WAKE-015 — HIGH — Cross-user cooldown denial
`_last_wake_ts` is one global timestamp; a request from any caller suppresses all other callers during the cooldown.

### KAI-WAKE-016 — HIGH — Multi-worker cooldown inconsistency
The timestamp/lock are process-local. Separate workers or replicas maintain unrelated debounce windows.

### KAI-WAKE-017 — HIGH — Threshold-rejected events consume cooldown
`detect_wake_word()` updates `_last_wake_ts` before the endpoint evaluates `WAKE_CONFIDENCE_THRESHOLD`. A configured threshold above the fixed confidence rejects the event but still blocks subsequent callers.

### KAI-WAKE-018 — HIGH — Cooldown can be disabled by threshold interaction
Cooldown returns confidence up to 0.49. If the configured threshold is 0.49 or lower, the endpoint returns `detected=true` despite the explicit cooldown rejection.

### KAI-WAKE-019 — HIGH — Unsafe numeric configuration
Environment floats are parsed without finite/range/cross-field validation. Negative/NaN cooldowns and thresholds below zero or above one create contradictory behaviour.

### KAI-WAKE-020 — HIGH — Non-monotonic debounce clock
The cooldown uses `time.time()` rather than a monotonic clock.

### KAI-WAKE-021 — HIGH — Fabricated wake confidence
A matched single phrase always scores 0.85 and a phrase containing spaces always scores 0.95, regardless of transcription quality or false-positive likelihood.

### KAI-WAKE-022 — HIGH — Phrase length becomes confidence
Multi-word phrases receive stronger confidence purely because they contain a space.

### KAI-WAKE-023 — HIGH — Intent prompt injection
Sanitised caller text remains an instruction-capable user prompt. The classifier has no structured field separation or independent validation of semantic intent.

### KAI-WAKE-024 — HIGH — Linguistically weak fallback
Substring/regex rules ignore statements such as “do not restart”, quoted commands, reports about another person and hypothetical questions.

### KAI-WAKE-025 — HIGH — Command-first classification bias
The command keyword test runs before question, task, emotion and chat tests, so any command word dominates the returned route label.

### KAI-WAKE-026 — HIGH — Silent classifier downgrade
LLM errors, stub/error text and malformed JSON all become ordinary heuristic output; callers cannot distinguish live-model classification from fallback.

### KAI-WAKE-027 — HIGH — Configured intent model is not selected
The service passes specialist `Ollama`; `common.llm` selects the globally configured `OLLAMA_MODEL`. `WAKE_INTENT_MODEL` is only written into the system prompt text.

### KAI-WAKE-028 — HIGH — Wrong readiness subject
Health checks only `/api/tags` reachability and never confirms that the actual global Ollama model is present and usable for chat completions.

### KAI-WAKE-029 — HIGH — 4xx health false positive
`llm_ok = resp.status_code < 500` marks authentication, not-found and malformed-request responses healthy.

### KAI-WAKE-030 — HIGH — Audio capability is optional but health stays green
`status` depends only on Ollama. `_whisper_available=false` does not degrade readiness.

### KAI-WAKE-031 — HIGH — Missing ASR looks like no wake word
When Faster Whisper is absent/model unavailable, transcription returns an empty string and the endpoint responds with a normal false detection.

### KAI-WAKE-032 — HIGH — Capability flag bypass
`FF_WAKE_INTENT_ROUTING` controls only one Agentic integration; direct `/wake/*` endpoints do not check the feature flag.

### KAI-WAKE-033 — HIGH — Open classifier workload
`/wake/intent` invokes the shared LLM router for any anonymous text, then may retry/fallback according to shared LLM settings.

### KAI-WAKE-034 — HIGH — NaN intent confidence accepted
The range checks `confidence < 0` and `confidence > 1` are both false for NaN, so the value survives and is rounded/returned.

---

## Medium-severity findings

### KAI-WAKE-035 — MEDIUM — Boolean confidence accepted
`bool` is a subclass of `int`; JSON `true`/`false` therefore pass numeric validation.

### KAI-WAKE-036 — MEDIUM — Unbounded wake configuration
Wake-word count and phrase lengths are not capped at startup.

### KAI-WAKE-037 — MEDIUM — Repeated regex compilation
Every detection recompiles every configured phrase pattern.

### KAI-WAKE-038 — MEDIUM — Missing Unicode normalisation
Whitespace/lowercasing occurs, but equivalent Unicode forms are not canonicalised.

### KAI-WAKE-039 — MEDIUM — Data-URI metadata ignored
The prefix before the first comma is discarded without verifying an audio media type or encoding declaration.

### KAI-WAKE-040 — MEDIUM — Decoder diagnostics exposed
Base64 exception text is copied into the HTTP error detail.

### KAI-WAKE-041 — MEDIUM — Ambiguous dual-input behaviour
The request validator permits both fields; any non-empty text causes the audio body to be ignored without warning.

### KAI-WAKE-042 — MEDIUM — Missing inference provenance
Responses omit transcript, ASR/model identity, live/stub/heuristic source, fallback reason and acoustic confidence.

### KAI-WAKE-043 — MEDIUM — Configuration disclosure
Public health reveals wake phrases, cooldown, threshold, model identifier and capability flags.

### KAI-WAKE-044 — MEDIUM — Health connection churn
Every health call constructs and closes a new `AsyncClient`.

### KAI-WAKE-045 — MEDIUM — False memU dependency
Compose waits for healthy memU although wake code does not reference the service, increasing startup coupling and misleading architecture documentation.

### KAI-WAKE-046 — MEDIUM — Non-reproducible dependencies
FastAPI, Starlette and Faster Whisper use lower-bound ranges rather than exact reviewed versions.

### KAI-WAKE-047 — MEDIUM — No decision audit
The service records only selected warning logs; it has no actor/device, input digest, detected phrase, classifier source, decision ID or downstream-use record.

### KAI-WAKE-048 — MEDIUM — Missing model lifecycle
There is no lifespan warm-up, readiness state transition, inference serialisation policy or graceful model/task drain.

---

## Batch totals

- Findings: **48**
- Critical: **2**
- High: **32**
- Medium: **14**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,005**
- Critical: **181**
- High: **986**
- Medium: **835**
- Low: **3**

## Files materially reviewed

`perception/wake/app.py`, `perception/wake/Dockerfile`, `perception/wake/requirements.txt`, `docs/wake_intent_j2.md`, `scripts/test_wake_intent.py`, `common/llm.py`, wake deployment in `docker-compose.full.yml`, and Agentic/Dashboard integration references.
