# Kai Code Audit — TTS Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_TTS_SERVICE.md`. The existing 10 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TTSX-001 | HIGH | Generic sanitisation silently truncates text before the advertised 5,000-character validation |
| KAI-TTSX-002 | HIGH | Generic sanitisation strips punctuation/operators and changes the spoken meaning |
| KAI-TTSX-003 | HIGH | Client disconnect cannot stop synthesis because all upstream audio is generated before response streaming begins |
| KAI-TTSX-004 | HIGH | Edge TTS streaming has no service-owned total synthesis timeout |
| KAI-TTSX-005 | HIGH | Generated audio has no maximum byte, duration or chunk-count limit |
| KAI-TTSX-006 | HIGH | Returned chunks are accepted as audio solely from a caller-controlled type field and are not validated as MP3 |
| KAI-TTSX-007 | HIGH | The service has no idempotency or synthesis cache, so identical requests repeatedly leave the trust boundary |
| KAI-TTSX-008 | HIGH | Synthesis requests have no source identity, purpose, consent revision or data-classification metadata |
| KAI-TTSX-009 | HIGH | The configured `TTS_VOICE` does not control the normal request default |
| KAI-TTSX-010 | HIGH | Configured default rate and volume values are never used |
| KAI-TTSX-011 | HIGH | Health reports a default voice that may differ from the voice actually selected for default requests |
| KAI-TTSX-012 | HIGH | No durable audit links caller, text digest, external provider, voice, audio digest and delivery consumer |
| KAI-TTSX-013 | MEDIUM | Voice identifiers are reflected into the `X-Voice` response header without header-safe validation |
| KAI-TTSX-014 | MEDIUM | Arbitrary voice values are written to logs without control-character sanitisation |
| KAI-TTSX-015 | MEDIUM | Public `/voices` exposes backend/provider voice identifiers and configuration |
| KAI-TTSX-016 | MEDIUM | Public health exposes usage volume, backend, voice and device information |
| KAI-TTSX-017 | MEDIUM | Unknown non-audio stream chunk types are silently ignored |
| KAI-TTSX-018 | MEDIUM | Audio responses have no content digest, generation ID or provider revision |
| KAI-TTSX-019 | MEDIUM | A static attachment filename prevents reliable correlation and can cause client-side overwrites |
| KAI-TTSX-020 | MEDIUM | Usage counters measure sanitised/truncated characters rather than requested characters or provider billing units |
| KAI-TTSX-021 | MEDIUM | Backend and voice configuration is loaded once with no rotation/reload contract |
| KAI-TTSX-022 | MEDIUM | Edge-provider version, endpoint and data-processing region are not surfaced in readiness or responses |
| KAI-TTSX-023 | MEDIUM | The service has no lifespan-owned provider client, cancellation drain or graceful synthesis shutdown |
| KAI-TTSX-024 | MEDIUM | Error logging and public details have no protected trace ID or credential/data redaction contract |

---

## High-severity findings

### KAI-TTSX-001 — HIGH — Advertised input limit is bypassed by silent truncation
**Issue:** `sanitize_string(request.text)` truncates to the shared sanitizer’s 1,024-character limit before `if len(text) > 5000`. The 5,000-character rejection is therefore effectively unreachable.  
**Risk:** Callers believe complete text was spoken, while most of a long response is silently discarded; consequential instructions and qualifications can disappear.  
**Recommendation:** validate the original byte/character/token length first and use a TTS-specific normalisation that never silently truncates.  
**Status:** OPEN

### KAI-TTSX-002 — HIGH — Sanitisation changes speech meaning
The generic sanitizer strips characters such as semicolons, pipes and ampersands. Those characters can encode pauses, lists, company names, formulae and instructions; synthesis does not report the alteration.

### KAI-TTSX-003 — HIGH — Client cancellation is ineffective
`_synthesize_edge()` completes and buffers all audio before `StreamingResponse` is constructed. A caller disconnect does not provide downstream backpressure or stop the provider request already in progress.

### KAI-TTSX-004 — HIGH — No total synthesis deadline
The service relies on edge-tts internals and wraps no `asyncio.timeout` around the full async stream.

### KAI-TTSX-005 — HIGH — Unbounded generated media
Text length does not safely bound encoded audio size, duration, provider chunk count or memory use.

### KAI-TTSX-006 — HIGH — Audio bytes are not validated
Every chunk with `chunk["type"] == "audio"` is concatenated, and the final bytes are labelled `audio/mpeg` without magic/decoder validation.

### KAI-TTSX-007 — HIGH — Repeated external disclosure
There is no content-addressed cache/idempotency key; repeated identical requests resend the text to Microsoft and regenerate complete audio.

### KAI-TTSX-008 — HIGH — Missing processing authority
Requests contain only text/voice/rate/volume. There is no authenticated caller, user consent, purpose, classification or permitted-provider decision.

### KAI-TTSX-009 — HIGH — `TTS_VOICE` is not the request default
`SynthesisRequest.voice` defaults to `kai-default`, which maps to hard-coded Ryan. `DEFAULT_VOICE` is used only when the resolved voice is empty.

### KAI-TTSX-010 — HIGH — Environment defaults are dead configuration
`DEFAULT_RATE` and `DEFAULT_VOLUME` are defined but request defaults are independently hard-coded to `+0%`.

### KAI-TTSX-011 — HIGH — Health misreports effective default
Health returns `DEFAULT_VOICE`, although a normal request selects `VOICE_MAP["kai-default"]` even when the environment specifies another voice.

### KAI-TTSX-012 — HIGH — Missing synthesis audit chain
No tamper-evident record binds initiating principal, original/sanitised text digests, selected provider/voice, audio digest, counters and downstream recipient.

---

## Medium-severity findings

### KAI-TTSX-013 — MEDIUM — Unvalidated response-header value
A caller-supplied unknown voice is reflected into `X-Voice`; newline/non-Latin/control values can cause response construction/protocol failures.

### KAI-TTSX-014 — MEDIUM — Voice log injection
The selected voice string is logged verbatim after successful synthesis.

### KAI-TTSX-015 — MEDIUM — Public provider inventory
`/voices` returns all preset-to-provider mappings and backend name without authentication.

### KAI-TTSX-016 — MEDIUM — Public usage/topology health
Health exposes provider choice, default voice, counts, characters and device.

### KAI-TTSX-017 — MEDIUM — Silent protocol drift
Metadata, boundary, error or unknown chunk types are ignored rather than validated against an expected provider stream schema.

### KAI-TTSX-018 — MEDIUM — No media integrity metadata
The response includes character/voice headers but no SHA-256, generation ID, provider/model revision or completion state.

### KAI-TTSX-019 — MEDIUM — Static filename
Every response is `kai_speech.mp3`, making browser/download workflows overwrite or confuse distinct outputs.

### KAI-TTSX-020 — MEDIUM — Misleading counters
Character totals represent post-sanitisation text and are neither original demand nor provider billable units.

### KAI-TTSX-021 — MEDIUM — Static configuration lifecycle
Provider/backend/voice settings cannot be atomically reloaded or versioned, and rolling workers may differ.

### KAI-TTSX-022 — MEDIUM — Provider processing is opaque
Responses/readiness expose no actual Edge endpoint, library/provider revision, region or privacy-processing state.

### KAI-TTSX-023 — MEDIUM — Missing provider lifecycle ownership
No lifespan manages a reusable provider session, in-flight cancellation, shutdown timeout or usage flush.

### KAI-TTSX-024 — MEDIUM — Weak diagnostics contract
Raw exceptions are both logged and returned without a stable public error code, redacted protected trace or assurance that text/provider URLs are absent.

---

## Batch totals

- Findings: **24**
- Critical: **0**
- High: **12**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,207**
- Critical: **189**
- High: **1,098**
- Medium: **917**
- Low: **3**

## Files materially reviewed

`output/tts/app.py`, the existing TTS audit and integrations with Dashboard, Telegram, Monitor and Supervisor.
