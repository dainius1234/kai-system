# Kai Code Audit — Audio Perception Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_AUDIO_PERCEPTION.md`. The existing 14 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AUDIOX-001 | HIGH | API transcription sends complete recordings to an environment-controlled external HTTP destination |
| KAI-AUDIOX-002 | HIGH | External transcription calls use no service authentication, TLS identity or response attestation |
| KAI-AUDIOX-003 | HIGH | Audio is labelled `audio/wav` for the API backend regardless of the uploaded format |
| KAI-AUDIOX-004 | HIGH | Local Whisper model identifiers can trigger unpinned runtime model downloads on the first request |
| KAI-AUDIOX-005 | HIGH | Concurrent first requests can race and load multiple large Whisper model instances |
| KAI-AUDIOX-006 | HIGH | Local transcription is forced to CPU/int8 even when health advertises CUDA capability |
| KAI-AUDIOX-007 | HIGH | Uploaded bytes are not validated as audio before persistence and decoder/model processing |
| KAI-AUDIOX-008 | HIGH | Audio content has no duration, sample-count, channel, codec or decompression-work limit |
| KAI-AUDIOX-009 | HIGH | Microphone allocation is controlled by unvalidated sample-rate configuration and can exceed expected memory bounds |
| KAI-AUDIOX-010 | HIGH | Prompt-injection checking occurs only after audio has already crossed the external STT boundary |
| KAI-AUDIOX-011 | HIGH | Blocked microphone transcripts are still placed into the publicly readable transcript buffer |
| KAI-AUDIOX-012 | HIGH | Blocked microphone audio is still retained on disk |
| KAI-AUDIOX-013 | HIGH | Injection evidence is stored in a plaintext append-only file with transcript/session snippets and no retention policy |
| KAI-AUDIOX-014 | HIGH | Newline/control characters survive sanitisation and can forge injection-log records |
| KAI-AUDIOX-015 | HIGH | Auto-memorisation ignores its `source` argument and discards capture provenance |
| KAI-AUDIOX-016 | HIGH | Audio memories are attributed to the generic `perception-audio` identity rather than an authenticated speaker/device |
| KAI-AUDIOX-017 | HIGH | Auto-memorised transcripts are silently truncated to the shared 1,024-character sanitizer limit |
| KAI-AUDIOX-018 | HIGH | No PII or secret redaction occurs before external STT, memory persistence or public transcript responses |
| KAI-AUDIOX-019 | HIGH | Wake-intent prompts interpolate caller text into an LLM classifier without a typed untrusted-data boundary |
| KAI-AUDIOX-020 | HIGH | Any classifier output containing the substring `command` is accepted as a direct command intent |
| KAI-AUDIOX-021 | HIGH | Intent-classifier failure falls back to a trivial position heuristic that marks caller text as a command |
| KAI-AUDIOX-022 | HIGH | An empty configured wake word makes the regular expression match essentially every transcript |
| KAI-AUDIOX-023 | HIGH | The configured `PORCUPINE_KEYWORD` hotword is unused while health presents it as active configuration |
| KAI-AUDIOX-024 | HIGH | No rate, concurrency, recording or transcription workload admission policy protects the service |
| KAI-AUDIOX-025 | MEDIUM | Injection-log writes are synchronous, unlocked and can interleave across requests/processes |
| KAI-AUDIOX-026 | MEDIUM | Injection-log and capture files use process-default permissions without ownership validation |
| KAI-AUDIOX-027 | MEDIUM | Upload filenames can create invalid/nested paths and are not canonicalised to a safe basename |
| KAI-AUDIOX-028 | MEDIUM | Upload/source filenames are returned to callers and memory/log contexts without a safe display schema |
| KAI-AUDIOX-029 | MEDIUM | File-extension detection trusts the caller filename and can create malformed temporary suffix paths |
| KAI-AUDIOX-030 | MEDIUM | API transcription unnecessarily writes the complete upload to disk before rereading it for HTTP upload |
| KAI-AUDIOX-031 | MEDIUM | Transcript buffers have item-count limits but no aggregate byte limit |
| KAI-AUDIOX-032 | MEDIUM | Raw session IDs are stored in transcript and wake-history buffers without validation |
| KAI-AUDIOX-033 | MEDIUM | Uploaded-file captures are not added to transcript history and do not receive emotion analysis |
| KAI-AUDIOX-034 | MEDIUM | Capture-file duration is always reported as zero rather than unknown or measured |
| KAI-AUDIOX-035 | MEDIUM | Health returns Boolean capability fields as strings |
| KAI-AUDIOX-036 | MEDIUM | Whisper/API errors are embedded in transcript-like strings and may contain backend diagnostics |
| KAI-AUDIOX-037 | MEDIUM | External/backend configuration and model artefact revisions are not included in transcript provenance |
| KAI-AUDIOX-038 | MEDIUM | Wake detections use wall-clock timestamps without audio/source event identity |
| KAI-AUDIOX-039 | MEDIUM | No immutable audit links audio digest, speaker/device, transcription backend, transcript and memory record |
| KAI-AUDIOX-040 | MEDIUM | The service has no lifespan-owned model/client/executor resources or graceful in-flight transcription drain |

---

## High-severity findings

### KAI-AUDIOX-001 — HIGH — Configurable external audio exfiltration
**Issue:** With `WHISPER_BACKEND=api`, every recording is sent to `WHISPER_API_URL`, an unvalidated environment URL using ordinary HTTP by default.  
**Risk:** Private microphone or uploaded audio can leave the trusted environment for an unintended internal/external destination.  
**Recommendation:** require a signed approved endpoint, authenticated transport, explicit consent and data classification before external transcription.  
**Status:** OPEN

### KAI-AUDIOX-002 — HIGH — No STT service identity
The API request contains no bearer/HMAC/mTLS identity, and the response is not signed or bound to the audio digest.

### KAI-AUDIOX-003 — HIGH — Incorrect media declaration
The API upload always uses `audio/wav`, even when the extension/bytes are OGG, MP3 or arbitrary data.

### KAI-AUDIOX-004 — HIGH — Request-time unpinned model acquisition
`WhisperModel(WHISPER_MODEL)` is constructed lazily; arbitrary model identifiers may resolve/download artefacts during a user request without a pinned digest.

### KAI-AUDIOX-005 — HIGH — Model-load race
`_whisper_model` has no async/thread lock. Concurrent first transcriptions can each initialise a large model.

### KAI-AUDIOX-006 — HIGH — CUDA readiness mismatch
The local model is always created with `device="cpu", compute_type="int8"`, irrespective of `DEVICE` and the advertised GPU design.

### KAI-AUDIOX-007 — HIGH — Arbitrary-file decoder surface
The service verifies only a filename and post-read size. Any bytes are persisted and passed to Whisper/FFmpeg/external STT.

### KAI-AUDIOX-008 — HIGH — No decoded-work budget
A compressed/container file under 50 MB can represent extreme duration, channels, samples or decoder complexity.

### KAI-AUDIOX-009 — HIGH — Configurable microphone allocation
`sd.rec(int(seconds * SAMPLE_RATE))` uses an unbounded startup sample rate; the request can select 60 seconds.

### KAI-AUDIOX-010 — HIGH — Injection control is post-egress
The transcript regex executes only after local/external transcription, so it cannot prevent disclosure of malicious/sensitive audio to the provider.

### KAI-AUDIOX-011 — HIGH — Blocked content remains public
`capture_mic()` appends the transcript and emotion to `_transcript_buffer` regardless of `injection`.

### KAI-AUDIOX-012 — HIGH — Blocked raw audio persists
The WAV is written before transcription/injection checking and is never deleted.

### KAI-AUDIOX-013 — HIGH — Sensitive plaintext injection log
Blocked session/text snippets accumulate in `/tmp/injection_events.log` without encryption, cap, rotation, deletion or access checks.

### KAI-AUDIOX-014 — HIGH — Injection-log record forgery
The shared sanitizer removes only `;|&`; newlines and control characters remain and can create fabricated lines/fields.

### KAI-AUDIOX-015 — HIGH — Provenance argument is discarded
`_auto_memorize(transcript, source)` never writes `source` into the memory request.

### KAI-AUDIOX-016 — HIGH — Speaker identity collapse
Every transcript becomes `user_id="perception-audio"`; no authenticated speaker, microphone, Telegram sender or upload principal is retained.

### KAI-AUDIOX-017 — HIGH — Silent memory truncation
The code slices to 2,000 and then calls a sanitizer capped at 1,024, with no truncated flag.

### KAI-AUDIOX-018 — HIGH — Sensitive data is unredacted
The available shared PII redactor is not used before STT egress, memory writes, transcript history or API output.

### KAI-AUDIOX-019 — HIGH — Caller text controls classifier prompt
Transcript content is formatted directly into the LLM prompt; the sanitizer does not delimit or neutralise classifier instructions.

### KAI-AUDIOX-020 — HIGH — Substring intent acceptance
A response such as “not command” or explanatory text containing `command` is accepted because the loop searches substrings.

### KAI-AUDIOX-021 — HIGH — Unsafe fallback command classification
If the classifier fails, any transcript beginning with the configured wake word or containing `, kai` is marked a command.

### KAI-AUDIOX-022 — HIGH — Empty wake word matches all
An empty environment value builds `\b\b`, which succeeds broadly and sends arbitrary text to intent classification.

### KAI-AUDIOX-023 — HIGH — Hotword configuration is fictional
`HOTWORD` is returned by health but never used in capture/listen/wake-word logic; a separate `WAKE_WORD` authority exists.

### KAI-AUDIOX-024 — HIGH — No workload admission
Microphone capture, file upload, model loading/inference, external STT, emotion and wake classification have no caller quota or global bounded concurrency.

---

## Medium-severity findings

### KAI-AUDIOX-025 — MEDIUM — Unsafe concurrent evidence append
Multiple threads/processes can interleave file writes and there is no fsync/record digest.

### KAI-AUDIOX-026 — MEDIUM — Weak file permissions
Files/directories rely on container umask and no owner/mode check.

### KAI-AUDIOX-027 — MEDIUM — Unsafe upload filename handling
`sanitize_string` does not canonicalise path separators, dot segments, Unicode or reserved/long names before constructing a filesystem path.

### KAI-AUDIOX-028 — MEDIUM — Filename disclosure
The original filename is returned as `source` and passed into memorisation context.

### KAI-AUDIOX-029 — MEDIUM — Extension-derived temporary paths
The suffix is extracted from arbitrary filename text and used in a local temporary path.

### KAI-AUDIOX-030 — MEDIUM — Duplicate disk I/O
API mode writes bytes to a temporary file and then opens that file for upload instead of streaming a bounded in-memory/file object directly.

### KAI-AUDIOX-031 — MEDIUM — Byte-unbounded histories
Fifty transcripts and 100 wake entries can each carry relatively large strings/session IDs/emotion structures.

### KAI-AUDIOX-032 — MEDIUM — Raw session metadata
Manual and wake buffers retain caller session IDs without length/character/owner validation.

### KAI-AUDIOX-033 — MEDIUM — Inconsistent capture semantics
Uploaded files are omitted from transcript history and emotion analysis, unlike microphone captures.

### KAI-AUDIOX-034 — MEDIUM — False duration value
Uploaded recordings report `duration_seconds=0.0`, which can be mistaken for measured zero duration.

### KAI-AUDIOX-035 — MEDIUM — Weak health types
Capability values are stringified, encouraging truthiness/parsing mistakes downstream.

### KAI-AUDIOX-036 — MEDIUM — Diagnostics masquerade as speech
Local/API exception text is returned inside bracketed transcript strings.

### KAI-AUDIOX-037 — MEDIUM — Missing transcription provenance
Results include backend name but no model digest/version, endpoint identity, language confidence for API/stub or audio digest.

### KAI-AUDIOX-038 — MEDIUM — Weak wake chronology
Wake history stores `time.time()` only and no source audio/transcript digest or event ID.

### KAI-AUDIOX-039 — MEDIUM — Missing end-to-end audit
No immutable chain links consent/principal, capture ID, raw audio hash/location, provider request, transcript, injection verdict, emotion and memory ID.

### KAI-AUDIOX-040 — MEDIUM — Missing lifecycle ownership
No lifespan owns a dedicated inference executor, model load, external client, temporary-file cleanup or shutdown completion.

---

## Batch totals

- Findings: **40**
- Critical: **0**
- High: **24**
- Medium: **16**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,247**
- Critical: **189**
- High: **1,122**
- Medium: **933**
- Low: **3**

## Files materially reviewed

`perception/audio/app.py`, the existing Audio Perception audit and integrations with memU, Telegram, Dashboard, local Whisper, external STT and wake-intent classification.
