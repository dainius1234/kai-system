# Kai Code Audit — Audio Perception Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AUDIO-001 | CRITICAL | Unauthenticated callers can remotely activate microphone capture |
| KAI-AUDIO-002 | CRITICAL | Unauthenticated transcript and upload endpoints can write attacker-controlled content into memory |
| KAI-AUDIO-003 | HIGH | Recent transcripts and wake-word history are exposed without authentication |
| KAI-AUDIO-004 | HIGH | Uploaded and microphone audio is retained on disk without lifecycle controls |
| KAI-AUDIO-005 | HIGH | Async endpoints execute blocking recording and transcription work on the event loop |
| KAI-AUDIO-006 | HIGH | Uploaded files are fully read into memory before the size limit is checked |
| KAI-AUDIO-007 | HIGH | Timestamp-only capture and temporary filenames collide under concurrency |
| KAI-AUDIO-008 | MEDIUM | Memory writes are treated as successful without checking HTTP responses |
| KAI-AUDIO-009 | MEDIUM | Prompt-injection detection is a narrow phrase blacklist |
| KAI-AUDIO-010 | MEDIUM | Transcription failures are returned as ordinary transcript strings |
| KAI-AUDIO-011 | MEDIUM | Health reports `ok` when configured transcription capability is unavailable |
| KAI-AUDIO-012 | MEDIUM | Emotion classification is based on unanchored substring heuristics and raw amplitude thresholds |
| KAI-AUDIO-013 | MEDIUM | Captured state is process-local and inconsistent across workers |
| KAI-AUDIO-014 | MEDIUM | Audio configuration and backend selection are not validated |

---

## Audio perception: `perception/audio/app.py`

### KAI-AUDIO-001 — CRITICAL — Remote microphone activation is unauthenticated
**Issue:** `POST /capture/mic` requires no authentication or authorisation and directly calls `sounddevice.rec` for up to 60 seconds.  
**Risk:** Any caller with network reachability can activate the host microphone, capture surrounding conversations, save the recording and receive a transcript. This is a direct privacy and surveillance breach.  
**Recommendation:** Disable remote capture by default and require explicit local user consent, authenticated capability tokens, visible recording state and hardware-level privacy controls.  
**Status:** OPEN — immediate remediation required

### KAI-AUDIO-002 — CRITICAL — Untrusted content can be persisted as memory
**Issue:** `POST /listen`, `POST /capture/file` and wake-word detection are unauthenticated. With `AUTO_MEMORIZE_AUDIO` enabled by default, accepted transcripts are sent to memu-core as `daily-logs` or wake-word memories under the trusted `perception-audio` identity.  
**Risk:** A caller can inject fabricated conversations, instructions, emotional signals and wake activations into durable memory, influencing later agent reasoning and operator modelling.  
**Recommendation:** Authenticate capture sources, bind provenance to a real device/session and require trust classification or approval before persistence.  
**Status:** OPEN — immediate remediation required

### KAI-AUDIO-003 — HIGH — Transcript history is publicly readable
**Issue:** `GET /transcripts` and `GET /wake-word/history` expose recent transcript text, timestamps, session IDs, sources, emotions and classified intents without access control.  
**Risk:** Sensitive spoken content and behavioural history are disclosed to any reachable caller.  
**Recommendation:** Require user-scoped authentication and minimise/expire retained transcript data.  
**Status:** OPEN

### KAI-AUDIO-004 — HIGH — Audio retention has no lifecycle or protection
**Issue:** Microphone and uploaded recordings are written into `AUDIO_DIR`, defaulting to `/tmp/audio-captures`. Microphone files are retained after transcription; uploaded source files are also retained. No encryption, permission hardening, deletion schedule, quota or consent record is implemented.  
**Risk:** Raw conversations accumulate on disk, may survive longer than expected and can be read by other processes or exposed through container/host access.  
**Recommendation:** Avoid retention by default; use protected temporary files with deterministic deletion and explicit retention consent.  
**Status:** OPEN

### KAI-AUDIO-005 — HIGH — Blocking work runs in async handlers
**Issue:** `/capture/mic` directly performs synchronous recording and waits for completion. `_transcribe_audio` performs local model inference or synchronous HTTP/file work directly inside async request handlers.  
**Risk:** One recording or transcription blocks the event-loop worker; concurrent requests can make the service unavailable.  
**Recommendation:** Move capture and inference to a bounded supervised worker queue and return job state.  
**Status:** OPEN

### KAI-AUDIO-006 — HIGH — Upload limit is applied after allocation
**Issue:** `audio_bytes = await file.read()` loads the entire request body before checking the 50 MB maximum.  
**Risk:** Oversized concurrent uploads can exhaust process memory despite the stated limit.  
**Recommendation:** Enforce request/body limits at the server boundary and stream with a hard byte counter.  
**Status:** OPEN

### KAI-AUDIO-007 — HIGH — Filenames collide within the same second
**Issue:** Microphone recordings, uploaded files and transcription temporary files use `int(time.time())`. Concurrent operations in the same second can target the same path; upload filenames add caller-controlled text but can still collide. Temporary transcription cleanup can delete a file being used by another request.  
**Risk:** Recordings are overwritten, mixed between requests or deleted during active processing, compromising integrity and privacy isolation.  
**Recommendation:** Use securely created unique temporary files and per-request immutable capture IDs.  
**Status:** OPEN

### KAI-AUDIO-008 — MEDIUM — Memory writes are not validated
**Issue:** `_auto_memorize` and `_send_wake_nudge` await HTTP POST calls but do not inspect response status or body.  
**Risk:** 4xx/5xx failures are treated as successful and transcript persistence silently fails.  
**Recommendation:** Validate responses and use a durable, idempotent outbox.  
**Status:** OPEN

### KAI-AUDIO-009 — MEDIUM — Injection control is trivially incomplete
**Issue:** Injection detection is a fixed regex matching a small set of English phrases. Content not matching those exact patterns is accepted and may be memorised.  
**Risk:** Rephrased, multilingual, encoded or indirect instructions bypass the control while the system presents the transcript as checked.  
**Recommendation:** Treat all transcripts as untrusted data, preserve provenance and prevent them from becoming executable instructions regardless of phrase matching.  
**Status:** OPEN

### KAI-AUDIO-010 — MEDIUM — Failures masquerade as transcripts
**Issue:** Model and API failures return strings beginning with `[transcript: error ...]` rather than structured failure states. Capture endpoints still return `status="ok"` unless injection is detected.  
**Risk:** Callers and downstream components can interpret failed transcription as valid content or a successful capture.  
**Recommendation:** Return typed error/degraded results and never mark failed transcription as ok.  
**Status:** OPEN

### KAI-AUDIO-011 — MEDIUM — Health is not capability-aware
**Issue:** `/health` always returns `status: ok`, even when the selected local backend is unavailable, microphone capture is unavailable or backend configuration is invalid.  
**Risk:** Orchestration routes work to a service incapable of performing its advertised function.  
**Recommendation:** Separate liveness from microphone, model and external-backend readiness.  
**Status:** OPEN

### KAI-AUDIO-012 — MEDIUM — Emotion output is not a reliable voice analysis
**Issue:** Emotion scores use substring presence in transcript text and fixed RMS thresholds. Keywords are not token-bound, contextual or negation-aware; RMS is affected by microphone gain, distance and file encoding. The code labels these outputs as voice emotion and may generate nudges.  
**Risk:** Ordinary phrases and audio levels produce false stress, fatigue or calm classifications that can contaminate behavioural context.  
**Recommendation:** Label output as low-confidence heuristic metadata and require validation before behavioural action or persistence.  
**Status:** OPEN

### KAI-AUDIO-013 — MEDIUM — Buffers are worker-local
**Issue:** Transcript and wake-detection histories are module-level lists.  
**Risk:** Multiple workers expose different histories, and restart clears evidence without an explicit retention event.  
**Recommendation:** Use a protected shared store with explicit retention or remove server-side history.  
**Status:** OPEN

### KAI-AUDIO-014 — MEDIUM — Configuration lacks validation
**Issue:** Sample rate, recording duration, backend name, model name, URLs and directory paths are accepted directly. Unknown backend values silently fall into stub behaviour.  
**Risk:** Misconfiguration causes false-success stub responses, excessive resource use or unsafe storage/network destinations.  
**Recommendation:** Validate typed configuration and fail readiness on unsupported combinations.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **2**
- High: **5**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **316**
- Critical: **34**
- High: **130**
- Medium: **149**
- Low: **3**

## Files materially reviewed in this batch

`perception/audio/app.py`.
