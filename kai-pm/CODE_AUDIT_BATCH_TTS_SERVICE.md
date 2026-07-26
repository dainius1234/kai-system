# Kai Code Audit — TTS Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TTS-001 | HIGH | Unauthenticated callers can use the service as an external Microsoft TTS relay |
| KAI-TTS-002 | HIGH | Default backend sends supplied text outside the sovereign/local trust boundary |
| KAI-TTS-003 | HIGH | Complete generated audio is buffered in memory before streaming |
| KAI-TTS-004 | MEDIUM | Caller-controlled voice identifiers are passed through without allowlisting |
| KAI-TTS-005 | MEDIUM | Caller-controlled rate and volume strings are not validated |
| KAI-TTS-006 | MEDIUM | Upstream exception details are exposed in HTTP responses |
| KAI-TTS-007 | MEDIUM | Health checks package import rather than upstream readiness |
| KAI-TTS-008 | MEDIUM | Synthesis counters are process-local and non-durable |
| KAI-TTS-009 | MEDIUM | No request throttling or synthesis-cost controls are implemented |
| KAI-TTS-010 | MEDIUM | Backend configuration accepts unsupported values without startup failure |

---

## TTS service: `output/tts/app.py`

### KAI-TTS-001 — HIGH — Unauthenticated synthesis relay
**Issue:** `POST /synthesize` requires no authentication or authorisation. Any reachable caller can submit up to 5,000 characters and cause the service to invoke the configured speech backend.  
**Risk:** The endpoint can be abused for repeated external requests, CPU/memory/network consumption and production of arbitrary speech under Kai’s voice identity.  
**Recommendation:** Require authenticated callers, per-identity quotas and authorised voice scopes.  
**Status:** OPEN

### KAI-TTS-002 — HIGH — Text crosses the local trust boundary by default
**Issue:** `TTS_BACKEND` defaults to `edge`; `_synthesize_edge` sends caller-supplied text to Microsoft Edge TTS over the internet. The module documentation itself states that this backend requires internet.  
**Risk:** Content provided by internal services or users leaves the sovereign stack, potentially including confidential responses, personal data or operational information. This contradicts an air-gapped/local-processing expectation unless explicitly disclosed and approved.  
**Recommendation:** Default to a local backend and require explicit policy approval, classification and redaction before any external synthesis.  
**Status:** OPEN

### KAI-TTS-003 — HIGH — Audio is fully buffered in memory
**Issue:** `_synthesize_edge` accumulates every audio chunk into `io.BytesIO`, calls `getvalue()`, then wraps the complete byte string in another `BytesIO` for `StreamingResponse`.  
**Risk:** Concurrent long synthesis requests duplicate and retain full audio payloads in memory, creating avoidable memory amplification and denial-of-service potential.  
**Recommendation:** Stream validated chunks directly with bounded buffering and cancellation handling.  
**Status:** OPEN

### KAI-TTS-004 — MEDIUM — Arbitrary voice identifiers are accepted
**Issue:** Unknown `request.voice` values are passed directly to `edge_tts.Communicate` rather than rejected against `VOICE_MAP`.  
**Risk:** Callers can access unreviewed voices, trigger repeated upstream errors or change the assistant’s identity characteristics outside approved presets.  
**Recommendation:** Restrict synthesis to an explicit allowlist of reviewed voice IDs.  
**Status:** OPEN

### KAI-TTS-005 — MEDIUM — Rate and volume are unvalidated
**Issue:** `rate` and `volume` are arbitrary strings supplied directly to `edge_tts.Communicate`.  
**Risk:** Invalid or extreme values generate upstream failures and can be used to amplify error traffic or produce unintended output.  
**Recommendation:** Parse and enforce bounded numeric percentage formats.  
**Status:** OPEN

### KAI-TTS-006 — MEDIUM — Internal/upstream errors are disclosed
**Issue:** Network and synthesis exceptions are interpolated directly into HTTP error details, including `TTS upstream unreachable: {e}`, `TTS upstream error: {e}` and `TTS synthesis error: {e}`.  
**Risk:** Callers receive network, protocol, dependency and endpoint diagnostics useful for reconnaissance.  
**Recommendation:** Return stable error categories and protected trace identifiers.  
**Status:** OPEN

### KAI-TTS-007 — MEDIUM — Health is not readiness-aware
**Issue:** `/health` reports `ok` whenever the Python `edge_tts` package imports successfully. It does not verify network access, upstream acceptance, selected voice validity or successful recent synthesis.  
**Risk:** Orchestration treats the service as ready while all synthesis requests can fail.  
**Recommendation:** Separate liveness, backend configuration and verified synthesis readiness.  
**Status:** OPEN

### KAI-TTS-008 — MEDIUM — Metrics are worker-local
**Issue:** `_synth_count` and `_synth_chars` are module-level counters.  
**Risk:** Multiple workers return inconsistent totals and restart erases usage history.  
**Recommendation:** Use shared telemetry or label counters explicitly as per-process.  
**Status:** OPEN

### KAI-TTS-009 — MEDIUM — No throttling or cost boundary
**Issue:** The service implements only a per-request character limit. There is no caller identity, request rate limit, concurrency bound, daily quota or circuit breaker around the external backend.  
**Risk:** Repeated requests can saturate workers, network connections and the upstream service.  
**Recommendation:** Apply authenticated quotas, bounded concurrency and upstream circuit breaking.  
**Status:** OPEN

### KAI-TTS-010 — MEDIUM — Unsupported backend values fail only at request time
**Issue:** `TTS_BACKEND` is accepted as an arbitrary environment string. Unsupported values do not fail startup; `/health` can still report `ok` if `edge_tts` imports, while `/synthesize` returns 503.  
**Risk:** Deployment misconfiguration remains hidden until runtime traffic arrives.  
**Recommendation:** Validate backend configuration at startup and fail readiness on unsupported combinations.  
**Status:** OPEN

---

## Batch totals

- Findings: **10**
- Critical: **0**
- High: **3**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **376**
- Critical: **40**
- High: **148**
- Medium: **185**
- Low: **3**

## Files materially reviewed in this batch

`output/tts/app.py`.
