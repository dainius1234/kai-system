# Kai Code Audit — Avatar Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AVATAR-001 | HIGH | Unauthenticated callers can submit arbitrary avatar speech requests |
| KAI-AVATAR-002 | HIGH | The endpoint reports requests queued without performing or confirming any queue operation |
| KAI-AVATAR-003 | MEDIUM | Submitted text, voice and emotion fields are unbounded before sanitisation |
| KAI-AVATAR-004 | MEDIUM | Caller-controlled voice and emotion values are accepted without allowlists |
| KAI-AVATAR-005 | MEDIUM | The full submitted text is echoed back to the caller |
| KAI-AVATAR-006 | MEDIUM | Health exposes internal TTS routing and WebRTC port information |
| KAI-AVATAR-007 | MEDIUM | Health always reports ok without checking TTS or WebRTC readiness |
| KAI-AVATAR-008 | MEDIUM | Configuration values are not validated |

---

## Avatar service: `output/avatar/app.py`

### KAI-AVATAR-001 — HIGH — Unauthenticated avatar speech submission
**Issue:** `POST /speak` requires no authentication or authorisation and accepts arbitrary text, voice and emotion fields.  
**Risk:** Any reachable caller can submit content represented as a Kai avatar speech request, potentially impersonating the assistant if a downstream implementation later consumes the same contract.  
**Recommendation:** Require authenticated, authorised callers and signed provenance for avatar output requests.  
**Status:** OPEN

### KAI-AVATAR-002 — HIGH — False queue acknowledgement
**Issue:** `/speak` returns `status: ok` and `message: avatar request queued`, but the function does not call TTS, WebRTC, a queue or any other downstream component.  
**Risk:** Callers and operators receive a positive delivery acknowledgement for an action that was never attempted. This can suppress retries and conceal a non-functional output channel.  
**Recommendation:** Return an explicit not-implemented state or perform a real durable queue operation and expose its identifier.  
**Status:** OPEN

### KAI-AVATAR-003 — MEDIUM — Input sizes are not bounded
**Issue:** Text, voice and emotion are parsed as unrestricted strings. Sanitisation occurs only after the complete request has been allocated.  
**Risk:** Oversized requests consume memory and response capacity and are echoed back.  
**Recommendation:** Enforce body and per-field limits at schema and transport boundaries.  
**Status:** OPEN

### KAI-AVATAR-004 — MEDIUM — Voice and emotion are unvalidated
**Issue:** Caller-supplied voice and emotion labels are sanitised but not checked against supported values.  
**Risk:** Consumers may receive unsupported or misleading avatar state, and callers can bypass intended identity/emotion presets.  
**Recommendation:** Use strict enums tied to implemented avatar capabilities.  
**Status:** OPEN

### KAI-AVATAR-005 — MEDIUM — Submitted text is reflected
**Issue:** The endpoint returns the sanitised full text in its response.  
**Risk:** Sensitive speech content is unnecessarily duplicated in API responses and intermediary logs/proxies.  
**Recommendation:** Return only an opaque request ID and status metadata.  
**Status:** OPEN

### KAI-AVATAR-006 — MEDIUM — Internal routing metadata is public
**Issue:** `/health` exposes `TTS_URL` and `WEBRTC_PORT` without authentication.  
**Risk:** Callers learn internal service names, ports and routing structure useful for reconnaissance.  
**Recommendation:** Restrict operational details or return only capability state.  
**Status:** OPEN

### KAI-AVATAR-007 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always reports `status: ok` and does not test TTS connectivity, WebRTC availability or any actual avatar renderer/queue.  
**Risk:** Orchestration treats a non-functional stub as ready.  
**Recommendation:** Separate liveness from verified downstream readiness and implementation status.  
**Status:** OPEN

### KAI-AVATAR-008 — MEDIUM — Configuration lacks validation
**Issue:** TTS URL, WebRTC port and service port are accepted directly from environment values.  
**Risk:** Invalid routing or port values remain hidden until runtime, while health continues reporting ok.  
**Recommendation:** Validate typed startup configuration and approved URL schemes/hosts.  
**Status:** OPEN

---

## Batch totals

- Findings: **8**
- Critical: **0**
- High: **2**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **531**
- Critical: **59**
- High: **190**
- Medium: **279**
- Low: **3**

## Files materially reviewed in this batch

`output/avatar/app.py`.
