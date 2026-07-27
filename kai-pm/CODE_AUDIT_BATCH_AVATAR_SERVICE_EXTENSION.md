# Kai Code Audit — Avatar Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_AVATAR_SERVICE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-AVATARX-001 | HIGH | Shared sanitisation silently truncates speech text before the service reports it queued |
| KAI-AVATARX-002 | HIGH | Shared sanitisation strips punctuation/operators and can change the intended spoken content |
| KAI-AVATARX-003 | HIGH | The default WebRTC port is the same as the Avatar HTTP service port |
| KAI-AVATARX-004 | HIGH | Avatar requests have no request ID, idempotency key, queue identity or delivery receipt |
| KAI-AVATARX-005 | MEDIUM | `/speak` exposes the internal TTS URL in every response |
| KAI-AVATARX-006 | MEDIUM | The service has no audit, metrics or structured operation logging |
| KAI-AVATARX-007 | MEDIUM | No feature/readiness state distinguishes the current stub from a functioning avatar renderer |
| KAI-AVATARX-008 | MEDIUM | Requests carry no authenticated speaker, recipient, avatar identity or output-purpose metadata |
| KAI-AVATARX-009 | MEDIUM | Returned voice/emotion/text contain no source digest, model/render version or safe presentation schema |
| KAI-AVATARX-010 | MEDIUM | The service has no lifespan-owned downstream client, queue, shutdown drain or delivery reconciliation |

---

### KAI-AVATARX-001 — HIGH — Silent speech truncation
**Issue:** `sanitize_string(request.text)` applies the shared 1,024-character cap after the entire body is parsed. The response then says the request was queued without indicating truncation.  
**Risk:** Important qualifications or instructions can be removed while callers believe the full speech was accepted.  
**Recommendation:** validate the original bounded input and return an explicit truncated/rejected state; do not silently alter output content.  
**Status:** OPEN

### KAI-AVATARX-002 — HIGH — Speech semantics are altered
The generic sanitizer removes characters such as semicolons, pipes and ampersands from voice/emotion/text. These may change names, formulae, pauses or instructions.

### KAI-AVATARX-003 — HIGH — Default port collision
`WEBRTC_PORT` defaults to 8081 and the FastAPI process also defaults to port 8081. A future WebRTC listener cannot bind the advertised port in the same network namespace.

### KAI-AVATARX-004 — HIGH — No operation identity
The endpoint returns no immutable request/queue/delivery ID, so retries, duplicates and committed-but-unknown outcomes cannot be reconciled.

### KAI-AVATARX-005 — MEDIUM — Per-request topology disclosure
Every speak response returns the internal TTS destination, duplicating the health disclosure and propagating topology into callers/logs.

### KAI-AVATARX-006 — MEDIUM — No operational accountability
The service has no request audit, metrics, delivery counters or structured failure state.

### KAI-AVATARX-007 — MEDIUM — Stub capability is not explicit
There is no feature flag or implementation-status field that prevents consumers from treating `status:ok` as a working avatar channel.

### KAI-AVATARX-008 — MEDIUM — Missing output authority
No authenticated initiator, intended audience, avatar identity, consent or message class accompanies the speech request.

### KAI-AVATARX-009 — MEDIUM — Missing render provenance
Returned content has no digest, renderer/TTS/model revision, sanitisation/truncation state or safe display contract.

### KAI-AVATARX-010 — MEDIUM — Missing lifecycle ownership
No lifespan manages a downstream TTS/WebRTC client, durable queue, cancellation or shutdown reconciliation.

---

## Batch totals

- Findings: **10**
- Critical: **0**
- High: **4**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,500**
- Critical: **191**
- High: **1,251**
- Medium: **1,055**
- Low: **3**

## Files materially reviewed

`output/avatar/app.py` and the existing Avatar Service audit.
