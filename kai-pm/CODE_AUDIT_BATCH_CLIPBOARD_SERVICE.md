# Kai Code Audit — Clipboard Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CLIP-001 | CRITICAL | Unauthenticated callers can poison clipboard-derived agent context |
| KAI-CLIP-002 | HIGH | Clipboard contents and history are exposed without authentication |
| KAI-CLIP-003 | HIGH | Unauthenticated callers can erase all clipboard history |
| KAI-CLIP-004 | HIGH | Request bodies are fully parsed before the content-size limit is enforced |
| KAI-CLIP-005 | MEDIUM | History limit accepts negative values with unintended slicing behaviour |
| KAI-CLIP-006 | MEDIUM | Source metadata is unbounded and caller-controlled |
| KAI-CLIP-007 | MEDIUM | Queue overflow silently discards older clipboard entries |
| KAI-CLIP-008 | MEDIUM | State and identifiers are process-local and volatile |
| KAI-CLIP-009 | MEDIUM | Counter and history updates are not synchronised |
| KAI-CLIP-010 | MEDIUM | Error-budget recording passes a Boolean and omits exceptions |
| KAI-CLIP-011 | MEDIUM | Configuration values are not validated |

---

## Clipboard service: `perception/clipboard/app.py`

### KAI-CLIP-001 — CRITICAL — Clipboard-derived context can be injected remotely
**Issue:** `POST /push` requires no authentication or source verification. The module documentation states that the agentic layer reads `/latest` to enrich context when the operator refers to “this” or “what I just copied.” Caller-controlled content and source labels are therefore inserted into a trusted contextual input channel.  
**Risk:** Any network-reachable caller can replace the apparent clipboard with fabricated instructions, secrets, URLs or misleading data that may influence subsequent agent reasoning and tool use as though it came from the operator’s local copy action.  
**Recommendation:** Bind clipboard events to an authenticated browser session/device, preserve signed provenance and treat clipboard text as untrusted quoted data rather than instructions.  
**Status:** OPEN — immediate remediation required

### KAI-CLIP-002 — HIGH — Clipboard data is publicly readable
**Issue:** `GET /latest` and `GET /history` return full clipboard contents, source labels, timestamps and IDs without authentication.  
**Risk:** Copied passwords, tokens, personal data, internal URLs, documents and messages can be disclosed to any reachable caller.  
**Recommendation:** Require user-scoped access and minimise clipboard retention and exposure.  
**Status:** OPEN

### KAI-CLIP-003 — HIGH — Clipboard evidence can be erased remotely
**Issue:** `DELETE /history` clears the complete history without authentication or audit.  
**Risk:** A caller can remove evidence of prior clipboard activity, suppress context expected by the operator and interfere with incident investigation.  
**Recommendation:** Require authenticated acknowledgement and retain immutable audit metadata where justified.  
**Status:** OPEN

### KAI-CLIP-004 — HIGH — Size check occurs after request allocation
**Issue:** Pydantic/FastAPI must parse the complete JSON string into `req.content` before `len(req.content.encode())` checks `MAX_CONTENT_BYTES`.  
**Risk:** Oversized request bodies can consume memory and CPU despite the apparent 256 KB content limit.  
**Recommendation:** Enforce body-size limits at the ASGI/reverse-proxy boundary before JSON parsing.  
**Status:** OPEN

### KAI-CLIP-005 — MEDIUM — Negative history limits are accepted
**Issue:** `/history` computes `limit = min(limit, MAX_HISTORY)` without a lower bound. Negative values are used directly in `list(_history)[-limit:]`, producing unintuitive subsets rather than rejection.  
**Risk:** API behaviour is inconsistent and can disclose a broader or different set of entries than the caller or consumer expects.  
**Recommendation:** Validate `limit` with explicit inclusive bounds.  
**Status:** OPEN

### KAI-CLIP-006 — MEDIUM — Source metadata is unbounded and unverified
**Issue:** `source` is an arbitrary optional string with no maximum length or allowlist and is stored and returned as provenance.  
**Risk:** Callers can forge trusted-looking origins and consume memory/response capacity with oversized source labels.  
**Recommendation:** Derive source identity from authenticated transport metadata and enforce a bounded enum.  
**Status:** OPEN

### KAI-CLIP-007 — MEDIUM — Overflow silently drops older entries
**Issue:** `_history` is a `deque(maxlen=MAX_HISTORY)`. New entries automatically evict the oldest entry without an event or metric.  
**Risk:** Flooding the unauthenticated push endpoint can rapidly remove legitimate clipboard history.  
**Recommendation:** Apply authenticated quotas and explicit overflow handling.  
**Status:** OPEN

### KAI-CLIP-008 — MEDIUM — Clipboard state is worker-local and non-durable
**Issue:** History and `_counter` exist only as module-level process memory.  
**Risk:** Restarts erase all state; multiple workers expose different histories and duplicate identifiers.  
**Recommendation:** Use a protected shared store or enforce a single authoritative process with explicit volatility semantics.  
**Status:** OPEN

### KAI-CLIP-009 — MEDIUM — Shared-state updates are unsynchronised
**Issue:** Deduplication, counter increment and append are separate operations with no lock or transaction.  
**Risk:** Concurrent requests can bypass consecutive deduplication, produce inconsistent IDs or reorder entries.  
**Recommendation:** Make insertion and ID allocation atomic.  
**Status:** OPEN

### KAI-CLIP-010 — MEDIUM — Reliability telemetry is incorrectly recorded
**Issue:** Middleware calls `budget.record(response.status_code >= 500)`, passing a Boolean instead of the actual status code, and does not record exceptions raised before a response.  
**Risk:** Error-budget metrics can be inaccurate or omit failures.  
**Recommendation:** Record actual HTTP status codes and exception outcomes consistently.  
**Status:** OPEN

### KAI-CLIP-011 — MEDIUM — Configuration lacks validation
**Issue:** Port, maximum history and maximum byte values are parsed directly. Zero, negative or invalid values can fail startup or produce unintended queue and rejection behaviour.  
**Risk:** Misconfiguration can disable retention, allow no content or make the service unavailable.  
**Recommendation:** Validate typed configuration with explicit safe ranges.  
**Status:** OPEN

---

## Batch totals

- Findings: **11**
- Critical: **1**
- High: **3**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **398**
- Critical: **41**
- High: **155**
- Medium: **199**
- Low: **3**

## Files materially reviewed in this batch

`perception/clipboard/app.py`.
