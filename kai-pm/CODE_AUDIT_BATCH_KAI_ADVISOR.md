# Kai Code Audit — KAI Advisor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-ADVISOR-001 | HIGH | Deployed advisor image contains no `docs` directory and therefore starts with an empty knowledge base |
| KAI-ADVISOR-002 | HIGH | Service reports a configured model and device despite performing no model inference |
| KAI-ADVISOR-003 | HIGH | Case-insensitive matching followed by case-sensitive splitting can return an entire matched document |
| KAI-ADVISOR-004 | HIGH | Unauthenticated callers can query and retrieve loaded documentation content |
| KAI-ADVISOR-005 | HIGH | Substring retrieval returns raw document suffixes rather than bounded answers |
| KAI-ADVISOR-006 | MEDIUM | All Markdown documents are loaded fully into memory at import time without limits |
| KAI-ADVISOR-007 | MEDIUM | Relative `docs` path makes behaviour depend on process working directory |
| KAI-ADVISOR-008 | MEDIUM | Document read failures are silently ignored |
| KAI-ADVISOR-009 | MEDIUM | Question size is unbounded and the full question is reflected in responses |
| KAI-ADVISOR-010 | MEDIUM | Knowledge is never refreshed after process startup |
| KAI-ADVISOR-011 | MEDIUM | Health reports ok with zero knowledge and no model/runtime readiness |
| KAI-ADVISOR-012 | MEDIUM | GPU and model configuration only alter labels and are weakly validated |
| KAI-ADVISOR-013 | MEDIUM | Service provides no metrics, audit trail or request provenance |

---

## KAI Advisor: `kai-advisor/app.py`, `kai-advisor/Dockerfile`

### KAI-ADVISOR-001 — HIGH — Deployed knowledge base is empty
**Issue:** The application walks the relative directory `docs` at import time. The service Dockerfile copies `app.py`, `common/` and `data/`, but does not copy `docs/`. In the deployed image the loop therefore finds no Markdown files and `knowledge` remains empty.  
**Risk:** The published, health-checked advisor cannot answer from the repository documentation and always falls back to its stub message, while deployment marks it healthy.  
**Recommendation:** Remove the service until implemented or package a versioned approved knowledge corpus and fail readiness when it is unavailable.  
**Status:** OPEN

### KAI-ADVISOR-002 — HIGH — False model/device capability representation
**Issue:** `MODEL` defaults to `deepseek-v4` and `DEVICE` is derived from `USE_GPU`, but `/ask` never loads or calls a model and performs no CPU/GPU inference. These labels are returned in health and every answer.  
**Risk:** Callers can be led to believe responses were produced by a named model on a stated device when they are only string-search/stub output.  
**Recommendation:** Report the actual implementation type and only expose model/device metadata after verified initialisation and inference.  
**Status:** OPEN

### KAI-ADVISOR-003 — HIGH — Case mismatch can disclose a whole document
**Issue:** Matching uses `if q.lower() in chunk.lower()`, but extraction uses the case-sensitive `chunk.split(q, 1)`. If the phrase exists only with different case, `split` finds no delimiter and `[-1]` returns the complete document.  
**Risk:** In deployments where documentation is mounted or copied, a caller can use case variants of known terms to retrieve entire Markdown files.  
**Recommendation:** Do not return raw source documents; use explicit indexed passages with access controls, bounded excerpts and correct match offsets.  
**Status:** OPEN

### KAI-ADVISOR-004 — HIGH — Documentation retrieval is unauthenticated
**Issue:** `POST /ask` requires no authentication or authorisation and returns matching loaded documentation content. The service is published on host port 8090 in the full Compose deployment.  
**Risk:** Any reachable caller can probe and exfiltrate whatever internal Markdown content is present in the runtime `docs` path.  
**Recommendation:** Require scoped access and classify/redact the knowledge corpus before retrieval.  
**Status:** OPEN

### KAI-ADVISOR-005 — HIGH — Retrieval returns arbitrary raw suffixes
**Issue:** On an exact-case match, the answer is everything after the first occurrence of the question string in the first matching document. There is no section detection, relevance scoring, maximum answer length or source boundary.  
**Risk:** A short/common query can return a very large unrelated remainder of an internal document, including content far beyond the matching passage.  
**Recommendation:** Return bounded passage windows with source identifiers and relevance validation.  
**Status:** OPEN

### KAI-ADVISOR-006 — MEDIUM — Import-time memory allocation is unbounded
**Issue:** Every `*.md` file beneath `docs` is read fully and retained as one string in the global list. No file count, per-file size or aggregate corpus limit exists.  
**Risk:** A large mounted documentation tree can delay startup or exhaust memory before the service begins listening.  
**Recommendation:** Build a bounded offline index and validate corpus size during controlled startup.  
**Status:** OPEN

### KAI-ADVISOR-007 — MEDIUM — Knowledge path depends on working directory
**Issue:** `os.walk("docs")` is relative to the process current working directory, not the application file or an explicit configured path.  
**Risk:** The same image/code can silently load no knowledge or an unintended directory depending on launch context.  
**Recommendation:** Use an explicit validated absolute knowledge path.  
**Status:** OPEN

### KAI-ADVISOR-008 — MEDIUM — Document failures disappear
**Issue:** All exceptions while reading Markdown files are suppressed. No failed-file count or error state is retained.  
**Risk:** Missing permissions, decoding errors or corrupted documents silently reduce the corpus while health still reports ok.  
**Recommendation:** Record per-file failures and fail readiness when required knowledge cannot be loaded.  
**Status:** OPEN

### KAI-ADVISOR-009 — MEDIUM — Query input is unbounded and reflected
**Issue:** `question` has no maximum length. The complete stripped string is scanned against every document and returned in the response or embedded in the stub text.  
**Risk:** Oversized input consumes CPU/memory and is duplicated into responses and intermediary logs.  
**Recommendation:** Enforce strict request/body length and return an opaque request ID rather than unnecessary reflection.  
**Status:** OPEN

### KAI-ADVISOR-010 — MEDIUM — Knowledge cannot update safely
**Issue:** The corpus is loaded once at module import. Changes to mounted documents are not detected, versioned or reloaded.  
**Risk:** The service serves stale documentation until restart and cannot prove which corpus version produced an answer.  
**Recommendation:** Use immutable versioned indexes with controlled atomic activation.  
**Status:** OPEN

### KAI-ADVISOR-011 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when `knowledge_chunks` is zero and no model exists.  
**Risk:** Compose and health tooling mark a non-functional advisor ready.  
**Recommendation:** Separate liveness from knowledge/model readiness and fail readiness when required capability is absent.  
**Status:** OPEN

### KAI-ADVISOR-012 — MEDIUM — Configuration is cosmetic and weakly validated
**Issue:** Any `KAI_MODEL` string is accepted; `USE_GPU` recognises a few truthy strings and simply changes the returned device label. Port parsing is direct.  
**Risk:** Deployment metadata can claim arbitrary model/device capability and invalid ports fail at runtime.  
**Recommendation:** Validate typed configuration against actually initialised components.  
**Status:** OPEN

### KAI-ADVISOR-013 — MEDIUM — No operational accountability
**Issue:** The service implements no structured logging, metrics, error budget, source citation, retrieval trace or caller provenance.  
**Risk:** Document access and misleading responses cannot be audited or attributed.  
**Recommendation:** Add authenticated request auditing, source-span metadata and reliability telemetry before production use.  
**Status:** OPEN

---

## Batch totals

- Findings: **13**
- Critical: **0**
- High: **5**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **757**
- Critical: **82**
- High: **266**
- Medium: **406**
- Low: **3**

## Files materially reviewed in this batch

`kai-advisor/app.py`, `kai-advisor/Dockerfile`, and the relevant `kai-advisor` deployment definition in `docker-compose.full.yml`.
