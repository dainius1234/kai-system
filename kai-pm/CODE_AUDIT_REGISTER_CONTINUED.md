# Kai Code Audit Register — Continued Findings

Repository: `dainius1234/kai-system`  
Parent register: `kai-pm/CODE_AUDIT_REGISTER.md`  
Status: ACTIVE CONTINUATION  
Started: 26 July 2026

This file continues the numbered master audit register from finding 39. It will be consolidated into the final full defect register at audit completion.

---

## Trust integration: `agentic/trust_integration.py`

### KAI-TRUST-001 — CRITICAL — Shared autonomy gate fails open when governance controls are unavailable

**Issue:** `gate_autonomous_action()` starts with `allowed = True` and catches failures from TrustCore and Ohana alignment checks without changing the decision to denied. Missing imports, corrupt state, programming failures or unavailable controls therefore result in permission being granted.

**Risk:** Any autonomous capability relying on this shared gateway can execute precisely when the trust or moral-governance layer is broken. This defeats the intended security boundary across web access, strategy, model council, trading and other autonomous callers.

**Recommendation:** Fail closed for all consequential autonomous actions. Distinguish `denied` from `governance_unavailable`, emit a critical audit event and require explicit operator override through a separately authenticated path.

**Status:** OPEN — immediate remediation required

### KAI-TRUST-002 — HIGH — Low moral alignment is warning-only rather than enforcement

**Issue:** Ohana alignment blocks only an exact value of `0.0`. Any value above zero but below `0.5` merely logs a warning and continues.

**Risk:** Actions assessed as materially misaligned can still execute. Floating-point or model-derived alignment values near zero bypass the only blocking condition.

**Recommendation:** Define documented policy thresholds by action class, fail closed below the applicable threshold and require explicit operator approval for borderline results. Validate alignment as finite and within `[0,1]`.

**Status:** OPEN

### KAI-TRUST-003 — HIGH — Trust increases from self-declared model confidence rather than verified behaviour

**Issue:** `record_chat_response()` marks responses with conviction at least 5 as successful and adds consistency evidence whenever conviction is at least 7. The evidence is based on Kai's own confidence score, not correctness, user confirmation or measured outcome.

**Risk:** An overconfident model can increase its own trust level through repeated high-confidence responses, creating a self-reinforcing autonomy escalation path unrelated to actual reliability.

**Recommendation:** Never use self-reported conviction as positive trust evidence by itself. Require externally verifiable outcomes, operator review, benchmark performance or later consistency checks; apply negative evidence for corrections and contradictions.

**Status:** OPEN

### KAI-TRUST-004 — MEDIUM — Trust-ledger recording is described as nonblocking but performs synchronous filesystem work

**Issue:** `_record_nonblocking()` is synchronous and constructs a ledger object and appends to storage inline on the caller's execution path. It is fire-and-forget only in the sense that exceptions are swallowed.

**Risk:** Slow or blocked filesystem operations can delay request handling and autonomous control paths, while swallowed failures create silent audit gaps.

**Recommendation:** Rename the function accurately or move writes to a bounded asynchronous queue with backpressure, health metrics, durable retry and explicit audit-gap alarms.

**Status:** OPEN

---

## Trust core: `agentic/trust_core.py`

### KAI-TRUST-005 — CRITICAL — Trust level and evidence persistence are tamperable local files

**Issue:** The current trust level, evidence scores and audit trail are stored in ordinary JSON/JSONL files without authentication, integrity protection, access-control verification, atomic writes or an external root of trust.

**Risk:** Any process or attacker with filesystem write access can raise Kai directly to GUARDIAN, inflate evidence scores, erase revocations or rewrite the audit history. Corruption can also reset state unpredictably.

**Recommendation:** Store trust state in a protected transactional store; cryptographically authenticate records and append-only events; restrict write identity; use monotonic revisions and independently anchored audit checkpoints.

**Status:** OPEN — immediate remediation required

### KAI-TRUST-006 — HIGH — Trust mutations and capability checks are concurrency-unsafe

**Issue:** `can_do()`, `record_evidence()`, `grant()` and `revoke()` mutate one in-memory record and rewrite the same JSON file without locks or compare-and-swap semantics.

**Risk:** Concurrent requests can lose evidence, overwrite revocations, corrupt counters or persist a stale higher trust level after a lower level was granted by another worker.

**Recommendation:** Use transactional updates with version checks and worker-shared state. Revocation must have precedence and should be guarded by an atomic monotonic policy version.

**Status:** OPEN

### KAI-TRUST-007 — HIGH — Auto-promotion accepts unbounded caller-supplied evidence

**Issue:** `record_evidence()` accepts arbitrary positive scores and immediately runs `_check_promotion()`. There is no per-event cap, evidence-source authentication, deduplication, expiry or requirement for independent validation.

**Risk:** A single internal caller, compromised route or programming error can add enough evidence to advance multiple trust levels. Repeated duplicate events can also accumulate permanent autonomy.

**Recommendation:** Authenticate evidence producers, cap and normalise event scores, require evidence IDs and deduplication, apply decay, and require operator approval for high-level promotions even when numerical thresholds are met.

**Status:** OPEN

### KAI-TRUST-008 — MEDIUM — Corrupt trust state silently resets to DORMANT without preserving forensic evidence

**Issue:** `_load_record()` logs a warning and creates a fresh default record when parsing fails. The corrupt file is neither quarantined nor preserved with a recovery marker.

**Risk:** A storage fault or partial write can erase the effective trust state and evidence history. The reset may conceal tampering and produce inconsistent behaviour across restarts.

**Recommendation:** Fail safe into a locked governance state, quarantine the corrupt file, preserve hashes and metadata, alert the operator and require an explicit recovery procedure.

**Status:** OPEN

---

## Continuation summary

- New findings in this continuation: 8
- Critical: 2
- High: 4
- Medium: 2
- Cumulative findings across both registers: 47
- Cumulative Critical: 8
- Cumulative High: 23
- Cumulative Medium: 15
- Cumulative Low: 1
- Additional files materially reviewed: `agentic/trust_integration.py`, `agentic/trust_core.py`
- Current security posture: HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE
- Audit state: IN PROGRESS
