# Kai Code Audit — Key Rotation Tooling Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-KEY-001 | CRITICAL | HMAC rotation prints active and previous secrets to standard output |
| KAI-KEY-002 | HIGH | HMAC secrets are stored in plaintext with no permission hardening |
| KAI-KEY-003 | HIGH | HMAC rotation state writes are non-atomic |
| KAI-KEY-004 | HIGH | Revoked HMAC secrets remain indefinitely in the same state file |
| KAI-KEY-005 | HIGH | Ed25519 private keys are stored unencrypted with no permission hardening |
| KAI-KEY-006 | HIGH | Ed25519 rotation state writes are non-atomic |
| KAI-KEY-007 | MEDIUM | Timestamp-derived Ed25519 key IDs can collide |
| KAI-KEY-008 | MEDIUM | Rotation state corruption has no safe recovery or forensic mode |
| KAI-KEY-009 | MEDIUM | Rotation configuration accepts unsafe interval values without validation |
| KAI-KEY-010 | MEDIUM | Rotation tooling does not verify consumer rollout before revocation or key removal |

---

### KAI-KEY-001 — CRITICAL — HMAC rotation prints secrets to standard output
**Issue:** After rotation, `auto_rotate_hmac.py` emits JSON containing `INTERSERVICE_HMAC_SECRET` and `INTERSERVICE_HMAC_SECRET_PREV`.  
**Risk:** Active authentication secrets can be captured by CI logs, shell history, process supervisors, log aggregation and operator transcripts. Any reader of those logs can forge inter-service requests.  
**Recommendation:** Never emit secret values. Write them directly to a managed secret backend and output only key identifiers and status.  
**Status:** OPEN — immediate remediation required

### KAI-KEY-002 — HIGH — HMAC secrets are stored in plaintext without permission hardening
**Issue:** All current, previous and revoked secrets are serialised to a normal JSON file. The script does not create the file with restrictive mode, validate ownership or use encrypted storage.  
**Risk:** Any process or user able to read the working directory can recover every inter-service signing secret.  
**Recommendation:** Store keys in a secret manager or OS-protected keystore and enforce least-privilege access.  
**Status:** OPEN

### KAI-KEY-003 — HIGH — HMAC rotation writes are non-atomic
**Issue:** State is replaced directly with `Path.write_text` and no lock, temporary file, fsync or compare-and-swap.  
**Risk:** Concurrent rotation or interruption can corrupt state, lose the current key or create inconsistent revocation metadata.  
**Recommendation:** Use a transactional secret backend or locked atomic durable replacement.  
**Status:** OPEN

### KAI-KEY-004 — HIGH — Revoked HMAC secrets remain indefinitely
**Issue:** Rotation appends old key IDs to `revoked` but keeps every secret value in the `secrets` dictionary.  
**Risk:** Compromise of the state file exposes the entire historical authentication key set and expands retrospective forgery risk where old signatures remain accepted or auditable.  
**Recommendation:** Retain only explicitly required overlap material, securely destroy retired keys and preserve non-secret revocation metadata separately.  
**Status:** OPEN

### KAI-KEY-005 — HIGH — Ed25519 private keys are stored unencrypted without permission hardening
**Issue:** Raw private keys are Base64-encoded and written into `security/ed25519_rotation_state.json` with no encryption, file-mode enforcement or ownership validation.  
**Risk:** Filesystem readers can forge every signature associated with retained keys. Base64 provides no confidentiality.  
**Recommendation:** Keep private keys in a hardware/managed signing service or protected keystore and export public keys only.  
**Status:** OPEN

### KAI-KEY-006 — HIGH — Ed25519 state writes are non-atomic
**Issue:** Initial creation and subsequent saves directly rewrite the complete key state file.  
**Risk:** A crash can destroy all active signing material and make historical signatures unverifiable or services unavailable.  
**Recommendation:** Use transactional key storage with versioned recovery and atomic activation.  
**Status:** OPEN

### KAI-KEY-007 — MEDIUM — Timestamp-derived key IDs can collide
**Issue:** Key IDs use `k{int(time.time())}`, providing only one-second resolution.  
**Risk:** Multiple invocations in the same second can overwrite or alias key entries.  
**Recommendation:** Use a cryptographically random or UUID identifier and reject duplicates.  
**Status:** OPEN

### KAI-KEY-008 — MEDIUM — Corrupt rotation state has no safe recovery mode
**Issue:** Both scripts parse state directly and allow exceptions to terminate execution. No backup, quarantine, integrity check or last-known-good recovery is implemented.  
**Risk:** A partial write or manual damage can halt rotation and leave consumers on stale or unknown keys.  
**Recommendation:** Sign/version state, preserve previous generations and enter an explicit degraded recovery mode.  
**Status:** OPEN

### KAI-KEY-009 — MEDIUM — Rotation intervals are unvalidated
**Issue:** Environment values are converted directly to integers. Zero or negative intervals cause rotation every invocation; extreme values can effectively disable rotation.  
**Risk:** Misconfiguration can create key churn, availability failures or indefinite key lifetime.  
**Recommendation:** Enforce policy-defined minimum and maximum intervals and fail startup on invalid values.  
**Status:** OPEN

### KAI-KEY-010 — MEDIUM — Consumer rollout is not verified
**Issue:** HMAC rotation marks the current key revoked immediately in local state, while Ed25519 rotation removes the previous key on a later timer. Neither script queries consumers, verifies adoption or coordinates an atomic activation point.  
**Risk:** Services can disagree about active keys, causing authentication outages or prolonged acceptance of retired material.  
**Recommendation:** Implement staged publish, consumer acknowledgement, activation and retirement with measurable quorum and rollback.  
**Status:** OPEN

---

## Batch totals

- Findings: **10**
- Critical: **1**
- High: **5**
- Medium: **4**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **169**
- Critical: **22**
- High: **77**
- Medium: **69**
- Low: **1**

## Files materially reviewed in this batch

`scripts/auto_rotate_hmac.py`, `scripts/auto_rotate_ed25519.py`.
