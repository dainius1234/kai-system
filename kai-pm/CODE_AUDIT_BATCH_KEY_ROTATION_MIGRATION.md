# Kai Code Audit — Key Rotation and Authentication Migration Scripts

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Existing request-signature defects in `common/auth.py` and Tool Gate are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-KEYS-001 | CRITICAL | HMAC rotation never persists or outputs the initial secret when state is absent |
| KAI-KEYS-002 | CRITICAL | Successful HMAC rotation prints current and previous secrets to stdout |
| KAI-KEYS-003 | HIGH | HMAC secrets are stored in plaintext with ordinary filesystem permissions |
| KAI-KEYS-004 | HIGH | The previous key is simultaneously marked revoked and emitted for overlap use |
| KAI-KEYS-005 | HIGH | Every historical HMAC secret is retained indefinitely |
| KAI-KEYS-006 | HIGH | Generated HMAC state is not consumed by any running service |
| KAI-KEYS-007 | HIGH | HMAC rotation state writes are non-atomic and not fsynced |
| KAI-KEYS-008 | HIGH | Concurrent HMAC rotations race and can reuse or lose key versions |
| KAI-KEYS-009 | HIGH | Corrupt HMAC state aborts rotation without a protected recovery path |
| KAI-KEYS-010 | MEDIUM | HMAC key-version parsing assumes a simple `v<number>` format |
| KAI-KEYS-011 | MEDIUM | Wall-clock rollback can postpone rotation indefinitely |
| KAI-KEYS-012 | MEDIUM | Zero or negative HMAC rotation intervals rotate on every invocation |
| KAI-KEYS-013 | MEDIUM | HMAC secret and revocation history has no retention bound |
| KAI-KEYS-014 | HIGH | HMAC state path is arbitrary and symlink/ownership checks are absent |
| KAI-KEYS-015 | MEDIUM | HMAC generation has no durable activation or consumer-acknowledgement audit |
| KAI-KEYS-016 | CRITICAL | Ed25519 private signing keys are stored unencrypted in plaintext JSON |
| KAI-KEYS-017 | HIGH | Ed25519 rotation state is not consumed by the running application |
| KAI-KEYS-018 | HIGH | Ed25519 key IDs use second-resolution timestamps and can collide |
| KAI-KEYS-019 | HIGH | Ed25519 rotation drops the immediately previous key during alternate rotations |
| KAI-KEYS-020 | HIGH | Orphaned historical Ed25519 private keys remain in the state file |
| KAI-KEYS-021 | HIGH | Ed25519 state mutation is not tied to verifier/signer activation acknowledgement |
| KAI-KEYS-022 | HIGH | Ed25519 state writes and rotations are concurrency-unsafe and non-atomic |
| KAI-KEYS-023 | HIGH | Corrupt Ed25519 state aborts rotation without fail-safe recovery |
| KAI-KEYS-024 | MEDIUM | Ed25519 rotation uses non-monotonic wall-clock age |
| KAI-KEYS-025 | MEDIUM | Ed25519 rotation interval is not safely validated |
| KAI-KEYS-026 | HIGH | Ed25519 state path is arbitrary and filesystem trust is not verified |
| KAI-KEYS-027 | HIGH | Ed25519 key state has no integrity signature, encryption or protected key provider |
| KAI-KEYS-028 | MEDIUM | Ed25519 retention and revocation semantics are incomplete |
| KAI-KEYS-029 | HIGH | One confirmed HMAC security incident still produces “STAY ON HMAC” |
| KAI-KEYS-030 | HIGH | A zero-trust mandate alone still produces “STAY ON HMAC” |
| KAI-KEYS-031 | HIGH | Migration-advisor defaults materially undercount the deployed service topology |
| KAI-KEYS-032 | HIGH | Migration recommendations rely entirely on unauthenticated environment assertions |
| KAI-KEYS-033 | MEDIUM | Invalid migration inputs silently fall back to benign defaults |
| KAI-KEYS-034 | MEDIUM | Negative and non-finite migration metrics are accepted |
| KAI-KEYS-035 | MEDIUM | Migration scoring gives every signal equal weight regardless of severity |
| KAI-KEYS-036 | MEDIUM | Migration advisor always exits successfully regardless of urgent recommendation |

---

## HMAC rotation: `scripts/auto_rotate_hmac.py`

### KAI-KEYS-001 — CRITICAL — Initialisation never completes
**Issue:** when the state file is absent, `_load()` creates a random in-memory v1 secret with `rotated_at=now`. `main()` sees age below the interval, prints only `rotated:false` and returns without calling `_save()` or outputting the secret.  
**Risk:** every invocation generates and discards a different initial secret; no service can be configured with a stable v1 key, while the script appears healthy.  
**Recommendation:** perform an atomic one-time initialisation and explicitly return a protected activation record.  
**Status:** OPEN — immediate remediation required

### KAI-KEYS-002 — CRITICAL — Secrets are emitted to logs/stdout
**Issue:** after rotation, JSON output includes `INTERSERVICE_HMAC_SECRET` and `INTERSERVICE_HMAC_SECRET_PREV` in plaintext.  
**Risk:** CI logs, shell history, schedulers and monitoring capture both valid current and previous shared credentials, compromising every service using the HMAC trust domain.  
**Recommendation:** write secrets directly to a protected secret manager and output only key IDs/status.  
**Status:** OPEN — immediate remediation required

### KAI-KEYS-003 — HIGH — Plaintext secret archive
The state JSON stores all raw HMAC secrets under a normal relative path with process-default ownership/mode and no encryption.

### KAI-KEYS-004 — HIGH — Revocation and overlap contradict each other
**Issue:** the current key ID is appended to `revoked` at rotation, while the same key’s secret is emitted as `SECRET_PREV` for migration overlap.  
**Risk:** consumers following revocation reject the overlap key; consumers accepting it contradict the revocation state.  
**Recommendation:** use explicit staged states: active, grace, revoked, destroyed.  
**Status:** OPEN

### KAI-KEYS-005 — HIGH — Revoked credentials remain recoverable
The `secrets` dictionary never deletes historical secrets, so compromise of the state file reveals every prior key.

### KAI-KEYS-006 — HIGH — Rotation is operationally disconnected
Repository search shows `HMAC_ROTATION_STATE` is used only by this script. Running services read environment secrets and do not load/reload this state.

### KAI-KEYS-007 — HIGH — Unsafe state commit
`write_text` replaces state directly without temporary-file atomic replacement, fsync, backup, integrity check or restrictive mode.

### KAI-KEYS-008 — HIGH — Rotation races
Two processes can load the same current version, generate the same `next_id`, overwrite state and emit conflicting secrets.

### KAI-KEYS-009 — HIGH — Corruption halts key management
Malformed JSON, missing structures or read errors propagate and terminate the script; no quarantine, previous-good recovery or alert contract exists.

### KAI-KEYS-010 — MEDIUM — Brittle version parser
`int(current.lstrip('v'))` accepts/removes every leading `v` and raises for non-numeric/custom key IDs.

### KAI-KEYS-011 — MEDIUM — Clock changes alter rotation
A backward clock makes age negative and can defer rotation far beyond policy.

### KAI-KEYS-012 — MEDIUM — Unsafe interval
Zero/negative values cause every invocation to rotate; malformed values fail at import.

### KAI-KEYS-013 — MEDIUM — Unbounded state growth
Revoked IDs and secret values accumulate without a maximum or destruction policy.

### KAI-KEYS-014 — HIGH — Untrusted state destination
The environment can select any path; canonical location, symlinks, owner, mode and filesystem type are not verified.

### KAI-KEYS-015 — MEDIUM — No activation evidence
The state records generation time only. It does not track which services loaded each key, grace completion, failed consumers or an authenticated rotation operation ID.

---

## Ed25519 rotation: `scripts/auto_rotate_ed25519.py`

### KAI-KEYS-016 — CRITICAL — Private keys are unencrypted files
**Issue:** raw Ed25519 private bytes are Base64-encoded using `NoEncryption()` and written into JSON.  
**Risk:** read access to the repository/workspace state file gives an attacker signing authority and enables forged service identities/evidence.  
**Recommendation:** keep private keys non-exportable in a protected key-management/HSM service.  
**Status:** OPEN — immediate remediation required

### KAI-KEYS-017 — HIGH — Generated keys are not integrated
Repository search shows `ED25519_STATE_PATH` is used only by this script; signers/verifiers do not consume or reload its keys.

### KAI-KEYS-018 — HIGH — Key-ID collisions
IDs are `k<int(time.time())>`; concurrent or repeated generation in one second overwrites the dictionary entry and creates ambiguous rotations.

### KAI-KEYS-019 — HIGH — Broken grace-key lifecycle
**Issue:** on a run where `drop_previous_on_next` is true, the script first designates the current key as `previous`, then immediately removes that key and sets mode to single.  
**Risk:** the newly activated key has no previous verification grace period, breaking consumers that have not yet updated.  
**Recommendation:** separate activation and retirement into independently acknowledged phases.  
**Status:** OPEN

### KAI-KEYS-020 — HIGH — Old private keys remain orphaned
Only the immediately previous key is optionally removed; earlier keys remain in `keys` indefinitely even when no longer referenced.

### KAI-KEYS-021 — HIGH — No distributed activation protocol
State changes do not wait for signer/verifier deployment, quorum acknowledgement or traffic verification before retiring keys.

### KAI-KEYS-022 — HIGH — Unsafe concurrent persistence
Initial creation and saves directly rewrite JSON with no lock, exclusive creation, atomic rename, fsync or revision compare.

### KAI-KEYS-023 — HIGH — Corruption stops rotation
Malformed/tampered state raises and terminates; no fail-closed service alert or verified backup recovery exists.

### KAI-KEYS-024 — MEDIUM — Wall-clock rotation age
Clock rollback/forward can defer or prematurely trigger key lifecycle transitions.

### KAI-KEYS-025 — MEDIUM — Interval validation missing
Zero/negative/extreme values are accepted and malformed values fail during import.

### KAI-KEYS-026 — HIGH — Arbitrary key-file target
Environment-controlled path is used without canonical, owner, permission, symlink or secure-filesystem validation.

### KAI-KEYS-027 — HIGH — State itself is unauthenticated
The file has no signature/MAC, versioned schema or encrypted envelope; an editor can replace public/private keys and lifecycle fields.

### KAI-KEYS-028 — MEDIUM — Incomplete revocation/retention model
There is no revoked-key list, compromise time, destruction proof, signer identity, activation timestamp per key or maximum retained-key policy.

---

## Migration advisor: `scripts/hmac_migration_advisor.py`

### KAI-KEYS-029 — HIGH — Confirmed compromise does not force migration
**Issue:** one or more incidents triggers one point. With no two other signals, the result remains “STAY ON HMAC”.  
**Risk:** the tool recommends retaining shared-secret architecture after a confirmed recent HMAC incident.  
**Recommendation:** make compromise/identity-boundary failures mandatory migration/escalation signals.  
**Status:** OPEN

### KAI-KEYS-030 — HIGH — Zero-trust mandate is only one vote
A true zero-trust target alone also results in “STAY ON HMAC”, contradicting the stated architecture requirement.

### KAI-KEYS-031 — HIGH — Defaults misrepresent this repository
The default service count is three, while the deployed stack contains substantially more services; absent environment configuration biases the recommendation toward staying on HMAC.

### KAI-KEYS-032 — HIGH — Inputs are assertions, not observations
All topology, teams, incidents, dependencies and auditability values come from local environment variables with no inventory, incident-log or deployment evidence.

### KAI-KEYS-033 — MEDIUM — Invalid values become safe-looking defaults
Parsing failures silently substitute defaults rather than failing the assessment as incomplete.

### KAI-KEYS-034 — MEDIUM — Numeric domains are unvalidated
Negative counts and NaN/infinite auditability values are accepted and alter trigger comparisons unexpectedly.

### KAI-KEYS-035 — MEDIUM — Severity is flattened
A minor scale threshold and a confirmed security incident each contribute one point; no mandatory or weighted risk logic exists.

### KAI-KEYS-036 — MEDIUM — Automation cannot detect urgency
`main()` returns exit code zero for STAY, PREPARE and MIGRATE, so CI/schedulers cannot enforce the recommendation without parsing prose.

---

## Batch totals

- Findings: **36**
- Critical: **3**
- High: **21**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,303**
- Critical: **103**
- High: **561**
- Medium: **636**
- Low: **3**

## Files materially reviewed in this batch

`scripts/auto_rotate_hmac.py`, `scripts/auto_rotate_ed25519.py`, `scripts/hmac_migration_advisor.py`, with repository-wide consumer searches for their generated state.
