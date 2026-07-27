# Kai Code Audit — HMAC Rotation Drill Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged assurance defects in `scripts/hmac_rotation_drill.py`. Underlying HMAC implementation, rotation service and Ed25519 rotation defects remain in their existing batches and are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HMACDRILL-001 | HIGH | The drill calls itself end-to-end while testing only in-process helper functions |
| KAI-HMACDRILL-002 | HIGH | No Tool Gate, caller service, network, container, secret mount or live rollout is exercised |
| KAI-HMACDRILL-003 | HIGH | Source-visible static secrets are used for every HMAC phase |
| KAI-HMACDRILL-004 | HIGH | Every signature uses one fixed nonce and one fixed historical timestamp |
| KAI-HMACDRILL-005 | HIGH | The drill certifies signatures from timestamp 1700000000 without any expiry or skew check |
| KAI-HMACDRILL-006 | HIGH | Nonce replay and nonce-store persistence are never tested |
| KAI-HMACDRILL-007 | HIGH | Request parameters, conviction and body digest binding are never tested |
| KAI-HMACDRILL-008 | HIGH | Actor-to-token/service identity binding is never tested |
| KAI-HMACDRILL-009 | HIGH | The drill does not test authenticated mode change, co-sign, ledger read or execution requests |
| KAI-HMACDRILL-010 | HIGH | Rotation overlap duration and old-key expiry are not represented |
| KAI-HMACDRILL-011 | HIGH | Partial fleet rollout with old and new service instances is not tested |
| KAI-HMACDRILL-012 | HIGH | Concurrent requests during rotation are not tested |
| KAI-HMACDRILL-013 | HIGH | Crash, interrupted write, rollback and failed activation are not tested |
| KAI-HMACDRILL-014 | HIGH | Secret entropy, length, uniqueness and source are not validated |
| KAI-HMACDRILL-015 | HIGH | Key-file ownership, permissions, symlink safety and atomic replacement are not tested |
| KAI-HMACDRILL-016 | HIGH | Revocation is tested by key ID only and not by compromised key material reused under another ID |
| KAI-HMACDRILL-017 | HIGH | Strict key-ID testing covers only a valid labelled signature and not relabelling attacks |
| KAI-HMACDRILL-018 | HIGH | Retirement does not verify old secret removal from files, environments, processes or caches |
| KAI-HMACDRILL-019 | HIGH | Dual-sign success does not prove every receiver accepts the correct signature during rollout |
| KAI-HMACDRILL-020 | HIGH | The test mutates process-global authentication environment without a `tearDown()` restoration |
| KAI-HMACDRILL-021 | HIGH | `setUp()` deletes any real caller-provided HMAC configuration from the test process |
| KAI-HMACDRILL-022 | HIGH | The drill mutates private module state (`_WARNED_DEFAULT_SECRET`) directly |
| KAI-HMACDRILL-023 | HIGH | The test named `test_load_secret_from_docker_path` does not use a `/run/secrets/` path and does not test Docker-secret loading |
| KAI-HMACDRILL-024 | HIGH | Missing secret files, empty files, permission denial and malformed secret content are not tested |
| KAI-HMACDRILL-025 | HIGH | The test explicitly accepts fallback secret values without distinguishing safe test fallback from production misconfiguration |
| KAI-HMACDRILL-026 | MEDIUM | `sys.path.insert(0, ...)` allows repository-local module shadowing to define the tested code |
| KAI-HMACDRILL-027 | MEDIUM | Exact source commit, module hash and test configuration are not recorded |
| KAI-HMACDRILL-028 | MEDIUM | No structured result or per-phase evidence report is produced |
| KAI-HMACDRILL-029 | MEDIUM | The drill does not record generated signature digests or operation IDs for reproducibility |
| KAI-HMACDRILL-030 | MEDIUM | Key identifiers, secrets and environment state are ordinary mutable process strings |
| KAI-HMACDRILL-031 | MEDIUM | Environment cleanup is incomplete after the final test in the process |
| KAI-HMACDRILL-032 | MEDIUM | Test execution order and shared module cache can influence later tests |
| KAI-HMACDRILL-033 | MEDIUM | No test covers empty, duplicate, whitespace or delimiter-containing key IDs |
| KAI-HMACDRILL-034 | MEDIUM | No test covers malformed signature encodings or extremely long signature inputs |
| KAI-HMACDRILL-035 | MEDIUM | No test covers revoked-ID whitespace, duplicates or Unicode confusables |
| KAI-HMACDRILL-036 | MEDIUM | No test covers strict-key mode disabled during a rotation transition |
| KAI-HMACDRILL-037 | MEDIUM | No test proves secondary-key verification stops after overlap expiry independent of revocation ID |
| KAI-HMACDRILL-038 | MEDIUM | Ed25519 state testing is unrelated to the stated HMAC lifecycle and dilutes the drill result |
| KAI-HMACDRILL-039 | MEDIUM | The Ed25519 test validates JSON state shape rather than cryptographic signing and verification |
| KAI-HMACDRILL-040 | MEDIUM | Time-based Ed25519 key-ID collision is manually patched in the test instead of causing a failure |
| KAI-HMACDRILL-041 | MEDIUM | The collision workaround masks the underlying key-ID uniqueness defect |
| KAI-HMACDRILL-042 | MEDIUM | Temporary Ed25519 state permissions and ownership are not checked |
| KAI-HMACDRILL-043 | MEDIUM | Tampered, truncated and concurrently written rotation-state files are not tested |
| KAI-HMACDRILL-044 | MEDIUM | Private-key persistence, encryption and secure deletion are not checked |
| KAI-HMACDRILL-045 | MEDIUM | No multi-process or distributed state consistency test exists |
| KAI-HMACDRILL-046 | MEDIUM | No service-restart test proves new key material is reloaded consistently |
| KAI-HMACDRILL-047 | MEDIUM | No audit-ledger event is inspected for correct rotation identity and phase transition |
| KAI-HMACDRILL-048 | MEDIUM | A passing unittest exit code is the only assurance output and cannot prove deployment completion |

---

## Findings

### KAI-HMACDRILL-001 — HIGH — False end-to-end claim
All HMAC phases invoke `common.auth` directly in one Python process.

### KAI-HMACDRILL-002 — HIGH — Deployment boundary absent
No HTTP request or running service is involved.

### KAI-HMACDRILL-003 — HIGH — Public fixture secrets
`old-secret-alpha`, `new-secret-beta` and `new-secret-gamma` are hard-coded.

### KAI-HMACDRILL-004 — HIGH — Reused nonce/time
Every call uses `drill-nonce-001` and `1700000000.0`.

### KAI-HMACDRILL-005 — HIGH — Ancient signature certified
The helper verifies the fixed historical signature because freshness is not part of this layer/test.

### KAI-HMACDRILL-006 — HIGH — Replay protection absent
Nonce reuse, persistence and restart are not tested.

### KAI-HMACDRILL-007 — HIGH — Incomplete signed subject
Only actor/session/tool/nonce/time helper fields are exercised.

### KAI-HMACDRILL-008 — HIGH — Identity binding absent
No trusted token, mTLS identity or actor mapping is included.

### KAI-HMACDRILL-009 — HIGH — Privileged endpoint lifecycle absent
Real administrative/execution use is untested.

### KAI-HMACDRILL-010 — HIGH — Overlap timing absent
Phase changes are immediate environment replacements.

### KAI-HMACDRILL-011 — HIGH — Mixed-version fleet absent
One process holds both old and new verification state.

### KAI-HMACDRILL-012 — HIGH — Concurrency absent
No request races occur.

### KAI-HMACDRILL-013 — HIGH — Failure recovery absent
No interrupted activation or rollback.

### KAI-HMACDRILL-014 — HIGH — Secret quality absent
Any short string is accepted.

### KAI-HMACDRILL-015 — HIGH — Secret-file controls absent
Only environment strings are used for HMAC phases.

### KAI-HMACDRILL-016 — HIGH — Revocation alias bypass untested
The same compromised bytes under a new key ID are not exercised.

### KAI-HMACDRILL-017 — HIGH — Strict relabelling untested
Only a correctly labelled old signature is verified.

### KAI-HMACDRILL-018 — HIGH — Secret retirement unproven
Environment replacement does not prove removal elsewhere.

### KAI-HMACDRILL-019 — HIGH — Receiver compatibility unproven
Both signatures are checked by one helper instance.

### KAI-HMACDRILL-020 — HIGH — No environment teardown
The class defines `setUp()` only.

### KAI-HMACDRILL-021 — HIGH — Caller configuration destroyed
Each test removes six HMAC environment variables.

### KAI-HMACDRILL-022 — HIGH — Private-state manipulation
The drill resets an internal warning flag.

### KAI-HMACDRILL-023 — HIGH — Misnamed Docker-secret test
A `/tmp` filename is placed in an env variable; `load_secret` deliberately returns the filename as raw text.

### KAI-HMACDRILL-024 — HIGH — Secret-source failures absent
No real Docker mount conditions are exercised.

### KAI-HMACDRILL-025 — HIGH — Fallback acceptance
The default-value test positively certifies returning a fallback.

### KAI-HMACDRILL-026 — MEDIUM — Import-path injection
Repository root is prepended to `sys.path`.

### KAI-HMACDRILL-027 — MEDIUM — Source identity absent
No SHA/digest.

### KAI-HMACDRILL-028 — MEDIUM — No machine-readable report
Unittest text only.

### KAI-HMACDRILL-029 — MEDIUM — Evidence detail absent
No retained phase artefact.

### KAI-HMACDRILL-030 — MEDIUM — Secret lifecycle in memory ungoverned
Strings remain in process environment/objects.

### KAI-HMACDRILL-031 — MEDIUM — Final environment contamination
Last test-specific variables may remain.

### KAI-HMACDRILL-032 — MEDIUM — Shared module-cache effects
Auth module/global state persists.

### KAI-HMACDRILL-033 — MEDIUM — Key-ID edge cases absent
Grammar/normalisation untested.

### KAI-HMACDRILL-034 — MEDIUM — Signature parser fuzz absent
Malformed/large inputs omitted.

### KAI-HMACDRILL-035 — MEDIUM — Revocation-list parser edge cases absent
Whitespace/duplicates/confusables omitted.

### KAI-HMACDRILL-036 — MEDIUM — Non-strict transition omitted
Default key-ID behaviour is not exercised in overlap.

### KAI-HMACDRILL-037 — MEDIUM — Independent expiry absent
Only secret removal/revoked IDs end acceptance.

### KAI-HMACDRILL-038 — MEDIUM — Scope dilution
Ed25519 state is mixed into an HMAC drill.

### KAI-HMACDRILL-039 — MEDIUM — Ed25519 cryptography untested
Only state fields are asserted.

### KAI-HMACDRILL-040 — MEDIUM — Collision manually hidden
The test edits a colliding key ID.

### KAI-HMACDRILL-041 — MEDIUM — Broken uniqueness still passes
The workaround converts failure into success.

### KAI-HMACDRILL-042 — MEDIUM — Temporary-state permissions absent
No file hardening check.

### KAI-HMACDRILL-043 — MEDIUM — State corruption/concurrency absent
Only clean save/load.

### KAI-HMACDRILL-044 — MEDIUM — Private-key protection absent
Storage contents/security are not validated.

### KAI-HMACDRILL-045 — MEDIUM — Distributed rotation absent
One process/file only.

### KAI-HMACDRILL-046 — MEDIUM — Reload absent
No service restart.

### KAI-HMACDRILL-047 — MEDIUM — Audit evidence absent
No ledger/rotation event is inspected.

### KAI-HMACDRILL-048 — MEDIUM — Exit code overstates completion
Passing local tests do not prove deployment rotation.

---

## Batch totals

- Findings: **48**
- Critical: **0**
- High: **25**
- Medium: **23**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,809**
- Critical: **182**
- High: **1,427**
- Medium: **1,197**
- Low: **3**

## Files materially reviewed

`scripts/hmac_rotation_drill.py`, with current `common/auth.py` and Ed25519 rotation interfaces used to evaluate what the drill actually covers.
