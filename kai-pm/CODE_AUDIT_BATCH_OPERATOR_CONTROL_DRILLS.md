# Kai Code Audit — Operator Control, Recovery, Self-Audit and Chaos Tooling Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers local operator-control, recovery-drill, self-audit, game-day and chaos scripts. Key-rotation-specific defects remain in the existing key-rotation batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-OPS-001 | CRITICAL | Destructive control authorisation proves only matching public-key files, not possession of either private key |
| KAI-OPS-002 | CRITICAL | TPM verification proves only that a handle exists, not that it matches the sealed primary key |
| KAI-OPS-003 | CRITICAL | Conviction/threat overrides can be added without the two-key destructive-action check |
| KAI-OPS-004 | CRITICAL | Emergency action logs use fixed text such as `TPM` as the alleged signature rather than a cryptographic signature |
| KAI-OPS-005 | CRITICAL | Self-audit treats command timeouts and execution exceptions as successful checks |
| KAI-OPS-006 | CRITICAL | The real recovery drill generates and validates a fresh unrelated recovery secret rather than recovering the deployed primary key |
| KAI-OPS-007 | CRITICAL | Chaos scorecard validates default production ports and fail-open gates rather than the chaos processes on ports 19000/19001/19007 |
| KAI-OPS-008 | CRITICAL | Chaos services inherit the caller environment and can connect to real persistent dependencies while running destructive failure scenarios |
| KAI-OPS-009 | HIGH | Any directory under user media paths is treated as a removable trusted USB device |
| KAI-OPS-010 | HIGH | USB discovery does not validate mount ownership, filesystem type, device identity, read-only status or removable-media provenance |
| KAI-OPS-011 | HIGH | Key discovery checks file existence only and does not verify that private and public key material correspond |
| KAI-OPS-012 | HIGH | Primary and backup USBs contain the same copied private key rather than independent recovery shares |
| KAI-OPS-013 | HIGH | Test mode makes TPM verification unconditional and switches Vault to a plaintext local file |
| KAI-OPS-014 | HIGH | Test mode is enabled by an ordinary environment variable with no production lockout |
| KAI-OPS-015 | HIGH | Vault root tokens are loaded from process environment and sent to a plaintext HTTP endpoint by default |
| KAI-OPS-016 | HIGH | Vault server identity and TLS certificate are not required or validated |
| KAI-OPS-017 | HIGH | Local Vault fallback is an unsigned, non-atomic plaintext JSON file |
| KAI-OPS-018 | HIGH | Corrupt local Vault fallback is silently treated as empty state |
| KAI-OPS-019 | HIGH | Recovery words are written in plaintext under the user home directory with default filesystem permissions |
| KAI-OPS-020 | HIGH | The UI displays the complete recovery phrase and returns the encrypted payload and words together |
| KAI-OPS-021 | HIGH | Paper recovery restore does not validate the Vault-stored recovery hint before overwriting key files |
| KAI-OPS-022 | HIGH | Restore writes private/public key files to a caller-selected mount without ownership, symlink or existing-file checks |
| KAI-OPS-023 | HIGH | The paper phrase derives from only 24 modulo-reduced bytes and a 48-word list, discarding substantial seed entropy |
| KAI-OPS-024 | HIGH | QR generation failure/absence still returns a QR path that may not exist |
| KAI-OPS-025 | HIGH | `kill_executor()` ignores Docker Compose command failure and logs the action as successful |
| KAI-OPS-026 | HIGH | Executor stop targets a hard-coded Compose file and service without verifying the deployed project or postcondition |
| KAI-OPS-027 | HIGH | memU rollback chooses the first returned commit without validating chronology, ownership or safety |
| KAI-OPS-028 | HIGH | memU rollback uses an unauthenticated HTTP mutation and validates no restored postcondition |
| KAI-OPS-029 | HIGH | Emergency logs are unsigned, append-only plaintext files with no ownership or rotation control |
| KAI-OPS-030 | HIGH | Conviction override text is unbounded and newline content can create multiple bypass rules or forge log structure |
| KAI-OPS-031 | HIGH | Conviction override files are non-atomic and have no authenticated operator revision |
| KAI-OPS-032 | HIGH | Unlock Logs exposes Executor output after only a single recognised public-key file and no private-key proof |
| KAI-OPS-033 | HIGH | Full Executor logs may contain commands, parameters, secrets and internal diagnostics |
| KAI-OPS-034 | HIGH | The scheduled real drill rotates the local paper-recovery material every run and overwrites prior recovery files |
| KAI-OPS-035 | HIGH | Drill success depends only on public-key hash presence and a freshly generated local restore simulation |
| KAI-OPS-036 | HIGH | Drill audit is optional and drill success is unaffected if Redis audit logging fails |
| KAI-OPS-037 | HIGH | Drill failure alert targets a stale/default service name and port and suppresses delivery failure |
| KAI-OPS-038 | HIGH | Self-audit logs results to memU without checking HTTP status or response schema |
| KAI-OPS-039 | HIGH | Self-audit uses a memory payload that does not preserve authenticated actor, command digest or exact evidence |
| KAI-OPS-040 | HIGH | Self-audit can persist false “All checks passed” lessons into memU |
| KAI-OPS-041 | HIGH | Self-audit overwrites one unsigned JSON report and destroys prior audit history |
| KAI-OPS-042 | HIGH | Self-audit reruns the extremely broad merge/test gates recursively and without workload isolation |
| KAI-OPS-043 | HIGH | Chaos services are launched directly from source outside the deployed container/security profile |
| KAI-OPS-044 | HIGH | Chaos Tool Gate is started in WORK mode with inherited shared-secret and trusted-token state |
| KAI-OPS-045 | HIGH | Chaos waits a fixed two seconds and never verifies that services became ready before killing them |
| KAI-OPS-046 | HIGH | Chaos restart declares success after process creation without a health or state-recovery check |
| KAI-OPS-047 | HIGH | Chaos process control does not use process groups and can leave descendants alive |
| KAI-OPS-048 | HIGH | Game-day scorecard includes fail-open go/no-go, fake-embedding hardening and dummy-cryptography drill checks as equal success evidence |
| KAI-OPS-049 | MEDIUM | USB timeout configuration is not validated and uses wall-clock timing |
| KAI-OPS-050 | MEDIUM | USB selection uses the first directory returned by filesystem iteration and is nondeterministic |
| KAI-OPS-051 | MEDIUM | Key files and recovery files are written directly without atomic replacement or fsync |
| KAI-OPS-052 | MEDIUM | Cryptography and QR import catches `BaseException`, hiding serious native/runtime failures |
| KAI-OPS-053 | MEDIUM | Vault read converts every recovered value to text and loses original type/encoding metadata |
| KAI-OPS-054 | MEDIUM | Emergency-action timestamps use wall-clock strings without an event sequence or target revision |
| KAI-OPS-055 | MEDIUM | Advisor Mode mixes English, Russian and private financial totals into one unaudited local dialog |
| KAI-OPS-056 | MEDIUM | UI refresh polls USB and Vault state every two seconds synchronously on the GUI thread |
| KAI-OPS-057 | MEDIUM | Direct helper functions can be imported and invoked without the GUI’s intended control flow |
| KAI-OPS-058 | MEDIUM | Self-audit runs relative to the caller working directory rather than an immutable repository root |
| KAI-OPS-059 | MEDIUM | Self-audit retains only the last 1,000 stdout/stderr characters and can hide the actual failure cause |
| KAI-OPS-060 | MEDIUM | Self-audit report and memU lesson have no tested commit, environment or dependency revision |
| KAI-OPS-061 | MEDIUM | Chaos randomness is fixed to seed 42 and repeatedly kills the same deterministic target sequence |
| KAI-OPS-062 | MEDIUM | Unknown chaos scenarios only log a warning rather than failing |
| KAI-OPS-063 | MEDIUM | Chaos stdout/stderr are discarded by default, removing failure evidence |
| KAI-OPS-064 | MEDIUM | Game-day checks have no individual timeout and can hang indefinitely |
| KAI-OPS-065 | MEDIUM | Game-day pass and duration thresholds are unvalidated environment floats and can be disabled |
| KAI-OPS-066 | MEDIUM | Game-day results are an unsigned non-atomic single-file report with only output tails |

---

## Critical findings

### KAI-OPS-001 — CRITICAL — Destructive authorisation lacks private-key proof
**Issue:** `key_presence()` hashes only `kai-primary.pub` files and compares them with Vault values. `can_destructive_actions()` requires both matching public files plus a TPM handle check; neither private file is opened or used in a challenge-response.  
**Risk:** Copies of two public files—or two mounts containing the same public file—unlock Kill/Rollback without possession of either private key.  
**Recommendation:** require cryptographic challenge signatures from two independently controlled private-key factors and bind them to the exact requested action.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-002 — CRITICAL — TPM handle is not identity-bound
**Issue:** `tpm2_readpublic -c 0x81000001` is accepted solely by return code. The returned public area is not hashed and compared with the expected sealed key.  
**Risk:** Any object at that persistent handle satisfies the TPM factor.  
**Recommendation:** verify object name/public digest, policy and key usage against an approved immutable TPM identity.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-003 — CRITICAL — Threat bypass lacks destructive authorisation
**Issue:** The “Ignore threat override” button is shown when the primary exists but backup is absent, and `add_conviction_override()` itself performs no key/TPM check.  
**Risk:** A local user with partial or no intended destructive-factor proof can create permanent phrases that bypass low-conviction safeguards.  
**Recommendation:** treat overrides as high-risk policy changes requiring authenticated two-factor approval, scope, expiry and review.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-004 — CRITICAL — Audit signature is fabricated text
**Issue:** `log_action()` stores its `signature` argument verbatim. Callers pass constants such as `TPM` and `override by Dainius`; no signature is created or verified.  
**Risk:** The log falsely claims cryptographic/operator evidence and can be forged or rewritten locally.  
**Recommendation:** sign a canonical action record with the actual factors used and anchor it in an immutable audit authority.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-005 — CRITICAL — Self-audit exceptions are passes
**Issue:** `run_make()` returns `{target,error}` on timeout/exception. `summarize_results()` uses `r.get("returncode",0)`, so missing return codes default to success.  
**Risk:** Hung, missing or crashed checks produce “All checks passed” and may be stored as a lesson in memU.  
**Recommendation:** represent execution errors as mandatory failures and preserve typed timeout/exception state.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-006 — CRITICAL — Drill does not test deployed recovery
**Issue:** The non-test drill checks public-key hashes, then calls `generate_paper_recovery()` to create a new random seed and immediately restores that new seed into a temporary directory.  
**Risk:** It passes even if the deployed TPM/private key cannot be restored from the existing emergency material.  
**Recommendation:** test a non-destructive challenge derived from the actual sealed recovery package and expected deployed public identity.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-007 — CRITICAL — Chaos scorecard targets another stack
**Issue:** Chaos starts Tool Gate/memU/Agentic on 19000/19001/19007, then runs `make game-day-scorecard`. Its go/no-go and many tests use default ports such as 8000, 8001, 8007 and Dashboard 8080.  
**Risk:** The scorecard can pass by testing an unrelated already-running stack while the chaos processes fail completely.  
**Recommendation:** inject an immutable chaos environment manifest into every check and reject any endpoint outside it.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-008 — CRITICAL — Chaos can mutate real dependencies
**Issue:** Each child inherits `os.environ`; only a few ports/URLs are overridden. Database, Redis, files, secrets, models and other service URLs may still point at real developer/production resources.  
**Risk:** Kill/restart/failure tests can write or corrupt persistent live state.  
**Recommendation:** run chaos in an isolated disposable network, filesystem and credentials namespace with explicit deny-by-default environment.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-OPS-009 — HIGH — Directory equals USB
Every directory under `/media/$USER` and `/run/media/$USER` is considered a candidate key device.

### KAI-OPS-010 — HIGH — Media provenance absent
No block-device, mount, owner, filesystem, removable flag or serial identity is checked.

### KAI-OPS-011 — HIGH — Keypair validity absent
Discovery requires both filenames but never verifies the private material corresponds to the public blob.

### KAI-OPS-012 — HIGH — Backup is a cloned secret
`seal_backup()` copies the same private file, creating two copies of one factor rather than independent shares/keys.

### KAI-OPS-013 — HIGH — Test mode bypasses hardware/Vault
It returns TPM success and writes secret state locally.

### KAI-OPS-014 — HIGH — Test mode is runtime mutable
One environment variable activates the bypass; no build/profile or production refusal exists.

### KAI-OPS-015 — HIGH — Root token and plaintext Vault
Root/client token comes from environment and the default address is HTTP localhost.

### KAI-OPS-016 — HIGH — No Vault identity
There is no CA pinning, TLS requirement, namespace/mount verification or response attestation.

### KAI-OPS-017 — HIGH — Unsafe local secret state
The fallback file is complete JSON rewritten directly with default permissions.

### KAI-OPS-018 — HIGH — Corruption resets authority
Any JSON error yields an empty mapping without quarantine.

### KAI-OPS-019 — HIGH — Plaintext recovery phrase
`recovery_words.txt` contains the complete phrase needed to derive the encryption key.

### KAI-OPS-020 — HIGH — Factors presented together
The function returns both encrypted payload and words; the GUI displays words and points to the QR on the same machine.

### KAI-OPS-021 — HIGH — Recovery hint unused
Restore never compares the supplied payload hash with `keeper_recovery_hint`.

### KAI-OPS-022 — HIGH — Unsafe key overwrite
Mount/path/file ownership, symlinks, existing content and free space are not validated before writes.

### KAI-OPS-023 — HIGH — Recovery phrase entropy reduction
Only the first 24 seed bytes are mapped modulo 48; the phrase cannot represent the full random seed entropy and modulo mapping is biased.

### KAI-OPS-024 — HIGH — False QR result
When qrcode is unavailable, the response still returns the expected path.

### KAI-OPS-025 — HIGH — False Executor-stop success
Docker command uses `check=False`; return code/output are ignored and success is logged/shown.

### KAI-OPS-026 — HIGH — Wrong deployment targeting
Compose project name, working directory, container identity and stopped-state postcondition are not verified.

### KAI-OPS-027 — HIGH — Unsafe rollback selection
The first commit entry is assumed to be the correct rollback target; ordering/type/integrity are not checked.

### KAI-OPS-028 — HIGH — Unauthorised rollback request
The HTTP mutation carries no authentication and validates no restored record/index/graph state.

### KAI-OPS-029 — HIGH — Weak emergency log
It has no signature, hash chain, lock, fsync, rotation or protected permissions.

### KAI-OPS-030 — HIGH — Override rule injection
Embedded newlines create multiple lines/rules and can alter audit/display structure.

### KAI-OPS-031 — HIGH — Override persistence race
Concurrent readers/appends can lose deduplication or create inconsistent rule sets.

### KAI-OPS-032 — HIGH — Logs require only weak factor state
The UI exposes Unlock Logs for one recognised key; private possession and TPM identity are not proven.

### KAI-OPS-033 — HIGH — Sensitive Executor output
Raw Docker logs can expose commands, credentials, paths, errors and private outputs.

### KAI-OPS-034 — HIGH — Scheduled drill overwrites recovery state
Every real drill creates new words/QR/hint and overwrites local files, potentially invalidating previously stored emergency material.

### KAI-OPS-035 — HIGH — Weak drill checks
Private-key correctness, TPM identity, Vault recovery and real restored signing are never tested.

### KAI-OPS-036 — HIGH — Audit is non-enforcing
`required=False`; logging failure is ignored.

### KAI-OPS-037 — HIGH — Failure alert is stale and fail-open
Default `perception-telegram:9000` does not match the audited Telegram deployment, and curl failure is suppressed.

### KAI-OPS-038 — HIGH — memU status ignored
`requests.post()` response is never checked.

### KAI-OPS-039 — HIGH — Weak self-audit evidence
The memory record has no command output digests, tested commit, actor signature or result schema.

### KAI-OPS-040 — HIGH — False lesson poisoning
Execution errors become “All checks passed” and are memorised as system lessons.

### KAI-OPS-041 — HIGH — Audit history overwritten
One `output/self_audit_log.json` replaces the prior run non-atomically.

### KAI-OPS-042 — HIGH — Recursive workload amplification
Self-audit runs `merge-gate`, which itself runs `test-core`, then runs `test-core` again and health sweep, each with a coarse timeout.

### KAI-OPS-043 — HIGH — Chaos bypasses deployment hardening
Raw Python processes do not use non-root Docker users, no-new-privileges, resource limits, volumes or network segmentation.

### KAI-OPS-044 — HIGH — Chaos execution-capable mode
The Tool Gate child is explicitly started in WORK mode and inherits secrets/token state.

### KAI-OPS-045 — HIGH — Readiness unverified before fault
A fixed two-second sleep precedes kills; startup/model/database completion is not tested.

### KAI-OPS-046 — HIGH — Recovery unverified
`restart()` records success immediately after `Popen` and checks no health or prior state.

### KAI-OPS-047 — HIGH — Descendant process leakage
Terminate/kill targets only the direct Popen PID and uses no process-group/session containment.

### KAI-OPS-048 — HIGH — Invalid scorecard evidence mix
Fail-open/static/stub/dummy tests each receive one equal pass vote and can produce 100% without real-system recovery.

---

## Medium-severity findings

### KAI-OPS-049 — MEDIUM — USB timing config
Negative/extreme values are silently clamped/extended using adjustable wall clock.

### KAI-OPS-050 — MEDIUM — Nondeterministic mount selection
Filesystem iteration order determines primary/backup/device actions.

### KAI-OPS-051 — MEDIUM — Non-atomic key/recovery writes
Direct writes can leave partial files and lack durable completion.

### KAI-OPS-052 — MEDIUM — BaseException suppression
Native panic, KeyboardInterrupt/SystemExit-like conditions can be converted into an unavailable optional feature.

### KAI-OPS-053 — MEDIUM — Vault type loss
Every value is coerced to string.

### KAI-OPS-054 — MEDIUM — Weak action chronology
Logs lack operation ID, target revision and trusted timestamp.

### KAI-OPS-055 — MEDIUM — Advisor data handling
Private finance totals and free text appear in a local dialog without access/audit/data-minimisation controls.

### KAI-OPS-056 — MEDIUM — Synchronous GUI polling
Vault/file/TPM checks can block the UI every two seconds.

### KAI-OPS-057 — MEDIUM — UI is not an enforcement boundary
Imported helpers can be invoked directly by any local Python caller.

### KAI-OPS-058 — MEDIUM — Working-directory dependence
`make` and output paths use the caller’s current directory.

### KAI-OPS-059 — MEDIUM — Failure evidence truncated
Only output tails survive.

### KAI-OPS-060 — MEDIUM — Missing run identity
Self-audit lacks source/environment/dependency revisions.

### KAI-OPS-061 — MEDIUM — Deterministic “random” chaos
Every run uses the same seed/selection sequence.

### KAI-OPS-062 — MEDIUM — Unknown scenario is non-fatal
The function warns and returns.

### KAI-OPS-063 — MEDIUM — Default evidence discard
Child stdout/stderr are sent to `/dev/null` unless verbose.

### KAI-OPS-064 — MEDIUM — Scorecard checks can hang
`subprocess.run` has no timeout.

### KAI-OPS-065 — MEDIUM — SLO policy can be disabled
Negative/zero/NaN/extreme environment values are not validated.

### KAI-OPS-066 — MEDIUM — Weak game-day report
The same file is directly overwritten, carries only tails and has no signature/revision.

---

## Batch totals

- Findings: **66**
- Critical: **8**
- High: **40**
- Medium: **18**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,780**
- Critical: **214**
- High: **1,414**
- Medium: **1,149**
- Low: **3**

## Files materially reviewed

`scripts/kai_control.py`, `scripts/kai-drill.sh`, `scripts/self_audit.py`, `scripts/chaos_ci.py`, `scripts/gameday_scorecard.py` and Makefile integration.
