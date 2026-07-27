# Kai Code Audit — Shell Contract, Health and Rotation Wrappers Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged wrapper/assurance findings in `scripts/contract_smoke.sh`, `scripts/health_sweep.sh`, `scripts/weekly_key_rotation.sh`, `scripts/monthly_paper_backup.sh` and `scripts/weekly_ed25519_rotation.sh`. Underlying key-rotation, backup and service implementation findings remain in their existing batches and are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SHELLWRAP-001 | HIGH | Contract smoke uses unauthenticated plaintext service calls |
| KAI-SHELLWRAP-002 | HIGH | The memU `/route` request omits the required `timestamp` field and cannot satisfy the current schema |
| KAI-SHELLWRAP-003 | HIGH | A default `SESSION_ID=bootstrap-token-1` is declared but never used for authentication or any request |
| KAI-SHELLWRAP-004 | HIGH | Contract validation checks only that JSON keys exist |
| KAI-SHELLWRAP-005 | HIGH | `ledger/verify` passes the smoke test when `valid` is false |
| KAI-SHELLWRAP-006 | HIGH | Dashboard root passes when `core_ready` is false |
| KAI-SHELLWRAP-007 | HIGH | Dashboard readiness passes when status is NO_GO or otherwise unhealthy |
| KAI-SHELLWRAP-008 | HIGH | Service identity, version and deployment digest are never verified |
| KAI-SHELLWRAP-009 | HIGH | Curl calls have no connection or total timeout |
| KAI-SHELLWRAP-010 | HIGH | Complete responses are materialised in shell variables without byte limits |
| KAI-SHELLWRAP-011 | HIGH | The memU route response, including memory vectors and metadata, is copied into process arguments |
| KAI-SHELLWRAP-012 | HIGH | Environment-controlled service URLs allow the smoke runner to contact arbitrary destinations |
| KAI-SHELLWRAP-013 | HIGH | Public/read-only endpoint reachability is substituted for protected control-path testing |
| KAI-SHELLWRAP-014 | MEDIUM | Response content type and JSON schema types are not validated |
| KAI-SHELLWRAP-015 | MEDIUM | Null, empty and malformed values satisfy the key-presence check |
| KAI-SHELLWRAP-016 | MEDIUM | Checks run sequentially without a total test deadline |
| KAI-SHELLWRAP-017 | MEDIUM | Raw JSON parsing/errors can expose internal response content in terminal diagnostics |
| KAI-SHELLWRAP-018 | MEDIUM | No authenticated execution, memory-write, denial or replay contract is exercised |
| KAI-SHELLWRAP-019 | MEDIUM | Smoke execution creates no structured report, source SHA or endpoint revision record |
| KAI-SHELLWRAP-020 | MEDIUM | The check does not verify that all expected services are present |
| KAI-SHELLWRAP-021 | MEDIUM | No cleanup or before/after state validation accompanies the requests |
| KAI-SHELLWRAP-022 | MEDIUM | Successful output overstates a full contract from four shallow endpoint checks |
| KAI-SHELLWRAP-023 | HIGH | Shell health sweep classifies every HTTP 2xx response as healthy |
| KAI-SHELLWRAP-024 | HIGH | Dashboard readiness HTTP 200 is logged OK even when the body says not ready |
| KAI-SHELLWRAP-025 | HIGH | Health response bodies are discarded and never semantically parsed |
| KAI-SHELLWRAP-026 | HIGH | Health calls carry no service authentication or expected service identity |
| KAI-SHELLWRAP-027 | HIGH | Health URLs are arbitrary environment values |
| KAI-SHELLWRAP-028 | HIGH | Curl calls have no timeout and one endpoint can hang the full sweep |
| KAI-SHELLWRAP-029 | HIGH | Only Tool Gate, memU, Executor and Dashboard are checked |
| KAI-SHELLWRAP-030 | HIGH | Health and readiness checks are sequential and can consume cumulative latency |
| KAI-SHELLWRAP-031 | HIGH | The log destination is an arbitrary environment-controlled path |
| KAI-SHELLWRAP-032 | HIGH | A privileged run can follow a malicious symlink at the log path |
| KAI-SHELLWRAP-033 | HIGH | Health logs grow without rotation or retention |
| KAI-SHELLWRAP-034 | HIGH | URLs containing credentials or sensitive query strings are persisted verbatim |
| KAI-SHELLWRAP-035 | MEDIUM | Log-directory permissions and ownership are not explicitly hardened |
| KAI-SHELLWRAP-036 | MEDIUM | Concurrent sweeps can interleave records in the same log |
| KAI-SHELLWRAP-037 | MEDIUM | Host wall-clock timestamps have no trusted source or monotonic sequence |
| KAI-SHELLWRAP-038 | MEDIUM | No response digest or evidence payload is retained |
| KAI-SHELLWRAP-039 | MEDIUM | No retry, jitter or transient/permanent failure distinction exists |
| KAI-SHELLWRAP-040 | MEDIUM | No alert or incident record is generated on failure |
| KAI-SHELLWRAP-041 | MEDIUM | Dashboard health and readiness duplicate expensive internal fan-out work |
| KAI-SHELLWRAP-042 | MEDIUM | The script does not verify TLS policy for non-local configured URLs |
| KAI-SHELLWRAP-043 | MEDIUM | The service inventory and expected statuses are not versioned |
| KAI-SHELLWRAP-044 | MEDIUM | The result is a single process exit code with no per-service machine-readable report |
| KAI-SHELLWRAP-045 | HIGH | Rotation/backup wrappers depend on the caller’s current working directory |
| KAI-SHELLWRAP-046 | HIGH | Wrapper commands use PATH-resolved `python` without verifying interpreter identity |
| KAI-SHELLWRAP-047 | HIGH | A malicious PATH can substitute `python`, `tee` or other invoked commands |
| KAI-SHELLWRAP-048 | HIGH | Child scripts inherit the complete scheduler/service environment |
| KAI-SHELLWRAP-049 | HIGH | Log directories are arbitrary environment-controlled paths |
| KAI-SHELLWRAP-050 | HIGH | Privileged wrappers can follow symlinks for their log files |
| KAI-SHELLWRAP-051 | HIGH | Rotation and backup output is persisted to plaintext logs without field redaction |
| KAI-SHELLWRAP-052 | HIGH | Only stdout is captured by `tee`; stderr diagnostics and partial failures are absent from the audit log |
| KAI-SHELLWRAP-053 | HIGH | Log files have no rotation, retention or maximum size |
| KAI-SHELLWRAP-054 | HIGH | Concurrent scheduled/manual runs have no mutual-exclusion lock |
| KAI-SHELLWRAP-055 | HIGH | “Weekly” and “monthly” names do not enforce schedule or minimum elapsed time at the wrapper boundary |
| KAI-SHELLWRAP-056 | HIGH | Wrapper execution has no overall timeout |
| KAI-SHELLWRAP-057 | HIGH | No post-run check verifies that keys were activated, old keys revoked or backups are restorable |
| KAI-SHELLWRAP-058 | HIGH | Wrapper success is not tied to a signed operation ID or immutable source/configuration revision |
| KAI-SHELLWRAP-059 | MEDIUM | Rotation interval environment values are forwarded without wrapper-level numeric validation |
| KAI-SHELLWRAP-060 | MEDIUM | HMAC, Ed25519 and paper-backup wrappers use inconsistent `PYTHONPATH` handling |
| KAI-SHELLWRAP-061 | MEDIUM | Log-directory creation uses the caller’s umask |
| KAI-SHELLWRAP-062 | MEDIUM | Log entries have no wrapper-generated timestamp or host identity |
| KAI-SHELLWRAP-063 | MEDIUM | A failed command can leave partial stdout appended as an apparently normal operation log |
| KAI-SHELLWRAP-064 | MEDIUM | No notification or escalation is issued after wrapper failure |
| KAI-SHELLWRAP-065 | MEDIUM | No lock protects underlying key/backup files from overlapping jobs |
| KAI-SHELLWRAP-066 | MEDIUM | The wrapper does not verify that it is running from the intended repository revision |
| KAI-SHELLWRAP-067 | MEDIUM | Monthly backup can be invoked repeatedly and create uncontrolled duplicate artefacts |
| KAI-SHELLWRAP-068 | MEDIUM | Weekly rotation can be invoked repeatedly and accelerate key churn |
| KAI-SHELLWRAP-069 | MEDIUM | Wrapper logs are not integrity-protected or externally anchored |
| KAI-SHELLWRAP-070 | MEDIUM | No structured summary links scheduler run, child result, generated artefact and validation outcome |

---

## Contract smoke — `scripts/contract_smoke.sh`

### KAI-SHELLWRAP-001 — HIGH — Open transport
All service calls are plain unauthenticated HTTP.

### KAI-SHELLWRAP-002 — HIGH — Stale memU contract
Current `MemoryRequest` requires query, session ID and timestamp; the smoke body supplies only the first two.

### KAI-SHELLWRAP-003 — HIGH — Dead credential theatre
The bootstrap-token variable is neither sent nor checked.

### KAI-SHELLWRAP-004 — HIGH — Key-only assertion
No values, types, invariants or status semantics are evaluated.

### KAI-SHELLWRAP-005 — HIGH — Invalid ledger passes
Presence of `valid` is sufficient.

### KAI-SHELLWRAP-006 — HIGH — Unready core passes
Presence of `core_ready` is sufficient.

### KAI-SHELLWRAP-007 — HIGH — NO_GO passes
Readiness status content is ignored.

### KAI-SHELLWRAP-008 — HIGH — Identity absent
Any endpoint returning matching JSON keys can impersonate the service.

### KAI-SHELLWRAP-009 — HIGH — Unbounded wait
Curl has no timeout options.

### KAI-SHELLWRAP-010 — HIGH — Shell response buffering
Full JSON is retained in variables.

### KAI-SHELLWRAP-011 — HIGH — Private data in argv
The route response is passed as `sys.argv[1]` to Python and can be visible through process inspection.

### KAI-SHELLWRAP-012 — HIGH — Arbitrary configured destinations
URLs are not constrained to loopback/private expected services.

### KAI-SHELLWRAP-013 — HIGH — Wrong contract layer
No authenticated Gate/Executor or memory write/read boundary is tested.

### KAI-SHELLWRAP-014 — MEDIUM — Media/schema validation absent
Only JSON decoding and dictionary membership occur.

### KAI-SHELLWRAP-015 — MEDIUM — Invalid values accepted
Keys with null/false/wrong type pass.

### KAI-SHELLWRAP-016 — MEDIUM — Sequential total latency
No parallelism/deadline.

### KAI-SHELLWRAP-017 — MEDIUM — Diagnostic exposure
Parser/curl errors can reveal returned data.

### KAI-SHELLWRAP-018 — MEDIUM — Security cases omitted
No deny, replay, auth, body-binding or mutation test exists.

### KAI-SHELLWRAP-019 — MEDIUM — No evidence report
Only terminal text/exit.

### KAI-SHELLWRAP-020 — MEDIUM — Partial inventory
Four endpoint families only.

### KAI-SHELLWRAP-021 — MEDIUM — State neutrality unproven
No before/after check.

### KAI-SHELLWRAP-022 — MEDIUM — Overstated pass
“contract smoke passed” lacks scope qualification.

---

## Shell health sweep — `scripts/health_sweep.sh`

### KAI-SHELLWRAP-023 — HIGH — HTTP status as health
`curl -f` accepts any 2xx body.

### KAI-SHELLWRAP-024 — HIGH — Readiness false positive
Dashboard returns HTTP 200 for advisory readiness/no-go data.

### KAI-SHELLWRAP-025 — HIGH — Semantic evidence discarded
Bodies go to `/dev/null`.

### KAI-SHELLWRAP-026 — HIGH — Service authentication absent
No token/signature/mTLS.

### KAI-SHELLWRAP-027 — HIGH — Arbitrary endpoint configuration
Environment URLs are trusted.

### KAI-SHELLWRAP-028 — HIGH — No deadline
Curl may hang indefinitely.

### KAI-SHELLWRAP-029 — HIGH — Incomplete fleet
Most deployed services are omitted.

### KAI-SHELLWRAP-030 — HIGH — Serial checks
Total sweep latency is cumulative.

### KAI-SHELLWRAP-031 — HIGH — Arbitrary log path
`LOG_DIR` is unrestricted.

### KAI-SHELLWRAP-032 — HIGH — Symlink write risk
Tee follows the resolved file target.

### KAI-SHELLWRAP-033 — HIGH — Unlimited log growth
No rotation/cap.

### KAI-SHELLWRAP-034 — HIGH — URL credential retention
Full strings are logged.

### KAI-SHELLWRAP-035 — MEDIUM — Weak log permissions
Umask only.

### KAI-SHELLWRAP-036 — MEDIUM — Concurrent interleaving
No lock.

### KAI-SHELLWRAP-037 — MEDIUM — Wall-clock evidence
No trusted sequence.

### KAI-SHELLWRAP-038 — MEDIUM — No retained response proof
Only OK/FAIL.

### KAI-SHELLWRAP-039 — MEDIUM — No failure classification
All curl failures are identical.

### KAI-SHELLWRAP-040 — MEDIUM — No alerting
Only log/exit.

### KAI-SHELLWRAP-041 — MEDIUM — Expensive duplicate probe
Dashboard endpoints perform broad internal checks.

### KAI-SHELLWRAP-042 — MEDIUM — Remote transport policy unstated
HTTPS identity/certificate policy is not pinned.

### KAI-SHELLWRAP-043 — MEDIUM — Expected fleet contract absent
No versioned manifest.

### KAI-SHELLWRAP-044 — MEDIUM — No structured output
Automation gets only aggregate exit.

---

## Scheduled wrappers — weekly/monthly scripts

### KAI-SHELLWRAP-045 — HIGH — Working-directory dependence
Relative child-script and PYTHONPATH values assume repository root.

### KAI-SHELLWRAP-046 — HIGH — Interpreter identity absent
`python` may be Python 2, wrong venv or attacker-controlled.

### KAI-SHELLWRAP-047 — HIGH — PATH substitution
No absolute trusted binaries are used.

### KAI-SHELLWRAP-048 — HIGH — Environment leakage
All secrets/configuration are inherited.

### KAI-SHELLWRAP-049 — HIGH — Arbitrary log destination
`LOG_DIR` is trusted.

### KAI-SHELLWRAP-050 — HIGH — Symlink logging
No no-follow/owner checks.

### KAI-SHELLWRAP-051 — HIGH — Sensitive plaintext output
Operational key/backup details may enter logs.

### KAI-SHELLWRAP-052 — HIGH — Incomplete audit stream
Stderr bypasses tee.

### KAI-SHELLWRAP-053 — HIGH — Unlimited logs
No rotation/retention.

### KAI-SHELLWRAP-054 — HIGH — Overlapping jobs
No flock/distributed lock.

### KAI-SHELLWRAP-055 — HIGH — Schedule names are advisory
The wrappers themselves accept any invocation time/frequency.

### KAI-SHELLWRAP-056 — HIGH — No execution deadline
Hung child processes persist.

### KAI-SHELLWRAP-057 — HIGH — No postcondition
Output/exit does not prove activation/revocation/restorability.

### KAI-SHELLWRAP-058 — HIGH — Missing operation attestation
No signed ID/revision.

### KAI-SHELLWRAP-059 — MEDIUM — Interval forwarding
Unsafe values reach child scripts.

### KAI-SHELLWRAP-060 — MEDIUM — Import-path inconsistency
Only HMAC wrapper exports `PYTHONPATH=.`.

### KAI-SHELLWRAP-061 — MEDIUM — Log mode unspecified
Umask governs access.

### KAI-SHELLWRAP-062 — MEDIUM — No wrapper timestamp
Child output alone is retained.

### KAI-SHELLWRAP-063 — MEDIUM — Partial-log ambiguity
Pipefail returns failure, but previous lines remain without status framing.

### KAI-SHELLWRAP-064 — MEDIUM — No notification
Scheduler must infer exit externally.

### KAI-SHELLWRAP-065 — MEDIUM — Artefact concurrency unprotected
No wrapper lock.

### KAI-SHELLWRAP-066 — MEDIUM — Source identity absent
Wrong checkout can run.

### KAI-SHELLWRAP-067 — MEDIUM — Duplicate backup churn
Monthly name provides no idempotency.

### KAI-SHELLWRAP-068 — MEDIUM — Accelerated rotation
Repeated calls can churn keys.

### KAI-SHELLWRAP-069 — MEDIUM — Log integrity absent
Files are mutable local evidence.

### KAI-SHELLWRAP-070 — MEDIUM — No end-to-end record
Scheduler, child, artefact and verification are not linked.

---

## Batch totals

- Findings: **70**
- Critical: **0**
- High: **35**
- Medium: **35**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,526**
- Critical: **181**
- High: **1,260**
- Medium: **1,082**
- Low: **3**

## Files materially reviewed

`scripts/contract_smoke.sh`, `scripts/health_sweep.sh`, `scripts/weekly_key_rotation.sh`, `scripts/monthly_paper_backup.sh`, `scripts/weekly_ed25519_rotation.sh`, with current memU, Tool Gate, Dashboard and wrapper-target semantics checked against source.
