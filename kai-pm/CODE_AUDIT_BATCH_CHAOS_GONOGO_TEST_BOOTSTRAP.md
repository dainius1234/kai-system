# Kai Code Audit — Chaos, Go/No-Go and Test Bootstrap Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/chaos_ci.py`, `scripts/go_no_go_check.py`, root `conftest.py`, and the directly related `Makefile` go/no-go wiring. Existing service/authentication defects are not duplicated; this batch records assurance-layer failures.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-ASSURE-001 | CRITICAL | Go/no-go exits successfully whenever Dashboard is unavailable, unreachable, malformed or returns an HTTP error |
| KAI-ASSURE-002 | HIGH | The standalone go/no-go fallback claims “static checks only” while performing no static checks itself |
| KAI-ASSURE-003 | HIGH | Go/no-go trusts one unauthenticated plaintext Dashboard response as the release decision |
| KAI-ASSURE-004 | HIGH | Any service at localhost:8080 returning `{"decision":"GO"}` can satisfy the gate |
| KAI-ASSURE-005 | HIGH | Go/no-go validates only the `decision` field and ignores `core_ready`, failed checks, evidence and policy state |
| KAI-ASSURE-006 | HIGH | Dashboard response identity, deployment version, policy revision and freshness are not verified |
| KAI-ASSURE-007 | HIGH | The full Dashboard response is read without a byte limit |
| KAI-ASSURE-008 | HIGH | HTTP status/content type and response schema types are not explicitly validated |
| KAI-ASSURE-009 | HIGH | All exceptions—transport, timeout, JSON, decoding and programming failures—are converted to a successful exit |
| KAI-ASSURE-010 | HIGH | The hard-coded localhost endpoint cannot represent remote, namespaced or authenticated deployments |
| KAI-ASSURE-011 | HIGH | The go/no-go script executes network and exit logic at import time |
| KAI-ASSURE-012 | HIGH | The Makefile go/no-go target compiles only a manually selected subset of Python files |
| KAI-ASSURE-013 | HIGH | The go/no-go target misses the Dashboard client’s confirmed fatal JavaScript syntax error |
| KAI-ASSURE-014 | HIGH | Active Python services and modules outside the selected compile list can be broken while the gate passes |
| KAI-ASSURE-015 | HIGH | `py_compile` does not execute imports, startup configuration, routes or lifecycle code |
| KAI-ASSURE-016 | HIGH | Compose, Dockerfile, shell, JavaScript, YAML, policy and secret configuration are outside the gate |
| KAI-ASSURE-017 | HIGH | The Makefile globally enables fake memU embeddings for tests and assurance targets |
| KAI-ASSURE-018 | HIGH | The go/no-go target uses PATH-resolved `python` without verifying interpreter or environment identity |
| KAI-ASSURE-019 | MEDIUM | No retries or transient/permanent dependency distinction exists |
| KAI-ASSURE-020 | MEDIUM | The three-second timeout is not accompanied by a total release-check deadline or timing evidence |
| KAI-ASSURE-021 | MEDIUM | A successful gate produces no structured signed report, source SHA or evidence digest |
| KAI-ASSURE-022 | MEDIUM | No operator, CI run, branch or commit identity is bound to the decision |
| KAI-ASSURE-023 | MEDIUM | The check does not retain the accepted Dashboard response for later audit |
| KAI-ASSURE-024 | MEDIUM | A stale cached or replayed GO response cannot be distinguished from a current evaluation |
| KAI-ASSURE-025 | HIGH | Chaos CI inherits the complete caller environment, including live database, Redis, secret and service configuration |
| KAI-ASSURE-026 | HIGH | Chaos services are not placed in isolated databases, Redis namespaces, vector indexes, ledgers or filesystems |
| KAI-ASSURE-027 | HIGH | Chaos runs can mutate real memU, Tool Gate and Agentic state |
| KAI-ASSURE-028 | HIGH | The repository trusted-token file is used directly by the chaos Tool Gate process |
| KAI-ASSURE-029 | HIGH | Service commands use PATH-resolved `python` and caller working-directory-relative source paths |
| KAI-ASSURE-030 | HIGH | Failure during partial startup leaves already-started processes running because cleanup begins only after `start()` returns |
| KAI-ASSURE-031 | HIGH | Fixed chaos ports are not checked for availability or ownership before processes start |
| KAI-ASSURE-032 | HIGH | Service readiness is replaced by fixed sleeps of two seconds and one second |
| KAI-ASSURE-033 | HIGH | A process that exits immediately can still be treated as successfully started |
| KAI-ASSURE-034 | HIGH | Standard output and error are discarded by default, hiding startup and failure evidence |
| KAI-ASSURE-035 | HIGH | The “random” kill is deterministic on every run because the RNG is always seeded with 42 |
| KAI-ASSURE-036 | HIGH | Only three services are included despite the much larger active deployment |
| KAI-ASSURE-037 | HIGH | No client load or in-flight request is generated while services are killed or restarted |
| KAI-ASSURE-038 | HIGH | The killed process is overwritten in the process list without first being waited/reaped |
| KAI-ASSURE-039 | HIGH | memU is restarted one second after SIGTERM without confirming shutdown or port/resource release |
| KAI-ASSURE-040 | HIGH | Restart can race the old process and create port, database or index conflicts |
| KAI-ASSURE-041 | HIGH | Process termination targets only direct children and not descendant process groups |
| KAI-ASSURE-042 | HIGH | State integrity, ledger continuity, memory consistency and pending-work preservation are never checked after restart |
| KAI-ASSURE-043 | HIGH | The game-day scorecard is not bound to the chaos service ports or process identities |
| KAI-ASSURE-044 | HIGH | Chaos can report passed from independent Makefile tests that never contacted the killed/restarted processes |
| KAI-ASSURE-045 | HIGH | The game-day scorecard subprocess has no timeout |
| KAI-ASSURE-046 | HIGH | Mutable Makefile targets, PATH and Python environment define the post-chaos validation at runtime |
| KAI-ASSURE-047 | MEDIUM | The deterministic scenario order exercises one narrow sequence only |
| KAI-ASSURE-048 | MEDIUM | No pre-chaos baseline is captured for comparison |
| KAI-ASSURE-049 | MEDIUM | No measured outage duration, recovery time, error rate or SLO evidence is collected |
| KAI-ASSURE-050 | MEDIUM | The `memu_degrade` scenario immediately restarts memU rather than testing an actual degraded interval |
| KAI-ASSURE-051 | MEDIUM | Unknown scenarios only warn inside `run_scenario()` rather than fail, if called programmatically |
| KAI-ASSURE-052 | MEDIUM | Dry-run output does not disclose exact commands, environment, ports, stores or validation steps |
| KAI-ASSURE-053 | MEDIUM | Chaos CI writes no structured result, source commit, process IDs or environment digest |
| KAI-ASSURE-054 | MEDIUM | Stop logic does not wait after SIGKILL to prove every process was reaped |
| KAI-ASSURE-055 | MEDIUM | Stop/kill failures are warnings and do not invalidate a prior pass result |
| KAI-ASSURE-056 | MEDIUM | Signals and process semantics are platform-dependent but no supported-platform check exists |
| KAI-ASSURE-057 | MEDIUM | No concurrency lock prevents two chaos runs using the same ports and stores |
| KAI-ASSURE-058 | MEDIUM | No cleanup reconciles records, files, nonces, ledgers or sessions created during the run |
| KAI-ASSURE-059 | HIGH | Root `conftest.py` globally enables use of the known `local-dev-shared-secret` for every pytest test |
| KAI-ASSURE-060 | HIGH | Tests without any configured HMAC secret silently sign and verify with a source-known shared secret |
| KAI-ASSURE-061 | HIGH | Production missing-secret and placeholder-secret failures are masked by the global test environment |
| KAI-ASSURE-062 | HIGH | The dev-secret allowance applies to all tests rather than an explicit local-auth test fixture or marker |
| KAI-ASSURE-063 | HIGH | Subprocesses spawned by tests inherit the dev-secret allowance |
| KAI-ASSURE-064 | HIGH | Integration tests can start services in a security-weakened mode without reporting that fact in results |
| KAI-ASSURE-065 | HIGH | Authentication tests can certify internally consistent signatures generated with the publicly known default secret |
| KAI-ASSURE-066 | HIGH | Test results do not distinguish a real configured secret from the default dev secret |
| KAI-ASSURE-067 | HIGH | Global environment mutation occurs before every test import and is not restored |
| KAI-ASSURE-068 | MEDIUM | Externally supplied `HMAC_ALLOW_DEV_SECRET` values control the suite because `setdefault()` does not enforce a known test state |
| KAI-ASSURE-069 | MEDIUM | Tests cannot exercise the secure default-deny path unless they explicitly undo global configuration |
| KAI-ASSURE-070 | MEDIUM | Parallel and in-process tests share the same mutable authentication environment |
| KAI-ASSURE-071 | MEDIUM | Module-level secret warnings/caches may persist after individual tests change environment variables |
| KAI-ASSURE-072 | MEDIUM | The bootstrap contains no fixture scope, teardown or per-test isolation |
| KAI-ASSURE-073 | MEDIUM | No marker requires a separate production-secret test lane |
| KAI-ASSURE-074 | MEDIUM | Test reports contain no flag indicating that dev-secret mode was enabled |
| KAI-ASSURE-075 | MEDIUM | The conftest policy affects third-party and future tests automatically without explicit opt-in |
| KAI-ASSURE-076 | MEDIUM | Test bootstrap state has no configuration digest or assurance record |

---

## Go/no-go — `scripts/go_no_go_check.py`, `Makefile`

### KAI-ASSURE-001 — CRITICAL — Runtime absence passes release gate
Every exception exits zero. Dashboard outage, timeout, malformed JSON, HTTP error and implementation failure all become a successful release decision.

### KAI-ASSURE-002 — HIGH — False “static checks” claim
The fallback branch performs no local validation; it merely prints the claim and exits.

### KAI-ASSURE-003 — HIGH — Unauthenticated release authority
One plaintext GET controls the result.

### KAI-ASSURE-004 — HIGH — Service impersonation
No expected identity or signature is checked.

### KAI-ASSURE-005 — HIGH — One-field semantics
Only exact `decision == GO` is inspected.

### KAI-ASSURE-006 — HIGH — Missing provenance/freshness
No revision, timestamp, policy hash or evidence is required.

### KAI-ASSURE-007 — HIGH — Unbounded response read
`resp.read()` has no limit.

### KAI-ASSURE-008 — HIGH — Weak response contract
Media type and field types are not checked.

### KAI-ASSURE-009 — HIGH — All errors fail open
The broad exception catches everything.

### KAI-ASSURE-010 — HIGH — Deployment inflexibility
URL is hard-coded to localhost and cannot authenticate.

### KAI-ASSURE-011 — HIGH — Import-time exit/network
The module is not reusable safely.

### KAI-ASSURE-012 — HIGH — Partial compile manifest
The Makefile list is manually selected.

### KAI-ASSURE-013 — HIGH — JavaScript outage invisible
The fatal Dashboard frontend parse error is outside Python compilation.

### KAI-ASSURE-014 — HIGH — Active-source omissions
Many services/modules are absent.

### KAI-ASSURE-015 — HIGH — Compile is not readiness
Import-time and runtime failures remain invisible.

### KAI-ASSURE-016 — HIGH — Non-Python controls omitted
Deployment and policy files are not checked.

### KAI-ASSURE-017 — HIGH — Fake embedding default
Make exports `MEMU_ALLOW_FAKE_EMBEDDINGS=true` broadly.

### KAI-ASSURE-018 — HIGH — Interpreter identity absent
`python` is PATH-resolved.

### KAI-ASSURE-019 — MEDIUM — No retry policy
Transient failure and permanent absence are identical.

### KAI-ASSURE-020 — MEDIUM — Weak timing evidence
No total duration/report.

### KAI-ASSURE-021 — MEDIUM — No attested result
Terminal output only.

### KAI-ASSURE-022 — MEDIUM — Actor/run identity absent
No branch/SHA/job binding.

### KAI-ASSURE-023 — MEDIUM — Evidence discarded
Accepted payload is not persisted.

### KAI-ASSURE-024 — MEDIUM — Replay/staleness undetectable
No nonce or as-of requirement.

---

## Chaos CI — `scripts/chaos_ci.py`

### KAI-ASSURE-025 — HIGH — Live environment inheritance
All caller variables are copied.

### KAI-ASSURE-026 — HIGH — Store isolation absent
No test-specific database/Redis/index/ledger paths.

### KAI-ASSURE-027 — HIGH — Real-state mutation risk
Started services can use production-like stores and credentials.

### KAI-ASSURE-028 — HIGH — Live token file
Repository token path is used.

### KAI-ASSURE-029 — HIGH — Untrusted command/cwd
PATH and working directory define executables/files.

### KAI-ASSURE-030 — HIGH — Partial-start leak
`start()` exceptions occur before the main `finally`.

### KAI-ASSURE-031 — HIGH — Port ownership unverified
Fixed ports may be occupied.

### KAI-ASSURE-032 — HIGH — Sleep-based readiness
No health polling.

### KAI-ASSURE-033 — HIGH — Dead child accepted
`Popen` success is the only startup proof.

### KAI-ASSURE-034 — HIGH — Diagnostics suppressed
DEVNULL by default.

### KAI-ASSURE-035 — HIGH — Fixed “random” target
Seed is constant.

### KAI-ASSURE-036 — HIGH — Tiny fleet coverage
Three services only.

### KAI-ASSURE-037 — HIGH — No traffic during failure
Recovery under use is untested.

### KAI-ASSURE-038 — HIGH — Zombie/reference loss
Killed child is replaced without wait.

### KAI-ASSURE-039 — HIGH — Unconfirmed memU shutdown
Restart follows fixed delay.

### KAI-ASSURE-040 — HIGH — Restart race
Port/files/store may remain in use.

### KAI-ASSURE-041 — HIGH — Descendants survive
No process group/session handling.

### KAI-ASSURE-042 — HIGH — State postconditions absent
No integrity comparison.

### KAI-ASSURE-043 — HIGH — Validator disconnected from subjects
Scorecard uses ordinary Make targets/default configuration.

### KAI-ASSURE-044 — HIGH — False chaos pass
Independent tests can pass while chaos processes never became ready or recovered.

### KAI-ASSURE-045 — HIGH — Validator can hang
No timeout.

### KAI-ASSURE-046 — HIGH — Mutable assurance authority
Makefile/PATH/environment define checks.

### KAI-ASSURE-047 — MEDIUM — One deterministic sequence
Coverage is narrow.

### KAI-ASSURE-048 — MEDIUM — Baseline absent
No pre-failure measurement.

### KAI-ASSURE-049 — MEDIUM — SLOs not measured
No RTO/error metrics.

### KAI-ASSURE-050 — MEDIUM — Degradation not exercised
memU is brought back immediately.

### KAI-ASSURE-051 — MEDIUM — Programmatic unknown scenario warning
No exception.

### KAI-ASSURE-052 — MEDIUM — Dry-run lacks detail
Not auditable.

### KAI-ASSURE-053 — MEDIUM — No run artefact
Only logs/stdout.

### KAI-ASSURE-054 — MEDIUM — Kill reap unverified
No wait after kill.

### KAI-ASSURE-055 — MEDIUM — Cleanup failure non-fatal
Warnings only.

### KAI-ASSURE-056 — MEDIUM — Platform assumptions
Signals/process semantics unvalidated.

### KAI-ASSURE-057 — MEDIUM — Concurrent-run collision
No lock.

### KAI-ASSURE-058 — MEDIUM — Test-data cleanup absent
State remains.

---

## Test bootstrap — `conftest.py`, `common/auth.py`

### KAI-ASSURE-059 — HIGH — Suite-wide known-secret allowance
Global conftest enables the dev secret for every pytest collection.

### KAI-ASSURE-060 — HIGH — Default secret becomes active test authority
`common.auth` falls back to `local-dev-shared-secret` when no real secret is supplied.

### KAI-ASSURE-061 — HIGH — Production failure modes masked
Missing/placeholder secret protection is not the default test condition.

### KAI-ASSURE-062 — HIGH — No explicit scope
All tests receive the exception.

### KAI-ASSURE-063 — HIGH — Subprocess inheritance
Spawned services inherit the weakened environment.

### KAI-ASSURE-064 — HIGH — Hidden insecure integration mode
Reports do not mark it.

### KAI-ASSURE-065 — HIGH — Self-consistent known-secret tests
Sign/verify can pass with a public source constant.

### KAI-ASSURE-066 — HIGH — Secret quality invisible
Results do not distinguish real versus default.

### KAI-ASSURE-067 — HIGH — Persistent global mutation
No teardown.

### KAI-ASSURE-068 — MEDIUM — External state ambiguity
`setdefault` allows caller overrides.

### KAI-ASSURE-069 — MEDIUM — Secure-path tests require manual undo
Default suite path is permissive.

### KAI-ASSURE-070 — MEDIUM — Shared parallel state
Environment is process-global.

### KAI-ASSURE-071 — MEDIUM — Module cache contamination
Imported auth state/warnings persist.

### KAI-ASSURE-072 — MEDIUM — No fixture lifecycle
Setup is module-level only.

### KAI-ASSURE-073 — MEDIUM — No production-secret lane marker
Coverage cannot be guaranteed.

### KAI-ASSURE-074 — MEDIUM — Dev mode absent from reports
Reviewers cannot see the exception.

### KAI-ASSURE-075 — MEDIUM — Future-test implicit opt-in
Every newly added test inherits it.

### KAI-ASSURE-076 — MEDIUM — Bootstrap provenance absent
No configuration digest/evidence.

---

## Batch totals

- Findings: **76**
- Critical: **1**
- High: **43**
- Medium: **32**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,722**
- Critical: **182**
- High: **1,381**
- Medium: **1,156**
- Low: **3**

## Files materially reviewed

`scripts/chaos_ci.py`, `scripts/go_no_go_check.py`, root `conftest.py`, related `Makefile` targets and current `common/auth.py` dev-secret semantics.
