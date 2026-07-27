# Kai Code Audit — Release, Bootstrap and Repository Gate Tooling Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers Makefile release targets and the scripts used as go/no-go, quality, Phase-closure, health, contract, documentation and import-shadow gates. CI workflow defects remain in `CODE_AUDIT_BATCH_CI_WORKFLOWS.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-RELEASE-001 | CRITICAL | `go_no_go_check.py` exits success whenever Dashboard is unavailable or its response cannot be parsed |
| KAI-RELEASE-002 | CRITICAL | `phase1_closure_check.py` declares every patch set closed from a few regex and file-presence checks |
| KAI-RELEASE-003 | CRITICAL | Phase 1 closure requires a TODO comment as positive proof of completion |
| KAI-RELEASE-004 | CRITICAL | `smoke_core.py` probes PostgreSQL’s native port with HTTP and therefore makes the required core smoke fail |
| KAI-RELEASE-005 | CRITICAL | The Makefile’s global default enables fake embeddings for every target, including release and hardening gates |
| KAI-RELEASE-006 | HIGH | The go/no-go gate compiles only a manually selected subset of Python services |
| KAI-RELEASE-007 | HIGH | Successful `py_compile` is treated as meaningful operational readiness |
| KAI-RELEASE-008 | HIGH | The dynamic go/no-go decision trusts Dashboard’s already-audited advisory and fabricated readiness metrics |
| KAI-RELEASE-009 | HIGH | The go/no-go URL is hard-coded to one unauthenticated localhost endpoint |
| KAI-RELEASE-010 | HIGH | The merge gate includes a dependency-audit target that always downgrades findings to a warning |
| KAI-RELEASE-011 | HIGH | Coverage phase one suppresses every test or coverage failure with `|| true` |
| KAI-RELEASE-012 | HIGH | Suppressed phase-one coverage data is still appended into the later coverage report |
| KAI-RELEASE-013 | HIGH | Quality Gate scans only top-level `scripts/*.py` and ignores nested operational scripts |
| KAI-RELEASE-014 | HIGH | Quality Gate skips every file beginning `test_` even when it is invoked as an operational smoke/evaluation script |
| KAI-RELEASE-015 | HIGH | Quality Gate misses bare `pass`, ellipsis, placeholder returns and most semantic stubs |
| KAI-RELEASE-016 | HIGH | Quality Gate checks only module docstrings and provides no implementation-quality or safety validation |
| KAI-RELEASE-017 | HIGH | Phase-closure parsing uses regular expressions over YAML text rather than the Compose data model |
| KAI-RELEASE-018 | HIGH | Phase-closure dependency checking recognises only list syntax and cannot validate readiness conditions |
| KAI-RELEASE-019 | HIGH | Phase-closure health checks validate only that the word `healthcheck:` exists |
| KAI-RELEASE-020 | HIGH | Phase closure does not validate ports, authentication, secrets, images, volumes, networks or runtime behaviour |
| KAI-RELEASE-021 | HIGH | `contract_smoke.sh` verifies only key presence and accepts false, error-shaped or semantically invalid values |
| KAI-RELEASE-022 | HIGH | Contract smoke never authenticates to the privileged services it exercises |
| KAI-RELEASE-023 | HIGH | Contract smoke reads Tool Gate ledger integrity without requiring `valid=true` |
| KAI-RELEASE-024 | HIGH | Contract smoke accepts Dashboard `core_ready=false` as long as the key exists |
| KAI-RELEASE-025 | HIGH | The declared contract-smoke session token is unused |
| KAI-RELEASE-026 | HIGH | Health Sweep treats any successful HTTP response as healthy without reading semantic state |
| KAI-RELEASE-027 | HIGH | Health Sweep calls Dashboard readiness, which performs expensive active fleet fan-out and stateful probes |
| KAI-RELEASE-028 | HIGH | Health Sweep writes raw endpoint URLs to a persistent log without credential redaction |
| KAI-RELEASE-029 | HIGH | Core Smoke waits five seconds while claiming a thirty-second warm-up |
| KAI-RELEASE-030 | HIGH | Core Smoke has no retry or service-specific readiness deadline |
| KAI-RELEASE-031 | HIGH | Every optional execution, perception and output service is explicitly non-fatal in Core Smoke |
| KAI-RELEASE-032 | HIGH | Documentation Sync hard-codes the published failure count to zero |
| KAI-RELEASE-033 | HIGH | Documentation test counts measure textual `def test_` declarations rather than collected or passing tests |
| KAI-RELEASE-034 | HIGH | Documentation milestone counts are derived from README’s own DONE labels, creating circular self-attestation |
| KAI-RELEASE-035 | HIGH | Missing README status table causes the documentation check to pass |
| KAI-RELEASE-036 | HIGH | Missing PROJECT_BACKLOG causes the documentation check to pass |
| KAI-RELEASE-037 | HIGH | PyPI-shadow checking relies on a static local blocklist rather than the actual dependency/package namespace |
| KAI-RELEASE-038 | HIGH | `KAI_SHADOW_ALLOW` lets the caller bypass any blocklisted package name without signed approval |
| KAI-RELEASE-039 | HIGH | The permanent `langgraph` exemption is not verified as the expected safe symlink target |
| KAI-RELEASE-040 | MEDIUM | Quality Gate’s TODO detector requires the marker at the start of a comment and misses many forms of unfinished work |
| KAI-RELEASE-041 | MEDIUM | Tokenisation failures return “no marker” and can hide unfinished code until a separate parse check happens |
| KAI-RELEASE-042 | MEDIUM | The `KNOWN_STUBS` mechanism has no expiry, owner or issue-reference schema |
| KAI-RELEASE-043 | MEDIUM | Health Sweep may require root permission to create `/var/log/sovereign` |
| KAI-RELEASE-044 | MEDIUM | Health Sweep has no per-endpoint timeout and can hang on a slow connection |
| KAI-RELEASE-045 | MEDIUM | Health Sweep has no immutable run ID, tested revision or structured result artefact |
| KAI-RELEASE-046 | MEDIUM | Core Smoke labels HTTP reachability as health and ignores response schemas |
| KAI-RELEASE-047 | MEDIUM | Contract Smoke embeds JSON responses in command-line arguments and can exceed shell argument limits |
| KAI-RELEASE-048 | MEDIUM | Contract Smoke provides no whole-run timeout, retry or target revision |
| KAI-RELEASE-049 | MEDIUM | Documentation service counting is a line-oriented Compose heuristic rather than parsed YAML |
| KAI-RELEASE-050 | MEDIUM | Documentation LOC counting silently ignores unreadable files and still publishes a normal total |
| KAI-RELEASE-051 | MEDIUM | The latest commit is computed by Documentation Sync but omitted from the generated status table |
| KAI-RELEASE-052 | MEDIUM | Documentation dates use the host locale/timezone and non-portable `%-d` formatting |
| KAI-RELEASE-053 | MEDIUM | README and backlog writes are direct non-atomic replacements without a retained generation |
| KAI-RELEASE-054 | MEDIUM | PyPI-shadow checking inspects only repository-root directories and misses files, nested paths and namespace-package collisions |
| KAI-RELEASE-055 | MEDIUM | Database initialisation retains a known `localdev` password fallback in the Makefile |
| KAI-RELEASE-056 | MEDIUM | Merge-gate stages share one mutable local Python/environment state and are order-dependent |
| KAI-RELEASE-057 | MEDIUM | Release and smoke outputs have no signed tested-commit, configuration or dependency digest |
| KAI-RELEASE-058 | MEDIUM | Full-up and core-up targets start privileged stacks without an explicit locked-mode safety precondition |

---

## Critical findings

### KAI-RELEASE-001 — CRITICAL — Unavailable Dashboard means GO
**Issue:** `scripts/go_no_go_check.py` catches every connection, timeout, HTTP and JSON error and exits with status zero while printing “static checks only”.  
**Risk:** The release gate succeeds precisely when its runtime decision authority is unavailable, broken or returning malformed data.  
**Recommendation:** fail closed when dynamic readiness is required; separate an explicitly named compile-only command from an enforcing runtime gate.  
**Status:** OPEN — immediate remediation required

### KAI-RELEASE-002 — CRITICAL — Static regexes declare Phase 1 complete
**Issue:** `phase1_closure_check.py` checks a few dependency strings, healthcheck labels and script existence, then reports patch sets A–F closed.  
**Risk:** Major security, availability and runtime defects are represented as completed closure evidence without executing the stack or validating controls.  
**Recommendation:** make closure evidence a versioned checklist of testable security/runtime postconditions tied to an immutable deployment revision.  
**Status:** OPEN — immediate remediation required

### KAI-RELEASE-003 — CRITICAL — TODO comment is proof of closure
**Issue:** Closure explicitly requires `# TODO: enable GPU when core is stable.` to exist in Compose.  
**Risk:** An acknowledged unfinished item becomes a required success condition and the script fails if the debt is actually removed or reworded.  
**Recommendation:** closure must require the completed capability or an approved tracked exception, never the presence of a TODO.  
**Status:** OPEN — immediate remediation required

### KAI-RELEASE-004 — CRITICAL — PostgreSQL is probed as HTTP
**Issue:** Core Smoke calls `httpx.get("http://localhost:5432")`. PostgreSQL speaks its own wire protocol, not HTTP.  
**Risk:** The required core-service set normally fails even when PostgreSQL is healthy, making the gate unusable and encouraging operators to ignore/bypass it.  
**Recommendation:** use `pg_isready` or an authenticated database query and validate the expected database/schema.  
**Status:** OPEN — immediate remediation required

### KAI-RELEASE-005 — CRITICAL — Fake embeddings are the global Make default
**Issue:** The first Makefile assignment exports `MEMU_ALLOW_FAKE_EMBEDDINGS ?= true` for all targets.  
**Risk:** Go/no-go, merge, hardening, integration and developer commands silently exercise hash vectors instead of the production semantic model unless every caller explicitly overrides the variable.  
**Recommendation:** default production/release targets to real-model-or-fail and confine fake embeddings to clearly named isolated test targets.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-RELEASE-006 — HIGH — Partial syntax inventory
`go_no_go` compiles a fixed list and omits many host-published services and scripts.

### KAI-RELEASE-007 — HIGH — Syntax is treated as readiness
`py_compile` does not import dependencies, execute startup, validate configuration or test endpoints.

### KAI-RELEASE-008 — HIGH — Gate trusts advisory Dashboard data
The dynamic decision is exactly the Dashboard decision already based on incomplete fleet inventory, total ledger counts and Dashboard HTTP error ratios.

### KAI-RELEASE-009 — HIGH — Hard-coded unauthenticated authority
The gate cannot securely target a selected environment or authenticate/verify the responding Dashboard.

### KAI-RELEASE-010 — HIGH — Non-fatal dependency gate
`make dep-audit` always succeeds after printing a warning.

### KAI-RELEASE-011 — HIGH — Coverage failures suppressed
The first coverage phase ends with `|| true`, hiding collection/test/import and coverage errors.

### KAI-RELEASE-012 — HIGH — Failed coverage contaminates later evidence
The second phase appends to whatever incomplete/stale `.coverage` data phase one left behind.

### KAI-RELEASE-013 — HIGH — Non-recursive script gate
Only direct children of `scripts/` ending `.py` are scanned.

### KAI-RELEASE-014 — HIGH — Operational `test_` scripts excluded
All such filenames are skipped even though the repository invokes many of them directly as operational checks and evaluations.

### KAI-RELEASE-015 — HIGH — Stub detection is lexical and incomplete
It recognises only `raise NotImplementedError` and narrow TODO/pass-comment patterns.

### KAI-RELEASE-016 — HIGH — Docstring presence becomes quality
A script with one module docstring and unsafe/no-op implementation passes.

### KAI-RELEASE-017 — HIGH — Regex-based Compose model
Comments, anchors, indentation and alternate valid YAML syntax can satisfy or defeat checks independently of actual Compose semantics.

### KAI-RELEASE-018 — HIGH — Dependency syntax blind spot
Only `- dependency` is recognised; mapping-form `depends_on` and conditions are not validated.

### KAI-RELEASE-019 — HIGH — Health semantics absent
A stub/no-token/always-green health route satisfies closure.

### KAI-RELEASE-020 — HIGH — Closure scope is grossly incomplete
Authentication, host exposure, secret quality, network segmentation, runtime permissions, persistence and actual service startup are outside the check.

### KAI-RELEASE-021 — HIGH — Key-presence-only contract
`check_keys` validates no types, values, enums or relationships.

### KAI-RELEASE-022 — HIGH — Smoke calls privileged APIs anonymously
A passing result validates the insecure public interface rather than the intended authentication boundary.

### KAI-RELEASE-023 — HIGH — Ledger invalidity can pass
The script checks only that `status` and `valid` keys exist.

### KAI-RELEASE-024 — HIGH — NO_GO readiness can pass
Dashboard values can be false/zero/error-shaped while all expected keys exist.

### KAI-RELEASE-025 — HIGH — Declared token is unused
`SESSION_ID=bootstrap-token-1` never enters any request, creating false assurance that contract calls are authenticated/session-bound.

### KAI-RELEASE-026 — HIGH — HTTP-only health sweep
The script discards bodies and validates only curl success status.

### KAI-RELEASE-027 — HIGH — Readiness sweep has active side effects
Dashboard readiness invokes its full root/fleet logic rather than a bounded passive readiness snapshot.

### KAI-RELEASE-028 — HIGH — URL credential leakage
Environment URLs may contain credentials or tokens and are written verbatim to the health log.

### KAI-RELEASE-029 — HIGH — False warm-up statement
The message says 30 seconds; the actual sleep is five seconds.

### KAI-RELEASE-030 — HIGH — No warm-up retry
Every core service is probed exactly once after five seconds.

### KAI-RELEASE-031 — HIGH — Privileged optional fleet is non-fatal
Executor, Agentic, Audio, Camera, Advisor, TTS and Avatar failures never affect exit status.

### KAI-RELEASE-032 — HIGH — Published failure metric is fabricated
README status rows always contain `| **Failures** | 0 |` without inspecting tests, audit or incidents.

### KAI-RELEASE-033 — HIGH — Declared tests are counted as passed capacity
Regex counting includes skipped, dead, parametrisation-independent and uncollected functions.

### KAI-RELEASE-034 — HIGH — Circular milestone evidence
The tool counts DONE labels already present in the document it updates.

### KAI-RELEASE-035 — HIGH — Missing README gate fails open
Failure to find the expected table prints a warning and returns true.

### KAI-RELEASE-036 — HIGH — Missing backlog gate fails open
A missing backlog file is accepted as current.

### KAI-RELEASE-037 — HIGH — Static shadow list is incomplete
Only names prelisted in `.pypi_shadow_blocklist` are checked; newly dangerous package-name collisions pass.

### KAI-RELEASE-038 — HIGH — Environment bypass of shadow policy
Any caller can add names to `KAI_SHADOW_ALLOW` without an approved revision or audit.

### KAI-RELEASE-039 — HIGH — Permanent exemption unverified
`langgraph/` is always allowed without proving it remains a safe shim to the intended source.

---

## Medium-severity findings

### KAI-RELEASE-040 — MEDIUM — Narrow TODO grammar
Inline comments such as `# later TODO` or other unfinished markers are not detected.

### KAI-RELEASE-041 — MEDIUM — Tokenisation failure hides marker result
`TokenizeError` returns false from the marker scanner rather than a dedicated failure.

### KAI-RELEASE-042 — MEDIUM — Stub exceptions lack governance
If populated later, `KNOWN_STUBS` contains no expiry/owner/issue/reference enforcement.

### KAI-RELEASE-043 — MEDIUM — Health-log permission assumption
Creating `/var/log/sovereign` may fail for non-root developer/CI execution.

### KAI-RELEASE-044 — MEDIUM — Curl timeout absent
A slow socket can hold the sweep indefinitely.

### KAI-RELEASE-045 — MEDIUM — No structured health artefact
Only append-only text lines remain, without commit/config identity or integrity.

### KAI-RELEASE-046 — MEDIUM — Core Smoke ignores health body
Any HTTP success is labelled ok.

### KAI-RELEASE-047 — MEDIUM — JSON passed via argv
Large Dashboard/memU payloads become shell argument data and may exceed OS limits or leak through process inspection.

### KAI-RELEASE-048 — MEDIUM — Contract run lacks deadline
Each curl uses no explicit timeout and the whole script has no run identifier/retry policy.

### KAI-RELEASE-049 — MEDIUM — Compose metrics are heuristic
Service counting reads indentation/lines rather than parsed files and merged profiles.

### KAI-RELEASE-050 — MEDIUM — LOC evidence silently incomplete
Unreadable Python files are skipped with no partial flag.

### KAI-RELEASE-051 — MEDIUM — Commit evidence discarded
`get_latest_commit()` populates a metric that is never rendered into the status table.

### KAI-RELEASE-052 — MEDIUM — Host-dependent date
Local timezone/locale and `%-d` affect generated documentation.

### KAI-RELEASE-053 — MEDIUM — Non-atomic documentation mutation
Files are overwritten directly without temporary replacement, fsync or concurrent-edit protection.

### KAI-RELEASE-054 — MEDIUM — Import-shadow coverage incomplete
Only root directories are examined; root modules, `.pth`, namespace packages, symlinked trees and nested working-directory collisions are outside scope.

### KAI-RELEASE-055 — MEDIUM — Known database fallback remains
`init-memu-db` constructs `keeper:localdev` when no password is provided.

### KAI-RELEASE-056 — MEDIUM — Merge stages are stateful
Tests, coverage and audit commands reuse environment variables, installed packages and generated files in one workspace.

### KAI-RELEASE-057 — MEDIUM — No release evidence envelope
Outputs do not bind exact source SHA, Compose/policy hashes, dependency lock resolution and machine/runtime identity.

### KAI-RELEASE-058 — MEDIUM — Unsafe startup convenience targets
`full-up`/`core-up` perform no locked-mode, secret-quality or access-control preflight before launching the stack.

---

## Batch totals

- Findings: **58**
- Critical: **5**
- High: **34**
- Medium: **19**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,714**
- Critical: **206**
- High: **1,374**
- Medium: **1,131**
- Low: **3**

## Files materially reviewed

`Makefile`, `scripts/go_no_go_check.py`, `scripts/quality_gate.py`, `scripts/phase1_closure_check.py`, `scripts/smoke_core.py`, `scripts/health_sweep.sh`, `scripts/contract_smoke.sh`, `scripts/check_pypi_shadow.sh` and `scripts/sync_docs.py`.
