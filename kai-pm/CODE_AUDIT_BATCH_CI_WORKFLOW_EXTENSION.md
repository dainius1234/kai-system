# Kai Code Audit — CI Workflow Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `.github/workflows/python-app.yml`, `.github/workflows/tech-watch-reminder.yml` and `.github/workflows/core-tests.yml`. Existing generic findings on mutable Action tags, Trivy `master`, ignored dependency installs, non-blocking audits, advisory maintenance and missing repository-wide lock manifests in `CODE_AUDIT_BATCH_CI_BACKUP.md` are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CIEXT-001 | HIGH | Python Application installs every service’s dependencies into one shared environment |
| KAI-CIEXT-002 | HIGH | Shared installation order can overwrite incompatible dependency versions between services |
| KAI-CIEXT-003 | HIGH | Tests can pass because another service installed an undeclared dependency |
| KAI-CIEXT-004 | HIGH | The root requirements file is optional and its absence is silently accepted |
| KAI-CIEXT-005 | HIGH | The enforced Flake8 pass checks only selected syntax/name error classes |
| KAI-CIEXT-006 | HIGH | Complexity and ordinary style findings are explicitly converted to warnings with `--exit-zero` |
| KAI-CIEXT-007 | HIGH | Coverage includes only five selected package roots and omits most active services |
| KAI-CIEXT-008 | HIGH | One aggregate 60% threshold can hide near-zero coverage in individual included modules |
| KAI-CIEXT-009 | HIGH | Dashboard tests are excluded from the coverage run and executed later without coverage |
| KAI-CIEXT-010 | HIGH | Fake memU embeddings are enabled for the principal test/coverage gate |
| KAI-CIEXT-011 | HIGH | JavaScript, shell, YAML, Dockerfile and policy syntax are not tested by this workflow |
| KAI-CIEXT-012 | HIGH | The workflow performs no service startup or end-to-end control-path test |
| KAI-CIEXT-013 | HIGH | One Python 3.11 environment is treated as sufficient compatibility evidence |
| KAI-CIEXT-014 | HIGH | The workflow produces no test, coverage, dependency or SBOM artefact for later audit |
| KAI-CIEXT-015 | MEDIUM | `ubuntu-latest` changes runner image and preinstalled tooling without repository revision |
| KAI-CIEXT-016 | MEDIUM | Pip is upgraded to the latest version during every run |
| KAI-CIEXT-017 | MEDIUM | Dependency installations are not cached or recorded as a resolved environment manifest |
| KAI-CIEXT-018 | MEDIUM | No job timeout or concurrency cancellation prevents long duplicate runs |
| KAI-CIEXT-019 | MEDIUM | Dashboard tests use a different working directory and import context from the main suite |
| KAI-CIEXT-020 | MEDIUM | Coverage omits branch coverage and minimums for security-critical functions |
| KAI-CIEXT-021 | MEDIUM | Test results are not linked to a deployment image digest |
| KAI-CIEXT-022 | MEDIUM | No secret scanner, JavaScript linter or static type checker runs |
| KAI-CIEXT-023 | MEDIUM | No test proves the package/service dependency declarations are individually sufficient |
| KAI-CIEXT-024 | MEDIUM | Runner environment and installed package versions are not persisted |
| KAI-CIEXT-025 | MEDIUM | The workflow has no explicit environment/profile distinguishing fake-embedding assurance from production assurance |
| KAI-CIEXT-026 | HIGH | Tech Watch issue identity includes the exact run date rather than the calendar review period |
| KAI-CIEXT-027 | HIGH | Manual runs on different dates create multiple issues for the same monthly review |
| KAI-CIEXT-028 | HIGH | Duplicate detection searches only open issues with an exact title match |
| KAI-CIEXT-029 | HIGH | The workflow does not inspect `TECH_WATCH.md` or determine whether any entry is actually stale |
| KAI-CIEXT-030 | HIGH | No package release, vulnerability or compatibility data is fetched |
| KAI-CIEXT-031 | HIGH | The fixed technology list omits most dependencies and cannot track repository changes automatically |
| KAI-CIEXT-032 | HIGH | Issue creation is the only result and does not block dependency or release decisions |
| KAI-CIEXT-033 | MEDIUM | All open issues are paginated and loaded merely to find one exact title |
| KAI-CIEXT-034 | MEDIUM | No workflow concurrency key prevents simultaneous duplicate issue creation |
| KAI-CIEXT-035 | MEDIUM | A missing `pm` or `tech-watch` label causes issue creation failure |
| KAI-CIEXT-036 | MEDIUM | The issue has no assignee, owner, due date, escalation or automatic closure policy |
| KAI-CIEXT-037 | MEDIUM | UTC date determines issue identity and can differ from operator/project timezone |
| KAI-CIEXT-038 | MEDIUM | API/rate-limit failures have no retry or durable pending state |
| KAI-CIEXT-039 | MEDIUM | The issue body contains no source SHA, dependency inventory or evidence snapshot |
| KAI-CIEXT-040 | MEDIUM | A checklist can be marked complete without any machine-verified review evidence |
| KAI-CIEXT-041 | HIGH | Core Tests globally enables the known HMAC dev-secret mode for the entire job |
| KAI-CIEXT-042 | HIGH | All unit, integration, Docker and third-party steps execute under the weakened HMAC environment |
| KAI-CIEXT-043 | HIGH | The go/no-go stage uses the confirmed fail-open release checker |
| KAI-CIEXT-044 | HIGH | The documentation-drift stage can pass from fabricated static metrics rather than executed tests/services |
| KAI-CIEXT-045 | HIGH | Numerous service tests are explicitly mocked and cannot validate real dependency, network or service identity boundaries |
| KAI-CIEXT-046 | HIGH | GPU-era permanent stubs are treated as successful foundation-test coverage |
| KAI-CIEXT-047 | HIGH | Global Workspace and other no-op foundation interfaces can pass interface tests without providing the advertised capability |
| KAI-CIEXT-048 | HIGH | GitHub Models live smoke failure is explicitly non-blocking |
| KAI-CIEXT-049 | HIGH | memU Graph live ingest/query/forget failure is explicitly non-blocking |
| KAI-CIEXT-050 | HIGH | Optional dependency installation failures are routinely tolerated before mocked tests run |
| KAI-CIEXT-051 | HIGH | The Trivy step is named container-image scanning but uses `scan-type: fs` against the repository filesystem |
| KAI-CIEXT-052 | HIGH | Built Docker images are not actually identified or scanned by image digest in the Trivy step |
| KAI-CIEXT-053 | HIGH | High and critical vulnerabilities without fixes are ignored |
| KAI-CIEXT-054 | HIGH | Minimal-stack readiness loops use `curl -s` and break on HTTP error responses as well as healthy responses |
| KAI-CIEXT-055 | HIGH | Health response bodies and semantic readiness are ignored in startup loops |
| KAI-CIEXT-056 | HIGH | Expiration of the startup wait loops does not itself fail the step |
| KAI-CIEXT-057 | HIGH | Fake embeddings are used in minimal-stack and restart-persistence validation |
| KAI-CIEXT-058 | HIGH | Kill-isolation validates only HTTP health and one memorize status, not index/database/graph consistency |
| KAI-CIEXT-059 | HIGH | The kill-isolation test creates a fixed synthetic memory record without a targeted cleanup step |
| KAI-CIEXT-060 | HIGH | `docker compose down` for the minimal stack does not remove volumes, leaving state for later steps/runs on the runner |
| KAI-CIEXT-061 | HIGH | Container logs are dumped broadly and may expose secrets, prompts, memory content and credentials |
| KAI-CIEXT-062 | HIGH | Full-stack images are built but not run as a complete integrated deployment |
| KAI-CIEXT-063 | HIGH | The sovereign boot uses the known `localdev` database password |
| KAI-CIEXT-064 | HIGH | Sovereign readiness checks accept any HTTP-success body without service identity or dependency semantics |
| KAI-CIEXT-065 | HIGH | The job does not execute authenticated Gate-to-Executor action approval as a mandatory integration test |
| KAI-CIEXT-066 | HIGH | The job does not test external host-port authentication boundaries that dominate the repository’s critical findings |
| KAI-CIEXT-067 | HIGH | One long serial job mixes unit, external, Docker, destructive and advisory checks without isolation between stages |
| KAI-CIEXT-068 | HIGH | Repository test/setup code executes while checkout credentials remain persisted by default |
| KAI-CIEXT-069 | MEDIUM | The job grants `models: read` to every step rather than only the GitHub Models smoke step |
| KAI-CIEXT-070 | MEDIUM | Third-party Actions and arbitrary repository test code run under one broad job token context |
| KAI-CIEXT-071 | MEDIUM | Coverage floors of 45%/60% leave most branches in security-critical modules untested |
| KAI-CIEXT-072 | MEDIUM | Combined coverage can mask weak coverage in a security-critical package |
| KAI-CIEXT-073 | MEDIUM | No mutation, property-based or differential tests validate the major security invariants |
| KAI-CIEXT-074 | MEDIUM | Health wait loops use fixed retry counts rather than one total deadline with explicit failure reason |
| KAI-CIEXT-075 | MEDIUM | The disk-cleanup step deletes runner tool caches and all Docker volumes without recording what was removed |
| KAI-CIEXT-076 | MEDIUM | `pip install requests` runs mid-workflow without pinning or environment manifest update |
| KAI-CIEXT-077 | MEDIUM | Docker builds are not linked to a generated SBOM, provenance attestation or signed digest |
| KAI-CIEXT-078 | MEDIUM | Integration-created records and files are not enumerated and cleaned deterministically |
| KAI-CIEXT-079 | MEDIUM | Test/report artefacts and complete logs are not uploaded for successful runs |
| KAI-CIEXT-080 | MEDIUM | The job has no overall timeout and can run through many cumulative model/service waits |
| KAI-CIEXT-081 | MEDIUM | No workflow concurrency group cancels superseded branch/PR runs |
| KAI-CIEXT-082 | MEDIUM | The workflow does not test multiple workers or replicas despite pervasive process-local state |
| KAI-CIEXT-083 | MEDIUM | The workflow does not test clock changes, restart order or concurrent maintenance against shared stores |
| KAI-CIEXT-084 | MEDIUM | A successful job has no signed consolidated assurance report linking every stage outcome |
| KAI-CIEXT-085 | MEDIUM | Best-effort live checks can disappear because of provider rate limits without reducing the final green status |
| KAI-CIEXT-086 | MEDIUM | Mocked dependency tests do not disclose mock-versus-live coverage in one consolidated result |
| KAI-CIEXT-087 | MEDIUM | The workflow uses three different Compose authorities without checking their service/configuration consistency |
| KAI-CIEXT-088 | MEDIUM | CI does not compare deployed port exposure, secrets and health contracts against an approved architecture manifest |

---

## Python Application workflow

### KAI-CIEXT-001 — HIGH — One dependency universe
All discovered requirements are installed into one Python environment.

### KAI-CIEXT-002 — HIGH — Order-dependent version replacement
Later installs can replace earlier constraints.

### KAI-CIEXT-003 — HIGH — Undeclared-dependency masking
A package installed for one service is available to every test.

### KAI-CIEXT-004 — HIGH — Optional root requirements
Absence is accepted with `|| true`.

### KAI-CIEXT-005 — HIGH — Narrow blocking lint
Only E9/F63/F7/F82 are enforced.

### KAI-CIEXT-006 — HIGH — Other lint is advisory
The second command always exits zero.

### KAI-CIEXT-007 — HIGH — Partial coverage scope
Most service directories are omitted.

### KAI-CIEXT-008 — HIGH — Aggregate masking
Only one combined threshold is used.

### KAI-CIEXT-009 — HIGH — Dashboard outside coverage
It is run separately.

### KAI-CIEXT-010 — HIGH — Fake semantic retrieval
The coverage gate enables fake embeddings.

### KAI-CIEXT-011 — HIGH — Non-Python source absent
Frontend/config/deployment syntax is not checked.

### KAI-CIEXT-012 — HIGH — No runtime integration
Only in-process tests/lint run.

### KAI-CIEXT-013 — HIGH — Single interpreter profile
No supported-version matrix.

### KAI-CIEXT-014 — HIGH — Evidence artefacts absent
Reports remain in logs.

### KAI-CIEXT-015 — MEDIUM — Mutable runner image
`ubuntu-latest` is not fixed.

### KAI-CIEXT-016 — MEDIUM — Mutable pip tool
Pip is upgraded live.

### KAI-CIEXT-017 — MEDIUM — Environment reproducibility absent
No package manifest/artifact.

### KAI-CIEXT-018 — MEDIUM — Run control absent
No timeout/concurrency.

### KAI-CIEXT-019 — MEDIUM — Different import context
Dashboard tests change CWD.

### KAI-CIEXT-020 — MEDIUM — Weak coverage semantics
No branch/security-function floor.

### KAI-CIEXT-021 — MEDIUM — Image linkage absent
Tests are not tied to built deployment artefacts.

### KAI-CIEXT-022 — MEDIUM — Static-analysis gaps
No JS/type/secret scanning.

### KAI-CIEXT-023 — MEDIUM — Per-service dependency completeness untested
Shared environment hides it.

### KAI-CIEXT-024 — MEDIUM — Runner/package evidence absent
No environment snapshot.

### KAI-CIEXT-025 — MEDIUM — Assurance profile not explicit
Fake behaviour is not encoded in result identity.

---

## Tech Watch Reminder workflow

### KAI-CIEXT-026 — HIGH — Date-specific review identity
Title uses the exact execution date.

### KAI-CIEXT-027 — HIGH — Same-month duplicates
Manual reruns on another date create another issue.

### KAI-CIEXT-028 — HIGH — Weak duplicate check
Only exact open-title matches count.

### KAI-CIEXT-029 — HIGH — No stale-entry analysis
The workflow never opens TECH_WATCH.md.

### KAI-CIEXT-030 — HIGH — No external evidence
No release/advisory API is queried.

### KAI-CIEXT-031 — HIGH — Static incomplete inventory
Technology list is hard-coded.

### KAI-CIEXT-032 — HIGH — Advisory-only outcome
Issue creation changes no control state.

### KAI-CIEXT-033 — MEDIUM — Full open-issue pagination
Potentially large repository query.

### KAI-CIEXT-034 — MEDIUM — Creation race
No concurrency key/idempotent month key.

### KAI-CIEXT-035 — MEDIUM — Label dependency
Missing labels fail the action.

### KAI-CIEXT-036 — MEDIUM — Ownership absent
No assignment/deadline/escalation.

### KAI-CIEXT-037 — MEDIUM — UTC identity
Project timezone omitted.

### KAI-CIEXT-038 — MEDIUM — No retry/pending state
API failures lose the reminder.

### KAI-CIEXT-039 — MEDIUM — Evidence-free issue
No dependency/source snapshot.

### KAI-CIEXT-040 — MEDIUM — Checkbox-only completion
No machine verification.

---

## Core Tests workflow

### KAI-CIEXT-041 — HIGH — Global dev-secret mode
The job-level environment enables it.

### KAI-CIEXT-042 — HIGH — Weakened mode spans every stage
The variable is inherited throughout the job.

### KAI-CIEXT-043 — HIGH — Fail-open gate integration
`make go_no_go` includes the unavailable-is-success checker.

### KAI-CIEXT-044 — HIGH — Self-certified documentation
`check-docs` uses source counts/static claims.

### KAI-CIEXT-045 — HIGH — Mock-heavy assurance
Many step names explicitly say mocked dependencies.

### KAI-CIEXT-046 — HIGH — Stub success
D95–D100 permanent stubs are a passing test target.

### KAI-CIEXT-047 — HIGH — No-op interface success
Foundation interfaces can pass without capability.

### KAI-CIEXT-048 — HIGH — External model failure non-blocking
Warning only.

### KAI-CIEXT-049 — HIGH — Graph live failure non-blocking
Warning only.

### KAI-CIEXT-050 — HIGH — Missing optional packages tolerated
Several pip installs use `|| true`.

### KAI-CIEXT-051 — HIGH — Scan label mismatch
Filesystem scan is called image scanning.

### KAI-CIEXT-052 — HIGH — Built images unscanned
No image ref/digest is supplied.

### KAI-CIEXT-053 — HIGH — Unfixed severe CVEs ignored
Configured true.

### KAI-CIEXT-054 — HIGH — HTTP errors satisfy startup loop
Curl lacks `-f`.

### KAI-CIEXT-055 — HIGH — Health semantics ignored
Only transport exit is used.

### KAI-CIEXT-056 — HIGH — Wait expiry non-fatal
No explicit post-loop failure at several waits.

### KAI-CIEXT-057 — HIGH — Fake embeddings in integration
Minimal/restart checks use them.

### KAI-CIEXT-058 — HIGH — Weak kill-isolation postcondition
No storage/index consistency.

### KAI-CIEXT-059 — HIGH — Fixed synthetic record remains
No targeted deletion.

### KAI-CIEXT-060 — HIGH — Minimal volumes persist
Down omits `-v`.

### KAI-CIEXT-061 — HIGH — Broad log disclosure
Full service logs are dumped.

### KAI-CIEXT-062 — HIGH — Full stack not exercised
Images build only.

### KAI-CIEXT-063 — HIGH — Known DB password
`localdev` is supplied.

### KAI-CIEXT-064 — HIGH — Sovereign readiness weak
Body/identity ignored.

### KAI-CIEXT-065 — HIGH — Mandatory action chain absent
No authenticated Gate grant consumed by Executor.

### KAI-CIEXT-066 — HIGH — Dominant exposure risks untested
Host authentication is not mandatory-tested.

### KAI-CIEXT-067 — HIGH — One shared mutable job
All stages share environment/files/system state.

### KAI-CIEXT-068 — HIGH — Persisted checkout credential exposure
Repository code/dependency hooks run after default checkout credential setup.

### KAI-CIEXT-069 — MEDIUM — Broad models permission
Job-wide rather than step-isolated.

### KAI-CIEXT-070 — MEDIUM — Shared token context
All actions/test code run in the same job authority.

### KAI-CIEXT-071 — MEDIUM — Low floors
Large untested surface remains.

### KAI-CIEXT-072 — MEDIUM — Combined floor masks packages
Aggregate coverage can pass.

### KAI-CIEXT-073 — MEDIUM — Security-test methods incomplete
No mutation/property/differential approach.

### KAI-CIEXT-074 — MEDIUM — Retry-count timing
No unified deadline/reason.

### KAI-CIEXT-075 — MEDIUM — Destructive runner cleanup unrecorded
Caches/volumes are removed.

### KAI-CIEXT-076 — MEDIUM — Mid-job unpinned install
Requests is added live.

### KAI-CIEXT-077 — MEDIUM — Image provenance absent
No attestation/SBOM/signature.

### KAI-CIEXT-078 — MEDIUM — Test residue inventory absent
Cleanup is broad, not deterministic.

### KAI-CIEXT-079 — MEDIUM — Successful artefacts absent
No uploaded reports/logs.

### KAI-CIEXT-080 — MEDIUM — Job deadline absent
Many long waits may accumulate.

### KAI-CIEXT-081 — MEDIUM — Superseded runs continue
No concurrency cancellation.

### KAI-CIEXT-082 — MEDIUM — Replica behaviour untested
Process-local-state defects remain.

### KAI-CIEXT-083 — MEDIUM — Time/order/maintenance races untested
No such scenarios exist.

### KAI-CIEXT-084 — MEDIUM — Consolidated attestation absent
No final signed evidence object.

### KAI-CIEXT-085 — MEDIUM — Live coverage can silently vanish
Best-effort failures stay green.

### KAI-CIEXT-086 — MEDIUM — Mock/live distinction fragmented
No consolidated matrix.

### KAI-CIEXT-087 — MEDIUM — Compose authorities unreconciled
Minimal/full/sovereign are tested separately.

### KAI-CIEXT-088 — MEDIUM — Architecture manifest absent
Ports/secrets/dependencies are not compared to an approved model.

---

## Batch totals

- Findings: **88**
- Critical: **0**
- High: **48**
- Medium: **40**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **3,054**
- Critical: **182**
- High: **1,563**
- Medium: **1,306**
- Low: **3**

## Files materially reviewed

`.github/workflows/python-app.yml`, `.github/workflows/tech-watch-reminder.yml`, `.github/workflows/core-tests.yml`, with prior CI findings reconciled against `CODE_AUDIT_BATCH_CI_BACKUP.md`.
