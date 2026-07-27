# Kai Code Audit — GitHub Actions and CI Release-Gate Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers all current GitHub Actions workflows. It records workflow and release-evidence defects not already counted in service, Dockerfile or runtime batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CI-001 | CRITICAL | Trivy executes from the mutable unpinned `aquasecurity/trivy-action@master` reference |
| KAI-CI-002 | CRITICAL | Friday lint status is captured from `tail`, so flake8 failures normally report a zero exit code |
| KAI-CI-003 | CRITICAL | Friday dependency-audit status is captured from `tail`, so vulnerable environments normally report success |
| KAI-CI-004 | CRITICAL | The workflow labelled “Scan container images” performs a repository filesystem scan instead of scanning the built images |
| KAI-CI-005 | CRITICAL | Pull-request-controlled code receives a GitHub token during the GitHub Models smoke-test step |
| KAI-CI-006 | HIGH | Checkout and Python setup actions use mutable major-version tags rather than immutable commit SHAs |
| KAI-CI-007 | HIGH | CI runs on the mutable `ubuntu-latest` runner image without a recorded runner-image revision |
| KAI-CI-008 | HIGH | Jobs have no overall timeout and can consume runner capacity indefinitely |
| KAI-CI-009 | HIGH | Workflows define no concurrency groups or cancellation of superseded runs |
| KAI-CI-010 | HIGH | Every service requirements file is installed into one shared Python environment |
| KAI-CI-011 | HIGH | Requirements installation failures are broadly suppressed with `|| true` |
| KAI-CI-012 | HIGH | Requirements installation order is filesystem-dependent and can resolve conflicts nondeterministically |
| KAI-CI-013 | HIGH | Root requirements may be installed twice in the Python application workflow |
| KAI-CI-014 | HIGH | Dependency-vulnerability scans are advisory and do not fail the main workflows |
| KAI-CI-015 | HIGH | Python application dependency auditing discards standard error and hides diagnostic evidence |
| KAI-CI-016 | HIGH | Dependency auditing examines a merged runner environment rather than each deployable service or image |
| KAI-CI-017 | HIGH | CI explicitly permits the known shared development HMAC secret |
| KAI-CI-018 | HIGH | Integration tests explicitly enable fake embeddings and can pass without the production retrieval model |
| KAI-CI-019 | HIGH | Sovereign-stack tests explicitly deploy the known database password `localdev` |
| KAI-CI-020 | HIGH | memU Graph live verification is marked best-effort and cannot fail the build |
| KAI-CI-021 | HIGH | GitHub Models smoke testing is best-effort and cannot fail the build |
| KAI-CI-022 | HIGH | Trivy ignores unfixed critical and high vulnerabilities |
| KAI-CI-023 | HIGH | Full-stack images are built, but the complete deployed fleet is never started and tested end-to-end |
| KAI-CI-024 | HIGH | Service wait loops can exhaust without explicitly failing at the timeout boundary |
| KAI-CI-025 | HIGH | Health waits check only HTTP success and not semantic readiness or required dependency state |
| KAI-CI-026 | HIGH | Failure log dumps can expose secrets, credentials, private prompts and operational data |
| KAI-CI-027 | HIGH | Weekly and Friday issue reports publish raw command and test output into repository issues |
| KAI-CI-028 | HIGH | Report body data is not safely escaped from Markdown fences, mentions or report-structure injection |
| KAI-CI-029 | HIGH | Weekly report-card checks intentionally exit zero regardless of go/no-go or test failure |
| KAI-CI-030 | HIGH | Friday lint, audit and documentation checks intentionally exit zero and cannot enforce cleanup quality |
| KAI-CI-031 | HIGH | The security scan produces no SBOM, signed provenance or image-attestation artefact |
| KAI-CI-032 | HIGH | Base images, service images and model tags remain mutable supply-chain identities |
| KAI-CI-033 | HIGH | No CodeQL, equivalent SAST or repository secret-scanning workflow is present |
| KAI-CI-034 | HIGH | `requests` is installed unpinned during the graph verification job |
| KAI-CI-035 | HIGH | Graph verification downloads external model and extension dependencies during CI |
| KAI-CI-036 | HIGH | A large portion of security-critical tests relies on mocks, stubs and fake backends |
| KAI-CI-037 | HIGH | Coverage enforcement omits most host-published and privileged services |
| KAI-CI-038 | HIGH | Duplicate CI workflows implement inconsistent test, lint, dependency and security gates |
| KAI-CI-039 | HIGH | Public report issues can disclose security findings and internal diagnostic details in a public repository |
| KAI-CI-040 | HIGH | The workflow token is available to PR-controlled Makefile and scripts during model testing |
| KAI-CI-041 | MEDIUM | Workflow actions lack exact source digest provenance in generated reports |
| KAI-CI-042 | MEDIUM | Coverage, test XML, dependency-audit and Trivy results are not retained as workflow artefacts |
| KAI-CI-043 | MEDIUM | Weekly and Friday workflows create recurring issue churn without stable report IDs or deduplication |
| KAI-CI-044 | MEDIUM | Drift detection examines only the first 100 branches |
| KAI-CI-045 | MEDIUM | Drift detection examines only the first 100 open issues |
| KAI-CI-046 | MEDIUM | Drift report reuse depends on mutable title and label heuristics |
| KAI-CI-047 | MEDIUM | Raw branch names and issue titles are inserted into Markdown reports |
| KAI-CI-048 | MEDIUM | Friday report output is truncated with `tail`, losing complete lint and vulnerability evidence |
| KAI-CI-049 | MEDIUM | Broad flake8 quality checking uses `--exit-zero` and never gates the build |
| KAI-CI-050 | MEDIUM | Scheduled workflows use UTC only and are not tied to the operator’s configured timezone |
| KAI-CI-051 | MEDIUM | Static `EOF` delimiters in `$GITHUB_OUTPUT` can be collided with by command output |
| KAI-CI-052 | MEDIUM | Large command output may exceed GitHub output and issue-body limits and become silently truncated |
| KAI-CI-053 | MEDIUM | Report publication can fail when labels are absent, and some workflows downgrade that loss to a warning |
| KAI-CI-054 | MEDIUM | Generated maintenance/report issues have no retention or closure policy |
| KAI-CI-055 | MEDIUM | Friday Cleanup performs no cleanup and only produces an advisory report |
| KAI-CI-056 | MEDIUM | Scheduled reports omit the exact tested commit SHA and dependency-policy digest from their body |
| KAI-CI-057 | MEDIUM | No branch-protection or release-deployment gate is evidenced by these workflows |
| KAI-CI-058 | MEDIUM | Cleanup and log-dump steps can obscure the original failure when their own commands fail |
| KAI-CI-059 | MEDIUM | Integration tests validate directly exposed unauthenticated ports rather than asserting access control |
| KAI-CI-060 | MEDIUM | Workflow stages share one mutable dependency environment without a clean isolation boundary |

---

## Critical findings

### KAI-CI-001 — CRITICAL — Mutable third-party action executes in CI
**Issue:** `core-tests.yml` uses `aquasecurity/trivy-action@master`. The `master` reference can move to different code without any repository change.  
**Risk:** Compromise or unexpected changes in that upstream repository execute with the job’s token, checked-out source and runner access.  
**Recommendation:** pin every action to a reviewed immutable commit SHA and manage updates through controlled dependency review.  
**Status:** OPEN — immediate remediation required

### KAI-CI-002 — CRITICAL — Friday lint failures are converted to success
**Issue:** The command substitution is `flake8 ... | tail -20` without `set -o pipefail`. `$?` is therefore the exit code of `tail`, normally zero.  
**Risk:** Syntax, undefined-name and quality failures are reported as passing and cannot trigger an enforcing workflow failure.  
**Recommendation:** enable `pipefail`, capture the originating command status and fail the job for mandatory lint classes.  
**Status:** OPEN — immediate remediation required

### KAI-CI-003 — CRITICAL — Friday vulnerability findings are converted to success
**Issue:** `pip-audit ... | tail -30` repeats the same pipeline-status defect.  
**Risk:** Known vulnerable dependencies are represented with a successful exit code and green report icon.  
**Recommendation:** preserve and enforce the `pip-audit` exit status before formatting output.  
**Status:** OPEN — immediate remediation required

### KAI-CI-004 — CRITICAL — Built images are not scanned
**Issue:** The step named “Scan container images for vulnerabilities” configures `scan-type: 'fs'` and `scan-ref: '.'`. It scans repository files, not the images built in the preceding step.  
**Risk:** OS packages, transitive image layers, installed service dependencies and runtime artefacts can contain critical vulnerabilities while the claimed image gate passes.  
**Recommendation:** enumerate every built image digest and scan each image with a fail-closed severity policy.  
**Status:** OPEN — immediate remediation required

### KAI-CI-005 — CRITICAL — PR code receives a GitHub token
**Issue:** `core-tests.yml` runs on `pull_request`, checks out the proposed code, and later exposes `${{ secrets.GITHUB_TOKEN }}` as `GITHUB_TOKEN` to `make test-github-models`. The checked-out Makefile and scripts control the command.  
**Risk:** Malicious or compromised PR code can read and exfiltrate the token or use its `models: read`/repository permissions.  
**Recommendation:** never expose credentials to untrusted PR code; move credentialled smoke tests to a protected post-merge or explicitly approved workflow using immutable trusted scripts.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-CI-006 — HIGH — Actions are pinned only to mutable major tags
`actions/checkout@v4` and `actions/setup-python@v5` do not identify the exact reviewed action commit.

### KAI-CI-007 — HIGH — Mutable runner identity
`ubuntu-latest` changes OS packages, tools and behaviour over time without a workflow revision.

### KAI-CI-008 — HIGH — No job timeout
Model pulls, Docker builds, tests, network calls or hung scripts can occupy a runner until platform limits intervene.

### KAI-CI-009 — HIGH — Overlapping runs
Pushes, pull requests, schedules and manual dispatches have no concurrency key or cancellation policy, multiplying costly Docker/model activity.

### KAI-CI-010 — HIGH — One mixed Python environment
Every discovered service requirements file is installed into one interpreter, unlike the isolated deployed images.

### KAI-CI-011 — HIGH — Dependency-install failure is suppressed
Several loops append `|| true`, allowing missing or incompatible service dependencies while tests continue under a partial environment.

### KAI-CI-012 — HIGH — Nondeterministic installation ordering
`find` output order determines which conflicting package version is installed last.

### KAI-CI-013 — HIGH — Duplicate root dependency installation
`python-app.yml` installs `requirements.txt` explicitly and then discovers it again in the find loop.

### KAI-CI-014 — HIGH — Vulnerability gates are advisory
All principal pip-audit invocations either use warning fallbacks or explicitly exit zero.

### KAI-CI-015 — HIGH — Dependency diagnostics are discarded
The Python application workflow redirects audit standard error to `/dev/null`.

### KAI-CI-016 — HIGH — Wrong audit target
The merged runner environment is audited, not the dependency set of each Docker image, lock file or production resolution.

### KAI-CI-017 — HIGH — Known HMAC secret permitted
`HMAC_ALLOW_DEV_SECRET=true` masks a security-readiness failure that production must reject.

### KAI-CI-018 — HIGH — Fake retrieval backend accepted
`MEMU_ALLOW_FAKE_EMBEDDINGS=true` permits integration and restart tests to succeed without the production embedding model.

### KAI-CI-019 — HIGH — Known database password in integration
The sovereign-stack job explicitly sets `DB_PASSWORD: localdev`.

### KAI-CI-020 — HIGH — Graph verification cannot gate
The live graph cycle is followed by `|| echo warning`.

### KAI-CI-021 — HIGH — Real-model smoke cannot gate
The GitHub Models check similarly downgrades every failure to a warning.

### KAI-CI-022 — HIGH — Unfixed critical vulnerabilities are ignored
Trivy uses `ignore-unfixed: true`, excluding precisely the vulnerabilities that may require blocking deployment.

### KAI-CI-023 — HIGH — Full fleet is never exercised
The full Compose file is built, but integration brings up minimal subsets and a separate graph subset; cross-service startup, access control and fleet readiness are not tested together.

### KAI-CI-024 — HIGH — Wait-loop exhaustion is not consistently enforced
Several loops simply finish after the maximum iteration and rely on a later or absent check rather than failing immediately with the timed-out service identity.

### KAI-CI-025 — HIGH — HTTP success is readiness
Waits use `curl`/urllib against health endpoints already audited as green in stub/no-token/degraded states.

### KAI-CI-026 — HIGH — Failure logs may expose secrets
Complete Compose logs can contain database URIs, tokens, prompts, private memory or model/provider diagnostics.

### KAI-CI-027 — HIGH — Raw evidence is published to issues
Weekly and Friday workflows embed command, test, lint and audit output in issue bodies.

### KAI-CI-028 — HIGH — Report-content injection
Untrusted command output can close Markdown fences, create mentions/links and spoof headings/status within generated reports.

### KAI-CI-029 — HIGH — Weekly regressions do not enforce anything
Both go/no-go and tests use `set +e`, capture status and `exit 0`; the job proceeds to issue creation and remains successful.

### KAI-CI-030 — HIGH — Friday quality failures do not enforce anything
Every principal check deliberately exits zero and only changes an icon.

### KAI-CI-031 — HIGH — No build provenance
Images have no SBOM, signed attestation, reproducible build manifest or link from source commit to image digest.

### KAI-CI-032 — HIGH — Mutable supply-chain tags
Docker/base/model references are not proven by immutable digest in the workflow.

### KAI-CI-033 — HIGH — Missing repository security analysis
No CodeQL/equivalent semantic SAST or explicit secret-scanning workflow is present.

### KAI-CI-034 — HIGH — Unpinned mid-job dependency
`pip install requests` resolves the current index version during the graph step.

### KAI-CI-035 — HIGH — Runtime dependency acquisition
Graph validation depends on model and extension downloads from external services during the job.

### KAI-CI-036 — HIGH — Security-critical behaviour is mocked
Browser, vision, email, feed, sensor, graph and other tests deliberately replace real trust/network/model boundaries.

### KAI-CI-037 — HIGH — Coverage excludes most exposed services
The explicit coverage set omits Tool Gate, Executor, Dashboard, Supervisor, Verifier, Fusion, perception, messaging and many other services.

### KAI-CI-038 — HIGH — Inconsistent duplicate gates
`python-app.yml` and `core-tests.yml` run overlapping suites with different excludes, dependencies, coverage and security semantics.

### KAI-CI-039 — HIGH — Security details may become public
In a public repository, generated issue content can expose vulnerabilities, internal topology and sensitive diagnostics to everyone.

### KAI-CI-040 — HIGH — Makefile/script token trust
Even where the platform restricts fork secrets, same-repository or otherwise credentialled PR execution still delegates the token to mutable checked-out commands rather than immutable trusted workflow code.

---

## Medium-severity findings

### KAI-CI-041 — MEDIUM — Action provenance absent from reports
Reports do not record the exact action SHAs or runner image used.

### KAI-CI-042 — MEDIUM — Evidence artefacts are not retained
No uploaded test XML, full coverage, audit JSON, SBOM or Trivy result supports later investigation.

### KAI-CI-043 — MEDIUM — Recurring report issue churn
Weekly Report Card and Friday Cleanup create new issues rather than one stable report object with structured history.

### KAI-CI-044 — MEDIUM — Branch pagination missing
The branch API requests one page of 100 entries.

### KAI-CI-045 — MEDIUM — Issue pagination missing
The issues API also inspects only the first 100 open entries.

### KAI-CI-046 — MEDIUM — Heuristic report reuse
Drift reporting selects the latest matching title/label, which can be renamed or duplicated.

### KAI-CI-047 — MEDIUM — Raw repository metadata in Markdown
Branch names and issue titles are not escaped before publication.

### KAI-CI-048 — MEDIUM — Evidence is deliberately truncated
Only the last 20/30 lines of lint/audit output are retained.

### KAI-CI-049 — MEDIUM — Broad lint is advisory
The second flake8 invocation uses `--exit-zero`.

### KAI-CI-050 — MEDIUM — Operator-timezone mismatch
Maintenance/report schedules are hard-coded UTC.

### KAI-CI-051 — MEDIUM — Output delimiter collision
Captured command text containing a line exactly equal to `EOF` can terminate the multi-line output early and alter later output fields.

### KAI-CI-052 — MEDIUM — Output-size truncation
Large test/audit output can exceed Actions output or GitHub issue limits without a durable complete artefact.

### KAI-CI-053 — MEDIUM — Report publication can disappear
Missing labels or issue API failures are sometimes reduced to warnings.

### KAI-CI-054 — MEDIUM — No report retention
Generated issue lifecycle, closure, access and deletion are not governed.

### KAI-CI-055 — MEDIUM — Cleanup is advisory only
The Friday workflow detects conditions but performs no cleanup or enforceable remediation.

### KAI-CI-056 — MEDIUM — Tested revision omitted
Issue bodies do not record the exact commit SHA, dependency resolution or policy digest.

### KAI-CI-057 — MEDIUM — No demonstrated release linkage
The workflows do not show a signed release/deployment gate requiring these exact checks.

### KAI-CI-058 — MEDIUM — Secondary failure noise
Always-run Docker log/teardown commands can fail and complicate identification of the originating failure.

### KAI-CI-059 — MEDIUM — Tests normalise open ports
Integration scripts call privileged host-published ports directly rather than asserting that unauthenticated access is denied.

### KAI-CI-060 — MEDIUM — Shared mutable stage environment
Package installs and job-side changes persist across later steps, making results order-dependent and reducing isolation.

---

## Batch totals

- Findings: **60**
- Critical: **5**
- High: **35**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,656**
- Critical: **201**
- High: **1,340**
- Medium: **1,112**
- Low: **3**

## Files materially reviewed

`.github/workflows/python-app.yml`, `.github/workflows/core-tests.yml`, `.github/workflows/weekly-report-card.yml`, `.github/workflows/drift-detector.yml` and `.github/workflows/friday-cleanup.yml`.
