# Kai Code Audit — CI, Supply Chain and Backup Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

This batch contains evidence-backed findings confirmed after the master register reached 104 findings. IDs are reserved for final consolidation.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CI-001 | HIGH | Weekly report validation failures are converted into successful workflow completion |
| KAI-CI-002 | HIGH | Dependency vulnerability findings do not fail the Friday Cleanup workflow |
| KAI-CI-003 | HIGH | Dependency installation failures are repeatedly ignored |
| KAI-CI-004 | MEDIUM | GitHub Actions are referenced through mutable major-version tags |
| KAI-CI-005 | MEDIUM | Trivy is executed from the mutable `master` branch |
| KAI-CI-006 | MEDIUM | CI dynamically resolves dependencies without a repository-wide lock manifest |
| KAI-CI-007 | MEDIUM | Maintenance workflows create advisory issues but do not enforce release or remediation gates |
| KAI-BACK-001 | CRITICAL | PostgreSQL restore is exposed without authentication or authorisation |
| KAI-BACK-002 | HIGH | Backup operations are exposed without authentication or authorisation |
| KAI-BACK-003 | HIGH | Backup listing discloses internal paths and backup metadata |
| KAI-BACK-004 | HIGH | Backup service has a predictable PostgreSQL credential fallback |
| KAI-BACK-005 | HIGH | Off-site backup can report completion after skipping the PostgreSQL dump |
| KAI-BACK-006 | HIGH | Ledger backup exports only statistics rather than recoverable ledger records |
| KAI-BACK-007 | MEDIUM | Memory export uses record-count-derived `top_k` without pagination or a bounded export protocol |
| KAI-BACK-008 | MEDIUM | Backup integrity checksum is stored beside the archive on the same storage |
| KAI-BACK-009 | MEDIUM | Backup-retention deletion errors are suppressed |
| KAI-BACK-010 | MEDIUM | The off-site workflow does not enforce or verify remote transfer |
| KAI-BACK-011 | MEDIUM | Backups are not automatically restore-tested before acceptance |

---

## GitHub Actions and dependency supply chain

### KAI-CI-001 — HIGH — Weekly report validation failures are converted into successful workflow completion
**Issue:** The workflow captures non-zero results from go/no-go validation and unit tests, records them for reporting, and explicitly exits zero.  
**Risk:** The workflow can complete successfully while core validation or test execution has failed, preventing branch and release controls from relying on its status.  
**Recommendation:** Preserve the original failure status after generating reports, or split advisory reporting from mandatory blocking checks.  
**Status:** OPEN

### KAI-CI-002 — HIGH — Dependency vulnerability findings do not fail the Friday Cleanup workflow
**Issue:** `pip-audit --strict` has its exit status captured and is followed by an unconditional successful exit.  
**Risk:** Known vulnerable dependencies can be detected without producing a failed security check.  
**Recommendation:** Make the security job blocking under an explicit vulnerability policy and use documented exception records where necessary.  
**Status:** OPEN

### KAI-CI-003 — HIGH — Dependency installation failures are repeatedly ignored
**Issue:** Dependency installation commands use patterns such as `pip install ... || true`.  
**Risk:** Tests and vulnerability scans may execute against an incomplete or materially different environment while appearing valid.  
**Recommendation:** Fail dependency setup when required packages cannot be installed and distinguish intentionally optional packages explicitly.  
**Status:** OPEN

### KAI-CI-004 — MEDIUM — GitHub Actions are referenced through mutable major-version tags
**Issue:** Workflows use references including `actions/checkout@v4`, `actions/setup-python@v5`, and `actions/github-script@v7`.  
**Risk:** The executed third-party workflow code can change without a repository commit, weakening reproducibility and review.  
**Recommendation:** Pin actions to reviewed commit SHAs and update them through a controlled dependency process.  
**Status:** OPEN

### KAI-CI-005 — MEDIUM — Trivy is executed from the mutable `master` branch
**Issue:** The workflow references `aquasecurity/trivy-action@master`.  
**Risk:** Unreviewed upstream changes can immediately alter security-scanning code and results.  
**Recommendation:** Pin to a reviewed release commit SHA.  
**Status:** OPEN

### KAI-CI-006 — MEDIUM — CI dynamically resolves dependencies without a repository-wide lock manifest
**Issue:** Workflows discover and install multiple service `requirements.txt` files dynamically; no resolved repository-wide dependency lock was observed in the reviewed workflow path.  
**Risk:** Identical commits can test against different transitive dependency versions over time.  
**Recommendation:** Produce reviewed, hashed lock manifests and use deterministic installation.  
**Status:** OPEN

### KAI-CI-007 — MEDIUM — Maintenance workflows are advisory rather than enforcing
**Issue:** Weekly Report Card, Drift Detector, and Friday Cleanup primarily create or update GitHub issues and do not establish mandatory approval, remediation or release gates.  
**Risk:** Identified drift and security failures can remain informational while deployment proceeds.  
**Recommendation:** Separate informational maintenance from defined blocking controls with explicit ownership and exception expiry.  
**Status:** OPEN

---

## Backup and recovery

### KAI-BACK-001 — CRITICAL — PostgreSQL restore is exposed without authentication or authorisation
**Issue:** `POST /restore/postgres` accepts a backup filename and invokes `psql` against the configured database without an authentication or operator-authorisation dependency.  
**Risk:** Any caller with network reachability can replace or destructively alter production database state using an available backup file.  
**Recommendation:** Remove network-exposed restore, or require strong operator identity, step-up approval, maintenance mode and immutable audit evidence.  
**Status:** OPEN — immediate remediation required

### KAI-BACK-002 — HIGH — Backup operations are exposed without authentication or authorisation
**Issue:** PostgreSQL, Redis, memory, ledger and full-backup endpoints are callable without visible access control.  
**Risk:** Reachable callers can trigger expensive and sensitive operational jobs, create storage pressure and obtain backup result metadata.  
**Recommendation:** Require authenticated service/operator scopes, rate limits and job-level audit records.  
**Status:** OPEN

### KAI-BACK-003 — HIGH — Backup listing discloses internal paths and metadata
**Issue:** `/backup/list` returns backup filenames, full internal paths, sizes and modification times without access control. Backup operations also return generated paths.  
**Risk:** Callers gain sensitive filesystem and recovery-state intelligence useful for further compromise.  
**Recommendation:** Restrict listing to authorised operators and expose opaque backup IDs rather than internal paths.  
**Status:** OPEN

### KAI-BACK-004 — HIGH — Predictable PostgreSQL credential fallback
**Issue:** `PG_URI` defaults to `postgresql://postgres:postgres@postgres:5432/postgres`.  
**Risk:** A deployment missing configuration runs with a source-known administrative-style credential.  
**Recommendation:** Refuse startup without managed credentials and reject known development values outside a development profile.  
**Status:** OPEN

### KAI-BACK-005 — HIGH — Off-site backup can report completion after skipping PostgreSQL
**Issue:** `backup_offsite.sh` logs a warning and continues when the PostgreSQL backup request fails, then creates the archive and logs `Backup complete`.  
**Risk:** Operators can treat an incomplete archive as a valid disaster-recovery backup even though the primary database was omitted.  
**Recommendation:** Define mandatory components, fail the job when any mandatory component is absent and include verified component status in a signed manifest.  
**Status:** OPEN

### KAI-BACK-006 — HIGH — Ledger backup is not a recoverable ledger export
**Issue:** `/backup/ledger` calls only `/ledger/stats` and writes those statistics to JSON.  
**Risk:** The resulting file cannot restore the trust/tool ledger despite being labelled as a ledger backup.  
**Recommendation:** Export immutable ledger records with completeness ranges, chain verification and restore validation.  
**Status:** OPEN

### KAI-BACK-007 — MEDIUM — Memory export lacks a bounded paginated protocol
**Issue:** Memory export obtains the reported record count and sends it as `top_k` in one retrieval request.  
**Risk:** Large stores can create excessive response size, timeout or memory pressure, and completeness depends on ordinary search semantics.  
**Recommendation:** Use a snapshot/export API with cursor pagination, stable ordering and completeness metadata.  
**Status:** OPEN

### KAI-BACK-008 — MEDIUM — Checksum is colocated with the archive
**Issue:** A SHA-256 sidecar is written beside the backup on the same destination.  
**Risk:** An attacker or storage fault able to replace the archive can also replace its checksum; the sidecar proves accidental integrity only when separately trusted.  
**Recommendation:** Sign manifests and publish integrity anchors to independent append-only storage.  
**Status:** OPEN

### KAI-BACK-009 — MEDIUM — Retention deletion errors are suppressed
**Issue:** Archive pruning ends with `2>/dev/null || true`.  
**Risk:** Retention failures and uncontrolled storage growth remain invisible.  
**Recommendation:** Treat cleanup failure as an observable job failure and alert on retention-policy drift.  
**Status:** OPEN

### KAI-BACK-010 — MEDIUM — Off-site transfer is not enforced or verified
**Issue:** The script writes to a local destination. Remote transfer appears only as an example command printed after completion.  
**Risk:** A backup labelled off-site may remain in the same host or failure domain as the live system.  
**Recommendation:** Integrate authenticated remote replication and verify remote object existence, integrity and retention before declaring success.  
**Status:** OPEN

### KAI-BACK-011 — MEDIUM — No automatic restore test
**Issue:** The reviewed backup flow creates files and checksums but does not restore them into an isolated environment and validate application-level integrity.  
**Risk:** Corrupt, incomplete or semantically unusable backups can remain undetected until an incident.  
**Recommendation:** Run scheduled isolated restore drills with database, ledger and memory consistency checks.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **1**
- High: **8**
- Medium: **9**
- Low: **0**

## Provisional repository totals after consolidation

- Findings: **122**
- Critical: **19**
- High: **54**
- Medium: **48**
- Low: **1**

## Files materially reviewed in this batch

`.github/workflows/weekly-report-card.yml`, `.github/workflows/drift-detector.yml`, `.github/workflows/friday-cleanup.yml`, `.github/workflows/pm-status.yml`, `.env.example`, `docker-compose.full.yml`, `scripts/backup_offsite.sh`, `backup-service/app.py`.
