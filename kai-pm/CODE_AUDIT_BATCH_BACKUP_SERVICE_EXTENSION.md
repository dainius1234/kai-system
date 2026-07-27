# Kai Code Audit — Backup Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_BACKUP_SERVICE.md` or the separate CI/off-site backup batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BACKUPX-001 | CRITICAL | Restoring a tampered “SQL backup” can execute psql client meta-commands as the Backup Service user |
| KAI-BACKUPX-002 | HIGH | The deployed image installs none of `pg_dump`, `psql` or `redis-cli` |
| KAI-BACKUPX-003 | HIGH | The Dockerfile neither creates nor grants access to the default `/data/backup` directory |
| KAI-BACKUPX-004 | HIGH | Compose mounts no persistent backup volume, so any successful artefacts remain container-local and disposable |
| KAI-BACKUPX-005 | HIGH | The Backup container cannot access Redis’s filesystem path or `dump.rdb` volume |
| KAI-BACKUPX-006 | HIGH | The deployed Redis backup path therefore reports `status=ok` without creating a backup artefact |
| KAI-BACKUPX-007 | HIGH | The mounted database-password secret is ignored by the application connection string |
| KAI-BACKUPX-008 | HIGH | Compose silently falls back to `localdev` through environment interpolation when `DB_PASSWORD` is absent |
| KAI-BACKUPX-009 | HIGH | Health remains green in the exact deployment where required binaries, storage and Redis data access are absent |
| KAI-BACKUPX-010 | HIGH | PostgreSQL credentials are passed in command-line arguments visible to process inspection |
| KAI-BACKUPX-011 | HIGH | Restore credentials are likewise exposed in the `psql` process command line |
| KAI-BACKUPX-012 | HIGH | Backup and restore subprocesses inherit the complete service environment |
| KAI-BACKUPX-013 | HIGH | Failed or timed-out `pg_dump` operations can leave partial files in the normal backup namespace |
| KAI-BACKUPX-014 | HIGH | Backup files are written directly to final paths without atomic temporary-file promotion or fsync |
| KAI-BACKUPX-015 | HIGH | Partial/in-progress files are immediately visible to listing and restore operations |
| KAI-BACKUPX-016 | HIGH | Backup listing follows symbolic links and can disclose metadata for files outside the backup directory |
| KAI-BACKUPX-017 | HIGH | PostgreSQL restore follows symbolic links inside the backup directory |
| KAI-BACKUPX-018 | HIGH | Restore accepts any regular file name in the directory rather than a verified PostgreSQL backup artefact |
| KAI-BACKUPX-019 | HIGH | Restore does not require a `.sql` extension, component prefix, manifest entry or creation job identity |
| KAI-BACKUPX-020 | HIGH | `psql` is invoked without `ON_ERROR_STOP`, so SQL statement failures can still lead to a reported successful restore |
| KAI-BACKUPX-021 | HIGH | Restore is not wrapped in a transaction and can leave a partially applied database |
| KAI-BACKUPX-022 | HIGH | No pre-restore snapshot or compensating rollback is created before destructive database mutation |
| KAI-BACKUPX-023 | HIGH | Restore does not quiesce writers, terminate sessions or enter a maintenance/fencing mode |
| KAI-BACKUPX-024 | HIGH | Backup metadata contains no source database identity, server version, schema revision or migration compatibility |
| KAI-BACKUPX-025 | HIGH | A backup from another database/environment can be restored into the configured target without compatibility checks |
| KAI-BACKUPX-026 | HIGH | Full backup components are captured sequentially and do not represent one coherent point-in-time system snapshot |
| KAI-BACKUPX-027 | HIGH | The “full backup manifest” is returned only in the HTTP response and is not durably stored with the artefacts |
| KAI-BACKUPX-028 | HIGH | Memory and ledger export files are created without checksums or signed manifests |
| KAI-BACKUPX-029 | HIGH | Memory export trusts a caller-visible memU record count as the requested retrieval size and has no completeness proof |
| KAI-BACKUPX-030 | HIGH | Memory export contains no pagination cursor, snapshot revision or end-of-data evidence |
| KAI-BACKUPX-031 | HIGH | Backup-to-memU and backup-to-Tool-Gate traffic uses unauthenticated plain HTTP |
| KAI-BACKUPX-032 | HIGH | Backup Service has no read-only snapshot identity or source authentication for downstream export responses |
| KAI-BACKUPX-033 | MEDIUM | PostgreSQL dump format is not recorded in a versioned manifest |
| KAI-BACKUPX-034 | MEDIUM | Restore does not validate SQL encoding, file header or expected dump structure before execution |
| KAI-BACKUPX-035 | MEDIUM | Restore timeout controls only the direct `psql` process and does not establish database rollback/postcondition semantics |
| KAI-BACKUPX-036 | MEDIUM | Backup list performs an unbounded glob/stat scan before returning only 50 entries |
| KAI-BACKUPX-037 | MEDIUM | Arbitrary unrelated files in the backup directory are included in inventory and restore selection |
| KAI-BACKUPX-038 | MEDIUM | Inventory contains no checksum, component type, job ID, verification state or restorability result |
| KAI-BACKUPX-039 | MEDIUM | There is no retention, maximum-volume, minimum-free-space or automatic cleanup policy |
| KAI-BACKUPX-040 | MEDIUM | Backup endpoints define no strict response models or schema version |
| KAI-BACKUPX-041 | MEDIUM | Backup inventory and operation responses lack `Cache-Control: no-store` |
| KAI-BACKUPX-042 | MEDIUM | Full-backup requests have no job ID, cancellation endpoint, progress state or retry identity |
| KAI-BACKUPX-043 | MEDIUM | Client/proxy retries can start duplicate full or component backups |
| KAI-BACKUPX-044 | MEDIUM | No restore-verification drill proves that produced PostgreSQL, Redis or memory artefacts are usable |
| KAI-BACKUPX-045 | MEDIUM | No audit record binds operator, source revisions, generated files, checksums and restore outcome |
| KAI-BACKUPX-046 | MEDIUM | Operation timestamps mix timezone-aware and naive UTC generation |
| KAI-BACKUPX-047 | MEDIUM | FastAPI/HTTPX dependencies and the Python base image are not reproducibly digest-pinned |
| KAI-BACKUPX-048 | MEDIUM | No dedicated service tests were found for real tools, permissions, symlinks, partial dumps or restore failure semantics |
| KAI-BACKUPX-049 | MEDIUM | The service has no lifespan-owned job queue, shared clients, graceful shutdown drain or incomplete-job reconciliation |
| KAI-BACKUPX-050 | MEDIUM | Public operation errors can expose the configured backup directory and backup filenames through filesystem exceptions |

---

## Critical finding

### KAI-BACKUPX-001 — CRITICAL — Restore can execute psql meta-commands
**Issue:** Restore accepts any regular file in `BACKUP_DIR` and executes it with `psql -f`. The service verifies neither that the file was produced by `pg_dump` nor that it contains only SQL. psql input files may contain client meta-commands, including shell-command execution.  
**Risk:** A replaced, symlinked or otherwise attacker-controlled “backup” can turn the unauthenticated restore endpoint into operating-system command execution under the Backup Service account.  
**Recommendation:** Restore only signed immutable dump artefacts in a non-executable format through an isolated restore worker; reject psql meta-command-bearing plaintext input.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-BACKUPX-002 — HIGH — Required client tools absent
**Issue:** The Dockerfile installs only Python packages. It does not install PostgreSQL or Redis client utilities.  
**Risk:** PostgreSQL backup/restore and Redis backup deterministically return unavailable in the normal image.  
**Recommendation:** Build a reviewed tool image with exact client versions and make successful probes part of readiness.  
**Status:** OPEN

### KAI-BACKUPX-003 — HIGH — Default storage path is not provisioned
The non-root image does not create/chown `/data/backup`; `os.makedirs` normally cannot create `/data` beneath the root filesystem.

### KAI-BACKUPX-004 — HIGH — Backup data is not persistent
Compose defines no volume at `/data/backup`.

### KAI-BACKUPX-005 — HIGH — Redis dump filesystem is isolated
Redis `CONFIG GET dir` describes a path inside the Redis container. Backup Service mounts neither that filesystem nor the Redis data volume.

### KAI-BACKUPX-006 — HIGH — False successful Redis backup
The known inaccessible-RDB branch still returns `status=ok` and is counted as a successful full-backup component.

### KAI-BACKUPX-007 — HIGH — Docker secret is unused
Compose mounts `db_password`, but the application reads only the already-expanded `PG_URI`; it never reads `/run/secrets/db_password`.

### KAI-BACKUPX-008 — HIGH — Known environment fallback
The Compose connection string uses `${DB_PASSWORD:-localdev}` independently from the mounted secret.

### KAI-BACKUPX-009 — HIGH — Deterministically false health
`/health` does not test binaries, target access, directory creation or durable storage.

### KAI-BACKUPX-010 — HIGH — Dump credential process exposure
The complete URI is an argument to `pg_dump`.

### KAI-BACKUPX-011 — HIGH — Restore credential process exposure
The complete URI is an argument to `psql`.

### KAI-BACKUPX-012 — HIGH — Broad subprocess environment
No minimal environment is supplied to backup/restore utilities.

### KAI-BACKUPX-013 — HIGH — Partial dump remains trusted-looking
On timeout/CalledProcessError, the final-path file is not removed or marked incomplete.

### KAI-BACKUPX-014 — HIGH — Non-atomic backup publication
Artefacts are written directly to their final names and returned before fsync/durable manifest commit.

### KAI-BACKUPX-015 — HIGH — In-progress artefact exposure
Listing/restore has no job-state filter and can see a path while another operation writes it.

### KAI-BACKUPX-016 — HIGH — Inventory symlink escape
`glob`, `isfile`, `getsize` and `getmtime` follow symlinks.

### KAI-BACKUPX-017 — HIGH — Restore symlink escape
`isfile` and `psql -f` follow a symlink whose basename passed the regex.

### KAI-BACKUPX-018 — HIGH — Component type not enforced
A Redis, memory, ledger or unrelated file can be passed to the PostgreSQL restore handler.

### KAI-BACKUPX-019 — HIGH — No manifest-bound selection
Filename syntax alone establishes eligibility.

### KAI-BACKUPX-020 — HIGH — SQL errors may be reported as success
Without `-v ON_ERROR_STOP=1`, psql can continue processing after statement errors and exit without the strict failure semantics expected by the endpoint.

### KAI-BACKUPX-021 — HIGH — Partial restore commit
No `--single-transaction` or equivalent controlled transaction is required.

### KAI-BACKUPX-022 — HIGH — No reversible precondition
The comment calls restore “safe, reversible,” but the handler takes no fresh pre-restore snapshot and retains no rollback operation.

### KAI-BACKUPX-023 — HIGH — Live database restore
Other services may continue writing while restore applies statements.

### KAI-BACKUPX-024 — HIGH — Missing database provenance
Checksums alone cannot identify the source database/version/schema.

### KAI-BACKUPX-025 — HIGH — Cross-environment restore accepted
The target URI and input file have no compatibility or environment binding.

### KAI-BACKUPX-026 — HIGH — No coherent full snapshot
PostgreSQL, Redis, memU and ledger are captured at different times while writes continue.

### KAI-BACKUPX-027 — HIGH — Manifest is ephemeral
The orchestration response is not written beside the generated files.

### KAI-BACKUPX-028 — HIGH — Missing export checksums
Memory and ledger return no checksum; no full manifest computes one later.

### KAI-BACKUPX-029 — HIGH — Untrusted memory-count request
The record count from one request controls the next retrieval size.

### KAI-BACKUPX-030 — HIGH — Memory completeness unprovable
The export cannot prove it captured all records from one stable revision.

### KAI-BACKUPX-031 — HIGH — Unauthenticated export traffic
The service trusts ordinary HTTP responses from internal names.

### KAI-BACKUPX-032 — HIGH — Missing source attestation
No response identity, digest, schema or revision is verified before writing an export.

---

## Medium-severity findings

### KAI-BACKUPX-033 — MEDIUM — Dump-format metadata absent
Consumers infer format from filename/implementation rather than a manifest.

### KAI-BACKUPX-034 — MEDIUM — No pre-execution file validation
Any bytes are passed to psql after filename/path checks.

### KAI-BACKUPX-035 — MEDIUM — Timeout is not recovery
Killing the client does not prove server-side rollback or restore consistency.

### KAI-BACKUPX-036 — MEDIUM — Unbounded inventory scan
Every file is statted and accumulated before slicing.

### KAI-BACKUPX-037 — MEDIUM — Directory pollution
No component filename/manifest filter exists.

### KAI-BACKUPX-038 — MEDIUM — Weak inventory contract
The listing cannot distinguish complete, partial, verified or restorable artefacts.

### KAI-BACKUPX-039 — MEDIUM — No storage governance
Disk exhaustion remains possible even if persistence is later correctly mounted.

### KAI-BACKUPX-040 — MEDIUM — Unversioned responses
All endpoints return free-form dictionaries.

### KAI-BACKUPX-041 — MEDIUM — Cacheable sensitive metadata
Backup locations and status are returned without privacy cache controls.

### KAI-BACKUPX-042 — MEDIUM — No durable job lifecycle
Long operations remain coupled to one HTTP request.

### KAI-BACKUPX-043 — MEDIUM — Duplicate retry work
No idempotency/correlation identity prevents repeated dumps.

### KAI-BACKUPX-044 — MEDIUM — Restoreability untested
Creation success is never followed by an isolated verification restore.

### KAI-BACKUPX-045 — MEDIUM — Missing operations audit
No immutable record joins actor, job, source revision, files and postconditions.

### KAI-BACKUPX-046 — MEDIUM — Timestamp inconsistency
Some paths use timezone-aware `datetime.now(timezone.utc)`, while others use naive `utcnow()`.

### KAI-BACKUPX-047 — MEDIUM — Non-reproducible service image
Dependencies/base are range/tag based.

### KAI-BACKUPX-048 — MEDIUM — Missing realistic tests
Repository search found only broader hardening tests, not real Backup Service integration coverage.

### KAI-BACKUPX-049 — MEDIUM — Missing lifecycle ownership
No job queue/task group/client/store shutdown model exists.

### KAI-BACKUPX-050 — MEDIUM — Filesystem diagnostics exposure
Permission/path/stat failures are returned through generic exception strings in component/full results.

---

## Batch totals

- Findings: **50**
- Critical: **1**
- High: **31**
- Medium: **18**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,907**
- Critical: **195**
- High: **1,488**
- Medium: **1,221**
- Low: **3**

## Files materially reviewed

`backup-service/app.py`, `backup-service/Dockerfile`, `backup-service/requirements.txt`, Backup/Redis/Postgres deployment in `docker-compose.full.yml`, existing service/CI/off-site backup audits and restore integration assumptions.
