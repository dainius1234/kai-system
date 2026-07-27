# Kai Code Audit — Backup Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BACKUP-001 | CRITICAL | Unauthenticated callers can restore PostgreSQL from any listed backup file |
| KAI-BACKUP-002 | CRITICAL | Unauthenticated callers can trigger full database, Redis, memory and ledger backups |
| KAI-BACKUP-003 | HIGH | Backup files are created unencrypted with no access-control or retention enforcement |
| KAI-BACKUP-004 | HIGH | Backup inventory discloses absolute paths, filenames, sizes and timestamps without authentication |
| KAI-BACKUP-005 | HIGH | PostgreSQL defaults embed known credentials |
| KAI-BACKUP-006 | HIGH | Database and Redis subprocesses execute synchronously inside async handlers |
| KAI-BACKUP-007 | HIGH | Memory export attempts to retrieve all records into one response and one in-memory JSON object |
| KAI-BACKUP-008 | HIGH | Backup and restore operations have no concurrency control or mutual exclusion |
| KAI-BACKUP-009 | MEDIUM | Redis BGSAVE completion is not awaited before copying `dump.rdb` |
| KAI-BACKUP-010 | MEDIUM | Redis URL parsing ignores passwords, usernames, TLS and Unix-socket forms |
| KAI-BACKUP-011 | MEDIUM | Backup filenames use second-resolution timestamps and can collide |
| KAI-BACKUP-012 | MEDIUM | Subprocess stderr and downstream exception text are exposed to callers |
| KAI-BACKUP-013 | MEDIUM | Full backup runs all components sequentially and can hold one request for several minutes |
| KAI-BACKUP-014 | MEDIUM | Partial/full success semantics can conceal stale or inaccessible backup artefacts |
| KAI-BACKUP-015 | MEDIUM | Health reports ok without checking tools, storage, credentials or dependencies |
| KAI-BACKUP-016 | MEDIUM | Configuration paths, URIs and service URLs are not validated |
| KAI-BACKUP-017 | MEDIUM | No integrity verification is performed before PostgreSQL restore |
| KAI-BACKUP-018 | MEDIUM | Backup list count includes all files while only the first 50 are returned |

---

## Backup service: `backup-service/app.py`

### KAI-BACKUP-001 — CRITICAL — Unauthenticated database restore
**Issue:** `POST /restore/postgres` requires no authentication, authorisation, approval gate or maintenance-mode check. A caller supplies any filename that exists in `BACKUP_DIR`, and the service invokes `psql PG_URI -f <path>`.  
**Risk:** Any reachable caller can overwrite or mutate the configured PostgreSQL database using an existing SQL dump. This is a direct destructive integrity and availability path affecting every service sharing that database.  
**Recommendation:** Require strong administrative authentication, multi-party/explicit approval, maintenance isolation, verified backup identity and a dedicated restricted restore workflow.  
**Status:** OPEN — immediate remediation required

### KAI-BACKUP-002 — CRITICAL — Unauthenticated full backup orchestration
**Issue:** `/backup`, `/backup/full`, `/backup/postgres`, `/backup/redis`, `/backup/memory` and `/backup/ledger` are all unauthenticated.  
**Risk:** Callers can repeatedly force expensive database dumps, Redis saves, memory exports and ledger requests, creating disk exhaustion, load spikes and bulk copies of sensitive system data.  
**Recommendation:** Restrict backup initiation to authenticated operators or a protected scheduler with quotas and audit logging.  
**Status:** OPEN — immediate remediation required

### KAI-BACKUP-003 — HIGH — Sensitive backups are stored unencrypted and ungoverned
**Issue:** SQL, RDB and JSON exports are written directly beneath `BACKUP_DIR` with default process permissions. No encryption, permission hardening, retention, secure deletion or total-size limit is implemented.  
**Risk:** Backups containing database rows, memory records and operational metadata remain readable to any process/user with directory access and can fill the volume indefinitely.  
**Recommendation:** Encrypt backups, enforce restrictive ownership/mode, bounded retention and secure lifecycle controls.  
**Status:** OPEN

### KAI-BACKUP-004 — HIGH — Backup inventory is publicly disclosed
**Issue:** `GET /backup/list` returns backup filenames, absolute filesystem paths, byte sizes and modification times without authentication.  
**Risk:** Callers can enumerate available restore targets, storage layout, backup cadence and data volume, directly supporting exploitation of the restore endpoint.  
**Recommendation:** Require scoped administrative access and avoid returning absolute paths.  
**Status:** OPEN

### KAI-BACKUP-005 — HIGH — Default PostgreSQL credentials are known and weak
**Issue:** `PG_URI` defaults to `postgresql://postgres:postgres@postgres:5432/postgres`.  
**Risk:** Deployments that omit configuration silently use a known superuser-style username/password combination. Other compromised containers on the network may connect directly.  
**Recommendation:** Fail startup when explicit secret-managed credentials are absent; use a least-privilege backup/restore role.  
**Status:** OPEN

### KAI-BACKUP-006 — HIGH — Blocking subprocess and filesystem work runs on the event loop
**Issue:** `pg_dump`, `redis-cli`, `psql`, SHA-256 hashing, file copying and JSON writes execute synchronously inside async request handlers.  
**Risk:** Long-running backup or restore operations block the event-loop worker, including health and other control endpoints.  
**Recommendation:** Execute jobs in isolated bounded workers with asynchronous status tracking.  
**Status:** OPEN

### KAI-BACKUP-007 — HIGH — Memory export is unbounded
**Issue:** The service reads memory stats, then requests `top_k = stats.records` and materialises the entire HTTP response, parsed JSON object and pretty-printed export in memory.  
**Risk:** Large memory stores can exhaust RAM and create very large backup files; a manipulated stats count can amplify the request.  
**Recommendation:** Export through paginated streaming with fixed batch and total limits.  
**Status:** OPEN

### KAI-BACKUP-008 — HIGH — Backup and restore operations can overlap
**Issue:** No lock, job registry or mutual exclusion prevents concurrent dumps, restores and full backups. Timestamp collisions and shared database/storage resources are not protected.  
**Risk:** Concurrent callers can overwrite same-second files, saturate I/O, race BGSAVE copies or restore while backups are being produced.  
**Recommendation:** Use a single authoritative job queue with component-level locking and idempotent job IDs.  
**Status:** OPEN

### KAI-BACKUP-009 — MEDIUM — Redis backup may copy stale/incomplete data
**Issue:** The service issues `BGSAVE` and immediately reads Redis `dir/dump.rdb`. It does not poll `LASTSAVE` or persistence state for completion.  
**Risk:** The copied RDB can be the previous snapshot or be read while background save has not completed, yet is reported as a successful current backup.  
**Recommendation:** Wait for confirmed new save completion and verify resulting file metadata/checksum.  
**Status:** OPEN

### KAI-BACKUP-010 — MEDIUM — Redis connection parsing is incomplete
**Issue:** A regex extracts only `redis://host:port`. Passwords, usernames, database index, `rediss://`, IPv6 and Unix sockets are ignored; parse failure silently falls back to `redis:6379`.  
**Risk:** Backups may target the wrong instance or fail authentication while operators believe the configured URL is honoured.  
**Recommendation:** Use a standard Redis URL parser and pass credentials/TLS securely.  
**Status:** OPEN

### KAI-BACKUP-011 — MEDIUM — Backup filenames can collide
**Issue:** All components use UTC timestamps with one-second resolution. Concurrent or repeated requests in the same second generate identical paths.  
**Risk:** Files can be overwritten or mixed, invalidating manifests and restore assumptions.  
**Recommendation:** Use collision-resistant job IDs and exclusive atomic file creation.  
**Status:** OPEN

### KAI-BACKUP-012 — MEDIUM — Internal diagnostics are exposed
**Issue:** `pg_dump`, Redis and restore stderr plus memory/ledger exception strings are returned directly in HTTP error details or full-backup component results.  
**Risk:** Callers receive database, network, filesystem, credential-policy and command diagnostics.  
**Recommendation:** Return stable error codes and store detailed diagnostics in protected logs.  
**Status:** OPEN

### KAI-BACKUP-013 — MEDIUM — Full backup is a long synchronous orchestration request
**Issue:** `backup_full` awaits PostgreSQL, Redis, memory and ledger operations sequentially. Individual subprocess timeouts total several minutes.  
**Risk:** Clients/proxies may time out while work continues, retries may duplicate jobs and one request monopolises a worker.  
**Recommendation:** Return a durable job ID immediately and process components asynchronously with progress state.  
**Status:** OPEN

### KAI-BACKUP-014 — MEDIUM — Reported success does not prove usable backup
**Issue:** Redis returns `status: ok` when BGSAVE was merely triggered but the RDB is inaccessible. Ledger export stores only `/ledger/stats`, not the ledger contents, despite being described as an export.  
**Risk:** Full backup counts components as succeeded even when no restorable artefact exists or only summary metadata was saved.  
**Recommendation:** Define success as a verified, durable, restorable artefact and test restore integrity.  
**Status:** OPEN

### KAI-BACKUP-015 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns ok without checking backup-directory writability/free space, `pg_dump`, `psql`, `redis-cli`, database access, Redis access, memu-core or tool-gate.  
**Risk:** Orchestration treats the service as ready while every backup/restore path may fail.  
**Recommendation:** Separate liveness from per-component backup and restore readiness.  
**Status:** OPEN

### KAI-BACKUP-016 — MEDIUM — Configuration lacks validation
**Issue:** Backup directory, database URI, Redis URL, service URLs and port are accepted directly. Relative paths, unsafe locations and invalid schemes are not rejected.  
**Risk:** Misconfiguration can write sensitive data to unintended locations, target unintended databases or break operations only at request time.  
**Recommendation:** Validate typed startup configuration and approved path/URL policies.  
**Status:** OPEN

### KAI-BACKUP-017 — MEDIUM — Restore does not verify integrity or provenance
**Issue:** Although backup creation returns a SHA-256 checksum, it is not stored in a manifest used by restore. `restore_postgres` accepts any matching filename and performs no checksum, format, ownership or creation-source verification.  
**Risk:** A corrupted or replaced SQL file can be restored as trusted backup content.  
**Recommendation:** Require signed manifests and verify checksum, format, ownership and expected database metadata before restore.  
**Status:** OPEN

### KAI-BACKUP-018 — MEDIUM — Backup listing is internally inconsistent
**Issue:** The endpoint returns only `files[:50]` but sets `total` to the count of all files, with no pagination token or offset.  
**Risk:** Consumers cannot enumerate or select older backups reliably and may misinterpret the returned list as complete.  
**Recommendation:** Implement explicit pagination and deterministic metadata.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **2**
- High: **6**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **549**
- Critical: **61**
- High: **196**
- Medium: **289**
- Low: **3**

## Files materially reviewed in this batch

`backup-service/app.py`.
