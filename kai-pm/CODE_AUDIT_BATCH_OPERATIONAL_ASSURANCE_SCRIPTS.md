# Kai Code Audit — Operational Assurance Scripts Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/init_memu_db.py`, `scripts/auto_session_log.py`, `scripts/hmac_migration_advisor.py`, `scripts/gameday_scorecard.py` and `scripts/phase1_closure_check.py`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-OPSCRIPT-001 | HIGH | Database initialisation falls back to a known `keeper:localdev` credential |
| KAI-OPSCRIPT-002 | HIGH | The complete PostgreSQL URI, including password, is printed to stdout |
| KAI-OPSCRIPT-003 | HIGH | The initialiser creates an obsolete minimal `memories` schema incompatible with current memU fields |
| KAI-OPSCRIPT-004 | HIGH | `CREATE TABLE IF NOT EXISTS` cannot migrate or repair an existing incompatible schema |
| KAI-OPSCRIPT-005 | HIGH | Database schema creation has no migration version, lock or authoritative revision |
| KAI-OPSCRIPT-006 | HIGH | The initialiser performs privileged extension creation without a least-privilege role boundary |
| KAI-OPSCRIPT-007 | MEDIUM | Database connections and cursors are not protected by context managers or `finally` cleanup |
| KAI-OPSCRIPT-008 | MEDIUM | No SSL requirement or server identity policy is enforced in the fallback connection |
| KAI-OPSCRIPT-009 | MEDIUM | The schema creates no retrieval, user, timestamp or vector indexes |
| KAI-OPSCRIPT-010 | MEDIUM | Success is printed without verifying columns, extension version or current memU compatibility |
| KAI-OPSCRIPT-011 | HIGH | Session logging’s date-only idempotency drops every later session/commit on the same day |
| KAI-OPSCRIPT-012 | HIGH | The next-day query includes the previous logged date from midnight and can duplicate earlier commits |
| KAI-OPSCRIPT-013 | HIGH | Git commit subjects are inserted into Markdown without escaping or trust boundaries |
| KAI-OPSCRIPT-014 | HIGH | The complete backlog file is rewritten non-atomically without a lock or backup |
| KAI-OPSCRIPT-015 | HIGH | Git subprocesses have no timeout and may hang automation indefinitely |
| KAI-OPSCRIPT-016 | HIGH | Git return codes and stderr are ignored, so command failure is reported as “no commits” success |
| KAI-OPSCRIPT-017 | HIGH | Concurrent logger runs can both pass the idempotency check and overwrite each other |
| KAI-OPSCRIPT-018 | MEDIUM | Local host time determines day boundaries rather than an explicit repository/operator timezone |
| KAI-OPSCRIPT-019 | MEDIUM | “Most recent session date” is the last matching heading, not the greatest parsed date |
| KAI-OPSCRIPT-020 | MEDIUM | `already_logged_today()` matches any heading containing today’s date, not an auto-session record |
| KAI-OPSCRIPT-021 | MEDIUM | Commit subjects are emitted newest-first, reversing the natural session chronology |
| KAI-OPSCRIPT-022 | MEDIUM | Commit count, backlog size and generated block size are unbounded |
| KAI-OPSCRIPT-023 | MEDIUM | Dry-run prints potentially sensitive commit messages to stdout |
| KAI-OPSCRIPT-024 | MEDIUM | File reads/writes omit explicit encoding and durable fsync semantics |
| KAI-OPSCRIPT-025 | HIGH | The HMAC migration adviser defaults to three services although the repository deploys far more |
| KAI-OPSCRIPT-026 | HIGH | One recorded HMAC security incident still produces “STAY ON HMAC” when no other signal triggers |
| KAI-OPSCRIPT-027 | HIGH | Security recommendations rely entirely on unauthenticated environment assertions |
| KAI-OPSCRIPT-028 | HIGH | Invalid environment values silently revert to reassuring defaults |
| KAI-OPSCRIPT-029 | HIGH | A simple unweighted count treats trivial scale signals as equivalent to real incidents or third-party trust |
| KAI-OPSCRIPT-030 | HIGH | A high `AUDITABILITY_SCORE` is treated as pressure to migrate, reversing the apparent meaning of the metric |
| KAI-OPSCRIPT-031 | MEDIUM | Negative and non-finite signal values are not rejected |
| KAI-OPSCRIPT-032 | MEDIUM | The adviser inspects no repository topology, live identities, key reuse, secret entropy or rotation failures |
| KAI-OPSCRIPT-033 | MEDIUM | The report has no as-of timestamp, evidence source, configuration digest or persistence |
| KAI-OPSCRIPT-034 | MEDIUM | Every recommendation exits with status zero, preventing enforcement by automation |
| KAI-OPSCRIPT-035 | MEDIUM | Boolean parsing recognises only a small value set and silently treats everything else as false |
| KAI-OPSCRIPT-036 | HIGH | Game-day subprocesses have no per-check timeout |
| KAI-OPSCRIPT-037 | HIGH | The total-duration SLO is evaluated only after every command completes and cannot stop a hang |
| KAI-OPSCRIPT-038 | HIGH | Full stdout and stderr are buffered in memory before only five lines are retained |
| KAI-OPSCRIPT-039 | HIGH | Test subprocesses inherit the complete environment, including service secrets and credentials |
| KAI-OPSCRIPT-040 | HIGH | Report tails may persist credentials or sensitive diagnostics in `output/gameday_scorecard.json` |
| KAI-OPSCRIPT-041 | HIGH | Mutable Makefile targets and PATH-resolved executables define the security test suite at runtime |
| KAI-OPSCRIPT-042 | HIGH | A zero exit code is treated as proof of hardening without validating what each check actually exercised |
| KAI-OPSCRIPT-043 | HIGH | The scorecard is unsigned and omits source commit, configuration and test-artefact digests |
| KAI-OPSCRIPT-044 | MEDIUM | Thresholds accept negative, non-finite and impossible values |
| KAI-OPSCRIPT-045 | MEDIUM | Wall-clock time is used for check and total durations |
| KAI-OPSCRIPT-046 | MEDIUM | All checks receive equal weight regardless of security criticality or coverage |
| KAI-OPSCRIPT-047 | MEDIUM | A missing executable raises before a complete failure report is written |
| KAI-OPSCRIPT-048 | MEDIUM | The scorecard output is overwritten non-atomically and has no retention/history |
| KAI-OPSCRIPT-049 | MEDIUM | Relative output and command paths depend on the caller’s working directory and environment |
| KAI-OPSCRIPT-050 | HIGH | Phase-1 closure validates an obsolete `docker-compose.sovereign.yml` rather than current full/minimal deployment files |
| KAI-OPSCRIPT-051 | HIGH | YAML dependencies are detected with raw substring searches that comments or unrelated lists can satisfy |
| KAI-OPSCRIPT-052 | HIGH | Health readiness is declared from the mere presence of the word `healthcheck:` |
| KAI-OPSCRIPT-053 | HIGH | The script reports every patch set closed from a very small set of static string checks |
| KAI-OPSCRIPT-054 | MEDIUM | The compose file is parsed with a fragile regular expression rather than a YAML parser |
| KAI-OPSCRIPT-055 | MEDIUM | A TODO comment is treated as a mandatory closure control |
| KAI-OPSCRIPT-056 | MEDIUM | Required scripts are checked only for filename existence, not content, permissions or execution success |
| KAI-OPSCRIPT-057 | MEDIUM | Network isolation, authentication, secrets, ports, volumes and resource limits are not checked |
| KAI-OPSCRIPT-058 | MEDIUM | The checker stops at the first failure and does not produce a complete gap report |
| KAI-OPSCRIPT-059 | MEDIUM | The successful report lacks source commit, compose digest and runtime verification evidence |

---

## Database initialisation — `scripts/init_memu_db.py`

### KAI-OPSCRIPT-001 — HIGH — Known fallback credential
The script defaults to `postgresql://keeper:localdev@postgres:5432/sovereign` whenever `PG_URI` is missing.

### KAI-OPSCRIPT-002 — HIGH — Credential disclosure
It prints the complete connection string before connecting.

### KAI-OPSCRIPT-003 — HIGH — Current schema incompatibility
The script creates only `id`, text timestamp/event type, content, embedding, relevance and pinned columns. Current memU persistence uses additional importance, access, stability, trust/source, poison/quarantine and other fields.

### KAI-OPSCRIPT-004 — HIGH — No migration behaviour
`IF NOT EXISTS` leaves an old table untouched and falsely permits the script to print “schema created”.

### KAI-OPSCRIPT-005 — HIGH — No schema revision authority
There is no migration ID/table, advisory lock, expected prior version or forward/backward migration contract.

### KAI-OPSCRIPT-006 — HIGH — Privileged DDL coupling
The same runtime URI is expected to create the `vector` extension, encouraging a broadly privileged application/database role.

### KAI-OPSCRIPT-007 — MEDIUM — Incomplete cleanup
Connection/cursor closure is skipped on exceptions and no explicit rollback occurs.

### KAI-OPSCRIPT-008 — MEDIUM — Transport policy absent
The fallback URI specifies no SSL mode, certificate or expected server identity.

### KAI-OPSCRIPT-009 — MEDIUM — Missing indexes
No indexes support user/timestamp/category/vector retrieval or maintenance paths.

### KAI-OPSCRIPT-010 — MEDIUM — No postcondition validation
The script does not introspect the resulting table or run a compatible insert/select.

---

## Session backlog generation — `scripts/auto_session_log.py`

### KAI-OPSCRIPT-011 — HIGH — Same-day work disappears
Once any heading for today exists, every later run skips, even after new commits.

### KAI-OPSCRIPT-012 — HIGH — Date-boundary duplication
`--since=<last date>T00:00:00` includes all commits from the already logged day.

### KAI-OPSCRIPT-013 — HIGH — Markdown/content injection
Commit subjects can add links, HTML, formatting and misleading checklist text to the operational backlog.

### KAI-OPSCRIPT-014 — HIGH — Unsafe complete-file rewrite
The script reads and replaces the complete file directly, with no temporary file, fsync, lock or compare-and-swap.

### KAI-OPSCRIPT-015 — HIGH — Unbounded Git execution
Neither Git command has a timeout.

### KAI-OPSCRIPT-016 — HIGH — Git failure looks empty
Return code/stderr are ignored and the run exits successfully with “no commits”.

### KAI-OPSCRIPT-017 — HIGH — Check/write race
Idempotency checking and writing are separate unsynchronised operations.

### KAI-OPSCRIPT-018 — MEDIUM — Ambiguous timezone
`datetime.now()` uses the executing host’s timezone.

### KAI-OPSCRIPT-019 — MEDIUM — File-order date selection
The final regex match is assumed to be newest.

### KAI-OPSCRIPT-020 — MEDIUM — Broad duplicate detection
Any `## YYYY-MM-DD` substring suppresses the auto-session.

### KAI-OPSCRIPT-021 — MEDIUM — Reverse chronology
Git’s default newest-first output is appended without reversal.

### KAI-OPSCRIPT-022 — MEDIUM — Unbounded generated history
No maximum commit count or file/block size exists.

### KAI-OPSCRIPT-023 — MEDIUM — Dry-run disclosure
All selected commit subjects are printed.

### KAI-OPSCRIPT-024 — MEDIUM — Weak file contract
Read/write encoding is platform-default and no durable-write guarantee exists.

---

## HMAC migration adviser — `scripts/hmac_migration_advisor.py`

### KAI-OPSCRIPT-025 — HIGH — Unsafe topology default
Absent configuration assumes only three services, materially understating the current system.

### KAI-OPSCRIPT-026 — HIGH — Incident does not force migration/preparation
One security incident contributes only one point; a score below two explicitly recommends staying on shared HMAC.

### KAI-OPSCRIPT-027 — HIGH — Self-asserted evidence
Every signal is an environment variable with no authenticated source.

### KAI-OPSCRIPT-028 — HIGH — Invalid data is hidden
Non-integer/non-float input silently becomes the default rather than failing the assessment.

### KAI-OPSCRIPT-029 — HIGH — Unweighted security scoring
Service count, team count, rotations, incidents, external trust and zero-trust mandate each count exactly one.

### KAI-OPSCRIPT-030 — HIGH — Auditability direction ambiguity
A score of 0.8 or more is called “auditability pressure” and increases migration urgency, although higher auditability ordinarily indicates improvement.

### KAI-OPSCRIPT-031 — MEDIUM — Numerical validation absent
Negative counts and NaN/infinite scores are accepted.

### KAI-OPSCRIPT-032 — MEDIUM — Critical properties omitted
The script checks no key sharing map, compromise radius, secret age/entropy, file permissions, replay store, identity binding or rotation failures.

### KAI-OPSCRIPT-033 — MEDIUM — No evidence record
Output is terminal text only, with no timestamp/configuration digest or signed report.

### KAI-OPSCRIPT-034 — MEDIUM — Non-enforcing exit status
Even “MIGRATE NEXT PHASE” returns zero.

### KAI-OPSCRIPT-035 — MEDIUM — Weak boolean contract
Unexpected values are silently false.

---

## Game-day scorecard — `scripts/gameday_scorecard.py`

### KAI-OPSCRIPT-036 — HIGH — No individual deadline
Any command can block forever.

### KAI-OPSCRIPT-037 — HIGH — Total SLO cannot constrain execution
`MAX_TOTAL_DURATION_S` is checked only after the last subprocess.

### KAI-OPSCRIPT-038 — HIGH — Unbounded subprocess buffering
`capture_output=True` retains complete output/error in memory.

### KAI-OPSCRIPT-039 — HIGH — Secret-bearing environment inheritance
Every Make/Python command receives all environment variables unchanged.

### KAI-OPSCRIPT-040 — HIGH — Sensitive report retention
The last five output/error lines are written to an ordinary JSON file.

### KAI-OPSCRIPT-041 — HIGH — Mutable test authority
The executed behaviour depends on the current Makefile, PATH and Python environment rather than immutable reviewed artefacts.

### KAI-OPSCRIPT-042 — HIGH — Exit code substitutes for evidence
No expected assertion count, coverage, endpoint identity or result schema is checked.

### KAI-OPSCRIPT-043 — HIGH — Unattested assurance result
The report can be modified and cannot prove which source/configuration produced it.

### KAI-OPSCRIPT-044 — MEDIUM — Unsafe SLO values
Negative/NaN/infinite values can make the SLO trivially pass or behave inconsistently.

### KAI-OPSCRIPT-045 — MEDIUM — Non-monotonic timing
Durations use `time.time()`.

### KAI-OPSCRIPT-046 — MEDIUM — Equal-weight scoring
A trivial smoke test and a security-critical Gate test each contribute one check.

### KAI-OPSCRIPT-047 — MEDIUM — Incomplete failure reporting
Process-launch errors abort the script before report construction.

### KAI-OPSCRIPT-048 — MEDIUM — Non-atomic latest-only report
The same path is overwritten without history or temporary-file replacement.

### KAI-OPSCRIPT-049 — MEDIUM — Caller-context dependence
Relative paths and PATH resolution change behaviour across working directories/hosts.

---

## Phase-1 closure checker — `scripts/phase1_closure_check.py`

### KAI-OPSCRIPT-050 — HIGH — Checks the wrong deployment authority
The active repository uses full/minimal compose variants, while this checker targets `docker-compose.sovereign.yml`.

### KAI-OPSCRIPT-051 — HIGH — Text-substring dependency proof
A comment or unrelated YAML list containing `- postgres` can satisfy the requirement.

### KAI-OPSCRIPT-052 — HIGH — Healthcheck presence equals readiness
No command, interval, dependency or semantic health result is validated.

### KAI-OPSCRIPT-053 — HIGH — Unsupported closure claim
All patch sets A–F are marked closed despite checking only three dependencies, four strings, one comment and two filenames.

### KAI-OPSCRIPT-054 — MEDIUM — Regex YAML parsing
Nested YAML, anchors, profiles, indentation changes and comments are not handled safely.

### KAI-OPSCRIPT-055 — MEDIUM — Comment as control
A literal GPU TODO string is treated as closure evidence.

### KAI-OPSCRIPT-056 — MEDIUM — Filename-only script check
Scripts need not be executable, correct or successful.

### KAI-OPSCRIPT-057 — MEDIUM — Major controls omitted
No authentication, secret, port exposure, network, volume, resource or user/capability checks exist.

### KAI-OPSCRIPT-058 — MEDIUM — First-error-only output
Operators receive no complete list of unmet requirements.

### KAI-OPSCRIPT-059 — MEDIUM — Missing provenance
The printed report lacks source SHA, compose SHA and runtime evidence.

---

## Batch totals

- Findings: **59**
- Critical: **0**
- High: **31**
- Medium: **28**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,136**
- Critical: **181**
- High: **1,053**
- Medium: **899**
- Low: **3**

## Files materially reviewed

`scripts/init_memu_db.py`, `scripts/auto_session_log.py`, `scripts/hmac_migration_advisor.py`, `scripts/gameday_scorecard.py`, `scripts/phase1_closure_check.py`, with schema/deployment context confirmed against current memU and compose sources.
