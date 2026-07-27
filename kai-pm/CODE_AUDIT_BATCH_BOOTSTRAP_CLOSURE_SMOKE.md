# Kai Code Audit — Database Bootstrap, Phase Closure and Smoke Checks

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Previously logged health-sweep and false-readiness findings are not duplicated where equivalent; this batch covers the separate shell/static scripts.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BOOT-001 | HIGH | Database bootstrap prints the complete credential-bearing PostgreSQL URI |
| KAI-BOOT-002 | HIGH | Database bootstrap defaults to the hard-coded `keeper:localdev` credential |
| KAI-BOOT-003 | HIGH | PostgreSQL TLS and server identity are not required |
| KAI-BOOT-004 | HIGH | An arbitrary configured database receives extension and schema mutations |
| KAI-BOOT-005 | HIGH | Bootstrap has no schema migration/version authority |
| KAI-BOOT-006 | MEDIUM | Database connection and statements have no bounded timeout |
| KAI-BOOT-007 | MEDIUM | Connection/cursor cleanup is not guaranteed after errors |
| KAI-BOOT-008 | MEDIUM | Memory schema uses weak types and missing integrity constraints |
| KAI-BOOT-009 | MEDIUM | Required query indexes and vector dimensions are absent |
| KAI-BOOT-010 | MEDIUM | Extension creation requires excessive bootstrap privileges |
| KAI-BOOT-011 | HIGH | Phase closure parses Compose using regular expressions rather than YAML semantics |
| KAI-BOOT-012 | HIGH | Dependency checks accept `- service` anywhere in the service block |
| KAI-BOOT-013 | HIGH | Healthcheck checks accept any literal `healthcheck:` text including comments |
| KAI-BOOT-014 | HIGH | The script declares every Phase-1 patch set closed from a small unrelated check set |
| KAI-BOOT-015 | HIGH | Security, authentication, secret, image and network requirements are absent from closure |
| KAI-BOOT-016 | MEDIUM | A specific TODO comment is treated as a closure requirement |
| KAI-BOOT-017 | MEDIUM | Required scripts are checked only for path existence |
| KAI-BOOT-018 | HIGH | Runtime validation is omitted while the report still returns `status: ok` |
| KAI-BOOT-019 | MEDIUM | Valid alternative Compose syntax can fail the text parser |
| KAI-BOOT-020 | MEDIUM | Invalid/comment-only Compose text can satisfy static checks |
| KAI-BOOT-021 | MEDIUM | Closure results are not tied to a Compose digest or repository revision |
| KAI-BOOT-022 | MEDIUM | Missing/unreadable Compose failures produce prose rather than a durable report |
| KAI-BOOT-023 | HIGH | Shell health sweep requests arbitrary configured URLs |
| KAI-BOOT-024 | HIGH | Shell health sweep equates any curl HTTP success with service health |
| KAI-BOOT-025 | HIGH | Shell health sweep omits most deployed services |
| KAI-BOOT-026 | HIGH | Shell health requests have no connection or total timeout |
| KAI-BOOT-027 | HIGH | Shell health checks use unauthenticated plaintext HTTP by default |
| KAI-BOOT-028 | MEDIUM | Health logs can persist credential-bearing URLs |
| KAI-BOOT-029 | MEDIUM | Health log grows without retention or size bounds |
| KAI-BOOT-030 | MEDIUM | Concurrent health sweeps append without locking |
| KAI-BOOT-031 | MEDIUM | Health response identity and body semantics are ignored |
| KAI-BOOT-032 | HIGH | Contract smoke invokes internal APIs without authentication or signed identity |
| KAI-BOOT-033 | HIGH | Contract validation checks only top-level key presence |
| KAI-BOOT-034 | HIGH | Ledger/readiness false values still satisfy the smoke contract |
| KAI-BOOT-035 | HIGH | Contract smoke accepts arbitrary service destinations and has no timeouts |
| KAI-BOOT-036 | MEDIUM | Configured `SESSION_ID` is unused |
| KAI-BOOT-037 | MEDIUM | Complete JSON responses are passed as process command-line arguments |
| KAI-BOOT-038 | MEDIUM | Contract responses are downloaded without byte limits |

---

## Database bootstrap: `scripts/init_memu_db.py`

### KAI-BOOT-001 — HIGH — Database credentials are printed
**Issue:** the script prints `PG_URI` verbatim before connection. PostgreSQL URIs commonly contain username/password.  
**Risk:** credentials enter terminal, CI and scheduler logs.  
**Recommendation:** print only a redacted host/database identity.  
**Status:** OPEN

### KAI-BOOT-002 — HIGH — Known default credential
Absent configuration uses `postgresql://keeper:localdev@postgres:5432/sovereign`.

### KAI-BOOT-003 — HIGH — No TLS requirement
The URI is accepted directly; no `sslmode=verify-full`, CA or host-identity policy is enforced.

### KAI-BOOT-004 — HIGH — Arbitrary privileged mutation target
Any environment-selected PostgreSQL endpoint receives `CREATE EXTENSION` and table DDL. Destination/role/schema approval is not verified.

### KAI-BOOT-005 — HIGH — No migration control
`CREATE TABLE IF NOT EXISTS` does not verify an existing table’s columns, types, indexes or expected version. Incompatible old schemas are reported as created.

### KAI-BOOT-006 — MEDIUM — No timeout policy
Connect and statement timeouts are not configured, so bootstrap can hang on network/locks.

### KAI-BOOT-007 — MEDIUM — Cleanup is not exception-safe
Cursor/connection are not managed by context/finally; errors can leave open transactions/connections.

### KAI-BOOT-008 — MEDIUM — Weak data model
Timestamps are text; content/relevance/pinned permit nulls; no user ID, provenance, uniqueness/version or value checks exist.

### KAI-BOOT-009 — MEDIUM — Missing performance/shape controls
The vector column has no dimension and no vector/timestamp/event indexes are created.

### KAI-BOOT-010 — MEDIUM — Privilege separation is absent
The application URI must possess extension-creation capability or the script fails; no separate controlled migration role is used.

---

## Phase closure: `scripts/phase1_closure_check.py`

### KAI-BOOT-011 — HIGH — Compose is not parsed
**Issue:** service blocks are extracted with regex over raw text. YAML anchors, mappings, profiles, indentation variants and merged configuration are not understood.  
**Risk:** the closure decision can disagree with the configuration Docker actually executes.  
**Recommendation:** parse and normalise Compose using the official model/config output.  
**Status:** OPEN

### KAI-BOOT-012 — HIGH — Dependency false positives
The test only searches for `- <dependency>` anywhere in the extracted service text, not specifically within `depends_on` or for readiness conditions.

### KAI-BOOT-013 — HIGH — Healthcheck false positives
Any `healthcheck:` substring, including a comment or irrelevant nested text, passes.

### KAI-BOOT-014 — HIGH — Closure is materially overclaimed
The report marks patch sets A–F closed although the script tests only three dependency lists, four strings and two file paths.

### KAI-BOOT-015 — HIGH — Security is outside the closure definition
No check covers authentication, exposed ports, default credentials, secret mounts, TLS, image pinning, privileges, capabilities, read-only filesystems or network segmentation.

### KAI-BOOT-016 — MEDIUM — Comment-driven closure
Presence of an exact `# TODO: enable GPU when core is stable.` comment is treated as required evidence.

### KAI-BOOT-017 — MEDIUM — Script path is enough
Companion scripts need not be executable, valid or contain any test logic.

### KAI-BOOT-018 — HIGH — Static-only result is `ok`
The notes acknowledge runtime health was not tested, but the top-level status is still `ok` and every patch is closed.

### KAI-BOOT-019 — MEDIUM — Format sensitivity
Valid Compose expressed using mappings, anchors, inline lists or different indentation can fail closure.

### KAI-BOOT-020 — MEDIUM — Text can satisfy checks without behaviour
Comments or unused/profile-disabled service fragments can satisfy literal checks even when the active stack lacks the requirement.

### KAI-BOOT-021 — MEDIUM — No revision binding
The report contains no Compose hash, Git commit, generated-config digest or execution environment identity.

### KAI-BOOT-022 — MEDIUM — No durable failure artefact
Read/parse/require failures terminate with prose and do not write a structured scorecard tied to the failed revision.

---

## Shell health sweep: `scripts/health_sweep.sh`

### KAI-BOOT-023 — HIGH — Configurable network probe
Environment values are concatenated with endpoint paths and requested without an approved host/scheme policy.

### KAI-BOOT-024 — HIGH — Transport success is operational health
`curl -f` validates status only; response schema, `status`, readiness, identity and freshness are ignored.

### KAI-BOOT-025 — HIGH — Incomplete fleet coverage
Only Tool Gate, memu-core, executor and dashboard are checked, excluding Agentic, verifier, trust ledger, LLM, databases and many deployed services.

### KAI-BOOT-026 — HIGH — Requests can hang
No `--connect-timeout` or `--max-time` is set.

### KAI-BOOT-027 — HIGH — Unauthenticated HTTP defaults
Every default URL is plaintext localhost HTTP and no caller/service identity is supplied.

### KAI-BOOT-028 — MEDIUM — URL leakage
Complete URLs are written to a persistent log; embedded credentials/query tokens would be retained.

### KAI-BOOT-029 — MEDIUM — Unlimited logging
The log has no rotation, retention, size limit or permission hardening.

### KAI-BOOT-030 — MEDIUM — Concurrent append races
Multiple sweeps can interleave output; no lock or unique run ID exists.

### KAI-BOOT-031 — MEDIUM — Wrong process can pass
Any HTTP responder on the configured port returning success is accepted as the intended service.

---

## Contract smoke: `scripts/contract_smoke.sh`

### KAI-BOOT-032 — HIGH — Contract calls are anonymous
The script invokes route, ledger and dashboard APIs without HMAC, bearer token, nonce or service identity.

### KAI-BOOT-033 — HIGH — Shape-only validation
`check_keys` verifies only that named top-level keys exist. Types, enums, values and nested contracts are not checked.

### KAI-BOOT-034 — HIGH — Failure values pass
A ledger response containing `valid:false` or readiness containing `core_ready:false` still passes because the keys exist.

### KAI-BOOT-035 — HIGH — Arbitrary unbounded requests
Configured destinations have no host/scheme policy and curl has no connect/total timeout.

### KAI-BOOT-036 — MEDIUM — Session setting is dead
`SESSION_ID` defaults to `bootstrap-token-1` but is never used; the request hard-codes `smoke-session`.

### KAI-BOOT-037 — MEDIUM — JSON is exposed through argv
Complete service responses are supplied to `python` as command-line arguments, visible to process inspection and limited by OS argument size.

### KAI-BOOT-038 — MEDIUM — Response allocation is unbounded
Curl captures complete bodies into shell variables before validation.

---

## Batch totals

- Findings: **38**
- Critical: **0**
- High: **21**
- Medium: **17**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,396**
- Critical: **105**
- High: **614**
- Medium: **674**
- Low: **3**

## Files materially reviewed in this batch

`scripts/init_memu_db.py`, `scripts/phase1_closure_check.py`, `scripts/health_sweep.sh`, and `scripts/contract_smoke.sh`.
