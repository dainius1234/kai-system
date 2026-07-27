# Kai Code Audit — Test Harnesses and Market Cache Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/hardening_smoke.py`, `common/market_cache.py`, `scripts/market_price_cache.py` and `scripts/kai_control_selftest.py`. The underlying memU, Executor and recovery implementations are covered in their own batches; this file records assurance-tool and cache-specific defects only.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HARNESS-001 | HIGH | The hardening smoke test can connect to and mutate the real configured memU database/store |
| KAI-HARNESS-002 | HIGH | The smoke test imports complete service modules and executes their import-time side effects |
| KAI-HARNESS-003 | HIGH | It calls endpoint functions directly and bypasses HTTP authentication, middleware and deployment boundaries |
| KAI-HARNESS-004 | HIGH | It inserts a pinned global `keeper` memory and never deletes it |
| KAI-HARNESS-005 | HIGH | It validates caller-selected keeper pinning and forced relevance 1.0 as a security success |
| KAI-HARNESS-006 | HIGH | It invokes compression against whichever store backend the environment configured |
| KAI-HARNESS-007 | HIGH | It treats an absent Redis audit stream returning `verify_or_halt() == true` as hardening success |
| KAI-HARNESS-008 | HIGH | It enables fake hash embeddings before testing memory behaviour |
| KAI-HARNESS-009 | HIGH | Executor hardening is reduced to checking that `/alive` is callable |
| KAI-HARNESS-010 | HIGH | No isolated database, Redis, filesystem or network namespace is established |
| KAI-HARNESS-011 | MEDIUM | The test uses a fixed historical timestamp unrelated to run identity |
| KAI-HARNESS-012 | MEDIUM | Python optimisation can remove every `assert` and make the smoke test pass silently |
| KAI-HARNESS-013 | MEDIUM | No negative authentication, injection, corruption, rollback or dependency-failure cases are exercised |
| KAI-HARNESS-014 | MEDIUM | The script emits only one success string and no structured evidence report |
| KAI-HARNESS-015 | MEDIUM | Imported module globals remain in-process and can contaminate later tests in the same interpreter |
| KAI-HARNESS-016 | HIGH | The market cache returns fabricated petrol and grocery values as the default operational payload |
| KAI-HARNESS-017 | HIGH | Default source URLs are intentionally invalid, so an unconfigured production run always publishes fabricated data |
| KAI-HARNESS-018 | HIGH | Partial source failure silently mixes live data with fabricated fallback values |
| KAI-HARNESS-019 | HIGH | Source URLs are arbitrary environment values accepted by `urllib.request.urlopen` |
| KAI-HARNESS-020 | HIGH | Network responses are read completely without a byte limit |
| KAI-HARNESS-021 | HIGH | Source HTTP status, content type, JSON schema and field types are not validated |
| KAI-HARNESS-022 | HIGH | Market prices, percentages and trends are accepted without finite/range or cross-field validation |
| KAI-HARNESS-023 | HIGH | Cache writes are non-atomic and unprotected by a lock |
| KAI-HARNESS-024 | HIGH | Cache corruption silently overwrites evidence with fresh fabricated defaults |
| KAI-HARNESS-025 | HIGH | Cached values have no maximum age and may be returned indefinitely stale |
| KAI-HARNESS-026 | MEDIUM | External requests run synchronously and sequentially |
| KAI-HARNESS-027 | MEDIUM | Source failures are swallowed and no per-source availability/error state is returned |
| KAI-HARNESS-028 | MEDIUM | `updated_at` is local wall-clock receipt time rather than source market time |
| KAI-HARNESS-029 | MEDIUM | Cache path is an arbitrary environment-controlled filesystem location |
| KAI-HARNESS-030 | MEDIUM | Cache file permissions, ownership, integrity and fsync are not enforced |
| KAI-HARNESS-031 | MEDIUM | Concurrent refresh/load calls can overwrite or read partial state |
| KAI-HARNESS-032 | MEDIUM | The CLI always prints JSON and exits successfully even when every source failed |
| KAI-HARNESS-033 | MEDIUM | Cache entries retain no source URL, provider identity, retrieval status or content digest |
| KAI-HARNESS-034 | HIGH | `KAI_CONTROL_TEST_MODE` is set only after `kai_control.py` has already been imported |
| KAI-HARNESS-035 | HIGH | Dynamic import can execute production-mode module side effects before test isolation is applied |
| KAI-HARNESS-036 | HIGH | Missing AES-GCM support is replaced with a dummy cipher and still considered a passing recovery test |
| KAI-HARNESS-037 | HIGH | The dummy cipher provides no nonce, AAD or ciphertext integrity validation |
| KAI-HARNESS-038 | HIGH | TPM verification is tested only through forced test mode rather than a real TPM boundary |
| KAI-HARNESS-039 | HIGH | USB discovery and independent-device checks are replaced with caller lambdas over temporary directories |
| KAI-HARNESS-040 | HIGH | The destructive-action test checks key-file presence, not independent cryptographic possession or operator identity |
| KAI-HARNESS-041 | HIGH | Paper recovery is validated only by creating a file, not by verifying exact key identity, permissions and trust state |
| KAI-HARNESS-042 | MEDIUM | The in-memory vault is global class state and is not explicitly reset between runs |
| KAI-HARNESS-043 | MEDIUM | No wrong-key, tampered-payload, reused-word, replay or corrupted-USB cases are tested |
| KAI-HARNESS-044 | MEDIUM | Python optimisation can remove all recovery assertions |
| KAI-HARNESS-045 | MEDIUM | The self-test produces no structured result, source digest or tested-configuration record |
| KAI-HARNESS-046 | MEDIUM | The harness does not test concurrency, rotation, revocation or partial-write recovery |
| KAI-HARNESS-047 | MEDIUM | Temporary USB directories do not model mount ownership, removable-media identity or filesystem semantics |
| KAI-HARNESS-048 | MEDIUM | Monkeypatched global functions/classes remain modified for the life of the imported module |

---

## Hardening smoke test — `scripts/hardening_smoke.py`

### KAI-HARNESS-001 — HIGH — Potential live-store mutation
The script changes only `MEMU_ALLOW_FAKE_EMBEDDINGS`. It does not replace `PG_URI`, vector-store mode, Redis URL or data paths before importing memU and calling `memorize_event()`.

### KAI-HARNESS-002 — HIGH — Import-time service execution
Dynamic imports construct memU/Executor module globals, clients, stores and configuration outside an isolated test process/container contract.

### KAI-HARNESS-003 — HIGH — Deployment controls bypassed
Direct function calls do not exercise ASGI parsing, authentication, middleware, network policy, body limits or service-to-service identity.

### KAI-HARNESS-004 — HIGH — Persistent test record
A pinned `keeper` record is written and no record ID is captured/deleted.

### KAI-HARNESS-005 — HIGH — Insecure behaviour asserted as correct
The test explicitly requires a caller-created keeper pin to force relevance 1.0.

### KAI-HARNESS-006 — HIGH — Environment-selected compression
`memu.store.compress()` executes on the active backend, which may be persistent or destructive depending on environment/code version.

### KAI-HARNESS-007 — HIGH — Audit fail-open certified
The test creates `AuditStream(..., redis_url='')` and asserts that verification returns true.

### KAI-HARNESS-008 — HIGH — Fake retrieval semantics
Hash embeddings are deliberately enabled while the test is labelled hardening behaviour.

### KAI-HARNESS-009 — HIGH — Executor assurance is meaningless
`callable(executor.alive)` proves only that a Python function exists.

### KAI-HARNESS-010 — HIGH — Missing sandbox
Database, Redis, paths, environment, network and imported modules are not isolated.

### KAI-HARNESS-011 — MEDIUM — Non-unique timestamp
Every run uses `2026-01-01T00:00:00`.

### KAI-HARNESS-012 — MEDIUM — Assertions removable
`python -O` removes the test logic and still reaches the success print.

### KAI-HARNESS-013 — MEDIUM — No adversarial cases
The harness contains only positive-path checks.

### KAI-HARNESS-014 — MEDIUM — No assurance artefact
It writes no JSON report, tested revision or evidence.

### KAI-HARNESS-015 — MEDIUM — Global contamination
Imported test modules remain registered in `sys.modules` with modified state.

---

## Market cache — `common/market_cache.py`, `scripts/market_price_cache.py`

### KAI-HARNESS-016 — HIGH — Fabricated market data
Fallback contains exact petrol price, tomorrow trend and grocery percentage presented in ordinary market fields.

### KAI-HARNESS-017 — HIGH — Fabrication is the default deployment mode
Both default URLs use `example.invalid`.

### KAI-HARNESS-018 — HIGH — Mixed provenance
One successful provider and one failed provider produce a single normal payload with no per-field distinction.

### KAI-HARNESS-019 — HIGH — Arbitrary source schemes/targets
`urlopen()` receives environment strings without an HTTP/HTTPS public-host allowlist.

### KAI-HARNESS-020 — HIGH — Unbounded response materialisation
`r.read()` has no maximum.

### KAI-HARNESS-021 — HIGH — No response contract
Any JSON object is accepted; status/media type/required fields are unchecked.

### KAI-HARNESS-022 — HIGH — Invalid numerical market state
NaN, infinity, negatives, strings and impossible trends can be cached.

### KAI-HARNESS-023 — HIGH — Unsafe cache replacement
The full JSON file is written directly with no temporary file or lock.

### KAI-HARNESS-024 — HIGH — Corruption becomes fake freshness
A parse error calls `refresh_cache()`, overwrites the damaged file and returns fabricated values stamped current.

### KAI-HARNESS-025 — HIGH — No staleness enforcement
`load_cache()` returns any parseable file regardless of age.

### KAI-HARNESS-026 — MEDIUM — Blocking serial network
Petrol and grocery requests run one after another in the caller thread.

### KAI-HARNESS-027 — MEDIUM — Failure state hidden
All exceptions are ignored.

### KAI-HARNESS-028 — MEDIUM — Wrong timestamp semantics
`updated_at` records local refresh completion, not provider event/as-of time.

### KAI-HARNESS-029 — MEDIUM — Arbitrary storage destination
`MARKET_CACHE_PATH` is not constrained to an approved data root.

### KAI-HARNESS-030 — MEDIUM — Weak file security/durability
No permissions, owner, integrity digest or fsync contract exists.

### KAI-HARNESS-031 — MEDIUM — Concurrent corruption risk
Refresh and load have no shared lock or revision.

### KAI-HARNESS-032 — MEDIUM — False CLI success
The command prints fallback JSON and exits zero after complete provider failure.

### KAI-HARNESS-033 — MEDIUM — Missing provenance
Cache fields contain no provider URL/name, status, source timestamp or response digest.

---

## Recovery self-test — `scripts/kai_control_selftest.py`

### KAI-HARNESS-034 — HIGH — Test mode activated too late
The target module executes before `KAI_CONTROL_TEST_MODE=true` is assigned.

### KAI-HARNESS-035 — HIGH — Production import side effects
Any import-time filesystem, hardware, key or environment behaviour occurs under the caller’s actual settings.

### KAI-HARNESS-036 — HIGH — Missing crypto dependency passes
When `AESGCM` is absent, the harness replaces it rather than failing the security test.

### KAI-HARNESS-037 — HIGH — Dummy cipher is not authenticated encryption
It appends key bytes and strips them during decrypt, ignoring nonce, AAD, tampering and key correctness.

### KAI-HARNESS-038 — HIGH — TPM boundary untested
`tpm_handle_verified()` is asserted only after test-mode activation.

### KAI-HARNESS-039 — HIGH — USB identity untested
Mount discovery is replaced with arbitrary temporary directories.

### KAI-HARNESS-040 — HIGH — Presence substitutes for possession
The destructive gate test proves only that two paths/files are visible.

### KAI-HARNESS-041 — HIGH — Weak restoration postcondition
Only existence of `kai-primary.pub` is checked.

### KAI-HARNESS-042 — MEDIUM — Shared fake vault state
`_MemVault.data` persists at class level.

### KAI-HARNESS-043 — MEDIUM — Missing negative crypto/recovery tests
No tamper, wrong key, replay, corrupted words or partial media cases exist.

### KAI-HARNESS-044 — MEDIUM — Assertions removable
Optimised Python can skip every validation.

### KAI-HARNESS-045 — MEDIUM — No test evidence record
Only a success string is printed.

### KAI-HARNESS-046 — MEDIUM — Lifecycle cases omitted
Rotation, revocation, concurrency and interrupted writes are not covered.

### KAI-HARNESS-047 — MEDIUM — Temporary directories are not removable media
Filesystem/mount identity, permissions and eject/reconnect semantics are absent.

### KAI-HARNESS-048 — MEDIUM — Monkeypatch leakage
Global target-module functions/classes are replaced without restoration.

---

## Batch totals

- Findings: **48**
- Critical: **0**
- High: **28**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,184**
- Critical: **181**
- High: **1,081**
- Medium: **919**
- Low: **3**

## Files materially reviewed

`scripts/hardening_smoke.py`, `common/market_cache.py`, `scripts/market_price_cache.py`, `scripts/kai_control_selftest.py`, with target behaviour checked against memU, Executor and the existing recovery-control audit.
