# Kai Code Audit — Keeper Recovery and Operational Scripts

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Existing market-model, Agentic override and service-backup findings are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-OPS-001 | CRITICAL | One USB satisfies both primary and backup key-presence checks |
| KAI-OPS-002 | CRITICAL | Monthly paper backup stores recovery payload, recovery words and Vault root token together |
| KAI-OPS-003 | HIGH | Backup USB is an exact copy rather than an independent recovery factor |
| KAI-OPS-004 | HIGH | TPM verification checks only that a handle exists |
| KAI-OPS-005 | HIGH | USB discovery does not verify removable media or mounted-device identity |
| KAI-OPS-006 | HIGH | Caller-selected mount paths can overwrite arbitrary `kai-primary.*` files |
| KAI-OPS-007 | HIGH | Private key and recovery material are written with ordinary filesystem permissions |
| KAI-OPS-008 | HIGH | Monthly recovery backups are plaintext and unencrypted |
| KAI-OPS-009 | HIGH | Same-day monthly backup runs overwrite the same filename |
| KAI-OPS-010 | HIGH | Environment test mode replaces Vault/TPM controls with plaintext and unconditional success |
| KAI-OPS-011 | HIGH | Vault root-token traffic defaults to unencrypted HTTP |
| KAI-OPS-012 | HIGH | Vault read failure is treated as missing sealed state |
| KAI-OPS-013 | HIGH | Paper restore writes a raw seed instead of reconstructing the TPM-created key object |
| KAI-OPS-014 | HIGH | Paper restore overwrites the authoritative primary public hash without recovery-policy validation |
| KAI-OPS-015 | HIGH | Stored recovery hint is never checked during restoration |
| KAI-OPS-016 | MEDIUM | Recovery payload and word inputs have no length or format bounds |
| KAI-OPS-017 | HIGH | Executor kill logs success even when Docker stop fails |
| KAI-OPS-018 | HIGH | Memory rollback selects `commits[0]` without validating order or target semantics |
| KAI-OPS-019 | HIGH | Rollback sends an unauthenticated POST to an environment-selected service URL |
| KAI-OPS-020 | HIGH | Rollback logs success without validating the response body or restored state |
| KAI-OPS-021 | HIGH | Emergency log `sig` fields are static labels, not cryptographic signatures |
| KAI-OPS-022 | MEDIUM | Emergency action logging is non-atomic, unlocked and not fsynced |
| KAI-OPS-023 | HIGH | Conviction overrides can be created without a verified two-key destructive-action gate |
| KAI-OPS-024 | HIGH | Unbounded substring overrides create broad persistent trust bypasses |
| KAI-OPS-025 | MEDIUM | Override persistence is concurrency-unsafe |
| KAI-OPS-026 | MEDIUM | Vault, accounting and storage paths are weakly validated |
| KAI-OPS-027 | MEDIUM | External commands have no execution timeout |
| KAI-OPS-028 | MEDIUM | USB wait logic depends on non-monotonic wall-clock time |
| KAI-OPS-029 | MEDIUM | Key discovery trusts filenames and a public-file hash only |
| KAI-OPS-030 | MEDIUM | Keeper-control application directory is not permission-hardened |
| KAI-OPS-031 | MEDIUM | Advisor mode displays private accounting totals without an explicit access check |
| KAI-OPS-032 | HIGH | Recovery words and QR payload are persisted in the same local application directory |
| KAI-OPS-033 | HIGH | Market-cache URLs permit arbitrary URL schemes and local/internal reads |
| KAI-OPS-034 | HIGH | Market-cache fetches buffer complete responses without a byte limit |
| KAI-OPS-035 | HIGH | Petrol and grocery payloads have no schema or numerical validation |
| KAI-OPS-036 | HIGH | Failed feeds are replaced with fabricated prices carrying a fresh timestamp |
| KAI-OPS-037 | HIGH | Mixed live and fallback values share one current-looking freshness timestamp |
| KAI-OPS-038 | MEDIUM | Cached market values are loaded without an age/freshness check |
| KAI-OPS-039 | MEDIUM | Market-cache writes are non-atomic and concurrency-unsafe |
| KAI-OPS-040 | MEDIUM | Cache corruption silently triggers replacement with fallback values |
| KAI-OPS-041 | MEDIUM | Market cache can be written to an arbitrary environment-selected path |
| KAI-OPS-042 | MEDIUM | Feed failures and source provenance are suppressed |
| KAI-OPS-043 | MEDIUM | Market-cache CLI prints fallback data as a normal successful result |
| KAI-OPS-044 | HIGH | Git commit messages are inserted into Markdown without escaping |
| KAI-OPS-045 | HIGH | Later commits on an already logged date are permanently skipped |
| KAI-OPS-046 | MEDIUM | “Since last date” includes the boundary date and can duplicate prior commits |
| KAI-OPS-047 | MEDIUM | Git command exit codes are ignored |
| KAI-OPS-048 | MEDIUM | Git failures are reported as “no commits” with exit code zero |
| KAI-OPS-049 | MEDIUM | Session backlog update is a non-atomic whole-file rewrite |
| KAI-OPS-050 | MEDIUM | Session grouping uses local calendar time without an explicit timezone |
| KAI-OPS-051 | HIGH | Hardening smoke uses fake embeddings and in-process imports instead of the deployed trust boundary |
| KAI-OPS-052 | HIGH | Hardening smoke treats an absent Redis-backed audit stream as a successful verification |
| KAI-OPS-053 | HIGH | Keeper-control self-test replaces Vault, TPM and optional encryption with test doubles |
| KAI-OPS-054 | HIGH | Game-day checks have no individual timeout and can hang indefinitely |
| KAI-OPS-055 | MEDIUM | Game-day scorecards persist command output tails without secret redaction or atomic writes |

---

## Keeper recovery/control: `scripts/kai_control.py`, `scripts/monthly_paper_backup.py`

### KAI-OPS-001 — CRITICAL — The two-key gate collapses to one key
**Issue:** `seal_backup()` copies the primary public/private files and stores the copied public hash as the backup hash. The hashes are therefore identical. `key_presence()` checks each discovered USB hash against both expected hashes, so one inserted copy sets both `primary=True` and `backup=True`.  
**Risk:** after backup enrolment, a single USB plus any existing TPM handle unlocks kill and rollback actions despite the stated two-key requirement.  
**Recommendation:** use cryptographically independent keys/shares with distinct identities and require simultaneous authenticated possession.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-002 — CRITICAL — Recovery and Vault root credential are bundled
**Issue:** the monthly backup JSON contains the encrypted recovery payload, all recovery words and `VAULT_ROOT_TOKEN` in one plaintext document.  
**Risk:** possession of one file gives the recovery secret and a root Vault credential, defeating separation and enabling complete secret-store compromise.  
**Recommendation:** never export root tokens; separate recovery shares into independently protected media.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-003 — HIGH — Backup is not independent
The backup USB receives byte-for-byte copies of the same public/private TPM blobs rather than a separate key or threshold share.

### KAI-OPS-004 — HIGH — TPM identity is not bound to the keys
`tpm_handle_verified()` checks only whether `tpm2_readpublic -c 0x81000001` returns zero. It does not compare that handle’s public key/name to the inserted USB or Vault hashes.

### KAI-OPS-005 — HIGH — Directory presence is treated as removable media
Every directory under `/media/$USER` or `/run/media/$USER` is considered a USB mount; device type, mount source, ownership and removable status are not checked.

### KAI-OPS-006 — HIGH — Arbitrary mount write targets
Seal/restore functions accept ordinary `Path` values and write fixed filenames without canonical mount-root or symlink checks.

### KAI-OPS-007 — HIGH — Sensitive file modes are uncontrolled
TPM private blobs, restored raw seeds, recovery words, QR images and logs are written with default process permissions.

### KAI-OPS-008 — HIGH — Paper backup is an ordinary JSON file
No encryption, signature, restrictive mode, atomic creation or secure print-only workflow protects the backup document.

### KAI-OPS-009 — HIGH — Daily filename collision
The filename contains only `YYYYMMDD`; repeat runs overwrite that day’s recovery record without exclusive creation or revision history.

### KAI-OPS-010 — HIGH — Test mode is environment-controlled
`KAI_CONTROL_TEST_MODE=true` makes TPM verification unconditional and stores Vault data in a plaintext local JSON file. No build/test-only enforcement prevents use in a real invocation.

### KAI-OPS-011 — HIGH — Vault root token over HTTP
Default `VAULT_ADDR` is `http://127.0.0.1:8200`; the root token is sent in a header with no TLS or server identity verification.

### KAI-OPS-012 — HIGH — Vault outage looks like absent key enrolment
`vault_read()` catches every error and returns `None`. UI/state logic can therefore prompt resealing and rewrite authoritative hashes during a transient Vault failure.

### KAI-OPS-013 — HIGH — Paper restore produces a different key format
`seal_primary()` creates TPM public/private objects, while `restore_from_paper()` writes `sha256(seed)` as `.pub` and the raw seed as `.priv`. These are not the TPM-created structures expected by the sealing workflow.

### KAI-OPS-014 — HIGH — Restoration redefines the authority
After writing the raw recovery files, restore writes their public hash to `keeper_primary_pubhash` without checking existing policy, backup identity, TPM binding or operator approval.

### KAI-OPS-015 — HIGH — Recovery hint is write-only
Generation stores a payload hash in Vault, but restoration never retrieves or compares it. Any decryptable payload/phrase pair is accepted.

### KAI-OPS-016 — MEDIUM — Unbounded recovery input
Base64 payload and word phrase are accepted without maximum sizes, expected word count or canonical word-list validation.

### KAI-OPS-017 — HIGH — Kill false acknowledgement
`docker compose stop executor` uses `check=False`; its return code is ignored and a successful emergency log entry is always written.

### KAI-OPS-018 — HIGH — Rollback target is ambiguous
The first item in `stats["commits"]` is selected without establishing sort order, current version, ancestry or operator-selected target.

### KAI-OPS-019 — HIGH — Rollback control request is unauthenticated
An environment-selected MEMU URL receives a bare POST to `/revert?version=...`, with no signature, token, nonce or body binding.

### KAI-OPS-020 — HIGH — Rollback success is not established
Any HTTP response body is discarded; memory state is not re-read and compared before a success log is written.

### KAI-OPS-021 — HIGH — Emergency signatures are fictional
`log_action` writes values such as `TPM` or `override by Dainius` into a field named `sig`; no signature or verified actor evidence exists.

### KAI-OPS-022 — MEDIUM — Weak emergency audit persistence
Logs append without locking, flush/fsync, integrity chaining, rotation or restrictive mode.

### KAI-OPS-023 — HIGH — Override creation bypasses destructive gating
`add_conviction_override()` has no key/TPM check. The UI exposes the override button during the “primary sealed, backup not sealed” state, and internal callers can invoke it directly.

### KAI-OPS-024 — HIGH — Persistent broad bypass rules
Complete lower-cased phrases are stored and Agentic later matches rules as substrings. Long/common rules can exempt broad classes of input from low-conviction protection.

### KAI-OPS-025 — MEDIUM — Override append races
Read/deduplicate/append is unsynchronised and lacks atomicity or durable acknowledgement.

### KAI-OPS-026 — MEDIUM — Configuration destinations are untrusted
Vault address, financial files, USB roots and application paths are environment/host controlled without canonical allowlists.

### KAI-OPS-027 — MEDIUM — Commands can hang
TPM, Docker and other subprocess calls have no timeout.

### KAI-OPS-028 — MEDIUM — USB timeout is wall-clock based
Clock changes can shorten or extend waiting loops; environment values are only minimally clamped.

### KAI-OPS-029 — MEDIUM — Filename/hash-only key recognition
Discovery requires two expected filenames and compares only the public file hash; no private-key proof-of-possession is performed.

### KAI-OPS-030 — MEDIUM — Application storage is not hardened
`~/.kai-control` is created without verifying owner, permissions, symlinks or filesystem security.

### KAI-OPS-031 — MEDIUM — Accounting disclosure is unguarded
Advisor mode reads and displays income and suggestions without a key-presence or separate privacy gate.

### KAI-OPS-032 — HIGH — Recovery factors are colocated
Generation saves `recovery_words.txt` and `recovery_qr.png` in the same application directory, making local compromise sufficient to collect both factors.

---

## Market price cache: `common/market_cache.py`, `scripts/market_price_cache.py`

### KAI-OPS-033 — HIGH — Arbitrary URL fetch capability
Environment URLs are passed directly to `urllib.request.urlopen`; no HTTPS-only rule, host allowlist, redirect policy or private/local destination block exists. File/internal URLs that return JSON can be read.

### KAI-OPS-034 — HIGH — Unlimited response buffering
`r.read()` consumes the complete response before JSON parsing, regardless of size.

### KAI-OPS-035 — HIGH — Untrusted data becomes financial context
Any JSON object is accepted for petrol/grocery data; prices, percentages, dates and trend strings are not validated.

### KAI-OPS-036 — HIGH — Fabricated freshness
Both default URLs are intentionally invalid. Failures retain hard-coded petrol/grocery values, then `updated_at` is set to the current time.

### KAI-OPS-037 — HIGH — Partial-source ambiguity
If one feed succeeds and one fails, both values share the same cache-level timestamp and there is no per-field source/fallback marker.

### KAI-OPS-038 — MEDIUM — No cache expiry
`load_cache()` returns any valid JSON regardless of age.

### KAI-OPS-039 — MEDIUM — Unsafe cache commit
The complete file is directly rewritten without a lock, temporary file, fsync or revision check.

### KAI-OPS-040 — MEDIUM — Corruption becomes fabricated data
Any parse/read error triggers refresh, which normally writes current-timestamp fallback values; corruption is not surfaced.

### KAI-OPS-041 — MEDIUM — Arbitrary cache destination
Environment configuration can target any writable path without path/ownership/symlink checks.

### KAI-OPS-042 — MEDIUM — Source failures are invisible
Exceptions are suppressed and no error/source/freshness status is included in the returned payload.

### KAI-OPS-043 — MEDIUM — CLI success is misleading
The CLI prints whatever `refresh_cache()` returns as normal JSON and does not set a failure exit code for complete feed outage/fallback operation.

---

## Session logging: `scripts/auto_session_log.py`

### KAI-OPS-044 — HIGH — Commit-to-Markdown injection
Git commit subjects are inserted verbatim as Markdown list items. Newlines/control syntax in commit messages can create headings, links or altered backlog structure.

### KAI-OPS-045 — HIGH — Daily idempotence loses work
Once any heading containing today’s date exists, later runs skip all commits added during the rest of that day.

### KAI-OPS-046 — MEDIUM — Boundary-day duplication
The script queries from midnight of the last logged date, so commits already represented in that session may be appended again on a later date.

### KAI-OPS-047 — MEDIUM — Git errors are ignored
Return codes and stderr are not checked; failed commands yield an empty list.

### KAI-OPS-048 — MEDIUM — Failure exits successfully
An empty list from a Git error prints “no commits to log” and returns zero.

### KAI-OPS-049 — MEDIUM — Backlog writes race
The entire Markdown file is read, modified and rewritten without locking, atomic replacement or revision checks.

### KAI-OPS-050 — MEDIUM — Timezone is implicit
Session dates use server-local `datetime.now()` and Git boundaries without a configured operator/repository timezone.

---

## Readiness and drill scripts: `scripts/hardening_smoke.py`, `scripts/kai_control_selftest.py`, `scripts/gameday_scorecard.py`

### KAI-OPS-051 — HIGH — Smoke test does not test deployed hardening
It imports applications directly, enables fake embeddings and invokes functions in-process. It does not exercise network exposure, authentication, service identity, real dependencies or container configuration.

### KAI-OPS-052 — HIGH — Missing audit backend is accepted
The test constructs `AuditStream(..., redis_url='')` and asserts `verify_or_halt() is True`, effectively treating absent external audit storage as the expected hardening result.

### KAI-OPS-053 — HIGH — Recovery self-test replaces security controls
Vault reads/writes are monkeypatched to an in-memory dictionary, test mode makes TPM verification true, and a dummy non-authenticating AES implementation may be installed. Passing does not verify real TPM/Vault/recovery security.

### KAI-OPS-054 — HIGH — Game-day can hang forever
Each subprocess has no timeout. The total-duration SLO is checked only after every command returns, so it cannot enforce the stated 120-second limit.

### KAI-OPS-055 — MEDIUM — Scorecard may retain secrets
The final five stdout/stderr lines from every command are written to mutable JSON without redaction, atomic replacement or restricted permissions.

---

## Batch totals

- Findings: **55**
- Critical: **2**
- High: **32**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,358**
- Critical: **105**
- High: **593**
- Medium: **657**
- Low: **3**

## Files materially reviewed in this batch

`scripts/kai_control.py`, `scripts/monthly_paper_backup.py`, `common/market_cache.py`, `scripts/market_price_cache.py`, `scripts/auto_session_log.py`, `scripts/hardening_smoke.py`, `scripts/kai_control_selftest.py`, and `scripts/gameday_scorecard.py`.
