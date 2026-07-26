# Kai Code Audit — Watchdog and Keeper Recovery Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WREC-001 | HIGH | Watchdog persistence does not restore failure counters or prior down state |
| KAI-WREC-002 | HIGH | Watchdog accepts arbitrary service URLs without trust-boundary validation |
| KAI-WREC-003 | MEDIUM | Watchdog treats every HTTP response below 400 as healthy without validating service readiness |
| KAI-WREC-004 | MEDIUM | Watchdog state persistence failures are silent and non-durable |
| KAI-WREC-005 | MEDIUM | Watchdog exposes raw service URLs and network errors in status state |
| KAI-WREC-006 | CRITICAL | Monthly paper backup stores the recovery payload and its decryption words together |
| KAI-WREC-007 | CRITICAL | Monthly paper backup copies the Vault root token into plaintext output |
| KAI-WREC-008 | HIGH | Recovery words and QR material are written to ordinary local files without permission hardening |
| KAI-WREC-009 | HIGH | Paper-backup creation is non-atomic and can overwrite the same-day backup |
| KAI-WREC-010 | HIGH | Test-mode local vault stores secrets in plaintext and silently resets after corruption |
| KAI-WREC-011 | HIGH | USB private-key material is copied and restored without permission enforcement or destination validation |
| KAI-WREC-012 | MEDIUM | Keeper recovery uses a small custom word list with modulo-biased encoding |
| KAI-WREC-013 | MEDIUM | Vault reads convert transport and authorisation failures into missing-secret results |
| KAI-WREC-014 | MEDIUM | Emergency action logging is unsigned, append-only only by convention and not fsynced |

---

## Service watchdog: `agentic/service_watchdog.py`

### KAI-WREC-001 — HIGH — Persisted failure state is not restored
**Issue:** `_load_state` parses saved service entries but executes only `pass`; it restores `last_checked_at` and discards consecutive-failure counters, prior health and `was_down` state.  
**Risk:** A restart clears accumulated failure evidence, delays `SERVICE_DOWN`, and prevents reliable `SERVICE_RESTORED` events despite documentation claiming persistence across restarts.  
**Recommendation:** Restore validated per-service state transactionally and test restart continuity.  
**Status:** OPEN

### KAI-WREC-002 — HIGH — Arbitrary service URLs can be probed
**Issue:** `check_all` accepts a caller-supplied list of service dictionaries and passes each URL directly to `httpx.Client.get`. No scheme, host, network-range or registry allowlist validation is applied.  
**Risk:** Any reachable caller path into this method can turn the watchdog into an SSRF/network-discovery primitive, including probes of local, cloud-metadata or privileged internal endpoints.  
**Recommendation:** Restrict checks to a fixed authenticated registry and validate canonical destinations before connection.  
**Status:** OPEN

### KAI-WREC-003 — MEDIUM — HTTP status alone defines health
**Issue:** Any response with status below 400 is marked healthy; response schema, dependency readiness and explicit service status are ignored.  
**Risk:** Redirects, degraded payloads and semantically failed health responses can reset failure counters and trigger false restoration.  
**Recommendation:** Require endpoint-specific readiness contracts and disable redirects unless explicitly needed.  
**Status:** OPEN

### KAI-WREC-004 — MEDIUM — Watchdog state failures are silent and not fully durable
**Issue:** Save errors are debug-only. Temporary-file replacement has no lock or fsync, and state is process-local.  
**Risk:** Acknowledged monitoring state may disappear, race across workers or become stale without affecting operation.  
**Recommendation:** Use shared transactional state or a monitored single-writer store with durable commits.  
**Status:** OPEN

### KAI-WREC-005 — MEDIUM — Status state exposes network details
**Issue:** `CheckResult` retains raw service URLs and truncated exception strings, and `status()` returns every result.  
**Risk:** Consumers can receive internal topology, hostnames, ports and network-error detail.  
**Recommendation:** Restrict diagnostics, return opaque service IDs publicly and retain detailed errors only in protected telemetry.  
**Status:** OPEN

---

## Keeper recovery: `scripts/monthly_paper_backup.py`, `scripts/kai_control.py`

### KAI-WREC-006 — CRITICAL — Recovery payload and decryption words are stored together
**Issue:** `generate_paper_recovery` encrypts the recovery seed using a key derived from the generated phrase. `monthly_paper_backup.py` then writes both `payload` and `words` into the same plaintext JSON file.  
**Risk:** Anyone obtaining the single backup file can derive the AES key, decrypt the seed and recreate private keeper material. The encryption provides no separation against file compromise.  
**Recommendation:** Separate recovery factors physically and cryptographically; never store the unlocking phrase with the encrypted payload.  
**Status:** OPEN — immediate remediation required

### KAI-WREC-007 — CRITICAL — Vault root token is copied into plaintext backup
**Issue:** The monthly backup reads `VAULT_ROOT_TOKEN` and writes it directly as `vault_root_token` in the recovery JSON.  
**Risk:** Theft or accidental disclosure of the backup grants root-level Vault access in addition to keeper recovery material.  
**Recommendation:** Never export a live root token. Use short-lived recovery procedures, sealed break-glass credentials and immediate revocation controls.  
**Status:** OPEN — immediate remediation required

### KAI-WREC-008 — HIGH — Recovery material lacks filesystem hardening
**Issue:** Recovery words are written to `~/.kai-control/recovery_words.txt`, QR data to `recovery_qr.png`, and backup JSON to an ordinary directory without explicit restrictive creation modes, ownership checks or encryption at rest.  
**Risk:** Other local users, backup agents, malware or permissive umasks may expose recovery secrets.  
**Recommendation:** Generate directly onto controlled offline media using restrictive permissions and securely erase transient files.  
**Status:** OPEN

### KAI-WREC-009 — HIGH — Paper backup write is non-atomic and same-day destructive
**Issue:** The filename contains only the date, and `write_text` directly overwrites it. There is no exclusive create, atomic replacement, fsync or integrity signature.  
**Risk:** Repeated runs overwrite previous recovery material; interruption can leave a partial file and there is no authenticated way to detect alteration.  
**Recommendation:** Use immutable unique generations, signed manifests and verified durable writes.  
**Status:** OPEN

### KAI-WREC-010 — HIGH — Test-mode local vault is plaintext and corruption resets state
**Issue:** Test mode stores all secret values in `local_vault.json` using an unlocked read-modify-write. Parse failure becomes `{}` and the next write replaces the damaged store.  
**Risk:** Development or accidentally enabled test deployments expose secrets, lose concurrent updates and erase forensic evidence after corruption.  
**Recommendation:** Isolate test credentials, reject test mode outside an explicit profile and use transactional protected storage.  
**Status:** OPEN

### KAI-WREC-011 — HIGH — USB private keys are written without destination protection
**Issue:** Primary private material is copied with `shutil.copy2` or written directly to caller-selected mounts. The code does not validate filesystem type, encryption, mount ownership, symlink behaviour or restrictive file permissions.  
**Risk:** Keeper private material can be placed on an attacker-controlled or insecure destination, exposed through inherited permissions, or redirected through filesystem tricks.  
**Recommendation:** Validate trusted removable media, reject symlinks, create files exclusively with strict modes and verify copies after durable flush.  
**Status:** OPEN

### KAI-WREC-012 — MEDIUM — Custom recovery words are modulo-biased
**Issue:** Each seed byte is mapped using `b % len(WORDLIST)` over a 48-word custom list. Since 256 is not divisible by 48, words are not uniformly distributed.  
**Risk:** The phrase has less cleanly characterised entropy than a standard mnemonic construction and lacks checksum/error-detection semantics.  
**Recommendation:** Use a reviewed standard such as a properly implemented BIP-39-style mnemonic or Shamir-based recovery scheme appropriate to the key material.  
**Status:** OPEN

### KAI-WREC-013 — MEDIUM — Vault failures are treated as absent values
**Issue:** `vault_read` catches all request, authentication, decoding and server failures and returns `None`.  
**Risk:** Control logic cannot distinguish “secret does not exist” from “Vault is unreachable or denied”, producing misleading key-presence and setup decisions.  
**Recommendation:** Return typed failure states and fail closed for security decisions.  
**Status:** OPEN

### KAI-WREC-014 — MEDIUM — Emergency action log lacks integrity protection
**Issue:** Emergency actions are appended as plain text with a constant-looking `sig` label supplied by the caller. There is no cryptographic signature, lock, fsync or external anchoring.  
**Risk:** Logs can be altered, truncated, interleaved or fabricated while appearing authoritative.  
**Recommendation:** Record signed immutable events in an independently anchored audit store.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **2**
- High: **6**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **183**
- Critical: **24**
- High: **83**
- Medium: **75**
- Low: **1**

## Files materially reviewed in this batch

`agentic/service_watchdog.py`, `scripts/monthly_paper_backup.py`, `scripts/kai_control.py`.
