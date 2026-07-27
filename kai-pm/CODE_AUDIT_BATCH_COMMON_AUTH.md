# Kai Code Audit — Shared Inter-Service Authentication Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records shared authentication defects not already counted in the original `KAI-AUTH-*`, Tool Gate or key-rotation batches.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-COMAUTH-001 | CRITICAL | Non-strict key-ID verification permits a valid signature to be relabelled with any unrevoked key ID, bypassing revocation |
| KAI-COMAUTH-002 | CRITICAL | Pipe-delimited canonicalisation is ambiguous because fields may contain the delimiter |
| KAI-COMAUTH-003 | CRITICAL | One shared secret allows every holder to forge every actor, session and service identity |
| KAI-COMAUTH-004 | HIGH | Strict key-ID enforcement defaults off |
| KAI-COMAUTH-005 | HIGH | Invalid strict-mode environment values silently disable strict key-ID enforcement |
| KAI-COMAUTH-006 | HIGH | Caller-supplied key IDs are not cryptographically bound to signatures |
| KAI-COMAUTH-007 | HIGH | `sign_gate_request` permits the caller to label a primary-key signature with an arbitrary key ID |
| KAI-COMAUTH-008 | HIGH | Revocation checks trust only the unbound key-ID label rather than the secret actually verifying the digest |
| KAI-COMAUTH-009 | HIGH | No algorithm, protocol or purpose domain separator is included in the signed payload |
| KAI-COMAUTH-010 | HIGH | Secret files are trusted without ownership, permission, symlink or regular-file validation |
| KAI-COMAUTH-011 | HIGH | Secret-file existence and reading are separate time-of-check/time-of-use operations |
| KAI-COMAUTH-012 | HIGH | Secret-file contents are synchronously reread for every sign and verification operation |
| KAI-COMAUTH-013 | HIGH | Secrets have no minimum length, entropy or approved-character policy |
| KAI-COMAUTH-014 | HIGH | Key IDs, revoked IDs and strict-mode state are reread independently and can change during one verification |
| KAI-COMAUTH-015 | HIGH | Primary and secondary key IDs may be equal and are not validated for uniqueness |
| KAI-COMAUTH-016 | HIGH | Dual-signature bundles require only one valid signature and provide no transition quorum semantics |
| KAI-COMAUTH-017 | HIGH | Signatures have no key activation, expiration, compromise time or not-before policy |
| KAI-COMAUTH-018 | HIGH | Authentication successes and failures are not durably audited with the accepted key ID |
| KAI-COMAUTH-019 | HIGH | Signature input strings are not length-bounded or Unicode-normalised before signing/verification |
| KAI-COMAUTH-020 | HIGH | Missing secret files fall back to caller-provided defaults rather than an explicit unavailable state |
| KAI-COMAUTH-021 | MEDIUM | Secret loading strips leading and trailing whitespace and silently changes key bytes |
| KAI-COMAUTH-022 | MEDIUM | Convention-based secret lookup uses the full lowercased environment name and does not match the repository’s normal `hmac_secret` Docker secret name |
| KAI-COMAUTH-023 | MEDIUM | Secret-path warnings disclose exact mounted filesystem locations |
| KAI-COMAUTH-024 | MEDIUM | Secret-file read failures after `is_file()` are not converted into a typed authentication-readiness state |
| KAI-COMAUTH-025 | MEDIUM | Integer timestamp canonicalisation discards sub-second precision |
| KAI-COMAUTH-026 | MEDIUM | Non-finite or non-numeric timestamps can raise during `int(ts)` rather than returning a typed verification failure |
| KAI-COMAUTH-027 | MEDIUM | Revoked-key environment lists are unbounded and unvalidated |
| KAI-COMAUTH-028 | MEDIUM | Signature strings and key-ID prefixes have no maximum length or character grammar |
| KAI-COMAUTH-029 | MEDIUM | UTF-8 text is the only secret representation and binary key material cannot be loaded without transformation |
| KAI-COMAUTH-030 | MEDIUM | The default development-secret warning is process-local and does not expose a machine-enforcing readiness failure |
| KAI-COMAUTH-031 | MEDIUM | Verification returns only Boolean and loses failure reason, key version and policy revision |
| KAI-COMAUTH-032 | MEDIUM | Authentication helpers have no immutable configuration snapshot or lifecycle-managed key reload contract |

---

## Critical findings

### KAI-COMAUTH-001 — CRITICAL — Revocation bypass by key relabelling
**Issue:** `verify_gate_signature()` rejects the supplied label only if that label is revoked. When strict mode is false, it then accepts a digest matching the primary or secondary secret regardless of the label. A valid primary-key signature can therefore be changed from `v1:<digest>` to `attacker:<digest>` and pass even when `v1` is revoked.  
**Risk:** Revocation does not revoke the cryptographic key; it revokes only a mutable unauthenticated string prefix. Compromised keys remain usable.  
**Recommendation:** bind key ID and algorithm/version inside the MAC input and select exactly the corresponding key before verification. Strict identity binding must be mandatory.  
**Status:** OPEN — immediate remediation required

### KAI-COMAUTH-002 — CRITICAL — Ambiguous signed tuple
**Issue:** `_payload()` joins actor, session, tool, nonce and timestamp with `|`, but none of the fields excludes or escapes `|`. Different field tuples can produce the same signed string.  
**Risk:** A valid signature for one identity/session/tool combination may be reinterpreted as another combination with shifted delimiters.  
**Recommendation:** sign a length-prefixed or canonical structured encoding with a schema/version and exact body digest.  
**Status:** OPEN — immediate remediation required

### KAI-COMAUTH-003 — CRITICAL — Shared-secret identity collapse
**Issue:** Every service holding the one HMAC secret can generate a valid signature for any `actor_did`, token/session and tool. No per-service key or delegation certificate exists.  
**Risk:** Compromise or misuse of one low-privilege service grants forgery authority across the entire trust domain.  
**Recommendation:** use service-bound asymmetric or mTLS identities with scoped delegation and independently revocable credentials.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-COMAUTH-004 — HIGH — Strict identity is opt-in
`INTERSERVICE_HMAC_STRICT_KEY_ID` defaults false, enabling the relabelling behaviour by default.

### KAI-COMAUTH-005 — HIGH — Strict-mode typo fails open
Only four truthy strings enable strictness; any misspelling or unsupported value silently becomes false.

### KAI-COMAUTH-006 — HIGH — Key label excluded from MAC
The digest covers only actor/session/tool/nonce/integer timestamp, not key ID.

### KAI-COMAUTH-007 — HIGH — Arbitrary signing label
`sign_gate_request(..., key_id=...)` always signs with the primary secret but emits the caller-selected label.

### KAI-COMAUTH-008 — HIGH — Revocation checks the wrong authority
The verifier does not map the label to one selected key before comparing; it tests all available keys after the label-only revocation check.

### KAI-COMAUTH-009 — HIGH — No domain separation
The signed bytes identify no protocol version, endpoint, HTTP method, audience/service, environment or signature purpose.

### KAI-COMAUTH-010 — HIGH — Untrusted secret source
`Path.is_file()` follows symlinks; file owner/mode/mount and regular-file status are not verified.

### KAI-COMAUTH-011 — HIGH — Secret TOCTOU
A file may be replaced or disappear between `is_file()` and `read_text()`.

### KAI-COMAUTH-012 — HIGH — Synchronous secret I/O in hot paths
Every sign/verify operation may read one or two files, blocking async callers and observing mid-rotation partial state.

### KAI-COMAUTH-013 — HIGH — Weak secrets are accepted
Any non-default string, including a one-character predictable secret, is used without entropy/length checks.

### KAI-COMAUTH-014 — HIGH — Mixed configuration snapshot
Key IDs, strict flag, revoked list and secrets are fetched separately from mutable environment/files and may not represent one atomic revision.

### KAI-COMAUTH-015 — HIGH — Key-ID collision
Primary/secondary IDs can be identical, making strict selection and audit attribution ambiguous.

### KAI-COMAUTH-016 — HIGH — Dual-signing is not quorum
The bundle contains two signatures, but verification accepts either one; no policy requires both, identifies overlap stage or enforces migration completion.

### KAI-COMAUTH-017 — HIGH — No key validity interval
Keys are accepted solely based on current environment presence and label revocation; activation and expiry are absent.

### KAI-COMAUTH-018 — HIGH — No authentication evidence
Helpers return strings/Boolean only and emit no tamper-evident accepted/rejected key/audience/body-digest event.

### KAI-COMAUTH-019 — HIGH — Unbounded canonical fields
Large actor/session/tool/nonce values are concatenated and HMACed with no grammar, byte or Unicode normalisation.

### KAI-COMAUTH-020 — HIGH — Missing-file fallback ambiguity
A missing configured secret file returns the caller-provided default. Depending on the caller, this may activate a known development secret or an empty key instead of a locked unavailable state.

---

## Medium-severity findings

### KAI-COMAUTH-021 — MEDIUM — Key bytes are altered
`.strip()` removes intentional whitespace from file-based secrets.

### KAI-COMAUTH-022 — MEDIUM — Convention mismatch
The automatic path for `INTERSERVICE_HMAC_SECRET` is `/run/secrets/interservice_hmac_secret`, while Compose normally mounts `/run/secrets/hmac_secret`; only explicit path configuration works.

### KAI-COMAUTH-023 — MEDIUM — Secret-path disclosure
Warnings include the complete missing secret path.

### KAI-COMAUTH-024 — MEDIUM — Untyped secret-read failure
Permission, decoding and race errors propagate rather than producing a controlled authentication-readiness result.

### KAI-COMAUTH-025 — MEDIUM — Timestamp precision loss
All times within the same second canonicalise identically.

### KAI-COMAUTH-026 — MEDIUM — Timestamp conversion failures
NaN, infinity and invalid types can raise before a Boolean verification result is produced.

### KAI-COMAUTH-027 — MEDIUM — Revocation-list configuration
The comma-separated list has no item/length/format limit or signed revision.

### KAI-COMAUTH-028 — MEDIUM — Unbounded signature grammar
Very long labels/digests are parsed and compared without a strict hexadecimal/length schema.

### KAI-COMAUTH-029 — MEDIUM — Text-only key material
Secrets are always UTF-8 strings, encouraging Base64/text transformation without explicit encoding metadata.

### KAI-COMAUTH-030 — MEDIUM — Development mode is advisory
Allowing the known dev secret only logs once per process; health/configuration does not necessarily expose a blocked production state.

### KAI-COMAUTH-031 — MEDIUM — Boolean-only verification contract
Callers cannot distinguish revoked, unknown ID, wrong digest, missing secret, malformed signature or configuration error.

### KAI-COMAUTH-032 — MEDIUM — No key lifecycle object
There is no validated immutable keyring snapshot, atomic reload, close/zeroisation or rollout revision shared across callers.

---

## Batch totals

- Findings: **32**
- Critical: **3**
- High: **17**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,596**
- Critical: **196**
- High: **1,305**
- Medium: **1,092**
- Low: **3**

## Files materially reviewed

`common/auth.py`, reconciled against the original authentication register, Tool Gate and both key-rotation batches.
