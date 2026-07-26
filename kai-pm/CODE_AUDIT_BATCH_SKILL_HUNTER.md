# Kai Code Audit — Skill Hunter Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SKILL-001 | CRITICAL | Unauthenticated callers can create autonomous skill files in the active skills directory |
| KAI-SKILL-002 | CRITICAL | PyPI package existence is misrepresented as package verification and trust |
| KAI-SKILL-003 | HIGH | Unauthenticated callers can disable skills by submitting error reports |
| KAI-SKILL-004 | HIGH | Generated skills instruct runtime package installation without version or hash pinning |
| KAI-SKILL-005 | HIGH | Skill files and metadata are overwritten non-atomically on name collisions |
| KAI-SKILL-006 | HIGH | Skill-acquisition provenance is self-asserted and not cryptographically bound to content |
| KAI-SKILL-007 | MEDIUM | Memory logging is fire-and-forget and HTTP failures are ignored |
| KAI-SKILL-008 | MEDIUM | Skill registry and health metadata are exposed without authentication |
| KAI-SKILL-009 | MEDIUM | Metadata corruption is silently converted into default healthy state |
| KAI-SKILL-010 | MEDIUM | Error counting is non-atomic and loses concurrent updates |
| KAI-SKILL-011 | MEDIUM | Capability-gap input lacks a maximum length |
| KAI-SKILL-012 | MEDIUM | Package discovery depends on live external PyPI access without policy control |
| KAI-SKILL-013 | MEDIUM | Disable threshold configuration is not validated |

---

## Skill Hunter: `skill-hunter/app.py`

### KAI-SKILL-001 — CRITICAL — Unauthenticated autonomous skill creation
**Issue:** `POST /hunt` requires no authentication or authorisation. A caller-controlled capability-gap string is converted into a Markdown skill file under `SKILLS_DIR`, which is the system’s active skills location.  
**Risk:** Any network-reachable caller can alter the assistant’s capability registry and create persistent instructions presented as learned skills. This crosses directly from untrusted network input into executable/operational guidance.  
**Recommendation:** Restrict skill creation to authenticated operators and require quarantine, review, signing and explicit activation before generated content enters the active registry.  
**Status:** OPEN — immediate remediation required

### KAI-SKILL-002 — CRITICAL — Package existence is labelled as verification
**Issue:** `_pypi_exists` only checks whether `https://pypi.org/pypi/{package}/json` returns HTTP 200. Generated front matter then states `pypi_verified: true`, and metadata repeats that assertion. No publisher identity, package ownership, version, release age, source code, signatures, hashes, dependency tree, vulnerabilities or malicious-package intelligence are checked.  
**Risk:** The system converts simple registry existence into a trust claim, enabling dependency-confusion, compromised-maintainer and malicious-release risks to be treated as verified capability acquisition.  
**Recommendation:** Do not represent registry existence as verification. Require approved package identities, pinned artefacts, integrity hashes, provenance attestations, vulnerability review and sandboxed evaluation.  
**Status:** OPEN — immediate remediation required

### KAI-SKILL-003 — HIGH — Unauthenticated skill disabling
**Issue:** `POST /skill/{name}/error` allows any caller to increment a skill’s error count and disable it after the configured threshold. There is no evidence requirement, caller identity or deduplication.  
**Risk:** A caller can disable valid capabilities through repeated requests, causing targeted denial of service and corrupting reliability history.  
**Recommendation:** Accept signed execution-result events from authorised runtimes and bind each report to a unique invocation.  
**Status:** OPEN

### KAI-SKILL-004 — HIGH — Generated installation instructions are unpinned
**Issue:** Every generated skill instructs `pip install {package}` without a fixed version, hash, index restriction or reproducible lockfile.  
**Risk:** Future execution installs whichever release is current at that time, making behaviour non-reproducible and exposing the system to later package compromise.  
**Recommendation:** Use reviewed, pinned, hashed artefacts from an internal allowlisted repository.  
**Status:** OPEN

### KAI-SKILL-005 — HIGH — Name collisions overwrite skills and metadata
**Issue:** `_skill_name` truncates normalised gaps to 30 characters. Different requests can map to the same name. `write_text` directly replaces both the skill and metadata files with no existence/version check or atomic transaction.  
**Risk:** A caller can overwrite an existing hunted skill and reset its error/disabled state by submitting a colliding gap. Concurrent hunts can leave mismatched skill and metadata generations.  
**Recommendation:** Use immutable unique IDs, content hashes and transactional versioned writes.  
**Status:** OPEN

### KAI-SKILL-006 — HIGH — Provenance is not integrity-protected
**Issue:** Provenance is plain YAML front matter and a separate JSON sidecar written by the same unauthenticated service. Neither is signed, hashed together or linked to a reviewed artefact.  
**Risk:** Files can be modified independently while retaining trusted-looking provenance labels. The sidecar can disagree with the skill content without detection.  
**Recommendation:** Sign an immutable manifest covering exact content, package artefact and approval state.  
**Status:** OPEN

### KAI-SKILL-007 — MEDIUM — Acquisition memory logging is unreliable
**Issue:** Logging is launched with untracked `asyncio.create_task`. The POST response is not checked, and all exceptions are swallowed.  
**Risk:** A skill can be created without the system memory recording that acquisition, while the API still reports success. Shutdown can abandon the task.  
**Recommendation:** Use a supervised durable outbox and validate acknowledgement.  
**Status:** OPEN

### KAI-SKILL-008 — MEDIUM — Skill inventory is unauthenticated
**Issue:** `/skills` and `/skill/{name}/health` expose package choices, acquisition times, probationary state, error counts and disabled status without access control.  
**Risk:** Callers can enumerate capabilities and identify disabled or weakly trusted components for targeted abuse.  
**Recommendation:** Require scoped operational read access.  
**Status:** OPEN

### KAI-SKILL-009 — MEDIUM — Corrupt metadata fails open
**Issue:** `_load_meta` catches every parse error and returns `{}`. Listing and health endpoints then default `probationary=True`, `disabled=False` and `error_count=0`.  
**Risk:** A corrupted or partially written metadata file is presented as an enabled skill with zero errors rather than an integrity failure.  
**Recommendation:** Fail closed and quarantine skills whose metadata cannot be validated.  
**Status:** OPEN

### KAI-SKILL-010 — MEDIUM — Error increments race
**Issue:** Error reporting performs an unlocked read-modify-write of the sidecar file.  
**Risk:** Concurrent reports overwrite one another, preventing the disable threshold from being reached reliably or producing malformed state.  
**Recommendation:** Use atomic counters or transactional storage.  
**Status:** OPEN

### KAI-SKILL-011 — MEDIUM — Gap input is unbounded
**Issue:** `HuntRequest.gap` has no maximum length. The value is processed, written into skill content and memory metadata, and returned in the response.  
**Risk:** Oversized requests consume memory, disk, log/context capacity and downstream memory storage.  
**Recommendation:** Apply strict schema length and character limits.  
**Status:** OPEN

### KAI-SKILL-012 — MEDIUM — External registry access lacks policy enforcement
**Issue:** Each hunt can make sequential live requests to PyPI. There is no offline catalogue, egress policy, caching, rate limit or audit of registry response provenance.  
**Risk:** Unauthenticated callers can generate repeated external traffic, and results vary with current external registry state.  
**Recommendation:** Use a controlled internal package catalogue and bounded authenticated requests.  
**Status:** OPEN

### KAI-SKILL-013 — MEDIUM — Disable threshold is not validated
**Issue:** `SKILL_DISABLE_THRESHOLD` is parsed directly. Zero or negative values cause immediate disabling on the first report; invalid text crashes startup.  
**Risk:** Misconfiguration can disable all reported skills or prevent service availability.  
**Recommendation:** Validate typed configuration with explicit bounds.  
**Status:** OPEN

---

## Batch totals

- Findings: **13**
- Critical: **2**
- High: **4**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **329**
- Critical: **36**
- High: **134**
- Medium: **156**
- Low: **3**

## Files materially reviewed in this batch

`skill-hunter/app.py`.
