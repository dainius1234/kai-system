# Kai Code Audit — Skill Hunter Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SKILL-001 | CRITICAL | Unauthenticated callers can create or overwrite persistent skill files in the active skills directory |
| KAI-SKILL-002 | CRITICAL | PyPI package existence is misrepresented as package verification and trust |
| KAI-SKILL-003 | CRITICAL | Unauthenticated skill hunts inject acquisition claims into long-term memory |
| KAI-SKILL-004 | HIGH | Unauthenticated callers can disable skills by submitting repeated error reports |
| KAI-SKILL-005 | HIGH | Generated skills instruct installation of unpinned latest packages |
| KAI-SKILL-006 | HIGH | “Disabled” skills remain present and no enforcement prevents their use |
| KAI-SKILL-007 | HIGH | Sanitised/truncated skill-name collisions overwrite existing skill and metadata files |
| KAI-SKILL-008 | HIGH | Skill and metadata persistence is non-atomic and lacks concurrency control |
| KAI-SKILL-009 | HIGH | Skill-acquisition provenance is self-asserted and not bound to reviewed content or execution evidence |
| KAI-SKILL-010 | MEDIUM | Capability-gap input is unbounded |
| KAI-SKILL-011 | MEDIUM | Substring heuristics can select unrelated packages |
| KAI-SKILL-012 | MEDIUM | Public hunts amplify sequential PyPI traffic without rate limits or connection reuse |
| KAI-SKILL-013 | MEDIUM | Memory logging is fire-and-forget, untracked and does not validate delivery |
| KAI-SKILL-014 | MEDIUM | Corrupt metadata is silently treated as absent/default metadata |
| KAI-SKILL-015 | MEDIUM | Skill inventory, package names and status are exposed without authentication |
| KAI-SKILL-016 | MEDIUM | Synchronous filesystem operations execute in async handlers |
| KAI-SKILL-017 | MEDIUM | Health reports ok without validating storage, PyPI or memory readiness |
| KAI-SKILL-018 | MEDIUM | Paths, thresholds and port configuration are not validated |

---

## Skill hunter: `skill-hunter/app.py`

### KAI-SKILL-001 — CRITICAL — Unauthenticated persistent skill modification
**Issue:** `POST /hunt` requires no authentication or authorisation. It derives a filename from caller-controlled `gap` text and writes a persistent `hunted_<name>.md` skill plus sidecar metadata. Existing files with the same derived name are overwritten.  
**Risk:** Any reachable caller can modify the assistant’s active capability instructions, replace an existing generated skill or create misleading operational guidance represented as an acquired skill.  
**Recommendation:** Restrict skill creation to authenticated administrative workflows with quarantine, review, signing and explicit activation.  
**Status:** OPEN — immediate remediation required

### KAI-SKILL-002 — CRITICAL — Package existence is labelled as verification
**Issue:** `_pypi_exists` only checks whether `https://pypi.org/pypi/{package}/json` returns HTTP 200. Generated front matter and metadata then state `pypi_verified: true`. No publisher identity, version, release, source, signature, hash, dependency, vulnerability or compatibility checks occur.  
**Risk:** Simple registry existence becomes a trust claim, allowing compromised, unsuitable or malicious releases to be treated as verified capability acquisition.  
**Recommendation:** Rename the result to package-exists and require approved identities, pinned artefacts, hashes, provenance attestations, vulnerability review and sandbox testing.  
**Status:** OPEN — immediate remediation required

### KAI-SKILL-003 — CRITICAL — Acquisition claims are injected into memory
**Issue:** Every successful unauthenticated hunt launches a call to `memu-core /memory/memorize` stating that Kai acquired the generated skill, including the caller-controlled gap and generated metadata.  
**Risk:** Callers can place persistent false capability claims into long-term memory even though no package is installed, imported or functionally tested. Those claims can influence later reasoning and planning.  
**Recommendation:** Record acquisition only after authenticated approval, installation, sandbox validation and signed evidence tied to the exact artefact.  
**Status:** OPEN — immediate remediation required

### KAI-SKILL-004 — HIGH — Public skill disabling
**Issue:** `POST /skill/{name}/error` is unauthenticated. Each call increments the error count and marks the skill disabled at the configured threshold.  
**Risk:** Any caller can suppress legitimate capabilities by repeatedly reporting fabricated errors.  
**Recommendation:** Accept signed execution outcomes only from the authoritative skill runtime and require review before disablement.  
**Status:** OPEN

### KAI-SKILL-005 — HIGH — Unpinned installation instructions
**Issue:** Generated skills instruct `pip install <package>` without a version, hash, index restriction or lockfile.  
**Risk:** Later execution installs whichever release is current, making behaviour non-reproducible and exposing the system to compromised or incompatible releases.  
**Recommendation:** Generate reviewed lockfile entries with exact versions and hashes from an approved repository.  
**Status:** OPEN

### KAI-SKILL-006 — HIGH — Disablement is metadata-only
**Issue:** Reporting errors sets `meta["disabled"] = True`, but the Markdown skill remains unchanged in `SKILLS_DIR`. This service provides no enforcement hook preventing consumers from loading or using it.  
**Risk:** The safety mechanism can report a skill disabled while its instructions remain active and discoverable.  
**Recommendation:** Enforce revocation at the authoritative loader and quarantine the artefact atomically.  
**Status:** OPEN

### KAI-SKILL-007 — HIGH — Derived-name collisions overwrite skills
**Issue:** `_skill_name` removes non-alphanumeric characters, truncates to 30 characters and strips underscores. Different gaps can resolve to the same filename; `/hunt` overwrites the skill and resets metadata/error count.  
**Risk:** A caller can replace an existing generated skill using a colliding phrase, erasing provenance and probation/error history.  
**Recommendation:** Use collision-resistant immutable IDs and versioned authorised updates.  
**Status:** OPEN

### KAI-SKILL-008 — HIGH — Persistence is race-prone
**Issue:** Skill Markdown and metadata are written as separate direct file writes with no lock, temporary file, atomic rename or transaction. Error reporting performs unsynchronised read-modify-write.  
**Risk:** Concurrent hunts/error reports can create mismatched files/metadata, lost error increments, truncation or partially updated capability state.  
**Recommendation:** Use transactional storage or locked atomic versioned writes.  
**Status:** OPEN

### KAI-SKILL-009 — HIGH — Provenance is self-asserted
**Issue:** Generated YAML states `source: skill-hunter`, `pypi_verified: true`, timestamp and probation status, but no signature or digest binds metadata to the Markdown content, package release or test results.  
**Risk:** Files can appear provenance-tracked without cryptographic integrity or evidence that the represented package/content was reviewed or executed.  
**Recommendation:** Sign an immutable manifest containing content hash, package artefact hash, reviewer, tests and activation decision.  
**Status:** OPEN

### KAI-SKILL-010 — MEDIUM — Gap text is unbounded
**Issue:** `HuntRequest.gap` has no maximum length. Complete input is tokenised, returned, persisted in metadata/Markdown and sent to memory.  
**Risk:** Oversized input consumes parsing, filesystem, response and memory-service capacity.  
**Recommendation:** Enforce strict body, field and token limits.  
**Status:** OPEN

### KAI-SKILL-011 — MEDIUM — Package selection is semantically weak
**Issue:** Candidates are selected when a keyword is a substring of a mapping key or vice versa. No contextual scoring or capability test occurs.  
**Risk:** Ambiguous terms can select unrelated packages, while the first package that merely exists is represented as the solution.  
**Recommendation:** Use reviewed mappings and functional tests against the requested capability.  
**Status:** OPEN

### KAI-SKILL-012 — MEDIUM — Public hunts amplify PyPI traffic
**Issue:** A hunt may probe up to eight packages sequentially. There is no authentication, request rate limit, cache or global concurrency bound, and each probe creates a new `httpx.AsyncClient`.  
**Risk:** Repeated callers can consume worker time, sockets and sustained upstream traffic.  
**Recommendation:** Authenticate, rate-limit, cache package metadata and reuse bounded clients.  
**Status:** OPEN

### KAI-SKILL-013 — MEDIUM — Memory logging is untracked and unverified
**Issue:** `_log_to_memory` is launched with `asyncio.create_task`, the task is not retained, downstream status is not checked and exceptions are suppressed.  
**Risk:** `/hunt` reports success without knowing whether memory was updated; shutdown abandons in-flight writes and failures are invisible.  
**Recommendation:** Use a supervised durable queue and verify memory acknowledgement separately.  
**Status:** OPEN

### KAI-SKILL-014 — MEDIUM — Metadata corruption is hidden
**Issue:** `_load_meta` catches parsing failures and returns `{}`. Callers then receive default probation/error/disabled values, and error reporting rewrites partial replacement metadata.  
**Risk:** Corrupt provenance and disablement state are silently lost or reset.  
**Recommendation:** Quarantine invalid metadata and fail closed until repaired.  
**Status:** OPEN

### KAI-SKILL-015 — MEDIUM — Capability inventory is public
**Issue:** `/skills` and `/skill/{name}/health` expose names, packages, acquisition times, probation status, error counts and disable thresholds without authentication. `/hunt` returns the absolute skill path.  
**Risk:** Callers can map capabilities and target specific skills for disablement or collision overwrite.  
**Recommendation:** Require scoped operational access and avoid exposing filesystem paths.  
**Status:** OPEN

### KAI-SKILL-016 — MEDIUM — Filesystem work blocks async handlers
**Issue:** Directory creation, globbing, reads, JSON parsing and writes run synchronously inside async endpoints.  
**Risk:** Slow storage or a large skill directory blocks the event-loop worker.  
**Recommendation:** Use bounded worker execution or asynchronous transactional storage.  
**Status:** OPEN

### KAI-SKILL-017 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns ok and does not test `SKILLS_DIR` writability/integrity, PyPI reachability or memu-core delivery.  
**Risk:** Orchestration treats an unusable capability-growth service as ready.  
**Recommendation:** Separate liveness, storage readiness and dependency readiness.  
**Status:** OPEN

### KAI-SKILL-018 — MEDIUM — Configuration lacks validation
**Issue:** Skills directory, disable threshold, memu-core URL and port are accepted directly. Zero/negative thresholds, unsafe paths or invalid URLs are not rejected.  
**Risk:** Misconfiguration can instantly disable skills, write to unintended locations or break acquisition logging.  
**Recommendation:** Validate typed startup configuration with safe ranges and approved paths/URLs.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **3**
- High: **6**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **642**
- Critical: **72**
- High: **223**
- Medium: **344**
- Low: **3**

## Files materially reviewed in this batch

`skill-hunter/app.py`.
