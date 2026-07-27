# Kai Code Audit — Screen Capture Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_SCREEN_CAPTURE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SCREENCAPX-001 | HIGH | The production image always installs MSS, preventing the advertised file-based fallback from being selected |
| KAI-SCREENCAPX-002 | HIGH | Runtime MSS/display failure raises before the watch-directory fallback is attempted |
| KAI-SCREENCAPX-003 | HIGH | The full-stack container has no display socket or display environment for MSS capture |
| KAI-SCREENCAPX-004 | HIGH | The full-stack container mounts no watch-directory source for simulated captures |
| KAI-SCREENCAPX-005 | HIGH | Screen Capture is omitted from the minimal topology even though Dashboard and Screen Watcher reference it |
| KAI-SCREENCAPX-006 | HIGH | Health remains green in the deterministic headless deployment failure state |
| KAI-SCREENCAPX-007 | HIGH | The image unnecessarily copies the repository security directory into an untrusted image-processing service |
| KAI-SCREENCAPX-008 | HIGH | Trusted-token and policy files become readable attack loot after any Pillow/Tesseract/service compromise |
| KAI-SCREENCAPX-009 | HIGH | Tesseract OCR has no subprocess deadline or cancellation boundary |
| KAI-SCREENCAPX-010 | HIGH | Tesseract output is fully materialised before response and memory truncation |
| KAI-SCREENCAPX-011 | HIGH | Shared sanitisation silently reduces the advertised 5,000-character OCR response to roughly 1,024 characters |
| KAI-SCREENCAPX-012 | HIGH | Shared sanitisation silently changes OCR semantics by removing punctuation/operators |
| KAI-SCREENCAPX-013 | HIGH | Capture, OCR, memory insertion and screenshot persistence are independent non-transactional operations |
| KAI-SCREENCAPX-014 | HIGH | A memory record can commit even when the corresponding screenshot fails to persist |
| KAI-SCREENCAPX-015 | HIGH | A screenshot can persist even when the corresponding memory record is rejected or lost |
| KAI-SCREENCAPX-016 | HIGH | Memory records contain no image digest, screenshot ID, display identity or storage revision |
| KAI-SCREENCAPX-017 | HIGH | Sensitive screenshots are stored with ordinary unencrypted filesystem semantics and no hardened permissions contract |
| KAI-SCREENCAPX-018 | HIGH | Fallback PNG discovery follows symlinks and can OCR an image outside the configured watch directory |
| KAI-SCREENCAPX-019 | HIGH | Capture persistence follows a pre-existing symlink at the timestamp-derived output path |
| KAI-SCREENCAPX-020 | HIGH | Uploaded content is not restricted by MIME type, filename extension or magic-byte policy |
| KAI-SCREENCAPX-021 | HIGH | All-monitor composite dimensions are unbounded before PNG encoding, OCR, disk write and response processing |
| KAI-SCREENCAPX-022 | HIGH | No OCR worker/process concurrency limit or admission queue protects Tesseract execution |
| KAI-SCREENCAPX-023 | HIGH | OCR text is returned as authoritative extracted content without confidence, layout or source-quality evidence |
| KAI-SCREENCAPX-024 | HIGH | OCR/screenshot responses lack `Cache-Control: no-store` and equivalent privacy headers |
| KAI-SCREENCAPX-025 | MEDIUM | Latest-file selection performs an unbounded blocking stat-and-sort over every PNG |
| KAI-SCREENCAPX-026 | MEDIUM | File selection has a time-of-check-to-time-of-use race between stat/sort and read |
| KAI-SCREENCAPX-027 | MEDIUM | Animated and multi-frame image handling is implicit and unreported |
| KAI-SCREENCAPX-028 | MEDIUM | EXIF orientation is not applied or represented before OCR |
| KAI-SCREENCAPX-029 | MEDIUM | OCR language, Tesseract version, page-segmentation mode and engine configuration are absent from results |
| KAI-SCREENCAPX-030 | MEDIUM | The response schema contains an unused `image_b64` field that the service never populates |
| KAI-SCREENCAPX-031 | MEDIUM | `/capture/file` has no response model or versioned result schema |
| KAI-SCREENCAPX-032 | MEDIUM | Capture, memory and API timestamps are generated independently and lack one monotonic operation sequence |
| KAI-SCREENCAPX-033 | MEDIUM | Health capability flags are encoded as strings rather than Booleans |
| KAI-SCREENCAPX-034 | MEDIUM | Public metrics expose request/error state without administrative authentication |
| KAI-SCREENCAPX-035 | MEDIUM | Tesseract and image-processing subprocesses inherit the complete service environment |
| KAI-SCREENCAPX-036 | MEDIUM | The watch directory is created at import before startup configuration/readiness can be reported |
| KAI-SCREENCAPX-037 | MEDIUM | Build dependencies and the Python base image are not reproducibly pinned by digest |
| KAI-SCREENCAPX-038 | MEDIUM | The test suite intentionally removes optional dependencies and therefore misses the production MSS-without-display failure path |
| KAI-SCREENCAPX-039 | MEDIUM | Tests accept capture failure and do not require real OCR, memory acknowledgement or persisted-image integrity |
| KAI-SCREENCAPX-040 | MEDIUM | The service has no lifespan-owned OCR pool, shared memU client, graceful job drain or storage reconciliation |
| KAI-SCREENCAPX-041 | MEDIUM | No tamper-evident audit links caller, capture source, image digest, OCR engine, returned text, memory result and stored file |

---

## High-severity findings

### KAI-SCREENCAPX-001 — HIGH — Fallback is unreachable in the production image
**Issue:** `screen-capture/requirements.txt` installs MSS, so `_mss_available=True`. `_capture_screen()` enters the MSS branch and only reaches the watch-directory fallback when MSS import is unavailable.  
**Risk:** The service’s documented container/file simulation mode is not selected in its own production image.  
**Recommendation:** Select capture backend explicitly and fall back on runtime capture failure only under an authenticated, provenance-safe policy.  
**Status:** OPEN

### KAI-SCREENCAPX-002 — HIGH — Runtime display failure does not fall back
MSS exceptions from `mss.mss()` or `grab()` are not caught inside `_capture_screen`; the endpoint returns 500 instead of using a valid governed fallback image.

### KAI-SCREENCAPX-003 — HIGH — No display is connected
The full Compose service provides neither X11/Wayland/display environment nor a desktop-capture portal/socket.

### KAI-SCREENCAPX-004 — HIGH — No simulated-capture source is mounted
Full Compose mounts no host/volume directory into `SCREEN_WATCH_DIR`, so even a corrected fallback starts empty and process-local.

### KAI-SCREENCAPX-005 — HIGH — Minimal topology references an absent service
The minimal deployment configures Dashboard/Screen Watcher integrations but contains no Screen Capture service definition.

### KAI-SCREENCAPX-006 — HIGH — Deployment failure is health-green
`/health` reports `status=ok` based on imports and paths, not a successful display capture/OCR/storage probe.

### KAI-SCREENCAPX-007 — HIGH — Unnecessary security bundle in parser image
The Dockerfile copies `security/` although Screen Capture imports no files from it.

### KAI-SCREENCAPX-008 — HIGH — Credential/policy exposure after parser compromise
The copied directory includes trusted-token and policy material. A vulnerability in the public Pillow/Tesseract processing surface gains access to unnecessary security data.

### KAI-SCREENCAPX-009 — HIGH — Unbounded OCR subprocess time
`pytesseract.image_to_string()` is invoked without its timeout option or an outer process deadline.

### KAI-SCREENCAPX-010 — HIGH — Unbounded OCR output allocation
Complete Tesseract output is constructed in memory before later slicing/sanitisation.

### KAI-SCREENCAPX-011 — HIGH — Misleading output-size contract
The endpoint slices to 5,000 characters, then `sanitize_string()` applies the shared 1,024-character limit without reporting truncation.

### KAI-SCREENCAPX-012 — HIGH — OCR meaning is altered
The generic sanitizer removes semicolons, pipes and ampersands, changing code, commands, formulae and natural text while reporting ordinary successful OCR.

### KAI-SCREENCAPX-013 — HIGH — Split capture transaction
Capture, OCR, memU POST and file write have no common operation ID, staging state, rollback or verified commit.

### KAI-SCREENCAPX-014 — HIGH — Memory without screenshot
Memory insertion runs before screenshot persistence and is not reversed if the later write fails.

### KAI-SCREENCAPX-015 — HIGH — Screenshot without memory
Memory failure is warning-only; the sensitive image remains on disk while callers cannot reconcile the missing derived record.

### KAI-SCREENCAPX-016 — HIGH — Memory lacks evidence identity
The memU payload contains only timestamp, event type, truncated text and a string user ID. It cannot be tied to exact image bytes, display or stored file.

### KAI-SCREENCAPX-017 — HIGH — Weak screenshot-at-rest protection
The service uses ordinary `mkdir`/`write_bytes` semantics with no encryption, restrictive mode assertion, ownership verification or isolated encrypted volume.

### KAI-SCREENCAPX-018 — HIGH — Fallback symlink escape
`glob("*.png")`, `stat()` and `read_bytes()` follow symbolic links; a link inside the directory may expose an image stored elsewhere.

### KAI-SCREENCAPX-019 — HIGH — Output symlink overwrite
`write_bytes()` follows an existing `capture_<second>.png` symlink, allowing the write to target another app-writable path when the directory is shared or compromised.

### KAI-SCREENCAPX-020 — HIGH — No media allowlist
Any uploaded bytes are passed to Pillow regardless of declared MIME, filename extension or magic signature.

### KAI-SCREENCAPX-021 — HIGH — Unbounded real-screen dimensions
The all-monitor composite is encoded and OCRed without maximum pixels, dimensions or combined-display count.

### KAI-SCREENCAPX-022 — HIGH — No OCR admission control
Anonymous callers can launch concurrent synchronous Tesseract subprocesses with no queue/semaphore/process budget.

### KAI-SCREENCAPX-023 — HIGH — Missing extraction evidence
The API returns plain text without OCR confidence, word/line boxes, language, page/frame selection or an explicit untrusted-data classification.

### KAI-SCREENCAPX-024 — HIGH — Sensitive response caching
OCR of the operator’s screen is returned without privacy-oriented cache controls.

---

## Medium-severity findings

### KAI-SCREENCAPX-025 — MEDIUM — Unbounded directory scan
Every fallback capture stats and sorts the full unbounded PNG directory before choosing one file.

### KAI-SCREENCAPX-026 — MEDIUM — File-selection race
A selected path may be replaced/deleted between sorting and reading; no descriptor-based validated open is used.

### KAI-SCREENCAPX-027 — MEDIUM — Multi-frame ambiguity
Pillow/Tesseract frame selection for GIF/TIFF/other multi-frame inputs is neither restricted nor reported.

### KAI-SCREENCAPX-028 — MEDIUM — Orientation ambiguity
EXIF orientation is not normalised, reducing OCR quality and making extracted layout/provenance unclear.

### KAI-SCREENCAPX-029 — MEDIUM — Missing OCR configuration provenance
Results do not identify Tesseract version, language data, OCR configuration or preprocessing revision.

### KAI-SCREENCAPX-030 — MEDIUM — Dead schema field
`CaptureResult.image_b64` advertises optional image delivery but is never populated.

### KAI-SCREENCAPX-031 — MEDIUM — Unmodelled upload response
The file endpoint returns a raw JSONResponse rather than the declared CaptureResult contract.

### KAI-SCREENCAPX-032 — MEDIUM — Fragmented chronology
Memory timestamp, filename second and response timestamp are separately generated and have no shared event sequence.

### KAI-SCREENCAPX-033 — MEDIUM — Stringified readiness flags
Values such as `"False"` are non-empty strings and can be misinterpreted by weak health consumers.

### KAI-SCREENCAPX-034 — MEDIUM — Public telemetry
Metrics requires no administrative identity.

### KAI-SCREENCAPX-035 — MEDIUM — Environment inheritance
OCR subprocesses inherit service environment values rather than a minimal controlled environment.

### KAI-SCREENCAPX-036 — MEDIUM — Import-time filesystem mutation
The watch directory is created during import; path/permission failure prevents a controlled degraded startup state.

### KAI-SCREENCAPX-037 — MEDIUM — Non-reproducible image
Several dependencies use lower-bound ranges and `python:3.11-slim` is not digest-pinned.

### KAI-SCREENCAPX-038 — MEDIUM — Tests miss deployed backend selection
Tests are explicitly designed without MSS/display, while the production image always installs MSS.

### KAI-SCREENCAPX-039 — MEDIUM — Failure-tolerant tests
The suite accepts 500/503 capture and does not verify Tesseract output, exact memory acknowledgement or image-to-memory integrity.

### KAI-SCREENCAPX-040 — MEDIUM — Missing lifecycle ownership
No lifespan owns OCR workers, a shared downstream client, active jobs or shutdown reconciliation.

### KAI-SCREENCAPX-041 — MEDIUM — Missing end-to-end audit
No immutable record joins initiating principal, source display/file, image hash, OCR revision, text digest, memory acknowledgement and persisted screenshot.

---

## Batch totals

- Findings: **41**
- Critical: **0**
- High: **24**
- Medium: **17**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,559**
- Critical: **191**
- High: **1,285**
- Medium: **1,080**
- Low: **3**

## Files materially reviewed

`screen-capture/app.py`, `screen-capture/Dockerfile`, `screen-capture/requirements.txt`, `scripts/test_screen_capture.py`, Screen Capture deployment and absence across full/minimal Compose topologies, `security/trusted_tokens.txt`, and integrations with Dashboard, memU and Screen Watcher.
