# Kai Code Audit — Screen Capture Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SCREENCAP-001 | CRITICAL | Unauthenticated callers can capture and OCR all connected monitors |
| KAI-SCREENCAP-002 | CRITICAL | Captured screen text can be written automatically into long-term memory without authenticated provenance |
| KAI-SCREENCAP-003 | HIGH | Every screen capture is written to disk with no retention limit |
| KAI-SCREENCAP-004 | HIGH | Screen capture and OCR execute synchronously inside async handlers |
| KAI-SCREENCAP-005 | HIGH | Uploaded hostile images are processed by Pillow and Tesseract without authentication |
| KAI-SCREENCAP-006 | HIGH | Uploaded file limit is checked only after the full body is read into memory |
| KAI-SCREENCAP-007 | HIGH | Image dimensions and decompression cost are not bounded |
| KAI-SCREENCAP-008 | MEDIUM | OCR failure and unavailability strings can be treated as genuine captured text |
| KAI-SCREENCAP-009 | MEDIUM | Auto-memory HTTP status is ignored |
| KAI-SCREENCAP-010 | MEDIUM | Capture filenames use second-resolution timestamps and can collide |
| KAI-SCREENCAP-011 | MEDIUM | Fallback mode reads arbitrary latest PNG content from the configured watch directory |
| KAI-SCREENCAP-012 | MEDIUM | Health exposes capture paths and reports ok without validating capture readiness |
| KAI-SCREENCAP-013 | MEDIUM | Internal capture errors are returned to callers |
| KAI-SCREENCAP-014 | MEDIUM | Uploaded filenames are reflected without length or provenance controls |
| KAI-SCREENCAP-015 | MEDIUM | Configuration values and Boolean settings are weakly validated |

---

## Screen capture service: `screen-capture/app.py`

### KAI-SCREENCAP-001 — CRITICAL — Unauthenticated all-monitor capture and OCR
**Issue:** `POST /capture` requires no authentication or authorisation. `_capture_screen` selects `sct.monitors[0]`, which represents all monitors combined, converts the frame to PNG and returns OCR text to the caller.  
**Risk:** Any network-reachable caller can remotely capture the operator’s complete desktop arrangement and extract visible text from messages, credentials, documents and internal systems.  
**Recommendation:** Require authenticated local-session authority, explicit user consent and visible capture indicators; restrict capture to approved displays and workflows.  
**Status:** OPEN — immediate remediation required

### KAI-SCREENCAP-002 — CRITICAL — Screen-derived memory injection lacks trusted provenance
**Issue:** When `AUTO_MEMORIZE=true`, every non-empty OCR result is posted directly to `memu-core /memory/memorize` as `event_type: screen_capture` and `user_id: screen-capture`. The endpoint itself is unauthenticated, and no signed capture/session provenance is attached.  
**Risk:** Any caller able to trigger `/capture` can cause currently displayed or fallback-file text to enter long-term memory. Manipulated on-screen text can therefore become persistent trusted context and influence later agent behaviour.  
**Recommendation:** Require authenticated capture initiation, signed sensor provenance, explicit memory policy and human confirmation before durable storage.  
**Status:** OPEN — immediate remediation required

### KAI-SCREENCAP-003 — HIGH — Unbounded disk retention
**Issue:** Every successful `/capture` writes the complete image to `WATCH_DIR`. No file count, total-size, age, privacy or cleanup policy is implemented.  
**Risk:** Repeated unauthenticated captures can fill disk and retain sensitive screenshots indefinitely.  
**Recommendation:** Disable default persistence or implement encrypted, access-controlled, bounded retention with explicit deletion policy.  
**Status:** OPEN

### KAI-SCREENCAP-004 — HIGH — Blocking capture and OCR run on the event loop
**Issue:** MSS capture, PIL conversion/PNG encoding, filesystem sorting/reads, `pytesseract.image_to_string` and disk writes execute directly inside async endpoints.  
**Risk:** Capture or OCR latency blocks the event-loop worker. Repeated callers can deny service to health, metrics and other requests.  
**Recommendation:** Use a bounded isolated worker queue with capture/OCR timeouts and concurrency limits.  
**Status:** OPEN

### KAI-SCREENCAP-005 — HIGH — Public hostile-image OCR surface
**Issue:** `POST /capture/file` accepts unauthenticated image uploads and passes the bytes to Pillow and Tesseract.  
**Risk:** Any parser vulnerability or expensive image construction becomes remotely reachable, while repeated OCR requests consume substantial CPU and subprocess capacity.  
**Recommendation:** Authenticate callers and isolate image decoding/OCR in a hardened resource-limited sandbox.  
**Status:** OPEN

### KAI-SCREENCAP-006 — HIGH — Upload size check occurs after allocation
**Issue:** `capture_file` calls `await file.read()` before checking the 10 MB limit.  
**Risk:** Oversized request bodies are fully materialised in memory before rejection. Concurrent uploads amplify memory exhaustion.  
**Recommendation:** Enforce body limits before multipart parsing and stream with a strict byte counter.  
**Status:** OPEN

### KAI-SCREENCAP-007 — HIGH — Decoded image cost is unbounded
**Issue:** `_ocr_image_bytes` opens image bytes with Pillow without enforcing dimensions, pixel count, frame count or decompression ratio before OCR.  
**Risk:** Compressed image bombs or extremely large dimensions can consume excessive memory and CPU even below the raw upload limit.  
**Recommendation:** Inspect image metadata safely, reject excessive dimensions/frames and apply decompression-bomb controls before full decode.  
**Status:** OPEN

### KAI-SCREENCAP-008 — MEDIUM — Failure strings masquerade as observations
**Issue:** OCR exceptions return the literal text `[OCR error]`; missing dependencies return `[OCR unavailable ...]`; disabled OCR returns `[OCR disabled]`. These values are returned with `status: ok`, and when auto-memory is enabled they satisfy `text.strip()` and can be memorised.  
**Risk:** Sensor failure is represented and potentially persisted as genuine screen-derived content rather than unavailable/error state.  
**Recommendation:** Separate status from extracted text and never memorise placeholder/error messages.  
**Status:** OPEN

### KAI-SCREENCAP-009 — MEDIUM — Memory persistence is not acknowledged
**Issue:** The auto-memory call does not inspect HTTP status or response content. Only transport exceptions are handled.  
**Risk:** Captures can report success while memory rejected the event, or error responses can be silently treated as persistence success.  
**Recommendation:** Validate the downstream acknowledgement and return separate capture/OCR/memory states.  
**Status:** OPEN

### KAI-SCREENCAP-010 — MEDIUM — Capture filenames collide
**Issue:** Stored files use `capture_{int(time.time())}.png`, providing only one-second resolution.  
**Risk:** Multiple captures in the same second overwrite one another, causing silent data loss and inconsistent fallback selection.  
**Recommendation:** Use collision-resistant identifiers and atomic file creation.  
**Status:** OPEN

### KAI-SCREENCAP-011 — MEDIUM — Fallback trust boundary is weak
**Issue:** When MSS is unavailable, the service selects the newest `*.png` in `WATCH_DIR` and treats its bytes as a screen capture. No ownership, signature, file-type or safe-path provenance is checked.  
**Risk:** Any process able to write to the directory can inject arbitrary image content into OCR, API responses and optional long-term memory.  
**Recommendation:** Use a private permissioned directory and signed/owned capture records; do not treat arbitrary files as sensor evidence.  
**Status:** OPEN

### KAI-SCREENCAP-012 — MEDIUM — Health is privacy- and readiness-blind
**Issue:** `/health` always reports `status: ok`, exposes the device type and full watch-directory path, and only stringifies dependency flags. It does not verify display permission, monitor availability, OCR execution or directory writability.  
**Risk:** Orchestration treats the service as ready when capture can fail, while unauthenticated callers learn filesystem configuration.  
**Recommendation:** Separate liveness, capture readiness, OCR readiness and storage readiness; restrict operational metadata.  
**Status:** OPEN

### KAI-SCREENCAP-013 — MEDIUM — Internal errors are disclosed
**Issue:** Unexpected capture exceptions are interpolated directly into HTTP 500 details.  
**Risk:** Display, filesystem and library diagnostics are exposed to unauthenticated callers.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-SCREENCAP-014 — MEDIUM — Filename metadata is unbounded and untrusted
**Issue:** Uploaded `file.filename` has no maximum length or sanitised provenance and is reflected in the `source` response field.  
**Risk:** Callers can create oversized or misleading source labels and inject control characters or trusted-looking names into downstream logs/UI.  
**Recommendation:** Bound and sanitise display filenames and derive source identity from authenticated metadata.  
**Status:** OPEN

### KAI-SCREENCAP-015 — MEDIUM — Configuration validation is weak
**Issue:** Watch path, service URL and port are accepted directly. Boolean settings accept only exact lowercase `true`; invalid values silently become false rather than failing validation.  
**Risk:** Misconfiguration silently disables OCR or memory behaviour, selects unintended storage paths or breaks startup/runtime routing.  
**Recommendation:** Validate typed startup configuration with explicit accepted values and approved paths/URLs.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **2**
- High: **5**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **507**
- Critical: **56**
- High: **184**
- Medium: **264**
- Low: **3**

## Files materially reviewed in this batch

`screen-capture/app.py`.
