# Kai Code Audit — Document Parser Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DOCPARSE-001 | CRITICAL | Unauthenticated callers can submit hostile files to multiple complex native/document parsers |
| KAI-DOCPARSE-002 | CRITICAL | ZIP extraction limit is checked only after a member is fully decompressed into memory |
| KAI-DOCPARSE-003 | HIGH | Uploaded files are fully read into memory without an outer size limit |
| KAI-DOCPARSE-004 | HIGH | Untrusted DWG files are passed to an external conversion binary |
| KAI-DOCPARSE-005 | HIGH | CPU- and I/O-intensive parsing executes directly in async request handlers |
| KAI-DOCPARSE-006 | HIGH | Extracted text and parser output sizes are not bounded |
| KAI-DOCPARSE-007 | MEDIUM | File type selection trusts the caller-controlled filename extension |
| KAI-DOCPARSE-008 | MEDIUM | Legacy DOC and PPT files are routed to incompatible OOXML parsers |
| KAI-DOCPARSE-009 | MEDIUM | ZIP member exceptions are embedded into returned text |
| KAI-DOCPARSE-010 | MEDIUM | DWG converter stderr is exposed to callers |
| KAI-DOCPARSE-011 | MEDIUM | `/formats` runs a blocking subprocess on every GET request |
| KAI-DOCPARSE-012 | MEDIUM | Parser resources are not consistently closed on exceptions |
| KAI-DOCPARSE-013 | MEDIUM | No rate limiting or bounded parser concurrency is implemented |
| KAI-DOCPARSE-014 | MEDIUM | Health reports ok regardless of missing parser capabilities |
| KAI-DOCPARSE-015 | MEDIUM | Error-budget telemetry is never recorded by request middleware |

---

## Document parser: `document-parser/app.py`

### KAI-DOCPARSE-001 — CRITICAL — Unauthenticated hostile-file parsing
**Issue:** `POST /parse` requires no authentication or authorisation and dispatches caller-supplied bytes into PyMuPDF, python-docx, openpyxl, xlrd, python-pptx, ezdxf, BeautifulSoup/lxml, ZIP and JSON/CSV processing based on the filename.  
**Risk:** Any reachable caller can repeatedly exercise a broad attack surface of complex document and native-code dependencies with malformed files. A vulnerability in any parser becomes remotely reachable through this service.  
**Recommendation:** Require authenticated callers, isolate parsing in a hardened sandbox with no secrets/network, and apply format-specific resource limits and dependency patch governance.  
**Status:** OPEN — immediate remediation required

### KAI-DOCPARSE-002 — CRITICAL — ZIP bomb limit is enforced after allocation
**Issue:** `_parse_zip` calls `member_data = zf.read(info.filename)` before adding its length to `total_extracted` and checking the 50 MB cumulative limit for subsequent members. A single member can therefore decompress far beyond the configured maximum.  
**Risk:** A small compressed upload can expand into an arbitrarily large in-memory object and exhaust process/container memory before the limit takes effect.  
**Recommendation:** Reject members from declared and streamed uncompressed sizes before allocation, enforce compression-ratio limits and stop decompression with a strict byte counter.  
**Status:** OPEN — immediate remediation required

### KAI-DOCPARSE-003 — HIGH — Outer upload size is unbounded
**Issue:** The endpoint calls `await file.read()` with no request or file-size limit.  
**Risk:** Large uploads consume memory before any format validation or parser-specific control is applied. Concurrent requests multiply the allocation.  
**Recommendation:** Enforce body limits at the ASGI/proxy boundary and stream into bounded quarantine storage.  
**Status:** OPEN

### KAI-DOCPARSE-004 — HIGH — External converter processes untrusted DWG input
**Issue:** `_parse_dwg` writes attacker bytes to disk and invokes `dwg2dxf` directly. The converter runs with the service process privileges and environment.  
**Risk:** Malformed DWG files can exploit the external binary or consume CPU, memory and disk. The 30-second timeout limits duration but does not sandbox privileges, filesystem or network access.  
**Recommendation:** Run conversion in a disposable, unprivileged, seccomp/AppArmor-confined worker with strict CPU, memory, disk and process limits.  
**Status:** OPEN

### KAI-DOCPARSE-005 — HIGH — Blocking parsing runs on the event loop
**Issue:** All document parsing, ZIP decompression, workbook iteration, CAD recovery and `subprocess.run` calls execute directly inside the async `/parse` handler.  
**Risk:** One expensive file blocks the event-loop worker, including health and metrics requests. Multiple unauthenticated uploads create straightforward denial of service.  
**Recommendation:** Use a bounded asynchronous job queue backed by isolated parser workers and hard execution deadlines.  
**Status:** OPEN

### KAI-DOCPARSE-006 — HIGH — Extracted output is unbounded
**Issue:** Parsers concatenate all PDF pages, paragraphs, spreadsheet cells, slide text, CSV rows and up to 50 MB of ZIP member content into Python strings and return them in one JSON response. There is no character, row, page, worksheet, entity or response-size ceiling.  
**Risk:** Files with huge logical content can cause memory amplification, expensive serialisation and oversized downstream context ingestion even when the raw upload is moderate.  
**Recommendation:** Apply per-format page/row/cell/entity and output-character limits with truncation metadata.  
**Status:** OPEN

### KAI-DOCPARSE-007 — MEDIUM — File type is determined by untrusted extension
**Issue:** Dispatch uses only the suffix from `file.filename`; MIME type and magic bytes are not validated.  
**Risk:** Callers can route arbitrary bytes into a chosen parser, bypass intended format assumptions and produce misleading `format` metadata.  
**Recommendation:** Verify file signatures and parser-detected type against an allowlisted extension/MIME combination.  
**Status:** OPEN

### KAI-DOCPARSE-008 — MEDIUM — Legacy formats are misrouted
**Issue:** `.doc` is sent to `python-docx`, and `.ppt` is sent to `python-pptx`. These libraries parse OOXML `.docx`/`.pptx`, not legacy binary Office formats.  
**Risk:** The service advertises support it does not actually provide, returning parser errors for legitimate files and expanding the malformed-input surface.  
**Recommendation:** Remove unsupported claims or use isolated, purpose-built legacy converters.  
**Status:** OPEN

### KAI-DOCPARSE-009 — MEDIUM — ZIP parser errors become document text
**Issue:** Generic exceptions from member parsing are interpolated into returned extracted text as `[error: {exc}]`.  
**Risk:** Internal paths, library diagnostics and parser details can be disclosed to callers and later ingested as document content by downstream systems.  
**Recommendation:** Return stable per-member error codes separately from extracted content.  
**Status:** OPEN

### KAI-DOCPARSE-010 — MEDIUM — Converter diagnostics are disclosed
**Issue:** Non-zero `dwg2dxf` stderr is decoded and returned in an HTTP 502 detail.  
**Risk:** Converter version, filesystem paths and internal diagnostics are exposed to unauthenticated callers.  
**Recommendation:** Log protected diagnostics and return a stable conversion error code.  
**Status:** OPEN

### KAI-DOCPARSE-011 — MEDIUM — GET `/formats` launches a subprocess
**Issue:** Every formats request synchronously executes `which dwg2dxf` inside an async endpoint.  
**Risk:** Repeated unauthenticated GET requests cause process-spawn overhead and event-loop blocking for information that changes only at deployment time.  
**Recommendation:** Resolve tool availability once at startup and expose cached readiness.  
**Status:** OPEN

### KAI-DOCPARSE-012 — MEDIUM — Resource closure is exception-sensitive
**Issue:** PDF documents and presentation objects are not explicitly closed. XLSX workbooks close only after successful complete iteration; exceptions before `wb.close()` skip cleanup.  
**Risk:** Repeated malformed files can retain native/library resources longer than necessary and increase memory/file-descriptor pressure.  
**Recommendation:** Use context managers or `try/finally` cleanup around every parser resource.  
**Status:** OPEN

### KAI-DOCPARSE-013 — MEDIUM — No abuse controls around expensive parsers
**Issue:** There is no caller identity, rate limit, queue size, concurrent-job bound, per-format timeout or circuit breaker except the DWG subprocess timeout.  
**Risk:** Attackers can saturate CPU, memory and disk with legitimate but expensive files.  
**Recommendation:** Apply authenticated quotas and isolated bounded worker concurrency.  
**Status:** OPEN

### KAI-DOCPARSE-014 — MEDIUM — Health is capability-blind
**Issue:** `/health` always returns `status: ok` even if all optional parsers are absent; it omits ZIP/plain parsing readiness and does not test the DWG tool.  
**Risk:** Orchestration treats the advertised multi-format parser as ready despite missing required capabilities.  
**Recommendation:** Separate liveness from per-format readiness and required deployment profile.  
**Status:** OPEN

### KAI-DOCPARSE-015 — MEDIUM — Error budget is not populated
**Issue:** `_budget` is created, but the application defines no middleware or endpoint logic that calls `_budget.record`. `/metrics` therefore exposes a snapshot without request outcomes being recorded in this module.  
**Risk:** Reliability metrics appear available while not reflecting parser failures.  
**Recommendation:** Record actual response status and exceptions through consistent middleware.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **2**
- High: **4**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **441**
- Critical: **46**
- High: **166**
- Medium: **226**
- Low: **3**

## Files materially reviewed in this batch

`document-parser/app.py`.
