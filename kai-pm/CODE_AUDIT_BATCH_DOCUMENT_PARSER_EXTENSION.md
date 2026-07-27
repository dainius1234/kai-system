# Kai Code Audit — Document Parser Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_DOCUMENT_PARSER.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DOCPARSEX-001 | CRITICAL | OOXML formats bypass all ZIP expansion, member-count and compression-ratio controls |
| KAI-DOCPARSEX-002 | HIGH | The Dockerfile uses invalid `HEALTHCHECK --start_period` syntax and can fail image construction |
| KAI-DOCPARSEX-003 | HIGH | LibreDWG installation failure is explicitly ignored during the image build |
| KAI-DOCPARSEX-004 | HIGH | Parser/converter workers retain unrestricted network egress despite processing hostile documents |
| KAI-DOCPARSEX-005 | HIGH | ZIP central-directory metadata is fully materialised before the first-100-member slice |
| KAI-DOCPARSEX-006 | HIGH | Directory entries consume the first-100 ZIP slots and can hide all actual files from parsing |
| KAI-DOCPARSEX-007 | HIGH | Duplicate ZIP names are read by filename rather than by the inspected `ZipInfo` entry |
| KAI-DOCPARSEX-008 | HIGH | Nested ZIP and unsupported binary members are decoded as plain text rather than rejected |
| KAI-DOCPARSEX-009 | HIGH | Unknown top-level extensions receive a success-shaped arbitrary-binary-to-text fallback |
| KAI-DOCPARSEX-010 | HIGH | Invalid JSON is silently returned as successful plain text with `format=json` |
| KAI-DOCPARSEX-011 | HIGH | JSON nesting/depth and object cardinality are unbounded before pretty-print expansion |
| KAI-DOCPARSEX-012 | HIGH | CSV parsing holds the complete decoded input, row matrix and formatted output simultaneously |
| KAI-DOCPARSEX-013 | HIGH | PDF extraction holds every page string plus the joined document string simultaneously |
| KAI-DOCPARSEX-014 | HIGH | Encrypted or permission-restricted PDFs can return empty successful extraction without an encryption state |
| KAI-DOCPARSEX-015 | HIGH | XLSX `data_only=True` returns potentially stale cached formula values without formula/freshness evidence |
| KAI-DOCPARSEX-016 | HIGH | DOCX extraction silently omits headers, footers, comments, text boxes, tracked changes and other content classes |
| KAI-DOCPARSEX-017 | HIGH | PPTX extraction silently omits notes, tables, grouped shapes, charts and non-text-frame content |
| KAI-DOCPARSEX-018 | HIGH | DXF iteration has no entity-count or processing-work limit |
| KAI-DOCPARSEX-019 | HIGH | ezdxf recovery diagnostics are discarded while recovered content is returned as authoritative |
| KAI-DOCPARSEX-020 | HIGH | CAD layer, block and annotation strings are returned without a safe presentation/control-character contract |
| KAI-DOCPARSEX-021 | HIGH | DWG converter stdout and stderr are fully buffered before any truncation |
| KAI-DOCPARSEX-022 | HIGH | Converted DXF output is read completely with no output-file size limit |
| KAI-DOCPARSEX-023 | HIGH | DWG timeout terminates only the direct process and does not guarantee descendant cleanup |
| KAI-DOCPARSEX-024 | HIGH | The external converter inherits the full environment and ordinary filesystem/network access |
| KAI-DOCPARSEX-025 | HIGH | Temporary-file and converter-output disk consumption has no per-job quota |
| KAI-DOCPARSEX-026 | HIGH | Parse responses have no strict response model or versioned output schema |
| KAI-DOCPARSEX-027 | HIGH | Results contain no file digest, parser/library versions, job ID, timing or truncation evidence |
| KAI-DOCPARSEX-028 | HIGH | Extracted text is not classified as untrusted document data before downstream prompt use |
| KAI-DOCPARSEX-029 | HIGH | Sensitive parsed-document responses lack `Cache-Control: no-store` and equivalent privacy headers |
| KAI-DOCPARSEX-030 | MEDIUM | XML files are parsed with BeautifulSoup’s `lxml` HTML parser rather than an XML-specific parser contract |
| KAI-DOCPARSEX-031 | MEDIUM | UTF-8 replacement decoding silently corrupts CSV, JSON, XML, HTML and unknown plain-text inputs |
| KAI-DOCPARSEX-032 | MEDIUM | CSV dialect, quoting and encoding are assumed rather than detected or supplied by a validated contract |
| KAI-DOCPARSEX-033 | MEDIUM | `page_count` inconsistently means PDF pages, worksheets, slides or ZIP files |
| KAI-DOCPARSEX-034 | MEDIUM | CAD layer, block and annotation truncation is not reported in metadata |
| KAI-DOCPARSEX-035 | MEDIUM | ZIP member-limit truncation is not reported and omitted files are invisible to callers |
| KAI-DOCPARSEX-036 | MEDIUM | ZIP member names may contain control/markup text that is copied into result text and metadata |
| KAI-DOCPARSEX-037 | MEDIUM | PDF title and author metadata are untrusted strings returned without presentation constraints |
| KAI-DOCPARSEX-038 | MEDIUM | `GET /formats` publicly reveals installed parser and converter capability inventory |
| KAI-DOCPARSEX-039 | MEDIUM | The per-request `which dwg2dxf` subprocess has no explicit timeout |
| KAI-DOCPARSEX-040 | MEDIUM | Very long DWG filenames can exceed temporary-path limits and fail outside a typed parser error |
| KAI-DOCPARSEX-041 | MEDIUM | DXF parser exception details are returned directly in HTTP 400 responses |
| KAI-DOCPARSEX-042 | MEDIUM | Raw caller filenames are written to logs without control-character or log-injection handling |
| KAI-DOCPARSEX-043 | MEDIUM | Public metrics expose a disabled/unpopulated reliability object without administrative authentication |
| KAI-DOCPARSEX-044 | MEDIUM | Optional-library initialisation catches only `ImportError`, not native/runtime initialisation failure |
| KAI-DOCPARSEX-045 | MEDIUM | Health exposes no parser versions or known-file self-test results |
| KAI-DOCPARSEX-046 | MEDIUM | Dependencies, system packages and the Python base image are not reproducibly digest-pinned |
| KAI-DOCPARSEX-047 | MEDIUM | No dedicated parser tests were found for archive bombs, duplicate entries, converter limits or malformed OOXML |
| KAI-DOCPARSEX-048 | MEDIUM | The service has no lifespan-owned parser pool, job queue, graceful cancellation or temporary-file reconciliation |
| KAI-DOCPARSEX-049 | MEDIUM | No tamper-evident audit binds caller, file digest, detected format, parser revision and extracted-output digest |
| KAI-DOCPARSEX-050 | MEDIUM | Parse results contain no source timestamp, processing timestamp or monotonic operation sequence |

---

## Critical finding

### KAI-DOCPARSEX-001 — CRITICAL — OOXML archive controls are bypassed
**Issue:** DOCX, XLSX and PPTX are ZIP-based containers, but they are dispatched directly to python-docx, openpyxl and python-pptx. `_MAX_ZIP_EXTRACTED` and `_MAX_ZIP_MEMBERS` apply only when the filename extension is `.zip`.  
**Risk:** A small OOXML archive can contain oversized/high-ratio XML and member structures that bypass the service’s only archive controls before reaching complex parsers.  
**Recommendation:** Apply one preflight archive policy to every ZIP-derived format before opening it in format-specific libraries.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-DOCPARSEX-002 — HIGH — Invalid Docker healthcheck option
The Dockerfile uses `--start_period`; Dockerfile HEALTHCHECK syntax requires `--start-period`. Image construction may fail before the service can be deployed.

### KAI-DOCPARSEX-003 — HIGH — Converter omission is deliberately hidden
APT installation of `libredwg-tools` is wrapped with `|| true`, allowing a successful image build with the advertised converter absent.

### KAI-DOCPARSEX-004 — HIGH — Hostile parser jobs have network egress
Neither the container nor subprocess limits outbound destinations; parser/converter compromise inherits the service network.

### KAI-DOCPARSEX-005 — HIGH — Unbounded ZIP central directory
`zf.infolist()` constructs metadata for every archive entry before slicing the first 100.

### KAI-DOCPARSEX-006 — HIGH — Directory-prefix hiding
The slice occurs before directories are skipped, so 100 directory entries can prevent every actual file from being considered.

### KAI-DOCPARSEX-007 — HIGH — Duplicate-name entry confusion
The loop inspects one `ZipInfo` but calls `zf.read(info.filename)`; duplicate names can resolve to a different archive entry than the inspected one.

### KAI-DOCPARSEX-008 — HIGH — Binary/nested archive data becomes text
At depth one, `.zip` no longer enters `_parse_zip` and falls through to replacement-decoded plain text. Other unsupported binary members do the same.

### KAI-DOCPARSEX-009 — HIGH — Arbitrary extension success fallback
Any unrecognised top-level extension is labelled with that caller-controlled format and returned as decoded text.

### KAI-DOCPARSEX-010 — HIGH — Invalid JSON is reported as parsed JSON
JSON exceptions are swallowed; original decoded bytes are returned with `format=json` and no validity flag.

### KAI-DOCPARSEX-011 — HIGH — JSON expansion workload
Nested/large JSON is fully parsed and then pretty-printed, potentially expanding memory/output substantially.

### KAI-DOCPARSEX-012 — HIGH — CSV triple representation
The service retains decoded text, `list(reader)` and a second joined string concurrently.

### KAI-DOCPARSEX-013 — HIGH — PDF double representation
Every page string is retained in a list before another complete joined string is created.

### KAI-DOCPARSEX-014 — HIGH — Protected PDF ambiguity
The result exposes no password/encryption/permission state; empty extraction can look like a genuinely empty document.

### KAI-DOCPARSEX-015 — HIGH — Stale spreadsheet values
`data_only=True` returns workbook cached formula results, which may be missing or stale, while formulas and calculation timestamps are omitted.

### KAI-DOCPARSEX-016 — HIGH — Incomplete DOCX extraction
Only body paragraphs and tables are processed; omitted content is not declared.

### KAI-DOCPARSEX-017 — HIGH — Incomplete PPTX extraction
Only text-frame paragraphs are processed; omitted notes/tables/groups/charts are not declared.

### KAI-DOCPARSEX-018 — HIGH — Unbounded CAD iteration
Every modelspace entity is visited before output slices are applied.

### KAI-DOCPARSEX-019 — HIGH — Recovery warnings discarded
`ezdxf.recover.readfile()` returns an auditor object, but it is assigned to `_` and ignored.

### KAI-DOCPARSEX-020 — HIGH — Unsafe CAD text
Names and annotations may contain HTML, Markdown, bidirectional or control characters and are propagated as ordinary document text/metadata.

### KAI-DOCPARSEX-021 — HIGH — Converter pipe buffering
`capture_output=True` stores complete stdout/stderr in memory; only the returned error excerpt is truncated.

### KAI-DOCPARSEX-022 — HIGH — Unbounded converter output
The generated DXF file is opened and read completely before DXF parsing.

### KAI-DOCPARSEX-023 — HIGH — Incomplete timeout containment
`subprocess.run(timeout=30)` controls the direct process but provides no process-group/session cleanup guarantee.

### KAI-DOCPARSEX-024 — HIGH — Converter privilege surface
The converter receives the service environment, filesystem access and network namespace without a minimal sandbox.

### KAI-DOCPARSEX-025 — HIGH — No temporary disk budget
Input/output/temp artefacts have no per-job or aggregate disk limit.

### KAI-DOCPARSEX-026 — HIGH — Unversioned result shape
The endpoint returns parser-specific dictionaries with no response model or stable schema revision.

### KAI-DOCPARSEX-027 — HIGH — Missing extraction provenance
Results cannot be tied to exact bytes, parser versions, truncation policy or processing job.

### KAI-DOCPARSEX-028 — HIGH — Prompt-injection boundary absent
Extracted document text is returned as ordinary text rather than explicitly untrusted evidence, despite Dashboard inserting it into Agentic chat.

### KAI-DOCPARSEX-029 — HIGH — Cacheable private documents
Responses containing complete extracted document contents have no privacy cache controls.

---

## Medium-severity findings

### KAI-DOCPARSEX-030 — MEDIUM — XML parser mismatch
For XML, the code selects parser string `lxml`, which is BeautifulSoup’s HTML-oriented lxml builder rather than an explicit XML parser contract.

### KAI-DOCPARSEX-031 — MEDIUM — Silent encoding corruption
Replacement decoding loses source bytes/encoding errors without returning a degraded flag.

### KAI-DOCPARSEX-032 — MEDIUM — Fixed CSV interpretation
Comma dialect and UTF-8 are assumed; semicolon/tab/quoted regional files can be materially misparsed.

### KAI-DOCPARSEX-033 — MEDIUM — Ambiguous page count
One field has incompatible semantics across formats.

### KAI-DOCPARSEX-034 — MEDIUM — Hidden CAD truncation
Lists are sliced but no total-versus-returned/truncated flag is emitted.

### KAI-DOCPARSEX-035 — MEDIUM — Hidden ZIP truncation
Only the first 100 central-directory entries are considered; the response never says the archive was truncated.

### KAI-DOCPARSEX-036 — MEDIUM — Unsafe archive names
Raw member names are copied into extracted text and metadata.

### KAI-DOCPARSEX-037 — MEDIUM — Unsafe PDF metadata
Title and author are untrusted document-controlled values.

### KAI-DOCPARSEX-038 — MEDIUM — Public parser inventory
`/formats` exposes installed libraries/converter availability.

### KAI-DOCPARSEX-039 — MEDIUM — Unbounded formats subprocess
The `which` call has no timeout even though the result should be cached at startup.

### KAI-DOCPARSEX-040 — MEDIUM — Filename/path-limit failure
A basename may exceed filesystem component limits before a controlled parser response is produced.

### KAI-DOCPARSEX-041 — MEDIUM — DXF diagnostics leak
Invalid-DXF exception text is placed directly in public HTTP detail.

### KAI-DOCPARSEX-042 — MEDIUM — Log-control injection
Raw filenames are interpolated into structured log messages without a canonical display schema.

### KAI-DOCPARSEX-043 — MEDIUM — Public misleading metrics
The service exposes `status=ok` plus an empty error budget without recording requests.

### KAI-DOCPARSEX-044 — MEDIUM — Incomplete optional-library startup handling
Native/library initialisation errors other than ImportError may crash or appear only during parsing.

### KAI-DOCPARSEX-045 — MEDIUM — No parser-version readiness
Health reports import Booleans but no versions or known-file verification.

### KAI-DOCPARSEX-046 — MEDIUM — Non-reproducible parser stack
All major parser dependencies use broad lower bounds; the base/system packages are not digest-locked.

### KAI-DOCPARSEX-047 — MEDIUM — Missing adversarial parser tests
No dedicated tests were found for the critical archive/converter/integrity cases.

### KAI-DOCPARSEX-048 — MEDIUM — Missing parser lifecycle
No lifespan owns worker isolation, temporary artefacts or graceful cancellation.

### KAI-DOCPARSEX-049 — MEDIUM — Missing extraction audit
No immutable event links principal, file digest, detected format, parser versions and output digest.

### KAI-DOCPARSEX-050 — MEDIUM — Missing chronology
Results contain no processing timestamp, source document time or sequence.

---

## Batch totals

- Findings: **50**
- Critical: **1**
- High: **28**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,648**
- Critical: **192**
- High: **1,333**
- Medium: **1,120**
- Low: **3**

## Files materially reviewed

`document-parser/app.py`, `document-parser/Dockerfile`, `document-parser/requirements.txt`, minimal deployment/Dashboard integration and the existing Document Parser audit.
