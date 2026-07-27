# Kai Code Audit — Test Stubs, External Model, OCR and Upload Fuzz Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/conftest.py`, `scripts/github_models_client.py`, `scripts/ocr_receipt.py` and `scripts/security_fuzz_upload.py`. Underlying Dashboard upload and memU/accounting defects remain in their existing batches and are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TESTEXT-001 | HIGH | scripts/conftest replaces real Redis whenever it has not already been imported, even if the package is installed |
| KAI-TESTEXT-002 | HIGH | The declared `importlib.util` dependency check is never used to decide whether Redis is missing |
| KAI-TESTEXT-003 | HIGH | Real Redis integration paths are silently converted into MagicMock behaviour for the entire scripts test tree |
| KAI-TESTEXT-004 | HIGH | The async Redis stub does not faithfully model connectivity failure, atomicity, expiry, pipelines or pub/sub |
| KAI-TESTEXT-005 | HIGH | Fake memU embeddings are enabled globally for every scripts test |
| KAI-TESTEXT-006 | HIGH | Vault Sync service modules are executed during pytest collection rather than inside controlled fixtures |
| KAI-TESTEXT-007 | HIGH | Import-time service side effects can access live files, Redis, memU URLs and environment configuration before tests begin |
| KAI-TESTEXT-008 | HIGH | Every exception during dynamic module execution is swallowed |
| KAI-TESTEXT-009 | HIGH | A module is inserted into `sys.modules` before execution and remains there after a failed partial import |
| KAI-TESTEXT-010 | HIGH | Service code can be imported twice under alias and canonical names, creating duplicate singleton state |
| KAI-TESTEXT-011 | HIGH | `vault-sync`, `common` and `memu-core` are prepended to `sys.path`, enabling module-shadowing and ambiguous imports |
| KAI-TESTEXT-012 | HIGH | Test collection success is prioritised over detecting missing declared production dependencies |
| KAI-TESTEXT-013 | HIGH | Redis-unavailable fallback tests cannot prove behaviour with the real Redis client library |
| KAI-TESTEXT-014 | HIGH | Subprocesses spawned by tests inherit fake-embedding and other global environment mutations |
| KAI-TESTEXT-015 | HIGH | The global Redis and module stubs are never restored or scoped per test |
| KAI-TESTEXT-016 | MEDIUM | Test outcome reports do not indicate that Redis and embeddings were replaced |
| KAI-TESTEXT-017 | MEDIUM | The stubbed Redis client has no deterministic data store or realistic return types |
| KAI-TESTEXT-018 | MEDIUM | MagicMock truthiness can make unimplemented Redis operations appear successful |
| KAI-TESTEXT-019 | MEDIUM | Async methods may return non-awaitable or unconstrained MagicMock values depending on call shape |
| KAI-TESTEXT-020 | MEDIUM | Import ordering determines whether real or fake Redis is used |
| KAI-TESTEXT-021 | MEDIUM | Path and module mutations affect every later test in the same interpreter |
| KAI-TESTEXT-022 | MEDIUM | No teardown removes alias modules, paths or environment variables |
| KAI-TESTEXT-023 | MEDIUM | Partial import failures produce no collection warning or failed test |
| KAI-TESTEXT-024 | MEDIUM | Test isolation has no configuration digest, source revision or structured bootstrap report |
| KAI-TESTEXT-025 | HIGH | GitHub Models availability accepts any token string of 20 characters or more as plausible |
| KAI-TESTEXT-026 | HIGH | Availability checks only a raw TCP connection and does not validate TLS, token permission or model access |
| KAI-TESTEXT-027 | HIGH | `query()` does not call `is_available()` and will send even short or invalid tokens |
| KAI-TESTEXT-028 | HIGH | Prompts and system instructions are sent to an external provider without PII, secret or repository-data filtering |
| KAI-TESTEXT-029 | HIGH | The automatic GitHub token may carry broader repository permissions than the model request requires |
| KAI-TESTEXT-030 | HIGH | Caller-selected model identifiers are accepted without an approved allowlist |
| KAI-TESTEXT-031 | HIGH | Prompt, system text and response bodies are unbounded |
| KAI-TESTEXT-032 | HIGH | Temperature, maximum tokens and timeout accept negative, non-finite and unsafe values |
| KAI-TESTEXT-033 | HIGH | The complete HTTP response is materialised without a byte or JSON-depth limit |
| KAI-TESTEXT-034 | HIGH | Response status success is followed by unchecked nested JSON indexing |
| KAI-TESTEXT-035 | HIGH | The returned model field repeats the requested model rather than the provider-confirmed served model |
| KAI-TESTEXT-036 | HIGH | Raw exception text is returned as model output in a normal response object |
| KAI-TESTEXT-037 | HIGH | `source="error"` output can be consumed as ordinary model text by weak callers |
| KAI-TESTEXT-038 | HIGH | Synchronous DNS, socket and HTTP calls block the test/async caller thread |
| KAI-TESTEXT-039 | HIGH | No rate-limit, retry, backoff, budget or GitHub Models quota control exists |
| KAI-TESTEXT-040 | HIGH | No data-purpose, repository, branch, actor or consent record accompanies external model requests |
| KAI-TESTEXT-041 | MEDIUM | Availability can be true while the API is unauthorised or the selected model is unavailable |
| KAI-TESTEXT-042 | MEDIUM | Availability performs network I/O on every call and has no cache or circuit breaker |
| KAI-TESTEXT-043 | MEDIUM | Token scope, expiry and source are never inspected |
| KAI-TESTEXT-044 | MEDIUM | Provider request IDs, usage, latency, finish reason and content filters are discarded |
| KAI-TESTEXT-045 | MEDIUM | Response source is a free string rather than a strict enum |
| KAI-TESTEXT-046 | MEDIUM | No exact endpoint/API revision is recorded in results |
| KAI-TESTEXT-047 | MEDIUM | The client has no audit log or prompt/response digest |
| KAI-TESTEXT-048 | MEDIUM | Test skips caused by provider unavailability can silently remove behavioural coverage |
| KAI-TESTEXT-049 | HIGH | Receipt OCR accepts any caller-selected readable input path |
| KAI-TESTEXT-050 | HIGH | Receipt output can overwrite any caller-accessible path or symlink target |
| KAI-TESTEXT-051 | HIGH | Missing OCR dependencies produce a successful £0.00 output file |
| KAI-TESTEXT-052 | HIGH | A missing image produces a successful £0.00 output file |
| KAI-TESTEXT-053 | HIGH | OCR or image-decoding failures are not converted into a controlled no-write failure |
| KAI-TESTEXT-054 | HIGH | The parser assumes the last decimal-looking number is the receipt total |
| KAI-TESTEXT-055 | HIGH | Dates, VAT values, card digits, item prices or change can be misclassified as the total |
| KAI-TESTEXT-056 | HIGH | Amount parsing ignores currency, sign, thousands separators and locale semantics |
| KAI-TESTEXT-057 | HIGH | Accounting values use binary floating-point arithmetic |
| KAI-TESTEXT-058 | HIGH | An extracted amount of zero is accepted without operator confirmation |
| KAI-TESTEXT-059 | HIGH | The script overwrites the CSV despite its docstring claiming it appends expenses |
| KAI-TESTEXT-060 | HIGH | Existing accounting history is destroyed on every run |
| KAI-TESTEXT-061 | HIGH | Full receipt OCR text containing names, addresses, card fragments and purchase details is stored in plaintext |
| KAI-TESTEXT-062 | HIGH | OCR text and source path can trigger spreadsheet formula execution when the CSV is opened |
| KAI-TESTEXT-063 | HIGH | Images are opened and decoded without byte, pixel, dimension or decompression-bomb limits |
| KAI-TESTEXT-064 | HIGH | Tesseract inference has no deadline or process cancellation contract |
| KAI-TESTEXT-065 | HIGH | Complete OCR output is materialised before a 2,000-character storage truncation |
| KAI-TESTEXT-066 | HIGH | No receipt hash, merchant, transaction date, currency or duplicate identifier is retained |
| KAI-TESTEXT-067 | MEDIUM | Output directories are created recursively without an approved accounting root |
| KAI-TESTEXT-068 | MEDIUM | Output writes are non-atomic and unlocked |
| KAI-TESTEXT-069 | MEDIUM | File permissions and ownership are controlled only by the caller’s umask |
| KAI-TESTEXT-070 | MEDIUM | Image type and file signature are not validated before PIL/Tesseract processing |
| KAI-TESTEXT-071 | MEDIUM | Truncated raw text may omit the line supporting the selected amount |
| KAI-TESTEXT-072 | MEDIUM | No confidence score or OCR provenance is stored |
| KAI-TESTEXT-073 | MEDIUM | No manual-review, correction or reconciliation workflow exists |
| KAI-TESTEXT-074 | MEDIUM | Success output does not prove that OCR ran or that the amount is valid |
| KAI-TESTEXT-075 | MEDIUM | No actor, input digest or accounting-write audit event is produced |
| KAI-TESTEXT-076 | HIGH | Upload “security” tests explicitly accept path-traversal filenames being forwarded downstream |
| KAI-TESTEXT-077 | HIGH | Upload tests explicitly accept shell-script content and media type being forwarded downstream |
| KAI-TESTEXT-078 | HIGH | Null-byte and 4,096-character filenames need only avoid HTTP 500 rather than be rejected |
| KAI-TESTEXT-079 | HIGH | Empty files and binary garbage need only avoid HTTP 500 rather than be rejected |
| KAI-TESTEXT-080 | HIGH | The exactly-at-limit test accepts HTTP 503 as a passing result |
| KAI-TESTEXT-081 | HIGH | The suite claims oversized content is rejected before forwarding but does not assert that OCR was never called |
| KAI-TESTEXT-082 | HIGH | Tests do not validate magic bytes, safe filename canonicalisation or accepted media types |
| KAI-TESTEXT-083 | HIGH | Dashboard application code is executed during test collection through dynamic import |
| KAI-TESTEXT-084 | HIGH | The suite bypasses production authentication, reverse proxy, network and body-streaming boundaries |
| KAI-TESTEXT-085 | HIGH | A global patch of `httpx.AsyncClient` can intercept unrelated Dashboard HTTP calls during each test |
| KAI-TESTEXT-086 | HIGH | Fixed hand-picked cases are labelled fuzzing despite no generated or mutation-based input corpus |
| KAI-TESTEXT-087 | HIGH | No archive bombs, image decompression bombs, malformed multipart boundaries or chunked-body cases are tested |
| KAI-TESTEXT-088 | HIGH | No concurrent upload, slow upload or resource-exhaustion scenario is tested |
| KAI-TESTEXT-089 | MEDIUM | The test allocates multiple 10 MB payloads in the Python process |
| KAI-TESTEXT-090 | MEDIUM | `_MAX_BYTES` duplicates the production constant and can drift independently |
| KAI-TESTEXT-091 | MEDIUM | Rejection responses are not checked for information leakage |
| KAI-TESTEXT-092 | MEDIUM | The passthrough tests do not validate response bodies or redaction |
| KAI-TESTEXT-093 | MEDIUM | The happy-path mock accepts arbitrary fake PNG bytes and a synthetic OCR response |
| KAI-TESTEXT-094 | MEDIUM | Downstream received filename, content type and bytes are not asserted |
| KAI-TESTEXT-095 | MEDIUM | Multiple-file, duplicate-field and missing-content-length cases are not tested |
| KAI-TESTEXT-096 | MEDIUM | Filename Unicode normalisation and separator variants are not tested |
| KAI-TESTEXT-097 | MEDIUM | Test module path injection can shadow installed modules |
| KAI-TESTEXT-098 | MEDIUM | Test outcomes contain no source SHA, dependency versions or test-environment digest |
| KAI-TESTEXT-099 | MEDIUM | There is no property asserting that rejected requests leave no downstream or local side effect |
| KAI-TESTEXT-100 | MEDIUM | A passing suite certifies availability and non-crashing behaviour rather than a secure upload policy |

---

## Test bootstrap — `scripts/conftest.py`

### KAI-TESTEXT-001 — HIGH — Real Redis replacement
The condition checks `sys.modules`, not whether Redis is installed; at normal collection time the real package is usually not imported yet.

### KAI-TESTEXT-002 — HIGH — Unused dependency detection
`importlib.util` is imported but `find_spec("redis")` is never used.

### KAI-TESTEXT-003 — HIGH — Integration semantics replaced
All later `import redis` and `import redis.asyncio` calls receive MagicMocks.

### KAI-TESTEXT-004 — HIGH — Unrealistic async/distributed behaviour
The stub cannot model real Redis guarantees or failures.

### KAI-TESTEXT-005 — HIGH — Fake embeddings suite-wide
`MEMU_ALLOW_FAKE_EMBEDDINGS=true` is set globally.

### KAI-TESTEXT-006 — HIGH — Collection-time service execution
Four Vault Sync modules execute before tests.

### KAI-TESTEXT-007 — HIGH — Side-effect boundary absent
Module imports may construct clients, load mappings and inspect environment/files.

### KAI-TESTEXT-008 — HIGH — Errors hidden
Every execution exception is ignored.

### KAI-TESTEXT-009 — HIGH — Broken module retained
The module object is registered before `exec_module()`.

### KAI-TESTEXT-010 — HIGH — Duplicate singleton universes
Alias and canonical imports can both exist.

### KAI-TESTEXT-011 — HIGH — Path shadowing
Three repository directories are inserted at index zero.

### KAI-TESTEXT-012 — HIGH — Missing dependency hidden
Tests collect rather than fail when Redis is absent.

### KAI-TESTEXT-013 — HIGH — Fallback proof invalid
Fallback is exercised against a MagicMock implementation, not the production client.

### KAI-TESTEXT-014 — HIGH — Child inheritance
Environment changes propagate to subprocesses.

### KAI-TESTEXT-015 — HIGH — No restoration
Global stubs remain for the session.

### KAI-TESTEXT-016 — MEDIUM — Assurance mode invisible
Reports do not identify substitutions.

### KAI-TESTEXT-017 — MEDIUM — No data semantics
The mock has no stable Redis state.

### KAI-TESTEXT-018 — MEDIUM — Truthy mocks
Unconfigured operations can pass conditions.

### KAI-TESTEXT-019 — MEDIUM — Async shape ambiguity
Return/await behaviour is not API-faithful.

### KAI-TESTEXT-020 — MEDIUM — Import-order dependence
Pre-importing Redis changes suite behaviour.

### KAI-TESTEXT-021 — MEDIUM — Session contamination
Paths/modules affect unrelated tests.

### KAI-TESTEXT-022 — MEDIUM — No teardown
Nothing is removed.

### KAI-TESTEXT-023 — MEDIUM — Partial load hidden
No failed test is generated.

### KAI-TESTEXT-024 — MEDIUM — Bootstrap provenance absent
No machine-readable state record.

---

## GitHub Models client — `scripts/github_models_client.py`

### KAI-TESTEXT-025 — HIGH — Token-length authentication heuristic
Twenty arbitrary characters pass the plausibility check.

### KAI-TESTEXT-026 — HIGH — TCP reachability is not availability
No TLS handshake, API authentication or model check occurs.

### KAI-TESTEXT-027 — HIGH — Availability bypass
`query()` sends any non-empty token.

### KAI-TESTEXT-028 — HIGH — Unfiltered external egress
Prompts may contain private code, logs or test data.

### KAI-TESTEXT-029 — HIGH — Broad token reuse
The ambient GitHub token is used directly.

### KAI-TESTEXT-030 — HIGH — Arbitrary model selection
Environment/caller model strings are accepted.

### KAI-TESTEXT-031 — HIGH — Unbounded request/response
No aggregate limits.

### KAI-TESTEXT-032 — HIGH — Unsafe generation parameters
No validation.

### KAI-TESTEXT-033 — HIGH — Full response materialisation
Requests reads complete content.

### KAI-TESTEXT-034 — HIGH — Weak response schema
Nested fields are indexed directly.

### KAI-TESTEXT-035 — HIGH — Requested identity returned as served identity
Provider response model is ignored.

### KAI-TESTEXT-036 — HIGH — Error disclosure as text
Raw exception enters the result.

### KAI-TESTEXT-037 — HIGH — Failure content usable as output
Normal dataclass shape is returned.

### KAI-TESTEXT-038 — HIGH — Blocking calls
Socket/requests are synchronous.

### KAI-TESTEXT-039 — HIGH — Quota handling absent
No retry/budget/backoff.

### KAI-TESTEXT-040 — HIGH — Request governance absent
No actor/purpose/consent record.

### KAI-TESTEXT-041 — MEDIUM — False positive availability
Permission/model may still fail.

### KAI-TESTEXT-042 — MEDIUM — Repeated network probe
No cache/breaker.

### KAI-TESTEXT-043 — MEDIUM — Token properties unknown
No scope/expiry check.

### KAI-TESTEXT-044 — MEDIUM — Provider metadata lost
Usage/request ID/finish reason discarded.

### KAI-TESTEXT-045 — MEDIUM — Free source string
No enum.

### KAI-TESTEXT-046 — MEDIUM — API revision absent
Result is not reproducible.

### KAI-TESTEXT-047 — MEDIUM — Audit absent
No digest/event.

### KAI-TESTEXT-048 — MEDIUM — Silent coverage reduction
Unavailable provider encourages test skips.

---

## Receipt OCR — `scripts/ocr_receipt.py`

### KAI-TESTEXT-049 — HIGH — Arbitrary source path
Any accessible file is passed to PIL.

### KAI-TESTEXT-050 — HIGH — Arbitrary destructive output
Any writable file/symlink can be replaced.

### KAI-TESTEXT-051 — HIGH — Missing OCR becomes £0 success
Dependency import failure is silent.

### KAI-TESTEXT-052 — HIGH — Missing image becomes £0 success
No existence error.

### KAI-TESTEXT-053 — HIGH — Processing error handling absent
Decode/OCR exceptions abort after possible setup and no controlled accounting result.

### KAI-TESTEXT-054 — HIGH — Last-number heuristic
No total label or structure is required.

### KAI-TESTEXT-055 — HIGH — Receipt-number ambiguity
Other decimal values may win.

### KAI-TESTEXT-056 — HIGH — Incomplete monetary parser
Currency/locale/sign/thousands are absent.

### KAI-TESTEXT-057 — HIGH — Float accounting
Binary float is stored.

### KAI-TESTEXT-058 — HIGH — Zero accepted
No review gate.

### KAI-TESTEXT-059 — HIGH — Append claim false
File mode is `w`.

### KAI-TESTEXT-060 — HIGH — History destruction
Prior CSV rows are lost.

### KAI-TESTEXT-061 — HIGH — Sensitive OCR retention
Raw text is stored.

### KAI-TESTEXT-062 — HIGH — Spreadsheet injection
Cells are not neutralised for spreadsheet formula semantics.

### KAI-TESTEXT-063 — HIGH — Image resource bounds absent
No pre-decode constraints.

### KAI-TESTEXT-064 — HIGH — OCR deadline absent
No timeout/cancellation.

### KAI-TESTEXT-065 — HIGH — Post-OCR truncation only
Full output exists first.

### KAI-TESTEXT-066 — HIGH — Transaction identity absent
No merchant/date/hash/duplicate control.

### KAI-TESTEXT-067 — MEDIUM — Unapproved directory creation
Parent paths are created.

### KAI-TESTEXT-068 — MEDIUM — Unsafe write durability
No temp file/lock.

### KAI-TESTEXT-069 — MEDIUM — File protection absent
Umask only.

### KAI-TESTEXT-070 — MEDIUM — File-type validation absent
Extension/magic not checked.

### KAI-TESTEXT-071 — MEDIUM — Supporting text may be lost
2,000-character cutoff.

### KAI-TESTEXT-072 — MEDIUM — OCR confidence absent
No evidence quality.

### KAI-TESTEXT-073 — MEDIUM — Correction workflow absent
No review/reconcile.

### KAI-TESTEXT-074 — MEDIUM — Misleading success
“wrote” does not prove OCR.

### KAI-TESTEXT-075 — MEDIUM — Audit absent
No actor/input/output event.

---

## Upload fuzz test — `scripts/security_fuzz_upload.py`

### KAI-TESTEXT-076 — HIGH — Traversal forwarding certified
The test expects only no HTTP 500.

### KAI-TESTEXT-077 — HIGH — Shell content forwarding certified
The test says this is intentional.

### KAI-TESTEXT-078 — HIGH — Dangerous filenames need not be rejected
Null/long names only need avoid 500.

### KAI-TESTEXT-079 — HIGH — Invalid files need not be rejected
Empty/garbage only need avoid 500.

### KAI-TESTEXT-080 — HIGH — Broken exact-limit test may pass
HTTP 503 is accepted.

### KAI-TESTEXT-081 — HIGH — Pre-forward rejection unproven
OCR call count is not asserted.

### KAI-TESTEXT-082 — HIGH — Policy checks absent
No approved type/name validation tests.

### KAI-TESTEXT-083 — HIGH — Collection-time app execution
Dashboard module runs at import.

### KAI-TESTEXT-084 — HIGH — Production boundary bypass
TestClient is in-process.

### KAI-TESTEXT-085 — HIGH — Broad client patch
Global AsyncClient constructor is replaced.

### KAI-TESTEXT-086 — HIGH — Not fuzzing
Corpus is fixed.

### KAI-TESTEXT-087 — HIGH — Complex malicious formats absent
No bombs/multipart mutations.

### KAI-TESTEXT-088 — HIGH — Resource/concurrency attacks absent
No slow/concurrent tests.

### KAI-TESTEXT-089 — MEDIUM — Test memory amplification
Large payloads are allocated directly.

### KAI-TESTEXT-090 — MEDIUM — Constant drift
Limit is duplicated.

### KAI-TESTEXT-091 — MEDIUM — Error leakage untested
Only status is checked.

### KAI-TESTEXT-092 — MEDIUM — Passthrough body unvalidated
Redaction/schema ignored.

### KAI-TESTEXT-093 — MEDIUM — Fake file happy path
No real PNG/OCR.

### KAI-TESTEXT-094 — MEDIUM — Downstream request unasserted
Forwarded data is not inspected.

### KAI-TESTEXT-095 — MEDIUM — Multipart variants absent
One file/field only.

### KAI-TESTEXT-096 — MEDIUM — Unicode filename cases absent
No canonicalisation corpus.

### KAI-TESTEXT-097 — MEDIUM — Import-path shadowing
Repository root is prepended.

### KAI-TESTEXT-098 — MEDIUM — Environment provenance absent
No dependency/source digest.

### KAI-TESTEXT-099 — MEDIUM — Side-effect absence unproven
No state assertion.

### KAI-TESTEXT-100 — MEDIUM — Security claim overstated
The suite tests non-crashing and proxy status mapping, not a secure upload boundary.

---

## Batch totals

- Findings: **100**
- Critical: **0**
- High: **59**
- Medium: **41**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,909**
- Critical: **182**
- High: **1,486**
- Medium: **1,238**
- Low: **3**

## Files materially reviewed

`scripts/conftest.py`, `scripts/github_models_client.py`, `scripts/ocr_receipt.py`, `scripts/security_fuzz_upload.py`, with current Dashboard/memU/test bootstrap context used to evaluate actual assurance coverage.
