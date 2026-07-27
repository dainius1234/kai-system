# Kai Code Audit — Baseline, Behavioural Scoreboard and Feedback Tools Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/capture_baseline_responses.py`, `scripts/behavioral_scoreboard.py`, `scripts/operator_feedback.py` and `scripts/feedback_summary.py`. Existing Agentic and memU API defects are not duplicated; this batch records the distinct behaviour of these operational clients.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BEHAVTOOL-001 | HIGH | Baseline capture accepts an arbitrary unvalidated host and sends chat requests to it |
| KAI-BEHAVTOOL-002 | HIGH | Baseline requests contain no authentication, operator identity or approved benchmark purpose |
| KAI-BEHAVTOOL-003 | HIGH | Running the benchmark against Agentic mutates sessions and global `keeper` memory |
| KAI-BEHAVTOOL-004 | HIGH | Repeated runs use predictable session IDs and merge with prior benchmark state |
| KAI-BEHAVTOOL-005 | HIGH | Benchmark-created memory/session records are never cleaned up or isolated |
| KAI-BEHAVTOOL-006 | HIGH | The output is labelled as a 0.5B baseline without verifying the exact loaded model |
| KAI-BEHAVTOOL-007 | HIGH | Model health metadata is accepted without artefact digest, backend identity or configuration proof |
| KAI-BEHAVTOOL-008 | HIGH | Streaming response content is accumulated without a byte or token limit |
| KAI-BEHAVTOOL-009 | HIGH | The socket timeout is not a total benchmark deadline and a continuously streaming response can run indefinitely |
| KAI-BEHAVTOOL-010 | HIGH | Non-JSON stream lines are accepted as model response text |
| KAI-BEHAVTOOL-011 | HIGH | Response content type and SSE schema are not validated |
| KAI-BEHAVTOOL-012 | HIGH | Model responses are written directly into tracked Markdown without sanitisation or provenance boundaries |
| KAI-BEHAVTOOL-013 | HIGH | The baseline file is overwritten non-atomically without locking or backup |
| KAI-BEHAVTOOL-014 | HIGH | Baseline output omits route, source, exact model, prompt/context digest, verifier result and backend configuration |
| KAI-BEHAVTOOL-015 | HIGH | The five public fixed prompts can be memorised or overfit without measuring general quality |
| KAI-BEHAVTOOL-016 | MEDIUM | Complete prompts and responses are printed to terminal output |
| KAI-BEHAVTOOL-017 | MEDIUM | A caller-selected session prefix is unbounded and not canonicalised |
| KAI-BEHAVTOOL-018 | MEDIUM | The chosen host URL is persisted in the tracked baseline document |
| KAI-BEHAVTOOL-019 | MEDIUM | One request failure aborts the full capture without a partial-result report |
| KAI-BEHAVTOOL-020 | MEDIUM | No per-prompt latency, token usage, response hash or completion status is recorded |
| KAI-BEHAVTOOL-021 | MEDIUM | The manual 1/2 scoring guide conflates “worse” and “same” into one score |
| KAI-BEHAVTOOL-022 | MEDIUM | The pass condition uses subjective comparison without authoritative expected answers |
| KAI-BEHAVTOOL-023 | MEDIUM | Output file writing omits explicit encoding, fsync and source-commit identity |
| KAI-BEHAVTOOL-024 | MEDIUM | Concurrent captures can overwrite one another and mix benchmark sessions |
| KAI-BEHAVTOOL-025 | MEDIUM | Health failure returns model `unknown` but capture continues under the 0.5B filename and heading |
| KAI-BEHAVTOOL-026 | HIGH | The behavioural scoreboard always exits with status zero regardless of grade or failure |
| KAI-BEHAVTOOL-027 | HIGH | Complete LLM unavailability is explicitly treated as a successful process exit |
| KAI-BEHAVTOOL-028 | HIGH | A factually wrong response can receive 100/100 |
| KAI-BEHAVTOOL-029 | HIGH | Score depends only on non-emptiness, marker absence, length and latency |
| KAI-BEHAVTOOL-030 | HIGH | No expected answer, calculation, citation or safety property is checked for any prompt |
| KAI-BEHAVTOOL-031 | HIGH | A response meeting only three superficial criteria is labelled PASS at 75/100 |
| KAI-BEHAVTOOL-032 | HIGH | The same selected specialist is used for all general, maths, construction, memory and safety prompts |
| KAI-BEHAVTOOL-033 | HIGH | If Ollama is unavailable, the first configured backend is selected without task capability validation |
| KAI-BEHAVTOOL-034 | HIGH | Static known prompts make the scoreboard vulnerable to benchmark-specific responses |
| KAI-BEHAVTOOL-035 | HIGH | Error/stub detection uses a short string-marker list and misses other failure wording |
| KAI-BEHAVTOOL-036 | HIGH | `source="live"` is accepted as proof of a real, correct model response without backend identity verification |
| KAI-BEHAVTOOL-037 | HIGH | The scoreboard is advisory by design and cannot block a regression |
| KAI-BEHAVTOOL-038 | HIGH | The scoreboard does not verify the model or configuration it claims to assess |
| KAI-BEHAVTOOL-039 | HIGH | Shared LLM telemetry side effects occur during assessment and are not included in the result |
| KAI-BEHAVTOOL-040 | MEDIUM | The five requests run sequentially and can consume the sum of backend timeouts |
| KAI-BEHAVTOOL-041 | MEDIUM | A raised router/import exception prevents a complete structured scorecard |
| KAI-BEHAVTOOL-042 | MEDIUM | Overall score uses integer floor division and discards fractional information |
| KAI-BEHAVTOOL-043 | MEDIUM | All prompts and criteria receive equal weight regardless of factual or safety impact |
| KAI-BEHAVTOOL-044 | MEDIUM | No deterministic seed, temperature, repeat count or variance estimate is recorded |
| KAI-BEHAVTOOL-045 | MEDIUM | Output contains no source commit, policy revision, backend URL or model digest |
| KAI-BEHAVTOOL-046 | MEDIUM | Score history is not persisted or compared with a calibrated baseline |
| KAI-BEHAVTOOL-047 | MEDIUM | The scoreboard does not preserve response text or hashes needed to audit the awarded scores |
| KAI-BEHAVTOOL-048 | MEDIUM | Latency is treated as quality without controlling network, load or warm-up state |
| KAI-BEHAVTOOL-049 | MEDIUM | The construction and safety questions are not checked against current authoritative sources |
| KAI-BEHAVTOOL-050 | HIGH | Operator feedback sends an incompatible memU payload whose `content` field is ignored |
| KAI-BEHAVTOOL-051 | HIGH | memU can return 200 after storing an empty feedback memory and the tool reports “Feedback logged” |
| KAI-BEHAVTOOL-052 | HIGH | Feedback is assigned the generic caller-controlled identity `operator`, not an authenticated keeper principal |
| KAI-BEHAVTOOL-053 | HIGH | The feedback message is exposed in shell history and process arguments |
| KAI-BEHAVTOOL-054 | HIGH | Feedback length is unbounded |
| KAI-BEHAVTOOL-055 | HIGH | The client uses a hard-coded unauthenticated plaintext localhost endpoint |
| KAI-BEHAVTOOL-056 | HIGH | Any local process/user able to run the script can impersonate the operator |
| KAI-BEHAVTOOL-057 | HIGH | Non-200 responses do not trigger fallback persistence and the feedback is lost |
| KAI-BEHAVTOOL-058 | HIGH | Network exceptions fall back to an unencrypted plaintext log |
| KAI-BEHAVTOOL-059 | HIGH | Failure paths still terminate with a successful process exit |
| KAI-BEHAVTOOL-060 | HIGH | Fallback log writes are unlocked and concurrent feedback can interleave |
| KAI-BEHAVTOOL-061 | HIGH | Embedded newlines allow forged fallback-log entries |
| KAI-BEHAVTOOL-062 | HIGH | The fallback has no rotation, retention, permissions or secure deletion policy |
| KAI-BEHAVTOOL-063 | HIGH | No memU record ID, verdict, category or committed revision is verified or retained |
| KAI-BEHAVTOOL-064 | MEDIUM | The fallback path depends on the caller’s working directory |
| KAI-BEHAVTOOL-065 | MEDIUM | Missing `output/` causes the fallback write itself to fail |
| KAI-BEHAVTOOL-066 | MEDIUM | Raw memU failure bodies are printed to the terminal |
| KAI-BEHAVTOOL-067 | MEDIUM | Timestamp identity has only one-second resolution and uses the local host clock |
| KAI-BEHAVTOOL-068 | MEDIUM | No idempotency key prevents duplicate feedback submission |
| KAI-BEHAVTOOL-069 | MEDIUM | Feedback lacks source session, affected response, context and explicit operator confirmation evidence |
| KAI-BEHAVTOOL-070 | MEDIUM | The promised manual-review fallback has no ingestion queue or replay mechanism |
| KAI-BEHAVTOOL-071 | MEDIUM | Script logic executes at import time |
| KAI-BEHAVTOOL-072 | HIGH | Feedback Summary calls a `/memory/query` endpoint that does not exist in current memU Core |
| KAI-BEHAVTOOL-073 | HIGH | HTTP 404/non-200 responses do not activate the documented local-log fallback |
| KAI-BEHAVTOOL-074 | HIGH | The summary tool prints complete memory content and local logs without redaction |
| KAI-BEHAVTOOL-075 | HIGH | The memU query contains no authenticated user scope and would aggregate global records if implemented |
| KAI-BEHAVTOOL-076 | HIGH | Complete HTTP results and fallback files are materialised without size limits |
| KAI-BEHAVTOOL-077 | HIGH | Memory/log text can inject terminal control sequences into operator output |
| KAI-BEHAVTOOL-078 | HIGH | All failure and empty-data outcomes still exit successfully |
| KAI-BEHAVTOOL-079 | MEDIUM | The tool does not summarise; it dumps raw event dictionaries and files |
| KAI-BEHAVTOOL-080 | MEDIUM | Local fallback paths depend on the current working directory |
| KAI-BEHAVTOOL-081 | MEDIUM | A 200 response with missing or empty `results` is reported as a successful empty summary |
| KAI-BEHAVTOOL-082 | MEDIUM | Response JSON and event structures are not schema validated |
| KAI-BEHAVTOOL-083 | MEDIUM | Local feedback and self-audit files are read in full with no retention window |
| KAI-BEHAVTOOL-084 | MEDIUM | No source, trust tier, record ID or event provenance is displayed consistently |
| KAI-BEHAVTOOL-085 | MEDIUM | The hard-coded localhost URL has no configuration, TLS or service-identity validation |
| KAI-BEHAVTOOL-086 | MEDIUM | Script logic executes immediately on import |
| KAI-BEHAVTOOL-087 | MEDIUM | No structured machine-readable output or nonzero failure contract exists |
| KAI-BEHAVTOOL-088 | MEDIUM | Running the summary leaves no audit record of who accessed the private feedback data |

---

## Baseline capture — `scripts/capture_baseline_responses.py`

### KAI-BEHAVTOOL-001 — HIGH — Arbitrary destination
`--host` is concatenated into health/chat URLs without an approved-host or scheme policy.

### KAI-BEHAVTOOL-002 — HIGH — No benchmark identity
Requests carry only message and session ID.

### KAI-BEHAVTOOL-003 — HIGH — Benchmark mutates the subject
Agentic chat appends session turns and auto-memorises exchanges under the global keeper identity.

### KAI-BEHAVTOOL-004 — HIGH — Predictable shared sessions
The default prefix and prompt IDs are reused on every run.

### KAI-BEHAVTOOL-005 — HIGH — No cleanup
The script never clears sessions or removes benchmark memories.

### KAI-BEHAVTOOL-006 — HIGH — Unverified model label
The output path/title claims 0.5B even if health says unknown or another backend serves the response.

### KAI-BEHAVTOOL-007 — HIGH — Missing artefact proof
A health string does not establish the exact model digest, quantisation, prompt template or backend.

### KAI-BEHAVTOOL-008 — HIGH — Unbounded streamed accumulation
Every token is appended to a Python list.

### KAI-BEHAVTOOL-009 — HIGH — No total deadline
The read timeout applies to socket operations, not an overall capture budget.

### KAI-BEHAVTOOL-010 — HIGH — Raw-line acceptance
Malformed/non-JSON lines become response content.

### KAI-BEHAVTOOL-011 — HIGH — Stream contract absent
Status/content type/event schema and completion metadata are not validated.

### KAI-BEHAVTOOL-012 — HIGH — Markdown injection/document poisoning
Generated output is inserted verbatim into a tracked Markdown artefact.

### KAI-BEHAVTOOL-013 — HIGH — Unsafe overwrite
The complete baseline file is replaced directly.

### KAI-BEHAVTOOL-014 — HIGH — Incomplete benchmark evidence
The recorded metadata cannot reproduce or audit the result.

### KAI-BEHAVTOOL-015 — HIGH — Fixed public corpus
Only five known prompts are used.

### KAI-BEHAVTOOL-016 — MEDIUM — Terminal disclosure
Prompt and complete response text are printed.

### KAI-BEHAVTOOL-017 — MEDIUM — Unbounded session input
The prefix has no length or character validation.

### KAI-BEHAVTOOL-018 — MEDIUM — Host persistence
The endpoint string is committed into the report.

### KAI-BEHAVTOOL-019 — MEDIUM — All-or-nothing run
One URL error exits before later prompts/report writing.

### KAI-BEHAVTOOL-020 — MEDIUM — No quantitative call metadata
Latency and usage are discarded.

### KAI-BEHAVTOOL-021 — MEDIUM — Ambiguous scoring scale
One score combines worse and unchanged.

### KAI-BEHAVTOOL-022 — MEDIUM — Subjective pass rule
Human impressions replace expected outputs and statistical comparison.

### KAI-BEHAVTOOL-023 — MEDIUM — Weak file provenance/durability
No explicit encoding, source SHA or fsync.

### KAI-BEHAVTOOL-024 — MEDIUM — Concurrent collision
Runs share file and default session names.

### KAI-BEHAVTOOL-025 — MEDIUM — Unknown model still accepted
Health failure does not stop capture.

---

## Behavioural scoreboard — `scripts/behavioral_scoreboard.py`

### KAI-BEHAVTOOL-026 — HIGH — Non-enforcing exit
The final line always calls `sys.exit(0)`.

### KAI-BEHAVTOOL-027 — HIGH — Offline success
Stub mode and no live responses both return zero and then exit zero.

### KAI-BEHAVTOOL-028 — HIGH — Wrong-answer maximum
Correctness is never inspected.

### KAI-BEHAVTOOL-029 — HIGH — Surface-only quality
The four 25-point dimensions are formatting/availability metrics.

### KAI-BEHAVTOOL-030 — HIGH — No ground truth
Even the multiplication and current regulatory answers are not checked.

### KAI-BEHAVTOOL-031 — HIGH — PASS without correctness
Three non-correctness criteria suffice.

### KAI-BEHAVTOOL-032 — HIGH — No specialist coverage
One backend answers every domain.

### KAI-BEHAVTOOL-033 — HIGH — First-backend fallback
List ordering selects the replacement.

### KAI-BEHAVTOOL-034 — HIGH — Benchmark overfitting
The prompt set is source-visible and constant.

### KAI-BEHAVTOOL-035 — HIGH — Incomplete failure vocabulary
Only four substrings identify stubs/errors.

### KAI-BEHAVTOOL-036 — HIGH — Source label as proof
`live` does not establish identity or correctness.

### KAI-BEHAVTOOL-037 — HIGH — Advisory-only regression signal
The script explicitly cannot gate CI/deployment.

### KAI-BEHAVTOOL-038 — HIGH — Subject identity absent
No exact model/configuration is asserted.

### KAI-BEHAVTOOL-039 — HIGH — Assessment changes telemetry
Shared LLM logging runs but is absent from score semantics.

### KAI-BEHAVTOOL-040 — MEDIUM — Sequential cumulative latency
All prompts run one after another.

### KAI-BEHAVTOOL-041 — MEDIUM — Incomplete exception handling
A raised call can abort the report.

### KAI-BEHAVTOOL-042 — MEDIUM — Precision loss
Integer floor division computes overall score.

### KAI-BEHAVTOOL-043 — MEDIUM — Equal risk weighting
Safety and trivial arithmetic contribute equally.

### KAI-BEHAVTOOL-044 — MEDIUM — No repeatability controls
No seed, retries or variance.

### KAI-BEHAVTOOL-045 — MEDIUM — Missing environment provenance
No source/config/backend digest.

### KAI-BEHAVTOOL-046 — MEDIUM — No durable trend
Results exist only in stdout/weekly issue capture.

### KAI-BEHAVTOOL-047 — MEDIUM — Award evidence unavailable
Response bodies/hashes are not preserved.

### KAI-BEHAVTOOL-048 — MEDIUM — Load-dependent quality
Latency points mix model quality with transient infrastructure state.

### KAI-BEHAVTOOL-049 — MEDIUM — Stale-fact risk
Current regulatory/safety facts are judged only for style.

---

## Operator feedback — `scripts/operator_feedback.py`

### KAI-BEHAVTOOL-050 — HIGH — Payload-schema mismatch
Current memU `MemoryUpdate` expects `result_raw`; the script sends `content`.

### KAI-BEHAVTOOL-051 — HIGH — Empty-memory false success
Pydantic ignores the extra field, `result_raw` remains empty and memU can append an empty record with HTTP 200.

### KAI-BEHAVTOOL-052 — HIGH — Unauthenticated generic identity
`user_id="operator"` is a body assertion, not the keeper identity used by planning.

### KAI-BEHAVTOOL-053 — HIGH — Command-line privacy exposure
Sensitive guidance appears in shell/process history.

### KAI-BEHAVTOOL-054 — HIGH — Unbounded feedback
No body or CLI size limit exists.

### KAI-BEHAVTOOL-055 — HIGH — Fixed insecure transport
The URL is plaintext localhost and not configurable.

### KAI-BEHAVTOOL-056 — HIGH — Local impersonation
No OS-user, token or signature verification exists.

### KAI-BEHAVTOOL-057 — HIGH — Non-200 data loss
Fallback executes only on exceptions.

### KAI-BEHAVTOOL-058 — HIGH — Plaintext fallback
Full feedback is appended to an ordinary file.

### KAI-BEHAVTOOL-059 — HIGH — Failure exit is zero
No `sys.exit()` failure path follows either error case.

### KAI-BEHAVTOOL-060 — HIGH — Concurrent append race
No file lock or atomic record framing.

### KAI-BEHAVTOOL-061 — HIGH — Log-entry injection
Feedback newlines are written verbatim.

### KAI-BEHAVTOOL-062 — HIGH — Missing fallback governance
No limits or protection.

### KAI-BEHAVTOOL-063 — HIGH — Commit/result not verified
A status code alone is treated as persistence success.

### KAI-BEHAVTOOL-064 — MEDIUM — CWD-dependent path
The fallback is relative.

### KAI-BEHAVTOOL-065 — MEDIUM — Missing-directory failure
Parent creation is absent.

### KAI-BEHAVTOOL-066 — MEDIUM — Raw backend disclosure
Failure response text is printed.

### KAI-BEHAVTOOL-067 — MEDIUM — Weak event identity
Second-resolution local event time only.

### KAI-BEHAVTOOL-068 — MEDIUM — Replay duplication
No idempotency key.

### KAI-BEHAVTOOL-069 — MEDIUM — Missing feedback target
No response/session/decision reference.

### KAI-BEHAVTOOL-070 — MEDIUM — No fallback replay
The comment says manual review only.

### KAI-BEHAVTOOL-071 — MEDIUM — Import side effect
CLI processing and network/filesystem operations occur at module import.

---

## Feedback summary — `scripts/feedback_summary.py`

### KAI-BEHAVTOOL-072 — HIGH — Dead endpoint
Current memU exposes no `/memory/query` route.

### KAI-BEHAVTOOL-073 — HIGH — Documented fallback does not run on 404
Non-200 is handled inside the successful request branch.

### KAI-BEHAVTOOL-074 — HIGH — Private raw-data dump
Memory and local file content is printed verbatim.

### KAI-BEHAVTOOL-075 — HIGH — No principal boundary
The proposed query has event filters but no user identity.

### KAI-BEHAVTOOL-076 — HIGH — Unbounded materialisation
HTTP JSON and local files have no size cap.

### KAI-BEHAVTOOL-077 — HIGH — Terminal injection
Stored text controls terminal output.

### KAI-BEHAVTOOL-078 — HIGH — Failure is successful process completion
No nonzero exit is produced.

### KAI-BEHAVTOOL-079 — MEDIUM — No summarisation
The tool only prints records.

### KAI-BEHAVTOOL-080 — MEDIUM — CWD-dependent logs
Both paths are relative.

### KAI-BEHAVTOOL-081 — MEDIUM — Empty 200 looks valid
No required-result check.

### KAI-BEHAVTOOL-082 — MEDIUM — Response schema absent
Arbitrary dictionaries/lists are assumed.

### KAI-BEHAVTOOL-083 — MEDIUM — Full historical file read
No date/window filtering applies locally.

### KAI-BEHAVTOOL-084 — MEDIUM — Incomplete provenance
Records are displayed without trust/source identity.

### KAI-BEHAVTOOL-085 — MEDIUM — Insecure fixed service URL
No configuration or service authentication.

### KAI-BEHAVTOOL-086 — MEDIUM — Import side effect
Execution begins at import.

### KAI-BEHAVTOOL-087 — MEDIUM — No machine-readable result
Only terminal text is emitted.

### KAI-BEHAVTOOL-088 — MEDIUM — No access audit
Reading private feedback leaves no evidence.

---

## Batch totals

- Findings: **88**
- Critical: **0**
- High: **50**
- Medium: **38**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,356**
- Critical: **181**
- High: **1,170**
- Medium: **1,002**
- Low: **3**

## Files materially reviewed

`scripts/capture_baseline_responses.py`, `scripts/behavioral_scoreboard.py`, `scripts/operator_feedback.py`, `scripts/feedback_summary.py`, with target-schema and side-effect confirmation against current Agentic/memU sources and existing CI integration.
