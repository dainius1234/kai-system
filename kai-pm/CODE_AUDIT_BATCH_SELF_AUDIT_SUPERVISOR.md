# Kai Code Audit — Self-Audit and Autonomous Supervisor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-OPS-001 | CRITICAL | Supervisor autonomously rewrites repository scripts using blind text replacement |
| KAI-OPS-002 | HIGH | Supervisor approval gating is based on crude sentiment rather than action risk |
| KAI-OPS-003 | HIGH | Supervisor treats empty or failed memory retrieval as a healthy system |
| KAI-OPS-004 | HIGH | Autonomous edits are non-atomic and bypass source-control isolation and verification |
| KAI-OPS-005 | HIGH | Supervisor logs action success without checking HTTP delivery or formatter result |
| KAI-OPS-006 | MEDIUM | Self-audit reports all checks passed when result objects lack return codes |
| KAI-OPS-007 | MEDIUM | Self-audit persistence is non-atomic and overwrites prior evidence |
| KAI-OPS-008 | MEDIUM | Self-audit lesson delivery ignores unsuccessful HTTP responses |
| KAI-OPS-009 | MEDIUM | Audit and supervisor logs can contain raw test output and internal errors |
| KAI-OPS-010 | MEDIUM | Supervisor embeds static external references as if they support each action |

---

## Autonomous supervisor: `scripts/kai_supervisor.py`

### KAI-OPS-001 — CRITICAL — Autonomous blind repository rewrites
**Issue:** `auto_apply_improvements` scans every Python file under `scripts/` and directly rewrites source. For stub handling it performs global string replacements removing `TODO`, `pass  # stub` and the token `NotImplementedError` without AST-aware semantics.  
**Risk:** Executable code, exception classes, comments and strings can be corrupted across the repository. The script can transform intentional safety stops or incomplete implementations into syntactically valid but unsafe behaviour.  
**Recommendation:** Disable autonomous source mutation. Require isolated branches, structured patches, tests, human review and signed approval before merge.  
**Status:** OPEN — immediate remediation required

### KAI-OPS-002 — HIGH — Approval gating uses operator sentiment instead of action impact
**Issue:** Automatic changes are blocked only when a simple keyword counter labels recent messages as `negative`. Neutral or positive sentiment permits edits.  
**Risk:** High-impact source changes can proceed because operator language lacks negative keywords, while unrelated frustration can block safe work. Sentiment is not an authorisation control.  
**Recommendation:** Gate by explicit authenticated approval, change scope, test evidence and policy-defined impact.  
**Status:** OPEN

### KAI-OPS-003 — HIGH — Retrieval failure becomes “system healthy”
**Issue:** `get_recent_events` returns an empty list on any request or parsing failure. `analyze_events` then emits `No immediate improvements detected. System is healthy.`  
**Risk:** Loss of memory connectivity or malformed responses is converted into a positive health conclusion and can still lead to autonomous formatting or experimentation.  
**Recommendation:** Distinguish no findings from unavailable evidence and fail closed for automated action.  
**Status:** OPEN

### KAI-OPS-004 — HIGH — Edits bypass transactional and source-control safeguards
**Issue:** Files are overwritten directly with `write_text`; no temporary branch, clean-tree check, backup, atomic replacement, lint/test gate or rollback snapshot is required.  
**Risk:** Partial writes, concurrent modifications and bad transformations can damage the active working tree and running system.  
**Recommendation:** Execute proposals in disposable worktrees, generate reviewable diffs and merge only after mandatory checks and approval.  
**Status:** OPEN

### KAI-OPS-005 — HIGH — Action logging overstates success
**Issue:** Memory logging calls do not inspect response status. `black` is run with `check=False`, after which the script records `auto_formatting` regardless of a non-zero exit code.  
**Risk:** Audit records can claim actions succeeded when delivery or formatting failed.  
**Recommendation:** Validate each outcome and log verified result states with traceable evidence.  
**Status:** OPEN

### KAI-OPS-010 — MEDIUM — Static references are presented as action rationale evidence
**Issue:** Every supervisor action defaults to the same two URLs, regardless of whether they support the specific change.  
**Risk:** Audit records create an appearance of evidence-backed action without actual source relevance or retrieval verification.  
**Recommendation:** Attach only verified, action-specific evidence and record retrieval provenance.  
**Status:** OPEN

---

## Self-audit: `scripts/self_audit.py`

### KAI-OPS-006 — MEDIUM — Exceptions can be summarised as passing checks
**Issue:** `run_make` returns `{target, error}` on exceptions. `summarize_results` evaluates `r.get("returncode", 0)`, so missing return codes default to zero and the failed check is omitted. If all checks raise, the summary becomes `All checks passed.`  
**Risk:** Tool absence, timeout or execution failure can produce a false clean audit.  
**Recommendation:** Treat any missing return code or `error` field as failure.  
**Status:** OPEN

### KAI-OPS-007 — MEDIUM — Self-audit overwrites prior evidence non-atomically
**Issue:** Every run writes directly to the same `output/self_audit_log.json`.  
**Risk:** Historical audit evidence is lost and interruption can leave a corrupt current report.  
**Recommendation:** Write immutable uniquely identified reports with atomic durable publication and an index.  
**Status:** OPEN

### KAI-OPS-008 — MEDIUM — Lesson delivery ignores HTTP outcome
**Issue:** `requests.post` to memu-core is not checked with `raise_for_status` or response validation.  
**Risk:** Lessons can be rejected while the script completes as though they were recorded.  
**Recommendation:** Validate delivery and persist retryable outbox records.  
**Status:** OPEN

### KAI-OPS-009 — MEDIUM — Raw outputs and errors are persisted
**Issue:** The self-audit stores the tail of stdout, stderr and exception text directly in JSON; the supervisor also stores arbitrary event-derived issue text.  
**Risk:** Secrets, paths, tokens or sensitive operational details printed by commands can enter broadly accessible audit and memory stores.  
**Recommendation:** Apply structural redaction, classification and access control before persistence.  
**Status:** OPEN

---

## Batch totals

- Findings: **10**
- Critical: **1**
- High: **4**
- Medium: **5**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **203**
- Critical: **26**
- High: **92**
- Medium: **84**
- Low: **1**

## Files materially reviewed in this batch

`scripts/kai_supervisor.py`, `scripts/self_audit.py`.
