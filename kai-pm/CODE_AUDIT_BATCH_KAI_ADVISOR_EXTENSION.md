# Kai Code Audit — KAI Advisor Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_KAI_ADVISOR.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-ADVISORX-001 | HIGH | Filesystem walk order determines which matching document becomes the answer |
| KAI-ADVISORX-002 | HIGH | Symlinked `.md` files can load unintended readable files into the knowledge corpus |
| KAI-ADVISORX-003 | HIGH | The async `/ask` endpoint performs a full synchronous corpus scan on the event loop |
| KAI-ADVISORX-004 | HIGH | Very short/common queries can force large scans and arbitrary document-suffix disclosure |
| KAI-ADVISORX-005 | HIGH | Answers contain no source filename, passage bounds, corpus revision or content digest |
| KAI-ADVISORX-006 | HIGH | Raw Markdown/control text is returned without a safe display or downstream rendering contract |
| KAI-ADVISORX-007 | HIGH | No rate limit, caller quota or workload-admission control protects corpus scanning |
| KAI-ADVISORX-008 | MEDIUM | Knowledge chunks discard their original file paths and cannot be independently revoked or traced |
| KAI-ADVISORX-009 | MEDIUM | Corpus/file iteration is not sorted, making identical deployments potentially return different answers |
| KAI-ADVISORX-010 | MEDIUM | Query matching uses simple lowercase substring semantics rather than token or Unicode case-folding |
| KAI-ADVISORX-011 | MEDIUM | Only the first matching chunk and first case-sensitive split are used, with no ranking or ambiguity state |
| KAI-ADVISORX-012 | MEDIUM | The stub response reflects arbitrary question text into a trusted-advisor voice |
| KAI-ADVISORX-013 | MEDIUM | Corpus content is not validated as approved documentation rather than generated, secret or hostile Markdown |
| KAI-ADVISORX-014 | MEDIUM | The service exposes no immutable corpus-generation identity in health or answers |
| KAI-ADVISORX-015 | MEDIUM | No lifecycle-owned index, bounded worker pool or graceful query drain exists |

---

### KAI-ADVISORX-001 — HIGH — Non-deterministic first-source authority
**Issue:** The first chunk encountered by `os.walk()` containing the substring is returned. Directory/file traversal order is not sorted or governed.  
**Risk:** Filesystem ordering, packaging or a newly added document can silently change the authoritative answer to the same question.  
**Recommendation:** build a deterministic versioned index with explicit source priority and ambiguity handling.  
**Status:** OPEN

### KAI-ADVISORX-002 — HIGH — Symlinked Markdown corpus expansion
`open()` follows file symlinks. A `.md` symlink under the docs tree can expose any readable target file, independent of intended corpus scope.

### KAI-ADVISORX-003 — HIGH — Event-loop corpus scan
The async route loops through and lowercases complete document strings synchronously.

### KAI-ADVISORX-004 — HIGH — Common-query amplification/disclosure
One-character/common strings trigger full-corpus scans and may return nearly an entire document suffix under the existing extraction defect.

### KAI-ADVISORX-005 — HIGH — No answer provenance
A caller cannot identify the source file, matched offset, corpus version or exact bytes supporting an answer.

### KAI-ADVISORX-006 — HIGH — Unsafe raw answer format
Markdown, links, HTML/control characters and instruction-like content are returned as ordinary answer text with no safe rendering/provenance contract.

### KAI-ADVISORX-007 — HIGH — No workload admission
Anonymous callers can issue concurrent large scans/reflections with no quotas.

### KAI-ADVISORX-008 — MEDIUM — Source identity discarded
`knowledge` stores strings only, preventing per-document audit, access policy and revocation.

### KAI-ADVISORX-009 — MEDIUM — Unsorted corpus construction
Neither `dirs` nor `files` is sorted before reading.

### KAI-ADVISORX-010 — MEDIUM — Weak matching semantics
`lower()` substring search is language/normalisation insensitive and matches inside unrelated words.

### KAI-ADVISORX-011 — MEDIUM — No ambiguity/ranking
The service ignores all later matches and provides no relevance or conflicting-source state.

### KAI-ADVISORX-012 — MEDIUM — Stub authority reflection
The fallback quotes the caller question under a service branded as KAI Advisor, without clearly typed non-answer semantics beyond prose.

### KAI-ADVISORX-013 — MEDIUM — No corpus approval policy
Every readable `.md` under the path is trusted equally; generated audit files, drafts or secrets can enter the answer corpus.

### KAI-ADVISORX-014 — MEDIUM — Missing corpus identity
Health reports only a chunk count; answers contain no generation/digest.

### KAI-ADVISORX-015 — MEDIUM — Missing retrieval lifecycle
No prebuilt index, bounded worker, cancellation or graceful in-flight shutdown exists.

---

## Batch totals

- Findings: **15**
- Critical: **0**
- High: **7**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,404**
- Critical: **189**
- High: **1,200**
- Medium: **1,012**
- Low: **3**

## Files materially reviewed

`kai-advisor/app.py`, existing KAI Advisor audit findings and its deployment/package context.
