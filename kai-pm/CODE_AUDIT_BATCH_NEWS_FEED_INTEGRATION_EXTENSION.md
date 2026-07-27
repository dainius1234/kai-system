# Kai Code Audit — News Feed Integration Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_NEWS_FEED.md` or `CODE_AUDIT_BATCH_NEWS_FEED_EXTENSION.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NEWSI-001 | HIGH | All feed sources have equal downstream authority regardless of publisher trust, registration origin or corroboration |
| KAI-NEWSI-002 | HIGH | Feed text has no prompt-injection classification before Agentic and Cortex consume it |
| KAI-NEWSI-003 | HIGH | Article identity concatenates link and title without a delimiter, creating ambiguous hash inputs |
| KAI-NEWSI-004 | HIGH | Adding an existing feed ID silently overwrites its name, tags, fetch state and error history |
| KAI-NEWSI-005 | HIGH | No per-host quota prevents one origin or URL variants from dominating scheduled refresh traffic |
| KAI-NEWSI-006 | HIGH | Article cache publication has no aggregate generation-completeness state across all feeds |
| KAI-NEWSI-007 | MEDIUM | Hard-coded default publishers are activated without a versioned source-governance manifest |
| KAI-NEWSI-008 | MEDIUM | Feed/article text is not Unicode/control-character normalised before comparison and display |
| KAI-NEWSI-009 | MEDIUM | Search is raw substring matching with no language, token or relevance semantics |
| KAI-NEWSI-010 | MEDIUM | Article retention is count-based only and has no maximum age or source-revocation generation |
| KAI-NEWSI-011 | MEDIUM | Public metrics expose request telemetry without administrative authentication |
| KAI-NEWSI-012 | MEDIUM | Missing shared-runtime imports silently replace structured telemetry with no-op fallbacks |

---

### KAI-NEWSI-001 — HIGH — Flat source authority
**Issue:** Hard-coded feeds, environment feeds and anonymously registered feeds all publish articles with the same structure and downstream authority.  
**Risk:** An attacker-controlled or weakly trusted source is indistinguishable from an approved publisher when Agentic, Cortex or Dashboard consumes the article.  
**Recommendation:** attach a signed source-policy identity, trust tier, registration actor and corroboration state to every article.  
**Status:** OPEN

### KAI-NEWSI-002 — HIGH — News-to-prompt injection
Feed titles and summaries are not classified or quoted as untrusted external content before downstream model context use.  
**Risk:** A malicious article can contain instructions designed to redirect Kai’s behaviour rather than inform it.  
**Recommendation:** preserve source text as evidence-only data and prohibit it from acquiring system/instruction authority.  
**Status:** OPEN

### KAI-NEWSI-003 — HIGH — Ambiguous article-ID input
The UUID input is `link + title` with no delimiter or canonical encoding. Different pairs can produce the same concatenated string.

### KAI-NEWSI-004 — HIGH — Silent feed-state overwrite
`POST /feeds` assigns directly to `_feeds[fid]`, resetting prior metadata without an explicit update operation, revision check or audit event.

### KAI-NEWSI-005 — HIGH — Origin concentration is unbounded
Many feeds or canonical URL variants may target one host, consuming all sequential refresh time and provider/network capacity.

### KAI-NEWSI-006 — HIGH — No whole-generation completeness
The published article list does not state which feeds succeeded, failed or were omitted in that exact generation.

### KAI-NEWSI-007 — MEDIUM — Default-source governance absent
BBC, NYT and HN sources are embedded in code without a policy revision, signed manifest or operator approval record.

### KAI-NEWSI-008 — MEDIUM — Text canonicalisation absent
Names, tags, titles and summaries may contain bidi/control/confusable Unicode and are stored, compared and displayed raw.

### KAI-NEWSI-009 — MEDIUM — Weak search semantics
Search matches substrings inside raw title/summary content and supplies no relevance score or normalised visible-text basis.

### KAI-NEWSI-010 — MEDIUM — No age-based retention
Articles remain until count displacement or refresh replacement, and source deletion has no durable revocation generation.

### KAI-NEWSI-011 — MEDIUM — Public telemetry
`/metrics` is unauthenticated.

### KAI-NEWSI-012 — MEDIUM — Silent runtime downgrade
If `common.runtime` cannot import, basic logging and no-op ErrorBudget are substituted without readiness degradation.

---

## Batch totals

- Findings: **12**
- Critical: **0**
- High: **6**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,319**
- Critical: **189**
- High: **1,159**
- Medium: **968**
- Low: **3**

## Files materially reviewed

`news-feed/app.py`, both existing News Feed audit batches and Agentic/Cortex/Dashboard consumption paths.
