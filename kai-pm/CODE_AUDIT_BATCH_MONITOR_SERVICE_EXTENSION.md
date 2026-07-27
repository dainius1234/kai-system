# Kai Code Audit — Monitor Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_MONITOR_SERVICE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MONITORX-001 | CRITICAL | Scrape rules persistently read whichever shared Browser Agent page another workflow left open |
| KAI-MONITORX-002 | HIGH | The configured scrape URL is never navigated because Browser Agent `/scrape` ignores the posted body |
| KAI-MONITORX-003 | HIGH | The configured scrape selector is never applied |
| KAI-MONITORX-004 | HIGH | Scrape rules can alert on or disclose an unrelated authenticated browser session while displaying a different configured URL |
| KAI-MONITORX-005 | HIGH | Manual checks execute disabled rules and can still fetch sources and fire actions |
| KAI-MONITORX-006 | HIGH | Deleting a rule does not cancel an in-flight check, which may still deliver an alert after deletion |
| KAI-MONITORX-007 | HIGH | Disabling a rule does not cancel an in-flight check or action |
| KAI-MONITORX-008 | HIGH | Updating a rule mutates the same dictionary used by in-flight checks, producing mixed old/new execution |
| KAI-MONITORX-009 | HIGH | Clearing alerts erases operational evidence without authentication or durable retention |
| KAI-MONITORX-010 | HIGH | Rule creation has no total rule-count, per-caller or persistence-size limit |
| KAI-MONITORX-011 | HIGH | Generated rule IDs retain only 32 bits of UUID entropy |
| KAI-MONITORX-012 | HIGH | Source type is a free string rather than a strict `http|scrape` enum |
| KAI-MONITORX-013 | HIGH | Condition operator is a free string and unknown values silently evaluate false |
| KAI-MONITORX-014 | HIGH | Action names are unrestricted and unknown actions are silently ignored |
| KAI-MONITORX-015 | HIGH | Urgency is unrestricted and forwarded directly to Notify Service |
| KAI-MONITORX-016 | HIGH | Rule creation has no cross-field validation between operator and required value/text/percent fields |
| KAI-MONITORX-017 | HIGH | Empty `contains` text evaluates true for every value |
| KAI-MONITORX-018 | HIGH | Missing or negative percentage thresholds create overly permissive percentage-change alerts |
| KAI-MONITORX-019 | HIGH | Condition numbers accept non-finite values without a deterministic policy |
| KAI-MONITORX-020 | HIGH | `changed` compares string representations and can alert on semantically identical reordered objects |
| KAI-MONITORX-021 | HIGH | Source/condition updates retain the old baseline value and can create false change/percentage alerts |
| KAI-MONITORX-022 | HIGH | The current value is committed before condition/action success and becomes the next baseline after failures |
| KAI-MONITORX-023 | HIGH | Caller-controlled message templates can traverse attributes/indexes of formatted values |
| KAI-MONITORX-024 | HIGH | Unbounded monitored values and templates are expanded into Notify/TTS payloads and logs |
| KAI-MONITORX-025 | HIGH | Notify and TTS actions bypass Tool Gate and any exact-action approval |
| KAI-MONITORX-026 | HIGH | Source results have no response digest, source event time, service identity or freshness evidence |
| KAI-MONITORX-027 | HIGH | Monitoring traffic has no authenticated service identity, mTLS or signed response contract |
| KAI-MONITORX-028 | HIGH | `RULES_FILE` is empty by default, so “persistent” rules disappear on restart in the deployed configuration |
| KAI-MONITORX-029 | HIGH | No rules volume is mounted, so even a configured file is container-local unless separately changed |
| KAI-MONITORX-030 | MEDIUM | All enabled persisted rules begin polling immediately at startup without an activation review or migration revision |
| KAI-MONITORX-031 | MEDIUM | Rule-file writes synchronously serialise the complete unbounded ruleset on the event loop |
| KAI-MONITORX-032 | MEDIUM | HTTP polling supports JSON only and cannot distinguish an expected non-JSON source from a failed source contract |
| KAI-MONITORX-033 | MEDIUM | Extract paths stop early on a scalar and silently return a partially traversed value |
| KAI-MONITORX-034 | MEDIUM | Negative list indexes are accepted in extract paths |
| KAI-MONITORX-035 | MEDIUM | JSON keys containing dots cannot be addressed unambiguously |
| KAI-MONITORX-036 | MEDIUM | Message-format errors silently replace the configured template with a fallback message |
| KAI-MONITORX-037 | MEDIUM | Exact floating-point equality/inequality is used for alert conditions without tolerance semantics |
| KAI-MONITORX-038 | MEDIUM | Percentage-change conditions suppress every transition from a zero previous value |
| KAI-MONITORX-039 | MEDIUM | Manual checks return only “triggered” and provide no job ID, result or durable completion state |
| KAI-MONITORX-040 | MEDIUM | Scheduler timestamps are committed before checks complete, so actual polling cadence is not represented |
| KAI-MONITORX-041 | MEDIUM | The scheduler scans the complete unbounded rule dictionary every second |
| KAI-MONITORX-042 | MEDIUM | Alert `limit=0` returns one alert rather than zero |
| KAI-MONITORX-043 | MEDIUM | Status omits in-flight checks, last successful check, source freshness and delivery outcomes |
| KAI-MONITORX-044 | MEDIUM | Wall-clock changes alter interval and cooldown calculations |
| KAI-MONITORX-045 | MEDIUM | Public metrics expose an unpopulated reliability object without administrative authentication |
| KAI-MONITORX-046 | MEDIUM | FastAPI/HTTPX dependencies and the Python base image are not reproducibly digest-pinned |
| KAI-MONITORX-047 | MEDIUM | No dedicated tests were found for scrape-contract integration, in-flight deletion or delivery state |
| KAI-MONITORX-048 | MEDIUM | No tamper-evident audit links actor, rule revision, source response, condition result and action delivery |

---

## Critical finding

### KAI-MONITORX-001 — CRITICAL — Persistent cross-workflow browser scraping
**Issue:** Monitor posts `{url, selector}` to Browser Agent `/scrape`, but that endpoint accepts no request model/body and scrapes its single current page.  
**Risk:** A persistent rule can repeatedly capture text from whichever authenticated/internal page another Browser Agent workflow left open, then expose it through Monitor status/alerts/notifications while claiming a different configured source.  
**Recommendation:** Use an isolated authenticated browser context per rule and bind navigation, selector, page identity and extraction to one approved operation.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-MONITORX-002 — HIGH — Scrape URL is dead configuration
Browser Agent never navigates to the rule URL during Monitor’s scrape request.

### KAI-MONITORX-003 — HIGH — Selector is dead configuration
Browser Agent’s scrape implementation extracts `document.body.innerText` and fixed links; it does not read the selector sent by Monitor.

### KAI-MONITORX-004 — HIGH — False source attribution
Rule listings show one URL/selector while last values and alerts may derive from an unrelated global page.

### KAI-MONITORX-005 — HIGH — Disabled-rule manual execution
`manual_check()` calls `_check_rule()` without checking `enabled`.

### KAI-MONITORX-006 — HIGH — Delete-after-start race
An in-flight task retains the rule dictionary and can fire after all runtime maps and persistence entry were removed.

### KAI-MONITORX-007 — HIGH — Disable-after-start race
Disabling controls only future scheduling, not current source/action work.

### KAI-MONITORX-008 — HIGH — Mutable in-flight rule
Update applies dictionary fields in place while `_check_rule` references that object.

### KAI-MONITORX-009 — HIGH — Public alert-evidence destruction
`DELETE /alerts` clears the complete history and requires no authority.

### KAI-MONITORX-010 — HIGH — Unbounded rule cardinality
Any number of rules may be accepted and scanned/persisted.

### KAI-MONITORX-011 — HIGH — Short generated IDs
`str(uuid.uuid4())[:8]` provides only eight hexadecimal characters and is not a durable distributed identity.

### KAI-MONITORX-012 — HIGH — Source enum absent
Creation-time Pydantic validation still accepts arbitrary `type` values.

### KAI-MONITORX-013 — HIGH — Operator enum absent
Typos/unknown conditions fail silently rather than making a rule invalid/degraded.

### KAI-MONITORX-014 — HIGH — Action enum absent
A rule can be accepted while performing no configured action.

### KAI-MONITORX-015 — HIGH — Urgency contract absent
Arbitrary strings are sent to the downstream notification authority.

### KAI-MONITORX-016 — HIGH — Condition schema is not semantic
Numeric, textual and percentage operators may be created without their required operands.

### KAI-MONITORX-017 — HIGH — Empty substring matches everything
For a rule updated/persisted with `text=""`, `"" in sval` is always true.

### KAI-MONITORX-018 — HIGH — Percentage defaults are permissive
`float(percent or 0)` makes a missing threshold zero; negative values lower the trigger further.

### KAI-MONITORX-019 — HIGH — Non-finite alert state
NaN/infinity are not rejected for threshold/percent/current values and can produce non-portable or permanently false comparisons.

### KAI-MONITORX-020 — HIGH — String-change false positives
Dictionary key order or formatting differences trigger `changed` despite equivalent data.

### KAI-MONITORX-021 — HIGH — Baseline not reset on semantic update
Changing source, extraction or condition leaves `_last_value[rule_id]` from the previous rule definition.

### KAI-MONITORX-022 — HIGH — Failed processing advances state
`_last_value` is written before evaluation/action completion.

### KAI-MONITORX-023 — HIGH — Format-string data traversal
Python `str.format` allows attribute/index traversal from `{value...}`, `{name...}` and `{rule_id...}` rather than a restricted placeholder substitution.

### KAI-MONITORX-024 — HIGH — Unbounded action/log expansion
Large dictionaries/strings can be formatted, posted to two services and logged without a message-size cap.

### KAI-MONITORX-025 — HIGH — Action governance bypass
Monitor sends notify/TTS directly; no Gate approval is bound to message, urgency, recipient or rule revision.

### KAI-MONITORX-026 — HIGH — Missing source evidence identity
The current value is detached from exact response bytes, upstream timestamp and verified service identity.

### KAI-MONITORX-027 — HIGH — Unauthenticated service traffic
Polling/actions trust configured URLs and ordinary HTTP responses only.

### KAI-MONITORX-028 — HIGH — Persistence disabled by default
Minimal Compose sets `RULES_FILE` from an empty default and mounts no rules storage.

### KAI-MONITORX-029 — HIGH — Container-local persistence
A rules file inside the image/container filesystem is lost with container replacement unless an external volume is separately configured.

---

## Medium-severity findings

### KAI-MONITORX-030 — MEDIUM — Automatic activation
Loaded enabled rules begin executing immediately after startup, with no approved configuration revision or dry-run.

### KAI-MONITORX-031 — MEDIUM — Blocking complete-file save
CRUD requests synchronously rewrite the full rules list.

### KAI-MONITORX-032 — MEDIUM — JSON-only HTTP source
A valid text/CSV/HTML endpoint is always treated as an error rather than a typed source.

### KAI-MONITORX-033 — MEDIUM — Partial extraction succeeds
Traversal breaks on a scalar and returns it even when path components remain.

### KAI-MONITORX-034 — MEDIUM — Negative indexing
A path such as `data.-1.value` selects from the end of a list.

### KAI-MONITORX-035 — MEDIUM — Dot-key ambiguity
No escaping grammar exists for literal dots in object keys.

### KAI-MONITORX-036 — MEDIUM — Template failure hidden
The caller is not told that the configured message could not be rendered.

### KAI-MONITORX-037 — MEDIUM — Exact float comparisons
Normal representation noise can trigger or suppress `eq/ne` rules.

### KAI-MONITORX-038 — MEDIUM — Zero-baseline blind spot
Percentage rules always return false when previous is zero, even for a large meaningful increase.

### KAI-MONITORX-039 — MEDIUM — No manual-job state
A background task may fail after the endpoint reports success.

### KAI-MONITORX-040 — MEDIUM — Scheduling timestamp ambiguity
`_last_check` records dispatch time, not completion or source event time.

### KAI-MONITORX-041 — MEDIUM — O(n) scheduler tick
One-second scans scale linearly with attacker-created rules.

### KAI-MONITORX-042 — MEDIUM — Incorrect zero limit
The limit is clamped to at least one.

### KAI-MONITORX-043 — MEDIUM — Incomplete operational status
No active-task/delivery/freshness state is exposed.

### KAI-MONITORX-044 — MEDIUM — Wall-clock scheduling
System clock adjustments affect due/cooldown logic.

### KAI-MONITORX-045 — MEDIUM — Public misleading metrics
The existing unrecorded ErrorBudget is returned with `status=ok`.

### KAI-MONITORX-046 — MEDIUM — Non-reproducible service image
Dependencies and base artefacts are range/tag based.

### KAI-MONITORX-047 — MEDIUM — Missing integration/race tests
Repository search found no dedicated Monitor test suite covering these execution paths.

### KAI-MONITORX-048 — MEDIUM — Missing end-to-end audit
No immutable record joins caller, exact rule, source bytes, evaluation and confirmed action outcome.

---

## Batch totals

- Findings: **48**
- Critical: **1**
- High: **28**
- Medium: **19**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,752**
- Critical: **193**
- High: **1,395**
- Medium: **1,161**
- Low: **3**

## Files materially reviewed

`monitor-service/app.py`, `monitor-service/Dockerfile`, `monitor-service/requirements.txt`, minimal/full deployment configuration, Browser Agent’s actual scrape contract, Notify/TTS integrations and the existing Monitor Service audit.
