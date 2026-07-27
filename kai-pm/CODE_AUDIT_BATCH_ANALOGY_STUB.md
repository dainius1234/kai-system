# Kai Code Audit — Analogical Reasoning Stub Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-ANALOGY-001 | HIGH | Analogical reasoning is permanently unavailable regardless of feature-flag state |
| KAI-ANALOGY-002 | HIGH | The feature flag is imported from the wrong top-level module path and import failure is silently ignored |
| KAI-ANALOGY-003 | HIGH | The operation returns a normal `Analogy` object rather than a typed unavailable result |
| KAI-ANALOGY-004 | HIGH | The class claims graph search but accepts no graph client, graph snapshot or concept-node evidence |
| KAI-ANALOGY-005 | HIGH | The class claims embedding similarity but accepts no embedding model or similarity results |
| KAI-ANALOGY-006 | HIGH | The injected LLM callable is never used |
| KAI-ANALOGY-007 | HIGH | Source and target domains are not checked for non-emptiness, distinctness or known concept identity |
| KAI-ANALOGY-008 | HIGH | Stub natural language can be propagated as a proposed solution despite no mapping or reasoning |
| KAI-ANALOGY-009 | MEDIUM | Source and target text are unbounded and retained in the result object |
| KAI-ANALOGY-010 | MEDIUM | Stub output echoes sensitive caller text fragments without normalisation or escaping |
| KAI-ANALOGY-011 | MEDIUM | Fifty-character truncation creates collisions between distinct domains |
| KAI-ANALOGY-012 | MEDIUM | Empty structural mappings and graph paths can be mistaken for evaluated empty results |
| KAI-ANALOGY-013 | MEDIUM | Mapping relation fields and confidence values have no enum, range or provenance validation |
| KAI-ANALOGY-014 | MEDIUM | The future activation interface has no actor, purpose, timeout, token budget or concurrency contract |
| KAI-ANALOGY-015 | MEDIUM | No operation ID, graph revision, model digest or durable audit event exists |

---

### KAI-ANALOGY-001 — HIGH — Permanently disabled capability
`can_find()` always returns false after any feature check.

### KAI-ANALOGY-002 — HIGH — Feature authority failure hidden
The code imports `feature_flags`, not the repository’s `common.feature_flags`, and suppresses ImportError.

### KAI-ANALOGY-003 — HIGH — Unavailability hidden in a domain result
Callers receive an ordinary `Analogy` instance.

### KAI-ANALOGY-004 — HIGH — Graph architecture absent
No graph dependency or node evidence is represented in the API.

### KAI-ANALOGY-005 — HIGH — Embedding architecture absent
No embedding/search dependency is represented.

### KAI-ANALOGY-006 — HIGH — Dead model dependency
`llm_chat_fn` is stored and never invoked.

### KAI-ANALOGY-007 — HIGH — Domain validity absent
Identical, empty or arbitrary strings are accepted.

### KAI-ANALOGY-008 — HIGH — Stub solution masquerades as output
`proposed_solution` receives descriptive prose even though no analogy was found.

### KAI-ANALOGY-009 — MEDIUM — Unbounded inputs
No character or aggregate limits exist.

### KAI-ANALOGY-010 — MEDIUM — Input reflection
The stub includes caller text.

### KAI-ANALOGY-011 — MEDIUM — Truncation collision
Only 50 characters per domain appear in the stub.

### KAI-ANALOGY-012 — MEDIUM — Unevaluated empties
Empty list fields lack an unavailable/not-evaluated marker.

### KAI-ANALOGY-013 — MEDIUM — Result schema weak
Relations and confidence are free values.

### KAI-ANALOGY-014 — MEDIUM — Future workload/governance contract absent
No security or capacity context is accepted.

### KAI-ANALOGY-015 — MEDIUM — Provenance absent
No traceable evidence/result record exists.

---

## Batch totals

- Findings: **15**
- Critical: **0**
- High: **8**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,924**
- Critical: **182**
- High: **1,494**
- Medium: **1,245**
- Low: **3**

## Files materially reviewed

`agentic/analogy.py`, with repository searches confirming no live production call sites beyond test coverage.
