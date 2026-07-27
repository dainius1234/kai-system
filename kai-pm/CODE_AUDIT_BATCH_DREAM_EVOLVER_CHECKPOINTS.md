# Kai Code Audit — Dream, Evolver and Checkpoint Internals

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

Public checkpoint endpoint and restore findings already recorded in `CODE_AUDIT_BATCH_AGENTIC_API.md` are not duplicated here.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-LEARN-001 | HIGH | Failure classification trusts unauthenticated caller-supplied episode outcomes |
| KAI-LEARN-002 | HIGH | Classification ordering can hide verifier contradiction behind low conviction |
| KAI-LEARN-003 | HIGH | One failed episode can generate a durable absolute “always/never” rule |
| KAI-LEARN-004 | HIGH | Caller-controlled episode text is embedded into metacognitive rules |
| KAI-LEARN-005 | MEDIUM | Topic extraction uses alphabetically sorted words rather than salient context |
| KAI-LEARN-006 | HIGH | Learning-value output can fall below zero for out-of-range conviction |
| KAI-LEARN-007 | MEDIUM | Learning value rewards rethink count without proving learning or correction |
| KAI-LEARN-008 | HIGH | Preference extraction converts lexical differences into operator preferences |
| KAI-LEARN-009 | HIGH | Preference extraction loses negation, order, entities and semantic context |
| KAI-LEARN-010 | HIGH | Knowledge-boundary analysis trusts self-reported outcomes and conviction |
| KAI-LEARN-011 | MEDIUM | Knowledge topics collapse to the first three alphabetically sorted words |
| KAI-LEARN-012 | MEDIUM | Knowledge-boundary numerical inputs and minimum sample threshold are unvalidated |
| KAI-LEARN-013 | HIGH | Dream cycles combine episodes without authenticated principal/session isolation |
| KAI-LEARN-014 | HIGH | Rule deduplication keeps the longest lexical rule, not the most correct or recent rule |
| KAI-LEARN-015 | HIGH | Contradiction detection produces false conflicts because `should` matches inside `should not` |
| KAI-LEARN-016 | HIGH | Dream confidence is derived from occurrence count rather than independent evidence |
| KAI-LEARN-017 | HIGH | Rising conviction is represented as learning even without improved outcomes |
| KAI-LEARN-018 | HIGH | Dream “actionable” insights are produced without verification or approval |
| KAI-LEARN-019 | HIGH | Dream cycles return success even when persistence fails |
| KAI-LEARN-020 | MEDIUM | Dream persistence is unsigned, plaintext and non-atomic |
| KAI-LEARN-021 | MEDIUM | Dream persistence/load failures are silently suppressed |
| KAI-LEARN-022 | MEDIUM | Dream rule comparison has quadratic cost and no input bounds |
| KAI-LEARN-023 | MEDIUM | Dream cycle identifiers are time-derived rather than collision-resistant operation IDs |
| KAI-LEARN-024 | HIGH | Evolver treats any non-`unknown` failure class as failure even when outcome is successful |
| KAI-LEARN-025 | HIGH | Evolver accepts arbitrary caller-defined failure classes |
| KAI-LEARN-026 | HIGH | Evolver groups by failure class only despite claiming class-plus-topic grouping |
| KAI-LEARN-027 | HIGH | One suggestion combines unrelated failures and emits a generic behaviour change |
| KAI-LEARN-028 | HIGH | Evolver suggestion IDs depend only on failure class and collide across topics/reports |
| KAI-LEARN-029 | HIGH | Evolver priority is driven by unverified frequency and self-reported labels |
| KAI-LEARN-030 | HIGH | Evolver confidence rises as average self-reported conviction falls |
| KAI-LEARN-031 | HIGH | Evolver recommendations are promoted into actionable DreamInsights without review |
| KAI-LEARN-032 | MEDIUM | Evolver report persistence is mutable, unsigned and non-atomic |
| KAI-LEARN-033 | MEDIUM | Evolver persistence/load failures are silently suppressed |
| KAI-LEARN-034 | MEDIUM | Failure-analysis inputs and numerical values are unbounded and weakly validated |
| KAI-LEARN-035 | HIGH | Checkpoint IDs are sanitised non-injectively and can collide |
| KAI-LEARN-036 | HIGH | An empty sanitised checkpoint ID resolves to the shared `.json` filename |
| KAI-LEARN-037 | HIGH | Checkpoint contents, labels and triggers have no schema or aggregate size bounds |
| KAI-LEARN-038 | HIGH | Zero or negative checkpoint retention can delete every checkpoint |
| KAI-LEARN-039 | HIGH | Checkpoint eviction uses mutable filesystem mtime rather than embedded chronology |
| KAI-LEARN-040 | HIGH | Checkpoint files use ordinary mutable temporary storage with no ownership hardening |
| KAI-LEARN-041 | MEDIUM | Corrupt checkpoints silently disappear from listing and load operations |
| KAI-LEARN-042 | MEDIUM | Checkpoint deserialisation accepts invalid breaker/guard/budget structures |
| KAI-LEARN-043 | MEDIUM | Checkpoint list limits and diff inputs are unvalidated |
| KAI-LEARN-044 | MEDIUM | Checkpoint cap, create, list, load and delete operations are concurrency-unsafe |

---

## Failure learning and preferences: `agentic/kai_config.py`

### KAI-LEARN-001 — HIGH — Failure labels are self-asserted
**Issue:** `classify_failure()` directly trusts caller-supplied outcome, conviction, verifier verdict, context count, override and gate dictionaries. No authenticated evaluator or immutable episode identity is required.  
**Risk:** poisoned episodes become the foundation for future warnings, rules and evolution suggestions.  
**Recommendation:** classify only independently evaluated immutable outcomes.  
**Status:** OPEN

### KAI-LEARN-002 — HIGH — Specific verifier failure can be masked
**Issue:** conviction/rethink checks occur before verifier verdict checks. A contradicted episode with low conviction and two rethinks is labelled `CONFIDENCE_LOW`, not `CONTRADICTED_BY_EVIDENCE`.  
**Risk:** future planning receives the wrong remediation rule and may merely gather more context rather than treat the claim as contradicted.  
**Recommendation:** use an explicit multi-cause record or severity-prioritised taxonomy.  
**Status:** OPEN

### KAI-LEARN-003 — HIGH — Single-event absolute rules
**Issue:** `extract_metacognitive_rule()` turns one classified episode into language such as “always check” and “never” without recurrence, operator approval or outcome corroboration.  
**Risk:** one erroneous/attacker-created failure permanently constrains future planning.  
**Recommendation:** keep hypotheses provisional until repeated independently verified outcomes and operator review.  
**Status:** OPEN

### KAI-LEARN-004 — HIGH — Episode text enters durable rule instructions
**Issue:** topic words derived from caller-controlled input are interpolated into behavioural rules.  
**Risk:** stored prompt injection and misleading topic labels enter trusted planning constraints.  
**Recommendation:** use a controlled ontology and quote untrusted evidence separately.  
**Status:** OPEN

### KAI-LEARN-005 — MEDIUM — Topic identity is arbitrary
Topic words are deduplicated, alphabetically sorted and truncated to five, discarding order and salience.

### KAI-LEARN-006 — HIGH — Learning value violates its 0–1 contract
**Issue:** conviction is not clamped. For values far outside 0–10, computed uncertainty becomes negative and the returned value can be below zero; NaN propagates.  
**Risk:** learning prioritisation and persistence receive invalid numerical evidence.  
**Recommendation:** reject non-finite/out-of-range inputs and enforce both lower and upper bounds.  
**Status:** OPEN

### KAI-LEARN-007 — MEDIUM — Rethinking is intrinsically rewarded
Up to 0.2 is added solely from rethink count, even if every rethink repeats the same failed reasoning.

### KAI-LEARN-008 — HIGH — Lexical correction becomes a personal preference
**Issue:** any word-set difference between original output and correction creates a statement that “keeper prefers/wants/does NOT want” the differing words.  
**Risk:** edits for factual accuracy, grammar or context become durable personality/preferences.  
**Recommendation:** require explicit authenticated preference confirmation.  
**Status:** OPEN

### KAI-LEARN-009 — HIGH — Preference semantics are lost
Set subtraction discards word order, negation, quantities and entity relationships, then alphabetically joins isolated words into a preference claim.

### KAI-LEARN-010 — HIGH — Competence map uses self-scored history
Successes, failures, conviction and learning value are taken directly from episodes without evaluator provenance or calibration.

### KAI-LEARN-011 — MEDIUM — Topic clusters are collision-prone
Only the first three alphabetically sorted words define the topic key, causing unrelated requests to share boundaries.

### KAI-LEARN-012 — MEDIUM — Boundary inputs are unsafe
NaN/malformed scores can abort or distort averages; zero/negative `min_episodes` accepts unsupported clusters.

---

## Dream consolidation: `agentic/kai_config.py`

### KAI-LEARN-013 — HIGH — Cross-user dream synthesis
**Issue:** `run_dream_cycle()` receives an ordinary episode list and performs no authenticated principal, session, purpose or consent partition.  
**Risk:** one user’s failures/preferences can produce behavioural insights applied to another.  
**Recommendation:** partition learning by principal and approved purpose.  
**Status:** OPEN

### KAI-LEARN-014 — HIGH — Longest rule wins
Deduplication sorts rules by length and preserves the longest member of each word-overlap cluster, irrespective of accuracy, provenance, recency or operator approval.

### KAI-LEARN-015 — HIGH — Broken contradiction heuristic
**Issue:** the test `"should" in r1` is true for `"should not"`; two similarly negative rules can be labelled contradictory.  
**Risk:** valid aligned rules generate false high-confidence conflict insights.  
**Recommendation:** parse propositions/polarity structurally.  
**Status:** OPEN

### KAI-LEARN-016 — HIGH — Count creates confidence
Pattern, cluster and boundary confidence increases mechanically with episode count; duplicate or poisoned records are not deduplicated by event/source.

### KAI-LEARN-017 — HIGH — Confidence inflation is called learning
A rise in self-reported conviction over time creates a “Learning detected” insight even when outcomes do not improve.

### KAI-LEARN-018 — HIGH — Unverified insights are marked actionable
Heuristic strings about struggling topics, contradictions and boundaries receive `actionable=True` without independent evaluation.

### KAI-LEARN-019 — HIGH — Persistence is not part of success
`run_dream_cycle()` returns a normal cycle after `save_dream_cycle()` silently fails.

### KAI-LEARN-020 — MEDIUM — Weak dream storage
The complete history is directly rewritten as plaintext JSON, normally under `/tmp`, without atomic replacement, locking, signature or fsync.

### KAI-LEARN-021 — MEDIUM — Dream evidence gaps disappear
Save/load exceptions are suppressed and loads return an empty list, making corruption indistinguishable from no dream history.

### KAI-LEARN-022 — MEDIUM — Unbounded quadratic comparison
Rule contradiction detection compares every rule pair and all episode/rule strings are unbounded.

### KAI-LEARN-023 — MEDIUM — Weak cycle identity
Default cycle ID is a truncated hash of wall-clock time and is not tied to the episode set or a durable operation record.

---

## Agent Evolver: `agentic/kai_config.py`

### KAI-LEARN-024 — HIGH — Successful episodes can be classified as failures
**Issue:** the filter includes an episode when `failure_class != "unknown"` even if `outcome_score >= 0.5`.  
**Risk:** stale/malformed failure labels turn successes into failure evidence and recommendations.  
**Recommendation:** require a validated evaluator outcome and consistent schema.  
**Status:** OPEN

### KAI-LEARN-025 — HIGH — Unknown failure labels are accepted
Arbitrary strings become groups, priorities and generic recommendations; no FailureClass enum validation occurs.

### KAI-LEARN-026 — HIGH — Claimed topic subgrouping is absent
The implementation groups only by `failure_class`; `_extract_topic` is run once across every episode in the class.

### KAI-LEARN-027 — HIGH — Unrelated failures produce one fix
A class spanning unrelated subjects yields one dominant-word topic and one generic recommendation applied across them.

### KAI-LEARN-028 — HIGH — Suggestion IDs collide
`suggestion_id` hashes only the failure-class string, so every report/topic for the same class receives the same ID.

### KAI-LEARN-029 — HIGH — Frequency creates severity
Two/three/five caller-provided records mechanically produce medium/high/critical priorities without unique-event or source validation.

### KAI-LEARN-030 — HIGH — Lower confidence raises Evolver confidence
The formula adds `(10 - avg_conviction) * 0.02`, so low self-confidence increases confidence in the generated fix, without demonstrating causal correctness.

### KAI-LEARN-031 — HIGH — Suggestions become actionable dream instructions
Every Evolver suggestion is converted directly to an `actionable=True` DreamInsight and may influence later behaviour.

### KAI-LEARN-032 — MEDIUM — Weak report storage
Reports are read-modify-written to mutable plaintext JSON with no atomicity, signature or worker coordination.

### KAI-LEARN-033 — MEDIUM — Report loss/corruption is hidden
Save errors are ignored; load errors return an empty history.

### KAI-LEARN-034 — MEDIUM — Analysis values are unbounded
Episode count/text, conviction, outcomes and configured minimum count accept invalid/extreme values; malformed records can abort the report.

---

## Checkpoint internals: `agentic/kai_config.py`

### KAI-LEARN-035 — HIGH — Sanitisation creates ID collisions
**Issue:** all characters outside `[A-Za-z0-9_-]` are removed. Distinct IDs such as `a/b`, `a.b` and `ab` resolve to the same file.  
**Risk:** load/delete operations can target a different checkpoint than the supplied logical ID.  
**Recommendation:** require exact generated IDs and reject any noncanonical value.  
**Status:** OPEN

### KAI-LEARN-036 — HIGH — Empty ID maps to a shared hidden file
An ID containing only removed characters resolves to `CHECKPOINT_DIR/.json`, enabling ambiguous load/delete behaviour.

### KAI-LEARN-037 — HIGH — Snapshot content is unbounded and untyped
Labels, triggers, nested breaker/guard/budget dictionaries and overrides are accepted and serialised without schema, size or sensitivity limits.

### KAI-LEARN-038 — HIGH — Retention can delete all evidence
`CHECKPOINT_MAX=0` deletes every checkpoint after creation; negative values continue deleting until `pop(0)` raises after the list is empty.

### KAI-LEARN-039 — HIGH — Retention order is filesystem-mutable
Eviction/listing uses file mtime rather than the embedded timestamp or an immutable sequence. Copying/touching a file changes which evidence is retained.

### KAI-LEARN-040 — HIGH — Checkpoints use weak temporary storage
The default directory is `/tmp`; files are plaintext, ordinary mutable files with no ownership/mode hardening or independent integrity anchor.

### KAI-LEARN-041 — MEDIUM — Corruption is silently omitted
Listing skips invalid files and loading returns `None`, so missing/corrupt evidence appears equivalent to an unknown ID.

### KAI-LEARN-042 — MEDIUM — Deserialisation is permissive
`from_dict` accepts arbitrary nested types for operational state and does not validate legal breaker states, guard fields, budget values or override strings.

### KAI-LEARN-043 — MEDIUM — Query/diff work is unbounded
Negative/extreme list limits have Python-slice semantics; deeply nested/large dictionaries and override lists are compared and returned wholesale.

### KAI-LEARN-044 — MEDIUM — Filesystem operations race
Create, retention deletion, listing, loading and deletion use no lock/revision/atomic transaction; concurrent calls can delete or read partially written files.

---

## Batch totals

- Findings: **44**
- Critical: **0**
- High: **26**
- Medium: **18**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,267**
- Critical: **100**
- High: **540**
- Medium: **624**
- Low: **3**

## Files materially reviewed in this batch

The failure-learning, preference, knowledge-boundary, dream, Evolver and checkpoint sections of `agentic/kai_config.py`.
