# Kingsman Contingency & Fail-Safe Library — Design Obligation

> **STATUS: MASTER-CANON / PHASE-2 DESIGN INPUT — NO IMPLEMENTATION AUTHORISED BY THIS FILE.**
>
> Operator intent: Kai must carry a reusable, governed library of fail-safe responses, degraded operating modes, recovery playbooks and contingencies so individual components do not invent recovery behaviour independently during failures.
>
> This must extend and professionalise existing resilience machinery rather than create another bolt-on recovery authority.

## 1. Existing foundation

The repository already contains shared resilience primitives in `common/resilience.py`, including:

- retry/backoff;
- circuit breakers;
- deep dependency health checks;
- task watchdogs;
- a bio-inspired `HealingEngine` with containment, diagnosis, meta-cognitive escalation and knowledge recording;
- an in-memory per-service mapping of known error → known fix.

This is useful foundation/prototype logic, not yet a production-grade contingency system.

Phase 2 must first qualify the current implementation and determine what is retained, reworked, moved or superseded.

## 2. Core architectural rule

> **KNOWN FAILURE MODES SHOULD HAVE KNOWN, TESTED, GOVERNED RESPONSES.**

Do not rediscover the same recovery strategy every time a service fails.

Do not allow every service to invent its own retry/restart/fallback semantics.

Do not allow a sophisticated diagnostic component to self-authorise consequential repair.

The desired relationship is:

`FAILURE / ANOMALY`
→ `OBSERVATION`
→ `DIAGNOSIS`
→ `MATCH CONTINGENCY`
→ `CHECK APPLICABILITY + AUTHORITY`
→ `CONTAIN / DEGRADE / PROPOSE RECOVERY`
→ `APPROVE where required`
→ `EXECUTE NARROW RECOVERY`
→ `INDEPENDENT VERIFY`
→ `UPDATE LEARNED EVIDENCE`

## 3. What the library should contain

Each contingency should be a structured, versioned record rather than free prose.

Candidate fields:

- `contingency_id`
- `version`
- `failure_class`
- `symptoms / trigger conditions`
- `affected component / contract`
- `subject/applicability constraints`
- `criticality`
- `expected blast radius`
- `containment action`
- `degraded operating mode`
- `safe fallback`, if one exists
- `fail_closed_required`
- `retry policy`
- `timeout / circuit-breaker policy`
- `recovery options`
- `operator approval required`
- `required capability / actuator`
- `preconditions`
- `rollback plan`
- `verification test`
- `success criteria`
- `known contraindications`
- `evidence/source that earned the contingency`
- `confidence / qualification status`
- `last qualified subject/version`
- `expiry/review condition`
- `postmortem / learning references`

## 4. Example contingency classes

The final taxonomy should be evidence-derived, but expected families include:

### Dependency unavailable

Examples: memory backend, sensor, model server, database, external API.

Possible response pattern:

`isolate → mark dependent capability degraded → do not fabricate missing data → retry within budget → expose operator status → recover when dependency proves healthy`

### Slow / hung dependency

`timeout → circuit open → shed optional work → preserve core loop → diagnostic event`

### Specialist model unavailable

`remove specialist from current council → mark missing viewpoint → continue if quorum/decision rules permit → never synthesize the missing specialist's result`

### Memory unavailable

`disable memory-dependent assumptions → operate in explicit reduced-context mode → preserve new writes in safe spool where designed → notify operator if consequential`

### Policy / authority unavailable

`consequential actuation FAIL CLOSED → cognition may continue → operator sees control-plane outage`

### Skill failure

`quarantine skill → fallback to base capability if truthful → record failure → probation/requalification before re-enable`

### Evidence/provenance failure

`refuse promotion/use of unqualified evidence → keep UNKNOWN → do not convert missing proof into negative evidence`

### Data/schema incompatibility

`reject incompatible payload → preserve prior compatible state where safe → migration/rollback path → do not silently coerce semantics`

### Resource pressure

`reduce optional workloads/models → preserve control/evidence/policy paths → explicit resource-degraded profile`

### Repeated failed recovery

`stop automatic retries → escalate → request operator decision → avoid restart loops`

## 5. Three different response classes must stay distinct

The contingency system must not collapse these into one generic "self-heal":

### A. Automatic containment

Low-risk action necessary to stop propagation, e.g. open circuit, quarantine worker, stop retries.

### B. Automatic truthful degradation

Continue operating with reduced capability and explicit status.

### C. Recovery / repair

Action intended to change system state and restore capability. Depending on consequence/risk, this may require operator approval.

A system may be authorised to **contain** a failure without being authorised to **repair** it.

## 6. Authority model

The contingency library provides **knowledge**, not authority.

House Doctor / future self-diagnosis may identify and recommend a contingency.

Supervisor may execute pre-authorised narrow containment/recovery classes.

Policy/Tool Gate must enforce authority.

Dainius remains final authority for consequential or unearned recovery permissions.

A learned contingency cannot grant itself a higher authority level because it worked previously.

## 7. Evidence and qualification

A contingency is not trusted merely because it is documented or because it succeeded once.

Maturity states should include something equivalent to:

- `DRAFT`
- `OBSERVED`
- `TESTED`
- `QUALIFIED`
- `PRODUCTION_APPROVED`
- `RETIRED`

Promotion requires appropriate evidence.

For load-bearing contingencies, qualification should include:

- known-positive failure injection;
- known-negative/no-trigger case;
- boundary/contraindication case;
- blast-radius verification;
- rollback verification where relevant;
- independent confirmation that recovery restored the intended condition;
- confirmation that unrelated components did not regress.

## 8. Learning loop

Confirmed incidents should improve the library.

`INCIDENT`
→ `EVIDENCE`
→ `ROOT CAUSE`
→ `RESPONSE USED`
→ `OUTCOME`
→ `LESSON`
→ `CANDIDATE CONTINGENCY UPDATE`
→ `REVIEW / TEST`
→ `QUALIFY`

Do not automatically promote every improvised fix into doctrine.

A fix that worked once becomes **candidate knowledge**, not instant canon.

## 9. Relationship to existing components

### `common/resilience.py`

Existing shared primitive/foundation. Phase 2 decides whether it remains the library implementation, becomes a lower-level runtime library, or is split into clearer contracts.

### Supervisor

Consumes approved resilience policies and executes allowed containment/recovery actions. It should not invent diagnoses.

### House Doctor

Diagnoses and selects/suggests candidate contingencies. It should not become an unrestricted actuator.

### Future A4 self-diagnosis

Provides deeper structural context, dependency graph, recent-change correlation, applicability and likely root cause.

### Evidence Plane

Carries evidence proving why a contingency was selected, what subject it applies to, and whether recovery succeeded.

### Workflow engine

Runs durable multi-step recovery plans where a contingency requires pause/retry/checkpoint/rollback.

### Operator Control Room

Shows:

- active incident;
- containment applied;
- degraded capabilities;
- matched contingency;
- recovery state;
- approval needed;
- verification result.

## 10. No dangerous universal fallback

A generic fallback value such as `{}`, `[]`, `None`, `0`, cached-old-data or "success" must never be assumed safe across all callers.

Every fallback must have semantics defined by the consuming contract.

Standing rule:

> **A FALLBACK THAT HIDES FAILURE IS NOT RESILIENCE.**

If the consumer cannot safely distinguish fallback from real data, the correct behaviour is explicit degradation/refusal.

## 11. Avoid retry storms and healing loops

The library must define budgets for:

- retries;
- restart attempts;
- recovery attempts;
- cooldowns;
- circuit reopening;
- escalation.

Repeated failure should move toward containment/operator escalation, not infinite autonomous activity.

## 12. Contingency composition

Complex incidents may require more than one playbook.

Example:

`memory backend unavailable`
+ `disk pressure`
+ `operator offline`

The architecture should eventually support safe composition while detecting contradictory actions.

A central policy/recovery planner should resolve composition; individual services should not independently execute conflicting recovery actions.

## 13. Growth and portability

The library should be extensible as Kai grows.

New model providers, hardware devices, sensors and capabilities should register their known failure modes and recovery contracts without rewriting the core diagnostic engine.

Where practical, contingencies should reference **capabilities/contracts**, not hard-coded container names, so implementations can evolve.

## 14. Phase-2 review requirement

For every subsystem being professionalised, identify:

1. known failure modes;
2. expected blast radius;
3. containment strategy;
4. degraded mode;
5. fail-closed conditions;
6. recovery options;
7. approval requirements;
8. rollback;
9. independent verification;
10. whether an existing contingency applies or a new one must be qualified.

No S4/S5 production/Kingsman maturity without an appropriate failure/contingency story for material dependencies.

## 15. Operator-facing visual

The future mission-control page should include a **Resilience / Contingency** view showing at least:

- healthy/degraded/unavailable organs;
- current containment actions;
- contingency/playbook ID in force;
- automatic vs approval-required next action;
- recovery attempts/budget remaining;
- verification result;
- unresolved risk.

This allows Dainius to understand not merely that something is red, but what Kai is doing about it and why.

## 16. DeepSeek review questions

Add to the master architecture review packet:

1. Is a shared contingency library the correct abstraction, or should portions be policy, workflow and diagnostic knowledge separately?
2. Which actions are safe enough for automatic containment versus operator-approved recovery?
3. How should contingency applicability be represented so a playbook for one implementation/version cannot silently apply to another?
4. How should conflicting/composed contingencies be resolved?
5. What design prevents a contingency library from becoming a second orchestration authority?
6. What evidence is sufficient to promote a learned incident response to production-approved contingency?
7. What failure classes are missing from the proposed taxonomy?
8. How should the library integrate with workflow durability and rollback?

## 17. Plain-language target

Kai should carry something like a tested emergency handbook inside the machine.

When an organ fails, Kai should not panic, improvise blindly or restart everything.

He should know:

> **what probably failed → what must be isolated → what can continue safely → what fallback is genuinely safe → what repair is allowed → when Dainius must approve → how to verify the repair actually worked.**

As Kai grows, that handbook grows with him — but every new contingency must earn its place through evidence and testing.
