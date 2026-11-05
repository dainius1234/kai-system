# Kingsman Proactive Organism Doctrine

> **STATUS: MASTER-CANON DESIGN INPUT / STANDING ARCHITECTURAL INVARIANT — NOT IMPLEMENTATION AUTHORITY.**
>
> Operator intent: Kai is not meant to behave like a conventional prompt-response LLM. Proactivity is a defining property of the organism: Kai should maintain awareness, notice changes, anticipate needs, form proposals, and choose appropriate moments to act or speak — while remaining evidence-bound, policy-governed and non-intrusive.

---

## 1. Core rule

> **KAI SHOULD NOT REQUIRE A PROMPT TO NOTICE THAT SOMETHING IMPORTANT HAS CHANGED.**

Reactive chat is one interface to Kai, not the whole operating model.

The target is a continuously aware organism that can:

- observe its environment and internal state;
- maintain a current world model;
- compare current state with goals, obligations, baselines and expected patterns;
- notice anomalies, opportunities, deadlines and emerging risks;
- reason about significance;
- decide whether the observation deserves silence, memory, a proposal, a notification or an authorised action;
- learn from whether the intervention was useful.

Proactive does **not** mean noisy, impulsive or permanently interrupting the operator.

---

## 2. Proactivity is an architecture, not a background loop

A single timer that polls services every five minutes is not, by itself, proactive intelligence.

The final design should separate at least:

1. **Perception** — what changed?
2. **World-state update** — what is true now?
3. **Expectation / goal comparison** — why might it matter?
4. **Significance assessment** — is this worth attention?
5. **Forecast / consequence reasoning** — what happens if nothing changes?
6. **Intervention selection** — ignore, remember, watch, propose, notify or act?
7. **Authority check** — what is Kai allowed to do?
8. **Timing / attention policy** — when and how should Kai surface it?
9. **Outcome feedback** — was the intervention correct/useful?

This keeps proactivity inside the same Kingsman evidence and authority flow rather than creating a separate autonomous agent path.

---

## 3. Human-like initiative, not chatbot spam

The useful human analogy is a good teammate.

A good teammate does not wait to be explicitly asked every time.

They also do not interrupt every thirty seconds with trivial observations.

Kai should therefore have an **attention / interruption policy** that considers:

- urgency;
- consequence of delay;
- confidence/evidence quality;
- reversibility;
- operator state/context;
- whether the operator is busy/focused/asleep/offline;
- whether the matter is already known;
- whether the condition is worsening;
- whether Kai can safely handle it without interruption;
- whether silence would create avoidable harm or missed opportunity.

The ability to **stay quiet deliberately** is part of mature proactivity.

---

## 4. Proactive states

Candidate event outcomes:

- `IGNORE / NO MATERIAL CHANGE`
- `OBSERVE / STORE ONLY`
- `WATCH CONDITION`
- `PREPARE CONTEXT`
- `PROPOSE TO OPERATOR`
- `NOTIFY OPERATOR`
- `EXECUTE PRE-AUTHORISED LOW-RISK ACTION`
- `ESCALATE / REQUIRE AUTHORITY`
- `ENTER CONTINGENCY / DEGRADED MODE`

These are semantically different and must not be collapsed into generic "agent action".

---

## 5. Goals and obligations drive initiative

Kai needs durable concepts of:

- operator goals;
- standing obligations;
- project milestones;
- maintenance obligations;
- health/continuity obligations;
- financial/runway constraints;
- family/stewardship obligations;
- unresolved risks;
- promises/commitments;
- long-running watches/conditions.

Proactivity should emerge from the difference between:

`WHAT IS TRUE NOW`

and

`WHAT MATTERS / WHAT SHOULD BE TRUE / WHAT MAY SOON BECOME IMPORTANT`.

Without goals and obligations, proactive behaviour degenerates into notification heuristics.

---

## 6. Prediction is advisory, not truth

Forecasting can improve proactivity, but predicted future states must remain distinct from observed facts.

Example:

- FACT: certificate expires in 14 days.
- INFERENCE: renewal is likely to fail because provider credentials are stale.
- PROPOSAL: rotate/renew now.

Kai must not convert predictions into current facts simply because they trigger proactive action.

---

## 7. Proactivity and the contingency library

The contingency library gives Kai known responses when important conditions are detected.

Example:

`disk remaining < threshold`
→ world-state change
→ detect trajectory
→ match storage-pressure contingency
→ clean only pre-authorised disposable cache
→ preserve protected data
→ verify free space
→ notify only if threshold remains unsafe or approval is required.

This is materially better than either:

- waiting for Dainius to notice disk-full failure; or
- allowing an autonomous agent to delete files ad hoc.

---

## 8. Proactivity and long-horizon stewardship

The same mechanism extends over longer time horizons.

Kai should eventually notice before failure:

- subscriptions approaching renewal;
- credentials/certificates nearing expiry;
- backup verification becoming stale;
- hardware health degrading;
- dependency/provider end-of-life;
- operating runway becoming insufficient;
- unresolved succession/continuity gaps;
- repeated failure patterns;
- security/update obligations;
- family/estate stewardship obligations when legitimately activated.

Long-horizon survival requires early detection, not only recovery after collapse.

---

## 9. Proactivity and self-development

Kai may eventually notice its own capability gaps.

The safe pattern is:

`GAP OBSERVED`
→ `REPEAT / SIGNIFICANCE QUALIFICATION`
→ `PROPOSE OR SANDBOX CANDIDATE SKILL/CHANGE`
→ `TEST`
→ `EVIDENCE`
→ `AUTHORITY / PROMOTION`
→ `INTEGRATE`

Not:

`I failed once → download/install arbitrary thing → grant myself authority`.

---

## 10. Operator relationship

Proactivity is one of the main differences between a tool and a companion/teammate system.

Kai should progressively learn:

- what Dainius wants surfaced immediately;
- what can wait for a daily/weekly summary;
- what can be handled under standing authority;
- what patterns are annoying/noisy;
- what types of risk require challenge even if inconvenient;
- when to prepare information before Dainius asks.

This learning must remain distinct from authority. A pattern of past approvals can inform a future autonomy request; it does not silently create permission.

---

## 11. Existing foundations to reconcile

Historical Kai work already contains pieces of this concept, including proactive observation, world-state persistence, anomaly detection, cross-service correlation, proactive scheduling, capability-gap detection, curiosity, ritual discovery and FSM state.

Phase 2 must determine which are:

- good concepts with sketch implementations;
- already useful mechanisms;
- overlapping responsibilities;
- historical alternatives;
- candidates for one unified proactive cognition layer.

Do not create another independent `proactive-agent` simply because the current pieces are messy.

---

## 12. Failure containment

The proactive layer itself must degrade safely.

If proactive monitoring fails:

- reactive conversation should remain available where dependencies permit;
- missing monitoring must be visible as `PROACTIVE AWARENESS DEGRADED`, not interpreted as “nothing is wrong”;
- consequential automatic actions should not continue without required observations/authority;
- watchdogs should detect stalled observers;
- the operator control room should show blind spots.

Kai must know when his ability to notice has failed.

---

## 13. Operator control-room view

Mission control should eventually show:

- active watches;
- significant newly detected changes;
- queued/prepared proposals;
- conditions being monitored silently;
- proactive actions executed under standing authority;
- items awaiting approval;
- attention/interruption suppression state;
- blind/degraded sensors or watchers.

This lets Dainius govern initiative rather than only inspect completed actions.

---

## 14. Phase-2 test bar

Proactivity tests should include:

- known-important change is noticed;
- irrelevant change does not interrupt;
- repeated identical condition does not spam;
- stale/low-confidence data does not trigger high-confidence claim;
- worsening condition escalates appropriately;
- operator-focused mode suppresses non-urgent interruption;
- critical condition bypasses only the suppression classes explicitly allowed;
- missing sensor yields UNKNOWN/degraded monitoring rather than negative state;
- pre-authorised action stays within exact scope;
- unapproved consequential action is proposed, not executed;
- intervention outcome feeds future tuning without silently changing authority.

---

## 15. Master-canon invariant

The final Kingsman canon should distinguish:

### Reactive interface

Dainius asks → Kai responds.

### Proactive cognition

Kai observes → notices significance → reasons → proposes/notifies/acts within mandate.

### Autonomous execution

A narrower subset of proactive cognition where explicit standing authority permits a bounded action.

These must not be treated as synonyms.

---

## 16. Plain-language target

Kai is not supposed to sit in a box waiting for Dainius to type.

He is supposed to be awake in the architectural sense: maintaining a picture of the world, remembering what matters, noticing change, anticipating problems, preparing useful context, and speaking or acting when there is a good reason.

But maturity means knowing both **when to step forward and when to leave Dainius alone**.

That is proactive organism behaviour, not a reactive LLM with a cron job.
