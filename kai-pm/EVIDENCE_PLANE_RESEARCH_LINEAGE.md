# KAI Evidence Plane — Research Lineage and Continuity Note

Status: **PROJECT CONTINUITY RECORD — NOT A NEW ARCHITECTURE DECISION**  
Purpose: prevent loss of design lineage between threads/reviews.  
Applies to: Evidence Plane, self-diagnostics, experiment intelligence, assurance, recovery/repair proposal work.

## 1. Why this note exists

The KAI Evidence Plane and self-diagnostics architecture did **not** arise accidentally and must not be treated in future reviews as a newly rediscovered design.

The governing direction was deliberately shaped by prior research into high-assurance engineering and operations, including NASA-style fault management / health management / IV&V, high-reliability SRE practice, AI/system TEVV, mission-critical health modelling, and controlled deployment/rollback practice.

Dainius had to remind Kai of this lineage on 20 August 2026 after the same external research was refreshed and appeared to "match" the existing roadmap. The match is expected: the earlier research materially influenced the roadmap.

Future reviewers/agents must therefore treat this lineage as **continuity context**, not as evidence that a new architecture should be invented.

## 2. Core high-assurance patterns already incorporated

The Evidence Plane / self-diagnostics programme intentionally carries these recurring patterns:

1. **Observation is separate from diagnosis.**
   - Detect/observe first.
   - Preserve what actually happened before interpretation.

2. **Instrument failure is separate from subject failure.**
   - A broken observer does not prove the observed system is broken.

3. **Diagnosis is separate from authority.**
   - Intelligence may explain and propose.
   - Intelligence does not grant itself permission to act.

4. **The actor does not certify its own success.**
   - Consequential changes require independent verification / IV&V-style separation.

5. **Fault isolation requires discriminating evidence.**
   - "Cause X fits" is insufficient.
   - Ask what else must be true if X is the cause, then test those predictions.

6. **Unknown remains unknown.**
   - Missing, stale, conflicting or inaccessible evidence must not collapse into PASS/healthy/neutral.

7. **Exact provenance matters.**
   - Tree/image/config/run/environment identity belongs with evidence.
   - Applicability to later revisions is a separate qualification question.

8. **Health is a derived model, not a self-reported string.**
   - Service health/dependency state should be built from qualified observations rather than shallow `status: ok` claims.

9. **Recovery is controlled and bounded.**
   - Diagnose before mutating.
   - Use small blast radius, reversible/canary-style change where possible.
   - Verify postconditions independently.

10. **Learning follows verified outcomes.**
    - Proposed explanations, model reflections and actuator responses are not outcomes.

## 3. Organisational / discipline lineage

The design has been informed by the following classes of practice:

### NASA / aerospace high assurance
- Fault Detection, Isolation and Recovery / fault management.
- Integrated System Health Management style separation of detection, diagnosis, propagation/effects and recovery.
- Independent Verification & Validation: technical and managerial independence, nominal and off-nominal evidence, objective verification.
- Test pedigree / exact test article, configuration and environment identity.
- Consistency checking: if a diagnosis is correct, its predicted secondary observations should also be visible.

### Site Reliability Engineering / production operations
- Separate "what is broken?" from "why?".
- Combine black-box symptoms with white-box internal evidence.
- Read-only diagnosis before mutation.
- Hypotheses and verification steps remain distinguishable from facts.
- Incident learning comes from observed outcomes, not confident narrative.

### TEVV / AI assurance
- Testing, evaluation, verification and validation continues through the lifecycle, including after deployment.
- Claim strength is proportional to evidence and claim class; no universal confidence threshold.
- Generated reasoning is not automatically evidence.

### Mission-critical health modelling
- Explicit dependency-aware health state.
- Distinguish healthy / degraded / unhealthy / unknown.
- Fault injection / chaos-style qualification to prove detectors and recovery mechanisms can fail correctly.

### Controlled deployment / rollback practice
- Predefined failure conditions.
- Progressive exposure / bounded blast radius.
- Roll back to known-good state rather than pushing through uncertainty.
- Post-deployment verification is part of the change, not optional follow-up.

## 4. Mapping to the KAI Evidence Plane roadmap

This lineage is reflected directly in the governing sequence:

`RAW OBSERVATION`
→ `QUALIFICATION`
→ `CLAIM`
→ `DIAGNOSTIC REASONING`
→ `EXPERIMENT`
→ `AUTHORITY`
→ `ACTION`
→ `INDEPENDENT VERIFICATION`
→ `LEARNING`

And in the phased programme:

- **V0** formal specification / design replay.
- **V1** observations + qualification.
- **V2** deterministic Claim Engine + executable historical replay.
- **V3** Diagnostic Reasoner.
- **V4** Experiment Intelligence.
- **V5** Lesson institutionalisation.
- **V6** evidence-backed repair proposals.
- **V7** only narrowly scoped self-maintenance if separately earned and explicitly authorised.

The Evidence Plane roadmap is therefore the machine-usable formalisation of lessons already learned through the KAI-048 campaign plus prior high-assurance research.

## 5. Relationship to current KAI self-diagnostics work

The current `memu-core/introspect_app.py` / `/memory/diagnostics` surface is **not** the intended self-diagnostic brain. It is primarily a memory maintenance/introspection service.

The deployed Supervisor is also **not** the intended diagnostic authority. Its historical design mixes health observation and recovery in ways the audit identified as unsafe.

The intended future self-diagnostic chain is:

`observe system`
→ `qualify evidence`
→ `derive current claims`
→ `identify contradictions / unknowns`
→ `generate surviving hypotheses`
→ `predict what else should be observed if each hypothesis is true`
→ `reuse existing evidence first`
→ `propose smallest discriminating experiment`
→ `request authority where required`
→ `execute approved test/change`
→ `independent postcondition verification`
→ `verified lesson / regression gate / repair proposal`

This is the capability Dainius refers to when discussing KAI "self-diagnostics", "finding what is wrong", "giving options how to fix it", and later controlled repair proposals.

## 6. Continuity rule for future threads / agents

When resuming Evidence Plane or self-diagnostics work:

1. Do **not** treat NASA/SRE/IV&V similarity as a new discovery.
2. Read the governing Evidence Plane roadmap and this continuity note first.
3. Use external research to **validate, challenge or improve** the baseline, not to erase its lineage.
4. Change architecture only when source-bound evidence materially falsifies an assumption or exposes a stronger reusable design.
5. Preserve the constitutional boundaries:

> **EVIDENCE CAN INFORM AUTHORITY. EVIDENCE CAN NEVER CREATE AUTHORITY.**

> **INTELLIGENCE NEVER CREATES AUTHORITY.**

> **THE COMPONENT THAT ACTS MAY NOT BE THE SOLE COMPONENT THAT CERTIFIES SUCCESS.**

## 7. Current programme position

The self-diagnostic capability remains downstream of the currently governed prerequisite sequence:

`finish KAI-GATE-048 / Item 8`
→ `formal 048 closure`
→ `A-4 authoritative provenance repair`
→ `A-4 adversarial review`
→ `A-4 freeze + exact-byte hash`
→ `Assurance / Evidence Plane foundations`
→ `V0 → V1 → V2`
→ `Evidence Plane Core qualified`
→ `V3 diagnostics → V4 experiments → V5 lessons → V6 repair proposals`

This note does not authorise implementation or alter that order.

---

Recorded for continuity because this research lineage had to be manually recalled during a later thread. Future work should not depend on that reminder recurring.
