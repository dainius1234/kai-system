# Kingsman Organic Resilience & Evolvability Doctrine

> **STATUS: MASTER-CANON DESIGN INPUT / STANDING ARCHITECTURAL INVARIANT — NOT IMPLEMENTATION AUTHORITY.**
>
> Operator intent: Kai is to become an **organic, connected system**, not a collection of bolt-on features. Components should participate in one coherent flow while remaining sufficiently isolated that the failure, upgrade or replacement of one organ does not unnecessarily take down the whole organism.
>
> This doctrine must be considered during Kingsman master-canon reconciliation, House-in-Order Phase 2 professionalisation, future A4 self-diagnosis design, runtime topology review and README/architecture visualisation.

---

## 1. Core rule

> **ORGANIC INTEGRATION WITHOUT SHARED-FATE COUPLING.**

Kai should feel and behave like **one organism**, but it must not be engineered as one fragile blob.

Connected does **not** mean tightly coupled.

Integrated does **not** mean monolithic.

Modular does **not** mean disconnected.

The target is:

> **ONE KAI — COHERENT FLOW — CLEAR ORGANS — STABLE CONTRACTS — BOUNDED FAILURE DOMAINS — GRACEFUL DEGRADATION — INDEPENDENT UPGRADES — CONTINUOUS GROWTH.**

---

## 2. The organism model

The final architecture should be understandable in biological terms without using biology as an excuse for vague engineering.

Candidate conceptual mapping:

- **Perception / sensors** = senses
- **Evidence Plane / provenance** = trusted sensory/evidence circulation
- **World state** = current internal model of reality
- **Memory / identity** = long-term memory and continuity
- **Unified Hunter / cognition** = brain-level orchestration and reasoning
- **Specialists / swarms / models** = specialised cognitive regions
- **Policy / approval / Tool Gate** = executive control and inhibition
- **Capabilities / actuators** = hands and external action pathways
- **Supervisor / health / recovery** = autonomic resilience layer
- **House Doctor + future self-diagnosis** = diagnostic system
- **Dream / Evolver / learning** = consolidation and adaptation
- **Operator mission control** = human-visible control surface through which Dainius governs the organism

These are **complementary organs of one flow**, not independent mini-Kais.

---

## 3. Architectural flow

A healthy runtime should broadly follow:

`OBSERVE`
→ `QUALIFY / PROVENANCE`
→ `UPDATE WORLD STATE`
→ `RETRIEVE RELEVANT MEMORY`
→ `SPECIALIST INTERPRETATION`
→ `DELIBERATE / CHALLENGE`
→ `PROPOSE`
→ `POLICY / AUTHORITY`
→ `OPERATOR APPROVAL where required`
→ `EXECUTE THROUGH NARROW CAPABILITY`
→ `OBSERVE RESULT INDEPENDENTLY`
→ `VERIFY`
→ `LEARN / DIAGNOSE / UPDATE TRUST`

Every major component should contribute to this coherent flow or have a clearly justified supporting role.

Avoid the historical failure mode:

> new idea → new service → new API → new truth source → new authority path → another partially overlapping feature.

That is **bolt-on soup**.

The preferred pattern is:

> new idea → identify existing organ/layer → extend existing contract or introduce one justified new organ → preserve single truth/authority flow → verify integration and isolation.

---

## 4. Fault containment is mandatory

The organism must remain useful when non-essential organs fail.

For each component/subsystem the final architecture should explicitly define:

- failure domain;
- criticality;
- upstream/downstream dependencies;
- timeout/budget;
- retry policy where safe;
- circuit-breaker/degradation behaviour;
- fallback behaviour where truthful;
- what must fail closed;
- what may degrade gracefully;
- health signal;
- recovery/rollback path;
- operator-visible effect;
- whether the rest of Kai may continue.

A component crash must not automatically become a whole-system crash merely because everything shares a process, event loop, database connection, import graph or synchronous call chain.

### Standing principle

> **FAILURE SHOULD STOP AT THE NARROWEST SAFE BOUNDARY.**

Examples:

- weather unavailable → Kai can still converse and reason; weather-dependent claims become unavailable/UNKNOWN;
- one specialist model unavailable → council degrades; it does not fabricate that specialist's view;
- House Doctor unavailable → health monitoring may degrade, but diagnostic absence must be visible and must not be interpreted as “healthy”;
- optional skill crashes → quarantine/disable that skill rather than crashing cognition;
- memory subsystem degraded → explicitly reduce memory-dependent behaviour rather than silently inventing continuity;
- policy/authority layer unavailable → consequential actuation fails closed even if cognition remains available.

---

## 5. Degradation is a first-class operating mode

The final Kai should not have only two states: WORKING and DEAD.

Subsystems and the overall organism should support explicit states such as:

- `HEALTHY`
- `DEGRADED`
- `RECOVERING`
- `UNAVAILABLE`
- `QUARANTINED`
- `UNKNOWN / UNMEASURED`

The system must preserve the difference between:

- component unavailable;
- observer unavailable;
- degraded capability;
- negative subject result;
- unknown result.

This is existing House-in-Order doctrine applied to runtime resilience.

---

## 6. Stable contracts, replaceable organs

Kai is expected to evolve for years. Therefore important architectural seams should be defined by **stable contracts**, not by today's implementation details.

Where practical, each organ should expose:

- typed/validated input/output contracts;
- explicit versioning;
- capability identity;
- provenance/evidence expectations;
- error taxonomy;
- health/readiness semantics;
- authority requirements;
- observable metrics/events.

The implementation behind a contract should be replaceable without forcing a whole-system rewrite.

Examples of intended replaceability:

- swap one LLM/model without rewriting Tool Gate;
- replace one memory backend without changing every caller's semantics;
- improve House Doctor's diagnostic engine without giving it new authority;
- replace a sensor provider without changing world-state meaning;
- move a prototype from Python module to isolated service, or vice versa, without redefining the product concept.

---

## 7. Service boundaries must be earned

Microservices are not automatically professional architecture.

A separate service should be justified by one or more real boundaries such as:

- independent failure containment;
- independent scaling/resource requirements;
- security/trust boundary;
- hardware/runtime isolation;
- independent deployment lifecycle;
- long-running/durable execution;
- distinct ownership/contract.

If none apply, a module/library/process boundary may be cleaner and safer.

Phase 2 must review historical service proliferation and ask:

> **Does this need to be a service, or was a service simply the easiest way to sketch the idea at the time?**

Equally, do not collapse everything into one process merely for tidiness. The objective is **correct boundaries**, not fewer containers as a vanity metric.

---

## 8. Shared planes, not duplicated plumbing

Cross-cutting concerns should become common platform planes/primitives where justified rather than being reimplemented inside every feature.

Candidate shared planes include:

- evidence/provenance;
- identity/authentication;
- policy/authority;
- event/observation transport;
- capability registry;
- health/telemetry;
- configuration/feature flags;
- workflow durability;
- operator status/visibility.

A new feature should consume these shared primitives rather than inventing a parallel version.

But shared infrastructure must itself be resilient: a shared plane must not become an uncontrolled single point of catastrophic failure without failover/degradation rules.

---

## 9. Growth must be designed in

Current Kai is an early developmental stage, not the final ceiling.

The architecture should assume:

- more models;
- better models;
- new hardware;
- new sensors;
- new skills;
- new workflows;
- new reasoning mechanisms;
- new safety/assurance rules;
- new diagnostic knowledge;
- new interfaces;
- new operator requirements.

Therefore the system must support controlled extension through:

- registries/discovery rather than giant hard-coded switch statements where appropriate;
- versioned contracts;
- feature/capability flags;
- sandbox/probation for new skills;
- migration paths;
- compatibility checks;
- explicit deprecation/supersession;
- evidence-backed promotion through maturity levels;
- rollback.

### Standing rule

> **GROWTH SHOULD ADD CAPABILITY WITHOUT REQUIRING ARCHITECTURAL AMNESIA.**

New capability should plug into the existing organism's evidence, policy, authority, health, operator-visibility and verification flows.

---

## 10. No bolt-on authority

The most dangerous form of bolt-on soup is not duplicate code; it is duplicate **authority**.

No new component may quietly create its own route from observation to consequential action.

New capabilities must connect through the canonical authority chain.

A useful default rule:

`INTELLIGENCE MAY PROPOSE`

`POLICY MAY CONSTRAIN`

`OPERATOR / DELEGATED AUTHORITY MAY AUTHORISE`

`CAPABILITY MAY EXECUTE`

`INDEPENDENT OBSERVER MUST VERIFY`

No subsystem self-promotes because it has become sophisticated.

---

## 11. Self-diagnosis must understand the organism

Future self-diagnosis should reason over the architecture as a living dependency graph, for example:

`component`
→ `contract`
→ `runtime instance`
→ `dependencies`
→ `reader/writer relations`
→ `evidence sources`
→ `health state`
→ `recent changes`
→ `owner/authority`
→ `known failure patterns`

This is where House-in-Order/Census concepts become useful to runtime Kai.

A future diagnosis should be able to say:

> "Service X is unavailable. Its failure domain should affect capabilities A and B only. C and D are healthy. The degradation propagated into E because dependency Y is currently synchronous/shared. That propagation is outside the intended architecture. Proposed containment repair: Z."

That is materially better than merely restarting containers.

---

## 12. Independent update and maintenance

A professional Kai must support controlled maintenance without taking the whole organism offline wherever practical.

Phase 2 should investigate, per organ:

- independent build/test;
- contract tests;
- rolling/restart-safe update where relevant;
- schema migration;
- compatibility window;
- health/readiness gates;
- rollback;
- state preservation;
- dependency-order handling;
- post-update verification.

A change to one low-level capability should not require requalifying unrelated components unless a shared invariant actually changed.

Conversely, shared-plane changes **must** trigger all affected qualification because their blast radius is genuinely wider.

---

## 13. Resilience testing bar

Production-grade status requires more than happy-path unit tests.

For load-bearing components, Phase 2 should include appropriate tests such as:

- dependency unavailable;
- slow dependency / timeout;
- malformed response;
- stale evidence;
- duplicate/out-of-order event;
- partial network partition;
- service restart;
- state/backend unavailable;
- incompatible contract version;
- corrupted/quarantined skill;
- authority service unavailable;
- recovery attempt fails;
- recovery succeeds but independent verification fails.

Test the **blast radius**, not only whether the failing component reports an error.

Key question:

> **When organ X fails, what else stops working — and is that exactly what the architecture intended?**

---

## 14. Operator visibility during degradation

The operator control room must show degradation clearly.

Dainius should be able to see:

- what failed;
- what remains available;
- what is degraded;
- what Kai has automatically isolated;
- what is recovering;
- what requires approval;
- whether evidence/currentness is affected;
- what the expected blast radius is.

Do not reduce the entire system to a single red/green status light.

---

## 15. Phase-2 review question for every component

For each old/new script, service or subsystem, add these questions to the professionalisation review:

1. What organ/layer does this belong to?
2. What other organs does it depend on?
3. What depends on it?
4. Is the coupling intentional?
5. What happens if it crashes?
6. What is the expected blast radius?
7. Can Kai continue in a truthful degraded mode?
8. Can this component be updated/replaced independently?
9. Is its contract explicit enough to permit future growth?
10. Is it duplicating a shared plane or authority path?
11. Does the operator see its health/current status?
12. Does future self-diagnosis know enough to reason about it?

---

## 16. DeepSeek adversarial review additions

When the Kingsman architecture packet is reviewed externally, add these questions:

1. Where does this architecture still contain **shared-fate coupling** that could cascade a local failure into a global outage?
2. Which proposed shared planes are legitimate platform primitives, and which risk becoming dangerous single points of failure?
3. Which current microservices should probably become modules/libraries, and which current in-process components deserve isolation?
4. What contracts are essential now to keep future model/hardware/backend changes cheap?
5. What graceful-degradation states should be designed explicitly rather than emerging accidentally?
6. How should an AI system this modular preserve one coherent world state and one authority path?
7. What update/deployment pattern best supports a continuously growing personal AI without turning every upgrade into whole-system requalification?
8. What are the three most likely cascading-failure paths in the proposed final architecture?

External answers remain review input until repository assumptions are verified.

---

## 17. Master-canon acceptance requirement

The final Kingsman canon should not freeze until it can answer, for every major organ:

- responsibility;
- contract;
- authority;
- dependencies;
- failure boundary;
- degraded behaviour;
- recovery path;
- observability;
- update/replace strategy;
- evidence requirements;
- operator-visible status.

An architecture diagram showing components and arrows without these semantics is not enough.

---

## 18. Plain-language target

Kai should grow like an organism, not like a garage covered in extension leads.

Every new capability should know where it belongs, share the same evidence and authority bloodstream, and communicate through clear interfaces.

But if one organ gets sick, we isolate it. Kai should lose **that capability**, not collapse completely.

And when a better organ is developed later, we should be able to replace it without performing brain surgery on the whole system.

The system being built now is the beginning. The architecture must be strong enough not only for today's Kai, but for the much larger Kai it is intended to become.
