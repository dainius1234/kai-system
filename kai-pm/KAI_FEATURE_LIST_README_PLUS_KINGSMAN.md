# KAI — Feature list: README + Kingsman

**Orion | 2026-11-03 | directed by Dainius: "the rest is written in Kingsman tier plan"**

This is the build list:
- **Part 1:** the README's **104 features**, unchanged (machine-built on 2026-10-07 from README.md, re-run on 2026-10-11; handoff block KAI-FEATURE-LIST).
- **Part 2:** **20 additions** from the two Kingsman documents that state the mission and the final design:
  - `kai-pm/KINGSMAN_PRIMARY_MISSION_IDENTITY_AND_LINEAGE_DOCTRINE.md` (MD);
  - `kai-pm/KINGSMAN_FINAL_VISION_MASTER_CANON_PLAN.md` (KP).

  Each addition gives its source lines. "Extends #n" means it adds to a README feature rather than standing alone.

**What this list does not say:** whether any feature works today. That's the next step, the capability map: genuine, partial, simulated, name-only, absent, unknown.

## Part 1 — README features (104)

| Organ | Features |
|---|---|
| O1 Soul / Identity / Inner Life | 1 Ohana Core · 2 PUB/WORK personality modes · 3 Soul console view · 4 confession engine · 5 conscience & values · 6 emotional memory & predictive empathy · 7 epistemic humility · 8 imagination engine · 9 narrative identity · 10 self-reflection · 11 temporal self-model |
| O2 Memory / Relationship / Continuity | 12 Letta archival memory · 13 MARS memory decay / consolidation · 14 Obsidian Brain / vault-sync · 15 knowledge graph · 16 memory diary / mind-map views · 17 memory stores · 18 memu-core memory service · 19 per-teammate memory slices |
| O3 Senses / World Awareness | 20 anomaly detection · 21 cortex (perception spine / NPU) · 22 cross-sensor correlation · 23 multi-modal fusion · 24 sensors & perception services · 25 sensory pattern learning · 26 world state & provenance |
| O4 Intelligence / Reasoning | 27 14-way context gather · 28 agentic reasoning service · 29 conviction scoring · 30 local LLM serving · 31 memory-driven planner · 32 priority queue · 33 specialist router |
| O5 Cognitive Depth / Specialists | 34 transitive reasoning · 35 causal world model · 36 Global Workspace · 37 swarm assembly · 38 Socratic self-questioning · 39 hypothesis engine · 40 temporal projection · 41 dialectical synthesis · 42 analogical reasoning · 43 concept blending · 44 GPU-era foundations · 45 SAGE critique · 46 adversary engine · 47 cognitive reasoning FSM · 48 counterfactual rehearsal · 49 debate / tree search · 50 multi-model consensus · 51 persistent teammates & reputation · 52 verifier (SAGE fact-check) |
| O6 Proactivity / Goals / Attention | 53 Goals view · 54 Watch rule engine · 55 anti-annoyance · 56 emergent ritual discovery · 57 proactive observer / agent · 58 proactive scheduling · 59 resource-aware curiosity · 60 struggle detection · 61 wake word & intent |
| O7 Governed Hands / Capabilities | 62 Tool Gate + HMAC/Ed25519 auth · 63 Unified Hunter · 64 action audit ledger · 65 browser agent · 66 governed decision path · 67 sandboxed executor · 68 secret management & rotation · 69 trust negotiation / autonomy requests |
| O8 Immune System / Doctor / Resilience | 70 House Doctor / self-diagnosis · 71 checkpoint / time-travel debug · 72 heartbeat / pulse · 73 no single point of failure · 74 observability · 75 security self-hacking · 76 self-capability map · 77 structured errors / circuit breakers · 78 supervisor self-healing · 79 system FSM |
| O9 Growth / Dream / Evolution | 80 Agent-Evolver · 81 synthetic experience · 82 agentic cold path · 83 capability gap logging · 84 dream cycles · 85 skills hub / skill hunter |
| O10 Continuity / Stewardship / Sustainability | 86 backup service · 87 broker-bridge (read-only) · 88 financial awareness / UK self-employment advisor · 89 sovereign offline operation |
| O11 Operator Relationship / Mission Control | 90 cognitive fingerprint · 91 Telegram interface · 92 dashboard → Mission Control · 93 impact oracle · 94 notifications · 95 operator model · 96 presence: chat / voice / TTS / avatar · 97 shadow branches · 98 workspace manager |
| ENG Engineering base | 99 toolchain / CI gates · 100 feature flags · 101 honest status / test substrate · 102 private overlay network · 103 target hardware (Flow Z13 / Strix Halo) · 104 zero telemetry / local core |

## Part 2 — Kingsman additions (20)

### The mission: things the README does not mention at all

| # | Feature | Source |
|---|---|---|
| K1 | **Protect and care for Dainius** within earned authority: understand, assist, challenge, protect, care | MD L14 |
| K2 | **Stewardship for Dainius's daughter**, as protection and care without handing over unrestricted authority | MD L22, L114 |
| K3 | **Succession:** survive beyond Dainius's lifetime under a designed succession arrangement; a successor holds defined authority and is never treated as "Dainius under another name" | MD L21, L322–332 |
| K4 | **Keeps running when Dainius is unavailable:** stays recoverable and maintainable | MD L20 |
| K5 | **Economic self-sufficiency:** lawful, authorised, budget-bounded, auditable, tax-compliant, revocable income or resources, **kept separate from family assets**. Extends #88. | MD L18, L300–318 |
| K6 | **Long-horizon survival:** anticipate provider loss, hardware death, expiring credentials and running costs; migrate; keep funding and authority; verify continuity. Extends #86, #89. | MD L286–296 |

### Identity and continuity

| # | Feature | Source |
|---|---|---|
| K7 | **One KAI:** many models, senses and tools, but one governed decision path and no parallel authority. "KAI ≠ any LLM, framework, service or device." Extends #66. | KP L34–56; MD L32–70 |
| K8 | **Replaceable organs:** any model, sensor, memory backend or hardware can be swapped without losing KAI; "the model is an organ, not identity" | MD L186–209, L235–245 |
| K9 | **Lineage Manifest / Identity Ledger:** proves a restored or upgraded KAI is still the same KAI (mission version, memory lineage, authority lineage, migration record) | MD L249–268 |
| K10 | **Three kinds of state with their own change rules:** core invariants (change only by high authority), evolvable organs, learned state (evidence-aware) | MD L103–154 |
| K11 | **Growth without architectural amnesia:** KAI keeps developing; no "final version" | MD L158–182 |

### How KAI decides and checks itself

| # | Feature | Source |
|---|---|---|
| K12 | **Kingsman control loop:** perception → evidence → world state → specialists → deliberation → review → proposal → policy and values → Dainius's approval where required → exact capability → workflow → actuator → outcome check → verification → learning. Extends #66. | KP L58–78 |
| K13 | **Evidence Plane:** how a claim becomes evidence (identity, provenance, version binding, authority, uncertainty, evidence graph, attestations, audit). Extends #26. | KP L141–159 |
| K14 | **A4 structural self-diagnosis:** component map, runtime topology, drift detection, root cause, repair proposals, post-repair verification. Extends #70, #76. | KP L119–139 |
| K15 | **One recovery loop:** see → understand → diagnose → explain → propose → approve → heal → verify → learn, uniting House Doctor, Supervisor, FSM and anomaly detection. Extends #70, #78. | KP L206–212 |
| K16 | **Truthful degradation:** detect → contain → degrade honestly → recover → verify, in bounded failure domains. Extends #73, #77. | MD L276–284; KP L95–96 |
| K17 | **Values kept separate from facts:** operator-confirmed values; loyalty never overrides safety; assistant text never becomes operator truth. Extends #1, #5. | KP L194–204 |
| K18 | **Model runtime manager and hardware modes:** CPU/GPU/NPU roles, model residency profiles, battery, portable, docked and maintenance modes, future extra nodes. Extends #30, #103. | KP L189–190, L220–232 |
| K19 | **Earned autonomy:** scoped, revocable, expiring and evidence-bound; sandboxed skill generation; future agent swarms. Extends #69, #85. | KP L234–247 |
| K20 | **Proactive organism loop:** notice → remember → forecast → prepare → decide (ignore, watch, propose, notify, act) → learn whether it helped; initiative, never authority. Extends #57. | MD L213–231 |

**Total: 104 + 20 = 124 features** (12 of the 20 extend an existing README feature; 8 are new).
