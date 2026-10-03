# DeepSeek Review Packet — Kai Kingsman Candidate Architecture v0.1

> **PURPOSE:** independent adversarial architecture/design review before Kai + Dainius reconcile and freeze a master canon.
>
> **DO NOT TREAT THIS PACKET AS IMPLEMENTATION AUTHORITY.** The architecture is a candidate. Existing frozen programme/experiment authority remains unchanged.

## Primary review subject

Read in full:

`kai-pm/KINGSMAN_CANDIDATE_ARCHITECTURE_V0_1_DEEPSEEK_REVIEW.md`

Candidate creation commit:

`905130f7210c203e3ce287ea896748d94ed5d571`

Root doctrine/context:

- `kai-pm/KINGSMAN_PRIMARY_MISSION_IDENTITY_AND_LINEAGE_DOCTRINE.md`
- `kai-pm/KINGSMAN_ROOT_ARCHITECTURE_AND_CANON_ALIGNMENT.md`
- `kai-pm/KINGSMAN_ORGANIC_RESILIENCE_ARCHITECTURE_DOCTRINE.md`
- `kai-pm/KINGSMAN_CONTINGENCY_AND_FAILSAFE_LIBRARY_DESIGN.md`
- `kai-pm/KINGSMAN_PROACTIVE_ORGANISM_DOCTRINE.md`
- `kai-pm/KINGSMAN_LONG_HORIZON_STEWARDSHIP_AND_SUCCESSION.md`

## Architectural intent you must preserve unless you make a strong case against it

Kai is the **whole persistent organism**, not any individual model/framework/service/device.

Kimi, DeepSeek, GLM, Dolphin and future models are replaceable cognitive organs/resources.

The system is intended to:

- be proactive rather than prompt-only;
- remain evidence-bound;
- preserve one governed authority path;
- contain failures locally;
- keep operating truthfully in degraded modes;
- grow through controlled replacement of organs;
- preserve identity/lineage across hardware/model generations;
- eventually become economically sustainable under bounded governance;
- survive beyond the original operator and support a future authorised successor/family stewardship purpose;
- remain operator-sovereign while the original operator is active.

This is a private personal system, not a commercial product.

## Non-negotiable programme facts

- The current programme is not authorised to implement this candidate merely because the candidate exists.
- Latest valid D-numbered programme authority controls execution.
- **ITEM 8 BEFORE A4** remains standing.
- `A-4 PROVENANCE` is a programme stage distinct from `FUTURE A4 SELF-DIAGNOSIS`.
- Do not recommend modifying frozen House/Census/048/Item8 experiments as an opportunistic architecture cleanup.

## Current implementation facts already verified by Kai

These are seeds to evaluate/reuse, not claims that the target architecture is already live:

1. `common/contracts/*` already contains typed Pydantic contracts for perception/world/action/assessment/autonomy.
2. `common/perception_spine/ingress.py` already enforces validation, payload bounds, stale marking, duplicate handling and append-to-journal semantics.
3. `common/perception_spine/journal.py` is a crash-conscious fsynced JSONL journal with replay/digest checks, but is not a production multi-writer event backbone.
4. `common/world_state/*` already has scoped immutable snapshots, reducer semantics, conflict preservation, replay and freshness, but persistent state is in-memory structures.
5. `common/proposal_workspace/*` exists as a proposal/Global-Workspace seed.
6. `common/policy_bridge/*` includes policy/approval/capability concepts.
7. `common/autonomy/authority.py` already implements scoped, expiring, revocable, evidence-earned autonomy grants, but authority state is in-memory.
8. `common/service_identity.py` implements per-service Ed25519 request signing where identity derives from the verifying key rather than caller assertion; real full service-image rollout/feasibility still requires qualification.
9. `common/actuator_registry/*` exists with handlers/mutating handlers/migration/verification concepts.
10. `common/resilience.py` has retries, breakers, health/watchdog and healing primitives.
11. `house-doctor/app.py` is currently a v0.1 rule/string diagnosis service with in-memory recent history; valuable idea, insufficient production diagnosis architecture.
12. `supervisor/app.py` currently mixes fleet health/recovery with proactive nudging; the candidate recommends splitting those concerns.
13. `backup-service/app.py` performs real component backups and PostgreSQL restore but does not yet provide long-horizon lineage/restore proof, off-device resilience or automated restore qualification.
14. `common/model_registry.py` is a hard-coded model/routing sketch and is not a real runtime/resource/qualification manager.
15. Current Compose already has network segmentation and many services; the architecture should reduce service-per-idea drift rather than create more containers by default.

If you believe any of these facts materially affects your recommendation, state which fact and what additional repo evidence you would need.

## External standards used only as reference models

The candidate deliberately avoids requiring third-party frameworks merely because they are popular.

Review the use of:

- CloudEvents — common event-envelope semantics
- W3C PROV — provenance vocabulary/model
- SPIFFE — workload-identity reference
- OpenTelemetry — traces/metrics/logs/context
- in-toto + SLSA — subject-bound attestations/build/source provenance
- OPA — optional policy-engine candidate
- transactional outbox — initial event/state reliability pattern
- Temporal — optional later durable-workflow backend

Challenge any adoption that adds more complexity than value.

## Your task

Act as a senior distributed-systems + security + AI-runtime architect and **try to break this design**.

Do not simply agree with it.

Evaluate:

1. logical architecture;
2. physical deployment boundaries;
3. state ownership;
4. evidence/provenance semantics;
5. world-state model;
6. memory boundary;
7. proactivity/attention design;
8. cognition/model-runtime split;
9. identity/policy/authority chain;
10. durable workflows/actuators/verifiers;
11. resilience/self-diagnosis;
12. long-horizon backup/lineage/succession;
13. financial sustainability boundary;
14. skill/evolution lifecycle;
15. operator mission control;
16. Strix Halo resource strategy;
17. migration from the existing repository;
18. over-engineering and simplification opportunities.

## The questions that matter most

The full candidate contains 45 review questions. Prioritise these first:

1. **What organ/component is still missing?**
2. **Where have we accidentally created a second authority/orchestrator?**
3. **Is `kai-core` too broad? What exactly should be in-process vs isolated?**
4. **Is PostgreSQL + transactional outbox the right first backbone for one-device Kai?**
5. **Is Goal/Obligation/Watch + Attention Engine the correct abstraction for real proactivity?**
6. **How should we implement workload identity on v1: current Ed25519, mTLS, SPIFFE-like, or SPIRE?**
7. **Should the policy backend remain custom, use OPA, or another approach?**
8. **Postgres workflow engine vs Temporal for v1?**
9. **How would you design Model Runtime Manager for Strix Halo shared memory, multi-model use and future replacement?**
10. **What should run on NPU, GPU and CPU today—not theoretically?**
11. **How should House Doctor / Supervisor / Future A4 / Contingency Library be separated cleanly?**
12. **How do we prove restored/migrated Kai is the intended lineage, not merely a bootable copy?**
13. **How do we reconcile privacy/erasure with append-only provenance/audit?**
14. **What are the five most dangerous cascading failures?**
15. **If you must simplify this architecture by 30%, what do you merge/remove and why?**

## Required classification

For every finding:

- `BLOCKER` — breaks mission, identity, evidence/truth, authority or failure-containment invariant.
- `MAJOR` — direction is viable but implementation is likely to be unsafe/incorrect/fragile.
- `MINOR` — worthwhile improvement, not required to approve direction.
- `QUESTION` — insufficient evidence; requires test or repo inspection.

Use:

```text
ID:
SEVERITY:
ARCHITECTURE SECTION:
CLAIM / PROBLEM:
WHY IT MATTERS:
PROPOSED CHANGE:
WHAT IT SIMPLIFIES / ADDS:
NEW RISKS CREATED:
REPO FACTS IT DEPENDS ON:
TEST / MEASUREMENT THAT WOULD DISCRIMINATE:
CONFIDENCE:
```

## Required final output

Finish your response with these exact sections:

1. `OVERALL VERDICT — APPROVE DIRECTION / APPROVE WITH CHANGES / REJECT DIRECTION`
2. `TOP 10 CHANGES BEFORE CANON FREEZE`
3. `MISSING ORGANS / MISSING SYSTEM PRIMITIVES`
4. `RECOMMENDED LOGICAL ARCHITECTURE`
5. `RECOMMENDED PHYSICAL DEPLOYMENT`
6. `RECOMMENDED DATA / EVENT / EVIDENCE ARCHITECTURE`
7. `RECOMMENDED STRIX HALO MODEL-RUNTIME ARCHITECTURE`
8. `RECOMMENDED RESILIENCE / SELF-DIAGNOSIS ARCHITECTURE`
9. `RECOMMENDED LONG-HORIZON / SUCCESSION PREPARATION`
10. `SIMPLIFY BY 30%`
11. `TOP 5 CASCADING FAILURE PATHS`
12. `EXPERIMENTS / SPIKES REQUIRED BEFORE FREEZE`
13. `TOP 3 STRONGEST PARTS`
14. `TOP 3 HIDDEN ASSUMPTIONS`

Do not redesign merely for novelty. If the current candidate is sound, say which boundaries should remain. If you recommend a framework/product, explain the operational cost and what existing Kai code it replaces rather than adding another layer beside it.
