# KAI KINGSMAN — Engineering Architecture Drawing Set v0.4

> **STATUS: DETERMINISTIC ENGINEERING DRAWING SET FOR `KINGSMAN_EXISTING_KAI_MASTER_ARCHITECTURE_PLAN_V0_4.md` — NOT FINAL CANON, NOT IMPLEMENTATION AUTHORITY.**
>
> **Architecture subject creation commit:** `391f0b48eb3166888f0941cc61adb642e9197bde`.
>
> These diagrams are engineering claims, not presentation art. Every box is marked by semantic status and every target responsibility must map to current Kai or be explicitly identified as a genuinely missing joint.

## Legend

- `[C]` CURRENT — present in repository/current architecture; not automatically runtime-live.
- `[X]` TRANSITIONAL — existing migration/shim or current component being reworked.
- `[T]` TARGET RESPONSIBILITY — intended final responsibility, not necessarily a new service.
- `[N]` NEW JOINT — genuinely missing/under-specified mechanism attached to existing architecture.
- `[U]` UNKNOWN — exact current ownership/currentness requires E0 qualification.
- solid arrow = intended/current principal path.
- dotted arrow = migration/shadow/fallback/compatibility path.
- `PROPOSAL` = non-authoritative cognitive output.
- `AUTHORITY` = policy/approval/autonomy/capability path.
- `EXECUTION` = side effect.
- `VERIFY` = independent outcome observation.
- `HEALTH` = health/degradation path.

---

# A. WHOLE KAI — ROOTED ORGANISM

```mermaid
flowchart TB
    ROOT["PRIMARY MISSION / IDENTITY / LINEAGE\nDainius stewardship • continuity • growth • succession constraints"]
    ROOT --> KAI["KAI — THE ORGANISM\nnot a model, service, framework or machine"]

    KAI --> P["SENSES / PERCEPTION\n[C/X]"]
    KAI --> W["EVIDENCE + WORLD STATE\n[C/X/T]"]
    KAI --> M["MEMORY / RELATIONSHIP / CONTINUITY\n[C/U/T]"]
    KAI --> G["GOALS / WATCHES / TIME / ATTENTION\n[C fragments + N semantics]"]
    KAI --> C["COGNITION / UNIFIED HUNTER\n[C/X/T]"]
    KAI --> A["POLICY / AUTHORITY\n[C/X/T]"]
    KAI --> H["WORKFLOW / HANDS\n[C/X/T]"]
    KAI --> V["INDEPENDENT VERIFICATION\n[C/X/T]"]
    KAI --> R["HEALTH / DOCTOR / RECOVERY\n[C/X/T]"]
    KAI --> L["LEARNING / GROWTH / RELEASE\n[C/X/T]"]
    KAI --> S["CONTINUITY / LINEAGE / SUSTAINABILITY\n[C fragments + N]"]
    KAI --> O["MISSION CONTROL / OPERATOR GOVERNANCE\n[C Dashboard → T Mission Control]"]

    MODELS["Kimi • DeepSeek • GLM • Dolphin • future models"] -. "replaceable cognitive organs" .-> C
    HW["Strix Halo / future hardware"] -. "replaceable body generation" .-> KAI
```

**Architectural law:** Kai survives replacement of models/frameworks/hardware because identity and authority are not located in those components.

---

# B. CURRENT TRANSITIONAL RUNTIME — LEGACY + BUILT UH MIGRATION

```mermaid
flowchart LR
  subgraph LEGACY["CURRENT DEFAULT / LEGACY-LEANING PATH"]
    SENS["[C] sensors / watchers / readers"]
    POLL["[C] direct polling / Cortex / agentic reads"]
    MEM["[C] memu/current context"]
    AG["[C] agentic FSM / swarm / teammates"]
    TG["[C] Tool Gate / service auth"]
    ROUTES["[C] existing mutating service routes"]
    EXT["external/local targets"]
    SENS --> POLL --> MEM --> AG --> TG --> ROUTES --> EXT
  end

  subgraph UH["BUILT UH MIGRATION LAYER — NOT DEFAULT CUT OVER"]
    ADAPT["[X] sensor adapters"]
    ING["[C/X] PerceptionIngress"]
    J["[C/X] EventJournal"]
    WS["[C/X] scoped World State"]
    PW["[C/X] proposal-only workspace"]
    POL["[C/X] policy / approval / capability"]
    REG["[C/X] ActuatorRegistry\n34 identities / 8 tiers"]
    VER["[C/X] verifier / graded evidence"]
    AUTO["[C/X] scoped autonomy + LegacyTrustBridge"]
    ADAPT --> ING --> J --> WS --> PW --> POL --> REG --> VER
    VER --> AUTO
  end

  SENS --> ADAPT
  WS -. "KAI_CORTEX_SOURCE=world_state" .-> POLL
  AUTO -. "KAI_AUTONOMY_ENFORCE" .-> AG
  REG -. "migration handlers" .-> ROUTES
  ING -. "KAI_PERCEPTION_MODE=shadow|active" .-> WS
```

**Meaning:** the migration layer already exists. The work is to finish, harden and cut it over—not to build another architecture beside it.

---

# C. FINAL CONSEQUENCE PATH — MANUAL VS AUTONOMOUS AUTHORITY LANES

```mermaid
flowchart TB
    OBS["Qualified World State + Memory Context"] --> COG["Unified Hunter / cognition"]
    COG -->|PROPOSAL| PROP["ActionProposal"]
    PROP --> POLICY["Deterministic Policy / Constraints"]

    POLICY --> DEC{Initiation lane?}

    DEC -->|operator-approved| APPROVAL["Authenticated exact operator approval"]
    DEC -->|autonomous| AUTO["Valid scoped autonomy grant"]

    APPROVAL --> CAP["Exact one-use ActionCapability"]
    AUTO --> CAP

    CAP --> WF["Durable WorkflowRecord / outbox / fencing"]
    WF --> ACT["Actual actuator service / side-effecting hand"]
    ACT --> FH["[N/M17] validate exact capability\naudience + op + body digest + expiry + nonce"]
    FH --> CONSUME["atomic one-time consume"]
    CONSUME --> EFFECT["side effect"]
    EFFECT --> RECEIPT["ActuatorReceipt"]
    RECEIPT --> VERIFY["Independent target-specific verifier"]
    VERIFY --> OUT["VerifiedOutcome"]
    OUT --> LEARN["World update / learning / trust calibration"]

    ID["verified workload identity"] --> FH
    TARGET["[N/M21] target / egress constraints"] --> FH
```

**Key distinction:** autonomy decides whether Kai may initiate without fresh operator approval. It does not replace exact execution authority.

---

# D. CURRENT FINAL-HAND GAP — WHY D349 MATTERS

```mermaid
flowchart LR
    P["Policy/Approval"] --> CAP["ActionCapability"]
    CAP --> REG["[C] ActuatorRegistry\ncurrently consumes capability"]
    REG --> HANDLER["[C] mutating handler"]
    HANDLER -->|params + service auth/signature| ACT["[C] actual browser/backup/etc endpoint"]
    ACT --> EFFECT["side effect"]

    BYPASS["another service holding legacy membership/auth"] -. "possible direct call on weaker routes" .-> ACT

    NEED["[N/M17] capability-at-hand validation + atomic consume"] -. "must move/extend enforcement here" .-> ACT
    TEST["[N/M18] runtime negative bypass probe"] -. "must prove direct weaker path rejected" .-> ACT
```

Target closure condition:

`authenticated direct route` is insufficient. The weaker mutation path must be unusable at runtime.

---

# E. PERCEPTION → EVENT → EVIDENCE → WORLD STATE

```mermaid
flowchart LR
    SRC["[C] sensors / watchers / user / files / providers"]
    AD["[C/X] existing adapters"]
    PI["[C] PerceptionIngress\nvalidation • bounds • dedup • stale"]
    EJ["[C→T] EventJournal\ncurrent interface retained"]
    EV["[C/X/T] Evidence / provenance qualification"]
    CL["[C/X/T] structured Claim / conflict / freshness"]
    WS["[C/X/T] versioned WorldStateSnapshot"]
    PROJ["[X/M03] legacy consumer projections"]
    CORTEX["[C] Cortex / old consumers"]

    SRC --> AD --> PI --> EJ --> EV --> CL --> WS
    WS --> PROJ --> CORTEX

    OLD["[C] direct polling"] -. "shadow comparison only" .-> CORTEX
    OLD -. "retire after runtime proof" .-> WS
```

Fallback rule:

`world_state selected + canonical path unavailable` → `COLD_START/DEGRADED/UNKNOWN`, not silent steady-state polling.

---

# F. MEMORY / IDENTITY / RELATIONSHIP — CURRENT FAMILY AND TARGET OWNERSHIP

```mermaid
flowchart TB
    subgraph CURRENT["CURRENT MEMORY FAMILY"]
      MEMU["[C/U] memu-core"]
      GRAPH["[C] memu-graph / vector indexes"]
      LETTA["[C] Letta archival"]
      COMP["[C] memory-compressor"]
      OBS["[C] Obsidian / vault sync"]
      REL["[C] emotional / narrative / operator / relationship data"]
    end

    subgraph TARGET["TARGET RESPONSIBILITY CLASSES — PHYSICAL OWNER UNFROZEN UNTIL E0"]
      AUTHMEM["[T/U] authoritative memory records"]
      DERIVED["[T] rebuildable retrieval / graph projections"]
      ARCH["[T] archival"]
      MIRROR["[T] human-readable mirror"]
      MAINT["[T] compression / decay / maintenance"]
      IDMEM["[T] identity / relationship learned state"]
    end

    MEMU -. "E0 determines source role" .-> AUTHMEM
    GRAPH --> DERIVED
    LETTA --> ARCH
    OBS --> MIRROR
    COMP --> MAINT
    REL --> IDMEM

    AUTHMEM --> CONTEXT["Cognitive context"]
    IDMEM --> CONTEXT
    DERIVED --> CONTEXT
```

No new parallel memory service is assumed. Exact source ownership remains evidence-dependent.

---

# G. PROACTIVITY — EXISTING DETECTORS INTO ONE ATTENTION SEMANTIC LOOP

```mermaid
flowchart LR
    MON["[C] monitor-service"]
    CORTEX["[C] Cortex"]
    CAL["[C] calendar / scheduling"]
    SCREEN["[C] screen watcher"]
    ANOM["[C] anomaly / correlation"]
    OBS["[C] proactive observer"]
    RITUAL["[C] rituals / capability gaps"]
    SUP["[C] Supervisor nudges"]

    MON --> NORM["[N/M09] normalize observation/condition"]
    CORTEX --> NORM
    CAL --> NORM
    SCREEN --> NORM
    ANOM --> NORM
    OBS --> NORM
    RITUAL --> NORM
    SUP --> NORM

    NORM --> SEM["[N/T] Watch / Timer / Obligation / Commitment / Goal"]
    SEM --> ATTN["[N/T] AttentionCandidate"]
    ATTN --> DEC{decision}

    DEC --> IGN["ignore/store"]
    DEC --> WATCH["watch/prepare"]
    DEC --> NOTIFY["notify/propose"]
    DEC --> ACT["act only through existing authority lane"]

    OLD["[C] current proactive loops"] -. "shadow comparator" .-> DEC
```

**Physical home remains provisional** until E0; no separate proactivity service is assumed.

---

# H. COGNITION / UNIFIED HUNTER / MODEL RESOURCES

```mermaid
flowchart TB
    TASK["TaskFrame / current situation"] --> WS["Qualified World State"]
    TASK --> MEM["Memory/relationship context"]
    WS --> H["[C/X/T] Unified Hunter / Cognitive Workspace"]
    MEM --> H

    H --> FSM["[C] deterministic reasoning FSM"]
    H --> SOCR["[C] Socratic / hypothesis / causal / forecast"]
    H --> TEAM["[C] Scout / Sage / Doctor / Oracle / Advisor roles"]
    H --> ADV["[C] adversary / conviction / conflict resolver"]
    H --> FUT["[C/STUB] dialectical / analogical / concept blend / synthetic / transitive"]

    REG["[C→T] model registry\nadd digest/runtime/resource/qualification"] --> OLL["[C] Ollama / current model host"]
    OLL --> MODELS["replaceable local models"]
    MODELS --> H

    H -->|PROPOSAL ONLY| PROP["ActionProposal / answer / recommendation"]
```

No new runtime-manager service is required now. Stronger runtime management is introduced only when measured multi-runtime/resource needs justify it.

---

# I. IDENTITY / AUTHORITY SEMANTIC STACK

```mermaid
flowchart TB
    MEMBER["Membership\nshared token compatibility"]
    ID["Workload Identity\nEd25519 verified principal"]
    STATIC["Static role/operation scope"]
    POLICY["Policy / constraints"]
    APPROVE["Operator approval OR scoped autonomy delegation"]
    CAP["One-time exact ActionCapability"]
    HAND["Final-hand validation / atomic consume"]
    EFFECT["Side effect"]

    MEMBER --> ID --> STATIC --> POLICY --> APPROVE --> CAP --> HAND --> EFFECT

    NOTE1["membership alone cannot identify principal"] -.-> MEMBER
    NOTE2["identity alone is not action authority"] -.-> ID
    NOTE3["autonomy alone is not one-time capability"] -.-> APPROVE
```

This stack must remain explicit in code, tests, diagrams and Mission Control.

---

# J. DURABLE WORKFLOW AROUND EXISTING ACTUATOR REGISTRY

```mermaid
stateDiagram-v2
    [*] --> PROPOSED
    PROPOSED --> POLICY_BLOCKED: deny
    PROPOSED --> WAITING_APPROVAL: operator required
    PROPOSED --> APPROVED: valid autonomous delegation
    WAITING_APPROVAL --> APPROVED: exact approval
    APPROVED --> CAPABILITY_ISSUED
    CAPABILITY_ISSUED --> DISPATCH_PENDING
    DISPATCH_PENDING --> RUNNING: Postgres outbox / worker lease
    RUNNING --> SUCCEEDED_UNVERIFIED: receipt
    RUNNING --> FAILED: known failure
    RUNNING --> OUTCOME_UNKNOWN: timeout/partition
    OUTCOME_UNKNOWN --> RUNNING: reconcile says not executed + retry allowed
    OUTCOME_UNKNOWN --> SUCCEEDED_UNVERIFIED: external state proves execution
    SUCCEEDED_UNVERIFIED --> VERIFIED_SUCCESS: independent verification
    SUCCEEDED_UNVERIFIED --> VERIFIED_FAILURE: verification contradicts
    FAILED --> COMPENSATING: valid compensation exists
    COMPENSATING --> COMPENSATED
    VERIFIED_SUCCESS --> CLOSED
    VERIFIED_FAILURE --> CLOSED
    COMPENSATED --> CLOSED
    POLICY_BLOCKED --> CLOSED
```

First implementation candidate: PostgreSQL workflow/outbox/fencing around current ActuatorRegistry. No new workflow platform is assumed.

---

# K. HEALTH / DIAGNOSIS / CONTINGENCY / RECOVERY

```mermaid
flowchart LR
    HB["[C] heartbeat / metrics / sysmetrics / watchers"]
    TEL["[X/T] normalized telemetry"]
    GRAPH["[N/T] Component / Dependency / Authority Graph"]
    HD["[C/X] House Doctor\nstructured diagnosis"]
    DTEAM["[C] Doctor cognitive teammate"]
    CONT["[N/T] qualified contingency records"]
    POLICY["[C/X] policy / authority"]
    SUP["[C/X] Supervisor\nrecovery coordinator"]
    WF["durable workflow"]
    ACT["actual recovery actuator"]
    VER["independent verifier"]

    HB --> TEL --> GRAPH --> HD --> CONT --> POLICY --> SUP --> WF --> ACT --> VER
    GRAPH --> DTEAM
    DTEAM -->|PROPOSAL / explanation| POLICY
    VER --> GRAPH
```

No component diagnoses, approves, repairs and certifies itself end-to-end.

---

# L. CONTINUITY / LINEAGE / RESTORE

```mermaid
flowchart TB
    LIVE["Current Kai state"] --> BACK["[C] backup-service"]
    BACK --> MAN["[N/M13] Lineage Manifest"]

    MAN --> DB["authoritative store hashes / schema versions"]
    MAN --> REL["release / commit / contracts"]
    MAN --> EVT["event/world offsets"]
    MAN --> KEY["key references / custody state"]
    MAN --> MOD["model/runtime identity when material"]

    MAN --> OFF["[N/T] off-device / offline copy"]
    OFF --> REST["[N/T] isolated restore drill"]
    REST --> QUAL["lineage + authority + invariant qualification"]
    QUAL --> MIG["hardware/provider migration readiness"]
    QUAL --> MC["Mission Control continuity status"]
```

“Containers started” is not restore success. The intended lineage and authority state must be proven.

---

# M. FINANCIAL SUSTAINABILITY BOUNDARY

```mermaid
flowchart LR
    DATA["[C] financial awareness / market data / accounting inputs"]
    ANALYSE["[C/X] analysis / strategy / opportunity"]
    PROPOSE["proposal-only sustainability / finance plan"]
    POLICY["risk / policy / authority"]
    CAP["exact financial execution capability"]
    PAPER["[C] paper-trade / simulation"]
    REAL["[T future] real bounded execution"]
    RECON["independent reconciliation"]

    DATA --> ANALYSE --> PROPOSE --> POLICY --> CAP
    CAP --> PAPER --> RECON
    CAP -. "future separately authorised" .-> REAL --> RECON

    OPEX["Kai operating capital"] --> POLICY
    FAMILY["protected Dainius/family assets"] -. "separate trust domain; no survival presumption" .-> POLICY
```

No “survival agent” and no automatic access expansion.

---

# N. CANONICAL DEPLOYMENT PROFILE MODEL

```mermaid
flowchart TB
    REG["[N/T] ONE component registry\ncomponent • path • role • networks • state • auth • profile • evidence"]

    REG --> MIN["MINIMAL profile\ndaily-driver baseline"]
    REG --> FULL["FULL profile\nminimal + heavy/optional capability"]
    REG --> SOV["SOVEREIGN overlay\nhardening / secrets / observability / isolation"]
    REG --> HW["hardware/model optional profiles"]

    MIN --> DEPLOY["rendered/overlaid Compose deployment"]
    FULL --> DEPLOY
    SOV --> DEPLOY
    HW --> DEPLOY

    DEPLOY --> NET["agent / control / data / edge / egress / execution / observability / sensor networks"]
```

The implementation may use generated YAML or base+override files; one component truth is the invariant.

---

# O. CURRENT → TARGET MIGRATION CONTROL LOOP

```mermaid
flowchart LR
    CUR["CURRENT SUBJECT"] --> INT["recover intent"]
    INT --> MAP["map readers/writers/state/authority/deps"]
    MAP --> GAP["identify real gap"]
    GAP --> SHIM["compatibility shim / adapter"]
    SHIM --> SHADOW["shadow / dual-read / compare"]
    SHADOW --> TEST["positive + negative + boundary + fault tests"]
    TEST --> SOAK["runtime soak where required"]
    SOAK --> CUT["governed cutover"]
    CUT --> NEG["runtime prove weaker old path dead"]
    NEG --> RET["retire / rehome legacy"]
    RET --> DOC["update Mission Control / docs / evidence"]
    DOC --> BANK["bank exact subject / result"]
```

No migration is complete at “new path works”.

---

# P. E0→E11 PROFESSIONALISATION ROADMAP

```mermaid
flowchart LR
    E0["E0\ncurrent machine census"] --> E1["E1\ncanonical deployment profiles"]
    E1 --> E2["E2\ntruth-path / World State cutover"]
    E2 --> E3["E3\nidentity + authority + final hand"]
    E3 --> E4["E4\ndurable workflow"]
    E4 --> E5["E5\nWorld State / memory ownership"]
    E5 --> E6["E6\nproactivity / attention"]
    E6 --> E7["E7\ncognition / model resources"]
    E7 --> E8["E8\nDoctor / resilience / contingency"]
    E8 --> E9["E9\ncontinuity / lineage / restore"]
    E9 --> E10["E10\nsustainability / succession scaffolding"]
    E10 --> E11["E11\nMission Control / docs / S5 release"]
```

This is architecture dependency order only, not current programme execution authority.

---

# Q. MISSION CONTROL — OPERATOR VIEW

```mermaid
flowchart TB
    MACHINE["machine-derived current state / evidence subjects"]
    PROGRAM["programme authority / D-number state"]
    DESIGN["current master architecture / migration plan"]
    HEALTH["health / incidents / contingencies"]
    AUTH["approvals / autonomy / capabilities / bypass status"]
    LIFE["backup / restore / lineage / EOL / runway"]

    MACHINE --> MC["[C→T] DASHBOARD → MISSION CONTROL"]
    PROGRAM --> MC
    DESIGN --> MC
    HEALTH --> MC
    AUTH --> MC
    LIFE --> MC

    MC --> A["WHOLE KAI\ncurrent + target + maturity"]
    MC --> B["PROGRAMME / MIGRATION\ncurrent phase / next / blockers"]
    MC --> C["DECISIONS / ATTENTION\napprovals / watches / autonomy"]
    MC --> D["RESILIENCE / AUTHORITY\ndegraded / blast radius / bypass / recovery"]
    MC --> E["CONTINUITY\nbackup / lineage / EOL / runway / succession readiness"]
```

Mission Control is a derived governance view. It must never become the only source of programme or system truth.

---

# R. FAULT-CONTAINMENT MAP

```mermaid
flowchart TB
    FAIL{failure}
    FAIL --> SENSOR["sensor/provider"]
    FAIL --> MODEL["one model/specialist"]
    FAIL --> MEM["memory"]
    FAIL --> AUTH["authority"]
    FAIL --> VER["verifier"]
    FAIL --> DOC["House Doctor"]
    FAIL --> DB["PostgreSQL"]
    FAIL --> UI["Dashboard/Mission Control"]

    SENSOR --> S1["dependent claims UNKNOWN; unrelated Kai continues"]
    MODEL --> S2["missing viewpoint explicit; council degrades"]
    MEM --> S3["reduced-context mode; no invented continuity"]
    AUTH --> S4["consequential actions fail closed; cognition continues"]
    VER --> S5["result remains UNVERIFIED"]
    DOC --> S6["diagnosis degraded; health not inferred healthy"]
    DB --> S7["only explicitly designed degraded/read-only modes"]
    UI --> S8["core may continue; operator visibility/approval capability degraded"]
```

Every component must have an intended blast radius and a tested truthful degraded mode.

---

# S. ARCHITECTURAL STATUS / EVIDENCE RELATIONSHIP

```mermaid
flowchart LR
    CODE["repository presence"] --> PRESENT["PRESENT"]
    TEST["tests / calibration"] --> QUAL["QUALIFIED"]
    RUN["runtime observation"] --> LIVE["LIVE"]
    ENF["negative enforcement proof"] --> ENFORCED["ENFORCED"]
    EVID["current subject / provenance"] --> CLAIM["operator-visible claim"]

    PRESENT -. "does not imply" .-> LIVE
    LIVE -. "does not imply" .-> ENFORCED
    QUAL -. "does not imply deployment" .-> LIVE
    ENFORCED --> CLAIM
    EVID --> CLAIM
```

This diagram exists to prevent the recurring category error: code presence, test success, runtime use and enforcement are separate properties.

---

# T. FINAL TARGET — ONE ORGANIC FLOW

```mermaid
flowchart LR
    OBS["OBSERVE"] --> QUAL["QUALIFY / PROVENANCE"]
    QUAL --> WORLD["UPDATE WORLD STATE"]
    WORLD --> MEMORY["RETRIEVE MEMORY / RELATIONSHIP"]
    MEMORY --> GOALS["GOALS / WATCHES / ATTENTION"]
    GOALS --> COG["SPECIALIST INTERPRETATION / DELIBERATION"]
    COG --> PROP["PROPOSE"]
    PROP --> POLICY["POLICY / AUTHORITY"]
    POLICY --> HAND["EXACT CAPABILITY AT FINAL HAND"]
    HAND --> ACT["ACT"]
    ACT --> VERIFY["INDEPENDENTLY OBSERVE / VERIFY"]
    VERIFY --> LEARN["LEARN / DIAGNOSE / UPDATE TRUST"]
    LEARN --> WORLD

    HEALTH["health / structure / Doctor"] --> POLICY
    CONT["qualified contingency knowledge"] --> POLICY
    OP["Dainius / delegated authority"] --> POLICY
```

> **ONE CONNECTED ORGANISM — CLEAR RESPONSIBILITY — NO PARALLEL SOVEREIGNTY — BOUNDED FAILURE — VERIFIED OUTCOMES — CONTINUOUS GROWTH.**

---

# Drawing-set review rules

Before final freeze:

1. CURRENT boxes must be reconciled to E0 exact subjects/status.
2. TARGET boxes must map to a v0.4 requirement or accepted later delta.
3. NEW-JOINT boxes require explicit justification and target home.
4. no diagram may imply v0.4 is already implemented;
5. no green/completion state may survive withdrawal of supporting evidence;
6. physical process/service diagrams must be regenerated after E0/E1 rather than inferred from these logical views;
7. final drawings must be exact-subject-bound to `KINGSMAN_MASTER_CANON_v1` when frozen.
