# KAI KINGSMAN — Current → Target Migration Matrix v0.4

> **STATUS: ARCHITECTURE MIGRATION CONTROL MATRIX — CANDIDATE / NOT IMPLEMENTATION AUTHORITY.**
>
> Companion to:
>
> - `KINGSMAN_EXISTING_KAI_MASTER_ARCHITECTURE_PLAN_V0_4.md`
> - `KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_4.md`
>
> **Important:** current status/ownership entries that are not already repo-proven are intentionally marked `E0-QUALIFY`. No row authorises deletion/merge/refactor.

## Status vocabulary

- `KEEP` — current responsibility/concept should survive substantially.
- `REWORK` — preserve intent/API where useful; improve implementation.
- `SPLIT` — current component mixes responsibilities that should separate logically/physically.
- `MERGE` — duplicate implementation responsibility may consolidate after E0.
- `REHOME` — capability remains but moves under another organ/module.
- `SUPERSEDE` — new implementation replaces old after verified cutover.
- `HISTORICAL` — keep lineage/evidence, not runtime authority.
- `E0-QUALIFY` — current reader/writer/state/failure/consumer map needed before disposition is final.

---

# 1. Core / truth / control

| Current | Current role | Candidate disposition | Shim / migration | Target responsibility | Cutover proof | Do not retire/change until |
|---|---|---|---|---|---|---|
| `common/perception_spine/*` | canonical observation ingress/shadow migration | KEEP + REWORK durability | E01/M02 | perception/event spine | shadow/active comparison + replay + consumer migration | E0 provider/consumer population known |
| file `EventJournal` | append/replay/digest | KEEP interface / REWORK backend if needed | M02 | durable event/outbox | digest/order/replay equivalence | measured durability/fan-out needs known |
| `common/world_state/*` | scoped immutable current state | KEEP + REWORK persistence | E02/M03 | qualified World State | consumer-by-consumer comparison + no silent fallback | E0 consumers known |
| Cortex direct polling | legacy/current context source | REWORK → transitional projection | E02/M03 | World State consumer | steady-state old poll path runtime-not-load-bearing | explicit COLD_START/DEGRADED semantics qualified |
| Tool Gate | current control point | KEEP + REWORK internals | M04/M19/M17/M21 | deterministic authority/control facade | restart/concurrency/final-hand/bypass tests | no parallel authority created |
| policy bridge | deterministic policy seed | KEEP/REHOME behind Tool Gate facade | M04/M19 | policy decision logic | exact decisions survive restart | current callers mapped |
| approval gate | exact operator approval concepts | KEEP + REWORK persistence/auth | M19 | authenticated exact approval | replay/restart/revocation tests | approval UI/identity requirements defined |
| capability bridge | one-time central capability | KEEP semantics + REWORK final-hand | M17/M19 | one-use execution authority | actual actuator consumes exact capability atomically | D349 blockers closed |
| autonomy authority | scoped delegation | KEEP + REWORK persistence/bootstrap | E03/M19/M20 | bounded autonomous initiation | durable grants + authenticated grant lifecycle | no self-grant / rollback weakening |
| LegacyTrustBridge | old TrustLevel migration | KEEP transitional | E03/M20 | legacy→scoped autonomy cutover | disagreement/soak + scalar authority unused | valid durable grants exist |

---

# 2. Identity / authentication

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| shared `KAI_SERVICE_TOKEN` | membership/auth compatibility | KEEP transitional / reduce authority | E05/M05 | simple membership only where appropriate | identity-sensitive routes reject shared-only caller | complete route matrix |
| `common/service_identity.py` | Ed25519 workload identity | KEEP + finish rollout | M05 | verified workload principal | route coverage, replay/rotation/revocation tests | do not reimplement existing nonce/timestamp/body/path logic |
| static service/operation grants | coarse route scope | KEEP | M05/M17 | identity + static scope below exact capability | grant enforcement + exact-capability tests | E0 identity-sensitive route census |

---

# 3. Actuation / workflow / external effects

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| ActuatorRegistry | 34 actuator catalogue/dispatch | KEEP | E04/M06/M17 | canonical actuator catalogue + dispatch | each migrated actuator final-hand gated | no duplicate actuator framework |
| migration driver | risk-tier migration | KEEP / generalize pattern | E04/M22 | evidence-bound cutover controller | active refused without evidence; retirement proof | flags remain selectors only |
| legacy source verifier | source-based closure | KEEP as partial evidence / strengthen | M18 | static + runtime closure | direct bypass mutation rejected | never equate auth with capability closure |
| executor | side-effect worker | REWORK around durable workflow | M06 | workflow worker/fenced dispatcher | crash/retry/idempotency tests | E0 exact executor responsibilities |
| browser-agent | privileged web action | KEEP separate candidate + harden | M17/M21 | browser actuator | exact capability + target policy + independent outcome | direct shared-token bypass denied |
| notify/TTS/avatar/Telegram | output channels | E0-QUALIFY; possibly shared orchestration/modules | M06/M17 | output actuator adapters | channel-specific capability/outcome tests | credential/failure boundaries mapped |
| file/vault/calendar mutation | local/external actions | KEEP handlers / final-hand harden | M06/M17 | narrow actuators | exact target/body capability consumed | current call routes known |
| backup/recovery actions | continuity mutation | KEEP separate high-risk path | M06/M17/M13 | continuity actuator | exact backup/restore capability + verification | restore semantics defined |
| broker/paper trade | financial action | KEEP capability / isolate risk | M14/M17 | finance actuator family | risk/capability/reconciliation | no protected-asset access expansion |

---

# 4. Memory / relationship / identity state

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| memu-core | hot memory/orchestration mix | E0-QUALIFY; likely KEEP/REWORK | M08 | authoritative memory responsibility candidate | writer/source map + projection drift tests | do not declare sole source before E0 |
| memu-core-introspect | maintenance/introspection | KEEP/REHOME | M08/M16 | memory maintenance/telemetry | consumer map | E0 |
| memu-graph | graph/retrieval | KEEP as projection candidate | M08 | derived knowledge graph | rebuild from source where intended | source semantics proven |
| pgvector/TurboVec | retrieval/index | KEEP projection candidate | M08 | derived vector index | drift/rebuild tests | source semantics proven |
| Letta | archival memory | KEEP/E0-QUALIFY | M08 | archival class | retrieval/restore semantics | actual readers/writers mapped |
| memory-compressor | maintenance | REHOME module candidate | M08 | compression/decay | no source loss; reversible evidence | authoritative state known |
| Obsidian/vault-sync | human-readable mirror/knowledge | KEEP / REHOME continuity module | M08 | operator-readable mirror | mirror divergence visible | not treated as hidden authority |
| emotional/narrative/operator/cognitive fingerprint | relationship/identity learned state | KEEP | M08/M13 | identity/relationship continuity | provenance/retention semantics | never promoted to security authority |

---

# 5. Proactivity / world awareness

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| agentic proactive observer | proactive decisions | KEEP/REWORK | M09 | attention decision logic candidate | old/new shadow usefulness | physical owner not frozen before E0 |
| monitor-service | rule/watch detection | KEEP detector / module candidate | M09 | Watch candidate producer | no lost detections | E0 service/failure need |
| calendar scheduling | time obligations | KEEP input/provider | M09 | Timer/Obligation source | missed/duplicate time tests | timezone/clock semantics defined |
| anomaly/correlation | change detection | KEEP detector | M09 | AttentionCandidate evidence | known-positive/negative calibration | source quality known |
| screen watcher | context detection | KEEP detector / boundary E0 | M09 | context/watch input | privacy/false-positive tests | device isolation need known |
| rituals/capability-gap | learned recurring needs | KEEP | M09/M15 | goals/growth candidates | usefulness/provenance | no direct authority |
| Supervisor nudges | mixed proactive behaviour | SPLIT logically | M10/M09 | user attention moves to proactivity; health stays Supervisor | no duplicate nudge loops | E0 call paths |

---

# 6. Cognition / models

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| agentic reasoning FSM | main cognitive orchestration | KEEP/REWORK | existing UH workspace | Unified Hunter/Cognitive Workspace | proposal-only authority tests | no second orchestrator |
| proposal workspace | proposal-only UH boundary | KEEP | none/contract evolution | Hunter proposal boundary | cannot issue capability/execute | retain invariant |
| Scout/Sage/Doctor/Oracle/Advisor | specialist roles | KEEP capability / likely modules | cognitive role adapter | specialist role library | role/maturity tests | E0 physical boundaries |
| swarm/conflict/reputation | multi-mind deliberation | KEEP/REWORK | workspace integration | specialist deliberation | disagreement preserved/budgeted | no sovereign agent authority |
| adversary/conviction | challenge/calibration | KEEP | role adapter | adversarial cognition | no authority leakage | evidence semantics retained |
| hypothesis/causal/temporal | deeper reasoning | KEEP | role adapter | specialist cognition | role qualification | data/model prerequisites |
| Global Workspace / higher-cognition stubs | future cognitive intent | KEEP dormant / E0-QUALIFY | later role adapters | future cognitive modules | prerequisites + qualification | do not delete due stub status |
| Ollama | model host | KEEP | M07 | current model runtime | model identity/health/metrics | no wrapper service without need |
| model registry/flags | model selection metadata | REWORK/enrich | M07 | exact model artifact/runtime/resource/role registry | digest/resource/qualification tests | multi-runtime manager deferred until earned |

---

# 7. Health / self-diagnosis / recovery

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| heartbeat/metrics/sysmetrics/watchers | health observations | KEEP + normalize | M16 | telemetry source family | missing observer visible | E0 source coverage |
| Prometheus/Grafana | observability | KEEP | M16/M12 | telemetry + Mission Control sources | currentness/identity labels | sovereign profile reconciliation |
| Supervisor | health/recovery + some nudges | KEEP/SPLIT logically | M10 | recovery coordinator | repairs route through authority/workflow | attention ownership migrated |
| House Doctor | rule diagnosis | KEEP/REWORK | M11 | structured diagnosis | evidence/differential/blast-radius outputs | no repair authority |
| Doctor teammate | cognitive diagnostic specialist | KEEP separate logical role | cognitive workspace | interactive reasoning | cannot certify repair | no third Doctor |
| common resilience | retry/breaker/healing primitives | KEEP | contingency mapping | low-level resilience primitives | fault injection | no private authority bypass |
| verifier | outcome truth | KEEP | verifier framework | target-specific verification | independence group tests | not merged with actuator authority |
| fusion-engine | evidence/verification support | E0-QUALIFY | verification adapter | verification/evidence module | consumer role proven | preserve independence semantics |
| Component/Dependency/Authority graph | not yet single machine graph | NEW JOINT | E0 component registry | structural self-understanding | generated from declarations/evidence | no hand-maintained second inventory |
| contingency library | scattered response knowledge | NEW JOINT | M11/M16 | qualified recovery knowledge | applicability/version/fault tests | knowledge ≠ authority |

---

# 8. Growth / release

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| skill-hunter | discovers skills | KEEP/REHOME growth module | M15 | candidate generation | provenance + sandbox entry | no self-install authority |
| Agent-Evolver | improvement candidate generation | KEEP/REHOME | M15 | candidate evolution | evidence/release gate | no self-promotion |
| Dream/introspection | consolidation/ideas | KEEP | M15 | candidate source | proposal-only | no runtime authority |
| skill probation/disable | lifecycle safety | KEEP | M15 | probation/rollback | failure disable works | release identity added |
| workspace-manager | skill/project workspace | KEEP/E0-QUALIFY | M15 | controlled build/sandbox workspace | isolation/cleanup tests | physical boundary E0 |
| release bundle/evidence | release qualification | KEEP/EVOLVE | M15/M13 | release/lineage attestation | exact subject bindings | Item8/A-4 lessons integrated only when authorised |

---

# 9. Continuity / finance / operator control

| Current | Current role | Candidate disposition | Shim | Target | Cutover proof | Condition |
|---|---|---|---|---|---|---|
| backup-service | real backup/restore | KEEP | M13 | continuity execution | isolated restore + manifest | do not replace storage first |
| sovereign security controls | hardening profile | KEEP/REHOME common overlay where appropriate | E1 | security profile | generated/overlay config parity | exact profile model |
| financial-awareness | read/analysis | KEEP | M14 | finance observation/analysis | typed boundary | no execution authority |
| broker/paper-trade | controlled financial execution seed | KEEP/isolate | M14/M17 | finance actuator | capability/reconciliation | future real-money authority separately governed |
| Dashboard | operator UI/status | KEEP/EVOLVE | M12 | Mission Control | same machine truth / no stale dual source | exact state schema |
| Grafana | observability UI | KEEP supporting view | M12 | Mission Control source/deep drill | telemetry currentness | not separate programme truth |
| PM/UH trackers/docs | governance/history | KEEP as governance evidence / derive volatile state | M12/doc sync | programme/architecture history | contradiction/drift checks | no live status solely manual |
| Lineage Manifest | absent as product-level proof | NEW JOINT | M13 | identity/restore continuity | restore qualification | exact invariant set open |
| provider/EOL registry | fragmented/absent | NEW JOINT | E9/E10 | long-horizon dependency watch | expiry/migration alerts | operator-visible |
| financial runway | partial awareness, no unified semantic state | NEW JOINT | M14 | sustainability planning | budget/runway calculations | proposal-only initially |
| succession state | doctrine/design only | NEW FUTURE JOINT | later | high-consequence stewardship transition | legal/human/crypto evidence | no auto-trigger from silence |

---

# 10. Cross-cutting new joints — implementation priority after programme authorisation

| ID | Joint | Attaches to | Why genuinely missing | Priority |
|---|---|---|---|---|
| M17 | Final-Hand Execution Capability | Tool Gate + ActuatorRegistry + actual actuators | current capability consumed centrally, not actual hand | BLOCKER |
| M18 | Runtime Legacy-Bypass Probe | migration verifier + live endpoints | source/auth proof does not prove weaker mutation path dead | BLOCKER |
| M19 | Durable Authority State | Tool Gate/policy/approval/capability/autonomy | current process-local authority state cannot survive/reconcile restart | MAJOR |
| M20 | Scoped-Autonomy Grant Bootstrap | autonomy + operator approval | current grant bootstrap not durable/authenticated enough | MAJOR |
| M21 | Egress/Target Constraints | policy/capability/browser/network hands | network reachability is not target authority | MAJOR |
| M22 | Evidence-Bound Migration Record | flags/preflight/CI | flags select paths but do not carry qualification proof | MAJOR |
| M13 | Lineage Manifest | existing backup/release/evidence | restore must prove intended Kai lineage | LONG-HORIZON P0/P1 |
| Component Graph | House/Census + deployment + telemetry | self-diagnosis/Mission Control | no one machine dependency/authority graph yet | P0 |
| Goal/Watch semantics | existing proactive detectors | durable proactivity | current signals lack one semantic ownership model | P0 |

---

# 11. Zero-loss review checklist

Before any service/module is merged, rehomed, superseded or deleted:

- [ ] original intent recovered;
- [ ] current implementation exact path known;
- [ ] callers/consumers known;
- [ ] state ownership known;
- [ ] authority/security role known;
- [ ] failure/isolation role known;
- [ ] product capability mapped to target;
- [ ] shim/cutover path defined;
- [ ] target replacement proven at least equivalent where capability retained;
- [ ] weaker legacy authority path proven dead if relevant;
- [ ] rollback safe;
- [ ] historical record preserved;
- [ ] Mission Control/docs updated;
- [ ] Dainius/Kai governed disposition recorded.

No bulk cleanup may bypass this matrix.
