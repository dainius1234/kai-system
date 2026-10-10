# VALIDITY-BINDING POPULATION AUDIT — READ-ONLY EVIDENCE

Banked under **D364, evidence only**. No repair authority, no ontology
implementation authority, no HOUSE_H3 authority, no freeze authority.

| | |
|---|---|
| subject | `d8aac4d49e6ba997e3eb38062c0917186ee3f197` |
| subject tree | `3abc9e9d8ca11966a6f996d5f0af68072ee5b117` |
| candidate audited | `be37a0aa5d56255a151c31361d93e8b4be94ab912ec9441c8ac3535a84fbf133` |
| population | 272 documents |
| non-abstention VALIDITY | **56** |

## Reproduction provenance

The measurement is history-sensitive and **the active repository cannot
reproduce it**. The date-drift results need full history:

| | history source used | active repository |
|---|---|---|
| shallow | `false` | `true` |
| ancestry at subject | 986 | 280 |
| oldest reachable | 2025-06-18 | 2026-08-05 |
| origin | `https://github.com/dainius1234/kai-system` | same |

A shallow source does not fail — it returns the graft boundary as a
plausible date (`2026-08-05` for `TECH_WATCH.md`, true answer
`2026-07-24`). `validity_binding_audit.py` therefore **aborts** on a
shallow source rather than measuring. Verified: the abort fires.

```sh
git clone https://github.com/dainius1234/kai-system full   # NOT --depth
git -C full worktree add ../subject d8aac4d49e6b   # or a fresh clone at the subject
python3 kai-pm/validity_binding_audit.py \
    --subject-repo <subject checkout> --history-repo full \
    --subject d8aac4d49e6ba997e3eb38062c0917186ee3f197 \
    --package kai-pm/house_in_order_h2_v11 --out audit.json
```

## Result

| emitted state | N | BINDING PROVEN | SELF_CLAIM_ONLY | REGION/CITATION | AMBIGUOUS | FALSE_POSITIVE |
|---|---|---|---|---|---|---|
| `RUN_ARTEFACT` | 0 | 0 | 0 | 0 | 0 | 0 |
| `EXACT_SNAPSHOT` | 26 | 6 | 0 | 13 | 0 | 7 |
| `TIME_BOUND` | 23 | 0 | 14 | 9 | 0 | 0 |
| `CURRENT_TREE` | 7 | 0 | 0 | 0 | 0 | 7 |
| **TOTAL** | **56** | **6** | **14** | **22** | **0** | **14** |

**Whole-file binding proven: 6 of 56 (11%).** 14 are adjudicated false
positives; the other 36 are not proven at the scope H2 claims. 12 of the 14
false positives lie **outside** the partially-revealed 24-row holdout.

## The two independent checks

Neither asks the pattern that produced a witness whether the witness is real.

### 1. Does a commit-shaped witness resolve as a commit?

19 of 26 do. **Seven do not:**

| witness | what it actually is | document |
|---|---|---|
| `ed25519` | an algorithm name | `docs/next_level_roadmap.md` |
| `1700000000` | a unix timestamp | `kai-pm/CODE_AUDIT_BATCH_HMAC_ROTATION_DRILL.md` |
| `31570714150` | a workflow run id | `kai-pm/EMBEDDING_BACKEND_STATE.md` |
| `ed25519` | an algorithm name | `kai-pm/MAKEFILE_TARGETS.md` |
| `31894868473` | a workflow run id | `kai-pm/ORION_FIELD_NOTES.md` |
| `31605138566` | a workflow run id | `kai-pm/RUNTIME_TOPOLOGY_CENSUS.md` |
| `b5e68a3` | a `sha256:` DIGEST fragment | `kai-pm/VERIFICATION_ARCHITECTURE_REVIEW.md` |

`SHA = \b[0-9a-f]{7,40}\b` is not a commit-shaped test.

> **Caveat carried in the data.** "Does not resolve" is not proof a token
> is not a commit elsewhere. It is proof the classifier never checked.

### 2. Does a self-claimed date survive the history?

`last` was already in the Pass A row and was never consulted.
**9 of 23 `TIME_BOUND` documents changed AFTER the date they claim:**

| drift | claims | git last | document |
|---|---|---|---|
| **+94d** | 2026-04-21 | 2026-07-24 | `kai-pm/TECH_WATCH.md` |
| **+92d** | 2026-03-02 | 2026-06-02 | `docs/agentic_patterns_spec.md` |
| **+90d** | 2026-03-04 | 2026-06-02 | `docs/unfair_advantages.md` |
| **+32d** | 2026-07-21 | 2026-08-22 | `docs/PROJECT_BACKLOG.md` |
| **+30d** | 2026-07-24 | 2026-08-23 | `kai-pm/DECISIONS.md` |
| **+3d** | 2026-06-18 | 2026-06-21 | `kai-pm/SHOPPING_LIST_PLAN.md` |
| **+3d** | 2026-08-01 | 2026-08-04 | `kai-pm/STATUS.md` |
| **+3d** | 2026-07-21 | 2026-07-24 | `kai-pm/STUBS_AND_PLACEHOLDERS.md` |
| **+1d** | 2026-08-07 | 2026-08-08 | `kai-pm/SERVICE_IDENTITY_MEASUREMENT.md` |

## The D340 date repair is semantically withdrawn as proof of correct VALIDITY

`docs/agentic_patterns_spec.md` is the document `RUN.md` advertises as the
repaired D340 false positive. It was moved `CURRENT_TREE` → `TIME_BOUND`
and proved with a fail-old/pass-new fixture.

The fixture proved **abbreviated date recognition was repaired**.
It did **not** prove **the date binds the document**.

The document claims `2 Mar 2026`; history shows it last changed
`2026-06-02` — 92 days later. One syntactic defect repair followed by an
unsupported semantic promotion. The green fixture certified the exchange.

## Per-row audit

| path | verdict | witness | line | binding | region/cite | self-claim | class |
|---|---|---|---|---|---|---|---|
| `data/teammates/scout.md` | CURRENT_TREE | `currently` | 10/25 | NO | NO | NO | FALSE_POSITIVE |
| `docs/ara_review_status.md` | CURRENT_TREE | `currently` | 17/34 | NO | NO | NO | FALSE_POSITIVE |
| `docs/hmac_rotation_runbook.md` | CURRENT_TREE | `current phase` | 51/64 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md` | CURRENT_TREE | `currently` | 12/539 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/MAKEFILE_AUDIT.md` | CURRENT_TREE | `currently` | 18/411 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/MEMORY_GRAPH_DESIGN.md` | CURRENT_TREE | `currently` | 96/205 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/PLAYBOOKS/post_merge_checklist.md` | CURRENT_TREE | `current focus` | 6/13 | NO | NO | NO | FALSE_POSITIVE |
| `SESSION_BACKLOG.md` | EXACT_SNAPSHOT | `39c677c` | 81/284 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `docs/next_level_roadmap.md` | EXACT_SNAPSHOT | `ed25519` | 14/67 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/ASSURANCE_COUNTERPART_RESEARCH_2026-08-23.md` | EXACT_SNAPSHOT | `b14e6f9ce7879c` | 9/236 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_HMAC_ROTATION_DRILL.md` | EXACT_SNAPSHOT | `1700000000` | 17/231 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/CODE_AUDIT_CONTINUATION_LOG.md` | EXACT_SNAPSHOT | `3112c21f8258d5` | 17/410 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/CODE_AUDIT_FINAL_REPORT.md` | EXACT_SNAPSHOT | `2d830f25d569ba` | 4/1537 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `kai-pm/CODE_AUDIT_MASTER.md` | EXACT_SNAPSHOT | `2d830f25d569ba` | 6/237 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `kai-pm/CODE_AUDIT_PLANNING_PACKAGE_QA.md` | EXACT_SNAPSHOT | `2d830f25d569ba` | 20/123 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `kai-pm/DEEPSEEK_BRIEF_2026-08-07_EMBEDDINGS.md` | EXACT_SNAPSHOT | `e0e9849` | 66/120 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/EMBEDDING_BACKEND_STATE.md` | EXACT_SNAPSHOT | `31570714150` | 14/312 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/MAKEFILE_TARGETS.md` | EXACT_SNAPSHOT | `ed25519` | 87/213 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/NEXT_STINT_PLAN.md` | EXACT_SNAPSHOT | `097c91d` | 4/432 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/ORION_FIELD_NOTES.md` | EXACT_SNAPSHOT | `31894868473` | 18/482 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/PHASE1_READINESS.md` | EXACT_SNAPSHOT | `0e5d659` | 22/419 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/POSTMORTEM_2026-08-06_RUNNER_STARVATION.md` | EXACT_SNAPSHOT | `cddc01c` | 34/213 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/REALITY_CHECK_2026-05-10.md` | EXACT_SNAPSHOT | `97a3a61` | 23/38 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/REALITY_CHECK_2026-06-18.md` | EXACT_SNAPSHOT | `fa18739` | 59/74 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/RUNTIME_TOPOLOGY_CENSUS.md` | EXACT_SNAPSHOT | `31605138566` | 13/273 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/SEQUENCE.md` | EXACT_SNAPSHOT | `97a3a61` | 30/33 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/SERVICE_IDENTITY_STATE.md` | EXACT_SNAPSHOT | `773d21d` | 3/229 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `kai-pm/UH0_EVIDENCE_MANIFEST.md` | EXACT_SNAPSHOT | `7adab8d291011f` | 4/333 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `kai-pm/UH_PROGRESS_TRACKER.md` | EXACT_SNAPSHOT | `7adab8d` | 18/376 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/VERIFICATION_ARCHITECTURE_REVIEW.md` | EXACT_SNAPSHOT | `b5e68a3` | 66/246 | NO | NO | NO | FALSE_POSITIVE |
| `kai-pm/W1_DASHBOARD_REMEDIATION_PLAN.md` | EXACT_SNAPSHOT | `7adab8d` | 13/323 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/WAYPOINTS.md` | EXACT_SNAPSHOT | `bc70c7d` | 43/191 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md` | EXACT_SNAPSHOT | `9d15bcd` | 2/47 | YES | NO | NO | WHOLE_FILE_BINDING_PROVEN |
| `docs/PROJECT_BACKLOG.md` | TIME_BOUND | `2026-07-21` | 15/1434 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `docs/agentic_patterns_spec.md` | TIME_BOUND | `2 Mar 2026` | 3/304 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `docs/unfair_advantages.md` | TIME_BOUND | `4 March 2026` | 7/681 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_CLIPBOARD_SERVICE_EXTENSION.md` | TIME_BOUND | `27 July 2026` | 5/121 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_COGNITIVE_STATE_STUBS.md` | TIME_BOUND | `27 July 2026` | 5/301 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_FILES_SERVICE.md` | TIME_BOUND | `26 July 2026` | 5/128 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_GPU_FOUNDATION_STUBS.md` | TIME_BOUND | `27 July 2026` | 5/207 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_GPU_UTILITIES.md` | TIME_BOUND | `27 July 2026` | 5/205 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_BATCH_SCREEN_CAPTURE.md` | TIME_BOUND | `27 July 2026` | 5/142 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/CODE_AUDIT_REGISTER.md` | TIME_BOUND | `26 July 2026` | 6/434 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/DECISIONS.md` | TIME_BOUND | `2026-07-24` | 5/28167 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md` | TIME_BOUND | `27 July 2026` | 4/1190 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/NAVIGATION.md` | TIME_BOUND | `2026-08-01` | 6/66 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/PHASE_0_5_BACKLOG.md` | TIME_BOUND | `2026-07-21` | 6/110 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/RISKS.md` | TIME_BOUND | `2026-07-21` | 3/12 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/SERVICE_IDENTITY_MEASUREMENT.md` | TIME_BOUND | `2026-08-07` | 3/476 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/SERVICE_IDENTITY_TRUST_BOUNDARIES.md` | TIME_BOUND | `2026-08-08` | 3/139 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/SESSION_BOOTSTRAP.md` | TIME_BOUND | `25 July 2026` | 15/210 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/SHOPPING_LIST_PLAN.md` | TIME_BOUND | `2026-06-18` | 5/298 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/STATUS.md` | TIME_BOUND | `2026-08-01` | 3/71 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/STUBS_AND_PLACEHOLDERS.md` | TIME_BOUND | `2026-07-21` | 3/160 | NO | NO | YES | SELF_CLAIM_ONLY |
| `kai-pm/TECH_WATCH.md` | TIME_BOUND | `2026-04-21` | 9/44 | NO | YES | NO | REGION_OR_CITATION_ONLY |
| `kai-pm/UH2_INTAKE_REDESIGN.md` | TIME_BOUND | `2026-08-07` | 3/728 | NO | YES | NO | REGION_OR_CITATION_ONLY |

Full witness, context, position and per-row reasoning:
`kai-pm/validity_binding_audit_d364.json`.

## What this audit does NOT establish

* It does not establish that the 36 unproven-but-not-false verdicts are
  **wrong** — only that they are not proven at whole-file scope.
* It does not audit `FUNCTION`, `AUTHORITY`, `GENERATION` or `SCOPE`.
  Those remain **UNMEASURED** for this defect class.
* The adjudication classes are an **author nomination**. Every row ships
  its witness and context so an independent reviewer can overturn it.
