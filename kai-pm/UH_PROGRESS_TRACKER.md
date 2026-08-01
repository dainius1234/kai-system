# Unified Hunter — Progress Tracker

> **This is the single source of truth for Unified Hunter work.**
> If this file and any other doc disagree, **this file wins** for UH status.
> Every UH change must update this file in the same commit.

**Last updated:** 2026-08-01
**Branch:** `claude/project-rework-plan-pgvp35`
**Verify everything:** `make test-uh` (one command, all suites)

---

## 1. Governing documents — read in this order

| Order | Document | What it governs |
|---|---|---|
| 1 | [`KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`](KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md) | **Authoritative.** Work packages UH-0…UH-8, invariants (§15), adversarial suite (§16), anti-patterns (§17) |
| 2 | [`UH0_EVIDENCE_MANIFEST.md`](UH0_EVIDENCE_MANIFEST.md) | Immutable pre-migration baseline at commit `7adab8d` |
| 3 | **This file** | Live status of every work package and exit gate |
| 4 | [`DECISIONS.md`](DECISIONS.md) | **Append-only.** Why each choice was made. Never edit an entry — correct with a new one |
| 5 | [`CODE_AUDIT_REMEDIATION_BACKLOG.md`](CODE_AUDIT_REMEDIATION_BACKLOG.md) | Findings from the wider code audit |

**Rule:** no UH code change lands without (a) a roadmap clause it satisfies, and (b) a row updated here.

---

## 2. Work package status

| Item | Status | Commit | Suite | Tests |
|---|---|---|---|---|
| UH-0 Evidence manifest | ✅ Done | — | — | — |
| UH-1 Freeze canonical contracts | ✅ Done | `4ac5187` | `make test-contracts` | 126 |
| UH-2 Perception spine (shadow) | ✅ Done | `2f53d16` | `make test-perception-spine` | 166 |
| UH-3 Scoped world state | ✅ Done | `ca136cd` | `make test-world-state` | 71 |
| UH-4 Proposal-only workspace | ✅ Done | `06c5414` | `make test-proposal-workspace` | 52 |
| UH-5 Policy / approval / capability | ✅ Done | `9904486` | `make test-policy-bridge` | 49 |
| UH-6 Paper-trade vertical slice | ✅ Done | `b752b1d` | `make test-vertical-slice` | 86 |
| UH-7 Actuator registry + migration | ⚠️ Machinery only | `78dbd60` | `make test-actuator-registry` | 75 |
| UH-8 Autonomy requalification | ✅ Done | `07d3614` | `make test-autonomy` | 174 |
| §16.4 Payload bounds | ✅ Done | this commit | `make test-payload-bounds` | 24 |
| §16.13 Ohana / assessments | ✅ Done | this commit | `make test-assessment` | 56 |
| §16.26 Rollback guards | ✅ Done | this commit | `make test-invariant-guards` | 17 |
| | | | **Total** | **896** |

**UH-7 is marked ⚠️ deliberately.** The registry, migration state machine and legacy-path gate are built and tested, but **every actuator is still at `LEGACY`** — nothing has been migrated against a live service. See §5.

---

## 3. Exit gate status

### UH-6 — vertical slice
| Gate | Status |
|---|---|
| No direct financial mutation path remains | ✅ in the new path; legacy paths still live (§5) |
| Correlation and stale-source tests fail safely | ✅ |
| One signal cannot close unrelated positions | ✅ |
| Partial/unknown outcomes reconcile safely | ✅ |
| Runs in shadow/test mode | ✅ |

### UH-7 — actuator migration
| Gate | Status |
|---|---|
| Old path disabled before new path verified | ✅ enforced by state machine — `advance_migration()` to VERIFIED raises while `legacy_path` is enabled |
| Migration proceeds in ascending risk order | ✅ `next_migration_candidates()` gates on lower tiers |
| Actuators actually migrated | ❌ **0 of 33** — see §5 |

### UH-8 — autonomy
| Gate | Status |
|---|---|
| Self-generated text or simulation cannot grant trust | ✅ structural — `EvidenceGrade.qualifies()` + provenance downgrade |
| High-consequence domains pass attack-chain tests | ✅ 13 chains in `test_autonomy.py` |
| Autonomy bounded, expiring, revocable | ✅ |

---

## 4. Roadmap §16 adversarial suite — 30 required

| # | Scenario | Covered by |
|---|---|---|
| 1 | Anonymous provider event injection | `test_policy_bridge` |
| 2 | Compromised provider impersonates another source | `test_perception_spine` |
| 3 | NaN/infinity/negative/out-of-range payloads | `test_contracts` |
| 4 | Oversized/deep/high-cardinality payloads | `test_payload_bounds` |
| 5 | Duplicate and out-of-order events | `test_perception_spine` |
| 6 | Stale source received recently | `test_perception_spine` |
| 7 | Conflicting sources / independence collision | `test_world_state` |
| 8 | Prompt injection in payload changes action fields | `test_contracts` |
| 9 | Cross-principal World State access | `test_world_state` |
| 10 | Reducer crash/restart/replay determinism | `test_world_state` |
| 11 | D102 unavailable or split-brain | `test_proposal_workspace` |
| 12 | One/duplicate/correlated/stub specialists | `test_proposal_workspace` |
| 13 | Ohana unavailable or poisoned values | `test_assessment` |
| 14 | Policy engine unavailable | `test_policy_bridge` |
| 15 | Approval anonymous/XSS/CSRF/replay | `test_policy_bridge` |
| 16 | Proposal changed after approval | `test_policy_bridge` |
| 17 | Capability used by wrong actuator | `test_policy_bridge`, `test_actuator_registry` |
| 18 | Capability replay / concurrent consumption | `test_policy_bridge` |
| 19 | Actuator timeout after possible side effect | `test_policy_bridge` |
| 20 | Blind retry prevention and reconciliation | `test_vertical_slice` |
| 21 | Partial multi-step execution and compensation | `test_vertical_slice` |
| 22 | Actuator lies about success | `test_autonomy` (verifier independence) |
| 23 | Outcome verifier unavailable or contradictory | `test_autonomy` |
| 24 | Self-generated prediction as outcome evidence | `test_autonomy` |
| 25 | Old direct path callable after migration | `test_actuator_registry` |
| 26 | Rollback restores fail-open/legacy authority | `test_invariant_guards` |
| 27 | Multi-worker, restart, clock-change, fencing | `test_perception_spine` (partial — see §6) |
| 28 | Audit persistence failure before protected effect | `test_perception_spine` |
| 29 | Event/trace context tampering | `test_contracts` |
| 30 | End-to-end data deletion across derivatives | `test_world_state` (partial — see §6) |

---

## 5. Known gaps — honest list

These are **open**, not hidden. Do not mark UH complete while this section is non-empty.

| ID | Gap | Impact | Where |
|---|---|---|---|
| G-01 | **0 of 33 actuators migrated.** All at `LEGACY` | The new path is enforced but unused; old paths still serve traffic | `common/actuator_registry/catalog.py` |
| G-02 | Perception spine runs in **shadow mode only** | Cortex still polls sensors point-to-point | `common/perception_spine/shadow.py` |
| G-03 | Six **unauthenticated side-effecting endpoints** still live | `backup-service /restore/postgres` can overwrite the DB with no auth | see table below |
| G-04 | Legacy `TrustLevel` scalar still referenced by `gate_autonomous_action()` | Two authority systems coexist | `agentic/trust_core.py` |
| G-05 | §16.27 multi-worker/clock-change coverage is **partial** | Leader-fencing untested under real concurrency | `test_perception_spine` |
| G-06 | §16.30 deletion-lineage coverage is **partial** | End-to-end erasure across learning derivatives untested | `test_world_state` |

### G-03 — unauthenticated endpoints awaiting disablement

| Service | Endpoint | Risk |
|---|---|---|
| backup-service | `POST /restore/postgres` | Database overwrite |
| browser-agent | `POST /click`, `POST /type` | Arbitrary web interaction |
| telegram-bot | `POST /alert` | Unauthenticated operator messaging |
| monitor-service | `/rules` CRUD | Alert-rule tampering |
| agentic | `POST /checkpoint/{id}/restore` | Breaker-state modification |
| notify-service | `POST /notify` | Notification spam |

---

## 6. Next authorised step

**Nothing is authorised to change live behaviour without an explicit operator decision.**

Recommended order when cutover is approved:

1. **G-03 first** — close the six unauthenticated endpoints. Highest risk, independent of UH migration.
2. **Tier 1 actuators** (11 read-only) — `disable_legacy_path()` → `MIGRATING` → `VERIFIED` → `ACTIVE`.
3. Re-run `make test-uh` after each tier; update §2 and §5 here.
4. Only then consider tier 2.

Do **not** advance a tier while any actuator in a lower tier is unmigrated —
`next_migration_candidates()` enforces this, and bypassing it is an §17 anti-pattern.

---

## 7. Working rules for this workstream

1. **`make test-uh` must be green before any commit.** No exceptions.
2. **Update this file in the same commit** as the code change. A status table that lags the code is how you get to 6,000 errors.
3. **`DECISIONS.md` is append-only.** Corrections are new entries.
4. **Every claim here is verifiable.** If a row says ✅, a named suite proves it. If it can't be proved, it is ⚠️ or ❌.
5. **No actuator advances past `LEGACY` without operator authorisation.**
6. **Never mark a gate green because the machinery exists.** Green means the behaviour is exercised by a passing test.
