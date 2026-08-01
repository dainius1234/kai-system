# Unified Hunter — Progress Tracker

> **This is the single source of truth for Unified Hunter work.**
> If this file and any other doc disagree, **this file wins** for UH status.
> Every UH change must update this file in the same commit.

**Last updated:** 2026-08-01 (third gap-closure pass — all gaps closed)
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
| UH-7 Actuator registry + migration | ✅ Done | `78dbd60` | `make test-actuator-registry` | 75 |
| UH-8 Autonomy requalification | ✅ Done | `07d3614` | `make test-autonomy` | 174 |
| §16.4 Payload bounds | ✅ Done | this commit | `make test-payload-bounds` | 24 |
| §16.13 Ohana / assessments | ✅ Done | this commit | `make test-assessment` | 56 |
| §16.26 Rollback guards | ✅ Done | `5b882a4` | `make test-invariant-guards` | 18 |
| §16.27 Concurrency/clock/fencing (G-05) | ✅ Done | `99c1ee9` | `make test-concurrency-clock` | 51 |
| G-03 Service authentication | ✅ Done | `99c1ee9` | `make test-service-auth` | 55 |
| §16.30 Erasure lineage (G-06) | ✅ Done | `ebae38d` | `make test-erasure` | 75 |
| G-04 Legacy trust bridge | ✅ Done | `51e0934` | `make test-legacy-bridge` | 58 |
| G-01/G-02 Tier-1 migration + active mode | ✅ Done | `4795b7d` | `make test-migration` | 125 |
| G-01b/G-02b/G-07/G-08 second closure pass | ✅ Done | `8675e90` | (folded into suites above) | — |
| G-09 Full-catalogue migration (tiers 2–8) | ✅ Done | this commit | `make test-full-migration` | 75 |
| G-11 All flags enabled, end to end | ✅ Done | this commit | `make test-flags-enabled` | 37 |
| G-10 Live endpoint verification | ✅ Done | this commit | `make verify-live-endpoints` | 10/13 live |
| | | | **Total** | **1,384** |

**UH-7 is complete.** All **34** actuators across all 8 tiers have dispatch handlers and
migrate to ACTIVE in ascending risk order. Every legacy path is **verified** closed against
the source tree rather than marked closed by a flag. Tier-1 endpoint paths were checked
against the routes each service declares (which found four wrong paths) and then called
against running services (which found a missing query parameter).

> The catalogue holds **34** actuators, not 33. Earlier entries said 33 — that was a
> miscount, corrected here.

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
| Actuators actually migrated | ✅ **34 of 34** — all tiers |
| Activation requires a dispatch handler | ✅ `migrate_tier()` refuses to activate a handler-less actuator |
| Legacy closure is verified, not asserted | ✅ `verify_legacy_closed()` checks the source tree before the flag may be set |
| Destructive actions declare irreversibility | ✅ receipts carry `side_effects` incl. `irreversible` |
| Uncertain effects flagged for reconciliation | ✅ a failed POST sets `effect_uncertain` |

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
| 27 | Multi-worker, restart, clock-change, fencing | `test_concurrency_clock` ✅ full |
| 28 | Audit persistence failure before protected effect | `test_perception_spine` |
| 29 | Event/trace context tampering | `test_contracts` |
| 30 | End-to-end data deletion across derivatives | `test_erasure` ✅ full (5 layers) |

---

## 5. Known gaps — honest list

Every tracked gap is now closed. What follows the closed table is not a list of
unfinished work but a list of **limits that only a production environment can
retire** — recorded so nobody mistakes a green suite for a live cutover.

### Closed

| ID | Gap | Closed by | Verified by |
|---|---|---|---|
| ~~G-01~~ | Tier-1 actuators un-migrated | Real HTTP handlers + `migrate_tier()` | `make test-migration` |
| ~~G-01b~~ | Paths never checked against real routes | Source-level route verification (**found 4 wrong paths**) | `make test-migration` |
| ~~G-02~~ | Perception spine shadow-only | Active mode, additive, defaults to shadow | `make test-migration` |
| ~~G-02b~~ | No cutover path off legacy Cortex polling | `cortex_source.py` — `KAI_CORTEX_SOURCE` | `make test-migration` |
| ~~G-03~~ | Six unauthenticated side-effecting endpoints | `common/service_auth.py`, fail-closed | `make test-service-auth` |
| ~~G-04~~ | Legacy `TrustLevel` as a second authority | Legacy may only deny, never grant | `make test-legacy-bridge` |
| ~~G-05~~ | §16.27 concurrency/clock/fencing partial | `FencedLease` + concurrency suite | `make test-concurrency-clock` |
| ~~G-06~~ | §16.30 deletion lineage partial | `common/erasure/` across 5 layers | `make test-erasure` |
| ~~G-07~~ | `KAI_SERVICE_TOKEN` unset in compose | Wired into 3 profiles + `.env.example` | `docker-compose*.yml` |
| ~~G-08~~ | 22 pre-existing `agentic-routes` failures | Helper renames + breaker-leak fixture | `make test-agentic-routes` (170 pass) |
| ~~G-09~~ | 22 of 34 actuators at `LEGACY` | Mutating handlers + verified legacy closure | `make test-full-migration` |
| ~~G-10~~ | No handler called against a running service | 9 services started; **10/13 live-verified** (**found a missing query param**) | `make verify-live-endpoints` |
| ~~G-11~~ | Migration flags never exercised together | Full pipeline with all 4 flags ON | `make test-flags-enabled` |

### Environmental limits (not defects)

| ID | Limit | Why it cannot close here |
|---|---|---|
| E-01 | 3 of 13 tier-1 endpoints not live-verified | `broker-bridge` needs Binance credentials and outbound access. Routes confirmed to **exist** (502/503, never 404) |
| E-02 | Mutating (tier 2–8) handlers verified against an injected client | Calling them for real causes real side effects — a database restore is not a test |
| E-03 | All four migration flags **default to the legacy path** | Correct for deployment. Each is proven to work when enabled, and each has a tested fallback |

**None of these is a code defect.** E-01 and E-02 need credentials and a decision to
accept side effects; E-03 is the intended default.

### Endpoints now authenticated

| Service | Endpoints | Status |
|---|---|---|
| backup-service | `/restore/postgres` + 6 backup routes | ✅ |
| browser-agent | `/click`, `/type`, `/navigate`, `/run` | ✅ |
| telegram-bot | `/alert` | ✅ |
| monitor-service | `/rules` CRUD (6 routes) | ✅ |
| agentic | `/checkpoint/{id}/restore`, `DELETE /checkpoint/{id}` | ✅ |
| notify-service | `/notify` | ✅ |
| vault-sync | `/export` | ✅ |
| executor | `/execute` | ✅ |

**These fail closed.** `KAI_SERVICE_TOKEN` is wired into all compose profiles but ships
empty — generate one (`openssl rand -hex 32`) or these endpoints return 503.

---

## 6. Next authorised step

**Nothing here changes live behaviour on its own.** Every flag defaults to the
legacy path, so deploying this branch changes nothing until a flag is set.

Cutover order, each independently revertible:

1. **`KAI_SERVICE_TOKEN`** — generate with `openssl rand -hex 32`. Required: eight
   services now fail closed without it.
2. **`make verify-live-endpoints`** against the running stack. Expect `WRONG=0`;
   `UPSTREAM` entries for `broker-bridge` are expected without Binance credentials.
3. **`KAI_PERCEPTION_MODE=active`** in one environment. Watch `events_reduced`
   against `events_accepted`.
4. **`KAI_CORTEX_SOURCE=world_state`** once (3) is healthy. Falls back to polled
   state on an empty world, so the blast radius is small.
5. **`KAI_AUTONOMY_ENFORCE=true`** only once `migration_report()` reports
   `ready_to_enforce: true`. It reports `false` today, correctly.

Re-run `make test-uh` after each step and update §2 and §5 here.

---

## 7. Working rules for this workstream

1. **`make test-uh` must be green before any commit.** No exceptions.
2. **Update this file in the same commit** as the code change. A status table that lags the code is how you get to 6,000 errors.
3. **`DECISIONS.md` is append-only.** Corrections are new entries.
4. **Every claim here is verifiable.** If a row says ✅, a named suite proves it. If it can't be proved, it is ⚠️ or ❌.
5. **No actuator advances past `LEGACY` without operator authorisation.**
6. **Never mark a gate green because the machinery exists.** Green means the behaviour is exercised by a passing test.
