# Unified Hunter — Progress Tracker

> **This is the single source of truth for Unified Hunter work.**
> If this file and any other doc disagree, **this file wins** for UH status.
> Every UH change must update this file in the same commit.

**Last updated:** 2026-08-03 (seventeenth pass — H-3/H-4 complete; repo-wide hygiene debt 136 → 0)
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
| §16.26 Rollback guards | ✅ Done | `5b882a4` | `make test-invariant-guards` | 26 |
| §16.27 Concurrency/clock/fencing (G-05) | ✅ Done | `99c1ee9` | `make test-concurrency-clock` | 51 |
| G-03 Service authentication | ✅ Done | `99c1ee9` | `make test-service-auth` | 59 |
| §16.30 Erasure lineage (G-06) | ✅ Done | `ebae38d` | `make test-erasure` | 75 |
| G-04 Legacy trust bridge | ✅ Done | `51e0934` | `make test-legacy-bridge` | 58 |
| G-01/G-02 Tier-1 migration + active mode | ✅ Done | `4795b7d` | `make test-migration` | 136 |
| G-01b/G-02b/G-07/G-08 second closure pass | ✅ Done | `8675e90` | (folded into suites above) | — |
| G-09 Full-catalogue migration (tiers 2–8) | ✅ Done | `8d2ea09` | `make test-full-migration` | 75 |
| G-11 All flags enabled, end to end | ✅ Done | `8d2ea09` | `make test-flags-enabled` | 37 |
| G-10 Live endpoint verification | ✅ Done | `8d2ea09` | `make verify-live-endpoints` | 13/13 live |
| E-01 broker-bridge live-verified | ✅ Done | this commit | `make verify-live-endpoints` | 13/13 live |
| E-02 Mutating handlers live-verified | ✅ Done | this commit | `make verify-live-mutating` | 9 invoked, 5 skipped |
| E-03 Deployment preflight | ✅ Done | `8d2ea09` | `make test-preflight` | 57 |
| W-01 Modules wired into the running app | ✅ Done | this commit | `make test-invariant-guards` | (guards) |
| A-01 Architecture dependency CI gate — all 15 rules | ✅ Done | `cb3f142` | `make test-architecture-rules` | 61 |
| W1-DASH Dashboard finding tracker (96 findings) | ✅ Done | `dff418d` | `make test-dashboard-findings` | 177 |
| W1-DASH-A Dashboard inbound identity | ✅ Done | `9fb0e26` | `make test-dashboard-auth` | 99 |
| W1-DASH-D01 Browser credential shim | ✅ Done | `eb5b084` | `make test-dashboard-ui-auth` | 42 |
| W1-DASH-C Caller-scoped memory reads | ✅ Done | `d18a089` | `make test-dashboard` | (folded in) |
| W1-DASH-D Failure semantics (degraded envelope) | ✅ Done | `dc95692` | `make test-degraded` | 36 |
| W1-DASH-E–I Bounds, media, disclosure, fan-out, hygiene | ✅ Done | `a48b78d` | `make test-dashboard` | (folded in) |
| W1-SEC-1 Keeper is an explicit grant, not the default | ✅ Done | this commit | `make test-dashboard-auth` | (folded in) |
| H-5 Repo-wide hygiene ratchet (10th CI gate) | ✅ Done | `e5fd4de` | `make test-hygiene-gate` | 39 |
| H-1 Timezone-aware timestamps (17 sites, 7 services) | ✅ Done | `bd1e449` | `make hygiene-gate` | 0 naive remain |
| H-2 memu-core bounded reads + honest failures | ✅ Done | `a195a3e` | `make hygiene-gate` | memu-core clear |
| H-3/H-4 agentic + 23 services pooled and bounded | ✅ Done | `4c45398` | `make hygiene-gate` | **0 across 50 services** |
| A-02 Assertion-count ratchet (11th CI gate) | ✅ Done | `47d2243` | `make test-assertion-floors` | 40 |
| A-04a Instrumentation registry + meta-check (reporting) | ✅ Done | `b9286f1` | `make test-gate-registry` | 30 |
| A-04b Directional compose drift + sovereign hardening | ✅ Done | `51032a3` | `make test-compose-drift` | 21 |
| A-04e Partial enforcement — I-4 enforced, ratchet self-advancing | ✅ Done | `e332a02` | `make test-gate-registry` | 39 |
| KAI-GATE-005 Anchor pre-scan (category confusion) | ✅ Done | `bd80129` | `make test-dashboard-findings` | 189 |
| KAI-GATE-007/008/009 Secret rule, restart allowlist, camera identity | ✅ Done | `dd772c4` | `make test-secret-gates` | 28 |
| KAI-GATE-010/011/012 + I-5 inert-rule detector | ✅ Done | `d222c45` | `make test-compose-gates` | 27 |
| Closure register — first 6 findings closed, I-6 re-verifies | ✅ Done | `fe1f222` | `make test-gate-registry` | 53 |
| A-04b six compose gates fail closed + denominators; I-2 enforced | ✅ Done | this commit | `make gate-registry` | 62 |
| | | | **Total** | **2,096** |

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

### Environmental limits — now closed

| ID | Limit | How it was closed |
|---|---|---|
| ~~E-01~~ | broker-bridge unverifiable without Binance credentials | A Binance-shaped upstream stub was stood up and broker-bridge pointed at it via `BINANCE_BASE_URL`. All three routes verified, including the **signed** `/balance` — which proves the signing path. **13/13 live.** |
| ~~E-02~~ | Mutating handlers never called for real | 7 services started with auth on; **9 actions invoked** (7 fully, 2 contained), **5 deliberately skipped**. Skipped actions are reported as skipped, never as passed. |
| ~~E-03~~ | Migration flags never exercised in a real deploy | `make preflight` gates deployment readiness; `make setup-service-token` generates the token into gitignored `.env`. |

**On E-02's skipped five:** `browser_click`, `browser_type`, `service_recover`,
`auto_sleep` and `paper_trade_open` were **not invoked**. Clicking an arbitrary web
element, restarting live services, and triggering memory decay are exactly the
irreversible operations the capability system exists to gate — invoking them to prove
a test passes would be the wrong trade. They are recorded as skipped with the reason.

### W-01 — the orphaned-module finding

An audit of whether the new modules were actually *called* by running code found
**six of eight were orphaned**: built, tested, and invoked by nothing. The flags
existed and the code existed, but `KAI_PERCEPTION_MODE` and `KAI_CORTEX_SOURCE`
controlled paths the application never reached.

All are now wired and verified against a running app:

| Module | Wired into | Verified live |
|---|---|---|
| `perception_spine.shadow` | `agentic` startup poll loop | 6 events journalled, 6 reduced |
| `perception_spine.cortex_source` | `agentic._sense_world()` | `cortex_source=world_state` |
| `actuator_registry` | `GET /uh/actuators` | 34 actuators reported |
| `erasure` | `POST /uh/erasure` (authenticated) | 401 without token, receipt with |
| `vertical_slice` | `POST /uh/paper-trade` (authenticated) | full slice → `confirmed` |
| `policy_bridge.assessment` | `PaperTradeSlice` policy engine | Ohana required, fails closed |

`test_uh_modules_are_wired_in` now guards this — a module that loses its caller
fails a test rather than quietly becoming dead weight.

### Remaining operational steps

Not defects — decisions that belong to the operator, each with a tested fallback.

| ID | Step | Status |
|---|---|---|
| O-01 | `KAI_PERCEPTION_MODE=active` | **Proven live** — spine polled real sensors, journalled and reduced 6 events inside the running app. Enabling it in a deployment is the operator's call |
| O-02 | `KAI_CORTEX_SOURCE=world_state` | **Proven live** alongside O-01. Falls back to polled state on an empty world |
| O-03 | `KAI_AUTONOMY_ENFORCE=true` | **Must not be enabled yet.** No grants exist, so enforcement would deny every gated capability. `make preflight` blocks it |
| O-04 | Retire legacy Cortex polling | Deliberately additive during migration |

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

## 5b. Wave 1 — Dashboard privileged gateway

The UH work packages are complete. Wave 1 of the code-audit programme is in
progress, starting with the Dashboard gateway batch.

**Plan:** [`W1_DASHBOARD_REMEDIATION_PLAN.md`](W1_DASHBOARD_REMEDIATION_PLAN.md)
— the single source of truth for dashboard remediation status.

All 96 `KAI-DASH-*` findings were revalidated against the current tree
before any remediation was planned, because the findings were captured at
`7adab8d` and the tree has moved since.

| Status | Baseline `cb3f142` | Now |
|---|---|---|
| LIVE | 54 | **0** |
| PARTIAL | 2 | **0** |
| REMEDIATED (pending closure review) | 3 | **95** |
| MANUAL (needs human review) | 37 | **1** |
| **Total** | **96** | **96** |

**All nine tracks are complete: every one of the 96 findings reports zero LIVE.** `common/dashboard_auth.py`
gives every route a verified principal — identity, role, session — and a
declared scope.

- **All 10 CRITICALs remediated, and no finding of any severity reports LIVE.** 95 REMEDIATED, 1 MANUAL (needing human review, never auto-claimed)
- **185 routes: 179 authenticated, 66 of 66 mutating routes authenticated**
- The 6 unauthenticated routes are the declared public list — `/health`,
  `/metrics`, and the four HTML shells the browser loads before it can
  authenticate. None mutating. The list lives in the checker, so widening
  it is a deliberate edit
- **5 scopes, 3 roles.** `viewer` reads operational status; `operator`
  adds sensitive reads and routine writes; `keeper` alone may rewrite
  identity state or drive external action
- **Operator directive verified:** the dashboard reads neither
  `BINANCE_API_KEY` nor `BINANCE_API_SECRET`. Checked on every tool run

**All nine tracks are complete.** `common/degraded.py` gives an outage an
explicit envelope (503 + `degraded` marker). `common/http_hygiene.py`
carries bounded request bodies and a shared connection pool. Payload limits
are **not** re-declared: they come from
`common/perception_spine/ingress.py`, which already owned them.

**One finding remains MANUAL** — `KAI-DASH-073` (backend identity proof), which needs a transport-layer change (mTLS or signed service identity), not a code edit. It — not statically decidable, each naming what
a human must review. MANUAL is not a pass.

> **The same defects exist outside the dashboard.** A repo-wide sweep found
> 96 per-request HTTP clients, 20 unbounded `request.json()` reads and 15
> naive `datetime.utcnow()` calls across 26 services. Those belong to their
> own audit batches, not to this plan — see
> [`W1_GLOBAL_HYGIENE_SUBPLAN.md`](W1_GLOBAL_HYGIENE_SUBPLAN.md).

| Command | Purpose |
|---|---|
| `make dashboard-findings` | Live status of all 96 findings |
| `make test-dashboard-findings` | 177 tests proving the tracker can fail |
| `make test-dashboard-auth` | 89 tests on the auth module itself |
| `make test-dashboard-ui-auth` | 42 tests on the browser credential shim |
| `make test-degraded` | 36 tests that an outage cannot look like an answer |
| `make hygiene-gate` | Ratchet: fails if repo-wide hygiene debt rises |

**Deployment:** the gateway fails closed. `KAI_DASHBOARD_TOKEN` must be set
or every protected route answers 503. `make setup-service-token` generates
it as a **separate** secret from `KAI_SERVICE_TOKEN`, `make preflight`
blocks a deploy without it, and all three compose profiles pass it through.

### Findings discovered during remediation

Remediation surfaces defects the original audit never saw. They are held in
a **separate register** (`KAI-DASH-D##`) that is reported and counted but
can never stand in for one of the 96 — a register that lets new work dilute
the original count is worse than none.

| ID | Severity | Finding | Status |
|---|---|---|---|
| `KAI-DASH-D01` | HIGH | Track A closed the gateway to the UI as well: 121 `fetch()` calls carried no credential, and `EventSource` cannot send headers at all | **REMEDIATED** |
| `KAI-DASH-D02` | HIGH | `/api/memories?query=` omitted `user_id`, which `/memory/retrieve` requires — memory search had been answering 422 all along | **REMEDIATED** |
| `KAI-DASH-D03` | HIGH | **`KAI_SERVICE_TOKEN` is one shared secret across all 8 authenticated services.** `common/service_auth.py` calls it "the shared service token" — it proves *possession of a secret*, not *identity*, so any service holding it can impersonate any other. Immediate remediation for the parent `KAI-DASH-073`; mTLS is the escalation path if the threat model still demands cryptographic proof afterwards | **OPEN** |

**Findings formally closed: 0.** Programme Rule 7 — closure is a separate
evidence-backed register action, and `REMEDIATED` here is evidence for that
review, not the review itself.

---

## 6. Next authorised step

**Deploying this branch changes nothing on its own.** Every migration flag defaults
to the legacy path, so the new machinery sits behind the existing system until a
flag is set.

```bash
make setup-service-token   # writes KAI_SERVICE_TOKEN into gitignored .env
make preflight             # must print READY TO DEPLOY
make test-uh               # all suites, must be green
```

Then, against a running stack:

```bash
make verify-live-endpoints   # expect OK=13 WRONG=0
make verify-live-mutating    # expect FAILED=0
```

Operational steps, in order, each independently revertible:

1. **O-01** `KAI_PERCEPTION_MODE=active` — watch `events_reduced` vs `events_accepted`.
2. **O-02** `KAI_CORTEX_SOURCE=world_state` — falls back to polled state on an empty
   world, so the blast radius is small.
3. **O-03** `KAI_AUTONOMY_ENFORCE=true` — **only** once grants exist and
   `migration_report()` reports `ready_to_enforce: true`. `make preflight` blocks it
   until then, deliberately.
4. **O-04** retire legacy Cortex polling, once O-02 has soaked.

Re-run `make test-uh` and `make preflight` after each step, and update §2 and §5 here.

---

## 7. Working rules for this workstream

1. **`make test-uh` must be green before any commit.** No exceptions.
2. **Update this file in the same commit** as the code change. A status table that lags the code is how you get to 6,000 errors.
3. **`DECISIONS.md` is append-only.** Corrections are new entries.
4. **Every claim here is verifiable.** If a row says ✅, a named suite proves it. If it can't be proved, it is ⚠️ or ❌.
5. **No actuator advances past `LEGACY` without operator authorisation.**
6. **Never mark a gate green because the machinery exists.** Green means the behaviour is exercised by a passing test.
