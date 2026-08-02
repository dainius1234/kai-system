# Wave 1 — Dashboard Privileged Gateway Remediation Plan

**Branch:** `claude/project-rework-plan-pgvp35`
**Created:** 2026-08-02
**Scope:** all 96 `KAI-DASH-*` findings
**Status tool:** `make dashboard-findings`
**Tool's own tests:** `make test-dashboard-findings`

---

## 1. Why this plan exists

The dashboard findings were captured at commit `7adab8d`. Since then, P0
containment and the whole Unified Hunter programme have changed the tree.
Building straight from the finding list as written would mean fixing things
already fixed and missing things that moved.

So this plan is **not** a restatement of the audit. Every finding was
revalidated against the tree as it is now, mechanically, before any line of
remediation was planned. The revalidation is a re-runnable tool, not a
one-time assertion in a document — because a hand-maintained checklist
drifts, and drift is what produced the mess this programme exists to undo.

### Governing documents

| Order | Document | What it governs |
|---|---|---|
| 1 | [`CODE_AUDIT_BATCH_DASHBOARD_GATEWAY.md`](CODE_AUDIT_BATCH_DASHBOARD_GATEWAY.md) | **Authoritative finding list.** All 96 IDs, severities, recommendations |
| 2 | [`CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`](CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md) §3.4 | Wave 1 sequencing; maps to `KAI-CHAIN-001/002`, `KAI-ARCH-007` |
| 3 | [`CODE_AUDIT_IMPLEMENTATION_SEQUENCE_AND_CLOSURE_MATRIX.md`](CODE_AUDIT_IMPLEMENTATION_SEQUENCE_AND_CLOSURE_MATRIX.md) | Closure rules, including Rule 7 |
| 4 | **This file** | Track breakdown and live status of the dashboard work |
| 5 | [`UH_PROGRESS_TRACKER.md`](UH_PROGRESS_TRACKER.md) | Single source of truth for UH status overall |
| 6 | [`DECISIONS.md`](DECISIONS.md) | **Append-only.** Never edit an entry — correct with a new one |

### Programme Rule 7 applies without exception

> Finding counts remain unchanged until formal closure review. Planning
> documents do not reduce the 4,580 total. Closure is a separate
> evidence-backed register action.

Nothing in this plan closes a finding. `REMEDIATED` in the tool means *the
condition the finding describes no longer holds in the code* — it is
evidence for a future closure review, not the review itself.

---

## 2. Revalidated baseline, and where it stands now

Produced by `scripts/security/check_dashboard_findings.py`:

| Status | At `cb3f142` (baseline) | Now (Track A done) | Meaning |
|---|---|---|---|
| **LIVE** | 54 | **22** | the condition still holds |
| **PARTIAL** | 2 | **0** | materially reduced, not resolved |
| **REMEDIATED** | 3 | **37** | condition no longer holds (pending closure review) |
| **MANUAL** | 37 | **37** | not statically decidable; named for human review |
| **Total** | 96 | **96** | coverage self-audit enforces this |

Of the 22 still LIVE: **0 CRITICAL, 12 HIGH, 10 MEDIUM.** Every one of the
10 CRITICALs is remediated.

**Route surface:** 185 routes — 119 GET, 61 POST, 5 DELETE.
**66 mutating routes, all 66 now authenticated.** The 6 that serve without
a principal are the declared public list: `/health`, `/metrics` (liveness)
and the four HTML shells the browser must load *before* it can
authenticate. None is mutating, and the list lives in the checker, so
widening it is a deliberate edit rather than a side effect.

### Per-track status

| Track | LIVE | REMEDIATED | MANUAL | State |
|---|---|---|---|---|
| **A** Inbound identity | 0 | 5 | 0 | **Done** |
| **B** Mutating authority | 0 | 20 | 1 | **Done** bar one manual review |
| **C** Sensitive reads | 1 | 10 | 1 | `DASH-023` (hard-coded `keeper`) outstanding |
| **D** Failure semantics | 9 | 1 | 3 | Not started |
| **E** Bounds | 3 | 0 | 9 | Not started |
| **F** Media trust | 2 | 0 | 3 | Not started |
| **G** Disclosure | 1 | 0 | 4 | Not started |
| **H** Fan-out | 3 | 0 | 6 | Not started |
| **I** Hygiene | 3 | 1 | 10 | Not started |

### What the baseline revalidation found (historical — before Track A)

This is the snapshot that shaped the plan. It is kept as the record of why
the tracks are ordered the way they are; for current state, read the table
above or run `make dashboard-findings`.

| Finding | Was | Now | Evidence |
|---|---|---|---|
| `KAI-DASH-002` | CRITICAL — anonymous mode change via server-held token | **REMEDIATED** | No `DASHBOARD_GATE_TOKEN` anywhere; `/api/mode` returns `{"status": "local_only"}` and does not call Tool Gate |
| `KAI-DASH-013` | HIGH — mode-sync failure returns 200 | **REMEDIATED** | Premise removed with 002: there is no sync left to fail. Invalid mode raises 400 |
| `KAI-DASH-081` | MEDIUM — deliberate mode split when token missing | **REMEDIATED** | Same condition as 002 |
| `KAI-DASH-001` | CRITICAL — open privileged gateway | **PARTIAL** | All three compose files bind `127.0.0.1:8080:8080` (P0 containment) — `full.yml:197`, `minimal.yml:279`, `sovereign.yml:276`. Loopback only, but still zero inbound auth |
| `KAI-DASH-012` | HIGH — token lacks delegation evidence | **PARTIAL** | The static token is gone, but with no inbound principal there is nothing to delegate *from*. Re-opens when backend credentials return |

At that point everything else was LIVE or MANUAL, and **8 of the 10
CRITICALs were fully live** — `KAI-DASH-003` through `KAI-DASH-010`. Track A
closed all eight, and moved `001` and `012` from PARTIAL to REMEDIATED.

### Standing operator directive — checked on every run

> `BINANCE_API_KEY` and `BINANCE_API_SECRET` must never be exposed to the
> dashboard layer. They stay inside the broker-bridge service.

**Status: REMEDIATED and enforced.** The dashboard reads neither variable.
`dashboard/static/app.html:1130` names `BINANCE_API_KEY` in help text
("Configure BINANCE_API_KEY to see balance") — naming is fine, reading is
not, and the check distinguishes the two. This directive outranks the
finding list and is verified on every tool run, not just when convenient.

---

## 3. Track breakdown

All 96 findings are partitioned into 9 tracks. The partition is enforced by
the tool's coverage self-audit: every finding belongs to exactly one track,
and a missing or unknown finding fails the run.

| Track | Name | Findings | Count | Depends on |
|---|---|---|---|---|
| **A** | Inbound identity (foundation) | 001, 002, 011, 012, 018 | 5 | — |
| **B** | Authority on mutating routes | 003–010, 014, 019, 026–030, 032, 035–039 | 21 | A |
| **C** | Sensitive read authorisation | 020–025, 031, 033, 034, 040, 041, 044 | 12 | A |
| **D** | Failure semantics | 013, 015, 016, 054, 061–067, 080, 082 | 13 | — |
| **E** | Request/response bounds | 017, 042, 045–049, 053, 056, 076, 092, 093 | 12 | — |
| **F** | Media and filename trust | 050, 051, 089, 090, 091 | 5 | — |
| **G** | Disclosure minimisation | 052, 055, 068, 069, 077 | 5 | A (partly) |
| **H** | Fan-out and lifecycle | 043, 057–060, 074, 075, 085, 087 | 9 | — |
| **I** | Config, validation and hygiene | 070–073, 078, 079, 081, 083, 084, 086, 088, 094–096 | 14 | — |
| | | **Total** | **96** | |

### Sequencing

**A is the hard prerequisite for B and C.** Together B and C are 33 of the
96 findings, including 8 of the 10 CRITICALs, and every one of them reduces
to the same sentence: *"anonymous callers can do X."* There is no point
fixing them one route at a time — the fix is a principal model plus a
per-route authority declaration. Do A first, then B and C become mechanical.

**D through I are independent of A** and can proceed in any order. They are
sequenced after B and C because a route that is still anonymous does not
become materially safer by returning a better status code.

**Order: A → B → C → D → E → F → G → H → I.**

---

## 4. Track A — inbound identity (the foundation)

**Findings:** `KAI-DASH-001`, `002`, `011`, `012`, `018`
**Blocks:** tracks B and C (33 findings, 8 CRITICALs)

### What is actually wrong

The dashboard has **zero** inbound authentication references. Every one of
185 routes is anonymous, and the dashboard proxies to Agentic, memU,
Supervisor, Tool Gate, Financial Awareness, Browser Agent, Monitor, Files,
Notify, Email and Broker. It is a single unified control plane for the whole
stack, reachable by anyone who can reach the port.

### Design constraints

1. **Fail closed.** No secret configured → `503`, never open. This matches
   `common/service_auth.py`, which already protects 32 routes across 8
   services. Do not invent a second, weaker convention.
2. **Per-route declaration, not blanket middleware.** Middleware makes
   authority invisible at the route. The tracker deliberately only counts a
   route as authenticated when it declares its own dependency — so that
   `KAI-DASH-018` (least privilege) has something to attach to, and so a
   new route is unauthenticated *by default and visibly so*.
3. **Broker credentials stay out.** Non-negotiable; see §2.
4. **A principal, not just a token.** `KAI-DASH-011` asks for a verified
   principal with role and session ownership. A shared bearer token alone
   satisfies 001 but leaves 011 and 012 live. Build the principal now.

### Definition of done for Track A — all met

- [x] `common/dashboard_auth.py` with a fail-closed `require_dashboard_auth`
      returning a `DashboardPrincipal` (identity, role, session)
- [x] Scope model so each route declares the authority it needs (018) —
      5 scopes, 3 roles, distribution verified rather than merely present
- [x] Every route declares auth or is on the explicit public list
- [x] `make test-dashboard-auth` — 89 tests: missing config → 503; absent
      header → 401; wrong token → 401; insufficient role → 403; valid →
      principal populated. Prefix and suffix token variants are refused,
      a duplicate token across identities fails closed, and a session
      header carries into the audit trail without conferring authority
- [x] `make dashboard-findings` shows Track A at zero LIVE
- [x] Architecture rule 6 still passes; 15/15 rules accounted for
- [x] `DECISIONS.md` D142 records the principal model and why

**Roles:** `viewer` reads operational status only; `operator` adds
sensitive reads and routine writes; `keeper` alone may rewrite identity
state or drive external action. An operator who can rewrite `SOUL.md` can
rewrite what the system is, so that stays with the keeper.

---

## 5. Tracks B and C — authority on routes

Once A lands, these are mechanical but not trivial: each route needs the
*right* scope, not merely *a* guard. Getting 66 mutating routes uniformly
"authenticated" while giving every caller full authority would satisfy the
tracker and miss `KAI-DASH-018` entirely.

**Track B — 66 mutating routes.** Highest-value first:
`/api/soul`, `/api/agents-registry`, `/api/chat`, `/api/values/learn`,
`/api/reminders/*`, `/api/browser/*`, `/api/monitor/rules*`,
`/api/files/watch` — the 8 live CRITICALs.

**Track C — 20 sensitive reads.** Memory, financial, broker, email,
security-audit, logs, and the SSE event bus. Note `KAI-DASH-023`
(hard-coded global `keeper` identity) is a *correctness* fix inside this
track: authenticating the caller is pointless while every request is
executed as `keeper` regardless of who asked.

`KAI-DASH-044` (per-event isolation on SSE) cannot be resolved until a
principal exists — it is MANUAL today for exactly that reason, and becomes
actionable the moment Track A lands.

---

## 6. Tracks D–I — summary of approach

| Track | Approach | Notable |
|---|---|---|
| **D** Failure semantics | Stop returning success-shaped bodies for failures. 28 handlers currently swallow backend failure into a 200; 10 exception paths return empty/neutral data as if authoritative | `061` (any 2xx = healthy node), `063`/`064` (go/no-go uses the wrong metrics entirely), `065`/`066` (fabricated timestamps) are evidence-integrity bugs, not cosmetics |
| **E** Bounds | Reuse the `§16.4` payload-bounds work already built in `common/perception_spine/ingress.py` rather than inventing new limits | `042` (SSE admission) is a denial-of-service vector: each client holds a dedicated Redis pubsub |
| **F** Media trust | Derive content types from the backend response instead of forcing them; canonicalise filenames before forwarding | `089`/`090` currently force `audio/mpeg` and `image/png` unconditionally |
| **G** Disclosure | Minimise `/health` and `/` payloads; strip internal URLs and transport diagnostics from error paths | `069` confirmed: `/health` discloses `tool_gate_url`, `policy_version`, `policy_hash` |
| **H** Fan-out | Concurrent node probes, shared HTTP client, cached app shell | `057` sequential fan-out can consume the sum of all timeouts; `074` counts 47 per-request client constructions |
| **I** Hygiene | Timezone-aware timestamps, security headers, input validation, mandatory audit with actor and operation digest | `096`: audit currently defaults to *not required* and records only method/path/status |

---

## 7. How this work is tracked

Three mechanisms, all re-runnable. None of them is a document that someone
has to remember to update.

| Mechanism | Command | What it guarantees |
|---|---|---|
| Finding revalidation | `make dashboard-findings` | Live status of all 96 findings against current code |
| Tracker self-tests | `make test-dashboard-findings` | The tracker can actually fail — 43 tests |
| Architecture gate | `python3 scripts/security/check_architecture_rules.py` | Rule 6 blocks new unprotected side-effect routes |

`make test-uh` runs the tracker tests as part of the aggregate suite.

### Why the tracker has its own test suite

Twice in this programme a check that looked green was checking nothing: a
negative test whose injected violation was never written to disk, and an
architecture gate that silently omitted 6 of its 15 rules while reporting a
clean pass. Both passed. Both were worthless.

So the tracker's tests substitute a synthetic dashboard with the
remediation applied and assert each check **flips** — LIVE when the defect
is present, REMEDIATED when it is not. A check that cannot fail is not
evidence, and the coverage self-audit fails the run if any of the 96
findings is missing from the table.

`MANUAL` placeholders are tagged and inert: a test asserts they can only
ever report `MANUAL`, so an unreviewed finding can never drift into looking
resolved.

---

## 8. Current position

- **Track A: complete.** `common/dashboard_auth.py` gives every route a
  verified principal with identity, role and session, and a declared
  scope. 179 of 185 routes are authenticated; the other 6 are the
  declared public list. 89 tests.
- **Track B: complete** bar `KAI-DASH-014`, which needs a per-backend
  idempotency review rather than a code change here.
- **Track C: one finding outstanding** — `KAI-DASH-023`, the hard-coded
  global `keeper` identity. Authenticating the caller does not help while
  every request still executes as `keeper`, so this is the next step.
- **Tracks D–I:** not started.
- **Findings formally closed: 0** (Rule 7 — closure is a separate
  evidence-backed register action).

### Deployment consequence

The gateway now fails closed, so `KAI_DASHBOARD_TOKEN` must be provisioned
or the dashboard answers 503 on every protected route. Three things enforce
that rather than leaving it to memory:

- `make setup-service-token` generates it alongside `KAI_SERVICE_TOKEN`,
  as a **separate** secret — a browser-held credential must not also
  authorise service-to-service calls
- `make preflight` blocks a deploy that is missing it, uses a placeholder,
  reuses the service token, or names an unknown role
- all three compose profiles pass it to the dashboard service, and
  preflight verifies that wiring

Progress is read from `make dashboard-findings`, not from this section.
This file records the plan; the tool records the state.
