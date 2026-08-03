# Sub-plan — Repository-wide HTTP and Time Hygiene

**Status:** proposed, **not started**. No code outside `dashboard/` has been changed.
**Parent:** [`W1_DASHBOARD_REMEDIATION_PLAN.md`](W1_DASHBOARD_REMEDIATION_PLAN.md)
**Survey command:** `make hygiene-survey`

---

## 1. Why this exists

Three of the dashboard findings — `KAI-DASH-017` (unbounded bodies),
`KAI-DASH-074` (a connection pool per request) and `KAI-DASH-083` (naive
timestamps) — were fixed inside `dashboard/app.py`.

That was the wrong altitude, twice over:

1. **The defects are not dashboard defects.** They are repository-wide
   habits. A sweep found them in 27 of 50 services.
2. **Fixing one file created a duplicate.** The dashboard's payload-bound
   implementation was a *second* set of limits, while
   `common/perception_spine/ingress.py` already owned one — the exact
   duplication the fix's own comment claimed to be avoiding.

The duplication is already corrected: bounds and pooling now live in
`common/http_hygiene.py`, and the limits come from the perception spine
rather than being re-declared. **This sub-plan is about the rest.**

### What this sub-plan does *not* do

It does not renumber or absorb anything. Each affected service has its own
audit batch and its own finding IDs. This is a **cross-cutting view** of
one class of defect, so that fixing it does not happen 27 times in 27
different ways. Programme Rule 7 is untouched: nothing here closes a
finding.

---

## 2. The survey

Counted from each service's `app.py`. Reproduce with `make hygiene-survey`.

| Service | Per-request HTTP clients | Unbounded `request.json()` | Naive `utcnow()` | 200-on-failure |
|---|---:|---:|---:|---:|
| `agentic` | **37** | 6 | 5 | 1 |
| `memu-core` | 0 | **14** | 0 | 2 |
| `supervisor` | 12 | 0 | 0 | 0 |
| `telegram-bot` | 6 | 0 | 0 | 1 |
| `heartbeat` | 4 | 0 | 0 | 0 |
| `perception/camera` | 4 | 0 | 0 | 0 |
| `fusion-engine` | 3 | 0 | 0 | 0 |
| `monitor-service` | 3 | 0 | 0 | 0 |
| `screen-watcher` | 3 | 0 | 0 | 0 |
| `vault-sync` | 3 | 0 | 0 | 0 |
| `backup-service` | 2 | 0 | 3 | 0 |
| `ledger-worker` | 2 | 0 | 3 | 0 |
| `broker-bridge` | 2 | 0 | 0 | 0 |
| `perception/audio` | 2 | 0 | 0 | 0 |
| `skill-hunter` | 2 | 0 | 0 | 0 |
| `verifier` | 2 | 0 | 0 | 0 |
| `financial-awareness` | 0 | 0 | 2 | 0 |
| `memory-compressor` | 1 | 0 | 1 | 0 |
| `metrics-gateway` | 1 | 0 | 1 | 0 |
| `memu-graph` | 0 | 0 | 0 | 1 |
| 8 others (1 client each) | 8 | 0 | 0 | 0 |
| **Total** | **96** | **20** | **15** | **5** |

**Grand total: 136**, across 27 of 50 services. The other 23 carry none of
these.

Two services carry most of it: **`agentic` (49 of 136)** and
**`memu-core` (16)** — together **48%**.

> The `200-on-failure` count is 5, not the 1 an initial `grep` suggested.
> The survey parses the AST and only counts *route* handlers, because a
> helper returning a dict is not an HTTP 200. The table above is generated
> from the tool, not typed by hand — which is the point.

### What each column costs

- **Per-request HTTP clients** — every handler opens and tears down a
  connection pool. Under load this is the dominant cost, and it is the
  reason a slow backend degrades a whole service rather than one route.
- **Unbounded `request.json()`** — the service becomes an amplifier. One
  oversized or deeply nested body becomes work in whatever it forwards
  to. `memu-core` is the sharpest case: it is the memory store, so a
  pathological body lands on persistent state.
- **Naive `utcnow()`** — timestamps with no timezone. These end up in the
  ledger, in backups and in memory records, where "is this before that?"
  needs to be answerable across a restart or a host in another zone.
- **200-on-failure** — an outage that reads as an answer. Only 5 outside
  the dashboard, which had 30: the dashboard was unusual, not
  representative. Worth knowing before assuming the fix must be applied
  as widely as the other three.

---

## 3. The mechanisms already exist

Nothing new needs designing. Adoption is an import and a call.

| Concern | Module | Adoption |
|---|---|---|
| Bounded bodies | `common/http_hygiene.bounded_json` | replace `await request.json()` |
| Pooled connections | `common/http_hygiene.pooled_client` | replace `httpx.AsyncClient(` |
| Degraded envelope | `common/degraded.degraded_response` | replace success-shaped fallbacks |
| Aware timestamps | `datetime.now(timezone.utc)` | replace `datetime.utcnow()` |

`pooled_client` is a genuine drop-in: it returns a real `httpx.AsyncClient`
with a shared transport, so existing call sites *and* existing tests that
patch `httpx.AsyncClient` keep working unchanged. That property was
deliberate and is worth preserving in any alternative approach — a
module-global client was tried first and rejected because it outlives its
event loop and survives test patching, making suites order-dependent.

---

## 4. Proposed sequence

Ordered by risk, lowest first. Each step is independently revertible and
should land with the affected service's tests green.

| Step | Scope | Why this order |
|---|---|---|
| **H-1** | `datetime.utcnow()` → aware, all 15 sites | Mechanical, no behaviour change, no API change |
| **H-2** | `memu-core` — 14 unbounded reads, 2 success-shaped fallbacks | Highest consequence per defect: pathological bodies reach persistent state |
| **H-3** | `agentic` — 37 clients, 6 reads, 5 timestamps, 1 fallback | Largest single concentration (49 of 136); do it once the pattern is proven on a smaller service |
| **H-4** | Remaining 24 services — pooling | Mechanical, one line each |
| **H-5** | A repo-wide gate | Prevents reintroduction; without it the sweep is a snapshot, not a guarantee |

**H-5 matters most.** Without a gate this is a one-time cleanup that
decays. With one, the count can only go down. The gate belongs alongside
the existing nine CI policy checks.

### What would make this go wrong

- **Bounding a body that is legitimately large.** Upload and audio routes
  carry real payloads. The limits are the perception spine's, tuned for
  events, not files. Any route handling binary content needs checking
  before it is bounded, not after.
- **Sharing a pool across services with very different timeouts.** A 2s
  health probe and a 180s chat stream sharing a pool is fine; sharing a
  *client* would not be. The distinction is the reason the fix is shaped
  the way it is.
- **Doing all 96 at once.** The dashboard conversion broke a route by
  attaching a decorator to the wrong function, and it was caught only
  because a test asserted on the served HTML. Per-service, tests green
  each time.

---

## 5. Open question for the operator

The sequence above is the conservative path: ~5 steps, each verified.

**If you have a different approach in mind, this is the point to say so** —
before any of the 96 sites are touched. Specifically worth knowing:

1. **Scope** — fix all 27 services, or only the two that carry most of it
   (`agentic`, `memu-core`) and gate the rest against getting worse?
2. **Timing** — now, or after the remaining Wave 1 items (human principal
   authentication, workload identity, Tool Gate rebuild)? These are
   MEDIUM-severity hygiene defects; none is a live security hole.
3. **The gate** — should H-5 come *first*? Gating before fixing means the
   gate starts red and every service's count becomes a visible debt, which
   is honest but noisy. Gating after means a quiet gate that never goes
   red — but nothing stops a regression in between.

My recommendation, stated plainly: **H-1 and H-5 first** — make timestamps
correct, then land the gate in *reporting* mode so the debt is visible and
counted without blocking. Then H-2 and H-3, which together are ~48% of the
total. H-4 can follow whenever convenient, because by then the gate stops
it getting worse.

---

## 6. Tracking

Nothing here is tracked as a `KAI-DASH` finding; these belong to each
service's own batch. Progress is measured by `make hygiene-survey`, whose
totals should only ever fall.

**Findings formally closed by this sub-plan: 0.** Rule 7.
