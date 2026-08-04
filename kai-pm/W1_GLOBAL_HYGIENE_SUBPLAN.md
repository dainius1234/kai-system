# Sub-plan — Repository-wide HTTP and Time Hygiene

**Status:** complete. All five steps done; the debt is **0** and ratcheted.
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

| Column | At baseline | Now |
|---|---:|---:|
| Per-request HTTP clients | 96 | **0** |
| Unbounded `request.json()` | 20 | **0** |
| Naive `utcnow()` | 15 | **0** |
| 200-on-failure | 5 | **0** |
| **Total** | **136** | **0** |

**All 50 services are clear.** 146 pooled call sites and 54 bounded body
reads are now in use. The ratchet is set at zero and proven to fail on a
single reintroduced client.

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
| ~~**H-1**~~ | ~~`datetime.utcnow()` → aware~~ | **Done.** 17 sites, 7 services. `.isoformat()` now carries `+00:00`; `.strftime()` output is byte-identical. Verified safe first: memU coerces naive→aware defensively, and `calendar-sync` is self-contained and naive-consistent |
| ~~**H-2**~~ | ~~`memu-core`~~ | **Done.** All 14 recording routes bounded; `/memory/persist` and the graph proxy now answer 503 on failure. A test asserted the defect — that a persistence failure returns 200 — and was corrected |
| ~~**H-3**~~ | ~~`agentic`~~ | **Done.** Also fixed a **false positive in the survey**: `agentic`'s nested `_ping()` helper returns `{"reachable": False}` as a correct per-node result, and counting it invited someone to "fix" working code |
| ~~**H-4**~~ | ~~Remaining services~~ | **Done.** 23 services. Every one verified by *import*, not just compile — a bad import anchor is a runtime `NameError`, not a syntax error |
| ~~**H-5**~~ | ~~A repo-wide gate~~ | **Done.** Ratchet, not a threshold — see D148 |

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

## 5. What the sweep exposed

Three problems were in the **tooling and the shared module**, not in the
services being cleaned:

1. **The survey undercounted.** It scanned only `*/app.py`, missing
   `agentic/introspect_app.py` entirely — 2 naive timestamps and 3 clients
   no number here had ever included.
2. **The survey had a false positive.** It counted except-handlers in
   *nested* helpers, so `agentic`'s per-node `_ping()` — which correctly
   returns `{"reachable": False}` while the route succeeds — looked like a
   defect. Scope-aware now.
3. **`common/http_hygiene.py` imposed import-time requirements.** It
   subclassed `httpx.AsyncHTTPTransport` and imported the perception spine
   at module level, so any service adopting it failed to load under tests
   that stub `httpx` or replace `common`. **That was the root cause behind
   four broken suites** — the transport and the limits are both lazy now.

The last one is the important one. Patching each broken test would have
been four fixes for one cause, and the fifth service to adopt the module
would have broken a fifth suite.

---

## 6. Open question for the operator

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

*(Resolved: the operator chose gate-first, then `memu-core`, then
`agentic`. All five steps are now complete.)*

The original recommendation, kept for the record: **H-1 and H-5 first** — make timestamps
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

---

## H-6 — `except Exception: pass` (KAI-GATE-021)

**Status:** OPEN, ratcheted at **120**. Added 2026-08-04.
**Prompt:** *"I know it's large but we'll need to fix it… we'll manage it."*

156 handlers discard the reason repo-wide; 120 of them in service entry
points, which is what the survey scans and therefore what the ratchet
holds.

### Why this is not a style preference

`scripts/test_soul_identity.py` carried exactly this shape. It loaded
`agentic/app.py`, swallowed the failure with `except Exception: pass`, and
returned a half-built module — so four tests asserted `False` with nothing
to say why. The moment the handler was changed to **record** the exception
instead of discarding it, the message named the cause in one line:

```
AssertionError: agentic/app.py did not finish loading: No module named 'system_fsm'
```

Four failures became zero, and the fix took a minute. In a service the
same shape means it degrades quietly and the operator is told nothing.

### Classified, because 156 undifferentiated handlers is not a work item

| bucket | n | the risk |
|---|---:|---|
| **network call** | 64 | a dependency fails, the caller reports success — the *success-shaped failure* H-2/H-3 already named |
| other | 45 | needs reading; no shared shape |
| parse/convert | 17 | malformed input silently becomes a default |
| filesystem | 14 | a write that did not happen, reported as though it did |
| optional-import | 11 | the OpenCV class — a guard that is right for an *absent* package and wrong for a *partial* one |
| cleanup/teardown | 5 | defensible; a close that fails during shutdown is usually noise |

**84 of 156 sit in two files** — `memu-core/app.py` (53) and
`agentic/app.py` (31). The concentration is the good news: two careful
reads cover half of it.

### The phases

**Phase 0 — done.** The ratchet. `hygiene_survey`'s fifth column,
`silent_swallows`, baseline 120, may only fall. Proven by planting one
handler and watching the gate name the delta. Nothing can grow while the
rest is worked, which is what makes it safe to work slowly.

**Phase 1 — the 11 optional-imports.** Smallest bucket, sharpest lesson,
already has a precedent: `perception/vision/app.py` was fixed today after
CI died on a partial OpenCV. Each needs the same question — *is this guard
correct when the package is present but incomplete?*

**Phase 2 — the 64 network calls.** The highest value, and the bucket with
a shared answer: a failed call should log at warning with the exception,
and the caller should return a degraded envelope rather than a success.
`common/degraded.py` already exists for this; the work is applying it, not
inventing it.

**Phase 3 — 17 parse + 14 filesystem.** Both have a shared shape too: a
default substituted for a failure needs to be *labelled* as a default.

**Phase 4 — the 45 "other".** Read individually. No shortcut, and the
smallest bucket per unit of risk, so it goes last.

**Phase 5 — the 5 cleanup handlers.** Likely to end as a documented
exclusion rather than a change. If so it is encoded as a rule ("handlers
whose try block only closes or cancels"), never a list of line numbers.

### The rule this must not break

Per the operator's rubric, none of these may be waived by a comment. A
handler that genuinely should discard its exception says so by *catching
the named condition* — `except FileNotFoundError: pass` is a decision
about a specific thing, and the detector already ignores it. A broad
`except Exception: pass` is a decision not to look, and that is what the
count measures.

### One thing found while building the ratchet

Adding the column exposed the same defect in the survey itself.
`COLUMNS` was a hand-typed tuple beside the detectors, so `silent_swallows`
was computed and then **excluded from every total** — the survey reported
0 across a repository holding 156. The table had four hard-coded columns
while the grand total summed five, so the entire finding was invisible in
the place a reader looks.

Both are now derived from the detectors, so a sixth column cannot repeat
it. That is the third hand-maintained list to fail this way in one day,
after the dead-test detector's file list and the deprecation rule's seven
filenames. **The pattern is worth stating plainly: a list of what to check,
maintained next to the thing that does the checking, will drift — and the
drift is silent, because the list is what defines "everything".**
