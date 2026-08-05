# Sub-plan — The Instrumentation Layer

**Status:** **A-04a done**, **A-04b started** — registry + meta-check live in
reporting mode (`make gate-registry`), now reporting **29 findings** (was 33).
`check_compose_drift` is the first of the eight fully retrofitted. Register
below holds 6 findings.
**Parent:** [`W1_DASHBOARD_REMEDIATION_PLAN.md`](W1_DASHBOARD_REMEDIATION_PLAN.md)
**Question this answers:** *"What does the instrumentation architecture look
like if you sketch it the way you'd sketch `dashboard/app.py`?"*

---

## 1. Why this exists

Nine defects in this programme were not in the system. They were in the
things watching the system:

| | Defect | Found by |
|---|---|---|
| 1 | A negative test whose injected violation was never written to disk | luck |
| 2 | An architecture gate silently omitting 6 of its 15 rules | luck |
| 3 | `MANUAL` placeholders reading as a clean bill of health | luck |
| 4 | A node suite exiting 0 with no output | luck |
| 5 | Ratchet tests that stopped testing each column as it reached zero | luck |
| 6 | A survey that scanned `*/app.py` and missed `*_app.py` entirely | luck |
| 7 | A survey with a false positive, inviting a "fix" to working code | luck |
| 8 | A shared module breaking four suites at import time | luck |
| 9 | `make test-uh` — 26 suites, 1,947 assertions — running nowhere in CI | luck |

Nine for nine. The application layer has scoped routes, a degraded
envelope, a finding register and an audit trail. The layer that watches
it has none of that, and it is now load-bearing: these ratchets are the
only thing standing between 1,947 assertions and silent atrophy.

**This sub-plan is not a sixth gate.** It is the four properties the
application layer already has, applied to the watching layer, enforced by
*one* mechanism rather than by remembering.

---

## 2. What was measured

Twelve check scripts live in `scripts/security/`. Reproduce every number
below with the commands in §6.

| Property | Have it | Lack it |
|---|---:|---:|
| Reports a **denominator** — says how much it inspected | **4** | 8 |
| Has a suite **proving it can fail** | **4** | 8 |
| **Fails closed** when its input is missing | **1** | 11 |
| Declared in **one** place rather than three | **0** | 12 |

**Now (A-04 complete):** **all six invariants at zero and all six enforced.** The 12 checks that had 4/4/1/0 of the properties now have all of them. Each invariant was added to `ENFORCED` because the gate *refused to pass without it* — never because someone remembered.

The first two columns hold *exactly the same four scripts*:
`check_architecture_rules`, `check_dashboard_findings`,
`check_assertion_floors`, `hygiene_survey`. Those are precisely the four
built or repaired **after** a self-consuming guard was discovered. The
eight that predate the lesson have neither property.

That correlation is the finding. The properties were learned one incident
at a time and applied only where the incident happened.

### The sharpest case, proven rather than argued

`check_port_bindings.py` skips compose files that do not exist:

```python
for name in COMPOSE_FILES:
    path = repo_root / name
    if not path.exists():
        continue          # ← a renamed compose file is not a violation
```

Point it at filenames that do not exist and it reports:

```
exit=0  output='PASS: No disallowed port bindings found.'
```

Byte-identical to a real pass. Rename `docker-compose.full.yml` and this
gate — the one that stops a service publishing a port to the world —
passes forever, having inspected nothing.

**Three are proven, not inferred**, each by pointing the gate at inputs
that do not exist and running it in-process:

| Gate | Inputs removed | Result |
|---|---|---|
| `check_port_bindings` | all 3 compose files | `exit=0`, `PASS: No disallowed port bindings found.` |
| `hygiene_survey --gate` | all service entry points | `exit=0`, `GATE PASSED: nothing has got worse.` — 0 services scanned |
| `check_dashboard_findings` | `dashboard/app.py` and the repo root | `exit=0`, **`REMEDIATED=52`** against a tree that does not exist |

Eight more carry the same `if not path.exists(): continue` shape by
inspection, including one of the good four (`rule_12` returns clean if
`common/contracts` is missing). **Only `check_assertion_floors` fails
closed** — and only because that rule was written this afternoon, in
response to this conversation.

The middle row is my own gate, built during this programme. The bottom
row is the one the entire Wave 1 headline rests on.

**This is the same pattern under a different name.** A self-consuming
guard shrinks because the system got healthier. This one shrinks because
a file moved. Both end at "the check passes while checking nothing," and
both are invisible.

### The taxonomy has three members, not two

The operator drew a distinction worth keeping, because the three have
different mechanisms and therefore different fixes:

| Name | Mechanism | Fix |
|---|---|---|
| **Self-consuming guard** | A precondition shrinks because the operation it guards succeeded | Drive the test from synthetic state, not live state |
| **Boundary blindness** | A check cannot distinguish *the system is correct* from *the system is absent* | Fail closed on a missing input (`gate_inputs.require`) |
| **Category confusion** | A check for the **absence of something bad** passes because *everything* is absent | Assert a positive anchor first — prove the thing exists before judging it |

The third is `KAI-GATE-005`. *"The dashboard never reads broker
credentials"* is **correctly true** when there is no dashboard. The
check's logic is not wrong; it does not know it is making a far weaker
claim than it appears to. Fail-closed inputs fix the first case and not
the second, which is why they are listed apart.

---

## 3. The sketch

Four invariants. Each is a property `dashboard/app.py` already has.

### I-1. Fail closed on a missing input

`dashboard/app.py`: no `KAI_DASHBOARD_TOKEN` → **503**, never open.
The gates: no input file → `continue` → **PASS**.

A gate that cannot find what it audits must **fail**, not shrug. The
absence of the thing being audited is the strongest possible reason not
to certify it.

### I-2. Report a denominator, or decline to report

`common/degraded.py` answers `unavailable` rather than substituting a
different number wearing the same name. `check_architecture_rules`
already does the gate-shaped version:

```
cover §15 rules accounted for: 15/15 (12 enforced, 3 declared uncheckable)
```

That line exists *because* of defect 2. Every gate should have one. A
`PASS` with no denominator is unfalsifiable — it reads identically
whether it inspected fifty services or zero.

The "declared uncheckable" half matters as much as the count: it is how a
gate says *I am not measuring this* instead of quietly not measuring it.

### I-5. No inert rules

A rule that exists in syntax and has no effect. Four instances were found
while reading the gates, all with the same signature — **the code's
self-description and its behaviour had diverged**:

- `check_compose_drift`: `if net_cfg.get("internal"): pass`
- `check_restart_recovery`: `ALLOWED_RESTART` declared, never read
- `check_network_zones`: `if svc_nets is None: pass`
- `check_dashboard_findings`: a condition computed, `pass`-ed, discarded

The operator's framing is the sharp one: *an unused import can be cruft,
but a declared-but-unreferenced constant with a security-shaped name is a
claim the code makes about itself that isn't true.* The docstring says
"we allowlist"; the constant exists to prove it; the constant is wired to
nothing.

**I-5 is enforced.** It reached zero, and the gate refused to pass until
it was added to `ENFORCED` — the self-advancing ratchet firing on its own
author.

### I-3. Prove it can fail

Four gates have a suite that injects a violation and asserts the gate
fires. Eight do not, and have never been observed failing. They may be
vacuous right now; nothing would tell us.

### I-4. One declaration, not three

Adding a gate today means remembering three places — `policy-check`, a
workflow, and a can-it-fail suite — with nothing cross-checking them.
The current state of that:

- `check_dashboard_findings.py` — in neither `policy-check` nor CI (it is
  a report, not a gate; but that intent is recorded nowhere, so it is
  indistinguishable from an oversight)
- `check_assertion_floors.py` — in CI, not in `policy-check`
- The other ten — in both

This is the `PUBLIC_ROUTES` problem. The dashboard solved it by making a
route declare its own scope, so a new unauthenticated route is *visibly*
unsafe rather than invisibly so. The gates need the same: a registry each
gate must appear in, cross-checked against the filesystem and against
what CI actually runs.

---

## 4. The mechanism — one meta-check, not four

```
scripts/security/gate_registry.py     # what each gate is and claims
scripts/security/check_gate_registry.py   # the single meta-check
scripts/test_gate_registry.py         # proves the meta-check can fail
```

Each gate declares itself once:

```python
Gate(
    module="check_port_bindings",
    kind=GATE,                       # or REPORT — a report is not a gate
    invoked_by=("policy-check", "policy-checks.yml"),
    proven_by="test_policy_gates.py::test_port_bindings_fires",
    denominator="compose files inspected",
)
```

The meta-check cross-references **three independent sources**:

1. **The filesystem** — every `check_*.py` that exists
2. **The invocations** — every gate named in the `Makefile` and in
   `.github/workflows/*.yml`
3. **The registry** — every gate declared

Any disagreement fails. A new gate that nobody registered fails. A
registered gate nobody runs fails. A gate whose `proven_by` suite does
not exist fails. Then it runs each gate and asserts its denominator is
non-zero — which is I-1 and I-2 enforced in one pass, for all twelve, by
one mechanism.

### Where the regress stops

*Who watches the watcher?* It terminates, and it is worth being explicit
about why. `check_gate_registry.py` is itself a gate, so it appears in
its own registry and is bound by its own rules: it must report a
denominator (gates cross-checked), fail closed (an unreadable registry is
a failure, not a skip), and have a `proven_by` suite. The recursion is
depth-one and closed. The terminus is `test_gate_registry.py` — a suite
driven entirely by synthetic registries, which is the same discipline
`test_assertion_floors.py` already follows.

Adding a *seventh* gate to watch the sixth would be the mistake. Making
the sixth obey the rule it enforces is not.

---

## 5. Register — `KAI-GATE-##`

A **separate register**, like `KAI-DASH-D##`. These are defects in the
instrumentation, and letting them dilute or stand in for one of the 96
`KAI-DASH` findings would be worse than not counting them.

| ID | Severity | Finding | Status |
|---|---|---|---|
| `KAI-GATE-001` | HIGH | 11 of 12 gates fail **open** when their input is missing: `if not path.exists(): continue` → `PASS`. Proven on three; the rest share the shape by inspection | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-005` | HIGH | `check_dashboard_findings` reports **`REMEDIATED=52`** against a source tree that does not exist. For half the checks, *"the code is not there"* is indistinguishable from *"the code is correct"* | **CLOSED 2026-08-03** — anchor pre-scan: exit 2 absent, exit 3 unrecognisable, no verdict rendered either way · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-007` | HIGH | `check_secret_fallbacks` catches only a **denylist of nine weak words**. `${DB_PASSWORD:-hunter2}`, `${JWT_SECRET:-a8f3c9d1e7b2}` and a hardcoded `BINANCE_API_SECRET` all pass. Its docstring advertises a third scan — hardcoded secrets in environment blocks — that **has no implementing pattern** | **REMEDIATED** — rewritten as a rule: a secret may be referenced, never valued |
| `KAI-GATE-008` | MEDIUM | `check_restart_recovery` declares `ALLOWED_RESTART` and **never references it**, denying exactly one string instead, so `restart: nonsense-value` passed. Same shape as the `if ...: pass` dead branch in `check_compose_drift` | **CLOSED 2026-08-03** — the declared allowlist is the one enforced · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-010` | MEDIUM | `check_port_bindings` reported a **correctly** loopback-bound dashboard as a violation (Compose long-form `host_ip:` was unparsed), and turned a malformed `ports:` string into nine violations about ports named `'8'`, `'0'` and `':'` — the misleading-message failure mode | **REMEDIATED** |
| `KAI-GATE-011` | MEDIUM | `check_image_tags` used a **denylist of four words**, so `myimg:main` and `node:alpine` passed | **REMEDIATED** — a rule: versioned or digest. All 18 tags in use already comply |
| `KAI-GATE-012` | MEDIUM | `check_network_zones` claimed *"every service has an explicit networks assignment"* and implemented it as `if svc_nets is None: pass`. A service with no `networks:` key joins the implicit default bridge, outside every trust zone. Latent — 0 services affected today | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-021` | MEDIUM | **156 exception handlers discard the reason** (`except Exception: pass`) — 120 of them in service entry points. Classified by what the guarded operation was: **64 network calls** (the success-shaped failure H-2/H-3 named — a dependency fails and the service reports success), 45 other, 17 parse/convert, 14 filesystem, 11 optional-import (the OpenCV class: correct for an absent package, wrong for a partial one), 5 cleanup (defensible). 84 of 156 sit in two files. Proven live, not argued: `test_soul_identity` carried one, and making it *record* the exception named the cause in a single line — `No module named 'system_fsm'` — turning 4 failures into 0 | **CLOSED 2026-08-05** — , ratcheted** — `hygiene_survey`'s fifth column, baseline 120, may only fall. Phased plan in `kai-pm/W1_GLOBAL_HYGIENE_SUBPLAN.md` · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-019` | **CRITICAL** | **The repo-wide pytest ran zero tests, on every run, for at least a week.** `python-app.yml` is the only job that executes the repository's ~4,200 tests. It aborted during *collection* with six errors, so the number that ran was not reduced, it was zero — and the workflow triggers only on `main`, so the branch the work happens on could not observe it. Five of the six errors came from one line in `test_cortex.py` replacing `sys.modules["common"]`; every file it broke passed when run alone. Root cause: `sys.modules` edited process-globally by test files, in seventeen places, with nothing scoping the edit | **CLOSED 2026-08-04** — run 30939788411, all ten steps green: lint · pytest 4,246 passed 0 failed 0 errors · suite floor · isolation · dashboard. Nine further layers had to be removed before the tests could run at all (main-only trigger, flake8 ahead of the tests, a partial OpenCV, a SIGSEGV importing the ML stack, and 42 failures from one machine's paths baked into the tests). Prevention: `scripts/module_stubs.stubbed()` scopes them; `check_test_isolation.py` fails on any real module left replaced (at **zero and enforced**); `added`/`env_set` ratchet from a declared baseline; `python-app.yml` now runs on `claude/**` too |
| `KAI-GATE-020` | HIGH | **Order-dependent failures, invisible until the suite could run.** With collection fixed, 63 failures and 11 errors surfaced — all but 15 of which passed when their file ran alone. Four mechanisms, none of them `sys.modules`: `asyncio.get_event_loop()` reusing a loop an earlier TestClient had closed (44); endpoints failing closed without a service token, asserted from before G-03 (12); a `import memu_core_app` naming a module that has never existed (5); greps that could not tell a route being *hardened* from being removed (2); and module-identity collisions on the generic name `app` (13) | **CLOSED** — 4,208 passed, 0 failed, 0 errors. Prevention: `check_suite_floor.py` ratchets failures and errors down and the pass count up, so deleting a test is not a way to go green |
| `KAI-GATE-017` | HIGH | **"Pre-existing" was doing the work of an investigation.** Of six failing suites I had repeatedly described as "needs a running stack", only three did. `test_executor_service` runs against `TestClient` in-process and was asserting the behaviour from *before* G-03 made `/execute` fail closed. Two more crashed on `pytest.skip` outside pytest — and under pytest would have counted as **green while verifying nothing** | **REMEDIATED** |
| `KAI-GATE-016` | **CRITICAL** | **Three workflow files do not parse as YAML** — including `core-tests.yml`, the main CI. An embedded `python3 -c "` block begins at column 0, terminating the enclosing `run: |` scalar. A workflow that does not parse runs nothing, and running nothing is indistinguishable from having no failures. **Needs confirmation in the Actions tab: if GitHub rejected these, the core suite has not been running** | **REMEDIATED** — all three parse; content byte-identical |
| `KAI-GATE-015` | MEDIUM | 7 CI steps could pass without doing their job, announced only by `\|\| echo "::warning::"`. `pip-audit` also had `2>/dev/null`, hiding that `--strict` exits 1 for an unauditable *system* package — so the warning named the wrong cause while real CVEs went unreported | **REMEDIATED** — declared, owned, dated, and each prints an explicit SKIPPED |
| `KAI-GATE-014` | HIGH | **15 dashboard checks read "the route is gone" as "the defect is fixed"** — `if not src: return REMEDIATED, "upload route removed"`. Correct if the route was removed deliberately, wrong if it was renamed, and the check cannot tell. The tracker disagreed with itself: 10 branches called the same situation MANUAL. Found by mutation — blinding each of 17 subject handlers showed 8 where no check reacted at all | **REMEDIATED** — all 32 branches answer MANUAL |
| `KAI-GATE-009` | HIGH | `CAMERA_GATE_TOKEN` defaulted to the literal `camera-gate-token-1` in **both** `docker-compose.full.yml` and `perception/camera/app.py:225`, so that string *was* the camera's tool-gate session ID in any deployment where nobody set the variable | **CLOSED 2026-08-03** — no default in either place; the camera refuses to speak unprompted without an identity · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-002` | MEDIUM | 8 of 12 gates print `PASS` with **no denominator**. A gate that inspected nothing is indistinguishable from one that inspected everything | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-003` | MEDIUM | 8 of 12 gates have **never been observed failing**. No suite injects a violation and asserts they fire | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-004` | MEDIUM | A gate is declared in up to three places with **nothing cross-checking them**. Two of twelve are already inconsistent | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |
| `KAI-GATE-006` | HIGH | **9 of 21 `sovereign` services had no `restart` and no `security_opt`** — including Vault, the rotator, Postgres and Redis. The profile named for being hardened was the least guarded, because the drift check only ever compared `full` against `minimal` | **CLOSED 2026-08-03** · prevention re-verified on every run by `closure_register.still_holds` |

**Findings formally closed: 0.** Programme Rule 7.

---

## 6. Reproducing every number above

```bash
# 12 check scripts
ls scripts/security/*.py | grep -v __init__

# which are invoked, and from where
sed -n '/^policy-check:/,/^$/p' Makefile | grep -o 'scripts/security/[a-z_]*\.py'
grep -ho 'scripts/security/[a-z_]*\.py' .github/workflows/*.yml | sort -u

# which fail open on a missing input
grep -n "exists()" scripts/security/check_*.py

# which have a suite that proves they can fail
for f in scripts/security/check_*.py; do
  grep -l "$(basename "$f" .py)" scripts/test_*.py; done

# the proof, in-process
python3 -c "
import importlib, io, contextlib, sys; sys.path.insert(0,'.')
m = importlib.import_module('scripts.security.check_port_bindings')
m.COMPOSE_FILES = ['docker-compose.RENAMED.yml']
b = io.StringIO()
with contextlib.redirect_stdout(b): rc = m.main()
print(rc, repr(b.getvalue().strip()))"
```

---

## 7. Proposed sequence

Ordered lowest-risk first. Each step is independently revertible.

| Step | Scope | Why this order |
|---|---|---|
| ~~**A-04a**~~ | ~~Registry + meta-check in reporting mode~~ | **Done.** `make gate-registry` reports 29 findings and exits 0 by design. 30 assertions in `test_gate_registry.py`, all from synthetic registries |
| **A-04b** | Fail-closed across all 12 | **Started.** `gate_inputs.require()` is the shared helper; `check_compose_drift` is the first adopter and now has all four invariants. 7 compose gates remain — **semantics first**, per the operator: a denominator on a broken check is architectural lipstick. The first one read (`check_secret_fallbacks`) had two semantic defects — see `KAI-GATE-007` |
| **A-04b** | I-1 fail-closed across all 12 | Mechanical, and the highest-severity finding. One shared helper, not 12 edits |
| **A-04c** | I-2 denominators across all 12 | Follows I-1 naturally — the count is what the fail-closed check already computes |
| **A-04d** | I-3 can-it-fail suites for the 8 | The largest piece. 8 suites, each injecting one real violation |
| ~~**A-04e**~~ | ~~Flip the meta-check to gate~~ | **Done, partially and by design.** I-4 is at zero and **enforced** in `policy-check` and CI; I-1/I-2/I-3 are reported while the retrofit proceeds. A ratchet does not need to cover the whole surface to be useful — it needs to never go backwards on the surface it covers. **And it advances itself:** an invariant that reaches zero without being added to `ENFORCED` *fails* the gate, because zero that nothing enforces will not stay zero |

### What would make this go wrong

- **Fail-closed on a genuinely optional input.** `docker-compose.sovereign.yml`
  may legitimately be absent in some checkouts. The fix is a declared
  `optional=True` per input, not a blanket `continue` — the declaration is
  the point.
- **Denominators that count the wrong thing.** "12 files inspected" is
  useless if the interesting unit is services. Each gate declares its own
  unit; a wrong unit is at least *visible*.
- **Doing all 8 suites in one commit.** Defect 8 came from exactly that
  shape. Per-gate, green each time.

---

## 8. The limit of all of this

This catches gates that inspect **nothing**. It does not catch gates that
inspect everything and **assert nothing meaningful** — `check("x", True)`
counts the same as a real assertion, and reports a healthy denominator
while doing so.

That is a different detector: mutation testing. The operator's framing is
the right one — an **audit tool, not a CI gate**, because a slow or flaky
gate becomes an ignored gate, which is defect 9 all over again. Their
*sentinel mutation* proposal is the tractable version: a fixed set of
~20 critical invariants, one operator flipped in each, fail if the suite
still passes. Fixed, predictable cost; no whole-repo mutation run.

Recorded here as **A-03**, deliberately unscheduled, and deliberately not
claimed by A-02. The assertion ratchet catches shrinkage, not vacuity.
Two different defects, two different detectors.

---

## 10. What extending the drift check found

The drift gate compared `full` against `minimal` and nothing else. A third
profile was added later and the comparison was never revisited, so
`docker-compose.sovereign.yml` was the only profile never checked.

**9 of its 21 services carried neither `restart` nor `security_opt`** —
Vault, `vault-rotator`, Postgres, Redis, Tailscale, Prometheus, Grafana,
Alertmanager and `perception-telegram`. The profile named for being
hardened was, by this measure, the least guarded of the three. Now fixed,
using the exact form `full` already uses for Postgres and Redis — a
pattern CI boots, rather than one invented here.

### Equality would have been the wrong test

A naive extension reports all six shared-service differences, and two of
them are sovereign being **stricter** (`runtime: gvisor`,
`apparmor:executor-aa`, `read_only`). The cheapest way to make that green
is to weaken sovereign. **A gate that pushes toward less security is
worse than no gate.**

So drift is directional — the same ratchet shape as everywhere else here.
Stricter is allowed *and recorded*, weaker fails, absent fails in every
direction. `restart` is presence-required but value-free, because
`on-failure` versus `unless-stopped` is a containment-versus-availability
choice a profile is entitled to make.

### Two defects that had to stay apart

Sovereign's own anchor is much stricter than the baseline's
(`cap_drop: [ALL]`, `read_only`, `user`, `tmpfs`). Folding "bypasses its
own anchor" together with "below the baseline floor" would have demanded
`cap_drop: ALL` on Postgres — which needs SETUID/SETGID to drop from root
at startup. The gate would have been pushing a change that breaks the
profile it protects. The 9 anchor bypasses are reported in their own
category and left for per-service capability analysis.

### Two false starts, both caught before landing

1. The first network rule flagged `minimal` for having no `execution-net`
   — but minimal runs no executor, so the absence is correct. Flagging it
   would have invited someone to declare a network nothing attaches to:
   defect 7's shape. The rule now compares only networks declared in both
   places, and separately requires that a network a service *attaches to*
   is declared.
2. The first remediation script matched service names too loosely and
   injected `restart:` into a `depends_on:` mapping, breaking the file.
   Restored from git and redone with an exact two-space indent match,
   plus assertions that no pre-existing key changed.

---

## 9. What A-04a found in itself

The first run of the meta-check **spawned itself by subprocess and
recursed until the process tree had to be killed.** The registry lists
every check including `check_gate_registry`, and the denominator probe
runs each listed check.

Depth-one was a property of the *design*, argued in §4 of this document
and in the operator's fixed-point criterion. Nothing in the code enforced
it — which is the exact class of defect this file exists to find, found
inside the fix for it, on the first execution.

The terminus is explicit now: `probe_denominator` refuses to spawn itself
and returns a `self` status, and `test_the_meta_check_never_probes_itself`
traps `subprocess.run` to prove no child is ever launched. The meta-check's
own denominator is asserted from *outside*, by driving `main()` in-process
and matching its real output against the pattern the registry declares.

**A design property that the code does not enforce is not a property.**
That belongs alongside boundary blindness and the self-consuming guard.

---

## 11. Closure register — the first six

Rule 7 held for the whole programme: nothing closed. That was right while
the work was in flight, but a register where nothing ever closes stops
carrying information.

The operator set the bar, and it is higher than "we fixed it":

> a confirmation that the remediation actually addressed the finding,
> **and that the finding's category of defect has a structural prevention
> in place** so it won't recur.

**Closed (9):** `KAI-GATE-001`–`006`, `008`, `009`, `012`.

`001`, `002` and `003` were **declined in the first batch** — 15, 6 and 2
sites remained, and a prevention covering most sites is not a prevention.
All three reached zero and are enforced now, so they close on the
criterion's own terms rather than by relaxing it.

**Deliberately excluded:** `010` (misleading messages) and `011` (the
image-tag denylist). Both are fixed and tested. Nothing structurally
prevents the *next* misleading message, so they are **remediated, not
prevented** — which is the distinction the whole register turns on.

### Closure is falsifiable, not asserted

Each record in `scripts/security/closure_register.py` carries a
`still_holds` predicate, re-evaluated on every run as **I-6**. A closure
whose prevention is removed — I-5 dropped from `ENFORCED`, a gate lifted
out of `policy-check` — **re-opens itself and fails the gate.**

Proven: removing I-5 from `ENFORCED` re-opens `KAI-GATE-008` and `012`
by name, and both close again when it is restored.

That matters because "closed" is precisely the kind of label that decays
into a rubber stamp. Here it is a claim the system re-checks.

---

**Findings closed by this sub-plan: 6, each with its prevention
re-verified on every run.** Rule 7 satisfied, not bypassed.
