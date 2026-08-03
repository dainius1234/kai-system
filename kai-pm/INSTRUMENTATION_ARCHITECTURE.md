# Sub-plan — The Instrumentation Layer

**Status:** measured, not yet built. Register below holds 5 findings.
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
| `KAI-GATE-001` | HIGH | 11 of 12 gates fail **open** when their input is missing: `if not path.exists(): continue` → `PASS`. Proven on three; the rest share the shape by inspection | **OPEN** |
| `KAI-GATE-005` | HIGH | `check_dashboard_findings` reports **`REMEDIATED=52`** against a source tree that does not exist. For half the checks, *"the code is not there"* is indistinguishable from *"the code is correct"* | **OPEN** |
| `KAI-GATE-002` | MEDIUM | 8 of 12 gates print `PASS` with **no denominator**. A gate that inspected nothing is indistinguishable from one that inspected everything | **OPEN** |
| `KAI-GATE-003` | MEDIUM | 8 of 12 gates have **never been observed failing**. No suite injects a violation and asserts they fire | **OPEN** |
| `KAI-GATE-004` | MEDIUM | A gate is declared in up to three places with **nothing cross-checking them**. Two of twelve are already inconsistent | **OPEN** |

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
| **A-04a** | Registry + meta-check in **reporting** mode | Makes the 4 findings visible and counted without blocking. Same shape as H-5 |
| **A-04b** | I-1 fail-closed across all 12 | Mechanical, and the highest-severity finding. One shared helper, not 12 edits |
| **A-04c** | I-2 denominators across all 12 | Follows I-1 naturally — the count is what the fail-closed check already computes |
| **A-04d** | I-3 can-it-fail suites for the 8 | The largest piece. 8 suites, each injecting one real violation |
| **A-04e** | Flip the meta-check to **gate** | Only once the four above are clean, so it never starts red-and-ignored |

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

**Findings formally closed by this sub-plan: 0.** Rule 7.
