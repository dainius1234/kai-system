# How I Write Tests Wrong — a review of my own defects

**Status:** review complete. Two rules mechanically enforced, five written.
**Prompted by:** *"review your code writing and logic for test as we lost
mostly time fixing tests and how they were getting wrong readings and
giving out false results."*

That is a fair charge. This is the count, the classes, and what is now
enforced rather than remembered.

---

## 1. The count

**Sixteen defects in my own test and detector code**, in one programme.
Every one produced a *confident wrong reading* rather than an error.

| | Defect | Wrong reading it produced |
|---|---|---|
| 1 | AST detector: "any `test_` never called is dead" | **1,555** phantom dead tests |
| 2 | Same, excluding files with `unittest.main()` | **1,813** phantom dead tests |
| 3 | Same, reading the file's `run()` — matched a *helper* named `run` | **54** phantom dead tests |
| 4 | Boundary-blindness detector counted every `return` as a skip | **15** sites, of which **4 were correct refusals** |
| 5 | `test_the_ast_detector_finds_the_real_shape` pinned `lines == [70]` | broke when an edit moved the line |
| 6 | Rewritten to "exactly one fail-open site in `check_port_bindings`" | **broke when that site was fixed** |
| 7 | `test_removing_a_prevention_reopens_its_findings` pinned `len(out) == 2` | broke when three more findings closed |
| 8 | `check("no verdict rendered", "REMEDIATED" not in out)` | matched its own refusal message's prose |
| 9 | Proof-file check split `proven_by` on `" "` and took `[0]` | a filename with a trailing comma, existing nowhere |
| 10 | Anchor symbols guessed (`require_principal`, `PUBLIC_ROUTES`) | neither exists; refused against a healthy tree |
| 11–12 | Fixture dicts missing `"inert"`, then `"lapsed"` | `KeyError` when the structure gained a key |
| 13 | `EXPECTED_SCENARIOS` off by one, four separate times | self-inflicted friction on every addition |
| 14 | Three tests added to `test_architecture_rules` and never called | silently absent; count unchanged |
| 15 | `Violation(0, rel, "...")` — wrong arity | `TypeError` at runtime |
| 16 | CI suppression pattern included `if …; then` | matched every shell conditional in the repo |

---

## 2. The classes, and which are mechanically preventable

### A — A proxy stood in for the mechanism *(1, 2, 3, 16 — the worst)*

Four detectors read **file contents** to answer a question decided by
something else entirely: *how the file is invoked*. `python -m pytest x.py`
collects every test in it; `python x.py` runs only what the file calls.
That fact lives in the **Makefile**, not in the file, and three successive
attempts guessed at it from the inside and were confidently wrong by
1,555, 1,813 and 54.

**The rule: find the mechanism that actually decides the answer, and read
*that*.** If the deciding fact lives somewhere else, go there.

**Enforced:** `check_test_wiring.py` reads the invocation, and
**calibrates against five known-good suites before reporting.** A
detector that cannot reproduce a known answer refuses instead of
reporting an unknown one. All three earlier versions would have been
stopped by that gate.

### B — The test was guarded on state the repository owns *(5, 6, 7)*

`== [70]`, `"exactly one fail-open site"`, `len(out) == 2`. Each pinned a
number the repository is free to change. **Case 6 is the sharpest defect
in this whole review: the test broke because the bug was *fixed*.** A
test that requires its own defect to persist is a self-consuming guard in
its most literal form.

**The rule: assert the property the code guarantees, never the count the
repository happens to hold today.** Where a count is genuinely the
subject, derive it (`{c for c in CLOSED if "I-5" in c.prevention}`) rather
than typing a literal.

### C — The assertion matched its own output *(8)*

`"REMEDIATED" not in out` failed because the refusal message *explains*
what a REMEDIATED-against-nothing would mean. Identical to `dash_015`,
whose grep once matched its own docstring.

**The rule: assert on structured output — a verdict line, an exit code, a
parsed field — not on a substring of prose that may discuss the thing it
is testing for.**

### D — Fixtures hand-mirrored a structure that grew *(11, 12)*

A literal dict copying `problems`' keys. When the real structure gained
`inert`, then `lapsed`, the fixture didn't.

**The rule: derive fixtures from the source of truth.** A fixture that
must be edited whenever production changes is a second definition.

### E — Tests that never ran *(14)*, and hand-maintained counters *(13)*

Three added and never dispatched. `EXPECTED_SCENARIOS` needing a manual
bump four times.

**Enforced:** `check_test_wiring.py` fails on any test defined and never
called in a self-run suite. It immediately found **7 more in
`test_dashboard_findings.py` — 16 assertions running nowhere, all
passing**, so nothing ever drew attention to them.

### A′ — The proxy was the *right kind* of thing, at the wrong *time* — added 2026-08-04

The first detector for cross-file leakage imported each test file in a
subprocess and asked what it left behind in `sys.modules`. That is not
class A: it reads the real mechanism, not a proxy for it. It still gave
a wrong answer, because it read the mechanism **at the wrong moment**.

`test_cognitive_mechanisms.py` replaces `fastapi` — an installed,
working library — with a two-attribute stub, from `setup_method`, once
per test. At *import* it leaves nothing behind. The probe reported it
clean. It was the single worst offender in the repository, responsible
for 223 errors in other files.

**The rule: when the question is "what does this do when it runs", the
detector has to watch it run.** `isolation_plugin.py` hooks
`pytest_runtest_protocol` and diffs global state across file boundaries
in the real session.

**Enforced:** `check_test_isolation.py`, in `python-app.yml`, calibrated
against a synthetic leaky/clean pair before being pointed at the repo.

Two things this cost, worth recording because both were avoidable:

- The leaks were **chained**. Fixing the first file that replaced
  `common.runtime` made a second one appear — until then it had been
  replacing something already replaced. Six iterations to reach zero. A
  single measurement would have said "one offender" and been wrong every
  time. *Re-measure after every fix; a detector's first number is a
  lower bound when the defects mask each other.*
- I broke two suites by scoping stubs that genuinely needed to outlive
  the import (`test_agentic_routes`: 3 failures became 56). Both were
  caught because I baselined each file **before** changing it. That
  habit is the only reason the numbers in the commit message can be
  trusted, and it is cheap: `git stash`, run, `git stash pop`.

### F — Ordinary coding errors *(9, 10, 15)*

Wrong arity, bad parsing, guessed symbols. Not a pattern; just wrong.
Worth noting only because **each was caught by a test rather than by
review**, which is the system working.

---

## 3. What changed

**Mechanically enforced:**

- **`check_test_wiring.py`** — a test defined and never called fails the
  build. In `policy-check` and `policy-checks.yml`. Calibrated, and
  refuses rather than reports when calibration fails.
- **A Makefile target naming a script that does not exist fails.** I-1
  caught that in this very file on its first run.

**Written rules (A–D above), and one measurement worth keeping:**

Of the 16, **10 were caught by a test or a gate, and 6 by reading.** The
harness works. The recurring failure is not that defects get through — it
is that I trust the first number a new detector produces. Three times in
this review a detector was believed for one command before being checked.

**Calibration is the answer to that, and it generalises beyond this
file:** a new detector should be pointed at a case whose answer is
already known before it is pointed at the repository.

---

## 4. What is deliberately not enforced

`EXPECTED_SCENARIOS` stays an equality, not a floor, despite four
off-by-one corrections. It is friction on purpose: the operator asked for
a meta-assertion that notices when the test surface shrinks, and a floor
would notice removals but not a scenario that is added and never
dispatched. The friction is the feature; the four corrections were the
cost of it working.

Classes B, C and D are conventions, not gates. Detecting "this assertion
is pinned to repository state" mechanically would need to distinguish a
derived count from a typed one, and a detector that guesses at that would
be defect class A all over again.
