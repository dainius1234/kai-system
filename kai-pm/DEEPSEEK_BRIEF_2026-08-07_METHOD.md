# Brief for DeepSeek — the method, not a bug

Sent 2026-08-07. Kept because the answer is worth reading against the
question that produced it.

## Context

Self-hosted multi-service AI system, ~49 Docker services, 30+ static
gates in CI, an append-only decision log, and operating rules in
`CLAUDE.md`. Over two weeks ~17 defects were fixed and they all had one
shape: **a check whose scope was smaller than its name implied.** The
remedy was always: state the denominator, derive it from the tree rather
than from a hand-written list beside it.

## The pattern I want an outside view on

Every instrument built to *see* failures has carried the same defect as
the system it watched. Four instances, in order of discovery:

1. The gates themselves had scopes smaller than their names — the
   systemic finding, inside the thing looking for it.
2. `check_gate_registry` resolved 3 of 8 modules to a directory where
   the file did not exist. Every AST check silently returned "no
   findings" and the gate passed.
3. The CI post-mortem printed 13 sections in fixed order whether or not
   they had content, with the image build **first**. When the build
   failed, 12 sections said "(did not run)". The GitHub Actions log API
   serves a **fixed byte window from the end** (~15,780 chars,
   measured). The empty sections evicted the only section with content.
   That run's cause is unrecoverable.
4. A decision record said "tune the healthcheck start-period once CI
   reports a real number" — and no instrument existed to report it.

## My hypothesis — please attack this, don't just agree

The mechanism is structural, not a discipline problem.

Diagnostics are **by construction the least-executed code in any
system**: a post-mortem runs only on failure; a gate's failing branch
only when something is wrong; an error message's text only in the case
nobody tested.

Meanwhile the dominant finding of this whole programme is that every
defect lived in code that had **never executed** — not one was code that
used to work and broke.

So the observability layer is composed almost entirely of the exact
category where all the defects live, *and* it is the layer you depend on
precisely when you can least afford it wrong.

If that is right, "write better diagnostics" is not a plan, because
willpower does not execute code. The plan must make the failure path
stop being never-executed:

* inject a failure on a schedule; assert the post-mortem produced a
  **readable** answer, not merely that it ran
* the assertion should be a **property of the output** — e.g. "under 20
  lines when only the build failed" — not "did not crash"
* any "we will tune this from a measured number" must ship the
  instrument producing that number **in the same commit**

`scripts/test_post_mortem.py` now does the second one, calibrated
against the failed run's actual shape (1 section with output, 12
without).

## Questions

**Q1.** Is the hypothesis right, and is it known art? *"Diagnostics are
the least-executed code and therefore inherit the defect class of
never-executed code."* If there is existing literature or a named
principle, I would rather stand on it than reinvent it. If it is wrong,
where?

**Q2.** How do you **test a diagnostic for legibility** rather than
function? "It printed something" is trivially satisfiable. "Under 20
lines" is a proxy chosen because it matched one specific failure. What
are better output properties to assert? Is there a way to assert "the
cause is present in the output" without hard-coding the cause?

**Q3.** Scheduled failure injection aimed at the **observability layer**
— known practice, with a name? Chaos engineering targets the system; I
want to target the instruments watching it. What should the injection
set be, and how do you stop the injections themselves rotting into
never-executed code?

**Q4. The rules problem.** Eight rules in `CLAUDE.md`. Four caught me
within an hour of being written, which reads as success and might not
be. The worry: rules accumulate, and a wall of rules has the same
failure mode as a warning printed on every run — printing forever is
what teaches everyone to ignore it.

Proposed ratchet:

1. If it can be a gate, it belongs in code, not prose.
2. Keep it only if it is a recognisable **tell** — a signal in flight —
   rather than good advice. R1's value is not "be careful"; it is five
   specific words that mean stop: *would have, cannot, is simply, should
   work, it's just a*.
3. A rule that has not fired in a month is either working or dead, **and
   those look identical.**

(3) is the one I cannot solve. It is the same problem as a ratchet gate
reporting zero because the detector broke — solved for gates by
requiring calibration against known-answer input. What is the equivalent
for a rule whose enforcement mechanism is a human or an agent noticing
something?

**Q5.** Which of these are mechanically enforceable and which are
irreducibly judgment? Honest answer wanted, including "fewer than you
think".

| | rule | status |
|---|---|---|
| R1 | don't assert what you haven't run | ? |
| R2 | always *run* the contingency, never merely write one | ? |
| R3 | `&&` never `;` in a chain containing a gate | ? |
| R4 | measure the population before fixing it | ? |
| R5 | state the denominator, derive it from the tree | already a gate (I-2) |
| R6 | fix the class, not the instance | ? |
| R7 | findings stay open until formal closure review | already enforced |
| R8 | never-executed code is where the defects are | ? |

**Q6.** Is there a rule we are **missing** that the four instances above
imply and I have not extracted? I am inside this and probably cannot see
it.
