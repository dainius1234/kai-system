"""Findings formally closed, and the conditions that keep them closed.

Programme Rule 7: *finding counts remain unchanged until formal closure
review; closure is a separate evidence-backed register action.* For the
length of this programme nothing has closed, which was correct while the
work was in flight — but a register where nothing ever closes stops
carrying information.

The operator set the bar for closure, and it is higher than "we fixed
it":

> The review is a confirmation that the remediation actually addressed
> the finding, **and that the finding's category of defect has a
> structural prevention in place** (a gate, a ratchet, a changed
> convention) so it won't recur.

That second half is why **remediated is not closed**. `KAI-GATE-010`
(misleading messages) and `011` (the image-tag denylist) are fixed and
tested, and are deliberately *not* here: nothing structurally prevents
the next misleading message.

**Closure is encoded, not asserted.** Each record names a `prevention`
and a `still_holds` predicate that must be true right now. The meta-check
evaluates them on every run, so a closure whose prevention was quietly
removed — I-5 dropped from `ENFORCED`, a gate lifted out of
`policy-check` — **re-opens itself and fails the gate**.

That matters because "closed" is exactly the kind of label that decays
into a rubber stamp. A closed finding here is a claim the system
re-checks, not a note someone once wrote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass(frozen=True)
class Closure:
    finding: str
    defect: str          # what was actually wrong
    fix: str             # what changed
    prevention: str      # what stops it recurring
    proven_by: str       # the test that proves the prevention fires
    verified_on: str     # ISO date of the verifying run
    still_holds: Callable[[], bool]   # re-evaluated every run


def _enforced(invariant: str) -> Callable[[], bool]:
    def predicate() -> bool:
        from scripts.security.check_gate_registry import ENFORCED
        return invariant in ENFORCED
    return predicate


def _in_policy_check(module: str) -> Callable[[], bool]:
    def predicate() -> bool:
        from scripts.security.check_gate_registry import discover_policy_check
        return module in discover_policy_check()
    return predicate


def _tracker_anchors() -> bool:
    """The dashboard tracker refuses to judge a tree it cannot recognise."""
    from scripts.security import check_dashboard_findings as tracker
    return (hasattr(tracker, "anchor_scan")
            and bool(getattr(tracker, "ANCHOR_SYMBOLS", ())))


CLOSED: Tuple[Closure, ...] = (
    Closure(
        finding="KAI-GATE-004",
        defect="A check was declared in up to three places — the "
               "`policy-check` target, a CI workflow, and a suite proving "
               "it can fail — with nothing cross-checking them. Two of "
               "twelve were already inconsistent.",
        fix="`gate_registry.py` declares every check once; "
            "`check_gate_registry.py` cross-references three independent "
            "sources (filesystem, invocations, registry) and fails on any "
            "disagreement.",
        prevention="I-4 is at zero and **enforced** in `policy-check` and "
                   "`policy-checks.yml`. A new check that nobody registers "
                   "fails the build, so it is visibly unregistered rather "
                   "than quietly unwatched.",
        proven_by="scripts/test_gate_registry.py — unregistered, phantom, "
                  "Makefile mismatch, workflow mismatch, REPORT wired as a "
                  "gate, gate nothing invokes",
        verified_on="2026-08-03",
        still_holds=_enforced("I-4"),
    ),
    Closure(
        finding="KAI-GATE-005",
        defect="`check_dashboard_findings` reported REMEDIATED=52 against "
               "a source tree that did not exist. Category confusion: a "
               "check for the absence of something bad passes when "
               "everything is absent.",
        fix="An anchor pre-scan runs before any verdict: 5 verified "
            "symbols and a minimum route count. Exit 2 when the tree is "
            "absent, exit 3 when it is unrecognisable, and no finding "
            "verdict is rendered in either case.",
        prevention="No verdict can be produced without first proving the "
                   "tree is real. The refusal is the default path, not an "
                   "extra check that could be skipped.",
        proven_by="scripts/test_dashboard_findings.py — absent tree, "
                  "unrecognisable tree, symbol-complete but route-less "
                  "tree, and every anchor symbol proven present",
        verified_on="2026-08-03",
        still_holds=_tracker_anchors,
    ),
    Closure(
        finding="KAI-GATE-006",
        defect="9 of 21 `sovereign` services carried neither `restart` nor "
               "`security_opt` — including Vault, the rotator, Postgres "
               "and Redis. The profile named for being hardened was the "
               "least guarded, because the drift check only ever compared "
               "`full` against `minimal`.",
        fix="All 9 given the treatment `full` already proves in CI. The "
            "drift check now covers all three profiles, directionally: "
            "stricter is recorded, weaker fails, absent fails.",
        prevention="`check_compose_drift` runs in `policy-check` and "
                   "requires every service in every profile to be guarded. "
                   "A tenth unguarded service fails the build.",
        proven_by="scripts/test_compose_drift.py — absent setting, weaker "
                  "profile, unshared service, and the stricter-is-allowed "
                  "case that stops the gate pushing toward weakening",
        verified_on="2026-08-03",
        still_holds=_in_policy_check("check_compose_drift"),
    ),
    Closure(
        finding="KAI-GATE-008",
        defect="`check_restart_recovery` declared `ALLOWED_RESTART` and "
               "never referenced it, denying exactly one string instead. "
               "The docstring promised an allowlist; `restart: "
               "nonsense-value` passed.",
        fix="The declared allowlist is the one enforced.",
        prevention="I-5 (no inert rules) is at zero and **enforced**. Any "
                   "future policy-shaped constant that nothing reads, or "
                   "conditional whose body is `pass`, fails the build.",
        proven_by="scripts/test_secret_gates.py (the allowlist is wired, "
                  "invalid values rejected) and scripts/test_gate_registry.py "
                  "(the inert-rule detector itself)",
        verified_on="2026-08-03",
        still_holds=_enforced("I-5"),
    ),
    Closure(
        finding="KAI-GATE-009",
        defect="`CAMERA_GATE_TOKEN` defaulted to the literal "
               "`camera-gate-token-1` in both `docker-compose.full.yml` "
               "and `perception/camera/app.py`. With no env var set "
               "anywhere, that string *was* the camera's tool-gate session "
               "ID — in a file anyone can read.",
        fix="No default in either place, and `_gate_allows_speak` refuses "
            "explicitly without an identity rather than failing at the far "
            "end.",
        prevention="`check_secret_fallbacks` runs in `policy-check` as a "
                   "rule rather than a word list: a secret may be "
                   "referenced or explicitly empty, never valued. It "
                   "inspects interpolated variable names, which is how "
                   "this one was found under a non-secret key.",
        proven_by="scripts/test_secret_gates.py — weak default, strong "
                  "default, hardcoded literal, and a secret hiding under a "
                  "non-secret key",
        verified_on="2026-08-03",
        still_holds=_in_policy_check("check_secret_fallbacks"),
    ),
    Closure(
        finding="KAI-GATE-012",
        defect="`check_network_zones` claimed 'every service has an "
               "explicit networks assignment' and implemented it as "
               "`if svc_nets is None: pass`. Such a service joins "
               "Compose's implicit default bridge, outside every trust "
               "zone. Latent — 0 services affected at the time.",
        fix="The branch reports the violation and names the consequence.",
        prevention="I-5 (no inert rules) is at zero and **enforced**, so a "
                   "rule that exists in syntax and does nothing fails the "
                   "build. This is the same prevention as KAI-GATE-008 "
                   "because it is the same defect.",
        proven_by="scripts/test_compose_gates.py — a service with no "
                  "networks assignment is caught and the consequence named",
        verified_on="2026-08-03",
        still_holds=_enforced("I-5"),
    ),
)

BY_FINDING = {c.finding: c for c in CLOSED}


def lapsed() -> list:
    """Closures whose prevention no longer holds. These re-open."""
    out = []
    for closure in CLOSED:
        try:
            holds = closure.still_holds()
        except Exception as exc:                     # noqa: BLE001
            holds, exc_note = False, f" ({type(exc).__name__})"
        else:
            exc_note = ""
        if not holds:
            out.append(f"{closure.finding}: prevention no longer holds"
                       f"{exc_note} — {closure.prevention.splitlines()[0]}")
    return out
