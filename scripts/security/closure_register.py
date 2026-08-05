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


def _swallows_ratcheted() -> bool:
    """The silent-swallow count is baselined and the survey still gates.

    Both halves matter. A baseline nothing runs is a number in a file,
    and a survey with no ceiling is a report. The closure depends on the
    pair, so the predicate checks the pair.
    """
    import json
    from pathlib import Path as _Path
    from scripts.security.check_gate_registry import discover_policy_check
    if "hygiene_survey" not in discover_policy_check():
        return False
    baseline = _Path(__file__).resolve().parent / "hygiene_baseline.json"
    try:
        totals = json.loads(baseline.read_text(encoding="utf-8"))["totals"]
    except (OSError, KeyError, ValueError):
        return False
    return "silent_swallows" in totals


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

# ── Second batch: the three declined in the first round ──────────────
#
# `001`, `002` and `003` were offered for closure and declined, because
# 15, 6 and 2 sites remained and a prevention covering most sites is not
# a prevention. All three reached zero and are enforced now, so the
# criterion is met on its own terms rather than by relaxing it.

CLOSED = CLOSED + (
    Closure(
        finding="KAI-GATE-001",
        defect="11 of 12 checks answered `continue` when an input was "
               "missing, so a renamed file was indistinguishable from a "
               "clean bill of health. Proven on `check_port_bindings`: "
               "pointed at filenames that did not exist it printed PASS "
               "and exited 0.",
        fix="`gate_inputs.require()` refuses and exits 1, naming the "
            "missing input. Adopted by every compose gate; the four "
            "architecture rules that skip a missing directory now report "
            "it as a violation instead.",
        prevention="I-1 is at zero and **enforced**. The AST detector "
                   "distinguishes a skip from a refusal, so a new "
                   "`if not X.exists(): continue` fails the build while a "
                   "`return <failure>` correctly does not.",
        proven_by="scripts/test_compose_gates.py, scripts/test_secret_gates.py, "
                  "scripts/test_architecture_rules.py, scripts/test_gate_registry.py",
        verified_on="2026-08-03",
        still_holds=_enforced("I-1"),
    ),
    Closure(
        finding="KAI-GATE-002",
        defect="8 of 12 checks printed PASS with no statement of how much "
               "they inspected — unfalsifiable, reading identically "
               "whether they examined fifty services or zero.",
        fix="`gate_inputs.inspected()` on every gate. The unit is service "
            "definitions, not compose files: '3 files' is 3 whatever "
            "happens. `check_architecture_rules` gained a *second* "
            "denominator in python files, because its existing "
            "`15/15 rules` counted the wrong dimension and could not "
            "reveal a scanner blind to half its inputs.",
        prevention="I-2 is at zero and **enforced**. Each gate declares a "
                   "denominator pattern in the registry and the meta-check "
                   "runs it and matches the output.",
        proven_by="scripts/test_gate_registry.py (a gate with no "
                  "denominator, and a declared-expensive probe) and "
                  "scripts/test_architecture_rules.py (files as well as rules)",
        verified_on="2026-08-03",
        still_holds=_enforced("I-2"),
    ),
    Closure(
        finding="KAI-GATE-003",
        defect="8 of 12 checks had never been observed failing. They may "
               "have been vacuous; nothing would have said so.",
        fix="Five new suites — compose drift, secret and restart, compose "
            "gates, gate registry, plus architecture additions — each "
            "injecting a real violation and asserting the gate fires, and "
            "asserting it does *not* fire on correct configuration.",
        prevention="I-3 is at zero and **enforced**. Every registered gate "
                   "declares a `proven_by` suite and the meta-check fails "
                   "if the file is absent.",
        proven_by="scripts/test_gate_registry.py — a gate with no "
                  "proven_by, and a proven_by pointing at nothing",
        verified_on="2026-08-03",
        still_holds=_enforced("I-3"),
    ),
    Closure(
        finding="KAI-GATE-021",
        defect="120 `except Exception: pass` handlers across the service "
               "entry points, 34 repo-wide. A dependency failed and the "
               "caller reported nothing: memU degraded from a shared "
               "durable store to twelve divergent in-memory lists with "
               "every health check green; `vault_delete` answered "
               "`{\"status\": \"ok\"}` whether or not the graph delete "
               "happened; `_sense_world` returned the empty string with "
               "the whole sensory tier down — byte-identical to a calm, "
               "fully-observed world — and Kai spoke about a world it had "
               "not seen.",
        fix="120 -> 4 at the entry points, 34 -> 7 repo-wide, against the "
            "operator's rubric: read/observe degrades visibly, "
            "mutate/act propagates, an aggregate names the source it "
            "lost. `common/degraded.record_degradation()` records a "
            "survived failure with a count and an age and surfaces it at "
            "/health; `degraded_partial()` carries a partial result and "
            "*raises* on an unnamed failure. The 7 survivors are two "
            "conn.close() teardowns, two logging handlers where the "
            "recorder would recurse, vault-sync's teardown, the "
            "recorder's own guard, and one security probe where raising "
            "is the hoped-for behaviour — each documented in place.",
        prevention="`hygiene_survey`'s `silent_swallows` column is "
                   "baselined and ratcheted in `policy-check`: the count "
                   "may fall and may not rise, so a new bare handler "
                   "fails the build. The survey's scope is derived from "
                   "the tree rather than a glob list, so a new module "
                   "cannot be added outside it.",
        proven_by="scripts/test_hygiene_gate.py (the ratchet refuses a "
                  "rise, every column is ratcheted, the scope covers "
                  "library modules); scripts/test_degraded.py (the "
                  "recorder aggregates, is bounded, survives contention, "
                  "and degraded_partial refuses an unnamed failure); "
                  "scripts/test_agentic_routes.py and "
                  "scripts/test_p16_operational.py (failure paths for "
                  "_sense_world and submit_feedback)",
        verified_on="2026-08-05",
        still_holds=_swallows_ratcheted,
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
