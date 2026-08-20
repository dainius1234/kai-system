#!/usr/bin/env python3
"""Item 8's six results, on two axes that may not launder one another.

WHY TWO AXES, AND A THIRD COLUMN
================================

This run does two jobs at once:

* **Axis 1 — the HuggingFace/network contingency.** Does the retry loop
  recover from a failing fetch and refuse after persistent denial?
* **Axis 2 — image provenance**, and the collectors' first qualification
  against a real Docker daemon.

Frozen R2: *"A collector fault leaves Axis 1's result standing and leaves
item 10's provenance unmoved; a clean binding cannot turn a failed
contingency into a success."*

The first implementation of this pair had **one** verdict field. A failed
`.Image` binding rewrote it to UNMEASURED, and this file then printed
that field under "AXIS 1". So an image-provenance fault silently became a
contingency measurement — precisely the laundering R2 forbids. Caught in
adversarial review before any build existed.

Two columns from the runner, computed independently, and a third derived
HERE and never taken from the producer of the first two:

    axis1_verdict          PASS / WRONG_FAILURE / UNMEASURED
    axis2_provenance       BOUND / MISMATCH / UNRECORDED /
                           IMAGE_NOT_PRODUCED_BY_DESIGN
    qualifies              derived below — and per branch, not by a
                           single "sound" set applied to all six; see
                           REQUIRED_A2, where a BOUND B3 is a
                           contradiction rather than acceptable evidence

WHY A ROW COUNT IS NOT A DENOMINATOR
====================================

The first implementation keyed rows by `(image, branch)` into a dict —
which silently collapses duplicates — and then checked only
`len(rows) == 6`. Six rows containing a duplicate and a missing branch
would have satisfied it while one of the six precommitted subjects had
never been measured at all.

**A denominator is the set of precommitted subjects, not a number of
lines.** This requires exactly the six expected keys, each once, no
extras, before any conclusion is drawn — and reports the mismatch
precisely when it is not so.

WHY THE TOOLCHAIN IS RE-HASHED HERE
===================================

The runner puts a `toolchain_sha256` in every row. That proves the six
rows agree with **each other** about which file they ran under; it does
not prove which file that was, and every one of them came from the same
producer. Rule 26 again, and I-8: the expected answer must not come from
the thing under test.

So `--toolchain` takes the artefact itself, recomputes its digest here,
and requires all six rows to carry exactly that value. A row bound to a
different toolchain than the one archived beside the results is a row
whose conditions are unknown, whatever its axes say.

**It is REQUIRED, and that is the whole point.** While the flag was
optional, six rows agreeing with each other reached ALL SIX QUALIFY on
the producer's word alone — the exact defect the paragraph above claims
to close, still reachable through the shipped entry point. Optional
independent evidence is not independent evidence.

Digest equality is also not the whole binding. `tree_sha` and `run_id`
are written into each row by the same runner that wrote the digest into
it, so comparing them with each other is the producer agreeing with
itself. Both are now reconciled against the artefact, as is the
base-image digest each branch observed at its own build — a mutable tag
that moves mid-experiment would otherwise become an unexplained
difference between arms.

**And the contract itself is imported, not restated.** This file used to
require only enough of the artefact to reconcile hash, tree, run and
base image, while `check_item8_toolchain.py` required eight identities
before build 1. A record holding only the four could therefore support a
closure claim — the package was not self-validating, it relied on
remembering that a stricter step had once run. Same function, both
boundaries.

WHY THIS FILE DERIVES THE AXES INSTEAD OF READING THEM
======================================================

Through D295 this file asked `row["axis1_verdict"] == "PASS"` — and the
calibration's own six-row fixture was the proof that this is not enough.
Those rows said `PASS` while carrying `runtime_retries_observed = 1` for
a B3 branch whose frozen contract requires **five**, with no execution
proof, no vertex error and no injection sequence anywhere. They were a
valid ALL SIX QUALIFY fixture.

So the last self-certification path was the biggest one: **the producer
of the observations was also the authority on what they meant.**

Both axes are now derived HERE, from artefacts produced by instruments
that are not the runner:

    BuildKit's own event stream   executed / cached / vertex error /
                                  runtime output, re-parsed from the
                                  archived captures
    docker's --iidfile            what the build says it produced
    the identity collectors       docker_image_id, execution_binding
    the container's own exit      the offline load, archived as a file
    the archived absence record   B3's no-image contract, plus this
                                  file's own check that no iidfile exists

The runner's `axis1_verdict` and `axis2_provenance` are still read — and
**compared**. A disagreement between the producer's classification and
the derived one REFUSES rather than being resolved in either direction:
one of the two is wrong, and which is not something to guess at.

    RAW OBSERVATION  →  QUALIFICATION  →  CLAIM
    (runner, BuildKit,   (here)            (here)
     collectors)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent.parent


def _sibling(name: str):
    """Import a sibling instrument rather than restating its contract.

    D272's failure was two records of the same thing drifting apart. The
    toolchain contract and the BuildKit event model each live in exactly
    one module; this reads them there.
    """
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_TC = _sibling("check_item8_toolchain")
_EV = _sibling("parse_buildkit_events")
_DV = _sibling("derive_item8_dockerfile")

# The frozen target instruction and the frozen markers, in one place.
TARGET = "for attempt in 1 2 3 4 5"
RETRY_MARK = "retrying in"
REFUSE_MARK = "REFUSING TO BUILD"
INJECT_MARK = "ITEM8-B2-INJECTED-FIRST-ATTEMPT"
BAKED_MARK = "BAKED "
B3_REQUIRED_RETRIES = 5

# The frozen denominator, in the frozen order.
EXPECTED = [(i, b) for i in ("memu-core", "memu-graph")
            for b in ("B1", "B2", "B3")]

PASS = "PASS"
WRONG = "WRONG_FAILURE"
UNMEASURED = "UNMEASURED"

# THE BRANCH CONTRACT, PER BRANCH. Not one "sound" set for all six.
#
# `SOUND_A2 = {BOUND, IMAGE_NOT_PRODUCED_BY_DESIGN}` was a set of states
# that are sound SOMEWHERE, applied EVERYWHERE. Under it a B3 row
# carrying BOUND qualified -- while B3's entire contract is that no image
# is produced -- and a B1 row carrying IMAGE_NOT_PRODUCED_BY_DESIGN could
# qualify on its iidfile alone. Both are contradictions, and a
# contradiction is not sound provenance to be tolerated: it is evidence
# that the row does not describe the branch it claims to.
#
# The frozen design assigns each branch exactly one admissible state, so
# this does too, and anything else REFUSES. (D295)
REQUIRED_A2 = {"B1": "BOUND", "B2": "BOUND",
               "B3": "IMAGE_NOT_PRODUCED_BY_DESIGN"}
SOUND_A2 = set(REQUIRED_A2.values())   # for reporting counts only


class Evidence:
    """What the artefact package says, read without the runner's help."""

    __slots__ = ("target", "runtime", "diagnostics", "unmet", "iid",
                 "identity", "binding", "offline_rc", "absence",
                 "identity_err", "binding_err", "offline_err", "absence_err",
                 "expected_command", "expected_flags", "binding_note")

    def __init__(self) -> None:
        self.target = None          # the BuildKit target vertex
        self.runtime = ""           # its runtime output, decoded
        self.diagnostics = 0
        self.unmet = ""             # why nothing could be read, if so
        self.iid = None             # ABSENT / "" / the id, per rule 20
        self.identity = None        # the explicit-collector record
        self.binding = None         # the executed-container comparison
        self.offline_rc = None      # the container's own exit status
        self.absence = None         # B3's archived no-image observation
        # WHY a record is unusable, kept rather than collapsed to None:
        # "absent" and "present but not about this subject" are different
        # facts and must not read the same (rule 20).
        self.identity_err = ""
        self.binding_err = ""
        self.offline_err = ""
        self.absence_err = ""
        self.expected_command = ""
        self.expected_flags: list[str] = []
        self.binding_note = ""


def _json_one(path: pathlib.Path) -> tuple[dict | None, str]:
    """EXACTLY ONE record. Not "the first one that parses".

    The identity and binding contracts are one subject, one record. The
    previous reader took the first non-empty JSON line and ignored
    everything after it, so a file holding two CONTRADICTORY records was
    silently reduced to whichever was written first — the reader
    choosing, invisibly, which evidence counts. Zero, two, malformed or
    trailing records all refuse now. (D297)
    """
    if not path.is_file():
        return None, f"{path.name} does not exist"
    objs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError as e:
            return None, f"{path.name} line {n} is not valid JSON ({e})"
    if not objs:
        return None, f"{path.name} holds no records"
    if len(objs) > 1:
        return None, (f"{path.name} holds {len(objs)} records; this contract "
                      f"is one subject and one record, and choosing among "
                      f"them would be this file deciding which evidence "
                      f"counts")
    if not isinstance(objs[0], dict):
        return None, f"{path.name} does not hold a JSON object"
    return objs[0], ""


# WHICH RECORD TYPES CARRY A COMMIT, and why one does not.
#
# The executed-container binding is written by `collect_image_identity.py`,
# which is BYTE-FROZEN at b53fd4e and whose record contract carries
# service, run and tree but not commit. Requiring a commit there would
# mean editing a frozen instrument to satisfy a checker -- the tail
# wagging the dog, and the exact thing "an unchanged instrument needs no
# argument that it is unchanged" exists to prevent.
#
# It is a NAMED exception, not a silent conditional: the binding record
# is still reconciled on service, image ref, run and tree, and its
# `collected_image_id` must equal the identity record's image id -- and
# THAT record is commit-checked. So the binding is tied to a
# commit-verified record transitively, by an id rather than by a field.
# (D299)
COMMIT_BEARING = {"identity", "offline-load", "absence"}


def subject_problems(rec: dict, kind: str, label: str, image_ref: str,
                     run_id: str | None, tree_sha: str | None,
                     commit_sha: str | None = None) -> list[str]:
    """Is this record about THE SUBJECT, or merely in the right filename?

    The collectors already stamp `service`, `image_ref`, `commit_sha`,
    `tree_sha` and `run_id` into every record. The claim engine read the
    expected FILENAME and then ignored all of it — so a record from
    another branch, another run or another tree, placed under the
    expected name, could participate in BOUND. D247's bar is exact claim
    → tree/image/run, and a filename is none of those. (D297)
    """
    out: list[str] = []
    if rec.get("service") != label:
        out.append(f"the {kind} record names service "
                   f"{rec.get('service')!r}, not {label!r}")
    if rec.get("image_ref") not in (None, image_ref):
        out.append(f"the {kind} record names image_ref "
                   f"{rec.get('image_ref')!r}, not {image_ref!r}")
    if run_id is not None and str(rec.get("run_id")) != str(run_id):
        out.append(f"the {kind} record names run {rec.get('run_id')!r}, "
                   f"the toolchain names {run_id!r}")
    if tree_sha is not None and rec.get("tree_sha") != tree_sha:
        out.append(f"the {kind} record names tree "
                   f"{str(rec.get('tree_sha'))[:12]}, the toolchain names "
                   f"{tree_sha[:12]}")
    # REQUIRED, not conditional. D298 added this check but skipped it
    # when the record carried no commit at all -- so absence bypassed it
    # silently, and two of the four record types did not carry one. That
    # made "all four compare the commit" mechanically false while reading
    # as true, which is the shape of statement this register exists to
    # prevent. Present, and equal, or it is a problem. (D299)
    if commit_sha is not None and kind in COMMIT_BEARING:
        if rec.get("commit_sha") is None:
            out.append(f"the {kind} record carries no commit_sha; absence "
                       f"is not agreement")
        elif rec["commit_sha"] != commit_sha:
            out.append(f"the {kind} record names commit "
                       f"{str(rec['commit_sha'])[:12]}, the toolchain names "
                       f"{commit_sha[:12]}")
    return out


def load_evidence(image: str, branch: str, derived: pathlib.Path,
                  ident: pathlib.Path, run_id: str | None = None,
                  tree_sha: str | None = None,
                  commit_sha: str | None = None,
                  binding_rule: dict | None = None) -> Evidence:
    """Re-read the raw artefacts. No row is consulted anywhere here."""
    e = Evidence()
    label = f"item8-{branch.lower()}-{image}"

    captures = [derived / f"{image}.{branch}.events-stderr.jsonl",
                derived / f"{image}.{branch}.events-stdout.jsonl"]
    present = [p for p in captures if p.is_file()]
    if not present:
        e.unmet = (f"no archived BuildKit event capture for {image}/{branch}; "
                   f"the raw evidence for this branch is not in the package")
        return e
    bearing = []
    for p in present:
        vx, diag, err = _EV.parse(p.read_text(), p.name)
        e.diagnostics += len(diag)
        if err and "held no BuildKit events" not in err:
            e.unmet = f"the archived capture {p.name} is unreadable: {err}"
            return e
        if vx:
            bearing.append((p.name, vx))
    if not bearing:
        e.unmet = (f"the archived captures for {image}/{branch} hold no "
                   f"BuildKit events at all")
        return e
    if len(bearing) > 1:
        # Same rule as the parser, for the same reason: file order is not
        # chronology, and B2's criterion is an order.
        e.unmet = (f"{len(bearing)} archived captures for {image}/{branch} "
                   f"contain events; their chronology relative to one "
                   f"another is unestablished")
        return e

    # ── THE CAPTURE MUST BE THIS SUBJECT'S, NOT MERELY THIS FILENAME'S ─
    #
    # The target used to be found by `for attempt in 1 2 3 4 5`, which
    # appears in ALL SIX derived Dockerfiles. So a memu-core/B3 capture
    # copied under memu-graph/B3's name could satisfy memu-graph's five
    # retries, refusal and vertex error, and nothing would notice --
    # D297's own rule, one layer lower: a filename is not an identity.
    #
    # The subject is now re-derived from the SHIPPED Dockerfile, required
    # to equal the archived derived file byte for byte, and its target
    # instruction is what selects the vertex. Measured over the six:
    # 13 of 15 pairs are separated by the command alone; the two that
    # are not are B1 vs B3 within one image, which differ only by the
    # RUN flag -- see the binding rule below. (D298)
    src = REPO / image / "Dockerfile"
    want_df = derived / f"Dockerfile.{image}.{branch}"
    if not src.is_file():
        e.unmet = f"the shipped {image}/Dockerfile is not in this tree"
        return e
    if not want_df.is_file():
        e.unmet = (f"no archived derived Dockerfile for {image}/{branch}; "
                   f"the subject of this capture cannot be established")
        return e
    expect_text, _n, derr = _DV.derive(src.read_text(), branch)
    if derr:
        e.unmet = f"{image}/{branch} could not be re-derived: {derr}"
        return e
    if want_df.read_text() != expect_text:
        e.unmet = (f"the archived derived Dockerfile for {image}/{branch} is "
                   f"NOT what deriving the shipped {image}/Dockerfile "
                   f"produces; the subject built is not the subject the "
                   f"design specifies")
        return e
    run_text = _DV.find_target_run(expect_text)
    if run_text is None:
        e.unmet = (f"the derived Dockerfile for {image}/{branch} holds no "
                   f"target RUN, so no vertex can be bound to it")
        return e
    full = _EV.normalise_command(run_text)
    body, flags = _EV.strip_run_flags(full)
    e.expected_command = body
    e.expected_flags = flags

    hits = [v for v in bearing[0][1].values()
            if body in _EV.normalise_command(v.name)]
    if not hits:
        e.unmet = (f"no vertex in the {image}/{branch} capture carries this "
                   f"subject's target instruction. The capture is not "
                   f"evidence about {image}/{branch}, whatever it is filed "
                   f"as")
        return e
    if len(hits) > 1:
        e.unmet = (f"{len(hits)} vertices carry {image}/{branch}'s target "
                   f"instruction; the subject must be unambiguous")
        return e
    target = hits[0]

    # THE FLAG, WHEN THE DAEMON EXPOSES IT. B1 and B3 differ only by
    # `--network=none`, and whether BuildKit keeps RUN flags in a vertex
    # NAME is a property of the daemon that no amount of reasoning here
    # settles. The preflight MEASURES it and archives the answer; this
    # applies the strongest rule that measurement supports, and says
    # plainly when it supports none.
    seen = _EV.normalise_command(target.name)
    # THERE IS NO DEGRADED MODE, and D298 was WRONG to build one.
    #
    # D298 §2 claimed B1 and B3 could fall back on "disjoint Axis-1
    # criteria" when the flag is unavailable. They are not disjoint.
    # memu-core/Dockerfile:92-107 -- the UNMUTATED control -- retries
    # five times, prints "REFUSING TO BUILD" and exits 1 when its
    # genuine fetch cannot reach upstream. That is byte for byte the
    # evidence shape B3 requires, and an upstream outage during B1 is an
    # explicitly recognised possibility, not a hypothetical.
    #
    # So a B1 outage capture filed as B3 would have been indistinguishable
    # from a real B3 under the fallback. Without the flag this
    # instrumentation cannot say WHICH frozen subject produced a capture,
    # and that is an UNMEASURED instrument capability -- not licence to
    # infer identity from the observed result. The preflight refuses
    # before build 1 instead. (D299)
    if binding_rule is None:
        e.unmet = ("no admissible binding rule; the preflight that measures "
                   "how this daemon represents a RUN did not run, or its "
                   "record does not describe this execution")
        return e
    for f in flags:
        if f not in seen:
            e.unmet = (f"{image}/{branch} requires {f} in its target "
                       f"instruction and this capture's target does not "
                       f"carry it; this is not {branch}'s evidence")
            return e
    for f in ("--network=none",):
        if f in seen and f not in flags:
            e.unmet = (f"the capture's target carries {f}, which "
                       f"{image}/{branch} does not; this is another "
                       f"branch's evidence")
            return e
    e.target = target
    e.runtime = "".join(target.log)

    iid = derived / f"{image}.{branch}.iid"
    e.iid = iid.read_text().strip() if iid.is_file() else None

    # EVERY per-branch record is checked for SUBJECT IDENTITY, not just
    # for being in the expected filename.
    image_ref = f"kai-item8:{branch.lower()}-{image}"
    if branch != "B3":
        e.identity, err = _json_one(ident / f"{label}.jsonl")
        if err:
            e.identity_err = err
        elif e.identity:
            probs = subject_problems(e.identity, "identity", label,
                                     image_ref, run_id, tree_sha, commit_sha)
            if probs:
                e.identity, e.identity_err = None, "; ".join(probs)
        e.binding, err = _json_one(ident / f"{label}.executed.jsonl")
        if err:
            e.binding_err = err
        elif e.binding:
            probs = subject_problems(e.binding, "binding", label,
                                     image_ref, run_id, tree_sha, commit_sha)
            if probs:
                e.binding, e.binding_err = None, "; ".join(probs)
        off, err = _json_one(ident / f"{label}.offline.json")
        if err:
            e.offline_err = err
        elif off:
            probs = subject_problems(off, "offline-load", label, image_ref,
                                     run_id, tree_sha, commit_sha)
            if probs:
                e.offline_err = "; ".join(probs)
            elif not isinstance(off.get("exit_status"), int):
                e.offline_err = ("the offline-load record names no integer "
                                 "exit_status")
            else:
                e.offline_rc = off["exit_status"]
    else:
        e.absence, err = _json_one(ident / f"{label}.absence.json")
        if err:
            e.absence_err = err
        elif e.absence:
            probs = subject_problems(e.absence, "absence", label, image_ref,
                                     run_id, tree_sha, commit_sha)
            if probs:
                e.absence, e.absence_err = None, "; ".join(probs)
    return e


def ordered(runtime: str) -> bool:
    """injection → a GENUINE retry → a later success, in that order.

    Derived from the runtime text here rather than taken from the
    runner's `awk`. Same criterion, computed by a different party.
    """
    i = runtime.find(INJECT_MARK)
    if i < 0:
        return False
    r = runtime.find(RETRY_MARK, i + len(INJECT_MARK))
    if r < 0:
        return False
    return runtime.find(BAKED_MARK, r + len(RETRY_MARK)) >= 0


def derive_axis1(branch: str, e: Evidence) -> tuple[str, str]:
    """The contingency verdict, from BuildKit's evidence and nothing else.

    Frozen R2's criteria, restated as questions about artefacts:

      B1  the target instruction EXECUTED, uncached, without its own
          error, and the image loaded its asset with the network denied
      B2  the same, plus exactly one injection and the ordered recovery
      B3  the target carries its OWN error, refused, and did so after
          exactly the five attempts the design names
    """
    if e.unmet:
        return UNMEASURED, e.unmet
    t = e.target
    retries = e.runtime.count(RETRY_MARK)
    refusals = e.runtime.count(REFUSE_MARK)
    injections = e.runtime.count(INJECT_MARK)

    if branch == "B3":
        if not t.error:
            if refusals:
                return WRONG, ("the target vertex carries no error of its "
                               "own; the refusal text is present but the "
                               "failure is not attributable to that step")
            return UNMEASURED, ("no error on the target vertex and no "
                                "refusal in its output; the intended "
                                "refusal is not evidenced")
        if not refusals:
            return WRONG, ("the target vertex errored without its refusal "
                           "marker; the failure was something else")
        if retries != B3_REQUIRED_RETRIES:
            return UNMEASURED, (f"{retries} runtime retry line(s) attributed "
                                f"to the target vertex, not the "
                                f"{B3_REQUIRED_RETRIES} the design requires")
        if e.iid is not None:
            return UNMEASURED, ("an iidfile exists in the artefact package, "
                                "so the no-image contract is not established")
        if e.absence_err:
            return UNMEASURED, (f"the archived absence record is unusable: "
                                f"{e.absence_err}")
        if not e.absence:
            return UNMEASURED, ("no archived absence observation; post-build "
                                "non-existence is asserted, not recorded")
        if (e.absence.get("post_build_tag") != "absent"
                or e.absence.get("post_build_iidfile") != "absent"
                or e.absence.get("pre_build_state") != "clean"):
            return UNMEASURED, (f"the archived absence record does not "
                                f"establish non-existence: {e.absence}")
        return PASS, (f"{retries} retries and the refusal, both attributed "
                      f"to the target vertex; no image at either end")

    # B1 and B2 — the positive branches.
    if not t.started:
        return UNMEASURED, "the target vertex did not execute in this build"
    if t.cached:
        return UNMEASURED, ("the target vertex was served FROM CACHE; the "
                            "genuine fetch path did not run")
    if t.error:
        return WRONG, f"the target vertex carries an error: {t.error[:120]}"
    if e.offline_err:
        return UNMEASURED, f"offline-load record unusable: {e.offline_err}"
    if e.offline_rc is None:
        return UNMEASURED, ("no archived offline-load result; the branch's "
                            "criterion is asserted, not recorded")
    if e.offline_rc != 0:
        return UNMEASURED, (f"the offline asset load exited {e.offline_rc} "
                            f"with the network denied")
    if branch == "B2":
        if injections != 1:
            return UNMEASURED, (f"{injections} injection marker(s) in the "
                                f"target vertex output; exactly one is "
                                f"required")
        if retries < 1:
            return UNMEASURED, ("no runtime retry line attributed to the "
                                "target vertex; recovery is not established")
        if not ordered(e.runtime):
            return UNMEASURED, ("the runtime output does not show injection, "
                                "then a genuine retry, then a success, in "
                                "that order")
    elif injections:
        return WRONG, (f"{injections} injection marker(s) in a branch that "
                       f"has no injection; this is not B1's subject")
    return PASS, "executed uncached, and loaded its asset offline"


def derive_axis2(branch: str, e: Evidence) -> tuple[str, str]:
    """Provenance, from the identity artefacts rather than from a word.

    For B1/B2 the three independent records already exist and were
    simply never read here: the explicit collector's identity, the
    executed-container binding, and docker's own iidfile.
    """
    if branch == "B3":
        if e.iid is not None:
            return "MISMATCH", ("an iidfile exists; B3's contract is that no "
                                "image was produced")
        if e.absence_err:
            return "UNRECORDED", f"absence record unusable: {e.absence_err}"
        if not e.absence:
            return "UNRECORDED", "no archived absence observation"
        if (e.absence.get("post_build_tag") != "absent"
                or e.absence.get("post_build_iidfile") != "absent"):
            return "MISMATCH", f"the absence record shows {e.absence}"
        return "IMAGE_NOT_PRODUCED_BY_DESIGN", "no image at either end"

    if e.identity_err:
        return "UNRECORDED", f"identity record unusable: {e.identity_err}"
    if not e.identity:
        return "UNRECORDED", "no archived image-identity record"
    if e.identity.get("identity_state") != "RECORDED":
        return "UNRECORDED", (f"the identity record is "
                              f"{e.identity.get('identity_state')}")
    collected = e.identity.get("docker_image_id")
    if not collected:
        return "UNRECORDED", "the identity record names no image id"
    if e.iid is None:
        return "MISMATCH", ("no iidfile in the package; R2 requires the "
                            "corroboration and ABSENT is not 'no objection'")
    if e.iid != collected:
        return "MISMATCH", (f"the iidfile says {e.iid[:19]}, the collector "
                            f"says {collected[:19]}")
    if e.binding_err:
        return "UNRECORDED", f"binding record unusable: {e.binding_err}"
    if not e.binding:
        return "UNRECORDED", "no executed-container binding record"

    # MATCH IS RE-DERIVED FROM THE RAW IDs, not accepted as a word.
    #
    # The binding artefact carries `execution_binding`, and it also
    # carries the two ids that verdict was computed from. Reading the
    # verdict and ignoring the ids left one classification still trusted
    # inside an artefact this file otherwise reparses — so a record
    # saying MATCH while its own ids differ would have qualified. The
    # ids are the evidence; MATCH is somebody's reading of them. (D297)
    b_coll = e.binding.get("collected_image_id")
    b_exec = e.binding.get("executed_image_id")
    if not b_coll or not b_exec:
        return "UNRECORDED", ("the binding record does not carry both image "
                              "ids, so the comparison cannot be redone here")
    if b_coll != collected:
        return "MISMATCH", ("the binding compared against a different "
                            "identity than the one collected")
    if b_exec != b_coll:
        return "MISMATCH", (f"the executed container ran {b_exec[:19]}, the "
                            f"collected identity is {b_coll[:19]}")
    if e.binding.get("execution_binding") != "MATCH":
        # The ids agree and the record says otherwise: a contradiction
        # inside one artefact, refused rather than resolved.
        return "MISMATCH", (f"the binding record says "
                            f"{e.binding.get('execution_binding')} while its "
                            f"own two ids are equal; the artefact "
                            f"contradicts itself")
    return "BOUND", ("identity recorded, iidfile corroborated, and the "
                     "executed image re-derived equal from the raw ids")


def qualifies(r: dict, a1: str, a2: str) -> tuple[bool, str]:
    """Whether DERIVED evidence supports closure. The row supplies neither
    axis: `a1` and `a2` come from `derive_axis1`/`derive_axis2`, computed
    from the artefact package.

    The row is still consulted for its toolchain binding, which is a fact
    about which conditions record it names — reconciled separately
    against the artefact — and not a verdict about itself.
    """
    tc = r.get("toolchain_sha256")
    if not tc or tc == "ABSENT":
        return False, f"toolchain binding is {tc or 'missing'}"
    if a1 != PASS:
        return False, f"Axis 1 is {a1}"
    branch = r.get("branch")
    want = REQUIRED_A2.get(branch)
    if want is None:
        return False, f"{branch} is not a precommitted branch"
    if a2 != want:
        if a2 in SOUND_A2:
            return False, (f"Axis 2 is {a2}, which {branch} may never be: "
                           f"the branch contract requires {want}, and a row "
                           f"claiming the other is describing a different "
                           f"branch than the one it is filed under")
        return False, f"Axis 2 is {a2}, not {want}"
    if branch == "B3":
        return True, "refused by design, no image to bind"
    # The iidfile corroboration R2 requires is now PART OF deriving
    # BOUND, from the archived iidfile itself rather than from a row
    # field reporting the comparison's outcome. Reaching BOUND above
    # already means the collector's id, the iidfile and the executed
    # container's `.Image` all agree.
    return True, "Axis 1 PASS, bound, iidfile corroborated (all derived)"


def refuse(reason: str, detail: str = "") -> int:
    print("ITEM 8 UNMEASURED — EXPERIMENT INSTRUMENT FAILURE")
    print(f"  unmet prerequisite: {reason}")
    if detail:
        print(f"  {detail}")
    print("  No conclusion is drawn about the contingency from a partial "
          "or malformed result set.")
    return 4


def validate_keys(rows: list[dict]) -> tuple[bool, list[str]]:
    """Exactly the six precommitted subjects, each exactly once."""
    seen = [(r.get("image"), r.get("branch")) for r in rows]
    problems: list[str] = []
    for key in EXPECTED:
        n = seen.count(key)
        if n == 0:
            problems.append(f"MISSING: {key[0]}/{key[1]} was never reported")
        elif n > 1:
            problems.append(f"DUPLICATE: {key[0]}/{key[1]} reported {n} times")
    for key in sorted(set(seen) - set(EXPECTED)):
        problems.append(f"UNEXPECTED: {key[0]}/{key[1]} is not a "
                        f"precommitted subject")
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    # REQUIRED. Optional independent evidence is not independent
    # evidence: with the flag omitted, six rows agreeing with each other
    # about a digest reached ALL SIX QUALIFY on the producer's word
    # alone, which is the exact I-8 defect D294 claimed to close. The
    # calibration was itself calling the summariser without it. (D295)
    ap.add_argument("--toolchain", required=True,
                    help="REQUIRED. The toolchain artefact itself. Its "
                         "digest is recomputed here, and its tree and run "
                         "identity are compared against every row -- six "
                         "rows agreeing with each other are six statements "
                         "from one producer")
    # THE RAW EVIDENCE PACKAGE. Both axes are derived from what is in
    # these directories, not from the row's classification of it.
    ap.add_argument("--derived-dir", default="item8-derived",
                    help="archived BuildKit event captures and iidfiles")
    ap.add_argument("--identity-dir", default="item8-identity",
                    help="archived identity, binding, offline-load and "
                         "absence records")
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_file():
        return refuse(f"{path} does not exist",
                      "No branch result was recorded, so there is nothing "
                      "to report.")
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines()
                if l.strip()]
    except json.JSONDecodeError as e:
        return refuse(f"{path} is not readable as JSONL: {e}")
    if not rows:
        return refuse(f"{path} is empty",
                      "Six branches were precommitted and none reported.")

    ok, problems = validate_keys(rows)

    # THE TOOLCHAIN, RE-HASHED FROM THE ARTEFACT. Computed before the
    # tables so the refusal below can be unconditional: a result set whose
    # conditions are unknown is not summarised under a heading that
    # implies they are known.
    tc_expected = None
    tc_rec: dict[str, str] = {}
    tc_problems: list[str] = []
    tc_path = pathlib.Path(args.toolchain)
    if not tc_path.is_file():
        tc_problems.append(
            f"TOOLCHAIN: {tc_path} does not exist. R2 records these "
            f"identities with every branch; the rows name a digest of "
            f"nothing this summary can read")
    else:
        raw = tc_path.read_bytes()
        tc_expected = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
                tc_rec[k.strip()] = v.strip()
        # THE HASH IS NOT THE WHOLE BINDING. Digest equality proves a row
        # names THIS file. It does not prove the row was produced under
        # the tree and run the file describes -- `tree_sha` and `run_id`
        # are written into the row by the same runner that wrote the
        # digest into it, so on their own they are the producer agreeing
        # with itself. The artefact is the second source; compare them.
        # (D295)
        want_tree = tc_rec.get("tree_sha")
        want_run = tc_rec.get("run_id")
        for r in rows:
            who = f"{r.get('image')}/{r.get('branch')}"
            got = r.get("toolchain_sha256")
            if got != tc_expected:
                tc_problems.append(
                    f"TOOLCHAIN: {who} is bound to {str(got)[:12]}, not the "
                    f"archived artefact's {tc_expected[:12]}")
            if want_tree and r.get("tree_sha") != want_tree:
                tc_problems.append(
                    f"TOOLCHAIN: {who} names tree "
                    f"{str(r.get('tree_sha'))[:12]}, the artefact names "
                    f"{want_tree[:12]}. A row and its conditions must "
                    f"describe the same tree")
            if want_run and str(r.get("run_id")) != want_run:
                tc_problems.append(
                    f"TOOLCHAIN: {who} names run {r.get('run_id')}, the "
                    f"artefact names {want_run}")
        # THE SAME EIGHT-FIELD CONTRACT the pre-build validator applies,
        # imported from it rather than restated. This file used to check
        # only what it needed for reconciliation, so an artefact holding
        # tree, run and base image alone could support closure while
        # lacking the frontend, the Docker and buildx versions, the
        # runner OS and the commit -- and reading the package later would
        # have meant REMEMBERING that a stricter step once ran. (D296)
        for p in _TC.contract_problems(tc_rec):
            tc_problems.append(f"TOOLCHAIN: {p}")
        if not want_tree or not want_run:
            tc_problems.append(
                "TOOLCHAIN: the artefact does not name a tree_sha and a "
                "run_id, so the rows cannot be reconciled against it")

        # THE BASE IMAGE IS A MUTABLE TAG. `python:3.11-slim` can move
        # under the experiment, and six arms built against two different
        # base images are not six arms of one experiment. Pinning it
        # would change the subject, so instead each branch RECORDS what
        # the tag resolved to at its own build, and all six must agree
        # with each other and with the pre-run record. Observation, not
        # mutation -- and a divergence blocks interpretation rather than
        # being discovered later as an unexplained difference. (D295)
        want_base = tc_rec.get("base_image_digest")
        seen_base = {str(r.get("base_image_digest")) for r in rows}
        if want_base and seen_base - {want_base}:
            tc_problems.append(
                f"BASE IMAGE: the tag resolved to more than one digest "
                f"across the experiment — recorded before build 1 as "
                f"{want_base[:19]}, observed {sorted(seen_base)}. Six arms "
                f"built on two base images are not six arms of one "
                f"experiment, and which arms differ is not recoverable "
                f"afterwards")

    # ── DERIVE BOTH AXES FROM THE ARTEFACTS, PER SUBJECT ─────────────
    #
    # Nothing below reads `axis1_verdict` or `axis2_provenance` to decide
    # anything. Those two are read exactly once each, to be COMPARED.
    derived_dir = pathlib.Path(args.derived_dir)
    ident_dir = pathlib.Path(args.identity_dir)

    # HOW THIS DAEMON REPRESENTS A RUN, measured before build 1 and
    # archived. Without it the STRENGTH of the subject binding below is
    # unknown, and an unknown-strength binding is not one (R11).
    binding_rule, br_err = _json_one(derived_dir / "binding-rule.json")
    binding_problems: list[str] = []
    if br_err:
        binding_rule = None
        binding_problems.append(f"BINDING RULE: {br_err}")
    else:
        # THE RULE IS EVIDENCE TOO, and is held to the same standard as
        # everything else it licenses: one record, from THIS run and
        # tree, and both capabilities actually present. A rule carried
        # forward from another run would authorise a binding nobody
        # measured here. (D299)
        for k in ("full_instruction_in_vertex_name", "flags_in_vertex_name"):
            if binding_rule.get(k) is not True:
                binding_problems.append(
                    f"BINDING RULE: {k} is {binding_rule.get(k)!r}. Without "
                    f"it the six subjects cannot be told apart by their "
                    f"instructions, and the preflight should have refused "
                    f"before build 1")
        if tc_rec.get("run_id") and str(binding_rule.get("run_id")) != \
                str(tc_rec["run_id"]):
            binding_problems.append(
                f"BINDING RULE: measured in run {binding_rule.get('run_id')!r}, "
                f"the toolchain names {tc_rec['run_id']!r}")
        if tc_rec.get("tree_sha") and binding_rule.get("tree_sha") != \
                tc_rec["tree_sha"]:
            binding_problems.append(
                f"BINDING RULE: measured against tree "
                f"{str(binding_rule.get('tree_sha'))[:12]}, the toolchain "
                f"names {tc_rec['tree_sha'][:12]}")
        if binding_problems:
            binding_rule = None
    tc_problems.extend(binding_problems)
    derived: dict[tuple, tuple[str, str, str, str]] = {}
    disagreements: list[str] = []
    for image, branch in EXPECTED:
        ev = load_evidence(image, branch, derived_dir, ident_dir,
                           tc_rec.get('run_id'), tc_rec.get('tree_sha'),
                           tc_rec.get('commit_sha'), binding_rule)
        d1, why1 = derive_axis1(branch, ev)
        d2, why2 = derive_axis2(branch, ev)
        derived[(image, branch)] = (d1, why1, d2, why2)
        for r in rows:
            if (r.get("image"), r.get("branch")) != (image, branch):
                continue
            # A DISAGREEMENT IS A FINDING, NOT A TIE TO BREAK. One of the
            # two computations is wrong, and choosing between them here
            # would be the summariser deciding which instrument to
            # believe -- which is the authority question all over again.
            if r.get("axis1_verdict") != d1:
                disagreements.append(
                    f"DISAGREEMENT: {image}/{branch} Axis 1 — the row says "
                    f"{r.get('axis1_verdict')}, the artefacts give {d1} "
                    f"({why1})")
            if r.get("axis2_provenance") != d2:
                disagreements.append(
                    f"DISAGREEMENT: {image}/{branch} Axis 2 — the row says "
                    f"{r.get('axis2_provenance')}, the artefacts give {d2} "
                    f"({why2})")
            if "qualified_for_closure" in r:
                got, _ = qualifies(r, d1, d2)
                if bool(r["qualified_for_closure"]) != got:
                    disagreements.append(
                        f"DISAGREEMENT: {image}/{branch} row claims "
                        f"qualified={r['qualified_for_closure']}, derived "
                        f"{got}")

    print("ITEM 8 — HUGGINGFACE/NETWORK CONTINGENCY")
    print("=" * 74)
    print()
    print("AXIS 1 — the contingency, DERIVED from BuildKit's own evidence")
    print("-" * 74)
    for image, branch in EXPECTED:
        matches = [r for r in rows if (r.get("image"), r.get("branch"))
                   == (image, branch)]
        if not matches:
            print(f"  {image:<12} {branch}  NOT REPORTED")
            continue
        d1, why1, _, _ = derived[(image, branch)]
        for r in matches:
            print(f"  {image:<12} {branch}  {d1:<14}"
                  f" (row said {r.get('axis1_verdict', '?')})"
                  f" elapsed={r.get('elapsed_seconds', '?')}s")
            print(f"  {'':<12}     {why1}")

    print()
    print("AXIS 2 — provenance, DERIVED from the identity artefacts")
    print("-" * 74)
    for image, branch in EXPECTED:
        for r in [r for r in rows if (r.get("image"), r.get("branch"))
                  == (image, branch)]:
            d1, _, d2, why2 = derived[(image, branch)]
            q, why = qualifies(r, d1, d2)
            print(f"  {image:<12} {branch}  {d2:<30}"
                  f" (row said {r.get('axis2_provenance', '?')})"
                  f" qualifies={'yes' if q else 'NO'}")
            print(f"  {'':<12}     {why2}")
            if not q:
                print(f"  {'':<12}     BLOCKED: {why}")

    a1 = {v: sum(1 for k, d in derived.items() if d[0] == v)
          for v in (PASS, WRONG, UNMEASURED)}
    a2_sound = sum(1 for d in derived.values() if d[2] in SOUND_A2)
    quals = {(r.get("image"), r.get("branch")):
             qualifies(r, *(derived.get((r.get("image"), r.get("branch")),
                                        (UNMEASURED, "", "UNRECORDED", ""))[i]
                            for i in (0, 2)))
             for r in rows}
    qualified = sum(1 for v in quals.values() if v[0])

    print()
    print(f"  inspected: {len(rows)} result row(s) against "
          f"{len(EXPECTED)} precommitted subject(s)")
    print(f"    AXIS 1   PASS {a1[PASS]}  WRONG_FAILURE {a1[WRONG]}  "
          f"UNMEASURED {a1[UNMEASURED]}")
    print(f"    AXIS 2   sound {a2_sound} of {len(rows)}")
    print(f"    QUALIFIED FOR CLOSURE  {qualified} of {len(EXPECTED)}")
    if tc_expected:
        print(f"    TOOLCHAIN  recomputed {tc_expected[:16]}… from the "
              f"artefact, reconciled against {len(rows)} row(s) on digest, "
              f"tree, run and base image, and validated against the same "
              f"{len(_TC.REQUIRED)}-field contract the pre-build check uses")
    else:
        print("    TOOLCHAIN  NOT RECOMPUTED — the artefact is missing")

    print()
    print("  Reading rules, frozen before these results existed:")
    print("   * B2 measures recovery from an INJECTED FETCH-COMMAND failure.")
    print("     It does NOT measure recovery from a real network outage.")
    print("   * UNMEASURED is never an adverse result about the contingency.")
    print("   * WRONG_FAILURE is never a PASS and never a FAIL of it.")
    print("   * An Axis-2 fault blocks closure and leaves Axis 1 standing.")
    print("   * Closure qualification is DERIVED here from the evidence,")
    print("     never taken from the producer of it (rule 26).")
    print("   * No re-draws. An UNMEASURED branch stays UNMEASURED and Item 8")
    print("     is incomplete for that subject. (D247 §5, D289)")

    if tc_problems:
        print()
        for p in tc_problems:
            print(f"FAIL: {p}")
        print()
        print("Six rows agreeing with each other about a digest are six "
              "statements from one producer. The artefact is the "
              "independent evidence (I-8), and a row bound to a different "
              "toolchain ran under conditions this summary cannot name.")
        return 4

    if disagreements:
        print()
        for d in disagreements:
            print(f"FAIL: {d}")
        print()
        print("A row whose classification contradicts the artefacts is a "
              "finding, in EITHER direction, and it is not resolved here: "
              "one of the two computations is wrong, and choosing between "
              "them would be this file deciding which instrument to "
              "believe — the authority question all over again (rule 26).")
        return 4

    if not ok:
        print()
        for p in problems:
            print(f"FAIL: {p}")
        print()
        print("The denominator is the six precommitted subjects, not a count "
              "of lines. It is not adjusted downward, and a duplicate does "
              "not substitute for a missing subject.")
        return 4

    if a1[PASS] == len(EXPECTED) and qualified == len(EXPECTED):
        print()
        print("ALL SIX QUALIFY: both contingencies recover from an injected "
              "failure and refuse after persistent denial, and every branch "
              "carries sound provenance.")
        return 0

    print()
    if a1[PASS] == len(EXPECTED):
        print(f"AXIS 1 COMPLETE, PROVENANCE INCOMPLETE: 6/6 contingency PASS "
              f"but only {qualified}/6 qualify for closure. The contingency "
              f"result stands; item 10's provenance does not move for the "
              f"branches whose Axis 2 is unsound.")
    else:
        print(f"NOT ALL SIX PASS: Axis 1 {a1[PASS]}/6. Item 8 is not "
              f"satisfied. Every outcome above is banked as it occurred.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
