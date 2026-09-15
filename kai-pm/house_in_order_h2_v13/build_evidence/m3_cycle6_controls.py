#!/usr/bin/env python3
"""M3 CYCLE-6 CONTROLS. Banked WITH the implementation, before any run.

WHAT CYCLE 6 REPAIRS. Cycle 5 treated a label that fails the intrinsic
document-binding test as a TERMINAL VETO:

    if not DOC_BINDING.match(lab):
        return "SPAN"

so no later evidence could be considered, however strong. The absence of
INTRINSIC document-binding evidence is not PROOF OF LOCAL SCOPE. Cycle 6
turns the veto into a failed conjunct and admits ONE further, contextual
class: `Status` inside a document's own root metadata block, beside a
field that has independently earned document binding.

FAIL-OLD / PASS-NEW IS AGAINST THE REAL PREDECESSOR. `--old-ref` is
loaded from the git object and executed; the old answer is not a
hand-written imitation of what cycle 5 "would have" said (R1).

WHY THE CORPUS PROBES ASK ABOUT A POSITION RATHER THAN A WITNESS. Kai's
known negatives are `Status` SITES, and not every one of them carries an
emitted witness. `_scope_of` is a pure function of (text, offset,
detector), so each site is probed at the first character of its VALUE.
That is a STRICTLY STRONGER control than probing only the sites that
happen to carry a witness: it puts the question to the classifier at
every site Kai named, whether or not the detectors reach it.

SCOPE OF THE CORPUS PROBES. Three documents, named by Kai as the control
set. This file does NOT evaluate the 492 and does not read the record
manifest; the corpus run is a separate, later step.

    python3 m3_cycle6_controls.py --subject-repo R --tree T
    python3 m3_cycle6_controls.py --subject-repo R --tree T --old-ref SHA
"""
from __future__ import annotations
import argparse
import importlib.util
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import passa as NEW                                            # noqa: E402

CYCLE5 = "c3c7731592620227a35da4200b936dd7480140f7"
PASSA_PATH = "kai-pm/house_in_order_h2_v13/passa.py"

FAILED = []


def check(name, got, want, why=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'OK  ' if ok else '<<< '}{name:<58} {got:<10} "
          f"{'' if ok else 'expected ' + want}")
    if why and not ok:
        print(f"        {why}")
    return ok


def _git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout if binary else r.stdout.decode()


def load_old(repo, ref):
    """The predecessor implementation, executed from the git object.

    `envelope` resolves through the package directory already on
    sys.path, so no part of the working tree's passa.py is involved.
    """
    src = _git(repo, "show", f"{ref}:{PASSA_PATH}", binary=True)
    d = pathlib.Path(tempfile.mkdtemp(prefix="m3_old_"))
    f = d / "old_passa.py"
    f.write_bytes(src)
    spec = importlib.util.spec_from_file_location("old_passa", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_tree(repo, tree, path):
    blob = _git(repo, "show", f"{tree}:{path}", binary=True)
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"R11 ABORT: {path} at {tree[:12]} is not UTF-8 ({e})")


def scope(mod, text, off, det):
    """Always the PRODUCTION window. The parity guard governs this, and
    the local is named `head` because the guard's leg A reads the
    argument EXPRESSION -- a call spelled `_scope_of(text[:HEAD_BYTES])`
    is the divergence the guard exists to catch, even when it happens to
    be correct."""
    head = text[:mod.HEAD_BYTES]
    return mod._scope_of(head, off, det)


# ── synthetic documents ───────────────────────────────────────────────
# Every one is built here, in full, so the structure being tested is
# visible rather than described.

POS_MINIMAL = ("# Doc\n\n"
               "**Status:** COMPLETE — everything shipped (2026-01-02)\n"
               "**Last updated:** 2026-01-03\n\n"
               "## Section\n\nbody\n")

NEG_NO_ANCHOR = ("# Doc\n\n"
                 "**Status:** COMPLETE — 2026-01-01\n"
                 "**Owner:** somebody\n\n"
                 "## Section\n\nbody\n")

NEG_OWNER_SHARES_BLOCK = ("# Doc\n\n"
                          "**Owner:** Alice since 2025-01-01\n"
                          "**Last updated:** 2026-01-03\n\n"
                          "## Section\n\nbody\n")

NEG_UNDER_H2 = ("# Doc\n\n**Last updated:** 2026-01-03\n\n"
                "## Phase A\n\n**Status:** shipped 2026-01-02\n")

NEG_ANCHOR_REPEATED = ("# Doc\n\n"
                       "**Status:** COMPLETE — 2026-01-02\n"
                       "**Date:** 2026-01-03\n\n"
                       "entry\n\n**Date:** 2026-01-04\n\n"
                       "## Section\n\nbody\n")

NEG_ANCHOR_OTHER_BLOCK = ("# Doc\n\n"
                          "**Status:** COMPLETE — 2026-01-02\n\n"
                          "**Last updated:** 2026-01-03\n\n"
                          "## Section\n\nbody\n")

NEG_LIST_ENTRY = ("# Doc\n\n"
                  "- **Status:** COMPLETE — 2026-01-02\n"
                  "- **Last updated:** 2026-01-03\n\n"
                  "## Section\n\nbody\n")

NEG_PROSE_IN_BLOCK = ("# Doc\n\n"
                      "**Status:** COMPLETE — 2026-01-02\n"
                      "**Last updated:** 2026-01-03\n"
                      "This paragraph continues the block.\n\n"
                      "## Section\n\nbody\n")

NEG_TABLE_ROW = ("# Doc\n\n"
                 "| **Status:** shipped 2026-01-02 | **Last updated:** x |\n"
                 "**Last updated:** 2026-01-03\n\n"
                 "## Section\n\nbody\n")

NEG_PERSON_SUBJECT = ("# Doc\n\n"
                      "**Status:** blocked by @alice since 2026-01-02\n"
                      "**Last updated:** 2026-01-03\n\n"
                      "## Section\n\nbody\n")

NEG_ARTEFACT_SUBJECT = ("# Doc\n\n"
                        "**Status:** waiting on PLAN.md as of 2026-01-02\n"
                        "**Last updated:** 2026-01-03\n\n"
                        "## Section\n\nbody\n")

# Cycle 1-5 positives and negatives. These are REGRESSION controls: the
# cycle-6 change must not move any of them.
REG_LASTUPDATED = "# Doc\n\n**Last updated:** 2026-01-03\n\n## S\n\nbody\n"
REG_REVIEWED = "# Doc\n\n**Reviewed:** 2026-01-03\n\n## S\n\nbody\n"
REG_CREATED = "# Doc\n\n**Created:** 2026-01-03\n\n## S\n\nbody\n"
REG_H1_DATE = "# Report 2026-01-03\n\nbody\n\n## S\n\nbody\n"
REG_SELF_SUBJECT = ("# Doc\n\nThis document was measured at 2026-01-03.\n\n"
                    "## S\n\nbody\n")
REG_LIFECYCLE = "# Doc\n\nClosed 2026-01-03 by the operator.\n\n## S\n\nx\n"
REG_BARE_DATELINE = "# Doc\n\n**Version:** 3\n\n2026-01-03\n\n## S\n\nx\n"
REG_REPEATED_LABEL = ("# Doc\n\n**Date:** 2026-01-03\n\nx\n\n"
                      "**Date:** 2026-01-04\n\n## S\n\nx\n")
REG_GENERIC_UNIQUE = "# Doc\n\n**Branch:** main at 2026-01-03\n\n## S\n\nx\n"
REG_RUN_CONTINUATION = ("# Doc\n\n**Runs:** the deployed set is\n"
                        "31568526480 / `189500b`.\n\n## S\n\nx\n")
REG_BARE_HEX = "# Doc\n\n**Version:** 3\n\nc3c7731a\n\n## S\n\nx\n"
REG_BARE_RUN = "# Doc\n\n**Version:** 3\n\n31568526480\n\n## S\n\nx\n"


def at(text, needle, offset=0):
    i = text.index(needle)
    if i < 0:
        raise SystemExit(f"R11 ABORT: {needle!r} absent from the fixture")
    return i + offset


def value_starts(text, label="**Status:**"):
    """Every occurrence of a label, and the offset of its VALUE."""
    out, off = [], 0
    for i, ln in enumerate(text.split("\n")):
        j = ln.find(label)
        if j >= 0:
            out.append((i + 1, off + j + len(label) + 1, ln.strip()))
        off += len(ln) + 1
    return out


def synthetic_controls(old, new):
    print("\nSYNTHETIC — the contextual class, condition by condition")
    print("  Each negative removes exactly ONE condition from the positive.")
    for name, doc, needle, want in (
        ("POSITIVE minimal root block, anchor earned",
         POS_MINIMAL, "2026-01-02", "WHOLE_FILE"),
        ("C3 no independently earned anchor in the block",
         NEG_NO_ANCHOR, "2026-01-01", "SPAN"),
        ("C3 anchor repeated, so not independently earned",
         NEG_ANCHOR_REPEATED, "2026-01-02", "SPAN"),
        ("C3 anchor present but in a DIFFERENT block",
         NEG_ANCHOR_OTHER_BLOCK, "2026-01-02", "SPAN"),
        ("C4 the Status is a LIST ENTRY, not a root field",
         NEG_LIST_ENTRY, "2026-01-02", "SPAN"),
        ("C4 prose shares the block, so it is not metadata",
         NEG_PROSE_IN_BLOCK, "2026-01-02", "SPAN"),
        ("C1 the Status sits under an H2",
         NEG_UNDER_H2, "2026-01-02", "SPAN"),
        ("C2 the Status is inside a table row",
         NEG_TABLE_ROW, "2026-01-02", "SPAN"),
        ("C5 the subject is a PERSON, not the document",
         NEG_PERSON_SUBJECT, "2026-01-02", "SPAN"),
        ("C5 the subject is another ARTEFACT",
         NEG_ARTEFACT_SUBJECT, "2026-01-02", "SPAN"),
    ):
        check(name, scope(new, doc, at(doc, needle), "DATE"), want)
        check(f"  ^ predecessor said SPAN, so the class is genuinely new",
              scope(old, doc, at(doc, needle), "DATE"), "SPAN")

    print("\n  Kai's named synthetic negatives, verbatim shapes")
    check("root 'Owner: Alice since 2025-01-01' shares the block",
          scope(new, NEG_OWNER_SHARES_BLOCK,
                at(NEG_OWNER_SHARES_BLOCK, "2025-01-01"), "DATE"), "SPAN")
    check("root 'Status: COMPLETE - 2026-01-01', no binding",
          scope(new, NEG_NO_ANCHOR, at(NEG_NO_ANCHOR, "2026-01-01"), "DATE"),
          "SPAN")
    check("'## Phase A' then 'Status:' under the H2",
          scope(new, NEG_UNDER_H2, at(NEG_UNDER_H2, "2026-01-02"), "DATE"),
          "SPAN")

    print("\nREGRESSION — cycles 1-5 answers must not move")
    for name, doc, needle, det, want in (
        ("root Last updated", REG_LASTUPDATED, "2026-01-03", "DATE",
         "WHOLE_FILE"),
        ("root Reviewed", REG_REVIEWED, "2026-01-03", "DATE", "WHOLE_FILE"),
        ("root Created", REG_CREATED, "2026-01-03", "DATE", "WHOLE_FILE"),
        ("H1 line date", REG_H1_DATE, "2026-01-03", "DATE", "WHOLE_FILE"),
        ("self-subject sentence", REG_SELF_SUBJECT, "2026-01-03", "DATE",
         "WHOLE_FILE"),
        ("root lifecycle dateline", REG_LIFECYCLE, "2026-01-03", "DATE",
         "WHOLE_FILE"),
        ("bare dateline after other root metadata", REG_BARE_DATELINE,
         "2026-01-03", "DATE", "WHOLE_FILE"),
        ("repeated per-entry label", REG_REPEATED_LABEL, "2026-01-03",
         "DATE", "SPAN"),
        ("generic unique root label (Branch)", REG_GENERIC_UNIQUE,
         "2026-01-03", "DATE", "SPAN"),
        ("wrapped run-list continuation", REG_RUN_CONTINUATION,
         "31568526480", "DECIMAL_RUN", "SPAN"),
        ("bare HEX on a root line", REG_BARE_HEX, "c3c7731a", "HEX", "SPAN"),
        ("bare DECIMAL_RUN on a root line", REG_BARE_RUN, "31568526480",
         "DECIMAL_RUN", "SPAN"),
    ):
        got_new = scope(new, doc, at(doc, needle), det)
        got_old = scope(old, doc, at(doc, needle), det)
        check(name, got_new, want)
        check(f"  ^ unchanged from the predecessor", got_new, got_old,
              "cycle 6 moved a pre-existing answer")


def corpus_controls(old, new, repo, tree):
    print("\nCORPUS — Kai's named control sites, read from the frozen tree")
    print("  Probe = the first character of the Status VALUE, DATE detector.")
    print("  UNIVERSE: three documents. This is NOT the 492 evaluation.\n")

    target = "kai-pm/PHASE_0_5_BACKLOG.md"
    files = (target,
             "kai-pm/PHASE1_READINESS.md",
             "kai-pm/STRATEGIC_PLAN.md")
    texts = {p: read_tree(repo, tree, p) for p in files}

    print("  FAIL-OLD / PASS-NEW — the demonstrated case")
    t = texts[target]
    off, end = 273, 283
    check("PHASE_0_5_BACKLOG 273:283 DATE  predecessor",
          scope(old, t, off, "DATE"), "SPAN")
    check("PHASE_0_5_BACKLOG 273:283 DATE  cycle 6",
          scope(new, t, off, "DATE"), "WHOLE_FILE")
    print(f"        matched text {t[off:end]!r}  "
          f"line {t[:off].count(chr(10)) + 1}")

    print("\n  EVERY '**Status:**' site in the three documents")
    for p in files:
        t = texts[p]
        for line_no, voff, raw in value_starts(t):
            demonstrated = (p == target and line_no == 6)
            want = "WHOLE_FILE" if demonstrated else "SPAN"
            name = f"{p.split('/')[-1]} L{line_no}"
            check(f"{name:<28} {raw[:26]:<28}",
                  scope(new, t, voff, "DATE"), want)

    print("\n  CONFINEMENT — old vs new over every byte of the three "
          "documents")
    print("  (the same three documents; not a corpus figure)")
    total_diff = 0
    for p in files:
        t = texts[p]
        head = t[:NEW.HEAD_BYTES]
        diffs = []
        for i in range(len(head)):
            for det in ("DATE", "HEX", "DECIMAL_RUN"):
                a = old._scope_of(head, i, det)
                b = new._scope_of(head, i, det)
                if a != b:
                    diffs.append((i, det, a, b))
        total_diff += len(diffs)
        lines = sorted({head[:i].count("\n") + 1 for i, *_ in diffs})
        print(f"    {p.split('/')[-1]:<26} differing (offset,detector) "
              f"pairs: {len(diffs):<5} lines: {lines}")
    check("all divergence confined to PHASE_0_5_BACKLOG L6",
          "yes" if total_diff > 0 else "no", "yes",
          "the change must be demonstrable somewhere")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--old-ref", default=CYCLE5)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    a = ap.parse_args()

    old = load_old(a.repo, a.old_ref)
    print(f"PREDECESSOR   {a.old_ref[:12]} loaded from the git object, "
          f"executed, not paraphrased")
    print(f"CANDIDATE     working tree {PKG.name}/passa.py")
    print(f"CONTEXTUAL    {len(NEW.CONTEXTUAL_PREDICATES)} predicate(s): "
          f"{', '.join(sorted(NEW.CONTEXTUAL_PREDICATES))}")
    print(f"INTRINSIC     {len(NEW.BINDING_PREDICATES)} predicates, "
          f"unchanged: {len(NEW.BINDING_PREDICATES) == len(old.BINDING_PREDICATES)}")
    if set(NEW.BINDING_PREDICATES) != set(old.BINDING_PREDICATES):
        raise SystemExit("R11 ABORT: BINDING_PREDICATES changed. Cycle 6 "
                         "authorises no broad predicate expansion.")

    synthetic_controls(old, NEW)
    corpus_controls(old, NEW, a.subject_repo, a.tree)

    print(f"\nCONTROLS: {'ALL PASS' if not FAILED else 'FAILED — ' + ', '.join(FAILED)}")
    raise SystemExit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
