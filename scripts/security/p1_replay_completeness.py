#!/usr/bin/env python3
"""P1 — is the captured request complete enough to replay it faithfully?

D247 froze a two-stage ownership-separation experiment whose Stage 1 is
an *exact-request replay*. Before that replay can mean anything, one
question has to be answered with evidence rather than confidence:

    does the capture actually hold the whole model-facing invocation?

`probe_llm_contract.py` records five arguments **with their values**
(`messages`, `model`, `temperature`, `response_format`, `tools`) and
records every other keyword argument as a **name only**. So "we replayed
the exact request" is a claim about two independent axes, and each needs
its own evidence:

  A. keyword arguments — is `other_params` empty for every relevant
     production row? Measured from the artifact. Empirical.
  B. positional arguments — did the production call path supply any?
     Two sources, and neither may stand in for the other: the call
     path's **source**, which says what the code can do, and the
     probe's recorded `positional_arg_count`, which says what actually
     arrived. Before D250 the probe recorded nothing here at all, and a
     capture showing `other_params == []` said NOTHING about this axis —
     which is exactly the substitution this instrument refuses.

This script measures both and refuses to collapse them. It emits exactly
one of five verdicts, per the frozen procedure:

    REQUEST_REPLAYABLE            both axes clean, fields reconstructable
    REQUEST_INCOMPLETE_KWARGS     unrecorded extra kwarg values exist
    REQUEST_INCOMPLETE_POSITIONAL positional completeness not established
    REQUEST_INCOMPLETE_MULTIPLE   both defects
    UNRESOLVED                    the artifact/path cannot establish it

There is deliberately no "probably complete". A replay that silently
omitted one argument would give a beautifully controlled experiment
answering the wrong question.

--- the certified capture boundary -------------------------------------

This script certifies completeness at the **client-callable boundary**:
the arguments passed to `Completions.create`, which is where the probe's
wrapper sits. It does NOT certify the serialized HTTP body. The OpenAI
client may default, normalize or re-serialize fields after this
boundary, so a Stage 1 replay built on this evidence may claim

    "replay of the complete model-facing client invocation"

and may NOT claim "byte-identical wire request". Claiming the latter
needs evidence from the HTTP/wire boundary, which this instrument does
not observe and does not pretend to.

--- what a certifiable capture must contain (D250) ----------------------

P1 run 2 refused run 24's capture because one line of it was the probe's
own prose. The repair was not a whitelist for that line; it was a stream
that structurally cannot hold prose, plus three further requirements:

  * **the data stream is machine rows only** — one JSON object per line,
    each naming its `event`. A line that is not one is a hole in the
    denominator, whatever it says;
  * **a `capture-manifest` row declares the production-call count**,
    counted from the wrapper's own entry counter rather than by counting
    the rows it wrote. Declared must equal parsed, or the census
    refuses. Same instrument, so this is internal corroboration and NOT
    independent proof — but a lost or truncated row now shows up as a
    mismatch instead of shrinking the denominator along with the
    evidence;
  * **`args_state` gives every model-facing key three states** —
    `ABSENT` / `NULL` / `VALUE`. `"temperature": None` alone cannot say
    whether the caller omitted the key or passed None, and at the
    client-invocation boundary those are not the same claim. A replay
    rule of "if None, omit it" is an inference presented as a record;
  * **`positional_arg_count` is recorded**, so axis B has a runtime
    record as well as source analysis. If positional arguments ever
    appear and their values were not recorded, the census refuses rather
    than inferring replayability from an absence of evidence.

Captures written before this format keep their historical meaning and
are **not** retroactively upgraded: with no manifest and no `args_state`
they are simply not certifiable under this standard, and the census says
so rather than reading them as clean.

Usage:

    p1_replay_completeness.py --capture capture.jsonl \\
        --call-site-root /path/to/site-packages/cognee \\
        --forwarder-root /path/to/site-packages/instructor
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

# The verdicts. Frozen by the operator's P1 procedure; do not add a sixth
# without a decision entry, and never collapse two into one.
REPLAYABLE = "REQUEST_REPLAYABLE"
INCOMPLETE_KWARGS = "REQUEST_INCOMPLETE_KWARGS"
INCOMPLETE_POSITIONAL = "REQUEST_INCOMPLETE_POSITIONAL"
INCOMPLETE_MULTIPLE = "REQUEST_INCOMPLETE_MULTIPLE"
UNRESOLVED = "UNRESOLVED"

EXIT = {
    REPLAYABLE: 0,
    UNRESOLVED: 2,
    INCOMPLETE_KWARGS: 3,
    INCOMPLETE_POSITIONAL: 4,
    INCOMPLETE_MULTIPLE: 5,
}

# The keys `probe_llm_contract.py` records WITH their values. Anything
# else is a name in `other_params` and its value is gone. This tuple is
# the instrument's own contract, so it is asserted against the probe by
# the calibration suite rather than trusted here.
VALUED_KEYS = ("messages", "model", "temperature", "response_format", "tools")

# Without these two a row does not describe a request at all.
REQUIRED_FOR_REPLAY = ("messages", "model")

# The production phase the probe stamps on rows driven through the real
# stack. Selftest rows exercise a stand-in and are NOT the replay
# population — counting them would be R11's failure in miniature: rows
# that look like measurements of a subject that was never the subject.
PRODUCTION_PHASE = "capture"


# ---------------------------------------------------------------- axis A


def read_rows(path: pathlib.Path) -> tuple[list[dict], list[str], dict | None]:
    """Every `llm-call` row, the parse complaints, and the manifest.

    A JSON line that is not an object, or an object with no `event`, is a
    malformed machine row and is a complaint too — the stream's contract
    is "one JSON object per line, each naming its event", and a line that
    breaks it is as much a hole in the denominator as prose is.
    """
    rows: list[dict] = []
    notes: list[str] = []
    manifest: dict | None = None
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # R10: a line we could not read is reported, not dropped —
            # and reported with its CONTENT, because "1 line did not
            # parse" is unactionable while the line itself is not. The
            # excerpt declares its own size so it cannot read as whole.
            head = line[:400].replace("\t", " ")
            notes.append(
                f"line {n}: unparseable ({exc.msg}); {len(line)} bytes; "
                f"first {min(400, len(line))}: {head!r}")
            continue
        if not isinstance(row, dict):
            notes.append(
                f"line {n}: parsed as {type(row).__name__}, not a JSON "
                f"object; {len(line)} bytes")
            continue
        if "event" not in row:
            notes.append(
                f"line {n}: JSON object with no `event` key, so it names "
                f"no record type; {len(line)} bytes; keys: "
                f"{sorted(row)[:8]}")
            continue
        if row["event"] == "capture-manifest":
            manifest = row
        elif row["event"] == "llm-call":
            rows.append(row)
    return rows, notes, manifest


def audit_kwargs(rows: list[dict], notes: list[str] | None = None,
                 manifest: dict | None = None) -> dict:
    """Axis A over the production population."""
    population = [r for r in rows if r.get("phase") == PRODUCTION_PHASE]
    offenders = [r for r in population if r.get("other_params")]
    unreconstructable = [
        r for r in population
        if any(r.get(k) is None for k in REQUIRED_FOR_REPLAY)
    ]
    # THREE STATES, COUNTED. A row that records only the value cannot say
    # whether a key was absent or explicitly None, so it is counted as
    # legacy rather than quietly read as one or the other.
    legacy = [r for r in population if not isinstance(r.get("args_state"), dict)]
    presence: dict = {k: {"ABSENT": 0, "NULL": 0, "VALUE": 0, "UNRECORDED": 0}
                      for k in VALUED_KEYS}
    for r in population:
        st = r.get("args_state") if isinstance(r.get("args_state"), dict) else {}
        for k in VALUED_KEYS:
            presence[k][st.get(k, "UNRECORDED")] = \
                presence[k].get(st.get(k, "UNRECORDED"), 0) + 1
    # Axis B as RECORDED at runtime, which is a different source from the
    # source-code analysis and may not be substituted for it.
    pos_rows = [r for r in population if r.get("positional_arg_count")]
    pos_unrecorded = [r for r in population
                      if r.get("positional_arg_count") is None]
    pos_unreconstructable = [
        r for r in pos_rows
        if len(r.get("positional_args") or []) != r.get("positional_arg_count")
    ]
    declared = (manifest or {}).get("declared_production_calls")
    return {
        "parse_notes": list(notes or ()),
        "manifest": manifest,
        "declared": declared,
        "rows_total": len(rows),
        "population": len(population),
        "offenders": offenders,
        "extra_names": sorted({
            name for r in offenders for name in (r.get("other_params") or [])
        }),
        "unreconstructable": len(unreconstructable),
        "legacy_rows": len(legacy),
        "presence": presence,
        "positional_rows": len(pos_rows),
        "positional_unrecorded": len(pos_unrecorded),
        "positional_unreconstructable": len(pos_unreconstructable),
    }


# ---------------------------------------------------------------- axis B


class CreateCallVisitor(ast.NodeVisitor):
    """Find `<...>.chat.completions.create(...)` calls and count their
    positional arguments.

    The scope is every file under the root, derived by walking the tree
    (R5) — never a hand-written list of call sites beside the check. A
    superset of the production path is the safe direction here: if no
    call site anywhere in the package passes a positional argument, the
    production one does not either.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.sites: list[dict] = []

    @staticmethod
    def _is_create(node: ast.Call) -> bool:
        f = node.func
        if not isinstance(f, ast.Attribute) or f.attr != "create":
            return False
        completions = f.value
        if not isinstance(completions, ast.Attribute):
            return False
        if completions.attr != "completions":
            return False
        chat = completions.value
        return isinstance(chat, ast.Attribute) and chat.attr == "chat"

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if self._is_create(node):
            self.sites.append({
                "file": self.path,
                "line": node.lineno,
                "positional": len(node.args),
                "starred": any(isinstance(a, ast.Starred) for a in node.args),
            })
        self.generic_visit(node)


class ForwarderVisitor(ast.NodeVisitor):
    """Find `func(*args, **kwargs)`-shaped forwarding calls.

    Instructor's retry layer is the last hop before the client callable.
    If it inserted a positional argument of its own, axis B would be
    unestablished even with a clean call site — so the forwarder is
    checked separately rather than assumed transparent.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.sites: list[dict] = []

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id == "func":
            literal = [a for a in node.args if not isinstance(a, ast.Starred)]
            self.sites.append({
                "file": self.path,
                "line": node.lineno,
                "literal_positional": len(literal),
            })
        self.generic_visit(node)


def walk_python(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def audit_positional(call_root: pathlib.Path | None,
                     fwd_root: pathlib.Path | None) -> dict:
    """Axis B. Absence of source is 'not established', never 'clean'."""
    result: dict = {
        "established": False,
        "reason": "",
        "files_scanned": 0,
        "call_sites": [],
        "forwarder_sites": [],
        "unparseable": [],
    }
    if call_root is None or not call_root.exists():
        result["reason"] = (
            "no call-path source supplied or path does not exist, so "
            "positional completeness was NOT established. The capture "
            "cannot answer this axis by construction.")
        return result

    sites: list[dict] = []
    scanned = 0
    for path in walk_python(call_root):
        try:
            tree = ast.parse(path.read_text(errors="replace"), filename=str(path))
        except SyntaxError as exc:
            # I-1: an unreadable file is a hole in the denominator and is
            # reported as one, not silently skipped into a clean result.
            result["unparseable"].append(f"{path}: {exc}")
            continue
        scanned += 1
        v = CreateCallVisitor(str(path))
        v.visit(tree)
        sites.extend(v.sites)
    result["files_scanned"] = scanned
    result["call_sites"] = sites

    fwd: list[dict] = []
    if fwd_root is not None and fwd_root.exists():
        for path in walk_python(fwd_root):
            if path.name not in ("retry.py", "patch.py"):
                continue
            try:
                tree = ast.parse(path.read_text(errors="replace"),
                                 filename=str(path))
            except SyntaxError as exc:
                result["unparseable"].append(f"{path}: {exc}")
                continue
            v = ForwarderVisitor(str(path))
            v.visit(tree)
            fwd.extend(v.sites)
    result["forwarder_sites"] = fwd

    if result["unparseable"]:
        result["reason"] = (
            f"{len(result['unparseable'])} source file(s) could not be "
            "parsed, so the call-site denominator is incomplete")
        return result
    if not sites:
        result["reason"] = (
            "no `chat.completions.create` call site was found under the "
            "supplied root, so this is not the production call path")
        return result
    dirty = [s for s in sites if s["positional"] or s["starred"]]
    if dirty:
        result["reason"] = (
            f"{len(dirty)} of {len(sites)} call site(s) pass positional "
            "arguments")
        return result
    if fwd_root is not None and not fwd:
        result["reason"] = (
            "forwarder source supplied but no `func(...)` forwarding call "
            "was found, so the last hop is unverified")
        return result
    fwd_dirty = [s for s in fwd if s["literal_positional"]]
    if fwd_dirty:
        result["reason"] = (
            f"{len(fwd_dirty)} forwarding call(s) insert a literal "
            "positional argument")
        return result

    result["established"] = True
    result["reason"] = (
        f"{len(sites)} call site(s) across {scanned} file(s) pass zero "
        f"positional arguments; {len(fwd)} forwarding call(s) insert none")
    return result


# ----------------------------------------------------------------- verdict


def classify(kw: dict | None, pos: dict) -> tuple[str, list[str]]:
    why: list[str] = []
    if kw is None:
        return UNRESOLVED, ["the capture could not be read at all"]
    if kw["parse_notes"]:
        # A line that did not parse, or parsed into something that is not
        # a record, is a hole in the denominator — and a denominator with
        # a hole cannot certify anything. It might be the `llm-call` row
        # carrying the very kwarg that breaks axis A, and "it probably is
        # not" is exactly the reasoning P1 exists to refuse.
        return UNRESOLVED, [
            f"{len(kw['parse_notes'])} capture line(s) are not valid "
            f"machine rows, so the production population of "
            f"{kw['population']} is a LOWER BOUND, not a certified "
            "denominator"] + kw["parse_notes"]
    if kw["manifest"] is None:
        # No manifest: either a legacy capture, or a run that died before
        # declaring. Both mean the same thing here — nothing independent
        # of the rows says how many there should be.
        return UNRESOLVED, [
            "the capture declares no `capture-manifest` row, so nothing "
            "states how many production calls there SHOULD be. A capture "
            "written before the manifest existed keeps its historical "
            "meaning and is simply not certifiable under this standard; "
            "a truncated one is indistinguishable from it here, which is "
            "why neither is upgraded"]
    if kw["declared"] != kw["population"]:
        # THE reconciliation. Two counters from different places in the
        # probe: the wrapper's entry counter, and the rows that reached
        # the file. Same instrument, so this is internal corroboration
        # and not independent proof — but a dropped, truncated or
        # never-written row shows up as a mismatch instead of quietly
        # shrinking the denominator along with the evidence.
        return UNRESOLVED, [
            f"reconciliation FAILED: the capture declares "
            f"{kw['declared']} production call(s) and "
            f"{kw['population']} row(s) were parsed. One of them is "
            "wrong, and until that is resolved neither is a denominator"]
    if kw["population"] == 0:
        return UNRESOLVED, [
            f"no production rows (phase={PRODUCTION_PHASE!r}) in the "
            f"capture; {kw['rows_total']} llm-call row(s) total"]
    if kw["legacy_rows"]:
        return UNRESOLVED, [
            f"{kw['legacy_rows']} of {kw['population']} production row(s) "
            "record no `args_state`, so ABSENT and NULL are "
            "indistinguishable in them. A replay rule of 'if None, omit "
            "it' would be an inference presented as a record"]
    if kw["unreconstructable"]:
        return UNRESOLVED, [
            f"{kw['unreconstructable']} of {kw['population']} production "
            f"row(s) lack {' or '.join(REQUIRED_FOR_REPLAY)}, so the "
            "artifact cannot establish completeness"]

    kwargs_bad = bool(kw["offenders"])
    # Axis B now has TWO sources that must agree: the source analysis,
    # and what the wrapper recorded arriving at runtime. Neither may
    # stand in for the other — source says what the code can do, the
    # record says what happened.
    recorded_bad = bool(kw["positional_unrecorded"]
                        or kw["positional_unreconstructable"])
    pos_bad = (not pos["established"]) or recorded_bad
    if kwargs_bad:
        why.append(
            f"{len(kw['offenders'])} of {kw['population']} production "
            f"row(s) carry unrecorded kwarg value(s): "
            f"{', '.join(kw['extra_names'])}")
    if not pos["established"]:
        why.append(f"positional axis, source: {pos['reason']}")
    if kw["positional_unrecorded"]:
        why.append(
            f"positional axis, record: {kw['positional_unrecorded']} row(s) "
            "do not record a positional argument count at all")
    if kw["positional_unreconstructable"]:
        why.append(
            f"positional axis, record: {kw['positional_unreconstructable']} "
            "row(s) report positional arguments whose values were not "
            "recorded, so they cannot be replayed and must not be inferred")

    if kwargs_bad and pos_bad:
        return INCOMPLETE_MULTIPLE, why
    if kwargs_bad:
        return INCOMPLETE_KWARGS, why
    if pos_bad:
        return INCOMPLETE_POSITIONAL, why
    return REPLAYABLE, [
        f"{kw['population']} production row(s), every one with "
        f"other_params == []",
        f"reconciliation: {kw['declared']} declared == {kw['population']} "
        "parsed (internal corroboration, not independent proof)",
        f"presence recorded on all {kw['population']} row(s): ABSENT, NULL "
        "and VALUE are distinguished",
        f"positional axis, record: {kw['positional_rows']} row(s) with "
        "positional arguments",
        f"positional axis, source: {pos['reason']}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", required=True,
                    help="capture.jsonl from a production run")
    ap.add_argument("--call-site-root",
                    help="package root of the production call path "
                         "(e.g. site-packages/cognee)")
    ap.add_argument("--forwarder-root",
                    help="package root of the last forwarding hop "
                         "(e.g. site-packages/instructor)")
    args = ap.parse_args()

    print("P1 — REPLAY COMPLETENESS OF THE CAPTURED REQUEST")
    print("=" * 64)
    print("certified boundary: the client callable "
          "(Completions.create arguments).")
    print("NOT certified: the serialized HTTP body. A replay built on this")
    print("evidence may claim 'complete model-facing client invocation';")
    print("it may NOT claim 'byte-identical wire request'.")
    print()

    cap = pathlib.Path(args.capture)
    kw: dict | None = None
    if not cap.exists():
        # R11: no subject, no observation. Say which prerequisite failed.
        print(f"AXIS A: capture not found at {cap} — not measured.")
    else:
        rows, notes = read_rows(cap)
        for note in notes:
            print(f"  capture note: {note}")
        kw = audit_kwargs(rows, notes)
        print(f"AXIS A — keyword arguments, from the artifact")
        print(f"  llm-call rows            : {kw['rows_total']}")
        print(f"  production population    : {kw['population']} "
              f"(phase={PRODUCTION_PHASE!r})")
        print(f"  rows with extra kwargs   : {len(kw['offenders'])}")
        if kw["extra_names"]:
            print(f"  unrecorded kwarg names   : {', '.join(kw['extra_names'])}")
        print(f"  rows missing "
              f"{'/'.join(REQUIRED_FOR_REPLAY)}: {kw['unreconstructable']}")
        print(f"  declared by the capture   : {kw['declared']}"
              f"  (manifest present: {kw['manifest'] is not None})")
        print(f"  rows without args_state   : {kw['legacy_rows']}")
        print("  argument presence, three states (D250):")
        for k, counts in kw["presence"].items():
            print(f"      {k:<16} " + "  ".join(
                f"{s}={counts.get(s, 0)}"
                for s in ("VALUE", "NULL", "ABSENT", "UNRECORDED")))
        print(f"  positional, recorded      : "
              f"{kw['positional_rows']} row(s) with args, "
              f"{kw['positional_unrecorded']} row(s) with no count, "
              f"{kw['positional_unreconstructable']} unreconstructable")
    print()

    pos = audit_positional(
        pathlib.Path(args.call_site_root) if args.call_site_root else None,
        pathlib.Path(args.forwarder_root) if args.forwarder_root else None)
    print("AXIS B — positional arguments, from the call path's source")
    print("  (the capture is silent on this axis BY CONSTRUCTION: a clean")
    print("   axis A says nothing whatsoever about axis B)")
    print(f"  files scanned            : {pos['files_scanned']}")
    print(f"  create call sites        : {len(pos['call_sites'])}")
    for s in pos["call_sites"]:
        print(f"      {s['file']}:{s['line']} positional={s['positional']}"
              f"{' STARRED' if s['starred'] else ''}")
    print(f"  forwarding call sites    : {len(pos['forwarder_sites'])}")
    for s in pos["forwarder_sites"]:
        print(f"      {s['file']}:{s['line']} "
              f"literal_positional={s['literal_positional']}")
    for u in pos["unparseable"]:
        print(f"  UNPARSEABLE: {u}")
    print(f"  established              : {pos['established']}")
    print(f"  reason                   : {pos['reason']}")
    print()

    verdict, why = classify(kw, pos)
    print("=" * 64)
    # I-2. The denominator, always, including when it is zero — a verdict
    # whose population is invisible cannot be argued with.
    print(f"  inspected: {kw['population'] if kw else 0} production "
          f"request row(s) across 2 completeness axes")
    print(f"P1 VERDICT: {verdict}")
    for line in why:
        print(f"  - {line}")
    if verdict != REPLAYABLE:
        print()
        print("  Stage 1 and Stage 2 remain BLOCKED. Per D247/D248 the")
        print("  repair must record the missing VALUES, not merely their")
        print("  presence, and must carry its own calibration and mutation")
        print("  proof before it is used for ownership evidence.")
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
