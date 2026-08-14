#!/usr/bin/env python3
"""P1 — is the captured request complete enough to replay it faithfully?

D247 froze a two-stage ownership-separation experiment whose Stage 1 is
an *exact-request replay*. Before that replay can mean anything, one
question has to be answered with evidence rather than confidence:

    does the capture actually hold the whole model-facing invocation?

`probe_llm_contract.py` records five arguments **with their values**
(`messages`, `model`, `temperature`, `response_format`, `tools`), records
every other keyword argument as a **name only**, and does not record
positional arguments at all. So "we replayed the exact request" is a
claim about two independent axes, and each needs its own evidence:

  A. keyword arguments — is `other_params` empty for every relevant
     production row? Measured from the artifact. Empirical.
  B. positional arguments — did the production call path supply any?
     The capture is silent on this by construction, so a capture showing
     `other_params == []` says NOTHING about axis B. It is established
     from the call path's source, or it is not established at all.

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

One residual, recorded rather than hidden: the probe writes `None` for a
recorded key that was absent, so absent and explicitly-None are
indistinguishable in the record for the five valued keys. The certified
replay rule is therefore **omit any key recorded as None**. For these
five that is materially equivalent — the OpenAI client omits unset
optionals — but the ambiguity is real and is reported, not assumed away.

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


def read_rows(path: pathlib.Path) -> tuple[list[dict], list[str]]:
    """Every `llm-call` row in the capture, plus any parse complaints."""
    rows: list[dict] = []
    notes: list[str] = []
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # R10: a line we could not read is reported, not dropped.
            notes.append(f"line {n}: unparseable ({exc.msg})")
            continue
        if isinstance(row, dict) and row.get("event") == "llm-call":
            rows.append(row)
    return rows, notes


def audit_kwargs(rows: list[dict]) -> dict:
    """Axis A over the production population."""
    population = [r for r in rows if r.get("phase") == PRODUCTION_PHASE]
    offenders = [r for r in population if r.get("other_params")]
    unreconstructable = [
        r for r in population
        if any(r.get(k) is None for k in REQUIRED_FOR_REPLAY)
    ]
    ambiguous = {
        k: sum(1 for r in population if r.get(k) is None) for k in VALUED_KEYS
    }
    return {
        "rows_total": len(rows),
        "population": len(population),
        "offenders": offenders,
        "extra_names": sorted({
            name for r in offenders for name in (r.get("other_params") or [])
        }),
        "unreconstructable": len(unreconstructable),
        "null_ambiguous": ambiguous,
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
    if kw["population"] == 0:
        return UNRESOLVED, [
            f"no production rows (phase={PRODUCTION_PHASE!r}) in the "
            f"capture; {kw['rows_total']} llm-call row(s) total"]
    if kw["unreconstructable"]:
        return UNRESOLVED, [
            f"{kw['unreconstructable']} of {kw['population']} production "
            f"row(s) lack {' or '.join(REQUIRED_FOR_REPLAY)}, so the "
            "artifact cannot establish completeness"]

    kwargs_bad = bool(kw["offenders"])
    pos_bad = not pos["established"]
    if kwargs_bad:
        why.append(
            f"{len(kw['offenders'])} of {kw['population']} production "
            f"row(s) carry unrecorded kwarg value(s): "
            f"{', '.join(kw['extra_names'])}")
    if pos_bad:
        why.append(f"positional axis: {pos['reason']}")

    if kwargs_bad and pos_bad:
        return INCOMPLETE_MULTIPLE, why
    if kwargs_bad:
        return INCOMPLETE_KWARGS, why
    if pos_bad:
        return INCOMPLETE_POSITIONAL, why
    return REPLAYABLE, [
        f"{kw['population']} production row(s), every one with "
        f"other_params == []",
        f"positional axis: {pos['reason']}"]


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
        kw = audit_kwargs(rows)
        print(f"AXIS A — keyword arguments, from the artifact")
        print(f"  llm-call rows            : {kw['rows_total']}")
        print(f"  production population    : {kw['population']} "
              f"(phase={PRODUCTION_PHASE!r})")
        print(f"  rows with extra kwargs   : {len(kw['offenders'])}")
        if kw["extra_names"]:
            print(f"  unrecorded kwarg names   : {', '.join(kw['extra_names'])}")
        print(f"  rows missing "
              f"{'/'.join(REQUIRED_FOR_REPLAY)}: {kw['unreconstructable']}")
        print("  recorded-None (absent and explicit-None indistinguishable):")
        for k, n in kw["null_ambiguous"].items():
            print(f"      {k:<16} {n}/{kw['population']}")
        print("    replay rule: omit any key recorded as None.")
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
