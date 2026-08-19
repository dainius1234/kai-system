#!/usr/bin/env python3
"""Separate what a build STEP IS from what it PRINTED, structurally.

WHY THIS REPLACES GREPPING A RENDERED LOG
=========================================

Item 8's verdicts rested on searching BuildKit's human-rendered output
for strings. Three successive reviews found the same class of defect
under three different disguises, and the last one names it exactly:

> *"the parser is still capable of measuring BuildKit's presentation of
> the command instead of execution of the command."*

Every marker we searched for — `retrying in`, `REFUSING TO BUILD`, an
injected `echo` line — **exists literally in the Dockerfile source**, and
BuildKit's rendered progress interleaves instruction text with container
output. Each round we made the string harder to forge; none of them
changed the fact that instruction and output arrived in the same stream.

`--progress=rawjson` emits BuildKit's `SolveStatus` events, in which the
two are **different fields of different objects**:

    vertexes[].name    INSTRUCTION METADATA. The text of the step. This
                       is what BuildKit would echo back, and it is NEVER
                       evidence that anything ran.
    vertexes[].cached  whether the step was served from cache.
    vertexes[].error   the step's own error, attributable to that step.
    logs[].data        RUNTIME OUTPUT, base64, tagged with the vertex
                       that produced it. Only execution creates this.

So "did the fetch retry five times" becomes a question about
`logs` belonging to one vertex, and it can no longer be answered by the
instruction that would have retried. That is a structural separation
rather than a cleverer pattern, which is why it ends the sequence.

IT ALSO REMOVES INSTRUMENTATION FROM THE SUBJECT
================================================

The previous repair added `ITEM8-MARK ATTEMPT=$attempt` to all three
derived Dockerfiles as scaffolding, and needed an argument about whether
scaffolding counts against the frozen mutation cardinality. With runtime
logs attributable per vertex, the Dockerfiles' OWN retry lines are
sufficient — they are runtime output now, not searchable prose — so the
markers are gone and B1's treatment count returns to a clean zero.

The argument is not won. It is unnecessary.

WHAT IT REFUSES
===============

R11 at the parse boundary: unparseable events, a missing target vertex,
and a target matched by more than one vertex are all refusals. A parser
that silently returns "no retries found" for a log it could not read is
the `/dev/null` with better manners that R10 exists to stop.

Exit 0 = the target vertex was located and its facts emitted.
Exit 1 = refused, with the unmet prerequisite named.
"""
from __future__ import annotations

import argparse
import base64
import json
import pathlib
import sys


class Vertex:
    __slots__ = ("digest", "name", "cached", "started", "completed",
                 "error", "log")

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.name = ""
        self.cached = False
        self.started = None
        self.completed = None
        self.error = ""
        self.log: list[str] = []

    def as_dict(self) -> dict:
        return {
            "digest": self.digest,
            "name": self.name,
            "cached": bool(self.cached),
            "executed": bool(self.started),
            "completed": bool(self.completed),
            "error": self.error,
            "runtime_log_lines": len(self.log),
        }


def parse(text: str) -> tuple[dict[str, Vertex], str]:
    """Every vertex in the stream, with its RUNTIME output attached."""
    vertices: dict[str, Vertex] = {}
    seen_any = False
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            return {}, (f"line {lineno} of the event stream is not JSON "
                        f"({e}). A partially-read event stream cannot be "
                        f"distinguished from a build that printed nothing")
        if not isinstance(ev, dict):
            continue
        seen_any = True
        for v in ev.get("vertexes") or []:
            d = v.get("digest")
            if not d:
                continue
            vx = vertices.setdefault(d, Vertex(d))
            # A later event may complete a vertex announced earlier.
            if v.get("name"):
                vx.name = v["name"]
            vx.cached = bool(v.get("cached", vx.cached))
            vx.started = v.get("started") or vx.started
            vx.completed = v.get("completed") or vx.completed
            if v.get("error"):
                vx.error = v["error"]
        for lg in ev.get("logs") or []:
            d = lg.get("vertex")
            if not d:
                continue
            vx = vertices.setdefault(d, Vertex(d))
            data = lg.get("data") or ""
            try:
                vx.log.append(base64.b64decode(data).decode("utf-8", "replace"))
            except Exception:
                # Undecodable payload is a fact, not something to drop.
                vx.log.append(f"<UNDECODABLE LOG PAYLOAD: {data[:40]!r}>")
    if not seen_any:
        return {}, ("the event stream held no BuildKit events at all. "
                    "Either the build produced none or --progress=rawjson "
                    "was not in effect; neither licenses a conclusion")
    return vertices, ""


def find_target(vertices: dict[str, Vertex], needle: str
                ) -> tuple[Vertex | None, str]:
    """The ONE vertex whose INSTRUCTION TEXT contains `needle`.

    Matching on the name is correct here and only here: we are asking
    *which step is the subject*, not *what happened*. What happened is
    read from that vertex's runtime log, which the name cannot forge.
    """
    hits = [v for v in vertices.values() if needle in v.name]
    if not hits:
        return None, (f"no vertex's instruction text contains {needle!r}. "
                      f"The target step is not in this build, so nothing "
                      f"about it can be measured")
    if len(hits) > 1:
        return None, (f"{len(hits)} vertices contain {needle!r}; the target "
                      f"must be unambiguous or the evidence belongs to an "
                      f"unknown step")
    return hits[0], ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", required=True)
    ap.add_argument("--target-substring", required=True)
    ap.add_argument("--count", action="append", default=[], metavar="STRING",
                    help="count occurrences of STRING in the target's "
                         "RUNTIME log; repeatable")
    ap.add_argument("--emit-log", metavar="PATH",
                    help="write ONLY the target vertex's runtime output "
                         "here, so a caller may search it without ever "
                         "touching instruction text")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.events)
    if not path.is_file():
        print(f"REFUSED: {path} does not exist. There is no event stream "
              f"to read, and an absent stream is not an empty one.")
        return 1

    vertices, err = parse(path.read_text())
    if err:
        print(f"REFUSED: {err}.")
        return 1
    target, err = find_target(vertices, args.target_substring)
    if err:
        print(f"REFUSED: {err}.")
        return 1

    runtime = "".join(target.log)
    facts = target.as_dict()
    facts["counts"] = {s: runtime.count(s) for s in args.count}
    facts["vertices_total"] = len(vertices)

    if args.emit_log:
        # RUNTIME ONLY. A caller grepping this file cannot match a
        # Dockerfile instruction, because no instruction text is in it.
        pathlib.Path(args.emit_log).write_text(runtime)
        facts["runtime_log_path"] = args.emit_log
        facts["runtime_log_bytes"] = len(runtime.encode())

    if args.json:
        print(json.dumps(facts))
        return 0

    print("BUILDKIT TARGET VERTEX")
    print("=" * 68)
    print(f"  digest   : {target.digest}")
    print(f"  name     : {target.name[:100]}")
    print(f"  executed : {facts['executed']}   cached: {facts['cached']}")
    print(f"  error    : {target.error[:120] or '(none)'}")
    print(f"  runtime  : {len(target.log)} log chunk(s)")
    for s, n in facts["counts"].items():
        print(f"    {n:>3} x {s!r}  (RUNTIME output only)")
    print()
    print(f"  inspected: 1 target vertex of {len(vertices)} in the stream")
    print()
    print("  Counts are over RUNTIME OUTPUT attributed to this vertex.")
    print("  The instruction text above is metadata and is never counted,")
    print("  which is what stops presentation being read as execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
