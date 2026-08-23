#!/usr/bin/env python3
"""DOCUMENT REFERENCE GRAPH — Census v1.1.

P10 holds unchanged: CONTEXT MUST NOT BECOME SEMANTICS. Every occurrence
carries a context class; context is recorded, never used to drop an edge
or change its kind. That adjudication is a later, human step.

D341 F2 IS FIXED HERE. v1.0 declared seven edge contexts including
OTHER, but OTHER is a LINE classification that `edge_context` always
converts to INLINE_CODE or PROSE_PATH before emission. An exhaustive
input probe proved it structurally unreachable as an edge class. Rather
than invent an output to satisfy the enum, the two alphabets are now
declared separately and honestly:

  LINE_CONTEXTS  4 values -- an internal, per-line intermediate
  EDGE_CONTEXTS  6 values -- what is actually emitted on an edge

Kai's ruling: "Declare the actual six emitted edge contexts; do not
fabricate a seventh output."
"""
from __future__ import annotations
import collections
import pathlib
import re
import subprocess

LINK = re.compile(r"\[[^\]]*\]\(\s*([^)\s]+)")

ALPHABETS = {
    "KINDS": ("MARKDOWN_LINK", "EXPLICIT_PATH_REFERENCE", "BASENAME_UNIQUE",
              "BASENAME_AMBIGUOUS", "BROKEN_LINK"),
    "EDGE_CONTEXTS": ("MARKDOWN_LINK", "PROSE_PATH", "INLINE_CODE",
                      "FENCED_CODE", "TABLE", "QUOTE"),
}

# INTERNAL intermediate, deliberately NOT in ALPHABETS: these are line
# classifications, and OTHER is always resolved before an edge is
# emitted. Declaring it as an output is what produced D341 F2.
LINE_CONTEXTS = ("FENCED_CODE", "QUOTE", "TABLE", "OTHER")

ATTRIBUTABLE = ("MARKDOWN_LINK", "EXPLICIT_PATH_REFERENCE", "BASENAME_UNIQUE")


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                          text=True).stdout


def tracked_md(repo):
    return sorted(p for p in git(repo, "ls-tree", "-r", "--name-only",
                                 "HEAD").splitlines() if p.endswith(".md"))


def untracked_md(repo):
    a = [p for p in git(repo, "ls-files", "--others", "--exclude-standard")
         .splitlines() if p.endswith(".md")]
    b = [p for p in git(repo, "ls-files", "--others", "--ignored",
                        "--exclude-standard").splitlines()
         if p.endswith(".md")]
    return sorted(a), sorted(b)


def line_contexts(txt: str):
    """Per-line structural context -- an INTERNAL intermediate."""
    out, fence = [], False
    for ln in txt.splitlines():
        s = ln.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            out.append("FENCED_CODE")
            continue
        if fence:
            out.append("FENCED_CODE")
        elif s.startswith(">"):
            out.append("QUOTE")
        elif s.startswith("|"):
            out.append("TABLE")
        else:
            out.append("OTHER")
    return out


def edge_context(txt, lines, lctx, pos, is_link):
    """The EMITTED context class of the occurrence at offset `pos`.

    Total over EDGE_CONTEXTS: the internal OTHER line class is always
    resolved to INLINE_CODE or PROSE_PATH here.
    """
    if is_link:
        return "MARKDOWN_LINK"
    upto = txt.count("\n", 0, pos)
    base = lctx[upto] if upto < len(lctx) else "OTHER"
    if base != "OTHER":
        return base
    line = lines[upto] if upto < len(lines) else ""
    col = pos - (txt.rfind("\n", 0, pos) + 1)
    if line[:col].count("`") % 2 == 1:
        return "INLINE_CODE"
    return "PROSE_PATH"


def _norm(srcdir, raw):
    cand = raw.lstrip("/") if raw.startswith("/") else (
        f"{srcdir}/{raw}" if str(srcdir) != "." else raw)
    parts = []
    for seg in pathlib.PurePosixPath(cand).parts:
        if seg == "..":
            if parts:
                parts.pop()
        elif seg != ".":
            parts.append(seg)
    return "/".join(parts)


def build_graph(repo, docs: list):
    """Returns edges: (src, dst|None, kind, raw, context)."""
    repo = pathlib.Path(repo)
    docset = set(docs)
    by_base = collections.defaultdict(list)
    for d in docs:
        by_base[pathlib.PurePosixPath(d).name].append(d)

    edges = []
    for src in sorted(docs):
        try:
            txt = (repo / src).read_text(errors="ignore")
        except OSError:
            continue
        lines, lctx = txt.splitlines(), line_contexts(txt)
        srcdir = pathlib.PurePosixPath(src).parent

        linked_spans = []
        for m in LINK.finditer(txt):
            raw = m.group(1).split("#")[0].strip()
            linked_spans.append((m.start(1), m.end(1)))
            if not raw or raw.startswith(("http://", "https://", "mailto:",
                                          "#")) or not raw.endswith(".md"):
                continue
            cand = _norm(srcdir, raw)
            if cand in docset:
                edges.append((src, cand, "MARKDOWN_LINK", raw,
                              "MARKDOWN_LINK"))
            else:
                edges.append((src, None, "BROKEN_LINK", raw, "MARKDOWN_LINK"))

        def inside_link(p):
            return any(a <= p < b for a, b in linked_spans)

        for d in sorted(docs):
            if d == src:
                continue
            for m in re.finditer(re.escape(d), txt):
                if inside_link(m.start()):
                    continue
                edges.append((src, d, "EXPLICIT_PATH_REFERENCE", d,
                              edge_context(txt, lines, lctx, m.start(), False)))

        for base in sorted(by_base):
            targets = by_base[base]
            if base not in txt or any(t in txt for t in targets):
                continue
            for m in re.finditer(re.escape(base), txt):
                if inside_link(m.start()):
                    continue
                c = edge_context(txt, lines, lctx, m.start(), False)
                if len(targets) == 1 and targets[0] != src:
                    edges.append((src, targets[0], "BASENAME_UNIQUE", base, c))
                elif len(targets) > 1:
                    edges.append((src, None, "BASENAME_AMBIGUOUS", base, c))
    return edges


def distinct_pairs(edges):
    return {(s, d) for s, d, k, _r, _c in edges if d and k in ATTRIBUTABLE}


def incoming(edges):
    c = collections.Counter()
    for _s, d in distinct_pairs(edges):
        c[d] += 1
    return c


def kind_tally(edges):
    return collections.Counter(k for _s, _d, k, _r, _c in edges)


def context_tally(edges):
    return collections.Counter(c for *_x, c in edges)
