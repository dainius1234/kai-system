#!/usr/bin/env python3
"""CANDIDATE OPERATION EXTRACTION AND ADMISSION — Census v1.1.

Census v1.0 admitted every regex hit as a candidate operation and then
explained the bad ones away with an EXCLUDED_FROM_T witness. Kai's D341
ruling: an extraction artefact must be REJECTED BEFORE ADMISSION. It is
diagnostic evidence about the extractor, not a constructive exclusion of
a target.

The difference is not cosmetic. "Candidate operation positively excluded
from the target" asserts something about the target. "This was never a
candidate operation" asserts something about the parser. Only the second
is true of a `>` inside a comment.

THE ADMISSION MODEL
  A shell redirection means redirection ONLY IN A SHELL CONTEXT.
  * .sh / .bash      whole file
  * YAML             `run:` block scalars ONLY -- never the whole file
  * Makefile         recipe lines (tab-indented)
  * .py              never; Python is read by AST, not by redirect syntax

D341 preflight earned the YAML rule: `expr: vram_percent > 90` in
alert.rules.yml and `placeholder: "... <sha>"` in an issue template were
both admitted by v1.0 as write operations, because v1.0 scanned whole
YAML files as though they were shell.

REJECTION REASONS ARE DECLARED AND COUNTED, NEVER SILENT (R10/R11).
Every raw match is accounted for:
    raw_candidate_matches == rejected_non_operations
                             + admitted_candidate_operations
"""
from __future__ import annotations
import ast
import collections
import pathlib
import re
import subprocess

ALPHABETS = {
    "REJECTION_REASONS": (
        "NOT_SHELL_CONTEXT",        # YAML/Make text that is not a recipe
        "COMMENT_CONTEXT",          # operator sits after an unquoted #
        "QUOTED_STRING_CONTENT",    # operator sits inside a quoted string
        "ARROW_OPERATOR",           # the > belongs to -> / => / -->
        "EXTRACTION_ARTEFACT",      # captured token is punctuation only
    ),
    "OP_MODES": ("R", "W", "RW"),
}

EXCLUDE_DIRS = ("__pycache__", ".git", "node_modules", ".pytest_cache",
                ".mypy_cache", ".ruff_cache", "htmlcov", ".venv", "venv")
SRC_SUFFIX = (".py", ".sh", ".bash", ".yml", ".yaml")

PY_READ = ("read_text", "read_bytes", "readlines", "read")
PY_WRITE = ("write_text", "write_bytes", "writelines")

# The redirection operator is located separately from its target so the
# OPERATOR's position can be tested for comment/quote/arrow context. v1.0
# matched both together and so could not ask where the operator was.
SH_WRITE_OP = re.compile(r">{1,2}|tee\s+(?:-a\s+)?|sed\s+-i\S*\s+")
SH_READ_OP = re.compile(r"\b(?:cat|less|head|tail)\s+")
SH_TARGET = re.compile(r"[^\s;|&()<>]+")


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo), capture_output=True,
                          text=True).stdout


def tracked(repo, suffix=None):
    out = [p for p in git(repo, "ls-tree", "-r", "--name-only",
                          "HEAD").splitlines() if p]
    if suffix:
        out = [p for p in out if p.endswith(suffix)]
    return sorted(out)


def source_population(repo):
    """Git-tracked source files at the frozen tree (never rglob)."""
    out = []
    for p in tracked(repo):
        if any(p == d or p.startswith(d + "/") or f"/{d}/" in p
               for d in EXCLUDE_DIRS):
            continue
        if p.endswith(SRC_SUFFIX) or pathlib.PurePosixPath(p).name == "Makefile":
            out.append(p)
    return out


# ── SHELL CONTEXT ────────────────────────────────────────────────────
def shell_spans(path: str, text: str):
    """Absolute [start, end) offsets that are genuinely shell context."""
    name = pathlib.PurePosixPath(path).name
    if path.endswith((".sh", ".bash")):
        return [(0, len(text))]
    if name == "Makefile":
        spans, off = [], 0
        for ln in text.splitlines(keepends=True):
            if ln.startswith("\t"):
                spans.append((off, off + len(ln)))
            off += len(ln)
        return spans
    if not path.endswith((".yml", ".yaml")):
        return []

    # YAML: `run:` block scalars only.
    spans, off = [], 0
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"(\s*)-?\s*run:\s*(\S?)", ln)
        if m:
            indent = len(m.group(1))
            if m.group(2) in ("|", ">"):
                start = off + len(ln)
                j, end = i + 1, off + len(ln)
                while j < len(lines):
                    nxt = lines[j]
                    if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                        break
                    end += len(nxt)
                    j += 1
                spans.append((start, end))
                off, i = end, j
                continue
            # single-line `run: cmd`
            k = ln.index("run:") + 4
            spans.append((off + k, off + len(ln)))
        off += len(ln)
        i += 1
    return spans


def _line_bounds(text, pos):
    s = text.rfind("\n", 0, pos) + 1
    e = text.find("\n", pos)
    return s, (len(text) if e == -1 else e)


def quoted_and_comment_mask(line: str):
    """(inside_quote, after_comment) boolean masks for one shell line.

    A `>` inside "..." or '...' is string content, not a redirection --
    `stale_list="No stale branches (>30d old)."` was admitted by v1.0 as
    a write operation. A `>` after an UNQUOTED `#` is a comment.
    """
    n = len(line)
    inq = [False] * n
    com = [False] * n
    q = None
    commented = False
    for i, ch in enumerate(line):
        if commented:
            com[i] = True
            continue
        if q:
            inq[i] = True
            if ch == q:
                q = None
            continue
        if ch in "\"'":
            q = ch
            inq[i] = True
            continue
        if ch == "#":
            commented = True
            com[i] = True
    return inq, com


class Op:
    __slots__ = ("src", "mode", "expr", "disposition", "target", "frags",
                 "dynamic", "line")

    def __init__(s, src, mode, expr, line=0):
        s.src, s.mode, s.expr, s.line = src, mode, expr, line
        s.disposition, s.target, s.frags, s.dynamic = None, None, [], False


# ── PYTHON EXTRACTION (AST) ──────────────────────────────────────────
def _subscript_keys(node):
    """Constant nodes that are SUBSCRIPT KEYS, not path components.

    D332: os.environ["P"] yielded "P" as a path fragment, so a dynamic
    expression looked FIXED. Real evidence bound to the wrong role.
    """
    keys = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript):
            for k in ast.walk(n.slice):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(id(k))
    return keys


def _consts(node):
    """String constants IN SOURCE ORDER.

    ast.walk is breadth-first, so ROOT / "docs" / "X.md" yielded
    ["X.md", "docs"] -> "X.md/docs", which matches nothing, so a
    resolvable path became UNRESOLVED. A FALSE NEGATIVE.
    """
    skip = _subscript_keys(node)
    got = [n for n in ast.walk(node)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and id(n) not in skip]
    got.sort(key=lambda n: (getattr(n, "lineno", 0),
                            getattr(n, "col_offset", 0)))
    return [n.value for n in got]


def _binds(tree):
    b = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            s = _consts(n.value)
            if s:
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        b.setdefault(t.id, []).extend(s)
    return b


def _has_dynamic_part(expr, binds):
    """True when a component cannot be resolved to a literal.

    str(tmp) + "/SOUL.md" yields the literal "/SOUL.md", which LOOKS
    absolute but is a CONCATENATION SUFFIX.
    """
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id not in binds \
                and n.id not in ("pathlib", "os", "Path"):
            return True
        if isinstance(n, ast.Subscript):
            return True
    return False


def _candidate_paths(expr, binds):
    s = _consts(expr)
    if not s and isinstance(expr, ast.Name):
        s = binds.get(expr.id, [])
    return s


def _python_ops(fn, txt):
    try:
        tree = ast.parse(txt)
    except SyntaxError:
        return []
    binds = _binds(tree)
    ops = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        # A read/write chained onto open(...) is the SAME physical
        # operation the open() already accounts for.
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) \
                and isinstance(f.value.func, ast.Name) \
                and f.value.func.id == "open":
            continue
        o = None
        if isinstance(f, ast.Attribute) and f.attr in PY_WRITE:
            o = Op(fn, "W", ast.dump(f.value)[:60], getattr(n, "lineno", 0))
            tgt = f.value
        elif isinstance(f, ast.Attribute) and f.attr in PY_READ:
            o = Op(fn, "R", ast.dump(f.value)[:60], getattr(n, "lineno", 0))
            tgt = f.value
        elif isinstance(f, ast.Name) and f.id == "open" and n.args:
            mode = "r"
            if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                mode = str(n.args[1].value)
            for kw in n.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            m = ("RW" if "+" in mode else
                 "W" if any(c in mode for c in "wax") else "R")
            o = Op(fn, m, ast.dump(n.args[0])[:60], getattr(n, "lineno", 0))
            tgt = n.args[0]
        if o is not None:
            o.frags = _candidate_paths(tgt, binds)
            o.dynamic = _has_dynamic_part(tgt, binds)
            ops.append(o)
    return ops


# ── SHELL EXTRACTION WITH ADMISSION ──────────────────────────────────
def _is_punctuation(tok: str) -> bool:
    s = tok.strip().strip("\"'`")
    return (not s) or all(c in "\"'`>|&;$ " for c in s)


def _shell_ops(fn, txt, rejects, raw):
    spans = shell_spans(fn, txt)
    in_shell = [False] * (len(txt) + 1)
    for a, b in spans:
        for i in range(a, min(b, len(txt))):
            in_shell[i] = True

    ops = []
    for rx, mode in ((SH_WRITE_OP, "W"), (SH_READ_OP, "R")):
        for m in rx.finditer(txt):
            raw[0] += 1
            p = m.start()
            if not in_shell[p]:
                rejects["NOT_SHELL_CONTEXT"] += 1
                continue
            ls, le = _line_bounds(txt, p)
            line = txt[ls:le]
            col = p - ls
            inq, com = quoted_and_comment_mask(line)
            if col < len(com) and com[col]:
                rejects["COMMENT_CONTEXT"] += 1
                continue
            if col < len(inq) and inq[col]:
                rejects["QUOTED_STRING_CONTENT"] += 1
                continue
            if mode == "W" and m.group(0).startswith(">") and col > 0 \
                    and line[col - 1] in "-=":
                rejects["ARROW_OPERATOR"] += 1
                continue
            # The operator's position is tested above; the TARGET may be
            # separated from it by whitespace (`> docs/x.md`). Advancing
            # past spaces/tabs only -- never past a newline, which would
            # let a bare `>` at end of line steal the next line's word.
            q = m.end()
            while q < len(txt) and txt[q] in " \t":
                q += 1
            tm = SH_TARGET.match(txt, q)
            tok = tm.group(0) if tm else ""
            if _is_punctuation(tok):
                rejects["EXTRACTION_ARTEFACT"] += 1
                continue
            o = Op(fn, mode, tok, txt.count("\n", 0, p) + 1)
            o.frags = [tok]
            o.dynamic = any(c in tok for c in "${}*")
            ops.append(o)
    return ops


def collect(repo: pathlib.Path, docs: list):
    """Returns (admitted_ops, accounting).

    accounting reconciles the whole denominator, per Kai's D341 ruling:
    raw_candidate_matches, rejected_non_operations (by reason),
    admitted_candidate_operations.
    """
    repo = pathlib.Path(repo)
    rejects = collections.Counter()
    raw = [0]
    ops = []
    for fn in source_population(repo):
        try:
            txt = (repo / fn).read_text(errors="ignore")
        except OSError:
            continue
        if fn.endswith(".py"):
            got = _python_ops(fn, txt)
            raw[0] += len(got)
            ops += got
        else:
            ops += _shell_ops(fn, txt, rejects, raw)

    acc = {
        "raw_candidate_matches": raw[0],
        "rejected_non_operations": dict(rejects),
        "rejected_total": sum(rejects.values()),
        "admitted_candidate_operations": len(ops),
    }
    assert acc["raw_candidate_matches"] == acc["rejected_total"] + len(ops), (
        "admission accounting does not reconcile")
    return ops, acc
