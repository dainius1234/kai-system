#!/usr/bin/env python3
"""GENERATOR/READER ANALYSIS v3 — operation-level accounting. Read-only.

P12: every candidate I/O operation gets EXACTLY ONE disposition, and
     candidates == sum(dispositions). Nothing may silently disappear.

NO BASENAME FALLBACK for writer evidence (Kai). A write target must
resolve to an exact tracked repository-relative path, or UNRESOLVED.
"""
from __future__ import annotations
import ast, pathlib, re, subprocess, collections

EXCLUDE_DIRS = ("__pycache__", ".git", "node_modules", ".pytest_cache",
                ".mypy_cache", ".ruff_cache", "htmlcov", ".venv", "venv")
SRC_SUFFIX = (".py", ".sh", ".bash", ".yml", ".yaml")
DISPOSITIONS = ("RESOLVED_READ", "RESOLVED_WRITE", "READ_AND_WRITE",
                "RESOLVED_NON_DOCUMENT_TARGET", "UNRESOLVED_TARGET",
                "UNRESOLVED_RELEVANCE")

# Kai's P12 correction: NON_DOCUMENT requires a POSITIVE WITNESS.
# "no fragment ends .md" is absence of a marker, and absence of a marker
# is not a negative conclusion. A dynamic path could resolve to Markdown
# at runtime.
NON_DOC_EXT = (".json", ".py", ".txt", ".log", ".yml", ".yaml", ".sh",
               ".jsonl", ".csv", ".html", ".css", ".js", ".toml", ".ini",
               ".cfg", ".lock", ".png", ".jpg", ".svg", ".pdf", ".zip",
               ".tar", ".gz", ".sql", ".env", ".pyc", ".xml")
PY_READ = ("read_text", "read_bytes", "readlines", "read")
PY_WRITE = ("write_text", "write_bytes", "writelines")
# `>>?` allowed ">>" to match as ">" plus a CAPTURED ">", so the
# redirection operator itself became a target. Excluding <> from the
# captured class and requiring the operator to be complete fixes it.
SH_WRITE = re.compile(r"(?:>{1,2}\s*|tee\s+(?:-a\s+)?|sed\s+-i\S*\s+)"
                      r"([^\s;|&()<>]+)")
SH_READ = re.compile(r"(?:\bcat|\bless|\bhead|\btail)\s+([^\s;|&()]+)")

def git(repo, *a):
    return subprocess.run(["git", *a], cwd=repo, capture_output=True,
                          text=True).stdout

def tracked(repo, suffix=None):
    out = [p for p in git(repo, "ls-tree", "-r", "--name-only",
                          "HEAD").splitlines() if p]
    if suffix:
        out = [p for p in out if p.endswith(suffix)]
    return sorted(out)

def source_population(repo):
    """Git-tracked source files at the frozen tree (Kai: not rglob)."""
    out = []
    for p in tracked(repo):
        if any(p == d or p.startswith(d + "/") or f"/{d}/" in p
               for d in EXCLUDE_DIRS):
            continue
        if p.endswith(SRC_SUFFIX) or pathlib.PurePosixPath(p).name == "Makefile":
            out.append(p)
    return out

def _subscript_keys(node):
    """Constant nodes that are SUBSCRIPT KEYS, not path components.

    MISBINDING, D332 F1: os.environ["P"] yielded "P" as a path fragment,
    so a dynamic expression looked FIXED and P14's dynamic branch was
    never reached -- which is why Kai's specified mutation passed while
    proving nothing. A lookup key is real evidence bound to the wrong
    role.
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

    ast.walk is breadth-first, so `ROOT / "docs" / "X.md"` yielded
    ["X.md", "docs"] -- the outer operand before the deeper one. Joining
    that gave "X.md/docs", which matches nothing, so a genuinely
    resolvable multi-segment path became UNRESOLVED. A FALSE NEGATIVE:
    a derived document would have been recorded as having no writer.

    Sorting by (lineno, col_offset) restores textual order.
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

PATH_CTORS = {"Path", "PurePath", "PurePosixPath", "join", "str"}


def _has_dynamic_part(expr, binds):
    """True when the path expression contains a component we cannot
    resolve to a literal.

    Needed because `str(tmp) + "/SOUL.md"` yields the literal
    "/SOUL.md", which LOOKS absolute but is a CONCATENATION SUFFIX.
    Treating it as an absolute system path excluded the canonical
    tmp-vs-repository MISBINDING case -- my fix for one misbinding
    reintroducing another.
    """
    for n in ast.walk(expr):
        if isinstance(n, ast.Name) and n.id not in binds \
                and n.id not in ("pathlib", "os", "Path"):
            return True
        if isinstance(n, ast.Subscript):
            return True
    return False


def _candidate_paths(expr, binds):
    """Ordered string fragments of a target expression, or [] if none."""
    s = _consts(expr)
    if not s and isinstance(expr, ast.Name):
        s = binds.get(expr.id, [])
    return s

def _resolve_exact(frags, docset):
    """EXACT tracked-path resolution only. No basename fallback."""
    if not frags:
        return None
    joined = "/".join(f.strip("/") for f in frags if f)
    for cand in (joined, frags[-1]):
        c = cand.lstrip("./")
        if c in docset:
            return c
    return None

class Op:
    __slots__ = ("src", "mode", "expr", "disposition", "target",
                 "_frags", "_dynamic")
    def __init__(s, src, mode, expr):
        s.src, s.mode, s.expr = src, mode, expr
        s.disposition, s.target, s._frags = None, None, []
        s._dynamic = False

def collect(repo: pathlib.Path, docs: list[str]):
    docset = set(docs)
    ops: list = []
    for fn in source_population(repo):
        try:
            txt = (repo / fn).read_text(errors="ignore")
        except OSError:
            continue
        if fn.endswith(".py"):
            try:
                tree = ast.parse(txt)
            except SyntaxError:
                continue
            binds = _binds(tree)
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                # A read/write chained onto open(...) is the SAME physical
                # operation the open() already accounts for. Counting both
                # inflates the denominator and manufactures a spurious
                # UNRESOLVED. (P12 caught this.)
                if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call) \
                        and isinstance(f.value.func, ast.Name) \
                        and f.value.func.id == "open":
                    continue
                if isinstance(f, ast.Attribute) and f.attr in PY_WRITE:
                    o = Op(fn, "W", ast.dump(f.value)[:60])
                    o._frags = _candidate_paths(f.value, binds)
                    o._dynamic = _has_dynamic_part(f.value, binds); ops.append(o)
                elif isinstance(f, ast.Attribute) and f.attr in PY_READ:
                    o = Op(fn, "R", ast.dump(f.value)[:60])
                    o._frags = _candidate_paths(f.value, binds)
                    o._dynamic = _has_dynamic_part(f.value, binds); ops.append(o)
                elif isinstance(f, ast.Name) and f.id == "open" and n.args:
                    mode = "r"
                    if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                        mode = str(n.args[1].value)
                    for kw in n.keywords:
                        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                            mode = str(kw.value.value)
                    m = ("RW" if "+" in mode else
                         "W" if any(c in mode for c in "wax") else "R")
                    o = Op(fn, m, ast.dump(n.args[0])[:60])
                    o._frags = _candidate_paths(n.args[0], binds)
                    o._dynamic = _has_dynamic_part(n.args[0], binds); ops.append(o)
        else:
            for m in SH_WRITE.finditer(txt):
                o = Op(fn, "W", m.group(1)); o._frags = [m.group(1)]; ops.append(o)
            for m in SH_READ.finditer(txt):
                o = Op(fn, "R", m.group(1)); o._frags = [m.group(1)]; ops.append(o)

    tracked_all = set(tracked(repo))
    for o in ops:
        frags = getattr(o, "_frags", [])
        dynamic = any(("$" in f or "{" in f or "*" in f) for f in frags)

        # POSITIVE WITNESS for non-document, checked BEFORE any
        # absence-based reasoning.
        if frags and not dynamic:
            joined = "/".join(f.strip("/") for f in frags if f)
            last = frags[-1]
            if (joined.lstrip("./") in tracked_all
                    and not joined.endswith(".md")):
                o.disposition = "RESOLVED_NON_DOCUMENT_TARGET"; continue
            if last.endswith(NON_DOC_EXT):
                o.disposition = "RESOLVED_NON_DOCUMENT_TARGET"; continue

        # No literal at all, or dynamic: we cannot even establish whether
        # the target is a document. Relevance itself is unresolved.
        if not frags or dynamic:
            o.disposition = "UNRESOLVED_RELEVANCE"; continue

        if not any(f.endswith(".md") for f in frags):
            # literals present, none .md, none a proven non-doc witness
            o.disposition = "UNRESOLVED_RELEVANCE"; continue

        t = _resolve_exact(frags, docset)
        if t is None:
            o.disposition = "UNRESOLVED_TARGET"; continue
        o.target = t
        o.disposition = {"R": "RESOLVED_READ", "W": "RESOLVED_WRITE",
                         "RW": "READ_AND_WRITE"}[o.mode]
    return ops

def account(ops):
    tally = collections.Counter(o.disposition for o in ops)
    return len(ops), tally, sum(tally.values())


# ── P13: NEGATIVE CLAIMS REQUIRE A CLOSED SEARCH SPACE ───────────────
UNRESOLVED_KINDS = ("UNRESOLVED_TARGET", "UNRESOLVED_RELEVANCE")


def writer_claim(ops, doc):
    """The strongest ADMISSIBLE statement about writers of `doc`.

    NO_WRITER is emitted only when the search space is CLOSED: zero
    operations remain whose relevance or target is unresolved. Otherwise
    the strongest admissible answer is NO_PROVEN_WRITER.

    Earned by D330: `data/SOUL.md` had no proven writer and I wrote "no
    test writes it" -- a closed-world claim over an open search space,
    one paragraph after recording a false negative.
    """
    writers = sorted({o.src for o in ops
                      if o.target == doc
                      and o.disposition in ("RESOLVED_WRITE",
                                            "READ_AND_WRITE")})
    if writers:
        return "PROVEN_WRITE_RELATION", writers
    open_ops = [o for o in ops if o.disposition in UNRESOLVED_KINDS]
    if open_ops:
        return "NO_PROVEN_WRITER", {"unresolved_operations": len(open_ops)}
    return "NO_WRITER", {"search_space": "CLOSED"}


# ── PATH DOMAINS ────────────────────────────────────────────────────
#
# D332 excluded 99 operations with the witness "fixed directory 'dev'
# disjoint from 'data'". All 99 are `/dev/null`. The VERDICT was right
# and the RATIONALE was false: a device node described as a repository
# directory. A correct answer with a false reason is not qualified
# evidence (Kai). Domain is established BEFORE any directory reasoning.
PATH_DOMAINS = ("REPO_RELATIVE", "ABSOLUTE_SYSTEM", "REMOTE_URI",
                "SHELL_VARIABLE", "PUNCTUATION_ARTEFACT", "DYNAMIC_UNKNOWN")


def path_domain(frags, dynamic=False, tracked_all=()):
    if not frags:
        return "DYNAMIC_UNKNOWN"
    joined = "/".join(f.strip("/") for f in frags if f)
    raw = frags[0] if len(frags) == 1 else joined
    stripped = raw.strip().strip("\"'`")
    if not stripped or all(c in "\"'`>|&;$ " for c in stripped):
        return "PUNCTUATION_ARTEFACT"
    if "://" in stripped:
        return "REMOTE_URI"
    if "$" in stripped or "{" in stripped or "*" in stripped:
        return "SHELL_VARIABLE"
    if joined.lstrip("./") in tracked_all:
        return "REPO_RELATIVE"          # positive match beats appearances
    if dynamic:
        return "DYNAMIC_UNKNOWN"        # a suffix is not an absolute path
    if raw.startswith("/"):
        return "ABSOLUTE_SYSTEM"
    return "REPO_RELATIVE"


# ── P14: CLAIM-SCOPED NEGATIVES REQUIRE CONSTRUCTIVE EXCLUSION ───────
#
# P13 demanded a repository-wide closed search space, which with 889 open
# operations meant NO_WRITER could never be emitted for ANY document.
# P14 closes the space FOR ONE CLAIM instead — but exclusion must be
# CONSTRUCTIVE. "It doesn't look like it points at T" recreates the
# dismissed-on-absence defect P13 exists to forbid.
TARGET_DISPOSITIONS = ("REACHES_T", "EXCLUDED_FROM_T", "COULD_REACH_T")


def _fixed(frags):
    """True when every fragment is a literal with no dynamic component."""
    return bool(frags) and not any(
        ("$" in f or "{" in f or "*" in f) for f in frags)


def target_disposition(op, target, tracked_all):
    """Exactly one disposition, with a POSITIVE witness for exclusion."""
    frags = op._frags
    tbase = pathlib.PurePosixPath(target).name
    tdir = str(pathlib.PurePosixPath(target).parent)

    if op.target == target:
        return "REACHES_T", "exact resolved target"

    dom = path_domain(frags, getattr(op, "_dynamic", False), tracked_all)
    if dom == "ABSOLUTE_SYSTEM":
        return "EXCLUDED_FROM_T", (
            "absolute system path, outside the repository target domain")
    if dom == "REMOTE_URI":
        return "EXCLUDED_FROM_T", "remote URI, not a repository path"
    if dom == "PUNCTUATION_ARTEFACT":
        return "EXCLUDED_FROM_T", "not a path expression (parser artefact)"

    if not _fixed(frags):
        # dynamic or no literal: cannot be excluded constructively
        return "COULD_REACH_T", None

    joined = "/".join(f.strip("/") for f in frags if f).lstrip("./")
    last = frags[-1]

    # WITNESS 1 — exact resolved path that is some OTHER tracked file
    if joined in tracked_all and joined != target:
        return "EXCLUDED_FROM_T", f"exact resolved path {joined!r} != target"

    # WITNESS 2 — decisive incompatible extension
    if last.endswith(NON_DOC_EXT):
        return "EXCLUDED_FROM_T", f"fixed non-Markdown extension {last!r}"

    # WITNESS 3 — fixed basename differs from the target basename
    if last.endswith(".md") and pathlib.PurePosixPath(last).name != tbase:
        return "EXCLUDED_FROM_T", (
            f"fixed basename {pathlib.PurePosixPath(last).name!r} != {tbase!r}")

    # WITNESS 4 — fixed directory root provably disjoint
    if "/" in joined and tdir not in (".", ""):
        odir = str(pathlib.PurePosixPath(joined).parent)
        if odir != tdir and not odir.startswith(tdir + "/") \
                and not tdir.startswith(odir + "/"):
            return "EXCLUDED_FROM_T", (
                f"fixed directory {odir!r} disjoint from {tdir!r}")

    return "COULD_REACH_T", None


def writer_claim_scoped(ops, target, tracked_all):
    """P14: the strongest admissible negative for ONE target."""
    writes = [o for o in ops
              if o.disposition in ("RESOLVED_WRITE", "READ_AND_WRITE")
              or o.mode in ("W", "RW")]
    buckets = {k: [] for k in TARGET_DISPOSITIONS}
    witnesses = {}
    for o in writes:
        d, w = target_disposition(o, target, tracked_all)
        buckets[d].append(o)
        if w:
            witnesses.setdefault(w, 0)
            witnesses[w] += 1
    total = sum(len(v) for v in buckets.values())
    assert total == len(writes), "P14 population does not reconcile"
    if buckets["REACHES_T"]:
        return ("PROVEN_WRITE_RELATION",
                sorted({o.src for o in buckets["REACHES_T"]}),
                {k: len(v) for k, v in buckets.items()}, witnesses)
    if buckets["COULD_REACH_T"]:
        return ("NO_PROVEN_WRITER", None,
                {k: len(v) for k, v in buckets.items()}, witnesses)
    return ("NO_WRITER", None,
            {k: len(v) for k, v in buckets.items()}, witnesses)
