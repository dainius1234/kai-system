#!/usr/bin/env python3
"""GENERATOR DETECTION — structural, not proximity. Scratchpad, read-only.

D328 withdrew the previous attempt: it matched any `>` within 300
characters of a mention, so every READER became a WRITER, and compiled
__pycache__ bytecode counted as source.

Here a generator relationship exists only when a WRITE OPERATION's own
target expression names the document.

  python : AST — .write_text/.write_bytes/.writelines/.write on a target,
           open(X, "w"|"a"|"x"), json.dump(..., open(X, "w"))
  shell  : `> path`, `>> path`, `tee [-a] path`, `sed -i ... path`
  yaml   : the same shell forms inside `run:` blocks

EXCLUSIONS ARE DECLARED, NEVER SILENT.
"""
from __future__ import annotations
import ast
import pathlib
import re

# Declared exclusions — generated artefacts are not source.
EXCLUDE_DIRS = ("__pycache__", ".git", "node_modules", ".pytest_cache",
                ".mypy_cache", ".ruff_cache", "htmlcov", ".venv", "venv")
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".so", ".egg-info")

SHELL_WRITE = re.compile(
    r"(?:>>?\s*|tee\s+(?:-a\s+)?|sed\s+-i[^\s]*\s+(?:[^\s]+\s+)?)"
    r"([A-Za-z0-9_./${}\"'-]+\.md)")


def source_files(repo: pathlib.Path):
    """The executable/config corpus, with exclusions declared above."""
    out = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(repo))
        if any(f"/{d}/" in f"/{rel}" for d in EXCLUDE_DIRS):
            continue
        if p.suffix in EXCLUDE_SUFFIX or p.suffix == ".md":
            continue
        if p.suffix in (".py", ".sh", ".yml", ".yaml", ".bash") or \
           p.name in ("Makefile",):
            try:
                out.append((rel, p.read_text(errors="ignore")))
            except OSError:
                pass
    return out


def _strings_in(node) -> list[str]:
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _binds(tree) -> dict:
    """name -> string constants reachable from its assignment.

    P11 caught the need for this: `p = pathlib.Path("x.md")` followed by
    `p.write_text(...)` yielded NO strings, so the write was SILENTLY
    DROPPED -- not reported UNRESOLVED. A silent miss is worse than an
    abstention, because nothing marks the gap.

    Deliberately shallow: single-level literal binding only. Anything
    beyond it must surface as UNRESOLVED, never be inferred.
    """
    out: dict = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            s = _strings_in(n.value)
            if s:
                for tgt in n.targets:
                    if isinstance(tgt, ast.Name):
                        out.setdefault(tgt.id, []).extend(s)
    return out


def _target_strings(expr, binds) -> tuple:
    """(strings, resolvable). Empty strings + resolvable=False means the
    caller MUST record UNRESOLVED rather than skip."""
    s = _strings_in(expr)
    if s:
        return s, True
    if isinstance(expr, ast.Name) and expr.id in binds:
        return binds[expr.id], True
    return [], False


def python_write_targets(src: str):
    """(targets, saw_unresolvable) for write operations."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], False
    binds = _binds(tree)
    unres = False
    hits: list[str] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        # X.write_text(...) / X.write_bytes(...) / X.writelines(...)
        if isinstance(f, ast.Attribute) and f.attr in (
                "write_text", "write_bytes", "writelines"):
            s, okr = _target_strings(f.value, binds)
            hits += s
            unres = unres or not okr
        # open(X, "w"/"a"/"x")
        if isinstance(f, ast.Name) and f.id == "open":
            mode = ""
            if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                mode = str(n.args[1].value)
            for kw in n.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if any(c in mode for c in "wax") and n.args:
                s, okr = _target_strings(n.args[0], binds)
                hits += s
                unres = unres or not okr
    return hits, unres


def shell_write_targets(src: str) -> list[str]:
    return [m.group(1).strip("\"'") for m in SHELL_WRITE.finditer(src)]


PY_READ = ("read_text", "read_bytes", "readlines", "read")


def python_read_targets(src: str):
    """(targets, saw_unresolvable) for read operations."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], False
    binds = _binds(tree)
    unres = False
    hits: list[str] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr in PY_READ:
            s, okr = _target_strings(f.value, binds)
            hits += s
            unres = unres or not okr
        if isinstance(f, ast.Name) and f.id == "open":
            mode = "r"
            if len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                mode = str(n.args[1].value)
            for kw in n.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if not any(c in mode for c in "wax") and n.args:
                s, okr = _target_strings(n.args[0], binds)
                hits += s
                unres = unres or not okr
    return hits, unres


SHELL_READ = re.compile(r"(?:cat|less|head|tail|grep[^\n]*?)\s+"
                        r"([A-Za-z0-9_./${}\"'-]+\.md)")


def shell_read_targets(src: str) -> list[str]:
    return [m.group(1).strip("\"'") for m in SHELL_READ.finditer(src)]


def directionality(repo: pathlib.Path, docs: list[str]):
    """doc -> {source_file: STATE}.

    STATES (Kai): PROVEN_WRITER | PROVEN_READER | READ_AND_WRITE
                  | POSSIBLE_WRITER | UNRESOLVED
    A mention alone is never a state. Unresolvable targets stay
    UNRESOLVED rather than being forced into a bucket.
    """
    byname: dict = {}
    for d in docs:
        byname.setdefault(pathlib.PurePosixPath(d).name, []).append(d)

    def resolve(t):
        t = t.strip().strip("\"'")
        if not t.endswith(".md"):
            return None
        if "$" in t or "{" in t:
            return "UNRESOLVED"
        if t in docs:
            return t
        c = byname.get(pathlib.PurePosixPath(t).name, [])
        return c[0] if len(c) == 1 else "UNRESOLVED"

    out: dict = {d: {} for d in docs}
    unresolved: list = []
    for fn, txt in source_files(repo):
        pw, uw = (python_write_targets(txt) if fn.endswith(".py")
                  else ([], False))
        pr, ur = (python_read_targets(txt) if fn.endswith(".py")
                  else ([], False))
        w = {resolve(t) for t in pw + shell_write_targets(txt)}
        r = {resolve(t) for t in pr + shell_read_targets(txt)}
        if uw or ur or "UNRESOLVED" in w or "UNRESOLVED" in r:
            unresolved.append(fn)
        w.discard(None); w.discard("UNRESOLVED")
        r.discard(None); r.discard("UNRESOLVED")
        for d in w & r:
            out[d][fn] = "READ_AND_WRITE"
        for d in w - r:
            out[d][fn] = "PROVEN_WRITER"
        for d in r - w:
            out[d][fn] = "PROVEN_READER"
    return out, sorted(set(unresolved))


def generators(repo: pathlib.Path, docs: list[str]):
    """doc -> sorted list of source files that WRITE it."""
    gen: dict[str, set] = {d: set() for d in docs}
    byname: dict[str, list[str]] = {}
    for d in docs:
        byname.setdefault(pathlib.PurePosixPath(d).name, []).append(d)

    for fn, txt in source_files(repo):
        pw, _u = (python_write_targets(txt) if fn.endswith(".py")
                  else ([], False))
        targets = pw + shell_write_targets(txt)
        for t in targets:
            t = t.strip()
            if not t.endswith(".md"):
                continue
            # exact repo-relative path wins
            if t in gen:
                gen[t].add(fn)
                continue
            base = pathlib.PurePosixPath(t).name
            cands = byname.get(base, [])
            if len(cands) == 1:
                gen[cands[0]].add(fn)
            # ambiguous basename target: attributed to NOBODY, by design
    return {d: sorted(v) for d, v in gen.items()}
