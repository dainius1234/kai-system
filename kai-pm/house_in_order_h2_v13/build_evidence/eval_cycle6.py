#!/usr/bin/env python3
"""THE CYCLE-6 EVALUATOR. Produces STEP2_M3_CYCLE6_RESULT.txt.

Banked so the figures in that file are REPRODUCIBLE rather than
asserted (R13: a derived claim travels with its derivation). It is run
ONCE, after the implementation is banked, and the implementation is not
altered afterwards.

THREE SOURCES, and they are deliberately different things:

  identities   STEP2_M3_IDENTITY_MANIFEST.tsv -- path/start/end/detector
               only. Identity-only by construction, so no scope answer
               enters this evaluation as an input.
  baseline     STEP2_M3_RECORD_MANIFEST.tsv -- the FROZEN PRE-REPAIR
               scope column, read for the baseline delta and nothing
               else. Never rewritten.
  content      `git show <tree>:<path>` from the frozen subject tree.
               Never the working filesystem.

The predecessor is LOADED FROM THE GIT OBJECT AND EXECUTED. No cycle-5
answer here is a paraphrase of what cycle 5 "would have" said.

Conservation is ASSERTED, not printed and hoped for: the identity sets
of the three sources must be equal or the run aborts.

    python3 eval_cycle6.py --subject-repo R --tree T --out F
"""
from __future__ import annotations
import argparse
import collections
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


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def load_ref(repo, ref):
    src = _git(repo, "show", f"{ref}:{PASSA_PATH}")
    f = pathlib.Path(tempfile.mkdtemp(prefix="m3_ref_")) / "ref_passa.py"
    f.write_bytes(src)
    spec = importlib.util.spec_from_file_location("ref_passa", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Corpus:
    """Documents AS THE FROZEN TREE HOLDS THEM, read once each."""

    def __init__(self, repo, tree):
        self.repo, self.tree, self._c = repo, tree, {}

    def text(self, path):
        if path not in self._c:
            blob = _git(self.repo, "show", f"{self.tree}:{path}")
            try:
                self._c[path] = blob.decode("utf-8")
            except UnicodeDecodeError as e:
                raise SystemExit(f"R11 ABORT: {path} is not UTF-8 ({e})")
        return self._c[path]

    def line_of(self, path, off):
        return self.text(path)[:off].count("\n") + 1

    def raw_line(self, path, off):
        t = self.text(path)
        a = t.rfind("\n", 0, off) + 1
        b = t.find("\n", off)
        return t[a:b if b >= 0 else len(t)]

    def __len__(self):
        return len(self._c)


def scope(mod, corpus, path, off, det):
    """The PRODUCTION window, bound to a local named `head` so the
    window-parity guard's leg A reads the production expression."""
    head = corpus.text(path)[:mod.HEAD_BYTES]
    return mod._scope_of(head, off, det)


def load_manifests():
    ids, base = [], {}
    for ln in (HERE / "STEP2_M3_IDENTITY_MANIFEST.tsv").read_text(
            encoding="utf-8").splitlines()[1:]:
        p, s, e, det, _sel = ln.split("\t")
        ids.append((p, int(s), int(e), det))
    for ln in (HERE / "STEP2_M3_RECORD_MANIFEST.tsv").read_text(
            encoding="utf-8").splitlines()[1:]:
        p, s, e, det, _sel, sc = ln.split("\t")
        base[(p, int(s), int(e), det)] = sc
    if len(set(ids)) != len(ids):
        raise SystemExit("R11 ABORT: the identity manifest is not unique.")
    if set(ids) != set(base):
        raise SystemExit(f"R11 ABORT: identity sets differ — "
                         f"{len(ids)} identities, {len(base)} baseline rows.")
    return ids, base


def evaluate(ids, corpus, old):
    new_s, old_s = {}, {}
    for k in ids:
        p, s, _e, det = k
        new_s[k] = scope(NEW, corpus, p, s, det)
        old_s[k] = scope(old, corpus, p, s, det)
    if set(new_s) != set(ids) or len(new_s) != len(ids):
        raise SystemExit("R11 ABORT: records out != records in.")
    return old_s, new_s


def render(ids, base, old_s, new_s, corpus, candidate):
    o = []
    A = o.append
    A("STEP 2 / M3 — CYCLE 6 RESULT")
    A(f"IMPLEMENTATION {candidate} banked before this run. BASELINE 0f3da09.")
    A(f"PREDECESSOR {CYCLE5[:7]}, loaded from the git object and executed.")
    A("PRODUCED BY ORION. NOT A REFERENCE KEY. ZERO ADJUDICATION WEIGHT.")
    A("")
    n = len(ids)
    wf = lambda d: sum(1 for k in ids if d[k] == "WHOLE_FILE")   # noqa: E731
    A(f"  baseline   WHOLE_FILE {wf(base):<5}SPAN {n - wf(base):<5}total {n}")
    A(f"  cycle 5    WHOLE_FILE {wf(old_s):<5}SPAN {n - wf(old_s):<5}total {n}")
    A(f"  cycle 6    WHOLE_FILE {wf(new_s):<5}SPAN {n - wf(new_s):<5}total {n}")
    pb = sorted(k for k in ids if base[k] == "SPAN"
                and new_s[k] == "WHOLE_FILE")
    db = sorted(k for k in ids if base[k] == "WHOLE_FILE"
                and new_s[k] == "SPAN")
    ch = sorted(k for k in ids if old_s[k] != new_s[k])
    A(f"  vs baseline  promotions {len(pb)}   demotions {len(db)}   "
      f"unchanged {n - len(pb) - len(db)}")
    A(f"  vs cycle 5   CHANGED IDENTITIES {len(ch)}   unchanged {n - len(ch)}")
    for k in ids:
        if corpus.text(k[0])[k[1]:k[2]] == "76dbba4":
            A(f"  D14-B 76dbba4 {k[0]} L{corpus.line_of(k[0], k[1])} "
              f"-> {new_s[k]}")
    A("")
    A("  CONSERVATION")
    A(f"    records in {n} · records out {len(new_s)} · "
      f"identity set unchanged: {set(base) == set(new_s)}")
    A(f"    documents carrying at least one record: {len({k[0] for k in ids})}")
    spans = collections.Counter((k[0], k[1], k[2]) for k in ids)
    dup = sorted(x for x, c in spans.items() if c > 1)
    splits = sum(1 for x in dup
                 if len({new_s[k] for k in ids if k[:3] == x}) > 1)
    A(f"    same-span detector pairs {len(dup)} · splits {splits}")
    A("")
    A("  PAIR REPORT — every duplicate physical span")
    A(f"  {'path':<44}{'start':>6}{'end':>6}  detA        scopeA  detB  scopeB")
    for x in dup:
        a, b = sorted((k for k in ids if k[:3] == x), key=lambda k: k[3])[:2]
        A(f"  {x[0]:<44}{x[1]:>6}{x[2]:>6}  {a[3]:<11} {new_s[a]:<7} "
          f"{b[3]:<5} {new_s[b]}")
    A("")
    A(f"  CHANGED FROM CYCLE 5 ({len(ch)}) — every one, with its raw source")
    for k in ch:
        p, s, e, det = k
        A(f"    {p}\t{s}\t{e}\t{det}\tL{corpus.line_of(p, s)}\t"
          f"{old_s[k]} -> {new_s[k]}")
        A(f"      raw line   : {corpus.raw_line(p, s)}")
        A(f"      matched    : {corpus.text(p)[s:e]}")
    A("")
    for title, rows in ((f"DEMOTIONS vs baseline ({len(db)})", db),
                        (f"PROMOTIONS vs baseline ({len(pb)})", pb)):
        A(f"  {title}")
        for k in rows:
            A(f"    {k[0]}\t{k[1]}\t{k[2]}\t{k[3]}\tL{corpus.line_of(k[0], k[1])}")
        A("")
    A("  HISTORICAL RECONCILIATION ONLY — ZERO DESIGN AUTHORITY")
    A("    historical 214 WHOLE_FILE / 278 SPAN, 47 up / 1 down")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--candidate", default="HEAD")
    ap.add_argument("--out", default=str(HERE / "STEP2_M3_CYCLE6_RESULT.txt"))
    a = ap.parse_args()

    ids, base = load_manifests()
    corpus = Corpus(a.subject_repo, a.tree)
    old = load_ref(a.repo, CYCLE5)
    if set(old.BINDING_PREDICATES) != set(NEW.BINDING_PREDICATES):
        raise SystemExit("R11 ABORT: BINDING_PREDICATES changed since "
                         "cycle 5. Cycle 6 authorises no broad expansion.")
    cand = _git(a.repo, "rev-parse", "--short", a.candidate).decode().strip()
    old_s, new_s = evaluate(ids, corpus, old)
    text = render(ids, base, old_s, new_s, corpus, cand)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
