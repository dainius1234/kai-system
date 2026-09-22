#!/usr/bin/env python3
"""M1 CONTROLS. Banked WITH the implementation, before any corpus run.

THE PREDECESSOR IS LOADED FROM THE GIT OBJECT AND EXECUTED. No "old"
answer in this file is a paraphrase of what the predecessor would have
said.

EVERY CORPUS CONTROL IS SOURCE-BACKED. Rows are built by running the
FROZEN Pass A scanner over the document as the frozen tree holds it, and
the axis under test is called on the result. Nothing is hand-written to
make a case pass, and no path is special-cased inside the repair -- the
paths here name WHERE the source evidence lives, exactly as Kai's ruling
names them.

WHAT EACH SECTION PROVES
  1  FAIL-OLD / PASS-NEW on the three named review cases
  2  the five measured strict-A positives survive
  3  C negatives -- origin and lifecycle events abstain
  4  D negatives -- H1 date, bare dateline, contextual Status
  5  E negative  -- a bare `Date:` label abstains
  6  RESCUE and SOURCE ORDER, on real corpus documents and synthetically
  7  PERMUTATION INVARIANCE over every corpus row with >1 document date
  8  routes M1 must not touch: RUN_ID, COMMIT, contradiction
  9  the shared helper `_binding_witness` is byte-identical to the
     predecessor, so SCOPE cannot have moved

    python3 m1_controls.py --subject-repo R --tree T [--old-ref SHA]
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import pathlib
import random
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import classify as NEW                                         # noqa: E402
import passa as PA                                             # noqa: E402

M1_PREDECESSOR = "b0da564"
CLASSIFY_PATH = "kai-pm/house_in_order_h2_v13/classify.py"
FAILED = []


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'OK  ' if ok else '<<< '}{name:<62}{str(got):<14}"
          f"{'' if ok else 'expected ' + str(want)}")
    if extra:
        print(f"        {extra}")
    return ok


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def load_old(repo, ref):
    src = _git(repo, "show", f"{ref}:{CLASSIFY_PATH}")
    f = pathlib.Path(tempfile.mkdtemp(prefix="m1_old_")) / "old_classify.py"
    f.write_bytes(src)
    spec = importlib.util.spec_from_file_location("old_classify", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Subject:
    """Documents AS THE FROZEN TREE HOLDS THEM, scanned by frozen Pass A."""

    def __init__(self, repo, tree):
        self.repo, self.tree, self._rows = repo, tree, {}
        self.subject = _git(repo, "rev-parse", tree + "^{tree}").decode().strip()

    def text(self, path):
        return _git(self.repo, "show", f"{self.tree}:{path}").decode("utf-8")

    def row(self, path):
        if path not in self._rows:
            t = self.text(path)
            w = PA.scan(path, t, self.repo, "HEAD")
            self._rows[path] = {"path": path,
                                "witnesses": {k: [x.asdict() for x in v]
                                              for k, v in w.items()}}
        return self._rows[path]

    def paths(self):
        return [n for n in _git(self.repo, "ls-tree", "-r", "--name-only",
                                self.tree).decode().splitlines()
                if n.endswith(".md")]


def verdict(mod, row):
    return mod.validity(row, None)["value"]


def det(mod, row):
    v = mod.validity(row, None)
    w = v.get("witness")
    return (f"{w['source_selector']} {w['witness_value']!r}" if w else "—")


def synth_row(*lines):
    """A row from a synthetic document, scanned by the FROZEN scanner.

    The document is real text put through real Pass A, so a control can
    never pass because a witness was hand-shaped to fit.
    """
    doc = "# Synthetic\n\n" + "\n".join(lines) + "\n\n## Section\n\nbody\n"
    w = PA.scan("synthetic.md", doc, ".", "HEAD")
    return {"path": "synthetic.md",
            "witnesses": {k: [x.asdict() for x in v] for k, v in w.items()}}, doc


def section(title):
    print(f"\n{title}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--old-ref", default=M1_PREDECESSOR)
    a = ap.parse_args()

    OLD = load_old(a.repo, a.old_ref)
    S = Subject(a.subject_repo, a.tree)
    print(f"PREDECESSOR   {a.old_ref} classify.py, loaded from the git "
          f"object and executed")
    print(f"CANDIDATE     working tree classify.py")
    print(f"STATE PREDICATES declared closed-world: "
          f"{len(NEW.STATE_PREDICATES)} "
          f"{sorted(NEW.STATE_PREDICATES)}")
    if hasattr(OLD, "STATE_PREDICATES"):
        raise SystemExit("R11 ABORT: the predecessor already has "
                         "STATE_PREDICATES; --old-ref is wrong.")

    # 1 ────────────────────────────────────────────────────────────────
    section("1. FAIL-OLD / PASS-NEW — the three review cases Kai named")
    for path in ("kai-pm/CODE_AUDIT_BATCH_AGENTIC_API.md",
                 "kai-pm/CODE_AUDIT_PLANNING_PACKAGE_QA.md",
                 "kai-pm/RISKS.md"):
        r = S.row(path)
        n = path.split("/")[-1]
        check(f"{n:<44} predecessor", verdict(OLD, r), "TIME_BOUND")
        check(f"{n:<44} M1", verdict(NEW, r), "UNKNOWN",
              f"determining before: {det(OLD, r)}")

    # 2 ────────────────────────────────────────────────────────────────
    section("2. PROTECTED A POSITIVES — all five measured strict-A rows")
    for path in ("docs/ERROR_LOG.md",
                 "docs/agentic_patterns_spec.md",
                 "docs/gaps_and_hardening.md",
                 "kai-pm/CODE_AUDIT_REGISTER.md",
                 "kai-pm/NAVIGATION.md"):
        r = S.row(path)
        check(f"{path:<52}", verdict(NEW, r), "TIME_BOUND",
              f"determining now: {det(NEW, r)}")
        check(f"  ^ unchanged from the predecessor",
              verdict(NEW, r), verdict(OLD, r))

    # 3 ────────────────────────────────────────────────────────────────
    section("3. C NEGATIVES — origin and lifecycle events, no A witness")
    for label, path in (("created", "kai-pm/CONVICTION_SCENARIOS.md"),
                        ("created", "kai-pm/GPU_ARRIVAL_RUNBOOK.md"),
                        ("generated", "kai-pm/COMPOSE_DRIFT.md"),
                        ("generated", "kai-pm/MAKEFILE_TARGETS.md"),
                        ("started", "kai-pm/CODE_AUDIT_REGISTER_CONTINUED.md"),
                        ("opened", "kai-pm/TECH_DEBT_E402.md"),
                        ("prepared",
                         "kai-pm/KAI_FINAL_PRODUCT_ARCHITECTURE_SPECIFICATION.md"),
                        ("planning date",
                         "kai-pm/KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md"),
                        ("Sent",
                         "kai-pm/DEEPSEEK_BRIEF_2026-08-07_EMBEDDINGS.md"),
                        ("Written", "kai-pm/NEXT_STINT_PLAN.md"),
                        ("Closed", "kai-pm/EMBEDDING_BACKEND_STATE.md")):
        r = S.row(path)
        check(f"{label:<15}{path.split('/')[-1]:<47}", verdict(NEW, r),
              "UNKNOWN")
        check(f"  ^ predecessor said TIME_BOUND, so the case is live",
              verdict(OLD, r), "TIME_BOUND")

    # 4 ────────────────────────────────────────────────────────────────
    section("4. D NEGATIVES — no predicate at all")
    for label, path in (("H1 title date", "kai-pm/REALITY_CHECK_2026-05-10.md"),
                        ("H1 title date", "kai-pm/RUNTIME_TOPOLOGY_CENSUS.md"),
                        ("bare dateline", "kai-pm/UH2_INTAKE_REDESIGN.md"),
                        ("bare dateline", "kai-pm/UH2_SENSOR_INGRESS_PLAN.md"),
                        ("self-subject", "kai-pm/CLEANUP_TODO.md")):
        r = S.row(path)
        check(f"{label:<15}{path.split('/')[-1]:<47}", verdict(NEW, r),
              "UNKNOWN")
        check(f"  ^ predecessor said TIME_BOUND, so the case is live",
              verdict(OLD, r), "TIME_BOUND")
    # The corpus holds exactly ONE cycle-6 contextual Status date, and its
    # document also carries `Last updated`, so the DOCUMENT is rescued in
    # section 6. The class control is therefore made at WITNESS level:
    # the Status date itself must not qualify.
    r = S.row("kai-pm/PHASE_0_5_BACKLOG.md")
    stat = [w for w in r["witnesses"]["DATE"]
            if w["source_selector"] == "L6"]
    check("contextual Status date is not a state binding (witness level)",
          all(NEW._predicate_of(w) not in NEW.STATE_PREDICATES for w in stat),
          True, f"predicate read: {[NEW._predicate_of(w) for w in stat]}")

    # 5 ────────────────────────────────────────────────────────────────
    section("5. E NEGATIVE — a bare `Date:` label")
    for path in ("kai-pm/SERVICE_IDENTITY_MEASUREMENT.md",
                 "kai-pm/VERIFICATION_ARCHITECTURE_REVIEW.md",
                 "docs/gpu_integration_phase0.md"):
        r = S.row(path)
        check(f"{path:<62}", verdict(NEW, r), "UNKNOWN")
        check(f"  ^ predecessor said TIME_BOUND, so the case is live",
              verdict(OLD, r), "TIME_BOUND")

    # 6 ────────────────────────────────────────────────────────────────
    section("6. RESCUE AND SOURCE ORDER")
    for path, why in (
            ("kai-pm/PHASE1_READINESS.md", "Created L3 first, Last updated L4"),
            ("kai-pm/PHASE_0_5_BACKLOG.md", "Status L6 first, Last updated L7")):
        r = S.row(path)
        check(f"{path.split('/')[-1]:<44} stays TIME_BOUND", verdict(NEW, r),
              "TIME_BOUND", f"{why}; determining now: {det(NEW, r)}")
        check(f"  ^ determining witness MOVED to the qualified one",
              det(NEW, r) != det(OLD, r), True,
              f"before {det(OLD, r)}  after {det(NEW, r)}")

    rev_first, _ = synth_row("**Reviewed:** 2026-01-02",
                             "**Last updated:** 2026-03-04")
    upd_first, _ = synth_row("**Last updated:** 2026-03-04",
                             "**Reviewed:** 2026-01-02")
    check("synthetic: review FIRST, qualified second", verdict(NEW, rev_first),
          "TIME_BOUND", f"determining: {det(NEW, rev_first)}")
    check("synthetic: source order REVERSED", verdict(NEW, upd_first),
          "TIME_BOUND", f"determining: {det(NEW, upd_first)}")
    check("  ^ same verdict under both orders — order does not decide",
          verdict(NEW, rev_first), verdict(NEW, upd_first))
    check("  ^ predecessor DID depend on order (control can fail)",
          verdict(OLD, rev_first) == verdict(OLD, upd_first)
          and det(OLD, rev_first) == det(OLD, upd_first), False,
          f"predecessor determining: {det(OLD, rev_first)} vs "
          f"{det(OLD, upd_first)}")

    # 7 ────────────────────────────────────────────────────────────────
    section("7. PERMUTATION INVARIANCE over every corpus row with >1 "
            "document-scoped date")
    rng = random.Random(20260908)
    multi = flips = 0
    for path in S.paths():
        r = S.row(path)
        dates = r["witnesses"].get("DATE", [])
        if sum(1 for w in dates if w["applicability_scope"] == "WHOLE_FILE") < 2:
            continue
        multi += 1
        base_v, base_d = verdict(NEW, r), det(NEW, r)
        orders = [list(reversed(dates))]
        for _ in range(6):
            p = dates[:]
            rng.shuffle(p)
            orders.append(p)
        for p in orders:
            alt = {"path": path, "witnesses": dict(r["witnesses"], DATE=p)}
            if verdict(NEW, alt) != base_v or det(NEW, alt) != base_d:
                flips += 1
                print(f"     <<< {path} flipped under permutation")
                break
    check(f"corpus rows with >1 document-scoped date (the DENOMINATOR: "
          f"{multi}); flips", flips, 0)
    # Three corpus rows is a thin denominator, so the invariant is also
    # exercised on a synthetic carrying FOUR document-scoped dates, one
    # per class, under ALL 24 orderings.
    lines = ["**Reviewed:** 2026-01-02", "**Created:** 2026-02-03",
             "**Date:** 2026-03-04", "**Last updated:** 2026-04-05"]
    seen = set()
    for order in itertools.permutations(lines):
        r, _ = synth_row(*order)
        seen.add((verdict(NEW, r), NEW.validity(r, None)["witness"]
                  ["witness_value"]))
    check("synthetic, 4 document dates, all 24 orderings agree",
          len(seen), 1, f"outcome(s): {sorted(seen)}")
    check("  ^ and the qualified date is the one that determines",
          sorted(seen)[0], ("TIME_BOUND", "2026-04-05"))

    # 8 ────────────────────────────────────────────────────────────────
    section("8. ROUTES M1 MUST NOT TOUCH")
    # SOURCE-BACKED: the five COMMIT-route rows measured under the
    # predecessor. These are the real regression, not a fixture.
    for path in ("kai-pm/CODE_AUDIT_FINAL_REPORT.md",
                 "kai-pm/CODE_AUDIT_MASTER.md",
                 "kai-pm/SERVICE_IDENTITY_STATE.md",
                 "kai-pm/UH0_EVIDENCE_MANIFEST.md",
                 "kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md"):
        r = S.row(path)
        check(f"COMMIT route {path.split('/')[-1]:<49}", verdict(NEW, r),
              "EXACT_SNAPSHOT")
        check("  ^ unchanged from the predecessor", verdict(NEW, r),
              verdict(OLD, r))
    # The corpus has NO RUN_ARTEFACT row, so this route needs a synthetic.
    # `**Last run:**` is NOT a document-binding label, so the run id there
    # is SPAN and never reaches the route -- a fixture built that way
    # would prove nothing. The run id is placed under a binding label so
    # the route is genuinely exercised.
    run_row, _ = synth_row("**Version:** run 31570714150",
                           "**Last updated:** 2026-03-04")
    check("RUN_ID route still wins over a qualified date",
          verdict(NEW, run_row), "RUN_ARTEFACT")
    check("  ^ unchanged from the predecessor", verdict(NEW, run_row),
          verdict(OLD, run_row))
    c = {"claimed": "2026-01-01", "git_last": "2026-06-01", "drift_days": 151}
    r = S.row("kai-pm/RISKS.md")
    check("contradiction still short-circuits to UNKNOWN",
          NEW.validity(r, c)["value"], "UNKNOWN")
    check("  ^ unchanged from the predecessor",
          NEW.validity(r, c)["value"], OLD.validity(r, c)["value"])
    check("  ^ and it is the CONTRADICTION abstention, not the M1 one",
          "BINDING_CONTRADICTION" in str(NEW.validity(r, c)), True)

    # 9 ────────────────────────────────────────────────────────────────
    section("9. SCOPE CANNOT HAVE MOVED — the shared helper is untouched")
    # AST, not text. A line filter cannot reliably strip a docstring --
    # its interior lines look like ordinary code -- and comparing text
    # would report a difference for a comment. The executable structure
    # is what must be identical, so the docstring node is dropped and the
    # rest is compared as a tree.
    import ast
    import inspect

    def executable(fn):
        tree = ast.parse(inspect.getsource(fn).lstrip()).body[0]
        if (tree.body and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
                and isinstance(tree.body[0].value.value, str)):
            tree.body = tree.body[1:]
        return ast.dump(tree)

    check("_binding_witness executable AST identical to predecessor",
          executable(NEW._binding_witness) == executable(OLD._binding_witness),
          True)
    check("scope() executable AST identical to predecessor",
          executable(NEW.scope) == executable(OLD.scope), True)
    # and behaviourally, on every affected-class document in section 1/3/5
    same = all(NEW.scope(S.row(p))["value"] == OLD.scope(S.row(p))["value"]
               for p in ("kai-pm/CODE_AUDIT_BATCH_AGENTIC_API.md",
                         "kai-pm/RISKS.md", "kai-pm/CONVICTION_SCENARIOS.md",
                         "kai-pm/REALITY_CHECK_2026-05-10.md",
                         "kai-pm/SERVICE_IDENTITY_MEASUREMENT.md"))
    check("SCOPE verdict unchanged on one document per affected class",
          same, True)

    print(f"\nCONTROLS: "
          f"{'ALL PASS' if not FAILED else 'FAILED — ' + ', '.join(FAILED)}")
    raise SystemExit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
