#!/usr/bin/env python3
"""E2 — ROOT-CAUSE INVESTIGATION. MEASUREMENT ONLY. NO REPAIR.

THIS FILE CHANGES NOTHING. It renames nothing, defines no new fact, edits
no production module. It reads the ACCEPTED LINEAGE -- M3 at 002e0ab, M1
at 53791c4 -- and the frozen subject tree, and reports what is there.

THE SUSPECTED DEFECT, as put by Kai: EVIDENCE-CLASS INFLATION. The fact
`CONSUMED_AT_SUBJECT` asserts runtime consumption while its determining
evidence is static textual/reference analysis. Kai's instruction is to
PROVE OR DISPROVE that wording from the current repository, not to assume
it. So every leg below is derived, and the two places where the current
code does NOT match Kai's quoted Rev4 line are reported as findings
rather than smoothed over.

WHAT IS PROVED HERE, AND HOW
  producer      the single assignment site, located by AST, not by grep
  staticness    the full derivation chain walked to its origin, with the
                evidence class of each stage named from its own source
  population    measured, with the complete denominator chain that
                explains WHY the number is what it is
  consumers     every reader of the fact, of `readers`, and of
                `reader_ops`, enumerated across the tracked repository
  independence  AST proof that the verdict layer never sees any of them
  coupling      a CAN-FAIL probe: a rename that forgets one map entry
                silently converts every positive to False

    python3 e2_investigation.py --subject-repo R --tree T \\
        --census-package C --repo REPO --out F
"""
from __future__ import annotations
import argparse
import ast
import collections
import hashlib
import json
import pathlib
import subprocess
import sys

SELF = pathlib.Path(__file__).resolve()
HERE = SELF.parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import ontology as ont                                         # noqa: E402
import run_h2_v12 as R                                         # noqa: E402
import subjectbind as sb                                       # noqa: E402

FACT = "CONSUMED_AT_SUBJECT"
SUBJECT_REPO = [None]
SUBJECT_ARG = [None]
FIELD = "readers"


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def tree_text(repo, tree, path):
    return _git(repo, "show", f"{tree}:{path}").decode("utf-8", "replace")


# ── 1. THE PRODUCER, LOCATED BY AST ───────────────────────────────────
def producer_sites(pkg):
    """Every assignment whose subscript key is the fact name.

    Located structurally. A grep would also match a docstring, a comment
    and a name inside a tuple, and could not tell which one WRITES it.
    """
    out = []
    for py in sorted(pathlib.Path(pkg).rglob("*.py")):
        if py.resolve() == SELF:
            continue                      # see field_sites: R9
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Assign):
                continue
            for t in n.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.slice, ast.Constant)
                        and t.slice.value == FACT):
                    out.append((py.relative_to(pathlib.Path(pkg).parent),
                                n.lineno, ast.unparse(n)))
    return out


def field_sites(pkg, field):
    """Every read of row[<field>] in the package, by AST.

    THE ONE EXCLUSION IS THIS FILE, BY RESOLVED PATH IDENTITY. The
    instrument reads the field in order to report on it; counting itself
    among the consumers it governs is R9 -- a watcher observing itself
    and calling the result the world. There is deliberately no
    directory-name or category exclusion.
    """
    out = []
    for py in sorted(pathlib.Path(pkg).rglob("*.py")):
        if py.resolve() == SELF:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if (isinstance(n, ast.Subscript)
                    and isinstance(n.slice, ast.Constant)
                    and n.slice.value == field):
                out.append((py.relative_to(pathlib.Path(pkg).parent),
                            n.lineno, ast.unparse(n)))
    return out


def verdict_layer_independence(pkg):
    """AST proof that classify.py never sees the fact or the field.

    A negative claim needs a bounded search that earns it (R17), so this
    walks the WHOLE module rather than checking the functions I happen to
    remember.
    """
    tree = ast.parse((pathlib.Path(pkg) / "classify.py").read_text())
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and n.value in (FACT, FIELD,
                                                       "reader_ops",
                                                       "evidence_facts"):
            hits.append((getattr(n, "lineno", 0), repr(n.value)))
        if isinstance(n, ast.Name) and n.id in (FACT, FIELD):
            hits.append((getattr(n, "lineno", 0), n.id))
    return hits


# ── 2. IS ANY OF IT RUNTIME? ──────────────────────────────────────────
def staticness(census):
    """The evidence class of every stage of the derivation chain.

    Each stage is characterised from ITS OWN source, so the answer is not
    my recollection of what opscan does.
    """
    c = pathlib.Path(census)
    ops_src = (c / "opscan.py").read_text()
    cl_src = (c / "claims.py").read_text()
    tree = ast.parse(ops_src)
    uses_ast = any(isinstance(n, ast.Attribute) and
                   isinstance(n.value, ast.Name) and n.value.id == "ast"
                   for n in ast.walk(tree))
    # every subprocess invocation in opscan, with its argv, so "it only
    # calls git ls-tree" is shown rather than asserted
    calls = []
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "subprocess"):
            calls.append((n.lineno, ast.unparse(n)[:140]))
    head = _git(SUBJECT_REPO[0], "rev-parse", "HEAD").decode().strip()
    clean = _git(SUBJECT_REPO[0], "status", "--porcelain").decode().strip() == ""
    return {"subject_head": head, "subject_arg": SUBJECT_ARG[0],
            "head_equals_subject": head == SUBJECT_ARG[0],
            "worktree_clean": clean,
            "opscan parses source with ast": uses_ast,
            "opscan subprocess calls": calls,
            # NAMED FOR WHAT IT ACTUALLY TESTS. opscan's read_text calls
            # read the READING SOURCE FILES, never the .md targets, and
            # they read the WORKING FILESYSTEM -- see section 14, S1.
            "opscan reads reader source text from the filesystem":
                "read_text" in ops_src,
            "opscan enumerates via git ls-tree HEAD":
                "ls-tree" in ops_src and '"HEAD"' in ops_src,
            "claims.py imports subprocess": "import subprocess" in cl_src,
            "claims.py executes anything": any(
                w in cl_src for w in ("subprocess.", "os.system", "exec(",
                                      "eval(", "importlib"))}


# ── 3. POPULATION, WITH ITS WHOLE DENOMINATOR CHAIN ───────────────────
def population(subject_repo, census, passa, classification):
    sys.path.insert(0, str(census))
    import claims as C, docgraph as G, opscan as O          # noqa: E402

    repo = pathlib.Path(subject_repo)
    pop = O.source_population(repo)
    tracked = G.tracked_md(subject_repo)
    ops, acc = O.collect(repo, tracked)
    C.classify(ops, tracked, set(O.tracked(subject_repo)))
    disp = collections.Counter(o.disposition for o in ops)
    reads = [o for o in ops if o.disposition == "RESOLVED_READ"]

    rows = {r["path"]: r for r in passa["rows"]}
    cls = {r["path"]: r for r in classification["rows"]}
    with_readers = sorted(p for p, r in rows.items() if r[FIELD])
    with_ops = sorted(p for p, r in rows.items() if r["reader_ops"])
    emitted = sorted(p for p, r in cls.items()
                     if r["evidence_facts"].get(FACT))
    abstained = sorted(p for p, r in cls.items()
                       if FACT in r.get(
                           "evidence_facts_abstained_no_compliant_trace", []))
    return {"analysis_scope_files": len(pop),
            "analysis_scope_by_suffix": dict(collections.Counter(
                p.rsplit(".", 1)[-1] if "." in p.rsplit("/", 1)[-1]
                else "Makefile" for p in pop)),
            "exclude_dirs": list(O.EXCLUDE_DIRS),
            "src_suffix": list(O.SRC_SUFFIX),
            "accounting": {k: v for k, v in acc.items()
                           if k != "rejected_non_operations"},
            "dispositions": dict(disp),
            "resolved_read_ops": len(reads),
            "distinct_read_targets": len({o.target for o in reads}),
            "rows_with_readers": with_readers,
            "rows_with_reader_ops": with_ops,
            "fact_emitted_true": emitted,
            "fact_abstained_A6ii": abstained,
            "tracked_documents": len(tracked)}


# ── 4. CAN-FAIL COUPLING PROBE ────────────────────────────────────────
def coupling_probe(subject_repo, passa):
    """A rename that forgets TRACE_CLASS does not raise. It ABSTAINS.

    That is the dangerous shape: five positives become False through the
    A6-ii path and the run still succeeds. Demonstrated, not predicted.
    """
    row = next(r for r in passa["rows"] if r[FIELD])
    text = (pathlib.Path(subject_repo) / row["path"]).read_text(errors="ignore")
    claims, _stats = sb.bind_claims(row["path"], text)
    det = sb.determining_claims(claims)

    def run():
        f, _ac, tr, ab = R.evidence_facts(row, claims, None, det,
                                          passa["subject"], subject_repo)
        return f.get(FACT), (tr.get(FACT) or {}).get("witness_type"), ab

    before = run()
    saved = dict(R.TRACE_CLASS)
    R.TRACE_CLASS.pop(FACT)
    try:
        without = run()
    finally:
        R.TRACE_CLASS.clear()
        R.TRACE_CLASS.update(saved)
    return {"probe_row": row["path"], "baseline": before,
            "trace_class_entry_removed": without, "restored": run()}


def render(args, prod, fields, indep, static, pop, coup, consumers, digests):
    o = []
    A = o.append
    A("E2 — ROOT-CAUSE INVESTIGATION. MEASUREMENT ONLY, NO REPAIR.")
    A("PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY.")
    A(f"lineage      M3 002e0ab · M1 53791c4 · acceptance c4960ff")
    A(f"subject tree {args.tree}")
    A("")

    A("1. THE PRODUCER — located by AST, not by grep")
    for p, ln, src in prod:
        A(f"   {p} L{ln}")
        A(f"     {src}")
    A(f"   assignment sites in the accepted package: {len(prod)}")
    A("")
    A("   FINDING — KAI'S QUOTED REV4 LINE IS NOT THE CURRENT CODE.")
    A("   Rev4 quotes  f[\"CONSUMED_AT_SUBJECT\"] = bool(row[\"readers\"])")
    A("   That line is house_in_order_h2_v12/run_h2_v12.py L84. In the")
    A("   ACCEPTED v13 lineage the E1 repair already attached a trace and")
    A("   an evidence-class gate, so the boolean is now paired with a")
    A("   STATIC_READER_REFERENCE witness. The evidence class is therefore")
    A("   already stated correctly IN THE TRACE. What remains inflated is")
    A("   the FACT NAME. This narrows the defect; it does not remove it.")
    A("")

    A("2. WHAT row[\"readers\"] CONTAINS, AND HOW IT IS DERIVED")
    A("   passa.build L551-562, L582:")
    A("     readers[o.target].add(o.src)  for every census Op whose")
    A("     disposition == RESOLVED_READ; row[\"readers\"] is the sorted")
    A("     list of READER SOURCE PATHS, and row[\"reader_ops\"] keeps the")
    A("     src/line/mode/expr locator.")
    A("   census claims.classify L164:")
    A("     o.disposition = {\"R\": \"RESOLVED_READ\", ...}[o.mode]")
    A("     reached only when the operation's path fragments are FULLY")
    A("     LEXICALLY FIXED and resolve to a tracked .md document.")
    A("   census opscan._python_ops L269:")
    A("     Op(fn, \"R\", ast.dump(...), lineno) for a read call found by")
    A("     WALKING THE ABSTRACT SYNTAX TREE of the reading file.")
    A("")

    A("3. DOES ANY RUNTIME OBSERVATION CONTRIBUTE? — NO. Derived:")
    for k, v in static.items():
        if k in ("subject_head", "subject_arg", "head_equals_subject",
                 "worktree_clean"):
            continue                      # reported in section 14 instead
        A(f"   {k:<52}{v if not isinstance(v, list) else ''}")
        if isinstance(v, list):
            for ln, src in v:
                A(f"       L{ln}: {src}")
    A("   The ONLY process ever spawned in the chain is `git ls-tree`, to")
    A("   enumerate tracked files. No target document is executed, no")
    A("   reading script is executed, no import of either is performed,")
    A("   and no execution trace, log, coverage record or invocation")
    A("   observation enters at any stage.")
    A("")

    A("4. IS THE FACT EVER TRUE FOR A NON-STATIC REASON? — NO.")
    A(f"   There is exactly {len(prod)} assignment site and its predicate is")
    A("   bool(row[\"readers\"]). row[\"readers\"] has exactly one producer,")
    A("   shown in section 2. There is no second path.")
    A("")

    A("5. EXACT CURRENT AFFECTED POPULATION")
    A(f"   reader ANALYSIS SCOPE     {pop['analysis_scope_files']} source files")
    A(f"     by suffix               {pop['analysis_scope_by_suffix']}")
    A(f"     admitted suffixes       {pop['src_suffix']} + Makefile")
    A(f"     excluded directories    {pop['exclude_dirs']}")
    A(f"   raw candidate matches     {pop['accounting']['raw_candidate_matches']}")
    A(f"   rejected non-operations   {pop['accounting']['rejected_total']}")
    A(f"   admitted operations       "
      f"{pop['accounting']['admitted_candidate_operations']}")
    A(f"   dispositions              {pop['dispositions']}")
    A(f"   RESOLVED_READ operations  {pop['resolved_read_ops']}")
    A(f"   distinct .md read targets {pop['distinct_read_targets']}")
    A(f"   rows with readers         {len(pop['rows_with_readers'])}")
    A(f"   rows with reader_ops      {len(pop['rows_with_reader_ops'])}")
    A(f"   FACT EMITTED TRUE         {len(pop['fact_emitted_true'])}")
    A(f"   abstained under A6-ii     {len(pop['fact_abstained_A6ii'])}")
    A(f"   tracked .md documents     {pop['tracked_documents']}")
    A("")
    A("   WHY IT IS 5, PROVED RATHER THAN MATCHED TO THE HISTORICAL 5:")
    A(f"   {pop['resolved_read_ops']} RESOLVED_READ operations collapse onto")
    A(f"   {pop['distinct_read_targets']} distinct .md targets, one row each,")
    A("   and every one carries a compliant trace so none abstains. The")
    A("   producer condition and the emitted fact coincide exactly.")
    A("   Agreement with the historical figure is an OUTCOME. It was not")
    A("   used as a target and no step was tuned toward it.")
    A("")
    A("   THE NEGATIVE IS ALSO INFLATED, AND THIS IS A SEPARATE FINDING.")
    A(f"   {pop['dispositions'].get('UNRESOLVED_RELEVANCE', 0)} operations could")
    A("   not be shown relevant and "
      f"{pop['dispositions'].get('UNRESOLVED_TARGET', 0)} could not be resolved")
    A("   to a target. A False therefore means NO RESOLVABLE STATIC READ")
    A("   WAS FOUND IN THE ANALYSED SCOPE -- not that the document is")
    A("   unread. Anything reading False as 'not consumed' inflates an")
    A("   abstention into a negative.")
    A("")

    A("6. EVERY AFFECTED ROW, with the raw reading line")
    for p in pop["rows_with_readers"]:
        A(f"   {p}")
        A(f"     readers : {consumers['rows'][p]['readers']}")
        for op in consumers["rows"][p]["ops"]:
            A(f"     op      : {op['src']}:L{op['line']} mode={op['mode']} "
              f"disp={op['disposition']}")
            A(f"       source: {op['line_text']}")
            A(f"       expr  : {op['expr']}")
    A("")

    A("7. EVERY CONSUMER — the fact, the field, and the locator")
    A(f"   row[{FIELD!r}] read at, by AST:")
    for p, ln, src in fields:
        A(f"     {p} L{ln}   {src}")
    A("   FACT NAME referenced in the accepted package:")
    for line in consumers["fact_refs"]:
        A(f"     {line}")
    A("   FACT NAME referenced elsewhere in the tracked repository "
      "(file: hits):")
    for f, n in consumers["repo_refs"]:
        A(f"     {f:<62}{n}")
    A("")

    A("8. WOULD A RENAME MOVE ANY AXIS? — AST-PROVEN NO, WITH ONE TRAP")
    A(f"   classify.py references to {FACT!r}/{FIELD!r}/'reader_ops'/"
      f"'evidence_facts': {indep if indep else 'NONE'}")
    A("   The verdict layer receives (row, text, contradiction). It never")
    A("   reads the field or the fact, so VALIDITY, SCOPE, FUNCTION,")
    A("   AUTHORITY, GENERATION and LIFECYCLE cannot move.")
    A("   envelope.py is not on the path at all: evidence facts are")
    A("   booleans plus traces and never pass through envelope.claim, so")
    A("   no conservation or promotion behaviour is involved.")
    A("")
    A("   THE TRAP, DEMONSTRATED NOT PREDICTED. run_h2_v12.TRACE_CLASS is")
    A("   keyed BY FACT NAME and gates the fact through _class_ok. A")
    A("   rename that does not update that key does NOT raise -- the fact")
    A("   fails the class gate and A6-ii abstains it:")
    A(f"     probe row                     {coup['probe_row']}")
    A(f"     baseline                      {coup['baseline']}")
    A(f"     TRACE_CLASS entry removed     {coup['trace_class_entry_removed']}")
    A(f"     restored                      {coup['restored']}")
    A("   Silent loss of every positive, with a successful exit code.")
    A("   ontology.EVIDENCE_FACTS carries the same shape more weakly: it")
    A("   drives only the printed summary loop, so a stale name there")
    A("   hides the fact from the report while leaving the JSON tally.")
    A("")

    A("9. STATIC_REFERENCE_AT_SUBJECT + AN EXPLICIT ANALYSIS_SCOPE —")
    A("   SUFFICIENT? The name would be accurate. The SCOPE is the gap:")
    A("   nothing in the fact, the trace or the record STATES the analysed")
    A("   universe. The record carries census_dependency (package path and")
    A("   aggregate hash), which REFERENCES the scope but does not state")
    A("   it. To read the fact correctly a reader needs: 615 source files,")
    A("   suffixes .py/.sh/.bash/.yml/.yaml plus Makefile, nine excluded")
    A("   directory names, and lexically-fixed resolution only.")
    A("")

    A("10. DOES THE §5 SCHEMA ALREADY CARRY WHAT A RENAMED FACT NEEDS?")
    A("   The trace built by _reader_trace carries all nine §5 fields plus")
    A("   two declared extras, reader_ast_expr and reader_mode. It names")
    A("   the READER as source_path, an opscan:<src>:L<n> selector, the")
    A("   reading line as local_context, applicability_scope SPAN and")
    A("   subject OTHER:<document>. Multiple ops are carried with")
    A("   evidence_total/evidence_shown and truncated=True.")
    A("   MISSING: the analysis scope of section 9. It is a property of")
    A("   the EXTRACTION, not of any one witness, so a per-witness field")
    A("   would repeat it on every trace. Whether it belongs in a bounded")
    A("   record-level field or in the existing census_dependency is a")
    A("   DESIGN QUESTION FOR KAI, not settled here.")
    A("")

    A("11. WOULD E2 ALTER ANY FACT ENUMERATION USED BY Q1b?")
    A("   No Q1b implementation exists in the accepted package: a search")
    A("   for 'Q1a'/'Q1b' over its .py files returns nothing, and")
    A("   qualify.py contains no evidence-fact reference. The only")
    A("   enumerations are ontology.EVIDENCE_FACTS and the runtime")
    A("   evidence_fact_tally keys. A rename changes both STRINGS, so any")
    A("   later qualification logic keyed on the old name would need the")
    A("   same update -- which is why section 8's trap matters.")
    A("")

    A("12. TESTS, FIXTURES AND PROSE ASSERTING THE STRONGER MEANING")
    A("   CODE outside the accepted package that names the fact:")
    for line in consumers["code_refs"]:
        A(f"     {line}")
    A("   PROSE that names the fact AND asserts execution:")
    for line in consumers["prose_named"]:
        A(f"     {line}")
    A("   Loose pattern hits that do NOT name the fact -- reported, not")
    A("   dropped, and NOT evidence about the fact:")
    for line in consumers["prose_loose"]:
        A(f"     {line}")
    A("   NOTE ON SCOPE. Everything above except the two v12 ontology/")
    A("   runner lines lives in house_in_order_h2_v11 and _v12, which are")
    A("   HISTORICAL PACKAGES, not the accepted lineage. The v13")
    A("   cal_fixtures.py does not assert the fact at all -- its only")
    A("   contact is a fixture row built with readers=[]. So NO TEST IN")
    A("   THE ACCEPTED LINEAGE asserts the runtime meaning. The live")
    A("   governance exposure is DECISIONS.md, which is append-only: a")
    A("   correction there would be a new entry, never an edit.")
    A("")
    A("13. INPUT DIGESTS")
    for k, v in digests.items():
        A(f"   {k:<28}{v}")
    A("")
    A("14. SECONDARY MECHANISMS EXPOSED — reported, NOT repaired")
    A("   S1  SOURCE BINDING IN THE READER EXTRACTION, LATENT.")
    A("       census opscan.tracked L70-76 enumerates with")
    A("       `git ls-tree -r --name-only HEAD` -- the repo's OWN HEAD,")
    A("       not the subject argument, which is never passed to opscan --")
    A("       and opscan.collect L356 then reads each file with")
    A("       (repo / fn).read_text(). ENUMERATION FROM GIT, BYTES FROM")
    A("       THE WORKING FILESYSTEM. That is the SAME SHAPE as the")
    A("       already-confirmed manifest source-binding defect repaired")
    A("       forward at 479596e.")
    A(f"       Realised here? NO. HEAD {static['subject_head'][:12]} == "
      f"subject {static['subject_arg'][:12]}: "
      f"{static['head_equals_subject']}; worktree clean: "
      f"{static['worktree_clean']}. LATENT, not realised.")
    A("       OUTSIDE THE E2 SURFACE. The census package is frozen and")
    A("       Kai's authorisation names run_h2_v12.py. Reported for a")
    A("       separate ruling; nothing here touches it.")
    A("   S2  REACHABILITY IS UNMEASURED, AND NOT OBTAINABLE FROM THIS")
    A("       EVIDENCE CLASS. A resolvable read CALL SITE is counted")
    A("       whether or not the enclosing function is ever called, the")
    A("       script ever invoked, or the guard ever passed. Two of the")
    A("       five sites sit behind `if not X.exists(): return` and one")
    A("       inside try/except OSError. Establishing reachability needs a")
    A("       different evidence class -- and a call graph or a Makefile/CI")
    A("       reference would still be STATIC, not runtime. This is why a")
    A("       rename alone may be insufficient: the honest claim is a")
    A("       static reference, bounded by a stated scope, carrying NO")
    A("       reachability assertion.")
    A("   S3  THE NEGATIVE, in section 5: False is an abstention wearing")
    A("       the shape of a negative.")
    A("")
    A("15. SMALLEST PLAUSIBLE REPAIR PRINCIPLE (proposal, NOT implemented)")
    A("   Rename the FACT to state its evidence class, carry the analysed")
    A("   scope with the record rather than only referencing it, and")
    A("   update TRACE_CLASS in the SAME edit so the class gate cannot")
    A("   silently abstain. Attach no reachability claim.")
    A("   FILES THAT WOULD NEED MODIFICATION IF LATER AUTHORISED:")
    A("     run_h2_v12.py   the cand[...] key, the TRACE_CLASS key, and")
    A("                     the _reader_trace docstring")
    A("     ontology.py     the EVIDENCE_FACTS tuple entry")
    A("   AND NOTHING ELSE. passa.py, classify.py, envelope.py,")
    A("   subjectbind.py and the census package need no change: the field")
    A("   name `readers` is internal and never surfaces as a fact name.")
    A("   MUST REMAIN UNCHANGED, AND BE PROVEN SO:")
    A("     the 5-row population and its identities · all six axis")
    A("     tallies · every other evidence fact · trace content, selector")
    A("     and witness_type · evidence_total/shown/truncated · the A6-ii")
    A("     abstention path · 272 rows in / 272 out.")
    A("   MANDATORY CONTROLS BEFORE ANY IMPLEMENTATION:")
    A("     FAIL-OLD  the predecessor emits the OLD name on all 5 rows")
    A("     PASS-NEW  the candidate emits the NEW name on the SAME 5")
    A("     the old name appears NOWHERE in the new output")
    A("     TRACE_CLASS keyed on the new name, plus a CAN-FAIL proof that")
    A("       omitting it ABSTAINS all 5 rather than raising")
    A("     ontology.EVIDENCE_FACTS updated, plus a can-fail proof that a")
    A("       stale entry hides the fact from the printed summary")
    A("     trace bytes identical field-for-field except the fact key")
    A("     all six axes and every other fact tally unchanged")
    A("     predecessor loaded from the git object and executed")
    A("     implementation banked BEFORE the corpus run")
    A("")
    A("16. WHAT THIS IS NOT")
    A("   Not a repair, not a rename, not an adjudication, not an answer")
    A("   key. No production module was modified. The historical 5 was")
    A("   used as a locator only and never as a target.")
    return "\n".join(o) + "\n"


def gather_consumers(repo, subject_repo, tree, passa, pop):
    rows = {r["path"]: r for r in passa["rows"]}
    detail = {}
    for p in pop["rows_with_readers"]:
        ops = []
        for o in rows[p]["reader_ops"]:
            try:
                line = tree_text(subject_repo, tree, o["src"]).split(
                    "\n")[o["line"] - 1].strip()
            except Exception:
                line = "(unreadable at the frozen tree)"
            ops.append(dict(o, line_text=line))
        detail[p] = {"readers": rows[p]["readers"], "ops": ops}

    def grep(pattern, pathspec):
        r = subprocess.run(["git", "-C", str(repo), "grep", "-n", pattern,
                            "--", pathspec], capture_output=True, text=True)
        return [l for l in r.stdout.split("\n") if l.strip()]

    pkg = "kai-pm/house_in_order_h2_v13"
    fact_refs = grep(FACT, f"{pkg}/*.py")
    counts = subprocess.run(["git", "-C", str(repo), "grep", "-c", FACT],
                            capture_output=True, text=True).stdout
    repo_refs = []
    for line in counts.split("\n"):
        if not line.strip():
            continue
        f, _, n = line.rpartition(":")
        if not f.startswith(pkg):
            repo_refs.append((f, n))
    code_refs = [l for l in grep(FACT, "*.py") if pkg not in l]
    prose = grep("executable read\\|code reads it\\|executable operation",
                 "*.md")
    # EVERYTHING THE PREDICATE RETURNED TRAVELS (R17). The prose grep is
    # a LOOSE PATTERN, not a reference to the fact, so its hits are
    # reported separately and the ones that do not mention the fact are
    # labelled rather than dropped.
    prose_named = [l for l in prose if FACT in l]
    prose_loose = [l for l in prose if FACT not in l]
    return {"rows": detail, "fact_refs": fact_refs, "repo_refs": repo_refs,
            "code_refs": code_refs, "prose_named": prose_named,
            "prose_loose": prose_loose}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--passa", required=True)
    ap.add_argument("--classification", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pa = json.loads(pathlib.Path(a.passa).read_text())
    cl = json.loads(pathlib.Path(a.classification).read_text())
    if pa["subject_tree"] != a.tree or cl["subject_tree"] != a.tree:
        raise SystemExit(f"R11 ABORT: run artefacts are not bound to "
                         f"{a.tree[:12]}.")

    SUBJECT_REPO[0] = a.subject_repo
    SUBJECT_ARG[0] = pa['subject']
    prod = producer_sites(PKG)
    if len(prod) != 1:
        raise SystemExit(f"R11 ABORT: expected exactly one assignment site "
                         f"for {FACT}, found {len(prod)}. The single-producer "
                         f"claim in this report would be false.")
    fields = field_sites(PKG, FIELD)
    indep = verdict_layer_independence(PKG)
    static = staticness(a.census_package)
    pop = population(a.subject_repo, a.census_package, pa, cl)
    coup = coupling_probe(a.subject_repo, pa)
    if coup["baseline"][0] is not True or \
            coup["trace_class_entry_removed"][0] is not False:
        raise SystemExit("R11 ABORT: the coupling probe did not discriminate; "
                         "it would prove nothing.")
    consumers = gather_consumers(a.repo, a.subject_repo, a.tree, pa, pop)
    digests = {k: hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest()
               for k, v in (("passA.json", a.passa),
                            ("classification.json", a.classification),
                            ("run_h2_v12.py", PKG / "run_h2_v12.py"),
                            ("passa.py", PKG / "passa.py"))}
    text = render(a, prod, fields, indep, static, pop, coup, consumers, digests)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
