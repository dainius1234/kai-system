#!/usr/bin/env python3
"""Calibration for the A/B/C model-load denominator.

The operator's rule for this sweep, and the reason this file is longer
than the usual smoke test:

    The detector must prove it can express both inclusion AND exclusion.
    Do not trust a denominator simply because it prints one.

Six capabilities were demanded before the number could be believed. Each
has a named scenario here:

    1. include memu-core                    known-good applicable case
    2. include memu-core-introspect         known-bad applicable case
    3. exclude a non-model service          known exclusion
    4. distinguish A from B                 the tree cannot demonstrate
                                            this — every member of A is
                                            egress-restricted — so it is
                                            demonstrated on a fixture,
                                            plus a real known-negative
                                            for the egress classifier
    5. preserve UNKNOWN in C                absence of evidence stays
                                            absence of evidence
    6. separate reachability from behaviour import != call, and
                                            module-scope != lazy

THE DEFECT THIS FILE ALREADY CAUGHT
===================================

The first tracer decided module scope with `lineno in module_level_lines`,
built by `ast.walk`-ing each top-level statement — which descends into
function bodies, so every line qualified. All four hits printed `IMPORT`.
Two of them load lazily inside a memoised getter. That error points at
the most expensive architecture (bake a model into four images) on
evidence that does not exist, and the printed output looked entirely
reasonable.

So `test_lazy_is_not_import` is not a nicety. It is the assertion the
instrument failed, and the one that must keep failing if the scope logic
regresses.

A SECOND DEFECT, CAUGHT THE SAME WAY
====================================

The COPY parser appended the source basename unconditionally, mapping
`COPY common/ ./common/` to `/app/common/common`. Every `from common…`
import in every image then resolved to nothing. It found the same five
services, so the answer looked stable — which is precisely why a wrong
traversal that agrees with the right one is worth an assertion of its
own: it will not agree on the next question.
"""
from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import report_model_load_denominator as md  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 25
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


_ROWS = None


def rows():
    """The real tree, surveyed once."""
    global _ROWS
    if _ROWS is None:
        _ROWS = md.survey()
    return _ROWS


def row(service: str):
    return next((r for r in rows() if r["service"] == service), None)


def hits(src: str):
    """Loader hits in a snippet: [(lineno, timing, what)]."""
    return md._loader_hits(ast.parse(textwrap.dedent(src)))


# ── fixture plumbing ─────────────────────────────────────────────────

def fixture(root: Path, compose: str, files: dict) -> None:
    """Write a miniature repo: all three compose names must exist,
    because `require()` refuses to run on a missing input and a fixture
    that silently skipped one would be testing a different code path."""
    (root / "docker-compose.full.yml").write_text(textwrap.dedent(compose))
    for other in ("docker-compose.minimal.yml", "docker-compose.sovereign.yml"):
        (root / other).write_text("services: {}\n")
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(body))


MODEL_APP = """
    from sentence_transformers import SentenceTransformer as _ST
    model = _ST("all-MiniLM-L6-v2")
    """
PLAIN_DF = """
    FROM python:3.11-slim
    WORKDIR /app
    COPY svc/requirements.txt ./
    RUN pip install --no-cache-dir -r requirements.txt
    COPY svc/app.py ./
    CMD ["python", "app.py"]
    """


# ══ 1 & 2 — the two known applicable cases ═══════════════════════════

def test_memu_core_is_included_with_its_proven_contract() -> None:
    """Known-good. Every column of the offline contract is satisfied, and
    each is asserted separately: a detector that cannot see the contract
    cannot see its absence either."""
    scenario("memu-core included")
    r = row("memu-core")
    check("present", r is not None, "not in the survey at all")
    if not r:
        return
    check("in population A", r["reach"] == md.REACH_TRACED, r["reach"])
    check("loads at IMPORT", r["timing"] == md.TIMING_IMPORT, r["timing"])
    check("traced to memu-core/app.py",
          r["path"].startswith("memu-core/app.py:"), r["path"])
    check("library installed", r["installed"] == md.YES, r["installed"])
    check("asset baked", r["baked"] == md.YES, r["baked"])
    check("cache path pinned", r["cache"].startswith("HF_HOME="), r["cache"])
    check("offline enforced in image", r["offline"] == "IMAGE", r["offline"])
    check("build fails closed", r["fail_closed"] == md.YES, r["fail_closed"])
    check("no mount shadows the cache", r["shadow"] == md.NO, r["shadow"])


def test_memu_core_introspect_is_included_and_exposes_the_gap() -> None:
    """Known-bad. Same population, same timing, none of the contract —
    which is the entire finding. If the detector reported it identically
    to memu-core it would be measuring the source and calling it the
    image."""
    scenario("memu-core-introspect included")
    r = row("memu-core-introspect")
    check("present", r is not None)
    if not r:
        return
    check("in population A", r["reach"] == md.REACH_TRACED, r["reach"])
    check("loads at IMPORT", r["timing"] == md.TIMING_IMPORT, r["timing"])
    check("shares memu-core/app.py",
          r["path"].startswith("memu-core/app.py:"), r["path"])
    check("no baked asset", r["baked"] == md.NO, r["baked"])
    check("no pinned cache", r["cache"] == md.NO, r["cache"])
    check("offline NOT enforced", r["offline"] == "NOT ENFORCED", r["offline"])
    check("listener is gated on the load", r["listener_gated"] == md.YES,
          r["listener_gated"])
    core = row("memu-core")
    check("differs from memu-core on the contract, not the path",
          core and core["path"] == r["path"] and core["baked"] != r["baked"],
          f"{core['path'] if core else '?'} / {r['path']}")


# ══ 3 — the known exclusion ══════════════════════════════════════════

def test_a_real_non_model_service_is_excluded_after_being_examined() -> None:
    """`tool-gate` is runnable, egress-restricted and loads no model.

    The second half matters more than the first: an exclusion is only
    evidence if the service was actually inspected. A Dockerfile that
    failed to parse would also produce "not in A", and would be
    indistinguishable from this — I-1, boundary blindness, in the one
    place nobody looks for it."""
    scenario("non-model service excluded")
    r = row("tool-gate")
    check("present", r is not None)
    if not r:
        return
    check("NOT in population A", r["reach"] == md.REACH_NONE, r["reach"])
    check("it was examined, not skipped", r["dockerfile"].endswith("Dockerfile"),
          r["dockerfile"])
    image = md.Image(REPO / r["dockerfile"])
    start, why = image.entry_module()
    check("its entrypoint resolved", start is not None, why)
    check("the entrypoint really exists", start is not None and start.exists())
    check("and the trace genuinely ran and found nothing",
          start is not None and md.trace(image, start)[0] is None)


def test_an_import_without_a_call_is_not_a_model_load() -> None:
    """The hazard rule, as an assertion. Importing is not loading."""
    scenario("import is not a call")
    check("bare package import", hits("import sentence_transformers") == [], "")
    check("from-import with no call",
          hits("from sentence_transformers import SentenceTransformer") == [], "")
    check("the name mentioned in a string",
          hits('x = "SentenceTransformer"') == [], "")
    check("a same-named local function is not the library's",
          hits("""
               def pipeline(x): return x
               y = pipeline(3)
               """) == [], "a local `pipeline` was counted as transformers.pipeline")


# ══ 4 — A is not B ═══════════════════════════════════════════════════

def test_a_model_service_with_egress_is_in_A_but_not_in_B() -> None:
    """The tree cannot show this: every member of A happens to be
    egress-restricted. A denominator whose separation is never exercised
    is a denominator that could have collapsed silently, so it is
    exercised on a fixture."""
    scenario("A minus B on a fixture")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root, """
            networks:
              inner: {internal: true}
              outer: {}
            services:
              open-svc:
                build: {context: ., dockerfile: svc/Dockerfile}
                networks: [outer]
              shut-svc:
                build: {context: ., dockerfile: svc/Dockerfile}
                networks: [inner]
            """, {
            "svc/Dockerfile": PLAIN_DF,
            "svc/app.py": MODEL_APP,
            "svc/requirements.txt": "sentence-transformers>=2.7.0\n",
        })
        surveyed = md.survey(root)
        a, b, c = md.populations(surveyed)
        names_a = {r["service"] for r in a}
        names_b = {r["service"] for r in b}
        check("both reach a loader", names_a == {"open-svc", "shut-svc"}, str(names_a))
        check("only the internal-only one is in B", names_b == {"shut-svc"},
              str(names_b))
        check("A is strictly larger than B here", len(a) > len(b),
              f"{len(a)} vs {len(b)}")
        check("C is empty without citations", c == [], str(c))


def test_the_egress_classifier_has_a_real_known_negative() -> None:
    """`weather-service` sits on `egress-net`, which is not internal. If
    this ever reads EGRESS-RESTRICTED the classifier has inverted and B
    would swallow the whole tree."""
    scenario("egress known-negative")
    r = row("weather-service")
    check("present", r is not None)
    if r:
        check("has egress", r["egress"] == md.EGRESS_AVAILABLE, r["egress"])
    core = row("memu-core")
    check("and the known-positive still restricts",
          core and core["egress"] == md.EGRESS_RESTRICTED,
          core["egress"] if core else "?")


def test_a_service_on_no_network_is_not_called_restricted() -> None:
    """Absence of a `networks:` key is the implicit default network, which
    has egress. Reading absence as restriction would be a scope LARGER
    than reality — the inverted form, which fails things that are right."""
    scenario("no networks key is not restriction")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root, """
            networks:
              inner: {internal: true}
            services:
              bare-svc:
                build: {context: ., dockerfile: svc/Dockerfile}
            """, {
            "svc/Dockerfile": PLAIN_DF,
            "svc/app.py": MODEL_APP,
            "svc/requirements.txt": "sentence-transformers\n",
        })
        r = md.survey(root)[0]
        check("classified as having egress", r["egress"] == md.EGRESS_AVAILABLE,
              r["egress"])


# ══ 5 — UNKNOWN survives ═════════════════════════════════════════════

def test_runtime_status_stays_unknown_without_a_citation() -> None:
    scenario("UNKNOWN preserved in C")
    a, b, c = md.populations(rows())
    unknown = [r["service"] for r in b if r not in c]
    check("some member of B has no runtime evidence", len(unknown) > 0,
          "every member is cited — check the evidence table is not "
          "inventing rows")
    for name in unknown:
        r = row(name)
        check(f"{name} says UNKNOWN explicitly",
              r["evidence"].startswith("UNKNOWN"), r["evidence"])
    check("C is a subset of B", all(r in b for r in c), "")
    check("C is smaller than B", len(c) < len(b), f"{len(c)} vs {len(b)}")


def test_every_citation_resolves_in_the_decision_log() -> None:
    """I-8: the source of the expected answer is not the thing under
    test. A citation that does not resolve is a claim with no artefact
    behind it, and the report refuses to print C at all in that case."""
    scenario("citations resolve")
    check("all citations resolve", md.citations_resolve() == [],
          str(md.citations_resolve()))
    check("evidence tables are non-empty", len(md.RUNTIME_EVIDENCE) > 0)


def test_an_unresolvable_citation_is_detected() -> None:
    """Known-positive for the citation checker itself — otherwise 'all
    citations resolve' is a sentence that has never been able to fail."""
    scenario("bad citation detected")
    saved = dict(md.RUNTIME_EVIDENCE)
    try:
        md.RUNTIME_EVIDENCE["ghost-service"] = {
            "observation": "invented", "cite": "D99999-NOT-A-REAL-ENTRY"}
        bad = md.citations_resolve()
        check("the fake citation is reported", any("ghost-service" in b for b in bad),
              str(bad))
    finally:
        md.RUNTIME_EVIDENCE.clear()
        md.RUNTIME_EVIDENCE.update(saved)
    check("the table is restored", md.citations_resolve() == [], "")


# ══ 6 — reachability is not behaviour ════════════════════════════════

def test_module_scope_is_import() -> None:
    scenario("module scope is IMPORT")
    got = hits("""
        from sentence_transformers import SentenceTransformer as _ST
        m = _ST("all-MiniLM-L6-v2")
        """)
    check("one hit", len(got) == 1, str(got))
    check("timing is IMPORT", got and got[0][1] == md.TIMING_IMPORT, str(got))


def test_lazy_is_not_import() -> None:
    """THE assertion. The first tracer failed exactly this."""
    scenario("lazy is not IMPORT")
    got = hits("""
        from sentence_transformers import SentenceTransformer as _ST
        _M = None
        def _get():
            global _M
            if _M is None:
                _M = _ST("all-MiniLM-L6-v2")
            return _M
        def handler():
            return _get()
        """)
    check("one hit", len(got) == 1, str(got))
    check("timing is LAZY", got and got[0][1] == md.TIMING_LAZY,
          f"{got} — a call inside a function nobody invokes at module "
          f"scope is not an import-time load")


def test_a_function_invoked_at_module_scope_is_still_import() -> None:
    """The other direction. Hiding a load behind a `def` that the module
    then calls changes nothing about when it runs, and a detector that
    only looked for bare module-level calls would grant a free pass for
    a one-line refactor."""
    scenario("module-scope invocation is IMPORT")
    got = hits("""
        from sentence_transformers import SentenceTransformer as _ST
        def _boot():
            return _ST("all-MiniLM-L6-v2")
        MODEL = _boot()
        """)
    check("one hit", len(got) == 1, str(got))
    check("timing is IMPORT", got and got[0][1] == md.TIMING_IMPORT, str(got))


def test_a_startup_hook_is_startup_not_import_and_not_lazy() -> None:
    scenario("startup hook is STARTUP")
    got = hits("""
        from sentence_transformers import SentenceTransformer as _ST
        @app.on_event("startup")
        async def _boot():
            global M
            M = _ST("all-MiniLM-L6-v2")
        """)
    check("one hit", len(got) == 1, str(got))
    check("timing is STARTUP", got and got[0][1] == md.TIMING_STARTUP, str(got))


def test_from_pretrained_is_a_known_positive() -> None:
    """`LOADER_METHODS` is the one name-based rule in the instrument, so
    it carries the burden of proof: it must fire, and the surrounding
    origin-resolved rules must not fire on look-alikes."""
    scenario("from_pretrained fires")
    got = hits("""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        """)
    check("fires", len(got) == 1, str(got))
    check("at IMPORT", got and got[0][1] == md.TIMING_IMPORT, str(got))
    check("a plain attribute call does not fire",
          hits("x = obj.encode('hi')") == [], "")


def test_the_real_tree_reports_both_import_and_lazy() -> None:
    """A tracer that could only ever say IMPORT would be reporting on
    itself. Both classes must occur in the measured population."""
    scenario("both timings present in the tree")
    a, _, _ = md.populations(rows())
    timings = {r["timing"] for r in a if r["reach"] == md.REACH_TRACED}
    check("IMPORT occurs", md.TIMING_IMPORT in timings, str(timings))
    check("LAZY occurs", md.TIMING_LAZY in timings, str(timings))
    lazy = sorted(r["service"] for r in a if r["timing"] == md.TIMING_LAZY)
    check("the lazy ones are agentic and fusion-engine",
          lazy == ["agentic", "fusion-engine"], str(lazy))
    for name in lazy:
        check(f"{name}'s listener is NOT gated on the model",
              row(name)["listener_gated"] == md.NO, row(name)["listener_gated"])


def test_an_uninstalled_library_is_reported_as_such() -> None:
    """Source-reachable is not runtime-reachable. `agentic` traces to a
    loader whose library its image never installs, so the guarded import
    raises and the fallback runs — which is the BACKEND_UNAVAILABLE that
    #47 measured at runtime, from an entirely different direction."""
    scenario("uninstalled library reported")
    for name in ("agentic", "fusion-engine"):
        r = row(name)
        check(f"{name} traced", r and r["reach"] == md.REACH_TRACED,
              r["reach"] if r else "missing")
        check(f"{name} library NOT installed", r and r["installed"] == md.NO,
              r["installed"] if r else "?")
        check(f"{name} import is guarded", r and r["guarded"] == md.YES,
              r["guarded"] if r else "?")
    core = row("memu-core")
    check("and memu-core's IS installed — the check can say yes",
          core and core["installed"] == md.YES, core["installed"] if core else "?")


def test_a_module_the_image_never_copies_is_unreachable() -> None:
    """The claim "the trace runs inside the image" needs a case where the
    repo and the image disagree. Nothing in the tree exercises it today,
    so it is exercised here — an untested capability is a capability
    nobody has."""
    scenario("uncopied module unreachable")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root, """
            networks: {inner: {internal: true}}
            services:
              partial:
                build: {context: ., dockerfile: svc/Dockerfile}
                networks: [inner]
            """, {
            "svc/Dockerfile": """
                FROM python:3.11-slim
                WORKDIR /app
                COPY svc/requirements.txt ./
                RUN pip install --no-cache-dir -r requirements.txt
                COPY svc/app.py ./
                CMD ["python", "app.py"]
                """,
            "svc/app.py": "import loader\n",
            "svc/loader.py": MODEL_APP,
            "svc/requirements.txt": "sentence-transformers\n",
        })
        r = md.survey(root)[0]
        # It does not become NOT REACHABLE, and that is the right answer:
        # the library IS installed, so a third-party loader could still
        # run. What must be true is that the TRACE did not reach the
        # uncopied module — the instrument degrades to "candidate,
        # timing unknown" instead of claiming a path it cannot see.
        check("the trace does not reach an uncopied module",
              r["reach"] != md.REACH_TRACED, f"{r['reach']} {r['path']}")
        check("and it claims no path", r["path"] == "", r["path"])
        check("and no timing", r["timing"] == md.TIMING_UNKNOWN, r["timing"])
        # and the same tree WITH the copy must be included, or the
        # exclusion above proves nothing about COPY.
        (root / "svc" / "Dockerfile").write_text(textwrap.dedent("""
            FROM python:3.11-slim
            WORKDIR /app
            COPY svc/requirements.txt ./
            RUN pip install --no-cache-dir -r requirements.txt
            COPY svc/app.py ./
            COPY svc/loader.py ./
            CMD ["python", "app.py"]
            """))
        r2 = md.survey(root)[0]
        check("in A once it IS copied", r2["reach"] == md.REACH_TRACED,
              f"{r2['reach']} {r2['path']}")
        check("and it resolved through the import", "loader.py" in r2["path"],
              r2["path"])


# ══ instrument integrity ═════════════════════════════════════════════

def test_the_copy_parser_maps_a_directory_to_its_contents() -> None:
    """Docker copies the CONTENTS of a directory. The first parser
    appended the basename and produced `/app/common/common`."""
    scenario("COPY of a directory")
    image = md.Image(REPO / "memu-core" / "Dockerfile")
    got = image.resolve("/app/common/runtime.py")
    check("common/ maps to the repo's common/",
          got == REPO / "common" / "runtime.py", str(got))
    check("app.py maps to memu-core/app.py",
          image.resolve("/app/app.py") == REPO / "memu-core" / "app.py",
          str(image.resolve("/app/app.py")))
    check("an uncopied path resolves to None",
          image.resolve("/app/introspect_app.py") is None,
          str(image.resolve("/app/introspect_app.py")))
    agentic = md.Image(REPO / "agentic" / "Dockerfile")
    check("`COPY agentic/ ./` maps the workdir itself",
          agentic.resolve("/app/router.py") == REPO / "agentic" / "router.py",
          str(agentic.resolve("/app/router.py")))
    check("and a nested COPY still wins by longest prefix",
          agentic.resolve("/app/common/runtime.py") == REPO / "common" / "runtime.py",
          str(agentic.resolve("/app/common/runtime.py")))


def test_a_uvicorn_entrypoint_resolves() -> None:
    """Four Dockerfiles start `uvicorn app:app`. A parser that only knew
    `python app.py` would silently exclude them, and the exclusion would
    read the same as a clean result."""
    scenario("uvicorn entrypoint")
    image = md.Image(REPO / "cortex" / "Dockerfile")
    start, why = image.entry_module()
    check("resolved", start is not None, why)
    check("to cortex/app.py", start == REPO / "cortex" / "app.py", str(start))


def test_a_fail_open_bake_is_not_fail_closed() -> None:
    scenario("fail-open bake detected")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fixture(root, """
            networks: {inner: {internal: true}}
            services:
              soft:
                build: {context: ., dockerfile: svc/Dockerfile}
                networks: [inner]
            """, {
            "svc/Dockerfile": """
                FROM python:3.11-slim
                WORKDIR /app
                COPY svc/requirements.txt ./
                RUN pip install --no-cache-dir -r requirements.txt
                ENV HF_HOME=/opt/hf
                RUN python -c "import sentence_transformers; \\
                    sentence_transformers.SentenceTransformer('x')" || true
                COPY svc/app.py ./
                CMD ["python", "app.py"]
                """,
            "svc/app.py": MODEL_APP,
            "svc/requirements.txt": "sentence-transformers\n",
        })
        r = md.survey(root)[0]
        check("the bake is seen", r["baked"] == md.YES, r["baked"])
        check("but it is NOT fail-closed", r["fail_closed"] == md.NO,
              r["fail_closed"])
    core = row("memu-core")
    check("and the real fail-closed bake still reads YES",
          core and core["fail_closed"] == md.YES,
          core["fail_closed"] if core else "?")


def test_mount_shadow_fires_and_does_not_over_fire() -> None:
    """Both directions on one line each. `memu-core` puts its cache in
    /opt precisely so no volume can shadow it; that decision is only
    protected if the check could have said YES."""
    scenario("mount shadow both ways")
    check("a volume at the cache path shadows",
          md.mount_shadows("HF_HOME=/data/hf_cache", ["vol:/data"]), "")
    check("a volume exactly at the cache shadows",
          md.mount_shadows("HF_HOME=/data/hf", ["vol:/data/hf"]), "")
    check("an unrelated mount does not",
          not md.mount_shadows("HF_HOME=/opt/hf_cache", ["vol:/data/turbovec"]), "")
    check("a sibling prefix does not",
          not md.mount_shadows("HF_HOME=/data/hf_cache", ["vol:/data/hf"]), "")
    check("no cache means no shadow", not md.mount_shadows(None, ["vol:/"]), "")
    for name in ("memu-core", "memu-graph"):
        r = row(name)
        check(f"{name} is not shadowed today", r and r["shadow"] == md.NO,
              r["shadow"] if r else "?")


def test_delegation_is_recognised_both_ways() -> None:
    """`ollama-pull` runs a fetch verb on an internal-only network and is
    nonetheless correct, because it points the fetch at `ollama`. A rule
    of "fetch verb + no egress = defect" would have reported a working
    design as broken."""
    scenario("delegation both ways")
    dep = md.deployments()
    check("ollama is a known service", "ollama" in dep, "")
    delegated = md._delegation("OLLAMA_HOST=ollama:11434 ollama pull x",
                               dep, "ollama-pull")
    check("delegation is found", "delegated to `ollama`" in delegated, delegated)
    alone = md._delegation("ollama pull x", dep, "ollama-pull")
    check("and its absence is reported", "performs the fetch itself" in alone,
          alone)
    r = row("ollama-pull")
    check("the row records it", r and "delegated" in r["why"],
          r["why"] if r else "missing")
    check("and it is in A by COMMAND, not by tracing",
          r and r["reach"] == md.REACH_COMMAND, r["reach"] if r else "?")


def test_the_populations_account_for_every_service() -> None:
    """No service may vanish between the survey and the printout. A
    denominator that does not add up is a denominator with a silent
    filter in it."""
    scenario("populations account for all")
    all_rows = rows()
    a, b, c = md.populations(all_rows)
    excluded = [r for r in all_rows if r not in a]
    check("A + excluded == all", len(a) + len(excluded) == len(all_rows),
          f"{len(a)} + {len(excluded)} != {len(all_rows)}")
    check("every excluded row carries a reason or a clean trace",
          all(r["why"] or r["reach"] == md.REACH_NONE for r in excluded), "")
    check("B is a subset of A", all(r in a for r in b), "")
    check("no row is in C without being in B", all(r in b for r in c), "")
    check("the three populations are not equal",
          not (len(a) == len(b) == len(c)),
          "A == B == C would mean the distinctions collapsed")
    check("the survey covers every compose service",
          len(all_rows) == len(md.deployments()), "")


def test_third_party_candidates_are_not_silently_dropped() -> None:
    """`memu-graph` installs `transformers` and no repo source calls a
    loader — the load lives inside cognee. A tracer-only population would
    have excluded it and called the denominator complete."""
    scenario("third-party candidate kept")
    r = row("memu-graph")
    check("present", r is not None)
    if not r:
        return
    check("classified as a candidate, not a member",
          r["reach"] == md.REACH_THIRD_PARTY, r["reach"])
    check("timing is UNKNOWN, not guessed", r["timing"] == md.TIMING_UNKNOWN,
          r["timing"])
    check("listener-gating is UNKNOWN too", r["listener_gated"] == md.UNKNOWN,
          r["listener_gated"])
    check("and it still enters population A", r in md.populations(rows())[0], "")


def run_all() -> None:
    test_memu_core_is_included_with_its_proven_contract()
    test_memu_core_introspect_is_included_and_exposes_the_gap()
    test_a_real_non_model_service_is_excluded_after_being_examined()
    test_an_import_without_a_call_is_not_a_model_load()
    test_a_model_service_with_egress_is_in_A_but_not_in_B()
    test_the_egress_classifier_has_a_real_known_negative()
    test_a_service_on_no_network_is_not_called_restricted()
    test_runtime_status_stays_unknown_without_a_citation()
    test_every_citation_resolves_in_the_decision_log()
    test_an_unresolvable_citation_is_detected()
    test_module_scope_is_import()
    test_lazy_is_not_import()
    test_a_function_invoked_at_module_scope_is_still_import()
    test_a_startup_hook_is_startup_not_import_and_not_lazy()
    test_from_pretrained_is_a_known_positive()
    test_the_real_tree_reports_both_import_and_lazy()
    test_an_uninstalled_library_is_reported_as_such()
    test_a_module_the_image_never_copies_is_unreachable()
    test_the_copy_parser_maps_a_directory_to_its_contents()
    test_a_uvicorn_entrypoint_resolves()
    test_a_fail_open_bake_is_not_fail_closed()
    test_mount_shadow_fires_and_does_not_over_fire()
    test_delegation_is_recognised_both_ways()
    test_the_populations_account_for_every_service()
    test_third_party_candidates_are_not_silently_dropped()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Model-Load Denominator Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
