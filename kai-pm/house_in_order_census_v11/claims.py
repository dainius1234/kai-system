#!/usr/bin/env python3
"""DISPOSITIONS, EXCLUSION WITNESSES AND SCOPED CLAIMS — Census v1.1.

THREE RULINGS FROM D341 ARE IMPLEMENTED HERE.

1. NO_WRITER IS RENAMED, NOT WEAKENED.
   Kai: "Do not remove the semantic possibility merely because the
   current subject does not exhibit it" -- that would let the corpus
   define the ontology, which is the inversion P10 forbids. But the
   unqualified name overstated its proof scope. It never meant "nothing
   writes this file"; it meant "no candidate writer within the declared
   static source population and analysis model". So it is now called
   NO_WRITER_WITHIN_ANALYZED_SCOPE and it carries its scope with it.
   The closure rules are NOT relaxed by one inch to make it reachable.

2. SYNTAX IS NOT SEMANTICS -- THE REMOTE_URI CORRECTION.
   v1.0 excluded any target containing "://" as a "remote URI, not a
   repository path". But this analyser observes FILESYSTEM writes and
   shell redirections. `open("https://x.md", "w")` performs no HTTP
   request; to Python it is a pathname. `echo x > https://x.md` is a
   filesystem redirection. Containing "://" proves nothing about the
   operation. URI syntax is therefore recorded as an OBSERVATION and is
   never, by itself, an exclusion witness; such a target is reasoned
   about as the path string it is.

   R6 (fix the class, not the instance) forces the same correction on
   ABSOLUTE paths, which are the identical defect wearing a different
   character: v1.0 excluded any target starting with "/" as "outside the
   repository target domain", which is false -- /home/user/repo/data/X.md
   is absolute AND inside the repository. Now an absolute literal is
   excluded only by the constructive test below, which compares it with
   the target instead of inspecting its first character.

3. EVERY EXCLUSION WITNESS IS POSITIVE AND CONSTRUCTIVE.
   "It doesn't look like it points at T" is the dismissed-on-absence
   defect P13 exists to forbid. Each witness below names a property the
   operation DEMONSTRABLY HAS.
"""
from __future__ import annotations
import collections
import pathlib

ALPHABETS = {
    "DISPOSITIONS": (
        "RESOLVED_READ", "RESOLVED_WRITE", "READ_AND_WRITE",
        "RESOLVED_NON_DOCUMENT_TARGET", "UNRESOLVED_TARGET",
        "UNRESOLVED_RELEVANCE"),
    # OBSERVATION ONLY. Never an exclusion witness on its own (ruling 2).
    "PATH_SYNTAX": ("REPO_RELATIVE", "ABSOLUTE", "URI_SYNTAX",
                    "SHELL_VARIABLE", "DYNAMIC_UNKNOWN"),
    "TARGET_DISPOSITIONS": ("REACHES_T", "EXCLUDED_FROM_T", "COULD_REACH_T"),
    "EXCLUSION_WITNESSES": (
        "FIXED_COMPLETE_PATH_DIFFERS",
        "FIXED_BASENAME_DIFFERS",
        "FIXED_DIRECTORY_DISJOINT"),
    "CLAIMS": ("PROVEN_WRITE_RELATION", "NO_PROVEN_WRITER",
               "NO_WRITER_WITHIN_ANALYZED_SCOPE"),
}

NON_DOC_EXT = (".json", ".py", ".txt", ".log", ".yml", ".yaml", ".sh",
               ".jsonl", ".csv", ".html", ".css", ".js", ".toml", ".ini",
               ".cfg", ".lock", ".png", ".jpg", ".svg", ".pdf", ".zip",
               ".tar", ".gz", ".sql", ".env", ".pyc", ".xml")


def _fixed(frags):
    """True when every fragment is a literal with no dynamic component."""
    return bool(frags) and not any(
        ("$" in f or "{" in f or "*" in f) for f in frags)


def _joined(frags):
    return "/".join(f for f in frags if f)


def path_syntax(frags, dynamic=False):
    """A SYNTACTIC OBSERVATION about the target expression.

    Deliberately has no authority over exclusion. D341 F3: v1.0 let this
    classification manufacture negative claims, and two of its values had
    never been calibrated at all.
    """
    if not frags:
        return "DYNAMIC_UNKNOWN"
    raw = frags[0] if len(frags) == 1 else _joined(frags)
    s = raw.strip().strip("\"'`")
    if "://" in s:
        return "URI_SYNTAX"
    if any(c in s for c in "${}*"):
        return "SHELL_VARIABLE"
    if dynamic:
        return "DYNAMIC_UNKNOWN"
    if s.startswith("/"):
        return "ABSOLUTE"
    return "REPO_RELATIVE"


def _resolve_exact(frags, docset):
    """EXACT tracked-path resolution only. No basename fallback: a write
    claim must resolve to a real repository-relative path or stay open."""
    if not frags:
        return None
    joined = _joined(frags)
    for cand in (joined, frags[-1]):
        c = cand.lstrip("./").lstrip("/")
        if c in docset:
            return c
    return None


def classify(ops, docs, tracked_all):
    """Assign EXACTLY ONE disposition to every admitted operation."""
    docset = set(docs)
    for o in ops:
        frags = o.frags
        if not frags:
            o.disposition = "UNRESOLVED_RELEVANCE"
            continue

        lexical_dynamic = any(
            ("$" in f or "{" in f or "*" in f) for f in frags)
        # RELEVANCE and TARGET are separate questions, and conflating
        # them is a MISBINDING. `(ROOT / "README.md").read_text()` has an
        # unresolvable PREFIX, but its relevance is not in doubt: the
        # fixed final component ends .md. Filing it under
        # UNRESOLVED_RELEVANCE would assert we cannot tell whether the
        # target is a document, which is false -- and would silently
        # erase a real read relation from the record.
        fully_fixed = not lexical_dynamic and not o.dynamic
        last = frags[-1]

        # POSITIVE WITNESS for non-document. A fixed final extension is
        # decisive however the prefix resolves.
        if not lexical_dynamic:
            joined = _joined(frags).lstrip("./").lstrip("/")
            if fully_fixed and joined in tracked_all \
                    and not joined.endswith(".md"):
                o.disposition = "RESOLVED_NON_DOCUMENT_TARGET"
                continue
            if last.endswith(NON_DOC_EXT):
                o.disposition = "RESOLVED_NON_DOCUMENT_TARGET"
                continue

        if not any(f.endswith(".md") for f in frags):
            # No .md evidence and no non-document witness: we genuinely
            # cannot establish whether this operation is relevant.
            o.disposition = "UNRESOLVED_RELEVANCE"
            continue

        if not fully_fixed:
            # Relevance PROVEN (.md), target NOT proven. v1.0 resolved
            # these by matching the literal suffix against the tracked
            # tree, which is the D332 misbinding: str(tmp) + "/SOUL.md"
            # has the same shape and denotes a temporary file, not the
            # repository document.
            o.disposition = "UNRESOLVED_TARGET"
            continue

        t = _resolve_exact(frags, docset)
        if t is None:
            o.disposition = "UNRESOLVED_TARGET"
            continue
        o.target = t
        o.disposition = {"R": "RESOLVED_READ", "W": "RESOLVED_WRITE",
                         "RW": "READ_AND_WRITE"}[o.mode]
    return ops


def account(ops):
    tally = collections.Counter(o.disposition for o in ops)
    return len(ops), dict(tally), sum(tally.values())


# ── CLAIM-SCOPED CONSTRUCTIVE EXCLUSION ──────────────────────────────
def target_disposition(op, target, tracked_all):
    """(disposition, witness_name, witness_detail). Exactly one."""
    if op.target == target:
        return "REACHES_T", None, "exact resolved target"

    frags = op.frags
    if not frags:
        return "COULD_REACH_T", None, None
    tbase = pathlib.PurePosixPath(target).name
    tdir = str(pathlib.PurePosixPath(target).parent)
    last = frags[-1]
    lbase = pathlib.PurePosixPath(last).name
    # The BASENAME is what the witness compares, so it is the basename
    # that must be fixed -- not the whole fragment. A shell target is a
    # single fragment, so testing the fragment made "$OUT/bar.md" look
    # wholly dynamic when its filename is in fact fixed.
    base_fixed = bool(lbase) and not any(c in lbase for c in "${}*")

    if _fixed(frags):
        joined = _joined(frags)

        # An ABSOLUTE or URI-shaped literal is expressed in a DIFFERENT
        # COORDINATE SYSTEM from a repository-relative target, so its
        # directory may not be compared with the target's -- that is the
        # v1.0 defect (D341 F3 / R6) reappearing one level down.
        # /home/user/repo/data/SOUL.md IS data/SOUL.md when the root is
        # /home/user/repo. The sound constructive test is that an
        # absolute path can only ever denote repository-relative paths
        # that are SUFFIXES of it.
        if joined.startswith("/") or "://" in joined:
            if joined == target or joined.endswith("/" + target):
                return "COULD_REACH_T", None, None
            return ("EXCLUDED_FROM_T", "FIXED_COMPLETE_PATH_DIFFERS",
                    f"target {target!r} is not a path suffix of the fixed "
                    f"literal {joined!r}")

        norm = joined.lstrip("./")

        # WITNESS 1 -- complete repository-relative literal naming a
        # different file.
        if lbase and "." in lbase and lbase != tbase:
            return ("EXCLUDED_FROM_T", "FIXED_COMPLETE_PATH_DIFFERS",
                    f"fixed literal {joined!r} names {lbase!r}, not {tbase!r}")

        # WITNESS 3 -- same basename: a directory proof is the only
        # remaining constructive route, and it is sound here because
        # both paths are repository-relative.
        if "/" in norm and tdir not in (".", ""):
            odir = str(pathlib.PurePosixPath(norm).parent)
            if odir != tdir and not odir.startswith(tdir + "/") \
                    and not tdir.startswith(odir + "/"):
                return ("EXCLUDED_FROM_T", "FIXED_DIRECTORY_DISJOINT",
                        f"fixed directory {odir!r} disjoint from {tdir!r}")

        if lbase and "." not in lbase and norm != target:
            return ("EXCLUDED_FROM_T", "FIXED_COMPLETE_PATH_DIFFERS",
                    f"fixed literal {joined!r} is not {target!r}")
        return "COULD_REACH_T", None, None

    # PARTIALLY DYNAMIC. The path cannot be resolved, but a dynamic
    # PREFIX cannot change the BASENAME: if the final component is a
    # fixed filename that differs from the target's, the operation is
    # constructively excluded however the prefix resolves.
    if base_fixed and "." in lbase and lbase != tbase:
        return ("EXCLUDED_FROM_T", "FIXED_BASENAME_DIFFERS",
                f"fixed final component {lbase!r} != {tbase!r} under any "
                f"resolution of the dynamic prefix")

    # Otherwise: asserting exclusion would be a negative claim about an
    # unknown value, which is what P13/P14 exist to forbid.
    return "COULD_REACH_T", None, None


ANALYSIS_SCOPE = {
    "source_population": "git-tracked .py/.sh/.bash/.yml/.yaml/Makefile "
                         "at the analysed tree, standard excludes applied",
    "analysis_model": "Python write/read calls resolved by AST with "
                      "single-level literal binding; shell redirection, "
                      "tee and sed -i within shell context only",
    "excluded_by_construction": "runtime behaviour, generated code, "
                                "external processes, human edits, and any "
                                "write whose target expression is dynamic",
}


def scoped_claim(ops, target, tracked_all):
    """The strongest ADMISSIBLE statement about writers of `target`.

    Returns (claim, sources, buckets, witnesses, scope).

    NO_WRITER_WITHIN_ANALYZED_SCOPE is emitted only when every write
    operation is either resolved elsewhere or POSITIVELY excluded, i.e.
    COULD_REACH_T == 0. The name states what was actually analysed; it
    does not claim that nothing anywhere writes the document.
    """
    writes = [o for o in ops
              if o.disposition in ("RESOLVED_WRITE", "READ_AND_WRITE")
              or o.mode in ("W", "RW")]
    buckets = {k: [] for k in ALPHABETS["TARGET_DISPOSITIONS"]}
    witnesses = collections.Counter()
    for o in writes:
        d, wname, _detail = target_disposition(o, target, tracked_all)
        buckets[d].append(o)
        if wname:
            witnesses[wname] += 1
    total = sum(len(v) for v in buckets.values())
    assert total == len(writes), "claim population does not reconcile"

    counts = {k: len(v) for k, v in buckets.items()}
    if buckets["REACHES_T"]:
        return ("PROVEN_WRITE_RELATION",
                sorted({o.src for o in buckets["REACHES_T"]}),
                counts, dict(witnesses), ANALYSIS_SCOPE)
    if buckets["COULD_REACH_T"]:
        return ("NO_PROVEN_WRITER", None, counts, dict(witnesses),
                ANALYSIS_SCOPE)
    return ("NO_WRITER_WITHIN_ANALYZED_SCOPE", None, counts,
            dict(witnesses), ANALYSIS_SCOPE)
