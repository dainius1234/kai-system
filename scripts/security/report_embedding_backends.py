#!/usr/bin/env python3
"""Every service that chooses between a semantic backend and a fallback.

The claim under test
--------------------

    MEMU_ALLOW_FAKE_EMBEDDINGS=false is the documented production
    default.

It is, in all three compose files. The Makefile exports `true` globally
and every CI workflow sets `true`, so **no repo-defined execution path
has ever run the production default.** That claim is deliberately scoped
to repo-defined paths: this instrument cannot see historical external
environments, and a wider claim would be one it cannot support.

Why the denominator is not one service
--------------------------------------

`MEMU_ALLOW_FAKE_EMBEDDINGS` is named for memu-core and governs
memu-core. The *decision it governs* — real semantic backend, or a
degraded substitute — is made in more than one place. A flag whose scope
is narrower than the behaviour it names is the defect shape this
programme keeps paying for, so the population here is derived from the
tree: every non-test module that imports `sentence_transformers`.

What "proven" requires
----------------------

Not that the service starts. **A service can start perfectly while its
semantic backend silently degraded**, which is precisely the failure
mode in two of the three rows below. Proof requires the runtime
signature: a real backend produces 384-dimensional vectors, the hash
fallback produces 8. That signal is independent of the configuration
that is supposed to control it, which is the point (I-8).

Exit 0 always: this is a report. `kind=REPORT` in the registry.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected  # noqa: E402

LIBRARY = "sentence_transformers"
FLAG = "MEMU_ALLOW_FAKE_EMBEDDINGS"

#: Real all-MiniLM-L6-v2 output width, and the hash fallback's width.
#: These are the observable signatures the whole report rests on.
REAL_DIM = 384
FAKE_DIM = 8

#: Judgements, declared so a reader can disagree with a line rather than
#: a number. Everything else in the table is derived.
_INTENT: Dict[str, str] = {
    "memu-core": "semantic memory retrieval — embedding quality IS the "
                 "product; a wrong vector returns confident nonsense",
    "agentic": "semantic specialist routing; falls back to keyword match",
    "fusion-engine": "semantic agreement between specialist responses; "
                     "falls back to Jaccard token overlap",
}


#: Runtime verdicts. Deliberately more than REAL/FAKE: several distinct
#: failures all *look* like a working service, and telling them apart is
#: the whole job.
REAL = "REAL"
FAKE = "FAKE"
WRONG_DIMENSION = "WRONG_DIMENSION"
NO_OBSERVATION = "NO_OBSERVATION"


def classify_signature(dimension: Optional[int],
                       log_text: str = "") -> Tuple[str, str]:
    """Classify a runtime observation. Returns (verdict, reason).

    The dimension is the authority and the log is corroboration, in that
    order, because the log is a claim the service makes about itself and
    the vector is what it actually produced. Where they disagree the
    vector wins and the disagreement is reported — a backend that logs
    success and returns the wrong width is a real failure mode, and it is
    the one that would otherwise pass every check ever written for this.

    **No observation is never a pass.** A service that started, a
    configuration that says `false`, a healthy container: none of these
    is an embedding. `NO_OBSERVATION` exists so that absence cannot be
    silently scored as success.
    """
    said_real = "sentence-transformers loaded" in log_text
    said_fake = "hash-based fake embeddings" in log_text

    if dimension is None:
        if said_real or said_fake:
            return NO_OBSERVATION, (
                "the log makes a claim about the backend but no embedding "
                "was produced; a claim is not a measurement")
        return NO_OBSERVATION, "nothing was measured"

    if dimension == REAL_DIM:
        if said_fake:
            return WRONG_DIMENSION, (
                f"produced {REAL_DIM} dimensions while logging the FAKE "
                f"backend — the service disagrees with itself")
        return REAL, f"{REAL_DIM}-dimensional vector from the real model"

    if dimension == FAKE_DIM:
        return FAKE, (
            f"{FAKE_DIM}-dimensional hash vector — the deterministic "
            f"fallback, not a semantic embedding")

    return WRONG_DIMENSION, (
        f"{dimension} dimensions is neither the real model's {REAL_DIM} "
        f"nor the fallback's {FAKE_DIM}; the backend returned something "
        f"no one designed")


@dataclass
class Row:
    service: str
    path: str
    intent: str = ""
    declared_in: str = ""
    model_source: str = ""
    fallback_control: str = ""
    on_library_missing: str = ""
    on_model_missing: str = ""
    silent: Optional[bool] = None
    ci_executes_production_default: Optional[bool] = None
    classification: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


def importers(root: Path) -> List[Tuple[str, str]]:
    """(service, relative path) for every non-test module importing the
    library. Derived; nothing is named here that the tree does not."""
    out: List[Tuple[str, str]] = []
    skip = {"_archive", ".venv", "__pycache__", ".git", "scripts", "output",
            "tests"}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(p in skip for p in parts) or path.name.startswith("test_"):
            continue
        if path.name == "conftest.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if LIBRARY not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        real = any(
            (isinstance(n, ast.Import)
             and any(a.name.split(".")[0] == LIBRARY for a in n.names))
            or (isinstance(n, ast.ImportFrom)
                and (n.module or "").split(".")[0] == LIBRARY)
            for n in ast.walk(tree))
        if real:
            out.append((parts[0], str(path.relative_to(root))))
    return out


def _declares_library(service: str) -> str:
    req = REPO / service / "requirements.txt"
    if not req.exists():
        return "no requirements.txt"
    for line in req.read_text(encoding="utf-8").splitlines():
        if "sentence" in line.lower():
            return line.strip()
    return "ABSENT"


def _model_source(service: str) -> str:
    dockerfile = REPO / service / "Dockerfile"
    if not dockerfile.exists():
        return "no Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    if "BAKED_REVISION" in text or "snapshot_download" in text:
        cache = re.search(r"HF_HOME=(\S+)", text)
        return f"baked at build time -> {cache.group(1) if cache else '?'}"
    return "none — nothing baked, and no runtime egress on internal nets"


def _flag_in_compose(service: str) -> Tuple[str, bool]:
    """(default expression, whether CI forces a different value)."""
    import yaml
    default = "not passed to this service"
    for path in compose_files(REPO):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        env = ((doc.get("services") or {}).get(service) or {}).get(
            "environment") or {}
        if isinstance(env, list):
            env = {x.split("=")[0]: x.split("=", 1)[-1] for x in env}
        if FLAG in env:
            default = str(env[FLAG])
            break
    forced = False
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        if re.search(rf'{FLAG}\s*:\s*"?true"?', text):
            forced = True
    return default, forced


def _behaviour(rel: str) -> Tuple[str, str, bool]:
    """(on library missing, on model missing, silent) — read from source."""
    text = (REPO / rel).read_text(encoding="utf-8", errors="replace")
    raises = "raise RuntimeError" in text and "Refusing to silently" in text
    if raises:
        return ("raises RuntimeError naming the missing dependency",
                "raises RuntimeError naming the missing MODEL, distinctly",
                False)
    lib = "unknown"
    if re.search(r"except ImportError:\s*\n\s*(return|_HAS_ST\s*=\s*False)",
                 text):
        lib = "swallowed — degrades with no error and no flag"
    model = "unknown"
    if re.search(r"except Exception:\s*\n\s*(return|_SMODEL\s*=\s*None)",
                 text):
        model = "swallowed — degrades with no error and no flag"
    return lib, model, True


def audit(root: Path = None) -> Tuple[List[Row], int, Dict[str, int]]:
    root = root or REPO
    found = importers(root)
    if not found:
        return ([], 0, {})

    rows: List[Row] = []
    for service, rel in found:
        row = Row(service=service, path=rel)
        row.intent = _INTENT.get(service, "NOT DECLARED — must not be read "
                                          "as unimportant")
        row.declared_in = _declares_library(service)
        row.model_source = _model_source(service)
        default, forced_true = _flag_in_compose(service)
        row.fallback_control = default
        row.on_library_missing, row.on_model_missing, row.silent = _behaviour(rel)

        governed = default != "not passed to this service"
        row.ci_executes_production_default = bool(governed and not forced_true)

        if row.declared_in == "ABSENT":
            # The operator's rule: an undeclared dependency proves the
            # DECLARATION is missing, not that the built image lacks it.
            # Only an import inside the image can upgrade this.
            row.classification = "DECLARATION DEFECT"
            row.notes.append(
                "requirements.txt does not declare the library; the "
                "Dockerfile installs only that file from a bare "
                "python:3.11-slim, and full transitive resolution of the "
                "declared set contains no sentence-transformers or torch. "
                "Runtime image state NOT YET PROVEN — needs an import "
                "inside the built image.")
            if row.silent:
                row.notes.append(
                    "degradation is SILENT: the service starts, serves, and "
                    "reports healthy while the semantic path is off.")
        elif not governed:
            row.classification = "UNKNOWN"
        elif row.ci_executes_production_default:
            row.classification = "UNKNOWN"
        else:
            row.classification = "UNKNOWN"
            row.notes.append(
                f"library declared and model {row.model_source}; the "
                f"production default is {default}, and every CI workflow "
                f"forces {FLAG}=true, so no repo-defined path has executed "
                f"it. Needs the real-backend signature (dim {REAL_DIM}) "
                f"from a built image.")
        rows.append(row)

    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    counts["_silent"] = sum(1 for r in rows if r.silent)
    return rows, len(rows), counts


def main() -> int:
    rows, n, counts = audit()
    print(inspected(n, "service(s) choosing between a semantic backend and "
                       "a fallback",
                    f"derived from every non-test module importing "
                    f"{LIBRARY}"))
    if not rows:
        print("\n  no importers found — this scan is broken, not the system: "
              "the library is known to be in use.")
        return 0

    for row in rows:
        print(f"\n  ── {row.service}  ({row.path}) ──")
        print(f"     intent                 {row.intent}")
        print(f"     declared in            {row.declared_in}")
        print(f"     model source           {row.model_source}")
        print(f"     fallback control       {row.fallback_control}")
        print(f"     real signature         dim {REAL_DIM}, "
              f"'sentence-transformers loaded'")
        print(f"     degraded signature     dim {FAKE_DIM} (hash) / "
              f"keyword / Jaccard")
        print(f"     library missing        {row.on_library_missing}")
        print(f"     model missing          {row.on_model_missing}")
        print(f"     degradation            "
              f"{'SILENT' if row.silent else 'explicit — refuses to start'}")
        print(f"     CI runs prod default   "
              f"{row.ci_executes_production_default}")
        print(f"     classification         {row.classification}")
        for note in row.notes:
            print(f"     note                   {note}")

    print()
    print(f"  {counts.get('_silent', 0)} of {n} degrade SILENTLY — the "
          f"service starts, serves and reports healthy")
    print("  while the semantic path is off. 'Service started' is therefore")
    print("  NOT evidence that the semantic backend started.")
    print()
    for label in sorted(k for k in counts if not k.startswith("_")):
        print(f"  {label} ....... {counts[label]}")
    print()
    print("  Nothing here is PROVEN. Proof requires the runtime signature")
    print("  from a built image, not the presence of configuration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
