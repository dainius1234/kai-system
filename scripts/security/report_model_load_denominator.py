#!/usr/bin/env python3
"""Who can execute model-loading code, where it is deployed, and what we
actually know about it at runtime — kept as THREE populations, not one.

    A   source reachability     the import graph of a container's real
                                entrypoint, walked INSIDE the image, can
                                reach a CALL that materialises model
                                weights
    B   deployment applicability   of A, those deployed under an
                                egress-restricted contract — every
                                attached network declared `internal: true`
    C   runtime-qualified       of B, those for which a CITED runtime
                                observation about the model/startup
                                contract exists

A is not "affected in production". B is not "broken". C is not "the rest
are fine". Merging any adjacent pair produces a confident wrong number,
and this programme has paid for that three times in eight days.

THE HAZARD THIS INSTRUMENT IS BUILT AGAINST
===========================================

Twice in one day a detector matched a CONSTRUCT and reported it as a
BEHAVIOUR: a fallback classifier called `{'ok': False}` a silent fallback
(22 false positives in 53), and a semantic tracer called every `store.*`
call an embedding (14 reported, 10 real). Both inflated a denominator,
and an inflated denominator selects the wrong architecture.

So, explicitly, for this sweep:

  * importing `sentence_transformers` is NOT evidence of model loading.
    Only a CALL that materialises weights is. `agentic/router.py` imports
    it at module scope and calls it inside a memoised getter; those are
    different facts with different consequences and are recorded
    separately.
  * a package in `requirements.txt` is NOT evidence that repo code loads
    a model. It is evidence that a THIRD-PARTY loader could — which this
    tracer cannot see, and says so rather than guessing.
  * a service name containing "memu" or "embed" is not evidence of
    anything.

THE FIRST VERSION OF THIS TRACER FAILED ITS OWN CALIBRATION
===========================================================

Its module-scope test was::

    for node in tree.body:
        for sub in ast.walk(node):
            module_level_lines.add(sub.lineno)

`ast.walk` descends into the bodies of module-level `def`s, so every line
in the file is a "module-level line" and every hit was reported as
`IMPORT`. It printed four services all loading at import. Two of them
load lazily inside a function. The number was wrong in the direction that
would have forced the heaviest architecture, and nothing about the output
looked wrong. That is the entire argument for calibrating before
believing a denominator, and it is why
`scripts/test_model_load_denominator.py` asserts LAZY as hard as it
asserts IMPORT.

WHY THE TRACE RUNS INSIDE THE IMAGE, NOT OVER THE REPO
======================================================

A repo-wide trace answers "could this source reach a loader", which is
not the question. The question is what a *runnable container path* can
execute, and a container holds only what its Dockerfile `COPY`s and only
the packages its `pip install` put there. So the COPY layout is parsed
into a container-path -> repo-path map and imports are resolved through
it. Two exclusions fall out of that for free, and both are real:

  * a module the Dockerfile never copies cannot execute, however
    reachable it looks in the repo;
  * a guarded `import` of a package the image never installs raises
    `ImportError` and takes the fallback branch — which is precisely the
    BACKEND_UNAVAILABLE that #47 measured in `agentic` at runtime. The
    static prediction and the runtime observation come from different
    places and agree (I-8).

RUNTIME EVIDENCE IS CITED, NEVER DERIVED
========================================

There is still no machine-readable evidence record (task #51). Population
C is therefore built from an explicit table in which every row names the
decision entry that established it, and `main()` refuses to print C at
all if a citation does not resolve in `kai-pm/DECISIONS.md`. A service
absent from that table is UNKNOWN — which means "not established here",
not "never ran".
"""
from __future__ import annotations

import ast
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import compose_files, inspected, require  # noqa: E402

try:
    import yaml
except ImportError:                                          # pragma: no cover
    print("ERROR: PyYAML required.", file=sys.stderr)
    sys.exit(2)

COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)

# ── vocabularies, declared once ──────────────────────────────────────
# Written out rather than implied, because every one of these was, at
# some point in this programme, silently inferred from its neighbour.

REACH_TRACED = "TRACED"                    # a call, found, in a copied module
REACH_THIRD_PARTY = "THIRD-PARTY-CANDIDATE"
REACH_COMMAND = "COMPOSE-COMMAND"          # the repo-defined command IS a fetch
REACH_NONE = "NOT REACHABLE"
REACH_UNKNOWN = "UNKNOWN"

TIMING_IMPORT = "IMPORT"
TIMING_STARTUP = "STARTUP"
TIMING_LAZY = "LAZY / REQUEST-TIME"
TIMING_UNKNOWN = "UNKNOWN"
#: IMPORT happens before STARTUP happens before a request. `max` over this
#: order is the weakest-link combiner: a lazily-imported module whose body
#: loads at import still only loads when that lazy import runs.
_TIMING_ORDER = {TIMING_IMPORT: 0, TIMING_STARTUP: 1,
                 TIMING_LAZY: 2, TIMING_UNKNOWN: 3}

ROLE_REQUIRED = "REQUIRED"
ROLE_PARTIAL = "PARTIAL"
ROLE_NOT_REQUIRED = "NOT REQUIRED"
ROLE_UNKNOWN = "UNKNOWN"
#: Applied, not merely declared. `NOT REQUIRED` has no member today, and
#: a value that is only ever written down is a rule the code claims and
#: does not enforce (I-5) — so the whole vocabulary is validated against
#: the evidence table instead of sitting beside it. If a future entry
#: invents "NOT-REQUIRED" or "optional", `main()` refuses.
ROLE_VOCABULARY = frozenset(
    {ROLE_REQUIRED, ROLE_PARTIAL, ROLE_NOT_REQUIRED, ROLE_UNKNOWN})

EGRESS_RESTRICTED = "EGRESS-RESTRICTED"
EGRESS_AVAILABLE = "EGRESS-AVAILABLE"
EGRESS_MIXED = "MIXED"                     # differs between compose files
EGRESS_UNKNOWN = "UNKNOWN"

YES, NO, NA, UNKNOWN = "YES", "NO", "N/A", "UNKNOWN"

# ── what counts as materialising model weights ───────────────────────
# Keyed by ORIGIN MODULE, so `from x import y as z` resolves to where the
# name came from rather than to a bare name that happens to collide. A
# local helper called `pipeline` is not `transformers.pipeline`.
LOADER_CALLS: Dict[str, Set[str]] = {
    "sentence_transformers": {"SentenceTransformer", "CrossEncoder"},
    "transformers": {"pipeline"},
    "huggingface_hub": {"snapshot_download", "hf_hub_download"},
    "spacy": {"load"},
    "whisper": {"load_model"},
}
#: Method call that materialises weights whatever the receiver is. This
#: one IS name-based, and deliberately: `.from_pretrained(` has exactly
#: one meaning in the HuggingFace ecosystem and no other library in this
#: tree defines it. Asserted as a known-positive in the calibration.
LOADER_METHODS = {"from_pretrained"}

#: Libraries that can load a model from inside THIRD-PARTY code, where no
#: repo-source trace can reach the call. Presence in an image's installed
#: set makes a service an A-CANDIDATE — never a member by tracing, and
#: never silently dropped either, which is the failure the `memu-graph`
#: case demonstrates.
THIRD_PARTY_MODEL_LIBS = {
    "sentence-transformers", "transformers", "spacy", "openai-whisper",
    "timm", "diffusers", "open_clip_torch", "ultralytics",
}
#: Observed in the tree and deliberately NOT treated as model loaders,
#: with the reason, so the exclusion is auditable rather than an absence.
NOT_A_LOADER = {
    "torch": "a tensor library; `import torch` is a capability probe in "
             "common/gpu_utils.py and common/runtime.py, and torch.load "
             "deserialises a local file rather than resolving a registry",
    "onnxruntime": "imported inside a function in agentic/cortex.py as an "
                   "availability probe (`# noqa: F401`); no session is "
                   "constructed from a registry",
    "numpy": "arithmetic",
}

#: Environment variables that pin a cache to a fixed, image-owned path.
CACHE_ENV = ("HF_HOME", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME",
             "HUGGINGFACE_HUB_CACHE", "TORCH_HOME")
#: Environment variables that forbid a runtime registry round-trip.
OFFLINE_ENV = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
#: A build step wearing one of these has been told not to fail.
FAIL_OPEN = ("|| true", "|| :", "|| echo", "; exit 0", "|| exit 0")

#: An explicit FETCH VERB in a repo-defined `command:`/`entrypoint:`. This
#: is how a third-party image becomes measurable: the repo wrote the
#: command, so the behaviour is derivable even though the image is not.
#: Verbs only — `--model X` selects, it does not fetch, so
#: `parakeet-server` stays NOT MEASURED rather than being counted on the
#: strength of a flag name.
FETCH_VERBS = ("ollama pull", "huggingface-cli download", "hf download",
               "snapshot_download")
_HOSTPORT = re.compile(r"\b([a-z][a-z0-9-]{1,40}):(\d{2,5})\b")


# ══ runtime evidence: cited, never derived ═══════════════════════════
#
# Each entry names the decision entry that established it. `main()`
# verifies every citation resolves in kai-pm/DECISIONS.md before printing
# population C, and refuses to print it if one does not. The point is
# I-8: the source of the expected answer is not the thing under test.
#
# A service ABSENT here is UNKNOWN. UNKNOWN is an evidence status, not a
# history.
RUNTIME_EVIDENCE: Dict[str, Dict[str, str]] = {
    "memu-core": {
        "observation": "REAL embedding backend confirmed in the deployed "
                       "runtime; service reached healthy",
        "cite": "D175",
    },
    "memu-core-introspect": {
        "observation": "PREREQUISITE STARTUP FAILURE — the listener "
                       "remained unavailable while startup performed the "
                       "external Hugging Face resolution/retry sequence, "
                       "and the service failed its readiness contract",
        "cite": "D183",
    },
    "agentic": {
        "observation": "BACKEND_UNAVAILABLE at runtime — the semantic "
                       "backend did not start; the service did",
        "cite": "D175",
    },
}

#: Semantic necessity, measured rather than inferred. Same citation rule.
#: D185's wording is deliberate and is reproduced rather than paraphrased:
#: what the ROLE requires and what the current STRUCTURE enforces are two
#: different claims, and collapsing them is how "REQUIRED" would become a
#: permanent property of an accident.
ROLE_EVIDENCE: Dict[str, Dict[str, str]] = {
    "memu-core": {
        "role": ROLE_REQUIRED,
        "note": "vector write and search are the service's role",
        "cite": "D175",
    },
    "memu-core-introspect": {
        "role": ROLE_PARTIAL,
        "note": "REQUEST-TIME REQUIRED BY ROLE for search_by_category; "
                "PRE-READINESS ENFORCED BY CURRENT STRUCTURE, not by role "
                "— /health has not been shown to require embeddings",
        "cite": "D185",
    },
}


# ══ Dockerfile: the image as a filesystem, not as text ═══════════════

class Image:
    """What a Dockerfile actually puts in a container.

    Only three things are needed and all three are load-bearing:
    the COPY layout (so an import can be resolved the way the container
    would), the installed package set (so a guarded import can be known
    to fail), and the ENV/RUN facts the offline contract is made of.
    """

    def __init__(self, dockerfile: Path, root: Path = REPO):
        self.path = dockerfile
        self.root = root
        self.rel = str(dockerfile.relative_to(root))
        self.env: Dict[str, str] = {}
        self.copies: List[Tuple[str, str]] = []      # (container, repo)
        self.run_steps: List[str] = []
        self.requirements: List[str] = []
        self.entry: Optional[str] = None             # container path or module
        self.workdir = "/"
        self._parse(dockerfile.read_text(encoding="utf-8"))

    # -- parsing ------------------------------------------------------
    def _parse(self, text: str) -> None:
        for raw in self._logical_lines(text):
            if not raw or raw.startswith("#"):
                continue
            verb, _, rest = raw.partition(" ")
            verb = verb.upper()
            rest = rest.strip()
            if verb == "WORKDIR":
                self.workdir = self._abs(rest)
            elif verb == "ENV":
                self._env(rest)
            elif verb == "COPY":
                self._copy(rest)
            elif verb == "RUN":
                self.run_steps.append(rest)
                self._pip(rest)
            elif verb in ("CMD", "ENTRYPOINT"):
                self.entry = self._entry(rest) or self.entry

    @staticmethod
    def _logical_lines(text: str) -> List[str]:
        """Join `\\`-continued lines — a RUN step is one instruction."""
        out, buf = [], ""
        for line in text.splitlines():
            stripped = line.rstrip()
            if stripped.endswith("\\"):
                buf += stripped[:-1] + " "
                continue
            out.append((buf + stripped).strip())
            buf = ""
        if buf:
            out.append(buf.strip())
        return out

    def _abs(self, p: str) -> str:
        p = p.strip().strip('"')
        if p.startswith("/"):
            return p.rstrip("/") or "/"
        return (self.workdir.rstrip("/") + "/" + p.lstrip("./")).rstrip("/") or "/"

    def _env(self, rest: str) -> None:
        tokens = shlex.split(rest)
        if not tokens:
            return
        if "=" in tokens[0]:                     # ENV A=1 B=2
            for token in tokens:
                if "=" in token:
                    k, _, v = token.partition("=")
                    self.env[k] = v
        else:                                    # ENV KEY value
            parts = rest.split(None, 1)
            if len(parts) == 2:
                self.env[parts[0]] = parts[1].strip().strip('"')

    def _copy(self, rest: str) -> None:
        """Map container paths back to repo paths, with Docker's own rule.

        `COPY <dir> <dest>` copies the CONTENTS of the directory, not the
        directory itself — with or without a trailing slash. A first
        version appended the source's basename unconditionally and mapped
        `COPY common/ ./common/` to `/app/common/common`, so every
        `from common...` import in every image resolved to nothing and
        the trace silently stopped one hop in. It found the same five
        services, which is exactly why it went unnoticed: a wrong
        traversal that happens to reach the same answer still cannot be
        trusted for the next question.
        """
        parts = [p for p in shlex.split(rest) if not p.startswith("--")]
        if len(parts) < 2:
            return
        *srcs, dest = parts
        dest_abs = self._abs(dest)
        dest_is_dir = dest.endswith("/") or len(srcs) > 1 or dest in (".", "./")
        for src in srcs:
            src_clean = src.rstrip("/")
            if (self.root / src_clean).is_dir():
                container = dest_abs
            elif dest_is_dir:
                container = dest_abs.rstrip("/") + "/" + Path(src_clean).name
            else:
                container = dest_abs
            self.copies.append((container.rstrip("/") or "/", src_clean))

    def _pip(self, rest: str) -> None:
        for m in re.finditer(r"pip\s+install[^&|;]*?-r\s+(\S+)", rest):
            target = self._abs(m.group(1))
            repo_path = self.resolve(target)
            if repo_path and repo_path.exists():
                self.requirements.append(str(repo_path.relative_to(self.root)))

    @staticmethod
    def _entry(rest: str) -> Optional[str]:
        rest = rest.strip()
        try:
            argv = json.loads(rest) if rest.startswith("[") else shlex.split(rest)
        except (ValueError, json.JSONDecodeError):
            return None
        if not argv:
            return None
        if argv[0].endswith("python") or argv[0].startswith("python"):
            for a in argv[1:]:
                if a.endswith(".py"):
                    return a
        if argv[0] == "uvicorn" or (len(argv) > 2 and argv[1] == "uvicorn"):
            for a in argv[1:]:
                if ":" in a and not a.startswith("-"):
                    return a.split(":")[0].replace(".", "/") + ".py"
        return None

    # -- the image as a filesystem -----------------------------------
    def resolve(self, container_path: str) -> Optional[Path]:
        """Container path -> repo path, or None if nothing COPYs it.

        Longest prefix wins, the way an overlay does. `None` is a real
        answer and the reason `agentic-introspect` does not inherit
        `router.py`'s loader: the file is not in the image.
        """
        best: Optional[Path] = None
        best_len = -1
        for container, src in self.copies:
            if container_path == container:
                cand = self.root / src
            elif container_path.startswith(container.rstrip("/") + "/"):
                tail = container_path[len(container.rstrip("/")) + 1:]
                cand = self.root / src / tail
            else:
                continue
            if len(container) > best_len:
                best, best_len = cand, len(container)
        return best

    def installed(self) -> Set[str]:
        """Distribution names this image pip-installs, normalised."""
        out: Set[str] = set()
        for rel in self.requirements:
            try:
                text = (self.root / rel).read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.split("#")[0].strip()
                if not line or line.startswith("-"):
                    continue
                name = re.split(r"[<>=!\[; ]", line, 1)[0].strip().lower()
                if name:
                    out.add(name.replace("_", "-"))
        return out

    def entry_module(self) -> Tuple[Optional[Path], str]:
        """(repo path of the entrypoint module, why-not if None)."""
        if not self.entry:
            return None, "no CMD/ENTRYPOINT naming a python module"
        target = self._abs(self.entry)
        path = self.resolve(target)
        if path is None:
            return None, f"{target} is not COPYed into the image"
        if not path.exists():
            return None, f"{target} maps to {path} which does not exist"
        return path, ""


# ══ the trace ════════════════════════════════════════════════════════

def _module_scope_calls(tree: ast.AST) -> Set[str]:
    """Names invoked AT MODULE SCOPE — not names invoked anywhere.

    The distinction is the whole timing column. A first version collected
    these with `ast.walk`, which sees `_get_smodel()` called from inside a
    request handler and concludes `_get_smodel` runs at import. Both
    lazily-loading services were then reported as IMPORT, which is the
    error that argues for the most expensive architecture.
    """
    out: Set[str] = set()

    def visit(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            deeper = in_func or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if not in_func and isinstance(child, ast.Call) \
                    and isinstance(child.func, ast.Name):
                out.add(child.func.id)
            visit(child, deeper)

    visit(tree, False)
    return out


def _scope_timing(func_stack: List[ast.AST], module_calls: Set[str]) -> str:
    """Timing of a call sitting inside `func_stack` (outermost first).

    Not `lineno in module_lines` — that was the defect. Nesting is the
    only thing that decides whether code runs at import.
    """
    if not func_stack:
        return TIMING_IMPORT
    outer = func_stack[0]
    for dec in getattr(outer, "decorator_list", []):
        text = ast.dump(dec)
        if "on_event" in text or "startup" in text or "lifespan" in text:
            return TIMING_STARTUP
    name = getattr(outer, "name", None)
    if name and name in module_calls:
        # defined in a function but that function is invoked at module
        # scope, so it executes at import after all.
        return TIMING_IMPORT
    return TIMING_LAZY


def _loader_hits(tree: ast.AST) -> List[Tuple[int, str, str]]:
    """[(lineno, timing, what)] for every call materialising weights."""
    # name -> origin module, from this file's own imports
    origin: Dict[str, str] = {}
    modules: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                origin[a.asname or a.name] = f"{node.module.split('.')[0]}:{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                modules[a.asname or a.name] = a.name.split(".")[0]

    hits: List[Tuple[int, str, str]] = []
    module_calls = _module_scope_calls(tree)

    def visit(node: ast.AST, stack: List[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            deeper = stack
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                deeper = stack + [child]
            if isinstance(child, ast.Call):
                what = _what_loads(child, origin, modules)
                if what:
                    hits.append((child.lineno,
                                 _scope_timing(stack, module_calls), what))
            visit(child, deeper)

    visit(tree, [])
    return hits


def _what_loads(call: ast.Call, origin: Dict[str, str],
                modules: Dict[str, str]) -> str:
    """Name the loader this call invokes, or "" — origin-resolved."""
    f = call.func
    if isinstance(f, ast.Name):
        src = origin.get(f.id)
        if src:
            mod, _, attr = src.partition(":")
            if attr in LOADER_CALLS.get(mod, ()):
                return f"{mod}.{attr}"
        return ""
    if isinstance(f, ast.Attribute):
        if f.attr in LOADER_METHODS:
            return f"<receiver>.{f.attr}"
        if isinstance(f.value, ast.Name):
            mod = modules.get(f.value.id)
            if mod and f.attr in LOADER_CALLS.get(mod, ()):
                return f"{mod}.{f.attr}"
    return ""


def _imports(tree: ast.AST) -> List[Tuple[str, str]]:
    """[(module, timing-of-the-import-statement)] for this module."""
    out: List[Tuple[str, str]] = []
    module_calls = _module_scope_calls(tree)

    def visit(node: ast.AST, stack: List[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            deeper = stack
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                deeper = stack + [child]
            if isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                out.append((child.module, _scope_timing(stack, module_calls)))
            elif isinstance(child, ast.Import):
                for a in child.names:
                    out.append((a.name, _scope_timing(stack, module_calls)))
            visit(child, deeper)

    visit(tree, [])
    return out


def _guarded_imports(tree: ast.AST) -> Set[str]:
    """Top-level packages imported inside a `try:` that catches ImportError.

    An unguarded import of an absent package crashes the container; a
    guarded one takes the fallback branch. Different defect, different
    remedy, so they are not merged.
    """
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        catches = any(
            (isinstance(h.type, ast.Name) and h.type.id in
             ("ImportError", "ModuleNotFoundError", "Exception"))
            or h.type is None
            for h in node.handlers
        )
        if not catches:
            continue
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.ImportFrom) and sub.module:
                    out.add(sub.module.split(".")[0])
                elif isinstance(sub, ast.Import):
                    out.update(a.name.split(".")[0] for a in sub.names)
    return out


def trace(image: Image, start: Path) -> Tuple[Optional[Dict], List[str]]:
    """Walk the import graph AS THE CONTAINER WOULD and find the load.

    Returns `(earliest-timing hit or None, branches not followed)`.
    The second half is the honest part: a walk that stopped early cannot
    support a claim of "nothing found". Import resolution goes
    through the image's COPY map, so a module the Dockerfile never copies
    is unreachable however present it is in the repo.
    """
    best: Optional[Dict] = None
    seen: Set[Path] = set()
    blind: List[str] = []

    def walk(path: Path, timing_so_far: str, depth: int) -> None:
        nonlocal best
        if path in seen:
            return
        seen.add(path)
        # A truncated walk is not a clean walk. Each of these ends a
        # branch of the graph, and a branch that was never followed
        # cannot support "no loader is reachable" — so they are collected
        # and printed rather than skipped. I-1: absence must not read as
        # correctness, and a recursion guard is exactly where that is
        # easiest to forget.
        if depth > 6:
            blind.append(f"{path.name}: depth limit reached, not followed")
            return
        if not path.exists():
            blind.append(f"{path}: resolved by the COPY map but absent on disk")
            return
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            blind.append(f"{path.name}: unreadable/unparseable ({exc.__class__.__name__})")
            return
        guarded = _guarded_imports(tree)
        for lineno, timing, what in _loader_hits(tree):
            combined = max(timing, timing_so_far, key=lambda t: _TIMING_ORDER[t])
            lib = what.split(".")[0]
            dist = lib.replace("_", "-")
            record = {
                "path": str(path.relative_to(image.root)),
                "line": lineno,
                "timing": combined,
                "loader": what,
                "library": lib,
                "installed": (dist in image.installed()
                              if lib in {m.replace("_", "-") for m in LOADER_CALLS}
                              or dist in THIRD_PARTY_MODEL_LIBS else None),
                "guarded": lib in guarded,
            }
            if best is None or (_TIMING_ORDER[combined]
                                < _TIMING_ORDER[best["timing"]]):
                best = record
        for module, timing in _imports(tree):
            top = module.split(".")[0]
            if top in LOADER_CALLS or top in NOT_A_LOADER:
                continue                     # an import is not a call
            nxt = max(timing, timing_so_far, key=lambda t: _TIMING_ORDER[t])
            for candidate in _candidates(image, path, module):
                walk(candidate, nxt, depth + 1)

    walk(start, TIMING_IMPORT, 0)
    return best, blind


def _candidates(image: Image, importer: Path, module: str) -> List[Path]:
    """Where the CONTAINER would look for `module`, in order."""
    rel = module.replace(".", "/")
    out: List[Path] = []
    for container in (f"{image.workdir}/{rel}.py", f"{image.workdir}/{rel}/__init__.py"):
        got = image.resolve(container)
        if got is not None and got.exists():
            out.append(got)
    # a sibling of the importer is only reachable if the container has it,
    # which the workdir lookup above already decided. Nothing else is
    # added here on purpose: guessing a repo path would re-introduce the
    # repo-wide trace this function exists to avoid.
    return out


# ══ deployment ═══════════════════════════════════════════════════════

def deployments(root: Path = REPO) -> Dict[str, Dict]:
    """service -> per-file networks/volumes/profiles/build, plus egress."""
    out: Dict[str, Dict] = {}
    for path in [root / f for f in COMPOSE_FILES]:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        declared = doc.get("networks") or {}
        internal = {n for n, s in declared.items() if (s or {}).get("internal")}
        for name, spec in (doc.get("services") or {}).items():
            spec = spec or {}
            nets = spec.get("networks") or []
            if isinstance(nets, dict):
                nets = list(nets)
            entry = out.setdefault(name, {
                "files": {}, "dockerfile": None, "image": spec.get("image"),
                "profiles": set(), "command": "",
            })
            for key in ("entrypoint", "command"):
                val = spec.get(key)
                if val:
                    text = " ".join(val) if isinstance(val, list) else str(val)
                    entry["command"] = (entry["command"] + " " + text).strip()
            entry["files"][path.name] = {
                "networks": list(nets),
                # No `networks:` key means the implicit `default` network,
                # which no file here declares internal. Absence is egress,
                # not restriction — the inverted reading would be a scope
                # LARGER than reality, reporting failure over things that
                # are right.
                "restricted": bool(nets) and set(nets) <= internal,
                "volumes": list(spec.get("volumes") or []),
            }
            entry["profiles"].update(spec.get("profiles") or [])
            build = spec.get("build")
            if build and not entry["dockerfile"]:
                if isinstance(build, dict):
                    entry["dockerfile"] = (build.get("dockerfile")
                                           or f"{build.get('context', '.')}/Dockerfile")
                else:
                    entry["dockerfile"] = f"{build}/Dockerfile"
    for entry in out.values():
        flags = {f["restricted"] for f in entry["files"].values()}
        entry["egress"] = (EGRESS_RESTRICTED if flags == {True}
                           else EGRESS_AVAILABLE if flags == {False}
                           else EGRESS_MIXED if flags else EGRESS_UNKNOWN)
    return out


# ══ image contract ═══════════════════════════════════════════════════

def cache_path(image: Image) -> Optional[str]:
    for var in CACHE_ENV:
        if var in image.env:
            return f"{var}={image.env[var]}"
    return None


def bake_step(image: Image) -> Optional[str]:
    """A RUN step that CALLS a loader — behaviour, not the word 'model'."""
    for step in image.run_steps:
        if any(m in step for m in LOADER_METHODS):
            return step
        for mod, names in LOADER_CALLS.items():
            if mod in step and any(f"{n}(" in step for n in names):
                return step
        if "huggingface-cli download" in step or "snapshot_download" in step:
            return step
    return None


def offline_enforced(image: Image, svc_env: Dict[str, str]) -> str:
    for var in OFFLINE_ENV:
        if image.env.get(var) in ("1", "true", "True"):
            return "IMAGE"
        if svc_env.get(var) in ("1", "true", "True"):
            return "COMPOSE"
    return "NOT ENFORCED"


def mount_shadows(cache: Optional[str], volumes: Iterable[str]) -> bool:
    """Does any mount sit at or above the cache path?

    Docker seeds a named volume from the image only when the volume is
    NEW. A pre-existing volume mounted at or above the cache shadows the
    baked asset, and CI's fresh volume never sees it.
    """
    if not cache:
        return False
    target = cache.split("=", 1)[1].rstrip("/")
    for vol in volumes:
        parts = str(vol).split(":")
        if len(parts) < 2:
            continue
        mount = parts[1].rstrip("/")
        if target == mount or target.startswith(mount + "/"):
            return True
    return False


# ══ the survey ═══════════════════════════════════════════════════════

def survey(root: Path = REPO) -> List[Dict]:
    dep = deployments(root)
    rows: List[Dict] = []
    for name in sorted(dep):
        entry = dep[name]
        row = {
            "service": name,
            "dockerfile": entry["dockerfile"] or (entry["image"] or "-"),
            "egress": entry["egress"],
            "reach": REACH_NONE,
            "path": "",
            "timing": TIMING_UNKNOWN,
            "library": "",
            "installed": UNKNOWN,
            "guarded": UNKNOWN,
            "why": "",
            "baked": UNKNOWN,
            "cache": UNKNOWN,
            "offline": UNKNOWN,
            "fail_closed": UNKNOWN,
            "shadow": UNKNOWN,
            "listener_gated": UNKNOWN,
            "role": ROLE_UNKNOWN,
            "role_note": "",
            "evidence": "UNKNOWN (no cited runtime observation)",
            "profiles": sorted(entry["profiles"]),
            "image_only": False,
            "trace_blind": [],
        }
        if not entry["dockerfile"]:
            verb = next((v for v in FETCH_VERBS if v in entry["command"]), None)
            if verb:
                # The image is opaque; the COMMAND is not. The repo wrote
                # it, so this is a derived behaviour rather than a guess
                # about someone else's entrypoint.
                row.update(reach=REACH_COMMAND,
                           path=f"compose command: `{verb}`",
                           timing=TIMING_STARTUP,
                           listener_gated=NA,
                           why=_delegation(entry["command"], dep, name))
                rows.append(row)
                continue
            # NOT MEASURED, not "clean". This tracer reads repo source and
            # repo Dockerfiles; it cannot see inside a published image.
            # Some of these (ollama, parakeet-server) plainly serve models.
            # Folding them into "no loader found" would be a scope smaller
            # than the name, reported as green.
            row["reach"] = REACH_UNKNOWN
            row["why"] = ("runs a third-party image; this instrument cannot "
                          "see inside one — NOT MEASURED, not clean")
            row["image_only"] = True
            rows.append(row)
            continue

        df = root / entry["dockerfile"]
        if not df.exists():
            row["reach"] = REACH_UNKNOWN
            row["why"] = f"{entry['dockerfile']} not found"
            rows.append(row)
            continue

        image = Image(df, root)
        start, why = image.entry_module()
        if start is None:
            row["reach"] = REACH_UNKNOWN
            row["why"] = why
            rows.append(row)
            continue

        hit, blind = trace(image, start)
        row["trace_blind"] = blind
        installed = image.installed()
        third_party = sorted(installed & THIRD_PARTY_MODEL_LIBS)

        if hit:
            row.update(reach=REACH_TRACED,
                       path=f"{hit['path']}:{hit['line']}",
                       timing=hit["timing"],
                       library=hit["library"],
                       installed=(YES if hit["installed"] else NO)
                       if hit["installed"] is not None else UNKNOWN,
                       guarded=YES if hit["guarded"] else NO)
            if hit["installed"] is False:
                row["why"] = (f"{hit['library']} is not installed by "
                              f"{', '.join(image.requirements) or 'this image'}"
                              f" — the import raises and the "
                              f"{'fallback branch runs' if hit['guarded'] else 'container dies'}")
        elif third_party:
            row.update(reach=REACH_THIRD_PARTY,
                       library=", ".join(third_party),
                       installed=YES,
                       why="installed, but no repo-source call reaches a "
                           "loader; whether a model loads is inside "
                           "third-party code and is not derivable here")

        if row["reach"] in (REACH_TRACED, REACH_THIRD_PARTY):
            bake = bake_step(image)
            cache = cache_path(image)
            svc_env = _service_env(root, name)
            vols = [v for f in entry["files"].values() for v in f["volumes"]]
            row.update(
                baked=YES if bake else NO,
                cache=cache or NO,
                offline=offline_enforced(image, svc_env),
                fail_closed=(NA if not bake else
                             NO if any(t in bake for t in FAIL_OPEN) else YES),
                shadow=YES if mount_shadows(cache, vols) else NO,
                listener_gated=(YES if row["timing"] in (TIMING_IMPORT, TIMING_STARTUP)
                                else NO if row["timing"] == TIMING_LAZY
                                else UNKNOWN),
            )
        role = ROLE_EVIDENCE.get(name)
        if role:
            row["role"] = role["role"]
            row["role_note"] = f"{role['note']} [{role['cite']}]"
        ev = RUNTIME_EVIDENCE.get(name)
        if ev:
            row["evidence"] = f"{ev['observation']} [{ev['cite']}]"
        rows.append(row)
    return rows


def _delegation(command: str, dep: Dict[str, Dict], me: str) -> str:
    """Does this command make some OTHER service do the fetching?

    `ollama-pull` runs `ollama pull` on an internal-only network, which
    reads like the same defect as memu-core-introspect until you notice
    it sets `OLLAMA_HOST=ollama:11434`: the egress belongs to `ollama`,
    which has it. A rule of "no egress + a fetch verb = broken" would
    have reported a correct design as a defect, which is the inverted
    scope — larger than reality, failing things that are right.
    """
    for host, _port in _HOSTPORT.findall(command):
        if host in dep and host != me:
            return (f"the fetch is delegated to `{host}` "
                    f"({dep[host]['egress']}); this container needs no "
                    f"egress of its own")
    return ("no delegation target found in the command — this container "
            "performs the fetch itself")


def _service_env(root: Path, service: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in [root / f for f in COMPOSE_FILES]:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spec = (doc.get("services") or {}).get(service) or {}
        env = spec.get("environment") or {}
        if isinstance(env, list):
            for item in env:
                k, _, v = str(item).partition("=")
                out[k] = v
        elif isinstance(env, dict):
            out.update({k: str(v) for k, v in env.items()})
    return out


def populations(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    a = [r for r in rows if r["reach"] in (REACH_TRACED, REACH_THIRD_PARTY,
                                          REACH_COMMAND)]
    b = [r for r in a if r["egress"] in (EGRESS_RESTRICTED, EGRESS_MIXED)]
    c = [r for r in b if not r["evidence"].startswith("UNKNOWN")]
    return a, b, c


def citations_resolve(root: Path = REPO) -> List[str]:
    """Unresolved citations. Population C is not printed while any exist."""
    try:
        text = (root / "kai-pm" / "DECISIONS.md").read_text(encoding="utf-8")
    except OSError:
        return ["kai-pm/DECISIONS.md is unreadable"]
    bad = []
    for table in (RUNTIME_EVIDENCE, ROLE_EVIDENCE):
        for svc, ev in table.items():
            if not re.search(rf"\b{re.escape(ev['cite'])}\b", text):
                bad.append(f"{svc} cites {ev['cite']}, which is not in DECISIONS.md")
    for svc, ev in ROLE_EVIDENCE.items():
        if ev["role"] not in ROLE_VOCABULARY:
            bad.append(f"{svc} claims role {ev['role']!r}, which is outside "
                       f"the declared vocabulary {sorted(ROLE_VOCABULARY)}")
    return bad


# ══ output ═══════════════════════════════════════════════════════════

def main() -> int:
    require(COMPOSE_FILES)
    rows = survey()
    a, b, c = populations(rows)

    print(inspected(len(rows), "service definition(s)",
                    f"across {len(COMPOSE_FILES)} compose files, "
                    f"{len(compose_files())} found on disk"))

    print(f"\n  POPULATION A  source reachability      {len(a)}")
    print(f"  POPULATION B  egress-restricted of A    {len(b)}")
    bad = citations_resolve()
    if bad:
        print("  POPULATION C  REFUSED — a citation does not resolve:")
        for line in bad:
            print(f"                  {line}")
    else:
        print(f"  POPULATION C  runtime-qualified of B    {len(c)}")
    print("\n  A is source reachability ONLY. It is not a count of services "
          "affected in\n  production. B is where the offline invariant "
          "applies. C is what we have\n  actually observed; every other "
          "row in B is UNKNOWN, which is an evidence\n  status and not a "
          "history.")

    print("\n" + "=" * 78)
    print("POPULATION A — a runnable container path can reach a model load")
    print("=" * 78)
    for r in a:
        print(f"\n  {r['service']}  [{r['reach']}]")
        print(f"    image/Dockerfile      {r['dockerfile']}")
        print(f"    egress contract       {r['egress']}")
        print(f"    model-loading path    {r['path'] or '(third-party, not traceable here)'}"
              + (f"  -> {r['library']}" if r['library'] else ""))
        print(f"    load timing           {r['timing']}")
        print(f"    library installed     {r['installed']}"
              + (f"   guarded import: {r['guarded']}" if r['guarded'] != UNKNOWN else ""))
        print(f"    role-required?        {r['role']}"
              + (f" — {r['role_note']}" if r['role_note'] else ""))
        print(f"    baked asset?          {r['baked']}")
        print(f"    deterministic cache?  {r['cache']}")
        print(f"    offline enforced?     {r['offline']}")
        print(f"    build fail-closed?    {r['fail_closed']}")
        print(f"    mount shadow risk?    {r['shadow']}")
        print(f"    listener gated?       {r['listener_gated']}")
        print(f"    runtime evidence      {r['evidence']}")
        if r["why"]:
            print(f"    NOTE                  {r['why']}")
        for line in r["trace_blind"]:
            print(f"    TRACE INCOMPLETE      {line}")

    blind_rows = [r for r in rows if r["trace_blind"]]
    print(f"\n  branches the trace could not follow: "
          f"{sum(len(r['trace_blind']) for r in blind_rows)} across "
          f"{len(blind_rows)} service(s)")
    for r in blind_rows:
        for line in r["trace_blind"]:
            print(f"    {r['service']:22} {line}")
    if not blind_rows:
        print("    (none — every resolved import was parsed)")

    print("\n" + "=" * 78)
    print("POPULATION B — of A, deployed under an egress-restricted contract")
    print("=" * 78)
    for r in b:
        print(f"  {r['service']:24} {r['egress']:20} timing={r['timing']}")
    outside = [r for r in a if r not in b]
    print(f"\n  in A but NOT in B: {[r['service'] for r in outside] or '(none)'}")
    print("  The offline-startup invariant does not apply to those.")

    if not bad:
        print("\n" + "=" * 78)
        print("POPULATION C — of B, with a CITED runtime observation")
        print("=" * 78)
        for r in c:
            print(f"  {r['service']:24} {r['evidence']}")
        unknown = [r["service"] for r in b if r not in c]
        print(f"\n  in B with runtime status UNKNOWN: {unknown or '(none)'}")

    print("\n" + "=" * 78)
    print("EXCLUDED, and why — the half of a denominator nobody prints")
    print("=" * 78)
    reasons: Dict[str, List[str]] = {}
    for r in rows:
        if r in a:
            continue
        reasons.setdefault(r["why"] or "no call to a model loader is "
                           "reachable from the entrypoint", []).append(r["service"])
    for why, names in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {len(names)}x  {why}")
        print(f"       {', '.join(sorted(names))}")

    not_measured = [r for r in rows if r["image_only"]]
    if not_measured:
        print("\n" + "=" * 78)
        print("NOT MEASURED — a stated blind spot, not a clean bill of health")
        print("=" * 78)
        print("  These run images this repository does not build. The tracer "
              "reads repo\n  source; it cannot open a published image. "
              "Whether any of them loads a\n  model at runtime is UNKNOWN "
              "here — and at least two of them serve models\n  for a living, "
              "so reading this section as 'clean' would be the exact\n  "
              "mistake the populations above are separated to prevent.\n")
        for r in sorted(not_measured, key=lambda r: r["service"]):
            print(f"    {r['service']:22} {r['dockerfile']:34} {r['egress']}")

    print("\n  Libraries observed and deliberately NOT counted as loaders:")
    for lib, why in sorted(NOT_A_LOADER.items()):
        print(f"    {lib:14} {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
