#!/usr/bin/env python3
"""HOUSE_H2 — STAGE-A CLOSED-MEMBERSHIP CONSTRUCTION AND STAGE-B BINDING.

Authorised by D379 ("Stage-A closed-membership construction; EXTERNAL
Stage-B artifact binding"), given H2_STAGE_A_V1 semantics by D380, and
given H2_STAGE_A_V2 semantics and the production-use rules by D381.
D382 and D383/D384 change no Stage-A proposition and are NOT in the
governance set.

WHY V2 IS A NEW SCHEMA AND NOT AN AMENDED V1. A schema identifier must
identify ONE deterministic validation contract. Amending V1 would make
`schema = "H2_STAGE_A_V1"` denote [D379, D380] before D381 and
[D379, D380, D381] after, so a validator holding only that identifier
could not tell which closed rule it names without consulting outside
state. That is not a collision problem -- the governance array is inside
the canonical bytes, so the two forms hash differently -- it is a
VALIDATOR DETERMINISM problem, and self-description is the property this
tranche exists to build.

NO PRODUCTION STAGE-A IDENTITY IS CREATED BY IMPORTING THIS MODULE.
Construction is a call, calibration descriptors are explicitly labelled,
and the D379/D381 release forbids a real Stage A.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# ── D381 §11/§17 — the V2 schema, and the V1 it does not amend ────────
SCHEMA_V1 = "H2_STAGE_A_V1"
SCHEMA_V2 = "H2_STAGE_A_V2"
DOMAIN_V1 = "H2-STAGE-A-V1"
DOMAIN_V2 = "H2-STAGE-A-V2"
STDLIB_SCHEMA = "H2_PY_STDLIB_V1"          # D380 §7, INHERITED UNCHANGED

MODE_PRODUCTION = "PRODUCTION"
MODE_CALIBRATION = "CALIBRATION"
MODES = (MODE_PRODUCTION, MODE_CALIBRATION)

# D381 §14 — the NEW CLOSED governance contract for V2. Exact identities
# only: never "latest", never a branch HEAD, never the current
# DECISIONS.md blob. D382/D383/D384 changed no Stage-A proposition and
# are deliberately absent; a V2 descriptor naming one is REFUSED below as
# an unknown governing decision, which is correct and not an oversight.
GOVERNANCE_V2 = (
    ("D379", "608d706d8452b8e578a484f7b75331a5cb9c28d9"),
    ("D380", "c50989779baf0485e2a4a5ceb2113093441691e3"),
    ("D381", "838b7637058c5ba3b8f3b6c5430ebdc324249b96"),
)
GOVERNANCE_V1 = GOVERNANCE_V2[:2]          # historical, NOT amended

# D380 §6.2 / D381 §15 — exactly ten members, no more and no fewer.
H2_SOURCES = (
    "kai-pm/house_in_order_h2_v13/cal_fixtures.py",
    "kai-pm/house_in_order_h2_v13/classify.py",
    "kai-pm/house_in_order_h2_v13/envelope.py",
    "kai-pm/house_in_order_h2_v13/holdout.py",
    "kai-pm/house_in_order_h2_v13/ontology.py",
    "kai-pm/house_in_order_h2_v13/passa.py",
    "kai-pm/house_in_order_h2_v13/qualify.py",
    "kai-pm/house_in_order_h2_v13/run_h2_v12.py",
    "kai-pm/house_in_order_h2_v13/stage_identity.py",
    "kai-pm/house_in_order_h2_v13/subjectbind.py",
)

TOP_LEVEL_FIELDS = ("schema", "mode", "h2_sources", "contract", "governance",
                    "subject", "tree_paths", "census", "history", "runtime")

RUNTIME_FIELDS = ("executable_sha256", "implementation_name", "cache_tag",
                  "version", "stdlib_identity", "dont_write_bytecode")


class StageIdentityError(AssertionError):
    """A Stage-A rule refused. Raised, never returned."""


# ── RFC 8785 JCS, bounded to the types this schema admits ─────────────
def _jcs(obj) -> bytes:
    """RFC 8785 canonical JSON for the value types this schema uses.

    BOUNDED ON PURPOSE. Objects, arrays, strings, integers and booleans
    are canonicalised; a float REFUSES rather than being serialised by a
    partial reimplementation of ECMAScript Number::toString. This schema
    contains no float, so the narrowing costs nothing real and removes a
    whole class of silent divergence between implementations.

    Keys sort by UTF-16 code unit, which is what RFC 8785 specifies and
    is NOT the same as sorting Python strings by code point above the
    BMP.
    """
    out = []

    def enc_str(s):
        out.append('"')
        for ch in s:
            o = ord(ch)
            if ch == '"':
                out.append('\\"')
            elif ch == "\\":
                out.append("\\\\")
            elif ch == "\b":
                out.append("\\b")
            elif ch == "\f":
                out.append("\\f")
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            elif o < 0x20:
                out.append("\\u%04x" % o)
            else:
                out.append(ch)
        out.append('"')

    def walk(v):
        if v is True:
            out.append("true")
        elif v is False:
            out.append("false")
        elif v is None:
            out.append("null")
        elif isinstance(v, str):
            enc_str(v)
        elif isinstance(v, int):
            out.append(str(v))
        elif isinstance(v, float):
            raise StageIdentityError(
                "REFUSE: a float reached canonicalisation. This schema "
                "admits no float, and a partial ECMAScript number "
                "serialiser is a silent-divergence class.")
        elif isinstance(v, (list, tuple)):
            out.append("[")
            for i, x in enumerate(v):
                if i:
                    out.append(",")
                walk(x)
            out.append("]")
        elif isinstance(v, dict):
            items = sorted(v.items(),
                           key=lambda kv: kv[0].encode("utf-16-be"))
            out.append("{")
            for i, (k, x) in enumerate(items):
                if i:
                    out.append(",")
                if not isinstance(k, str):
                    raise StageIdentityError("REFUSE: non-string object key")
                enc_str(k)
                out.append(":")
                walk(x)
            out.append("}")
        else:
            raise StageIdentityError(
                f"REFUSE: unserialisable type {type(v).__name__}")

    walk(obj)
    return "".join(out).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm_path(p: str) -> str:
    """D380 §6.10 path rules. REFUSE rather than repair."""
    import unicodedata
    if not isinstance(p, str) or not p:
        raise StageIdentityError(f"REFUSE: empty or non-string path {p!r}")
    if p != unicodedata.normalize("NFC", p):
        raise StageIdentityError(f"REFUSE: path not Unicode NFC: {p!r}")
    if "\\" in p or p.startswith("/"):
        raise StageIdentityError(f"REFUSE: non-POSIX or absolute path {p!r}")
    if any(seg in ("", ".", "..") for seg in p.split("/")):
        raise StageIdentityError(f"REFUSE: bad path segment in {p!r}")
    return p


# ── D380 §7 — H2_PY_STDLIB_V1, NON-PLUGGABLE (D381 §16) ───────────────
#
# THE MECHANISM IS THE ABSENCE OF A SEAM, NOT A FUNCTION-IDENTITY CHECK.
# `build_stage_a` calls `_stdlib_identity()` directly. There is no
# parameter, no constructor argument, no schema selector and no callback
# by which a caller can supply a digest or an alternative algorithm, so
# there is nothing to compare a function object against. A control proves
# this by attempting each substitution and being refused by the signature
# itself.
def _governed_roots():
    """D380 §7.1 — stdlib/platstdlib roots, with purelib/platlib captured
    ONLY to exclude external package populations (§7.3)."""
    import sysconfig
    paths = sysconfig.get_paths()
    def real(k):
        v = paths.get(k)
        return os.path.realpath(v) if v else None
    stdlib, plat = real("stdlib"), real("platstdlib")
    external = [d for d in (real("purelib"), real("platlib")) if d]
    if stdlib is None:
        raise StageIdentityError("REFUSE: no stdlib path from sysconfig")
    if plat is None or plat == stdlib:                       # CASE A
        roots = {"stdlib": stdlib}
    else:                                                    # CASE B / C
        roots = {"stdlib": stdlib, "platstdlib": plat}
    return roots, external


def _is_external(path: str, external) -> bool:
    """D380 §7.3 — external package population BEATS stdlib containment."""
    for e in external:
        if path == e or path.startswith(e + os.sep):
            return True
    parts = path.split(os.sep)
    return "site-packages" in parts or "dist-packages" in parts


def _owning_root(path: str, roots):
    """D380 §7.2 — MOST-SPECIFIC governed root wins. Exactly one owner."""
    owners = [(rid, r) for rid, r in roots.items()
              if path == r or path.startswith(r + os.sep)]
    if not owners:
        return None
    rid, r = max(owners, key=lambda t: len(t[1]))
    return rid, r


def build_stdlib_identity():
    """The governed H2_PY_STDLIB_V1 object and its digest.

    Returns (identity_hex, object, canonical_bytes). Enumeration uses
    lstat semantics, never follows directory symlinks, excludes mutable
    bytecode, and stores no absolute path.
    """
    roots, external = _governed_roots()
    entries, seen = [], {}

    for rid in sorted(roots):
        root = roots[rid]
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    children = list(it)
            except (PermissionError, FileNotFoundError):
                continue
            for e in children:
                full = e.path
                name = e.name
                if name == "__pycache__":
                    continue
                if _is_external(full, external):      # §7.3, BEFORE accept
                    continue
                own = _owning_root(full, roots)
                if own is None or own[0] != rid:      # §7.2 single owner
                    continue
                rel = _norm_path(os.path.relpath(full, root).replace(os.sep, "/"))
                st = os.lstat(full)
                import stat as _stat
                if _stat.S_ISLNK(st.st_mode):
                    tgt = os.path.realpath(full)
                    if not os.path.exists(tgt):
                        raise StageIdentityError(
                            f"REFUSE: dangling symlink {rel} under {rid}")
                    town = _owning_root(tgt, roots)
                    if town is None:
                        raise StageIdentityError(
                            f"REFUSE: symlink {rel} under {rid} resolves "
                            f"OUTSIDE the governed root set. There is no "
                            f"external-dependency escape hatch in this schema.")
                    trid, troot = town
                    trel = _norm_path(
                        os.path.relpath(tgt, troot).replace(os.sep, "/"))
                    ent = {"type": "symlink", "root_id": rid, "path": rel,
                           "target_root_id": trid, "target": trel}
                elif _stat.S_ISDIR(st.st_mode):
                    stack.append(full)                # not a symlink: descend
                    continue
                elif _stat.S_ISREG(st.st_mode):
                    if name.endswith((".pyc", ".pyo")):
                        continue
                    try:
                        data = open(full, "rb").read()
                    except (PermissionError, FileNotFoundError):
                        continue
                    ent = {"type": "file", "root_id": rid, "path": rel,
                           "sha256": sha256_hex(data)}
                else:
                    raise StageIdentityError(
                        f"REFUSE: unsupported filesystem type at {rid}:{rel}")
                key = (ent["root_id"], ent["path"])
                if key in seen:                       # §7.7
                    raise StageIdentityError(
                        f"REFUSE: duplicate canonical member {key}")
                seen[key] = True
                entries.append(ent)

    def sort_key(e):                                  # §7.8
        disc = (e["sha256"] if e["type"] == "file"
                else e["target_root_id"] + "\x00" + e["target"])
        return (e["type"], e["root_id"], e["path"], disc)

    entries.sort(key=sort_key)
    obj = {"schema": STDLIB_SCHEMA, "entries": entries}
    S = _jcs(obj)
    return sha256_hex(S), obj, S


def _stdlib_identity() -> str:
    return build_stdlib_identity()[0]


def build_runtime():
    """D380 §6.9 — portable, location-independent runtime identity.

    `dont_write_bytecode = True` is the ONLY valid value: a producer that
    may write .pyc files mutates the population it just hashed.
    """
    exe = os.path.realpath(sys.executable)
    return {
        "executable_sha256": sha256_hex(open(exe, "rb").read()),
        "implementation_name": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": sys.version,
        "stdlib_identity": _stdlib_identity(),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
    }


# ── governance validation ─────────────────────────────────────────────
def validate_governance(gov, *, schema):
    """D381 §14 REFUSE conditions, order checked BEFORE canonicalisation."""
    expected = GOVERNANCE_V2 if schema == SCHEMA_V2 else GOVERNANCE_V1
    if not isinstance(gov, list) or not gov:
        raise StageIdentityError("REFUSE: governance is not a non-empty list")
    ids = []
    for e in gov:
        if (not isinstance(e, dict) or set(e) != {"decision_id", "bank_commit_sha"}
                or not isinstance(e["decision_id"], str)
                or not isinstance(e["bank_commit_sha"], str)):
            raise StageIdentityError(f"REFUSE: malformed governance entry {e!r}")
        ids.append(e["decision_id"])
    if len(set(ids)) != len(ids):
        raise StageIdentityError(f"REFUSE: duplicate governing decision in {ids}")
    try:
        nums = [int(i.lstrip("D")) for i in ids]
    except ValueError:
        raise StageIdentityError(f"REFUSE: non-numeric decision id in {ids}")
    if nums != sorted(nums):
        raise StageIdentityError(
            f"REFUSE: governance not in ascending numeric decision order: "
            f"{ids}. Order is checked BEFORE canonicalisation.")
    exp = {d: c for d, c in expected}
    for e in gov:
        d, c = e["decision_id"], e["bank_commit_sha"]
        if d not in exp:
            raise StageIdentityError(
                f"REFUSE: {d} is an unknown governing decision for {schema}. "
                f"The closed set is {[x for x, _ in expected]}.")
        if c != exp[d]:
            raise StageIdentityError(
                f"REFUSE: wrong bank commit for {d}: {c} != {exp[d]}")
    missing = [d for d, _ in expected if d not in ids]
    if missing:
        raise StageIdentityError(
            f"REFUSE: {schema} governance is missing {missing}")
    return True


def validate_descriptor(desc):
    """Every structural Stage-A rule, applied BEFORE any identity is taken.

    D381 §12: a Stage-A production path governed after D381 MUST REFUSE
    schema=H2_STAGE_A_V1 with mode=PRODUCTION, UNCONDITIONALLY, decided
    from the artefact in hand. No lineage inference, no branch state, no
    external mutable state.
    """
    if not isinstance(desc, dict):
        raise StageIdentityError("REFUSE: descriptor is not an object")
    extra = set(desc) - set(TOP_LEVEL_FIELDS)
    if extra:
        raise StageIdentityError(f"REFUSE: unknown top-level field(s) {sorted(extra)}")
    missing = set(TOP_LEVEL_FIELDS) - set(desc)
    if missing:
        raise StageIdentityError(f"REFUSE: missing top-level field(s) {sorted(missing)}")

    schema, mode = desc["schema"], desc["mode"]
    if schema not in (SCHEMA_V1, SCHEMA_V2):
        raise StageIdentityError(f"REFUSE: unknown schema {schema!r}")
    if mode not in MODES:
        raise StageIdentityError(f"REFUSE: unknown mode {mode!r}")

    if schema == SCHEMA_V1 and mode == MODE_PRODUCTION:
        raise StageIdentityError(
            "REFUSE: H2_STAGE_A_V1 + PRODUCTION. D381 §12 forbids any "
            "post-D381 Stage-A production construction or validation on the "
            "V1 schema, unconditionally and from the artefact alone. V1 "
            "remains available ONLY through the explicitly calibration-only "
            "path, carrying zero production, admission and holdout weight.")

    validate_governance(desc["governance"], schema=schema)

    src = desc["h2_sources"]
    if not isinstance(src, list):
        raise StageIdentityError("REFUSE: h2_sources is not a list")
    paths = []
    for m in src:
        if not isinstance(m, dict) or set(m) != {"path", "sha256"}:
            raise StageIdentityError(f"REFUSE: malformed h2_sources member {m!r}")
        paths.append(_norm_path(m["path"]))
    if len(set(paths)) != len(paths):
        raise StageIdentityError("REFUSE: duplicate normalised h2_sources path")
    if set(paths) != set(H2_SOURCES):
        miss = sorted(set(H2_SOURCES) - set(paths))
        extra = sorted(set(paths) - set(H2_SOURCES))
        raise StageIdentityError(
            f"REFUSE: h2_sources population is not the governed ten. "
            f"missing={miss} additional={extra}")

    rt = desc["runtime"]
    if not isinstance(rt, dict) or set(rt) != set(RUNTIME_FIELDS):
        raise StageIdentityError(f"REFUSE: malformed runtime {sorted(rt) if isinstance(rt, dict) else rt!r}")
    if rt["dont_write_bytecode"] is not True:
        raise StageIdentityError(
            "REFUSE: dont_write_bytecode must be true — a producer that may "
            "write .pyc mutates the population it just hashed")
    return True


def canonical_bytes(desc) -> bytes:
    """Validate, canonicalise, and prove the round-trip reproduces D."""
    validate_descriptor(desc)
    D = _jcs(desc)
    if _jcs(json.loads(D.decode("utf-8"))) != D:
        raise StageIdentityError(
            "REFUSE: reparse+recanonicalise did not reproduce the exact "
            "canonical bytes (D380 §6.10)")
    return D


def stage_a_identity(desc) -> str:
    """D381 §17. The domain separator is bound to the SCHEMA VALUE.

    DUAL VERSIONING IS DEFENCE IN DEPTH. The schema tag lives inside the
    canonical bytes and the separator outside them, and each ALONE already
    separates the identities. A later reader may notice one looks
    redundant; NEITHER MAY BE REMOVED ON THAT GROUND (D381 §17).
    """
    D = canonical_bytes(desc)
    dom = DOMAIN_V2 if desc["schema"] == SCHEMA_V2 else DOMAIN_V1
    return sha256_hex(dom.encode("utf-8") + b"\x00" + D)


def stage_a_descriptor_digest(desc) -> str:
    return sha256_hex(canonical_bytes(desc))


# ── D379 §5 — runtime-derived producer population ─────────────────────
#
# INC-2026-09-19-37. The previous implementation opened with
#
#     f = getattr(mod, "__file__", None)
#     if not f:
#         continue                      # built-in / frozen
#
# and that comment stated an assumption the code never tested. Three
# synthetic origins and three REAL live modules (__main__, typing.io,
# typing.re) were silently discarded: none is mechanically built-in and
# none is mechanically frozen.
#
# FILESYSTEM PRESENCE IS NOT THE DISCRIMINATOR, IN EITHER DIRECTION.
# Measured on this interpreter, `os` carries a __file__ AND reports
# spec.origin 'frozen'. So a missing file does not imply built-in, and a
# present file does not imply not-frozen. IMPORT METADATA DECIDES FIRST.
#
# There is no SKIP class. Every observed origin earns exactly one of
# BUILTIN, FROZEN, H2, CENSUS, STDLIB — or it REFUSES.

CLASS_BUILTIN = "BUILTIN"
CLASS_FROZEN = "FROZEN"
CLASS_H2 = "H2"
CLASS_CENSUS = "CENSUS"
CLASS_STDLIB = "STDLIB"
ORIGIN_CLASSES = (CLASS_BUILTIN, CLASS_FROZEN, CLASS_H2, CLASS_CENSUS,
                  CLASS_STDLIB)


class Origin:
    """ONE OBSERVED IMPORT ORIGIN, as data.

    Separating observation from classification is what lets a hostile
    control supply a synthetic origin WITH ITS OWN PREDECLARED EXPECTED
    ANSWER, instead of asking the classifier what it thinks and calling
    that the expectation. The qualifier-side version of that mistake is
    INC-36.
    """
    __slots__ = ("name", "has_spec", "spec_origin", "file", "is_main")

    def __init__(self, name, *, has_spec=True, spec_origin=None, file=None,
                 is_main=False):
        self.name = name
        self.has_spec = has_spec
        self.spec_origin = spec_origin
        self.file = file
        self.is_main = is_main

    def __repr__(self):
        return (f"Origin({self.name!r}, spec_origin={self.spec_origin!r}, "
                f"file={self.file!r}, is_main={self.is_main})")


def observe_origin(name, mod):
    """Read one live module's import metadata. NO classification here."""
    spec = getattr(mod, "__spec__", None)
    return Origin(name, has_spec=spec is not None,
                  spec_origin=getattr(spec, "origin", None),
                  file=getattr(mod, "__file__", None),
                  is_main=(name == "__main__"))


def classify_origin(o, *, repo_root, roots=None, external=None):
    """Classify ONE origin into the closed set, or REFUSE. D379 §5/§6.

    Returns (class, identity_detail). Raises StageIdentityError to REFUSE.
    """
    if roots is None or external is None:
        roots, external = _governed_roots()
    root = pathlib.Path(repo_root).resolve()
    h2root = (root / "kai-pm" / "house_in_order_h2_v13").resolve()
    census = (root / "kai-pm" / "house_in_order_census_v11").resolve()

    # 1. IMPORT METADATA FIRST, and it OUTRANKS __file__ (Kai §4).
    #    A frozen module carrying a convenience path stays FROZEN.
    if o.spec_origin == "built-in":
        return CLASS_BUILTIN, o.name
    if o.spec_origin == "frozen":
        return CLASS_FROZEN, o.name

    # 2. __main__ is handled EXPLICITLY, as D379 §5 requires in terms.
    #    An unsourced production entry point cannot satisfy that clause,
    #    and pretending it is built-in is exactly the INC-37 equivalence.
    if o.is_main and not o.file and not _is_fs_origin(o.spec_origin):
        raise StageIdentityError(
            f"REFUSE: the executing entry point __main__ has no mechanically "
            f"establishable source (spec_origin={o.spec_origin!r}, "
            f"file={o.file!r}). D379 §5 requires the entry-point source to be "
            f"included EXPLICITLY; an unsourced __main__ is not built-in and "
            f"is not frozen.")

    # 3. A filesystem origin. Where BOTH a filesystem spec.origin and a
    #    __file__ exist they must identify the SAME source object; picking
    #    whichever is convenient is how two implementations drift apart.
    cands = []
    if _is_fs_origin(o.spec_origin):
        cands.append(os.path.realpath(o.spec_origin))
    if o.file:
        cands.append(os.path.realpath(o.file))
    if len(cands) == 2 and cands[0] != cands[1]:
        raise StageIdentityError(
            f"REFUSE: {o.name} has a filesystem spec.origin and a __file__ "
            f"that resolve to DIFFERENT sources: {cands[0]} != {cands[1]}")
    if not cands:
        # 4. No file, and the origin is neither built-in nor frozen.
        raise StageIdentityError(
            f"REFUSE: {o.name} has no filesystem source and its origin is "
            f"neither 'built-in' nor 'frozen' (has_spec={o.has_spec}, "
            f"spec_origin={o.spec_origin!r}). A missing __file__ is NOT an "
            f"answer (INC-2026-09-19-37).")

    rp = cands[0]
    p = pathlib.Path(rp)
    if str(p).startswith(str(h2root) + os.sep):
        return CLASS_H2, p.relative_to(root).as_posix()
    if str(p).startswith(str(census) + os.sep):
        return CLASS_CENSUS, p.relative_to(root).as_posix()
    # external package population BEATS stdlib containment (D380 §7.3)
    if _is_external(rp, external):
        raise StageIdentityError(
            f"REFUSE: {o.name} is a loaded non-stdlib EXTERNAL module at "
            f"{rp}, outside every governed Stage-A root, with no explicit "
            f"Stage-A dependency identity. None is invented here (D379 §6).")
    if _owning_root(rp, roots) is not None:
        return CLASS_STDLIB, rp
    raise StageIdentityError(
        f"REFUSE: {o.name} at {rp} lies outside the governed H2 root, the "
        f"governed Census root and the governed Python stdlib classification.")


def _is_fs_origin(origin):
    """A spec origin that names a real filesystem location."""
    return bool(origin) and origin not in ("built-in", "frozen") \
        and os.path.exists(origin)


def producer_population(repo_root):
    """Every observed import origin, classified or REFUSED. D379 §5.

    Directional, per D379 §5: governed runtime modules actually loaded
    MUST be represented in Stage A; NOT every Stage-A module must appear
    in every process.

    Returns (members, offenders). `members` carries
    (class, identity, sha256-or-"") and is deduplicated BY RESOLVED SOURCE
    IDENTITY, so an entry point reached both explicitly and through the
    ordinary traversal yields ONE canonical member. NO SILENT MEMBER: an
    origin that cannot be classified appears in `offenders` with the exact
    refusal text, never by omission.
    """
    root = pathlib.Path(repo_root).resolve()
    roots, external = _governed_roots()
    members, offenders, seen = [], [], set()

    def take(o):
        try:
            cls, ident = classify_origin(o, repo_root=root, roots=roots,
                                         external=external)
        except StageIdentityError as e:
            offenders.append((o.name, str(e)))
            return
        key = (cls, ident)
        if key in seen:                       # D379 §5: deduplicate
            return
        seen.add(key)
        digest = ""
        if cls in (CLASS_H2, CLASS_CENSUS):
            digest = sha256_hex((root / ident).read_bytes())
        members.append((cls, ident, digest))

    # D379 §5 — "the executing entry-point source (__main__) is included
    # EXPLICITLY". Explicitly means BEFORE and INDEPENDENTLY of the general
    # traversal, not "incidentally, if it happens to carry a path".
    main = sys.modules.get("__main__")
    if main is not None:
        take(observe_origin("__main__", main))

    for name, mod in sorted(sys.modules.items()):
        if name == "__main__":
            continue                          # already taken, explicitly
        if mod is None:
            continue
        take(observe_origin(name, mod))

    members.sort()
    return members, offenders


def check_population(repo_root, descriptor, when="production"):
    """Observe this process's population and verify it against Stage A.

    ONE AUTHORITY, BOTH PRODUCERS. This check previously existed only as a
    private helper inside passa.py, so the Pass-A producer verified its own
    executing bytes and the CLASSIFICATION producer did NOT: it observed
    its population, recorded it, and never compared a single digest against
    the descriptor. D379 Q1a-3 -- "one governed classification-producer byte
    changed after Stage A fixed -> classification production REFUSES" -- had
    no implementation in the shipped executable at all.

    That is INC-38's shape a second time: correct logic reachable in one
    place while the real program walks around it. The remedy is not a copy
    in the other producer; it is one authority that both consume.

    Returns the observed members. Raises StageIdentityError to REFUSE.
    """
    members, offenders = producer_population(repo_root)
    if offenders:
        raise StageIdentityError(
            f"REFUSE ({when}): the producer runtime population contains "
            f"origins outside every governed Stage-A root, with no Stage-A "
            f"dependency identity: "
            + "; ".join(f"{n}: {str(w)[:90]}" for n, w in offenders[:4]))
    stage_h2 = {m["path"]: m["sha256"] for m in descriptor["h2_sources"]}
    for cls, identity, digest in members:
        if cls != CLASS_H2:
            continue
        if identity not in stage_h2:
            raise StageIdentityError(
                f"REFUSE ({when}): loaded H2 source {identity} is NOT "
                f"represented in Stage A. No silent runtime expansion.")
        if digest != stage_h2[identity]:
            raise StageIdentityError(
                f"REFUSE ({when}): loaded H2 source {identity} byte mismatch "
                f"against Stage A: {digest} != {stage_h2[identity]}")
    return members


# ── Stage-B: EXTERNAL binding on FINAL bytes (D379 §4) ────────────────
def stage_b_binding(artifact_path, *, artifact_kind, identity,
                    producer_component, producer_provenance_digest):
    """Hash the EXACT FINAL bytes of a finished artefact, from outside it.

    THE SELF-HASH PROHIBITION IS STRUCTURAL HERE. This function takes a
    path to an ALREADY-FINALISED file and hashes what is on disk. There is
    no code path by which a producer embeds its own whole-file digest into
    the bytes being digested, because the digest is never written back.
    """
    p = pathlib.Path(artifact_path)
    data = p.read_bytes()
    return {"artifact_path": _norm_path(p.name) if not p.is_absolute() else p.name,
            "artifact_sha256": sha256_hex(data),
            "artifact_kind": artifact_kind,
            "stage_a_identity": identity,
            "producer_component": producer_component,
            "producer_provenance_digest": producer_provenance_digest}


def stage_b_aggregate(bindings):
    """Canonical ordered aggregate over EXTERNAL bindings only."""
    rows = sorted(bindings, key=lambda b: (b["artifact_path"],
                                           b["artifact_sha256"]))
    return sha256_hex(_jcs({"schema": "H2_STAGE_B_V1", "bindings": rows}))


if __name__ == "__main__":
    print(json.dumps({"schema_v2": SCHEMA_V2, "domain_v2": DOMAIN_V2,
                      "stdlib_schema": STDLIB_SCHEMA,
                      "governance_v2": [list(g) for g in GOVERNANCE_V2],
                      "h2_sources": list(H2_SOURCES)}, indent=1))


# ── Q1a — PRODUCER-BYTE PROVENANCE (D379 §4/§5) ───────────────────────
#
# "WHO PRODUCED THE RESULT?" — a separate question from §8(6)'s "are the
# QUALIFIER's own executing bytes governed?". The two are never collapsed:
# this verifies a RECORDED provenance block against the Stage-A identity
# it claims, and it deliberately does NOT consult today's sys.modules,
# because today's module state cannot establish yesterday's producer bytes.
# D379 §4 names the canonical in-band block EXACTLY, and it names a
# DIFFERENT field set per component. A verifier that accepts either shape
# accepts a PASS_A block emitted by classification, and a classification
# block with no input_binding at all.
_PROV_COMMON = ("stage_a_identity", "stage_a_descriptor_digest",
                "producer_component", "producer_population",
                "producer_denominator", "runtime_identity",
                "subject_commit", "subject_tree", "tree_paths_identity")
PROV_SHAPE = {
    "PASS_A": _PROV_COMMON + ("census_identity", "history_source_identity"),
    "CLASSIFICATION": _PROV_COMMON + ("input_binding",),
}
INPUT_BINDING_FIELDS = ("pass_a_artifact_sha256", "pass_a_stage_a_identity",
                        "pass_a_producer_provenance_digest")


def provenance_digest(prov):
    """Canonical digest of an in-band provenance block (D379 §4)."""
    return sha256_hex(_jcs(prov))


def _prov_comparators(descriptor):
    """The AUTHORITATIVE expected value of every mechanically verifiable
    D379 §4 field, derived from the Stage-A descriptor — never from the
    provenance block being verified."""
    return {
        "stage_a_identity": stage_a_identity(descriptor),
        "stage_a_descriptor_digest": stage_a_descriptor_digest(descriptor),
        "runtime_identity": descriptor.get("runtime"),
        "subject_commit": descriptor["subject"]["commit"],
        "subject_tree": descriptor["subject"]["tree"],
        "tree_paths_identity":
            descriptor["tree_paths"]["tree_paths_identity"],
        "census_identity": descriptor["census"]["aggregate_sha256"],
        "history_source_identity":
            descriptor["history"]["reachable_set_sha256"],
    }


def verify_provenance(recorded, descriptor, *, pass_a_bytes=None,
                      artifact_bytes=None):
    """Verify a RECORDED producer provenance against Stage A, or REFUSE.

    NOT COMPLETE, AND THE LIMIT IS NAMED. Kai, on 0523108: calling this
    path complete is still too wide. Two things it does NOT do:

      1. `producer_population` members outside CLASS_H2 get shape checking
         only. No semantic check is applied to CENSUS, STDLIB, BUILTIN or
         FROZEN members.
      2. A SELF-CONSISTENT tamper -- one that shortens the recorded
         population AND its denominator together -- is NOT detectable
         here, and cannot be, because D379 §5 says a process need not
         load every Stage-A member. Distinguishing "legitimately fewer"
         from "deleted after production" requires the EXTERNAL Stage-B
         binding's `producer_provenance_digest`, which no caller of this
         function is currently given. That is banked case Q1a-7 and it is
         RETURNED TO KAI as a CLI contract question, not worked around
         here with another in-object proxy.

    What it DOES do, which the previous implementation did not:
    `stage_a_identity` and `producer_population` and returned success — so
    a block whose `subject_commit`, `runtime_identity`, `census_identity`,
    `tree_paths_identity` or `producer_denominator` were wrong verified
    CLEAN. Two of eleven fields checked is not "provenance verified"; it is
    a narrower check wearing a wider name (R5).

    FAIL CLOSED ON SHAPE. The accepted field set is COMPONENT-SPECIFIC and
    compared for EXACT EQUALITY: a missing field REFUSES, and so does an
    unexpected one, because an unexpected field is either a schema drift
    nobody governed or a place to hide a self-output digest.

    Returns (stage_a_identity, verified_h2_identities, unverified_fields).
    `unverified_fields` is NEVER empty-by-assumption: it names every field
    this call could not establish against an authority, so a caller cannot
    mistake a partial verification for a complete one (R17).
    """
    ident = stage_a_identity(descriptor)
    if not isinstance(recorded, dict):
        raise StageIdentityError("REFUSE: provenance is not an object")

    comp = recorded.get("producer_component")
    if comp not in PROV_SHAPE:
        raise StageIdentityError(
            f"REFUSE: producer_component {comp!r} is not one of the D379 §4 "
            f"governed components {sorted(PROV_SHAPE)}. An unrecognised "
            f"component has no governed field set to verify against.")
    expect_fields = set(PROV_SHAPE[comp])
    got = set(recorded)
    if got != expect_fields:
        raise StageIdentityError(
            f"REFUSE: {comp} provenance does not carry the D379 §4 field "
            f"set. missing={sorted(expect_fields - got)} "
            f"unexpected={sorted(got - expect_fields)}")

    # ── D379 §4: NO SELF-OUTPUT DIGEST IN EITHER ──────────────────────
    # BEFORE the value comparators. A planted self-digest necessarily makes
    # some field diverge from Stage A, so a comparator running first
    # reports a stale-field mismatch and the structural prohibition is
    # never the reason given. Same red process, wrong predicate -- which is
    # the defect this tranche exists to remove.
    #
    # WHAT IS ACTUALLY CONSTRUCTIBLE, stated plainly. A digest of the exact
    # final bytes INCLUDING that digest is a fixed point and no producer
    # can compute one. The self-references a producer CAN construct are:
    #
    #   (a) the digest of the artefact bytes as they stand, planted by a
    #       later rewrite -- caught by comparing against artifact_bytes;
    #   (b) the digest of the artefact WITH ITS OWN PROVENANCE BLOCK
    #       REMOVED -- the classic self-referential manifest, and the form
    #       a real producer would reach for.
    #
    # Both are computed HERE, from the artefact, and neither is supplied by
    # the caller. Checking only (a) would leave the constructible one open.
    if artifact_bytes is not None:
        forbidden = {sha256_hex(artifact_bytes): "the artefact's own bytes"}
        try:
            doc = json.loads(artifact_bytes.decode("utf-8"))
            if isinstance(doc, dict) and "producer_provenance" in doc:
                stripped = {k: v for k, v in doc.items()
                            if k != "producer_provenance"}
                forbidden[sha256_hex(_jcs(stripped))] = (
                    "the artefact with its own provenance block removed")
        except (ValueError, UnicodeDecodeError):
            pass                      # a non-JSON artefact still gets (a)

        # THE WHOLE OBJECT, NOT ITS TOP LEVEL. The previous scan read
        # `recorded.items()` and tested `isinstance(v, str)`, so a digest
        # inside input_binding, or inside any producer_population member,
        # satisfied a structural prohibition by being nested. D379 does not
        # say "top-level strings".
        hits = []

        def walk(node, path):
            if isinstance(node, str):
                if node in forbidden:
                    hits.append((path, forbidden[node]))
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{path}.{k}" if path else str(k))
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    walk(v, f"{path}[{i}]")

        walk(recorded, "")
        if hits:
            raise StageIdentityError(
                "REFUSE AS INVALID IDENTITY CONSTRUCTION: the in-band "
                "provenance declares a digest of its own output at "
                + "; ".join(f"{loc} (= {what})" for loc, what in hits)
                + ". D379 §4 -- no self-output digest, no fixed-point hash.")

    # ── every field with an authoritative comparator, mechanically ────
    cmps = _prov_comparators(descriptor)
    for field in sorted(expect_fields & set(cmps)):
        if recorded[field] != cmps[field]:
            raise StageIdentityError(
                f"REFUSE: {comp} provenance field {field!r} does not match "
                f"the Stage-A descriptor it claims. recorded="
                f"{str(recorded[field])[:80]!r} "
                f"stage_a={str(cmps[field])[:80]!r}")

    # ── producer_population, member by member ─────────────────────────
    stage_h2 = {m["path"]: m["sha256"] for m in descriptor["h2_sources"]}
    members = recorded.get("producer_population")
    if not isinstance(members, list) or not members:
        raise StageIdentityError(
            "REFUSE: provenance records no producer_population member")
    seen = set()
    for m in members:
        if not isinstance(m, dict) or set(m) != {"class", "identity", "sha256"}:
            raise StageIdentityError(f"REFUSE: malformed provenance member {m!r}")
        if m["class"] != CLASS_H2:
            continue
        if m["identity"] not in stage_h2:
            raise StageIdentityError(
                f"REFUSE: provenance names H2 member {m['identity']}, which "
                f"is NOT represented in Stage A. No silent runtime expansion.")
        if m["sha256"] != stage_h2[m["identity"]]:
            raise StageIdentityError(
                f"REFUSE: provenance H2 member {m['identity']} byte mismatch "
                f"against Stage A.")
        seen.add(m["identity"])

    if recorded["producer_denominator"] != len(members):
        raise StageIdentityError(
            f"REFUSE: producer_denominator {recorded['producer_denominator']} "
            f"does not equal the recorded producer_population size "
            f"{len(members)}. A denominator that does not count its own "
            f"population is not a denominator (R5).")

    unverified = set()
    if comp == "CLASSIFICATION":
        ib = recorded["input_binding"]
        if not isinstance(ib, dict) or set(ib) != set(INPUT_BINDING_FIELDS):
            raise StageIdentityError(
                f"REFUSE: classification input_binding does not carry the "
                f"D379 §4 field set {list(INPUT_BINDING_FIELDS)}; got "
                f"{sorted(ib) if isinstance(ib, dict) else type(ib).__name__}")
        if ib["pass_a_stage_a_identity"] != ident:
            raise StageIdentityError(
                f"REFUSE: input_binding.pass_a_stage_a_identity does not "
                f"match the Stage-A identity. Stale input across two "
                f"Stage As.")
        # THE EXACT BYTES, OR NAMED AS UNVERIFIED. These two fields are
        # derived from the Pass-A artefact, not from Stage A, so a caller
        # holding only the descriptor CANNOT establish them. Saying so is
        # the whole point: a silent skip here is indistinguishable from a
        # verified field.
        if pass_a_bytes is None:
            unverified |= {"input_binding.pass_a_artifact_sha256",
                           "input_binding.pass_a_producer_provenance_digest"}
        else:
            actual = sha256_hex(pass_a_bytes)
            if ib["pass_a_artifact_sha256"] != actual:
                raise StageIdentityError(
                    f"REFUSE: input_binding.pass_a_artifact_sha256 "
                    f"{ib['pass_a_artifact_sha256']} is not the digest of "
                    f"the exact Pass-A bytes consumed ({actual}).")
            pp = json.loads(pass_a_bytes.decode("utf-8")).get(
                "producer_provenance")
            if pp is None or ib["pass_a_producer_provenance_digest"] !=                     provenance_digest(pp):
                raise StageIdentityError(
                    "REFUSE: input_binding.pass_a_producer_provenance_digest "
                    "is not the canonical digest of the provenance block "
                    "inside the exact Pass-A bytes consumed.")

    return ident, seen, unverified


def verify_runtime_identity(descriptor):
    """D379 DEP-3. OBSERVE the executing runtime and compare it with the
    identity Stage A expects. REFUSE on mismatch.

    A producer that copies `descriptor["runtime"]` into its own provenance
    is attesting from the EXPECTED value: it says "my runtime is whatever
    Stage A says it should be", which cannot fail and therefore proves
    nothing. This observes, then compares, then returns the OBSERVED block
    for recording.
    """
    observed = build_runtime()                    # the existing authority
    expected = descriptor.get("runtime") or {}
    diff = [k for k in RUNTIME_FIELDS if observed.get(k) != expected.get(k)]
    if diff:
        raise StageIdentityError(
            f"REFUSE: DEP-3 executing runtime identity differs from the "
            f"identity Stage A expects, on {diff}. "
            f"observed stdlib_identity={observed.get('stdlib_identity')} "
            f"expected={expected.get('stdlib_identity')}")
    return observed


def reconcile_provenance(recorded_members, observed_members):
    """D379 §5: the emitted provenance population must EQUAL the canonical
    runtime-observed population. No provenance list defines its own
    completeness, so this compares against an INDEPENDENT observation."""
    r = {(m["class"], m["identity"]) for m in recorded_members}
    o = {(c, i) for c, i, _ in observed_members}
    if r != o:
        raise StageIdentityError(
            f"REFUSE: recorded provenance does not match the observed "
            f"producer population. observed-only={sorted(o - r)[:4]} "
            f"recorded-only={sorted(r - o)[:4]}")
    return True


def contains_self_digest(provenance, artifact_bytes):
    """Q1a-9 SELF-HASH PROHIBITION. An identity containing its own OUTPUT
    digest is the forbidden shape; the accepted path finalises the file
    first and binds it EXTERNALLY."""
    own = sha256_hex(artifact_bytes)
    return own in json.dumps(provenance)
