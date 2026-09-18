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
def producer_population(repo_root):
    """Governed modules ACTUALLY LOADED, classified by the closed rule.

    Directional, per D379 §5: governed runtime modules actually loaded
    MUST be represented in Stage A; NOT every Stage-A module must appear
    in every process. Returns (members, external_offenders).
    """
    root = pathlib.Path(repo_root).resolve()
    h2root = (root / "kai-pm" / "house_in_order_h2_v13").resolve()
    census = (root / "kai-pm" / "house_in_order_census_v11").resolve()
    roots, external = _governed_roots()
    members, offenders = [], []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue                                   # built-in / frozen
        rp = os.path.realpath(f)
        p = pathlib.Path(rp)
        if str(p).startswith(str(h2root) + os.sep):
            members.append(("H2", p.relative_to(root).as_posix(),
                            sha256_hex(p.read_bytes())))
        elif str(p).startswith(str(census) + os.sep):
            members.append(("CENSUS", p.relative_to(root).as_posix(),
                            sha256_hex(p.read_bytes())))
        elif _is_external(rp, external):
            offenders.append((name, rp))
        elif _owning_root(rp, roots) is not None:
            members.append(("STDLIB", name, ""))       # covered by runtime id
        else:
            offenders.append((name, rp))
    members.sort()
    return members, offenders


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
