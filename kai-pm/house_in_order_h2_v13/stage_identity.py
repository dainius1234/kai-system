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
    import errno as _errno
    import stat as _stat

    # v4.5 §22 / v4.1 F9 — FAIL CLOSED. Every enumeration error REFUSES.
    # The previous walk caught PermissionError/FileNotFoundError and
    # `continue`d, so an unreadable or disappearing governed member produced
    # a successful SMALLER identity: indistinguishable from "file removed".
    for rid in sorted(roots):
        root = roots[rid]
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    children = list(it)
            except OSError as exc:
                raise StageIdentityError(
                    f"REFUSE: governed stdlib directory under {rid} cannot be "
                    f"enumerated ({type(exc).__name__}: errno {exc.errno}). "
                    f"No successful smaller identity (v4.5 §22).") from None
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
                try:
                    st = os.lstat(full)
                except OSError as exc:
                    raise StageIdentityError(
                        f"REFUSE: governed stdlib member {rid}:{rel} disappeared "
                        f"or cannot be stat'ed during enumeration "
                        f"({type(exc).__name__}). No successful smaller "
                        f"identity (v4.5 §22).") from None
                if _stat.S_ISLNK(st.st_mode):
                    # v4.1 F10 / D380 §7.4: a LOOP is a cycle, not "dangling".
                    try:
                        tgt = os.path.realpath(full, strict=True)
                    except OSError as exc:
                        if exc.errno == _errno.ELOOP:
                            raise StageIdentityError(
                                f"REFUSE: symlink cycle at {rel} under {rid} "
                                f"(D380 §7.4: loops are detected as cycles)") \
                                from None
                        if exc.errno == _errno.ENOENT:
                            raise StageIdentityError(
                                f"REFUSE: dangling symlink {rel} under {rid}") \
                                from None
                        raise StageIdentityError(
                            f"REFUSE: symlink {rel} under {rid} cannot be "
                            f"resolved ({type(exc).__name__}: errno "
                            f"{exc.errno})") from None
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
                        with open(full, "rb") as fh:
                            data = fh.read()
                    except OSError as exc:
                        raise StageIdentityError(
                            f"REFUSE: governed stdlib member {rid}:{rel} is "
                            f"unreadable or disappeared ({type(exc).__name__}: "
                            f"errno {exc.errno}). No successful smaller "
                            f"identity (v4.5 §22, v4.1 F9).") from None
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
    with open(exe, "rb") as fh:
        exe_bytes = fh.read()
    return {
        "executable_sha256": sha256_hex(exe_bytes),
        "implementation_name": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": sys.version,
        "stdlib_identity": _stdlib_identity(),
        "dont_write_bytecode": bytecode_disabled_from_startup(),
    }


def bytecode_disabled_from_startup() -> bool:
    """v4.1 F7 / D380 §6.9: bytecode writing disabled FROM PROCESS STARTUP.

    `sys.flags.dont_write_bytecode` is fixed at interpreter initialisation
    (-B or PYTHONDONTWRITEBYTECODE) and cannot be set later; a LATE
    `sys.dont_write_bytecode = True` after governed imports began changes
    only the mutable attribute. Both must hold: the startup flag proves the
    state existed before any governed import, and the attribute proves it
    was not switched off afterwards.
    """
    return bool(sys.flags.dont_write_bytecode) and bool(sys.dont_write_bytecode)


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

    # ── v4.1 C1 — every mode: exact nested shapes, formats, ordering,
    #    cross-field consistency. CHECKED, never re-sorted or repaired.
    if [(m["path"], m["sha256"]) for m in src] != sorted(
            (m["path"], m["sha256"]) for m in src):
        raise StageIdentityError(
            "REFUSE: h2_sources is not in canonical (path, sha256) order. "
            "Order is checked, never re-sorted (D380 §6.2): an unsorted "
            "array would otherwise be a second identity for the same set.")
    for m in src:
        _fmt_hex64(m["sha256"], f"h2_sources[{m['path']}].sha256")
    _shape(desc["contract"], ("path", "sha256"), "contract")
    if desc["contract"]["path"] != CONTRACT_PATH:
        raise StageIdentityError(f"REFUSE: contract.path is not {CONTRACT_PATH}")
    _fmt_hex64(desc["contract"]["sha256"], "contract.sha256")
    sub = _shape(desc["subject"], ("commit", "tree", "population"), "subject")
    _fmt_oid(sub["commit"], "subject.commit")
    _fmt_oid(sub["tree"], "subject.tree")
    _fmt_count(sub["population"], "subject.population")
    tp = _shape(desc["tree_paths"], ("population", "tree_paths_identity"),
                "tree_paths")
    _fmt_count(tp["population"], "tree_paths.population")
    _fmt_hex64(tp["tree_paths_identity"], "tree_paths.tree_paths_identity")
    if tp["population"] != sub["population"]:
        raise StageIdentityError(
            "REFUSE: tree_paths.population != subject.population (cross-field)")
    ce = _shape(desc["census"], ("logical_package", "aggregate_sha256"), "census")
    if ce["logical_package"] != CENSUS_PACKAGE:
        raise StageIdentityError(
            f"REFUSE: census.logical_package is not {CENSUS_PACKAGE}")
    _fmt_hex64(ce["aggregate_sha256"], "census.aggregate_sha256")
    hi = _shape(desc["history"], HISTORY_FIELDS, "history")
    _fmt_oid(hi["subject_commit"], "history.subject_commit")
    if not isinstance(hi["is_shallow"], bool):
        raise StageIdentityError("REFUSE: history.is_shallow is not a boolean")
    _fmt_count(hi["reachable_count"], "history.reachable_count")
    _fmt_oid(hi["oldest_commit"], "history.oldest_commit")
    _fmt_date(hi["oldest_date"], "history.oldest_date")
    _fmt_hex64(hi["reachable_set_sha256"], "history.reachable_set_sha256")
    if hi["subject_commit"] != sub["commit"]:
        raise StageIdentityError(
            "REFUSE: history.subject_commit != subject.commit (cross-field)")
    _fmt_hex64(rt["executable_sha256"], "runtime.executable_sha256")
    _fmt_hex64(rt["stdlib_identity"], "runtime.stdlib_identity")
    for k in ("implementation_name", "cache_tag", "version"):
        if not isinstance(rt[k], str) or not rt[k]:
            raise StageIdentityError(
                f"REFUSE: runtime.{k} is not a non-empty string")

    # ── PRODUCTION additionally requires the frozen D380 values ──────────
    if mode == MODE_PRODUCTION:
        for path, want in PRODUCTION_FROZEN:
            node = desc
            for k in path:
                node = node[k]
            if node != want:
                raise StageIdentityError(
                    f"REFUSE: PRODUCTION {'.'.join(path)} = {node!r} is not "
                    f"the frozen D380 value {want!r}")
    return True


# D380 §§6.3, 6.5–6.8 frozen PRODUCTION values (implementation of banked
# authority, v4.5 §31). reachable_set_sha256 is NOT frozen by D380 and is
# established only by re-derivation (build_stage_a).
HISTORY_FIELDS = ("subject_commit", "is_shallow", "reachable_count",
                  "oldest_commit", "oldest_date", "reachable_set_sha256")
FROZEN_SUBJECT_COMMIT = "d8aac4d49e6ba997e3eb38062c0917186ee3f197"
PRODUCTION_FROZEN = (
    (("contract", "sha256"),
     "0ce5792ed72e6e7051ecc050664490899a847d01de2f62cff564f460d46800bb"),
    (("subject", "commit"), FROZEN_SUBJECT_COMMIT),
    (("subject", "tree"), "3abc9e9d8ca11966a6f996d5f0af68072ee5b117"),
    (("subject", "population"), 272),
    (("tree_paths", "tree_paths_identity"),
     "3af69867813d336f014931e2cfbad8fa4dae14df66cf8df25a5efb1ec2110b40"),
    (("census", "aggregate_sha256"),
     "29064d650a61296806df3c3bcab3322f7364da7df674ac93e79d0671475d757a"),
    (("history", "subject_commit"), FROZEN_SUBJECT_COMMIT),
    (("history", "is_shallow"), False),
    (("history", "reachable_count"), 986),
    (("history", "oldest_commit"), "e8c3209c0131e1401f6002ab984c0728a40424e4"),
    (("history", "oldest_date"), "2025-06-18"),
)


def _shape(node, fields, where):
    if not isinstance(node, dict) or set(node) != set(fields):
        raise StageIdentityError(
            f"REFUSE: {where} does not have exactly the fields {list(fields)}; "
            f"got {sorted(node) if isinstance(node, dict) else type(node).__name__}")
    return node


def _fmt_hex64(v, where):
    import re as _re
    if not isinstance(v, str) or not _re.fullmatch(r"[0-9a-f]{64}", v):
        raise StageIdentityError(
            f"REFUSE: {where} is not a lower-case 64-hex digest")


def _fmt_oid(v, where):
    import re as _re
    if not isinstance(v, str) or not _re.fullmatch(r"[0-9a-f]{40}", v):
        raise StageIdentityError(
            f"REFUSE: {where} is not a lower-case 40-hex Git object id")


def _fmt_count(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise StageIdentityError(
            f"REFUSE: {where} is not a non-negative non-bool integer")


def _fmt_date(v, where):
    import datetime as _dt
    import re as _re
    if not isinstance(v, str) or not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        raise StageIdentityError(f"REFUSE: {where} is not an ISO YYYY-MM-DD date")
    try:
        _dt.date.fromisoformat(v)
    except ValueError:
        raise StageIdentityError(
            f"REFUSE: {where} is not a valid calendar date") from None


def parse_descriptor_bytes(data: bytes):
    """Parse a MATERIALISED descriptor strictly (v4.1 C1, D380 §6.10).

    Duplicate JSON keys REFUSE (json.loads silently keeps the last one);
    floats and NaN REFUSE; and the bytes must be EXACTLY the canonical D —
    no added newline, no pretty printing, no reordering.
    """
    def pairs(kv):
        keys = [k for k, _ in kv]
        if len(set(keys)) != len(keys):
            raise StageIdentityError(
                f"REFUSE: duplicate JSON key in descriptor {keys}")
        return dict(kv)

    def no_float(s):
        raise StageIdentityError(f"REFUSE: float or constant {s!r} in descriptor")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise StageIdentityError("REFUSE: descriptor is not valid UTF-8") from None
    try:
        desc = json.loads(text, object_pairs_hook=pairs, parse_float=no_float,
                          parse_constant=no_float)
    except ValueError as exc:
        raise StageIdentityError(f"REFUSE: descriptor is not JSON ({exc})") from None
    if canonical_bytes(desc) != data:
        raise StageIdentityError(
            "REFUSE: the materialised descriptor is not exactly its canonical "
            "bytes D (D380 §6.10)")
    return desc


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


# ── v4.5 §8.1 — the INSTRUMENT root, derived, never caller-supplied ───
H2_DIR = "kai-pm/house_in_order_h2_v13"
SELF_MEMBER = H2_DIR + "/stage_identity.py"
CONTRACT_PATH = "kai-pm/H2_REPAIR_CONTRACT_D367.md"
CONTRACT_SHA256 = ("0ce5792ed72e6e7051ecc050664490899a847d01de2f62cff5"
                   "64f460d46800bb")       # implementation of banked D380 §6.3 (v4.5 §31)


def instrument_root() -> pathlib.Path:
    """The repository root of the H2 implementation ACTUALLY EXECUTING.

    v4.5 §8.1: taken from the loaded stage_identity.py filesystem source,
    resolved to its real path (a symlinked source resolves first), whose
    location must correspond to the governed Stage-A member. No argv[0], no
    environment variable, no --subject-repo and no CWD defines it.
    """
    f = globals().get("__file__")
    if not f or not os.path.isfile(f):
        raise StageIdentityError(
            "REFUSE: stage_identity is not loaded from a filesystem-backed "
            "source; the instrument root cannot be derived (v4.5 §8.1)")
    rp = pathlib.Path(os.path.realpath(f))
    parts = tuple(SELF_MEMBER.split("/"))
    if rp.parts[-len(parts):] != parts:
        raise StageIdentityError(
            f"REFUSE: the loaded stage_identity source {rp} does not sit at "
            f"the governed member location {SELF_MEMBER} (v4.5 §8.1)")
    return rp.parents[len(parts) - 1]


def _read_regular_once(path) -> bytes:
    """lstat, require a REGULAR file (never a symlink, FIFO, device or
    directory), then read it ONCE without following a final symlink."""
    p = os.fspath(path)
    try:
        st = os.lstat(p)
    except OSError as exc:
        raise StageIdentityError(
            f"REFUSE: {p} cannot be stat'ed ({type(exc).__name__})") from None
    import stat as _stat
    if _stat.S_ISLNK(st.st_mode):
        raise StageIdentityError(f"REFUSE: {p} is a symlink, not a regular file")
    if not _stat.S_ISREG(st.st_mode):
        raise StageIdentityError(f"REFUSE: {p} is not a regular file")
    try:
        fd = os.open(p, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise StageIdentityError(
            f"REFUSE: {p} cannot be opened ({type(exc).__name__})") from None
    with os.fdopen(fd, "rb") as fh:
        return fh.read()


# ── v4.5 §10 — Census EXACT-BYTE execution, closed ordering ───────────
CENSUS_PACKAGE = "house_in_order_census_v11"
CENSUS_MANIFEST = "MANIFEST.sha256"
CENSUS_EXECUTION_ROOTS = ("docgraph", "opscan", "claims")
_CENSUS = None          # the verified-byte registry of THIS process, or None


def _parse_census_manifest(man_bytes):
    import re as _re
    members = {}
    for ln in man_bytes.decode("utf-8").splitlines():
        if not ln.strip():
            continue
        m = _re.fullmatch(r"([0-9a-f]{64}) [ *]?(\S.*)", ln)
        if not m:
            raise StageIdentityError(f"REFUSE: malformed Census manifest line {ln!r}")
        name = _norm_path(m.group(2))
        if name in members:
            raise StageIdentityError(f"REFUSE: duplicate Census manifest member {name}")
        members[name] = m.group(1)
    return members


def _census_imports(src_bytes, member):
    import ast as _ast
    names = set()
    for node in _ast.walk(_ast.parse(src_bytes, filename=member)):
        if isinstance(node, _ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, _ast.ImportFrom):
            if node.level:
                raise StageIdentityError(
                    f"REFUSE: relative import in governed Census member "
                    f"{member}; its execution closure is not statically "
                    f"resolvable (v4.5 §10.2)")
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def load_governed_census(package_dir, roots=CENSUS_EXECUTION_ROOTS, *,
                         expected_aggregate=None):
    """Install the governed Census execution closure from VERIFIED BYTES.

    Order (v4.5 §10, reconciliation §9): refuse a preloaded governed module
    (S19); read the manifest once; derive the local import closure from the
    verified bytes, refusing an unaccounted local dependency; read each
    member ONCE and compare it with the manifest; pre-register every module
    object in sys.modules BEFORE executing; compile and execute THOSE SAME
    bytes; on any failure remove every governed entry installed here, then
    refuse. Returns the registry; Pass A uses its module objects directly.
    """
    global _CENSUS
    pkg = pathlib.Path(package_dir)
    pre = sorted(n for n in roots if n in sys.modules)
    if pre:
        raise StageIdentityError(
            f"REFUSE (S19): governed Census module(s) {pre} already present in "
            f"sys.modules before the verified-byte loader established its "
            f"boundary. Not silently replaced (v4.5 §10.1).")
    man_bytes = _read_regular_once(pkg / CENSUS_MANIFEST)
    aggregate = sha256_hex(man_bytes)
    if expected_aggregate is not None and aggregate != expected_aggregate:
        raise StageIdentityError(
            f"REFUSE: Census manifest aggregate {aggregate} != governed "
            f"{expected_aggregate}")
    members = _parse_census_manifest(man_bytes)
    py = {n[:-3]: n for n in members if n.endswith(".py") and "/" not in n}

    verified, order, queue, seen = {}, [], list(roots), set()
    while queue:
        mod = queue.pop(0)
        if mod in seen:
            continue
        seen.add(mod)
        if mod not in py:
            raise StageIdentityError(
                f"REFUSE: Census execution module {mod!r} is not a manifest "
                f"member (v4.5 §10.2)")
        b = _read_regular_once(pkg / py[mod])
        if sha256_hex(b) != members[py[mod]]:
            raise StageIdentityError(
                f"REFUSE: Census member {py[mod]} bytes do not match the "
                f"governed manifest")
        verified[mod] = b
        for dep in sorted(_census_imports(b, py[mod])):
            if dep in py:
                queue.append(dep)
            elif (pkg / f"{dep}.py").exists() or (pkg / dep).is_dir():
                raise StageIdentityError(
                    f"REFUSE: governed Census member {py[mod]} requires the "
                    f"unaccounted local Census module {dep!r}; the execution "
                    f"closure changed and it is not manifest-verified "
                    f"(v4.5 §10.6 dependency expansion)")
    pre = sorted(n for n in verified if n in sys.modules)
    if pre:
        raise StageIdentityError(
            f"REFUSE (S19): governed Census module(s) {pre} already present in "
            f"sys.modules before the verified-byte loader (v4.5 §10.1)")

    # dependency order: a member's local Census imports execute first
    deps = {m: sorted(d for d in _census_imports(verified[m], py[m]) if d in py)
            for m in verified}
    state = {}

    def visit(m):
        if state.get(m) == 2:
            return
        if state.get(m) == 1:
            return                      # a cycle: pre-registration handles it
        state[m] = 1
        for d in deps[m]:
            visit(d)
        state[m] = 2
        order.append(m)
    for m in sorted(verified):
        visit(m)

    import importlib.machinery as _mach
    import types as _types
    installed, registry = [], {}
    try:
        for m in order:
            rp = os.path.realpath(pkg / py[m])
            mod = _types.ModuleType(m)
            spec = _mach.ModuleSpec(m, None, origin=rp)
            spec.has_location = True
            mod.__spec__, mod.__file__, mod.__loader__ = spec, rp, None
            sys.modules[m] = mod                   # registered BEFORE execution
            installed.append(m)
        for m in order:
            code = compile(verified[m], os.path.realpath(pkg / py[m]), "exec")
            exec(code, sys.modules[m].__dict__)   # noqa: S102 — verified bytes
            registry[m] = {"member": py[m], "sha256": sha256_hex(verified[m]),
                           "realpath": os.path.realpath(pkg / py[m]),
                           "module": sys.modules[m]}
    except BaseException as exc:
        for m in installed:
            sys.modules.pop(m, None)                # reconciliation §9
        if isinstance(exc, StageIdentityError):
            raise
        raise StageIdentityError(
            f"REFUSE: governed Census execution failed "
            f"({type(exc).__name__}: {exc}); partially initialised governed "
            f"modules removed from sys.modules") from None
    _CENSUS = {"package": CENSUS_PACKAGE, "aggregate": aggregate,
               "manifest_bytes": man_bytes, "members": members,
               "realdir": os.path.realpath(pkg), "modules": registry}
    return _CENSUS


def census_assert_installed():
    """v4.5 §10.5: an ordinary import must resolve to the installed object."""
    import importlib as _il
    if _CENSUS is None:
        raise StageIdentityError("REFUSE: no verified Census registry in this process")
    for m, rec in _CENSUS["modules"].items():
        if _il.import_module(m) is not rec["module"] or sys.modules.get(m) is not rec["module"]:
            raise StageIdentityError(
                f"REFUSE (S18): {m} no longer resolves to the verified-byte "
                f"module object")
    return True


def classify_origin(o, *, instrument=None, census=None, roots=None,
                    external=None, snapshot=None):
    """Classify ONE origin into the closed set, or REFUSE. D379 §5/§6.

    v4.5 §11: H2 authority comes ONLY from the derived instrument root
    (§8.1) and Census authority ONLY from the verified-byte registry (§10).
    There is deliberately no repo_root parameter: a subject or history
    repository can never supply H2 or Census code authority, so a
    same-named module inside one is never an instrument module.
    Returns (class, identity). Raises StageIdentityError to REFUSE.
    """
    if roots is None or external is None:
        roots, external = _governed_roots()
    inst = pathlib.Path(instrument) if instrument is not None else instrument_root()
    h2root = str(inst / H2_DIR)
    census = census if census is not None else _CENSUS

    # 1. IMPORT METADATA FIRST, and it OUTRANKS __file__ (Kai §4).
    if o.spec_origin == "built-in":
        return CLASS_BUILTIN, o.name
    if o.spec_origin == "frozen":
        return CLASS_FROZEN, o.name

    # 2. __main__ is handled EXPLICITLY, as D379 §5 requires in terms.
    if o.is_main and not o.file and not _is_fs_origin(o.spec_origin):
        raise StageIdentityError(
            f"REFUSE: the executing entry point __main__ has no mechanically "
            f"establishable source (spec_origin={o.spec_origin!r}, "
            f"file={o.file!r}). D379 §5 requires the entry-point source to be "
            f"included EXPLICITLY; an unsourced __main__ is not built-in and "
            f"is not frozen.")

    # 3. A filesystem origin; spec.origin and __file__ must agree.
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
        raise StageIdentityError(
            f"REFUSE: {o.name} has no filesystem source and its origin is "
            f"neither 'built-in' nor 'frozen' (has_spec={o.has_spec}, "
            f"spec_origin={o.spec_origin!r}). A missing __file__ is NOT an "
            f"answer (INC-2026-09-19-37).")
    rp = cands[0]

    if rp.startswith(h2root + os.sep):
        rel = pathlib.Path(rp).relative_to(inst).as_posix()
        if rel not in H2_SOURCES:
            raise StageIdentityError(
                f"REFUSE: {o.name} at {rel} lies in the instrument H2 "
                f"directory but is not one of the ten governed H2_SOURCES "
                f"(v4.5 §8.1 step 7)")
        return CLASS_H2, rel
    if census is not None:
        rec = census["modules"].get(o.name)
        if rec is not None and rec["realpath"] == rp:
            return CLASS_CENSUS, f"{census['package']}/{rec['member']}"
        if rp.startswith(census["realdir"] + os.sep):
            raise StageIdentityError(
                f"REFUSE: {o.name} at {rp} is a Census-package file that was "
                f"NOT installed by the verified-byte loader (v4.5 §10.5)")
    if os.sep + CENSUS_PACKAGE + os.sep in rp:
        raise StageIdentityError(
            f"REFUSE: {o.name} at {rp} is a Census module loaded outside the "
            f"verified-byte loader; Census authority is its verified content, "
            f"never its pathname (v4.5 §§8.4, 11.2)")
    if _is_external(rp, external):                 # D380 §7.3
        raise StageIdentityError(
            f"REFUSE: {o.name} is a loaded non-stdlib EXTERNAL module at "
            f"{rp}, outside every governed Stage-A root, with no explicit "
            f"Stage-A dependency identity. None is invented here (D379 §6).")
    own = _owning_root(rp, roots)
    if own is not None:
        rid, root = own
        rel = _norm_path(os.path.relpath(rp, root).replace(os.sep, "/"))
        # D380 §7.11 / v4.1 F8: a filesystem-backed stdlib module must be
        # REPRESENTED in the governed snapshot (a sourceless .pyc is not).
        if snapshot is not None and (rid, rel) not in snapshot:
            raise StageIdentityError(
                f"REFUSE: loaded stdlib module {o.name} ({rid}:{rel}) is NOT "
                f"represented in the governed H2_PY_STDLIB_V1 snapshot "
                f"(D380 §7.11)")
        return CLASS_STDLIB, f"{rid}:{rel}"
    raise StageIdentityError(
        f"REFUSE: {o.name} at {rp} lies outside the governed H2 root, the "
        f"governed Census registry and the governed Python stdlib "
        f"classification.")


def _is_fs_origin(origin):
    """A spec origin that names a real filesystem location."""
    return bool(origin) and origin not in ("built-in", "frozen") \
        and os.path.exists(origin)


def _snapshot_files():
    _ident, obj, _S = build_stdlib_identity()
    return {(e["root_id"], e["path"]) for e in obj["entries"] if e["type"] == "file"}


def producer_population():
    """Every observed import origin, classified or REFUSED. D379 §5.

    Returns (members, offenders); members carry (class, identity, sha256).
    H2 digests are the exact bytes at the derived instrument root; CENSUS
    digests are the bytes the verified-byte loader EXECUTED (not a later
    read of the file, which a byte-mutation attack could have changed);
    STDLIB/BUILTIN/FROZEN digests are empty by rule. NO SILENT MEMBER.
    """
    inst = instrument_root()
    roots, external = _governed_roots()
    snapshot = _snapshot_files()
    members, offenders, seen = [], [], set()

    def take(o):
        try:
            cls, ident = classify_origin(o, instrument=inst, roots=roots,
                                         external=external, snapshot=snapshot)
        except StageIdentityError as e:
            offenders.append((o.name, str(e)))
            return
        key = (cls, ident)
        if key in seen:                       # D379 §5: deduplicate
            return
        seen.add(key)
        digest = ""
        if cls == CLASS_H2:
            digest = sha256_hex(_read_regular_once(inst / ident))
        elif cls == CLASS_CENSUS:
            digest = _CENSUS["modules"][o.name]["sha256"]
        members.append((cls, ident, digest))

    main = sys.modules.get("__main__")
    if main is not None:
        take(observe_origin("__main__", main))
    for name, mod in sorted(sys.modules.items()):
        if name == "__main__" or mod is None:
            continue
        take(observe_origin(name, mod))
    members.sort()
    return members, offenders


# ── v4.5 §9 — COMPLETE STAGE-A CONSTRUCTION, every field re-derived ───
def _git_out(repo, *args) -> bytes:
    import subprocess as _sp
    p = _sp.run(["git", "-C", os.fspath(repo), *args], capture_output=True)
    if p.returncode:
        raise StageIdentityError(
            f"REFUSE: git {' '.join(args[:3])} failed in a Stage-A context "
            f"(rc={p.returncode}): {p.stderr.decode(errors='replace')[:160]}")
    return p.stdout


def _path_c1(raw: bytes) -> str:
    """v4.1 C1 path rule, in its normative order: valid UTF-8; ALREADY NFC
    (never normalised into acceptance); D380 §6.10 structure. The duplicate
    check is the caller's, on the exact validated strings."""
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise StageIdentityError(f"REFUSE: path is not valid UTF-8: {raw!r}") from None
    return _norm_path(s)


def derive_tree_paths(subject_repo, tree):
    """D380 §6.6 from the exact Git tree: (population, identity, paths)."""
    raw = _git_out(subject_repo, "ls-tree", "-r", "-z", "--name-only", tree)
    paths = [_path_c1(p) for p in raw.split(b"\x00") if p]
    md = [p for p in paths if p.endswith(".md")]
    if len(set(md)) != len(md):
        raise StageIdentityError("REFUSE: duplicate tracked .md path in the subject tree")
    md.sort(key=lambda p: p.encode("utf-8"))
    data = "".join(p + "\n" for p in md).encode("utf-8")
    return len(md), sha256_hex(data), md


def derive_history(history_repo, subject_commit):
    """D380 §6.8 from the actual history source. The oldest_* derivation is
    the one that produced D380's frozen values (kai-pm/
    validity_binding_audit.py: git log --reverse --format='%H %ad'
    --date=short), applied at the subject commit."""
    shallow = _git_out(history_repo, "rev-parse",
                       "--is-shallow-repository").decode().strip()
    if shallow not in ("true", "false"):
        raise StageIdentityError(f"REFUSE: cannot determine shallowness ({shallow!r})")
    oids = sorted(set(_git_out(history_repo, "rev-list", subject_commit)
                      .decode().split()))
    first = _git_out(history_repo, "log", "--reverse", "--format=%H %ad",
                     "--date=short", subject_commit).decode().splitlines()[0].split()
    return {"subject_commit": subject_commit,
            "is_shallow": shallow == "true",
            "reachable_count": len(oids),
            "oldest_commit": first[0], "oldest_date": first[1],
            "reachable_set_sha256": sha256_hex("".join(o + "\n" for o in oids)
                                               .encode("utf-8"))}


def build_stage_a(mode, *, subject_repo, history_repo):
    """Construct the FULL H2_STAGE_A_V2 descriptor from actual contexts.

    v4.5 §9: schema and mode derived; h2_sources from the exact bytes at the
    DERIVED instrument root (§8.1); contract bytes hashed and, in
    PRODUCTION, compared with banked D380 §6.3; governance the closed V2
    set; subject, tree_paths and history from the actual Git objects;
    census from the verified-byte objects ACTUALLY EXECUTED by
    load_governed_census (§9.8, so it must have run first); runtime by
    build_runtime() against this process. No caller-supplied value enters.
    The stdlib digest is still taken ONLY inside build_runtime(): there is
    deliberately no parameter through which a digest could be supplied.
    """
    if mode not in MODES:
        raise StageIdentityError(f"REFUSE: unknown mode {mode!r}")
    inst = instrument_root()
    srcs = sorted(({"path": p, "sha256": sha256_hex(_read_regular_once(inst / p))}
                   for p in H2_SOURCES), key=lambda m: (m["path"], m["sha256"]))
    contract = {"path": CONTRACT_PATH,
                "sha256": sha256_hex(_read_regular_once(inst / CONTRACT_PATH))}
    commit = _git_out(subject_repo, "rev-parse", "HEAD^{commit}").decode().strip()
    tree = _git_out(subject_repo, "rev-parse", commit + "^{tree}").decode().strip()
    pop, tp_ident, _paths = derive_tree_paths(subject_repo, tree)
    if _CENSUS is None:
        raise StageIdentityError(
            "REFUSE: Stage-A census must come from the verified Census bytes "
            "actually executed (v4.5 §9.8); load_governed_census has not run")
    desc = {
        "schema": SCHEMA_V2, "mode": mode, "h2_sources": srcs,
        "contract": contract,
        "governance": [{"decision_id": d, "bank_commit_sha": c}
                       for d, c in GOVERNANCE_V2],
        "subject": {"commit": commit, "tree": tree, "population": pop},
        "tree_paths": {"population": pop, "tree_paths_identity": tp_ident},
        "census": {"logical_package": CENSUS_PACKAGE,
                   "aggregate_sha256": _CENSUS["aggregate"]},
        "history": derive_history(history_repo, commit),
        "runtime": build_runtime(),
    }
    validate_descriptor(desc)
    return desc


def require_supplied_equals_rederived(supplied, rederived):
    """v4.5 §9.11 / S16, reconciliation §10: the SUPPLIED descriptor is a
    commitment, not its own authority. Its canonical bytes must equal the
    canonical bytes of the INDEPENDENTLY rederived descriptor, and then the
    identities must be equal. REFUSE BEFORE OUTPUT, naming the first field."""
    S, R = canonical_bytes(supplied), canonical_bytes(rederived)
    if S != R:
        diff = [k for k in TOP_LEVEL_FIELDS if supplied.get(k) != rederived.get(k)]
        raise StageIdentityError(
            f"REFUSE (S16): supplied Stage A != independently rederived Stage A "
            f"on {diff}")
    if stage_a_identity(supplied) != stage_a_identity(rederived):
        raise StageIdentityError("REFUSE (S16): Stage-A identities differ")
    return stage_a_identity(rederived)


def check_population(descriptor, when="production"):
    """Observe this process's population and verify it against Stage A.

    ONE AUTHORITY, BOTH PRODUCERS (INC-38). H2 members against the Stage-A
    h2_sources; CENSUS members against the governed manifest bound by the
    Stage-A census aggregate. Returns the observed members or REFUSES.
    """
    members, offenders = producer_population()
    if offenders:
        raise StageIdentityError(
            f"REFUSE ({when}): the producer runtime population contains "
            f"origins outside every governed Stage-A root, with no Stage-A "
            f"dependency identity: "
            + "; ".join(f"{n}: {str(w)[:120]}" for n, w in offenders[:4]))
    stage_h2 = {m["path"]: m["sha256"] for m in descriptor["h2_sources"]}
    for cls, identity, digest in members:
        if cls == CLASS_H2:
            if identity not in stage_h2:
                raise StageIdentityError(
                    f"REFUSE ({when}): loaded H2 source {identity} is NOT "
                    f"represented in Stage A. No silent runtime expansion.")
            if digest != stage_h2[identity]:
                raise StageIdentityError(
                    f"REFUSE ({when}): loaded H2 source {identity} byte mismatch "
                    f"against Stage A: {digest} != {stage_h2[identity]}")
        elif cls == CLASS_CENSUS:
            if _CENSUS is None or _CENSUS["aggregate"] != \
                    descriptor["census"]["aggregate_sha256"]:
                raise StageIdentityError(
                    f"REFUSE ({when}): loaded Census member {identity} is not "
                    f"bound to the Stage-A Census aggregate")
            member = identity.split("/", 1)[1]
            if _CENSUS["members"].get(member) != digest:
                raise StageIdentityError(
                    f"REFUSE ({when}): executed Census member {identity} "
                    f"digest is not the governed manifest digest")
    return members


# ── Stage-B: EXTERNAL binding on FINAL bytes (D379 §4, v4.5 §13) ──────
STAGE_B_SCHEMA = "H2_STAGE_B_BINDING_V1"
ARTIFACT_KIND = {"PASS_A": "PASS_A_RESULT",            # BANKED producer/output role
                 "CLASSIFICATION": "CLASSIFICATION_RESULT"}
BINDING_FIELDS = ("schema", "artifact_path", "artifact_sha256", "artifact_kind",
                  "stage_a_identity", "producer_component",
                  "producer_provenance_digest", "producer_exit_status")


def strict_json(data: bytes, what: str):
    """Parse JSON bytes refusing duplicate keys, floats and non-UTF-8."""
    def pairs(kv):
        keys = [k for k, _ in kv]
        if len(set(keys)) != len(keys):
            raise StageIdentityError(f"REFUSE: duplicate JSON key in {what}")
        return dict(kv)

    def no_float(s):
        raise StageIdentityError(f"REFUSE: float or constant {s!r} in {what}")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                          parse_float=no_float, parse_constant=no_float)
    except (UnicodeDecodeError, ValueError) as exc:
        raise StageIdentityError(f"REFUSE: {what} is not UTF-8 JSON ({exc})") from None


def bind_artifact(artifact_path, *, producer_component, stage_a_desc,
                  producer_exit_status):
    """THE BINDER (v4.5 §13.3, v4.1 C3-i): a separate governed invocation,
    AFTER the producer exited. Refuses to bind a producer that exited
    non-zero. Reads the final regular file ONCE, hashes those bytes, parses
    THE SAME bytes for the provenance block. The producer supplies no
    expected digest of any kind. Returns (binding, canonical binding bytes).
    """
    if producer_component not in ARTIFACT_KIND:
        raise StageIdentityError(f"REFUSE: unknown producer component {producer_component!r}")
    if producer_exit_status != 0:
        raise StageIdentityError(
            f"REFUSE: the binder will not bind an artefact whose producer "
            f"exited {producer_exit_status} (v4.1 C3-i)")
    data = _read_regular_once(artifact_path)
    doc = strict_json(data, "artefact")
    prov = doc.get("producer_provenance") if isinstance(doc, dict) else None
    if not isinstance(prov, dict) or prov.get("producer_component") != producer_component:
        raise StageIdentityError(
            f"REFUSE: artefact carries no {producer_component} producer_provenance")
    ident = stage_a_identity(stage_a_desc)
    if prov.get("stage_a_identity") != ident:
        raise StageIdentityError("REFUSE: artefact provenance names a different Stage A")
    binding = {"schema": STAGE_B_SCHEMA,
               "artifact_path": os.path.basename(os.fspath(artifact_path)),
               "artifact_sha256": sha256_hex(data),
               "artifact_kind": ARTIFACT_KIND[producer_component],
               "stage_a_identity": ident,
               "producer_component": producer_component,
               "producer_provenance_digest": provenance_digest(prov),
               "producer_exit_status": 0}
    return binding, _jcs(binding)


def load_binding(binding_path, expected_binding_sha256, *, stage_a_desc,
                 producer_component):
    """Read ONE Stage-B binding (regular file, once), require its digest to
    equal the PARENT-HELD anchor, and validate it as a canonical binding for
    this Stage A and producer role. Used alone by a consumer that holds a
    REFERENCED binding but not its artefact (the qualifier, v4.5 §15.4)."""
    _fmt_hex64(expected_binding_sha256, "expected binding sha256 (anchor)")
    bb = _read_regular_once(binding_path)
    if sha256_hex(bb) != expected_binding_sha256:
        raise StageIdentityError(
            "REFUSE: Stage-B binding does not match the independently held "
            "anchor (coordinated rewrite or substituted binding)")
    binding = strict_json(bb, "Stage-B binding")
    if not isinstance(binding, dict) or set(binding) != set(BINDING_FIELDS) \
            or binding["schema"] != STAGE_B_SCHEMA or _jcs(binding) != bb:
        raise StageIdentityError("REFUSE: Stage-B binding is not a canonical "
                                 f"{STAGE_B_SCHEMA} object")
    if binding["producer_component"] != producer_component or \
            binding["artifact_kind"] != ARTIFACT_KIND.get(producer_component):
        raise StageIdentityError("REFUSE: Stage-B binding is for a different producer role")
    if binding["stage_a_identity"] != stage_a_identity(stage_a_desc):
        raise StageIdentityError("REFUSE: Stage-B binding names a different Stage A")
    if binding["producer_exit_status"] != 0:
        raise StageIdentityError("REFUSE: Stage-B binding records a failed producer")
    return binding


def consume_bound_artifact(artifact_path, binding_path, expected_binding_sha256,
                           *, stage_a_desc, producer_component):
    """THE CONSUMER (v4.5 §§13.4, 17; v4.1 C3). Refuses a symlink or
    non-regular artefact or binding; reads each ONCE; the binding's digest
    must equal the PARENT-HELD anchor, which never comes from the artefact,
    the binding file or the artefact's directory (reconciliation §17); the
    artefact digest, kind, component, Stage A and provenance digest must
    all match the ORIGINAL binding. Returns (artifact_bytes, doc, binding).
    """
    binding = load_binding(binding_path, expected_binding_sha256,
                           stage_a_desc=stage_a_desc,
                           producer_component=producer_component)
    data = _read_regular_once(artifact_path)
    if sha256_hex(data) != binding["artifact_sha256"]:
        raise StageIdentityError(
            "REFUSE: artefact bytes do not match the ORIGINAL Stage-B binding")
    doc = strict_json(data, "artefact")
    prov = doc.get("producer_provenance") if isinstance(doc, dict) else None
    if not isinstance(prov, dict) or \
            provenance_digest(prov) != binding["producer_provenance_digest"]:
        raise StageIdentityError(
            "REFUSE: artefact provenance digest does not match the ORIGINAL "
            "Stage-B binding (Q1a-7)")
    return data, doc, binding


def stage_b_aggregate(bindings):
    """Canonical ordered aggregate over EXTERNAL bindings only."""
    rows = sorted(bindings, key=lambda b: (b["artifact_path"],
                                           b["artifact_sha256"]))
    return sha256_hex(_jcs({"schema": "H2_STAGE_B_V1", "bindings": rows}))




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

# ── v4.5 §16 / v4.1 C4 — the provenance FORMAT schema, and from it,
#    MECHANICALLY, the digest-bearing slot population ──────────────────
F_HEX64, F_OID, F_TEXT, F_COUNT, F_BOOL, F_ROLE = (
    "HEX64", "OID", "TEXT", "COUNT", "BOOL", "ROLE")
PROV_FORMAT = {
    "stage_a_identity": F_HEX64, "stage_a_descriptor_digest": F_HEX64,
    "producer_component": F_ROLE, "producer_denominator": F_COUNT,
    "subject_commit": F_OID, "subject_tree": F_OID,
    "tree_paths_identity": F_HEX64, "census_identity": F_HEX64,
    "history_source_identity": F_HEX64,
    "runtime_identity.executable_sha256": F_HEX64,
    "runtime_identity.implementation_name": F_TEXT,
    "runtime_identity.cache_tag": F_TEXT, "runtime_identity.version": F_TEXT,
    "runtime_identity.stdlib_identity": F_HEX64,
    "runtime_identity.dont_write_bytecode": F_BOOL,
    "input_binding.pass_a_artifact_sha256": F_HEX64,
    "input_binding.pass_a_stage_a_identity": F_HEX64,
    "input_binding.pass_a_producer_provenance_digest": F_HEX64,
}
for _c in ORIGIN_CLASSES:
    PROV_FORMAT[f"producer_population[{_c}].sha256"] = F_HEX64
    PROV_FORMAT[f"producer_population[{_c}].identity"] = F_TEXT

AUTH_PRECOMMITTED = "PRECOMMITTED_STAGE_A"
AUTH_ARTIFACT = "EXACT_ARTIFACT_BYTES"
AUTH_EMPTY = "EMPTY_BY_RULE"
AUTHORITIES = (AUTH_PRECOMMITTED, AUTH_ARTIFACT, AUTH_EMPTY)
SLOT_AUTHORITY = {
    "stage_a_identity": AUTH_PRECOMMITTED,
    "stage_a_descriptor_digest": AUTH_PRECOMMITTED,
    "subject_commit": AUTH_PRECOMMITTED, "subject_tree": AUTH_PRECOMMITTED,
    "tree_paths_identity": AUTH_PRECOMMITTED,
    "census_identity": AUTH_PRECOMMITTED,
    "history_source_identity": AUTH_PRECOMMITTED,
    "runtime_identity.executable_sha256": AUTH_PRECOMMITTED,
    "runtime_identity.stdlib_identity": AUTH_PRECOMMITTED,
    "producer_population[H2].sha256": AUTH_PRECOMMITTED,
    "producer_population[CENSUS].sha256": AUTH_PRECOMMITTED,
    "producer_population[STDLIB].sha256": AUTH_EMPTY,
    "producer_population[BUILTIN].sha256": AUTH_EMPTY,
    "producer_population[FROZEN].sha256": AUTH_EMPTY,
    "input_binding.pass_a_artifact_sha256": AUTH_ARTIFACT,
    "input_binding.pass_a_stage_a_identity": AUTH_PRECOMMITTED,
    "input_binding.pass_a_producer_provenance_digest": AUTH_ARTIFACT,
}


def digest_slots(component=None):
    """Digest-bearing slots, derived from PROV_SHAPE x PROV_FORMAT. A slot
    is digest-bearing iff its governed format is HEX64 or OID."""
    fields = set(PROV_SHAPE[component]) if component else \
        set().union(*PROV_SHAPE.values())
    out = []
    for slot, fmt in PROV_FORMAT.items():
        top = slot.split(".")[0].split("[")[0]
        if top in fields and fmt in (F_HEX64, F_OID):
            out.append(slot)
    return sorted(out)


def authority_coverage():
    """Q1a-9 coverage gate: slots=N · authority-named=N · unknown=0."""
    slots = digest_slots()
    named = [s for s in slots if SLOT_AUTHORITY.get(s) in AUTHORITIES]
    unknown = sorted(set(slots) - set(named))
    stray = sorted(set(SLOT_AUTHORITY) - set(slots))
    return {"slots": len(slots), "authority_named": len(named),
            "unknown": unknown, "mapped_but_not_a_slot": stray}


_cov = authority_coverage()
if _cov["unknown"] or _cov["mapped_but_not_a_slot"]:
    raise StageIdentityError(f"REFUSE: provenance authority coverage {_cov}")


def provenance_digest(prov):
    """Canonical digest of an in-band provenance block (D379 §4)."""
    return sha256_hex(_jcs(prov))


def _precommitted(descriptor, census_manifest_bytes):
    """Every PRECOMMITTED_STAGE_A expected value, from the descriptor (and,
    for Census members, from manifest bytes BOUND by the descriptor's
    aggregate) — never from the provenance being verified."""
    exp = {
        "stage_a_identity": stage_a_identity(descriptor),
        "stage_a_descriptor_digest": stage_a_descriptor_digest(descriptor),
        "subject_commit": descriptor["subject"]["commit"],
        "subject_tree": descriptor["subject"]["tree"],
        "tree_paths_identity": descriptor["tree_paths"]["tree_paths_identity"],
        "census_identity": descriptor["census"]["aggregate_sha256"],
        "history_source_identity": descriptor["history"]["reachable_set_sha256"],
        "input_binding.pass_a_stage_a_identity": stage_a_identity(descriptor),
    }
    for k in RUNTIME_FIELDS:
        exp[f"runtime_identity.{k}"] = descriptor["runtime"][k]
    h2 = {m["path"]: m["sha256"] for m in descriptor["h2_sources"]}
    census = None
    if census_manifest_bytes is not None:
        if sha256_hex(census_manifest_bytes) != descriptor["census"]["aggregate_sha256"]:
            raise StageIdentityError(
                "REFUSE: supplied Census manifest bytes are not the Stage-A "
                "Census aggregate")
        census = {f"{CENSUS_PACKAGE}/{n}": d for n, d in
                  _parse_census_manifest(census_manifest_bytes).items()}
    return exp, h2, census


def verify_provenance(recorded, descriptor, *, pass_a_bytes=None,
                      artifact_bytes=None, census_manifest_bytes=None,
                      pass_a_binding=None):
    """Verify a RECORDED producer provenance, SLOT BY SLOT, against the
    authority each slot is mapped to, or REFUSE naming the slot.

    v4.5 §16 / v4.1 C4. Every digest-bearing slot has a named authority
    (SLOT_AUTHORITY, coverage-gated at import). No recorded value defines
    its own expected value. CENSUS member digests are checked against the
    manifest bytes BOUND by the Stage-A aggregate (census_manifest_bytes);
    STDLIB/BUILTIN/FROZEN digests must be exactly "" (EMPTY_BY_RULE);
    input_binding artefact slots need the exact Pass-A bytes.
    Returns (stage_a_identity, verified_h2_identities, unverified_slots).
    `unverified_slots` names every slot this call could NOT establish;
    governed consumers REFUSE unless it is empty (R17).
    """
    ident = stage_a_identity(descriptor)
    if not isinstance(recorded, dict):
        raise StageIdentityError("REFUSE: provenance is not an object")
    comp = recorded.get("producer_component")
    if comp not in PROV_SHAPE:
        raise StageIdentityError(
            f"REFUSE: producer_component {comp!r} is not one of the D379 §4 "
            f"governed components {sorted(PROV_SHAPE)}.")
    expect_fields = set(PROV_SHAPE[comp])
    if set(recorded) != expect_fields:
        raise StageIdentityError(
            f"REFUSE: {comp} provenance does not carry the D379 §4 field "
            f"set. missing={sorted(expect_fields - set(recorded))} "
            f"unexpected={sorted(set(recorded) - expect_fields)}")

    # D379 §4: NO SELF-OUTPUT DIGEST, checked BEFORE the value comparators,
    # over the WHOLE object (forms (a) and (b) are both computed here).
    if artifact_bytes is not None:
        forbidden = {sha256_hex(artifact_bytes): "the artefact's own bytes"}
        try:
            doc = json.loads(artifact_bytes.decode("utf-8"))
            if isinstance(doc, dict) and "producer_provenance" in doc:
                stripped = {k: v for k, v in doc.items() if k != "producer_provenance"}
                forbidden[sha256_hex(_jcs(stripped))] = (
                    "the artefact with its own provenance block removed")
        except (ValueError, UnicodeDecodeError):
            pass
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

    if recorded["producer_component"] != comp:          # BANKED role
        raise StageIdentityError("REFUSE: producer_component slot mismatch")
    exp, stage_h2, census = _precommitted(descriptor, census_manifest_bytes)
    rt = recorded["runtime_identity"]
    if not isinstance(rt, dict) or set(rt) != set(RUNTIME_FIELDS):
        raise StageIdentityError("REFUSE: runtime_identity does not carry the D380 §6.9 fields")
    flat = {f: recorded[f] for f in expect_fields
            if f not in ("runtime_identity", "producer_population", "input_binding")}
    flat.update({f"runtime_identity.{k}": v for k, v in rt.items()})
    if comp == "CLASSIFICATION":
        ib = recorded["input_binding"]
        if not isinstance(ib, dict) or set(ib) != set(INPUT_BINDING_FIELDS):
            raise StageIdentityError(
                f"REFUSE: classification input_binding does not carry the "
                f"D379 §4 field set {list(INPUT_BINDING_FIELDS)}")
        flat.update({f"input_binding.{k}": v for k, v in ib.items()})

    unverified = set()
    # every PRECOMMITTED scalar slot, and every non-digest runtime field
    for slot, value in sorted(flat.items()):
        if slot in exp and value != exp[slot]:
            raise StageIdentityError(
                f"REFUSE: {comp} provenance slot {slot!r} does not match its "
                f"authority {SLOT_AUTHORITY.get(slot, AUTH_PRECOMMITTED)}. "
                f"recorded={str(value)[:80]!r} expected={str(exp[slot])[:80]!r}")

    # producer_population, member by member, by class authority
    members = recorded["producer_population"]
    if not isinstance(members, list) or not members:
        raise StageIdentityError("REFUSE: provenance records no producer_population member")
    seen_keys, seen_h2 = set(), set()
    for m in members:
        if not isinstance(m, dict) or set(m) != {"class", "identity", "sha256"} \
                or m["class"] not in ORIGIN_CLASSES:
            raise StageIdentityError(f"REFUSE: malformed provenance member {m!r}")
        key = (m["class"], m["identity"])
        if key in seen_keys:
            raise StageIdentityError(f"REFUSE: duplicate provenance member {key}")
        seen_keys.add(key)
        slot = f"producer_population[{m['class']}].sha256"
        auth = SLOT_AUTHORITY[slot]
        if auth == AUTH_EMPTY:
            if m["sha256"] != "":
                raise StageIdentityError(
                    f"REFUSE: slot {slot} for {m['identity']} must be empty by "
                    f"rule (EMPTY_BY_RULE); got a value")
        elif m["class"] == CLASS_H2:
            if m["identity"] not in stage_h2:
                raise StageIdentityError(
                    f"REFUSE: provenance names H2 member {m['identity']}, which "
                    f"is NOT represented in Stage A. No silent runtime expansion.")
            if m["sha256"] != stage_h2[m["identity"]]:
                raise StageIdentityError(
                    f"REFUSE: slot {slot}: H2 member {m['identity']} byte "
                    f"mismatch against Stage A")
            seen_h2.add(m["identity"])
        elif m["class"] == CLASS_CENSUS:
            if census is None:
                unverified.add(slot)
            elif census.get(m["identity"]) != m["sha256"]:
                raise StageIdentityError(
                    f"REFUSE: slot {slot}: Census member {m['identity']} is not "
                    f"the governed manifest digest bound by Stage A")
    if recorded["producer_denominator"] != len(members):
        raise StageIdentityError(
            f"REFUSE: producer_denominator {recorded['producer_denominator']} "
            f"does not equal the recorded producer_population size "
            f"{len(members)} (R5).")

    if comp == "CLASSIFICATION":
        ib = recorded["input_binding"]
        if pass_a_bytes is None and pass_a_binding is not None:
            # v4.5 §15.4: the qualifier closes both artefact-derived slots
            # from the REFERENCED Pass-A Stage-B binding, which the caller
            # has already verified against its independently held anchor.
            if pass_a_binding.get("producer_component") != "PASS_A" or \
                    pass_a_binding.get("stage_a_identity") != ident:
                raise StageIdentityError(
                    "REFUSE: the referenced Pass-A binding is not a PASS_A "
                    "binding under this Stage A")
            if ib["pass_a_artifact_sha256"] != pass_a_binding["artifact_sha256"]:
                raise StageIdentityError(
                    "REFUSE: slot input_binding.pass_a_artifact_sha256 does not "
                    "match the referenced Pass-A Stage-B binding")
            if ib["pass_a_producer_provenance_digest"] != \
                    pass_a_binding["producer_provenance_digest"]:
                raise StageIdentityError(
                    "REFUSE: slot input_binding.pass_a_producer_provenance_digest "
                    "does not match the referenced Pass-A Stage-B binding")
        elif pass_a_bytes is None:
            unverified |= {"input_binding.pass_a_artifact_sha256",
                           "input_binding.pass_a_producer_provenance_digest"}
        else:
            actual = sha256_hex(pass_a_bytes)
            if ib["pass_a_artifact_sha256"] != actual:
                raise StageIdentityError(
                    f"REFUSE: slot input_binding.pass_a_artifact_sha256 is not "
                    f"the digest of the exact Pass-A bytes consumed ({actual}).")
            pdoc = strict_json(pass_a_bytes, "Pass-A artefact")
            pp = pdoc.get("producer_provenance") if isinstance(pdoc, dict) else None
            if pp is None or ib["pass_a_producer_provenance_digest"] != provenance_digest(pp):
                raise StageIdentityError(
                    "REFUSE: slot input_binding.pass_a_producer_provenance_digest "
                    "is not the canonical digest of the provenance block inside "
                    "the exact Pass-A bytes consumed.")
    return ident, seen_h2, unverified


def require_complete(unverified, who):
    """v4.1 C4: a governed consumer REFUSES when any slot is unverified."""
    if unverified:
        raise StageIdentityError(
            f"REFUSE ({who}): provenance slot(s) left unverified "
            f"{sorted(unverified)}; a partial verification is not a verification")


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


def _cli(argv):
    """`stage_identity.py bind` — the SEPARATE governed binder invocation
    (v4.5 §13.3). Writes the canonical binding create-only and prints its
    sha256, which the PARENT retains as the independent anchor."""
    import argparse
    ap = argparse.ArgumentParser(prog="stage_identity.py")
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("bind")
    b.add_argument("--artifact", required=True)
    b.add_argument("--component", required=True, choices=sorted(ARTIFACT_KIND))
    b.add_argument("--stage-a", required=True)
    b.add_argument("--producer-exit", required=True, type=int)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd != "bind":
        print(json.dumps({"schema_v2": SCHEMA_V2, "domain_v2": DOMAIN_V2,
                          "stdlib_schema": STDLIB_SCHEMA,
                          "governance_v2": [list(g) for g in GOVERNANCE_V2],
                          "h2_sources": list(H2_SOURCES),
                          "provenance_authority_coverage": authority_coverage()},
                         indent=1))
        return 0
    try:
        desc = parse_descriptor_bytes(_read_regular_once(a.stage_a))
        binding, bb = bind_artifact(a.artifact, producer_component=a.component,
                                    stage_a_desc=desc,
                                    producer_exit_status=a.producer_exit)
        with open(a.out, "xb") as fh:                 # create-only
            fh.write(bb)
    except (StageIdentityError, FileExistsError) as exc:
        print(f"BIND REFUSED: {exc}", file=sys.stderr)
        return 3
    print(sha256_hex(bb))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
