"""Change Authority Interlock (CAI) v1.0 — shared verification library.

Support module for scripts/security/check_cai_*.py. It has no main(): it
is a helper, not a check, and the gate registry classifies it by that
property.

PURE. Every function reads Git objects and returns facts or refusals. No
function here writes to the repository, creates a ref, merges, admits or
creates authority. The only filesystem writes are to a private temporary
GnuPG home holding the pinned public key, which is deleted afterwards.

FAIL CLOSED. Anything that cannot be established is a refusal with a
reason, never a pass. The contract is
kai-pm/change-control/KAI_CHANGE_AUTHORITY_INTERLOCK.md; section numbers
below refer to it.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "kai-pm" / "change-control" / "schemas"

REPOSITORY_ID = 1004463473
REPOSITORY_FULL_NAME = "dainius1234/kai-system"

NS_WA = "refs/tags/kai-wa/"
NS_ADMITTED = "refs/tags/kai-admitted/"
ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
WA_REF_RE = re.compile(r"^refs/tags/kai-wa/wa/(%s)$" % ID_RE)
EV_REF_RE = re.compile(r"^refs/tags/kai-wa/ev/(%s)/(%s)$" % (ID_RE, ID_RE))
ADM_REF_RE = re.compile(r"^refs/tags/kai-admitted/(%s)$" % ID_RE)
HEX40 = re.compile(r"^[0-9a-f]{40}$")

# Signature states, contract §5. Only VALID_PINNED is eligible.
VALID_PINNED = "VALID_PINNED"
WRONG_KEY = "WRONG_KEY"
UNSIGNED = "UNSIGNED"
INVALID = "INVALID"
UNKNOWN = "UNKNOWN"
SIGNATURE_STATES = (VALID_PINNED, WRONG_KEY, UNSIGNED, INVALID, UNKNOWN)

PASS = "PASS"
REFUSE = "REFUSE"
UNRESOLVED = "UNRESOLVED"

GIT_TIMEOUT = 60
GPG_TIMEOUT = 60


class CaiError(Exception):
    """A fact could not be established. Callers turn this into REFUSE."""


# ── Git, with the environment pinned ──────────────────────────────────
#
# A user's global or system Git configuration must not be able to change
# an authority decision: diff.renames, core.quotepath, gpg.format,
# gpg.program and gpg.ssh.* all would. So system and global config are
# disabled and every relevant option is forced on the command line.
def _git_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GIT_", "GNUPG"))}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    if extra:
        env.update(extra)
    return env


FORCED = ["-c", "core.quotepath=false", "-c", "diff.renames=false",
          "-c", "gpg.format=openpgp", "--literal-pathspecs"]


def git(repo, *args, env_extra=None, check=True, timeout=GIT_TIMEOUT,
        input_bytes=None) -> subprocess.CompletedProcess:
    cmd = ["git", "-C", str(repo)] + FORCED + list(args)
    try:
        pr = subprocess.run(cmd, capture_output=True, env=_git_env(env_extra),
                            timeout=timeout, input=input_bytes)
    except FileNotFoundError as e:
        raise CaiError(f"git unavailable: {e}")
    except subprocess.TimeoutExpired:
        raise CaiError(f"git timed out: {' '.join(args[:3])}")
    if check and pr.returncode != 0:
        raise CaiError(f"git {' '.join(args[:4])} failed rc={pr.returncode}: "
                       f"{pr.stderr.decode('utf-8', 'replace').strip()[:300]}")
    return pr


def object_type(repo, sha) -> Optional[str]:
    pr = git(repo, "cat-file", "-t", sha, check=False)
    return pr.stdout.decode().strip() if pr.returncode == 0 else None


def commit_tree(repo, commit) -> str:
    if object_type(repo, commit) != "commit":
        raise CaiError(f"{commit} is not a commit object in this repository")
    return git(repo, "rev-parse", "--verify", f"{commit}^{{tree}}"
               ).stdout.decode().strip()


def is_ancestor(repo, ancestor, descendant) -> bool:
    pr = git(repo, "merge-base", "--is-ancestor", ancestor, descendant,
             check=False)
    if pr.returncode == 0:
        return True
    if pr.returncode == 1:
        return False
    raise CaiError("ancestry could not be established (history unavailable?): "
                   + pr.stderr.decode("utf-8", "replace").strip()[:200])


# ── The schemas are the single definition ─────────────────────────────
#
# A validator that silently ignores a schema keyword it does not
# implement passes records the schema forbids. So the supported keyword
# set is closed, and an unsupported keyword is a CaiError, not a skip.
_SCHEMA_META = {"$schema", "$id", "title"}
_SCHEMA_KW = {"type", "const", "enum", "pattern", "minLength", "minimum",
              "required", "additionalProperties", "properties", "items"}
_TYPES = {"object": dict, "array": list, "string": str, "null": type(None)}


def load_schema(name: str) -> dict:
    p = SCHEMA_DIR / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CaiError(f"schema {name} unavailable: {e}")


def _type_ok(v, t) -> bool:
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(inst, schema, path="$") -> List[str]:
    errs: List[str] = []
    unknown = set(schema) - _SCHEMA_KW - _SCHEMA_META
    if unknown:
        raise CaiError(f"schema at {path} uses unsupported keywords "
                       f"{sorted(unknown)}; refusing rather than ignoring them")
    if "type" in schema:
        ts = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(inst, t) for t in ts):
            return [f"{path}: expected type {ts}, got {type(inst).__name__}"]
    if "const" in schema and inst != schema["const"]:
        errs.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and inst not in schema["enum"]:
        errs.append(f"{path}: {inst!r} not in {schema['enum']}")
    if isinstance(inst, str):
        if "minLength" in schema and len(inst) < schema["minLength"]:
            errs.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], inst):
            errs.append(f"{path}: does not match {schema['pattern']}")
    if _type_ok(inst, "integer") and "minimum" in schema \
            and inst < schema["minimum"]:
        errs.append(f"{path}: below minimum {schema['minimum']}")
    if isinstance(inst, dict):
        for r in schema.get("required", []):
            if r not in inst:
                errs.append(f"{path}: missing required field {r!r}")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for k in inst:
                if k not in props:
                    errs.append(f"{path}: unexpected field {k!r}")
        for k, sub in props.items():
            if k in inst:
                errs += validate(inst[k], sub, f"{path}.{k}")
    if isinstance(inst, list) and "items" in schema:
        for i, v in enumerate(inst):
            errs += validate(v, schema["items"], f"{path}[{i}]")
    return errs


# ── Canonical payloads, contract §4 ───────────────────────────────────
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False) + "\n"


def _no_dupes(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise CaiError(f"duplicate keys in payload: "
                       f"{sorted(k for k in keys if keys.count(k) > 1)}")
    return dict(pairs)


def _no_float(s):
    raise CaiError(f"floats are not permitted in payloads: {s}")


def parse_canonical(text: str) -> dict:
    try:
        obj = json.loads(text, object_pairs_hook=_no_dupes,
                         parse_float=_no_float,
                         parse_constant=_no_float)
    except ValueError as e:
        raise CaiError(f"payload is not JSON: {e}")
    if not isinstance(obj, dict):
        raise CaiError("payload is not a JSON object")
    if canonical(obj) != text:
        raise CaiError("payload is not in canonical form (one byte "
                       "representation per record is required)")
    return obj


# ── Tag objects ───────────────────────────────────────────────────────
_SIG_MARKERS = {
    "-----BEGIN PGP SIGNATURE-----": "openpgp",
    "-----BEGIN SSH SIGNATURE-----": "ssh",
    "-----BEGIN SIGNED MESSAGE-----": "x509",
}


@dataclass
class TagObject:
    sha: str
    target: str
    target_type: str
    tag_name: str
    tagger_time: Optional[int]
    message: str
    signature_format: Optional[str]


def read_tag(repo, ref) -> TagObject:
    """Read an ANNOTATED tag object. A lightweight tag is not a record."""
    sha = git(repo, "rev-parse", "--verify", ref).stdout.decode().strip()
    if object_type(repo, sha) != "tag":
        raise CaiError(f"{ref} is not an annotated tag object "
                       f"(a lightweight tag is not a control record)")
    raw = git(repo, "cat-file", "tag", sha).stdout
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise CaiError(f"{ref}: tag object is not UTF-8")
    head, sep, body = text.partition("\n\n")
    if not sep:
        raise CaiError(f"{ref}: tag object has no message")
    hdr = {}
    for line in head.split("\n"):
        k, _, v = line.partition(" ")
        if k in hdr:
            raise CaiError(f"{ref}: duplicate tag header {k!r}")
        hdr[k] = v
    for k in ("object", "type", "tag"):
        if k not in hdr:
            raise CaiError(f"{ref}: tag header {k!r} missing")
    tagger_time = None
    m = re.search(r" (\d+) [+-]\d{4}$", hdr.get("tagger", ""))
    if m:
        tagger_time = int(m.group(1))
    fmt, msg = None, body
    for marker, name in _SIG_MARKERS.items():
        i = body.find(marker)
        if i != -1:
            fmt, msg = name, body[:i]
            break
    return TagObject(sha=sha, target=hdr["object"], target_type=hdr["type"],
                     tag_name=hdr["tag"], tagger_time=tagger_time,
                     message=msg, signature_format=fmt)


# ── Signature classification, contract §5 ─────────────────────────────
@dataclass(frozen=True)
class AuthorityKey:
    """The pinned authority. fingerprint None = nothing pinned."""
    fingerprint: Optional[str]
    armored_path: Optional[str]
    gpg_program: str = "gpg"


def load_authority_config(path) -> AuthorityKey:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CaiError(f"authority configuration unavailable: {e}")
    if cfg.get("schema") != "kai.cai.authority_keys.v1":
        raise CaiError("authority configuration has the wrong schema")
    fpr = cfg.get("pinned_primary_fingerprint")
    ap = cfg.get("public_key_armored_path")
    if ap is not None and not os.path.isabs(ap):
        ap = str((Path(path).resolve().parent / ap).resolve())
    return AuthorityKey(fingerprint=fpr.upper() if fpr else None,
                        armored_path=ap,
                        gpg_program=cfg.get("gpg_program") or "gpg")


class PinnedKeyring:
    """A private GnuPG home holding ONLY the pinned public key."""

    def __init__(self, key: AuthorityKey):
        self.key = key
        self.home: Optional[str] = None
        self.error: Optional[str] = None

    def __enter__(self):
        if not self.key.fingerprint or not self.key.armored_path:
            self.error = "no authority key is pinned"
            return self
        self.home = tempfile.mkdtemp(prefix="cai-gnupg-")
        os.chmod(self.home, 0o700)
        try:
            pr = subprocess.run(
                [self.key.gpg_program, "--batch", "--homedir", self.home,
                 "--import", self.key.armored_path],
                capture_output=True, timeout=GPG_TIMEOUT)
            if pr.returncode != 0:
                self.error = ("pinned key import failed: "
                              + pr.stderr.decode("utf-8", "replace")[:200])
                return self
            ls = subprocess.run(
                [self.key.gpg_program, "--batch", "--homedir", self.home,
                 "--with-colons", "--fingerprint", "--list-keys"],
                capture_output=True, timeout=GPG_TIMEOUT)
            prim = [l.split(":")[9] for l in
                    ls.stdout.decode("utf-8", "replace").splitlines()
                    if l.startswith("fpr:")]
            pubs = [l for l in ls.stdout.decode("utf-8", "replace").splitlines()
                    if l.startswith("pub:")]
            if len(pubs) != 1 or not prim or prim[0].upper() != self.key.fingerprint:
                self.error = (f"pinned key file does not contain exactly the "
                              f"pinned primary key {self.key.fingerprint} "
                              f"(found {len(pubs)} keys, first fpr "
                              f"{prim[0] if prim else None})")
        except (OSError, subprocess.TimeoutExpired) as e:
            self.error = f"gpg unavailable: {e}"
        return self

    def __exit__(self, *exc):
        if self.home:
            shutil.rmtree(self.home, ignore_errors=True)


def classify_signature(repo, ref, keyring: PinnedKeyring,
                       tag: TagObject) -> Tuple[str, str]:
    """Return (state, reason). Status lines decide, never the exit code."""
    if tag.signature_format is None:
        return UNSIGNED, "the tag object carries no signature block"
    if tag.signature_format != "openpgp":
        return UNKNOWN, (f"{tag.signature_format} signatures are not "
                         f"implemented (contract P-4); not verifiable here")
    if keyring.error or not keyring.home:
        return UNKNOWN, keyring.error or "no keyring"
    try:
        pr = git(repo, "-c", f"gpg.program={keyring.key.gpg_program}",
                 "verify-tag", "--raw", ref,
                 env_extra={"GNUPGHOME": keyring.home}, check=False,
                 timeout=GPG_TIMEOUT)
    except CaiError as e:
        return UNKNOWN, f"verification unavailable: {e}"
    status = [l[len("[GNUPG:] "):] for l in
              pr.stderr.decode("utf-8", "replace").splitlines()
              if l.startswith("[GNUPG:] ")]
    if not status:
        return UNKNOWN, ("verification produced no GnuPG status output "
                         "(rc=%d): %s" % (pr.returncode, pr.stderr.decode(
                             "utf-8", "replace").strip()[:200]))
    kw = [s.split(" ", 1)[0] for s in status]
    if kw.count("NEWSIG") > 1:
        return INVALID, "more than one signature"
    for bad in ("BADSIG", "EXPSIG", "EXPKEYSIG", "REVKEYSIG"):
        if bad in kw:
            return INVALID, f"GnuPG reported {bad}"
    if "ERRSIG" in kw:
        errsig = next(s for s in status if s.startswith("ERRSIG "))
        parts = errsig.split()
        if "NO_PUBKEY" in kw or (len(parts) > 6 and parts[6] == "9"):
            return WRONG_KEY, "the signer is not the pinned authority key"
        return INVALID, f"GnuPG reported {errsig[:80]}"
    valid = [s.split() for s in status if s.startswith("VALIDSIG ")]
    if len(valid) == 1 and "GOODSIG" in kw:
        primary = valid[0][-1].upper()
        if primary == keyring.key.fingerprint:
            return VALID_PINNED, f"signed by pinned key {primary}"
        return WRONG_KEY, f"valid signature by non-pinned key {primary}"
    return UNKNOWN, f"unrecognised GnuPG status sequence {kw}"


# ── Git-blob references: a digest field proves nothing until resolved ─
def resolve_blob(repo, ref: dict) -> bytes:
    """Resolve {kind: git-blob, commit, path} to the EXACT committed bytes."""
    if ref.get("kind") != "git-blob":
        raise CaiError(f"unsupported reference kind {ref.get('kind')!r}")
    commit, path = ref.get("commit", ""), ref.get("path", "")
    if object_type(repo, commit) != "commit":
        raise CaiError(f"referenced commit {commit} is not available")
    pr = git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path)
    ents = [e for e in pr.stdout.split(b"\0") if e]
    if len(ents) != 1:
        raise CaiError(f"{path} not found at {commit[:12]} "
                       f"({len(ents)} tree entries matched)")
    meta, _, name = ents[0].partition(b"\t")
    mode, otype, sha = meta.decode().split()
    if name.decode("utf-8", "replace") != path or otype != "blob" \
            or mode not in ("100644", "100755"):
        raise CaiError(f"{path} at {commit[:12]} is not a regular blob "
                       f"(mode {mode}, type {otype})")
    return git(repo, "cat-file", "blob", sha).stdout


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── Records ───────────────────────────────────────────────────────────
@dataclass
class Record:
    ref: str
    kind: str
    ok: bool = False
    reasons: List[str] = field(default_factory=list)
    payload: Optional[dict] = None
    tag: Optional[TagObject] = None
    signature: Optional[str] = None
    ids: Tuple[str, ...] = ()


SCHEMA_FOR = {"wa": "work_authority.v1.schema.json",
              "ev": "terminal_event.v1.schema.json",
              "admission": "admission.v1.schema.json"}


def enumerate_namespace(repo, prefix) -> List[str]:
    pr = git(repo, "for-each-ref", "--format=%(refname)", prefix)
    return sorted(l for l in pr.stdout.decode("utf-8", "replace").splitlines()
                  if l)


def classify_ref(ref) -> Tuple[Optional[str], Tuple[str, ...]]:
    for kind, rx in (("wa", WA_REF_RE), ("ev", EV_REF_RE),
                     ("admission", ADM_REF_RE)):
        m = rx.match(ref)
        if m:
            return kind, m.groups()
    return None, ()


def verify_record(repo, ref, kind, ids, keyring) -> Record:
    rec = Record(ref=ref, kind=kind, ids=ids)
    try:
        tag = read_tag(repo, ref)
    except CaiError as e:
        rec.reasons.append(str(e))
        return rec
    rec.tag = tag
    short = ref[len("refs/tags/"):]
    if tag.tag_name != short:
        rec.reasons.append(f"tag object names itself {tag.tag_name!r} but "
                           f"is published as {short!r}")
    state, why = classify_signature(repo, ref, keyring, tag)
    rec.signature = state
    if state != VALID_PINNED:
        rec.reasons.append(f"signature {state}: {why}")
    try:
        payload = parse_canonical(tag.message)
    except CaiError as e:
        rec.reasons.append(str(e))
        return rec
    rec.payload = payload
    try:
        errs = validate(payload, load_schema(SCHEMA_FOR[kind]))
    except CaiError as e:
        errs = [str(e)]
    rec.reasons += [f"schema: {e}" for e in errs]
    if payload.get("repository_id") != REPOSITORY_ID or \
            payload.get("repository_full_name") != REPOSITORY_FULL_NAME:
        rec.reasons.append("repository identity mismatch")
    if kind == "wa":
        if payload.get("wa_id") != ids[0]:
            rec.reasons.append("ref WA id != payload wa_id")
        if tag.target_type != "commit" or \
                tag.target != payload.get("baseline_commit"):
            rec.reasons.append("WA tag must target its baseline_commit")
    elif kind == "ev":
        if payload.get("wa_id") != ids[0] or payload.get("event_id") != ids[1]:
            rec.reasons.append("ref ids != payload wa_id/event_id")
        if tag.target_type != "tag":
            rec.reasons.append("terminal event must target the WA tag object")
    elif kind == "admission":
        if payload.get("admission_id") != ids[0]:
            rec.reasons.append("ref admission id != payload admission_id")
        if tag.target_type != "commit":
            rec.reasons.append("admission tag must target a commit")
    rec.ok = not rec.reasons
    return rec


@dataclass
class Namespace:
    was: Dict[str, Record] = field(default_factory=dict)
    events: Dict[str, List[Record]] = field(default_factory=dict)
    admissions: List[Record] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)


def load_namespace(repo, keyring) -> Namespace:
    """Enumerate EXACTLY the two authoritative directories and verify every
    record. P-3: anything that is not a verifying record is an anomaly."""
    ns = Namespace()
    for prefix in (NS_WA, NS_ADMITTED):
        for ref in enumerate_namespace(repo, prefix):
            kind, ids = classify_ref(ref)
            if kind is None or not ref.startswith(prefix) or \
                    (prefix == NS_ADMITTED) != (kind == "admission"):
                ns.anomalies.append(f"{ref}: look-alike — does not match the "
                                    f"record grammar of its namespace")
                continue
            rec = verify_record(repo, ref, kind, ids, keyring)
            if not rec.ok:
                ns.anomalies.append(f"{ref}: " + "; ".join(rec.reasons))
                continue
            if kind == "wa":
                ns.was[ids[0]] = rec
            elif kind == "ev":
                ns.events.setdefault(ids[0], []).append(rec)
            else:
                ns.admissions.append(rec)
    return ns


# ── Path / mode semantics, contract §6 ────────────────────────────────
@dataclass
class Change:
    status: str
    old_mode: str
    new_mode: str
    paths: Tuple[bytes, ...]


def change_population(repo, baseline, candidate) -> List[Change]:
    pr = git(repo, "diff-tree", "-r", "-z", "--raw", "--no-renames",
             "--no-ext-diff", baseline, candidate)
    parts = pr.stdout.split(b"\0")
    out, i = [], 0
    while i < len(parts) and parts[i]:
        meta = parts[i].decode("ascii")
        if not meta.startswith(":"):
            raise CaiError(f"unparseable diff-tree record {meta[:60]!r}")
        om, nm, _os, _ns, st = meta[1:].split(" ")
        if st[0] in "RC":
            raise CaiError("rename/copy record present despite --no-renames")
        out.append(Change(st, om, nm, (parts[i + 1],)))
        i += 2
    return out


def tree_paths(repo, commit) -> List[bytes]:
    pr = git(repo, "ls-tree", "-r", "-z", "--name-only", "--full-tree", commit)
    return [p for p in pr.stdout.split(b"\0") if p]


def _fold(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold()


def path_problems(raw: bytes) -> Tuple[Optional[str], List[str]]:
    try:
        p = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        return None, ["path is not strict UTF-8"]
    probs = []
    if unicodedata.normalize("NFC", p) != p:
        probs.append("path is not NFC-normalised")
    if re.search(r"[\x00-\x1f\x7f]", p):
        probs.append("path contains a control character")
    if "\\" in p:
        probs.append("path contains a backslash")
    comps = p.split("/")
    if any(c in ("", ".", "..") for c in comps):
        probs.append("path has an empty, '.' or '..' component")
    if any(c.casefold() == ".git" for c in comps):
        probs.append("path has a .git component")
    return p, probs


def in_scope(path: str, scope: dict) -> Tuple[bool, str]:
    fp, fpre = scope.get("forbidden_paths", []), scope.get("forbidden_prefixes", [])
    ap, apre = scope.get("allowed_paths", []), scope.get("allowed_prefixes", [])
    for pre in fpre + apre:
        if not pre.endswith("/"):
            raise CaiError(f"prefix {pre!r} does not end in '/'")
    if path in fp:
        return False, "forbidden exact path"
    if any(path.startswith(pre) for pre in fpre):
        return False, "under a forbidden prefix"
    if path in ap or any(path.startswith(pre) for pre in apre):
        return True, "allowed"
    return False, "outside every allowed path and prefix"


def evaluate_scope(repo, baseline, candidate, scope) -> Tuple[str, List[str], int]:
    """Return (PASS|REFUSE, reasons, denominator of changed entries)."""
    reasons = []
    changes = change_population(repo, baseline, candidate)
    cand = tree_paths(repo, candidate)
    folded: Dict[str, List[str]] = {}
    for raw in cand:
        s = raw.decode("utf-8", "replace")
        folded.setdefault(_fold(s), []).append(s)
    for ch in changes:
        for mode in (ch.old_mode, ch.new_mode):
            if mode == "120000":
                reasons.append(f"{ch.paths[0]!r}: symlink (mode 120000) refused")
            if mode == "160000":
                reasons.append(f"{ch.paths[0]!r}: gitlink (mode 160000) refused")
        for raw in ch.paths:
            p, probs = path_problems(raw)
            if probs:
                reasons += [f"{raw!r}: {x}" for x in probs]
                continue
            if ch.status != "D" and len(folded.get(_fold(p), [])) > 1:
                reasons.append(f"{p}: collides after case-fold/NFC with "
                               f"{folded[_fold(p)]}")
            ok, why = in_scope(p, scope)
            if not ok:
                reasons.append(f"{p} ({ch.status}): {why}")
    return (PASS if not reasons else REFUSE), reasons, len(changes)


# ── Work Authority state, contract §7 ─────────────────────────────────
def parse_utc(s: str) -> int:
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def terminal_events(ns: Namespace, wa: Record) -> List[Record]:
    """Valid terminal events that are bound to THIS exact WA tag object."""
    out = []
    for ev in ns.events.get(wa.ids[0], []):
        if ev.tag.target == wa.tag.sha:
            out.append(ev)
    return out


def wa_standing(repo, wa: Record, at_time: int) -> List[str]:
    """Reasons a WA record is not usable at `at_time` (empty = usable)."""
    p = wa.payload
    reasons = []
    try:
        if commit_tree(repo, p["baseline_commit"]) != p["baseline_tree"]:
            reasons.append("baseline_tree does not match baseline_commit")
    except CaiError as e:
        reasons.append(str(e))
    if at_time >= parse_utc(p["expires_at"]):
        reasons.append(f"expired at {p['expires_at']}")
    for g in p.get("governing_refs", []):
        try:
            got = sha256_hex(resolve_blob(repo, g))
            if got != g["sha256"]:
                reasons.append(f"governing ref {g['path']} digest mismatch")
        except CaiError as e:
            reasons.append(f"governing ref {g.get('path')}: {e}")
    return reasons


# ── Admission chain, contract §8 ──────────────────────────────────────
@dataclass
class Derivation:
    state: str
    reasons: List[str]
    chain: List[Record] = field(default_factory=list)
    baseline: Optional[str] = None


def derive(repo, ns: Namespace) -> Derivation:
    if ns.anomalies:
        return Derivation(REFUSE, ["authoritative namespace contains records "
                                   "that do not verify (P-3):"] + ns.anomalies)
    adm = ns.admissions
    if not adm:
        return Derivation(REFUSE, ["no admission records: no programme "
                                   "baseline exists (EMPTY chain)"])
    bad = [a.ref for a in adm if (a.payload["sequence"] == 1) !=
           (a.payload["previous_admission_tag_object"] is None)]
    if bad:
        return Derivation(REFUSE, [f"genesis rule violated (sequence 1 <=> "
                                   f"null predecessor): {bad}"])
    genesis = [a for a in adm if a.payload["sequence"] == 1]
    if len(genesis) != 1:
        return Derivation(UNRESOLVED if genesis else REFUSE,
                          [f"{len(genesis)} genesis records: "
                           f"{[g.ref for g in genesis]}"])
    chain = [genesis[0]]
    while True:
        cur = chain[-1]
        succ = [a for a in adm
                if a.payload["sequence"] == cur.payload["sequence"] + 1
                and a.payload["previous_admission_tag_object"] == cur.tag.sha]
        if len(succ) > 1:
            return Derivation(UNRESOLVED, [
                f"fork at sequence {cur.payload['sequence'] + 1}: "
                f"{[s.ref for s in succ]} — no baseline is derived and none "
                f"is chosen"])
        if not succ:
            break
        chain.append(succ[0])
    orphans = [a for a in adm if a not in chain]
    if orphans:
        return Derivation(REFUSE, [
            f"{o.ref}: sequence {o.payload['sequence']} is not on the chain "
            f"(skipped sequence or wrong predecessor tag object)"
            for o in orphans])
    reasons: List[str] = []
    consumed = set()
    for i, a in enumerate(chain):
        reasons += [f"{a.ref}: {r}" for r in
                    admission_problems(repo, ns, a, chain[i - 1] if i else None,
                                       consumed)]
        consumed.add(a.payload["wa_id"])
    if reasons:
        return Derivation(REFUSE, reasons, chain)
    return Derivation(PASS, [], chain, chain[-1].payload["candidate_commit"])


def admission_problems(repo, ns, a: Record, prev: Optional[Record],
                       consumed) -> List[str]:
    p, r = a.payload, []
    wa = ns.was.get(p["wa_id"])
    if wa is None:
        return [f"work authority {p['wa_id']} has no valid record"]
    if p["wa_id"] in consumed:
        r.append(f"work authority {p['wa_id']} already backed an earlier "
                 f"admission (replay, P-6)")
    for ev in terminal_events(ns, wa):
        if p["sequence"] > ev.payload["admission_sequence_at_issue"]:
            r.append(f"work authority {p['wa_id']} is dominated by "
                     f"{ev.payload['event']} {ev.payload['event_id']}")
    if a.tag.tagger_time is None:
        r.append("admission tag has no signed tagger time")
    else:
        r += [f"work authority at admission time: {x}"
              for x in wa_standing(repo, wa, a.tag.tagger_time)]
    w = wa.payload
    if prev is not None and w["baseline_commit"] != \
            prev.payload["candidate_commit"]:
        r.append("stale work authority: its baseline is not the previous "
                 "admitted candidate (P-7)")
    if a.tag.target != p["candidate_commit"]:
        r.append("admission tag target != candidate_commit")
    try:
        if commit_tree(repo, p["candidate_commit"]) != p["candidate_tree"]:
            r.append("candidate_tree does not match candidate_commit")
    except CaiError as e:
        r.append(str(e))
    try:
        blob = resolve_blob(repo, p["ivv_evidence_ref"])
        if sha256_hex(blob) != p["ivv_evidence_sha256"]:
            r.append("IV&V evidence digest mismatch")
        else:
            ev = parse_canonical(blob.decode("utf-8"))
            errs = validate(ev, load_schema("ivv_evidence.v1.schema.json"))
            r += [f"IV&V evidence schema: {e}" for e in errs]
            if not errs:
                if ev["candidate_commit"] != a.tag.target:
                    r.append("admission target is not the exact commit IV&V "
                             "reviewed")
                if ev["candidate_tree"] != p["candidate_tree"]:
                    r.append("IV&V candidate_tree != admission candidate_tree")
                if ev["wa_id"] != p["wa_id"]:
                    r.append("IV&V evidence is for another work authority")
                if ev["disposition"] != "ACCEPT_FOR_ADMISSION":
                    r.append(f"IV&V disposition is {ev['disposition']}")
    except (CaiError, UnicodeDecodeError) as e:
        r.append(f"IV&V evidence unavailable: {e}")
    try:
        if not is_ancestor(repo, w["baseline_commit"], p["candidate_commit"]):
            r.append("candidate does not descend from the WA baseline (P-8)")
        st, why, n = evaluate_scope(repo, w["baseline_commit"],
                                    p["candidate_commit"], w)
        r += [f"scope ({n} changed entries): {x}" for x in why]
    except CaiError as e:
        r.append(f"scope could not be evaluated: {e}")
    return r
