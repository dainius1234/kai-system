"""Per-service Ed25519 request signing — identity derived, never claimed.

The defect this closes
----------------------

    SHARED AUTHENTICATION PROVES MEMBERSHIP, NOT IDENTITY.

Measured 2026-08-07 (`scripts/security/report_service_identity.py`):
**26 of 32** protected endpoints need to know *which* service called, and
none of them could. Two mechanisms were in use and both had the same
hole:

* ``KAI_SERVICE_TOKEN`` — one bearer token every service holds.
* ``INTERSERVICE_HMAC_SECRET`` — a real signed envelope, but one key
  shared by three services, signing an ``actor_did`` the **caller
  supplies**. Caller-asserted identity, cryptographically sealed: it
  reads as stronger than a bearer token while guaranteeing the same
  thing.

Hence the one rule this module exists to enforce, and the reason nothing
here reads a name from a header:

    THE PRINCIPAL IS DERIVED FROM WHICH KEY VERIFIED THE SIGNATURE.
    A caller never states who it is.

What is reused, and what is new
-------------------------------

Reused from `common/auth.py` and `tool-gate`, which already had them:
the signed-envelope shape, key ids, timestamp skew, a persisted replay
cache, rotation overlap and revocation. `load_secret()` is reused
verbatim for Docker-secret delivery.

New, because the existing envelope could not carry identity:

1. the key is **per service**, not shared;
2. the principal comes from the key map entry, not from a payload field;
3. **method, path and body hash are inside the signed string.** Without
   them a valid signature could be replayed onto a different route of the
   same service, which was true of the existing envelope.

Why Ed25519 and why only Ed25519
--------------------------------

A receiver holds only public keys, so compromising a receiver does not
yield the power to sign as any caller — which per-service HMAC cannot
offer, since there the verifier's key is also a forging key. An HMAC
path is deliberately **not** implemented: an unused algorithm is a
downgrade surface and never-executed code, and R8 says that is where the
defects are. The algorithm is still a field in the signed string, so
adding one later cannot be done silently by a caller.

Runtime status, stated because it is not proven
-----------------------------------------------

Ed25519 library feasibility is PARTIALLY PROVEN — the pinned
`cryptography` wheel is `abi3`/`manylinux_2_28`, needs no toolchain, and
signs and verifies correctly on this platform. **Real service-image
feasibility is UNKNOWN**: no `python:3.11-slim` image has been built with
this dependency. `import_status()` exists so a service can report that
honestly at runtime rather than crash obscurely.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("kai.service_identity")

#: The only algorithm. Present as a field so a future second one has to
#: be added explicitly on both sides rather than negotiated by a caller.
ALG_ED25519 = "ed25519"
_ALGORITHMS = (ALG_ED25519,)

SIGNATURE_VERSION = b"KAI-SIG-v1"

#: `<algorithm>:<key-id>:<signature-hex>`. Note what is absent: any
#: header naming the caller. Adding one would restore the defect.
SIGNATURE_HEADER = "x-kai-signature"
TIMESTAMP_HEADER = "x-kai-timestamp"
NONCE_HEADER = "x-kai-nonce"

KEYMAP_ENV = "KAI_SERVICE_KEYMAP"
PRIVATE_KEY_ENV = "KAI_SERVICE_PRIVATE_KEY"
KEY_ID_ENV = "KAI_SERVICE_KEY_ID"
REVOKED_ENV = "KAI_SERVICE_REVOKED_KEY_IDS"
SKEW_ENV = "KAI_SIGNATURE_SKEW_SECONDS"
NONCE_TTL_ENV = "KAI_NONCE_TTL_SECONDS"
NONCE_CACHE_ENV = "KAI_NONCE_CACHE_PATH"

_DEFAULT_SKEW = 300
_DEFAULT_NONCE_TTL = 900

#: Recorded when a caller authenticated with the shared token during the
#: transition window. Deliberately not a service name, so a provenance
#: record built from it is obviously worthless rather than subtly wrong.
UNVERIFIED_IDENTITY = "unverified"


class IdentityError(Exception):
    """Configuration or key-material problem.

    Never raised for a bad signature. A bad signature is a verdict the
    caller needs returned, not an exception that might be caught by
    something with an opinion about errors.
    """


# ── the principal ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class ServicePrincipal:
    """Who called, and whether that was proven.

    ``identity`` is copied from the key map entry whose key verified the
    signature. There is no code path that fills it from a request.
    """

    identity: str
    key_id: str = ""
    algorithm: str = ""
    verified: bool = False

    @property
    def usable_for_provenance(self) -> bool:
        """Whether a provenance record may name this caller.

        Transition-window callers authenticate with the shared token and
        are honestly anonymous; recording them by name would be a lie
        with a signature-shaped confidence attached.
        """
        return self.verified and self.identity != UNVERIFIED_IDENTITY


def unverified_principal() -> ServicePrincipal:
    return ServicePrincipal(identity=UNVERIFIED_IDENTITY, verified=False)


# ── the cryptography dependency, reported rather than assumed ───────────

def import_status() -> Tuple[bool, str]:
    """(available, detail) for the Ed25519 backend.

    Exists because the failure this guards against is real and was
    measured: the *distro* `cryptography` package on the development host
    panics on import (`pyo3_runtime.PanicException`) while the pinned
    wheel is fine. A service must be able to say which it got.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: F401
            Ed25519PrivateKey)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    import cryptography
    return True, f"cryptography {cryptography.__version__}"


def _ed25519():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey, Ed25519PublicKey)
    return Ed25519PrivateKey, Ed25519PublicKey


def _require_backend() -> None:
    ok, detail = import_status()
    if not ok:
        raise IdentityError(
            f"Ed25519 signing requires the `cryptography` package and it "
            f"could not be imported ({detail}). Install the pinned wheel "
            f"(cryptography==43.0.1); the distro package is known broken.")


# ── canonical request string ────────────────────────────────────────────

def canonical_request(
    *, algorithm: str, key_id: str, destination: str, method: str,
    path: str, body: bytes, timestamp: int, nonce: str,
) -> bytes:
    """The exact bytes both ends sign.

    **Length-prefixed, not delimiter-joined.** A delimiter can be forged
    from inside a field — a path containing the separator would let a
    caller shift a boundary and sign a different meaning with the same
    bytes. A length cannot be forged that way.

    ``destination`` is included so a signature captured by one service
    cannot be replayed against another; ``method``, ``path`` and the body
    hash are included so it cannot be replayed onto another route of the
    same service.
    """
    if algorithm not in _ALGORITHMS:
        raise IdentityError(f"unknown algorithm {algorithm!r}")
    body_hash = hashlib.sha256(body or b"").hexdigest()
    fields = (algorithm, key_id, destination, method.upper(), path,
              body_hash, str(int(timestamp)), nonce)
    out = bytearray(SIGNATURE_VERSION)
    for field in fields:
        raw = str(field).encode("utf-8")
        out += str(len(raw)).encode("ascii") + b":" + raw
    return bytes(out)


# ── key material ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeyEntry:
    key_id: str
    identity: str
    algorithm: str
    public_key: bytes


class KeyMap:
    """kid -> (identity, algorithm, public key).

    Rotation needs no special machinery: two key ids for one identity are
    simply two valid entries, so an overlap window is a map edit rather
    than a dual-signing protocol. Revocation is `KAI_SERVICE_REVOKED_KEY_IDS`,
    matching the existing envelope's convention.

    Integrity rests on how the file is delivered, so this refuses the two
    states that make delivery meaningless — writable by group or other,
    and empty — and exposes a digest so drift between services is
    observable instead of assumed.
    """

    def __init__(self, entries: Dict[str, KeyEntry], digest: str) -> None:
        self._entries = entries
        self.digest = digest

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key_id: str) -> Optional[KeyEntry]:
        return self._entries.get(key_id)

    def identities(self) -> Tuple[str, ...]:
        return tuple(sorted({e.identity for e in self._entries.values()}))

    @classmethod
    def from_text(cls, text: str) -> "KeyMap":
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            doc = json.loads(text)
        except ValueError as exc:
            raise IdentityError(f"key map is not valid JSON: {exc}") from exc
        keys = doc.get("keys")
        if not isinstance(keys, dict) or not keys:
            # I-1 applied to configuration: an empty map verifies nothing
            # and must not read as "no callers are configured yet".
            raise IdentityError(
                "key map declares no keys — an empty map can verify nothing "
                "and must not be read as a system with no callers")
        entries: Dict[str, KeyEntry] = {}
        for key_id, spec in keys.items():
            if not isinstance(spec, dict):
                raise IdentityError(f"key {key_id!r} is not an object")
            identity = spec.get("identity")
            algorithm = spec.get("algorithm")
            material = spec.get("public_key")
            if not identity or not algorithm or not material:
                raise IdentityError(
                    f"key {key_id!r} needs identity, algorithm and public_key")
            if algorithm not in _ALGORITHMS:
                raise IdentityError(
                    f"key {key_id!r} declares unsupported algorithm "
                    f"{algorithm!r}")
            try:
                public = bytes.fromhex(material)
            except ValueError as exc:
                raise IdentityError(
                    f"key {key_id!r} public_key is not hex") from exc
            if len(public) != 32:
                raise IdentityError(
                    f"key {key_id!r} public_key is {len(public)} bytes; "
                    f"ed25519 public keys are 32")
            entries[key_id] = KeyEntry(key_id=key_id, identity=identity,
                                       algorithm=algorithm, public_key=public)
        return cls(entries, digest)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "KeyMap":
        raw_path = path or os.getenv(KEYMAP_ENV, "")
        if not raw_path:
            raise IdentityError(
                f"{KEYMAP_ENV} is not set — this service cannot verify any "
                f"caller's identity and must not pretend that it can")
        file_path = Path(raw_path)
        if not file_path.is_file():
            raise IdentityError(f"key map {raw_path} does not exist")
        mode = file_path.stat().st_mode & 0o777
        if mode & 0o022:
            raise IdentityError(
                f"key map {raw_path} is writable by group or other (mode "
                f"{mode:o}) — anything that can rewrite the map can mint an "
                f"identity, which is the whole property this map provides")
        keymap = cls.from_text(file_path.read_text(encoding="utf-8"))
        logger.info("service key map loaded: %d key(s), %d identity(ies), "
                    "sha256 %s", len(keymap), len(keymap.identities()),
                    keymap.digest[:16])
        return keymap


def revoked_key_ids() -> frozenset:
    raw = os.getenv(REVOKED_ENV, "")
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def load_private_key() -> Tuple[str, str, bytes]:
    """(key_id, algorithm, private material) for this service.

    Accepts an inline ``ed25519:<hex>`` value or a path — the Docker
    secret convention `common/auth.load_secret` already established. A
    private key readable by group or other is refused.
    """
    key_id = os.getenv(KEY_ID_ENV, "")
    if not key_id:
        raise IdentityError(f"{KEY_ID_ENV} is not set")
    source = os.getenv(PRIVATE_KEY_ENV, "")
    if not source:
        raise IdentityError(f"{PRIVATE_KEY_ENV} is not set")
    if source.startswith("/"):
        path = Path(source)
        if not path.is_file():
            raise IdentityError(f"private key {source} does not exist")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise IdentityError(
                f"private key {source} is readable by group or other "
                f"(mode {mode:o})")
        source = path.read_text(encoding="utf-8").strip()
    algorithm, _, material = source.partition(":")
    if algorithm not in _ALGORITHMS or not material:
        raise IdentityError(
            "private key must be '<algorithm>:<hex>' with a known algorithm")
    try:
        raw = bytes.fromhex(material)
    except ValueError as exc:
        raise IdentityError("private key material is not hex") from exc
    if len(raw) != 32:
        raise IdentityError(
            f"private key is {len(raw)} bytes; ed25519 private keys are 32")
    return key_id, algorithm, raw


def generate_keypair() -> Tuple[bytes, bytes]:
    """(private, public) raw 32-byte halves."""
    _require_backend()
    private_cls, _ = _ed25519()
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, PublicFormat)
    key = private_cls.generate()
    return (key.private_bytes(Encoding.Raw, PrivateFormat.Raw,
                              NoEncryption()),
            key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def public_from_private(private_material: bytes) -> bytes:
    _require_backend()
    private_cls, _ = _ed25519()
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat)
    return private_cls.from_private_bytes(private_material).public_key(
        ).public_bytes(Encoding.Raw, PublicFormat.Raw)


# ── sign / verify ───────────────────────────────────────────────────────

def sign(message: bytes, algorithm: str, private_material: bytes) -> str:
    if algorithm != ALG_ED25519:
        raise IdentityError(f"unknown algorithm {algorithm!r}")
    _require_backend()
    private_cls, _ = _ed25519()
    return private_cls.from_private_bytes(private_material).sign(
        message).hex()


def verify(message: bytes, signature_hex: str, entry: KeyEntry) -> bool:
    """True only if this key signed this message. Returns a verdict for
    every failure that is a caller's fault, and raises only when the
    backend itself is missing."""
    if entry.algorithm != ALG_ED25519:
        return False
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    _require_backend()
    _, public_cls = _ed25519()
    try:
        public_cls.from_public_bytes(entry.public_key).verify(
            signature, message)
        return True
    except Exception:
        return False


# ── replay defence ──────────────────────────────────────────────────────

class NonceCache:
    """Seen ``(key_id, nonce)`` inside a TTL, surviving restart.

    The restart gap is the part that matters and was an open question in
    the design note. If the cache cannot be restored, the entries it held
    are unknown — so for one skew window this refuses any timestamp older
    than start-up. A request captured before the restart carries an older
    timestamp by construction, so it is rejected without needing the
    cache at all. `scripts/test_service_identity_auth.py` proves this by
    corrupting the file.
    """

    def __init__(self, path: Optional[str] = None, ttl: Optional[int] = None,
                 now: Optional[float] = None) -> None:
        self.path = Path(path or os.getenv(NONCE_CACHE_ENV,
                                           "/tmp/kai-service-nonces.json"))
        self.ttl = int(ttl if ttl is not None
                       else os.getenv(NONCE_TTL_ENV, _DEFAULT_NONCE_TTL))
        self._seen: Dict[str, float] = {}
        self.started_at = float(now if now is not None else time.time())
        self.restored = self._restore()

    def _restore(self) -> bool:
        if not self.path.exists():
            # Nothing to restore is not a failed restore: a first start
            # has no window behind it that needs protecting.
            return True
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("nonce cache is not an object")
            self._seen = {str(k): float(v) for k, v in data.items()}
            return True
        except Exception as exc:
            logger.error(
                "SECURITY: nonce cache at %s could not be restored (%s). "
                "Refusing pre-restart timestamps for one skew window.",
                self.path, exc)
            self._seen = {}
            return False

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._seen), encoding="utf-8")
        except OSError:
            logger.warning("nonce cache could not be persisted to %s",
                           self.path)

    def floor(self, skew: int, now: Optional[float] = None) -> float:
        """Oldest timestamp acceptable right now; 0 when unconstrained."""
        if self.restored:
            return 0.0
        moment = float(now if now is not None else time.time())
        if moment - self.started_at > skew:
            return 0.0            # the unprotected window has passed
        return self.started_at

    def check_and_record(self, key_id: str, nonce: str,
                         now: Optional[float] = None) -> bool:
        """False if this ``(key_id, nonce)`` was already seen in the TTL."""
        moment = float(now if now is not None else time.time())
        for seen_key, seen_at in list(self._seen.items()):
            if moment - seen_at > self.ttl:
                self._seen.pop(seen_key, None)
        cache_key = f"{key_id}:{nonce}"
        if cache_key in self._seen:
            return False
        self._seen[cache_key] = moment
        self._persist()
        return True


def new_nonce() -> str:
    return secrets.token_hex(16)


def skew_seconds() -> int:
    try:
        return int(os.getenv(SKEW_ENV, _DEFAULT_SKEW))
    except ValueError:
        return _DEFAULT_SKEW


# ── the two ends ────────────────────────────────────────────────────────

def signed_headers(
    *, destination: str, method: str, path: str, body: bytes,
    key_id: Optional[str] = None, algorithm: Optional[str] = None,
    private_material: Optional[bytes] = None, timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
) -> Dict[str, str]:
    """Headers proving this service signed this exact request."""
    if key_id is None or algorithm is None or private_material is None:
        key_id, algorithm, private_material = load_private_key()
    stamp = int(timestamp if timestamp is not None else time.time())
    used_nonce = nonce or new_nonce()
    message = canonical_request(
        algorithm=algorithm, key_id=key_id, destination=destination,
        method=method, path=path, body=body, timestamp=stamp,
        nonce=used_nonce)
    return {
        SIGNATURE_HEADER: f"{algorithm}:{key_id}:"
                          f"{sign(message, algorithm, private_material)}",
        TIMESTAMP_HEADER: str(stamp),
        NONCE_HEADER: used_nonce,
    }


def verify_request(
    headers: Dict[str, str], *, destination: str, method: str, path: str,
    body: bytes, keymap: "KeyMap", cache: Optional[NonceCache] = None,
    now: Optional[float] = None,
) -> Tuple[Optional[ServicePrincipal], int, str]:
    """(principal, status, detail). Principal is None on refusal.

    Every refusal that could distinguish "unknown key" from "bad
    signature" returns the same detail, because that difference is
    information an attacker wants and a legitimate caller does not need.
    The log says which; the response does not.
    """
    lower = {str(k).lower(): v for k, v in headers.items()}
    raw_signature = lower.get(SIGNATURE_HEADER, "")
    if not raw_signature:
        return None, 401, "missing request signature"

    parts = raw_signature.split(":")
    if len(parts) != 3 or not all(parts):
        return None, 401, "signature must be '<algorithm>:<key-id>:<hex>'"
    algorithm, key_id, signature_hex = parts

    stamp_raw = lower.get(TIMESTAMP_HEADER, "")
    nonce = lower.get(NONCE_HEADER, "")
    if not stamp_raw or not nonce:
        return None, 401, "signature requires a timestamp and a nonce"
    try:
        stamp = int(stamp_raw)
    except ValueError:
        return None, 401, "timestamp is not an integer"

    moment = float(now if now is not None else time.time())
    skew = skew_seconds()
    if abs(moment - stamp) > skew:
        return None, 401, "timestamp outside the accepted window"

    if cache is not None:
        floor = cache.floor(skew, moment)
        if floor and stamp < floor:
            return None, 401, ("timestamp predates this instance and the "
                               "replay cache could not be restored")

    if key_id in revoked_key_ids():
        logger.warning("SECURITY: signature from REVOKED key id %r", key_id)
        return None, 401, "signature could not be verified"

    entry = keymap.get(key_id)
    if entry is None:
        logger.warning("SECURITY: signature from unknown key id %r", key_id)
        return None, 401, "signature could not be verified"
    if entry.algorithm != algorithm:
        # A downgrade also fails the signature, since the algorithm is
        # inside the signed string. Refused here first so the log names it.
        logger.warning("SECURITY: key %r is %s, request claimed %s",
                       key_id, entry.algorithm, algorithm)
        return None, 401, "signature could not be verified"

    message = canonical_request(
        algorithm=algorithm, key_id=key_id, destination=destination,
        method=method, path=path, body=body, timestamp=stamp, nonce=nonce)
    if not verify(message, signature_hex, entry):
        logger.warning("SECURITY: bad signature for key %r on %s %s",
                       key_id, method, path)
        return None, 401, "signature could not be verified"

    # Recorded last, so a request that fails verification cannot burn a
    # nonce and make the legitimate retry look like a replay.
    if cache is not None and not cache.check_and_record(key_id, nonce, moment):
        return None, 401, "this request has already been seen"

    return (ServicePrincipal(identity=entry.identity, key_id=key_id,
                             algorithm=entry.algorithm, verified=True),
            200, "identity verified")
