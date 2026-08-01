"""Erasure coordinator — subject-scoped deletion across every layer.

Closes roadmap §16.30: end-to-end data deletion across source events,
views, proposals, audit-allowed references and learning derivatives.

Each subsystem owns its own deletion through a registered handler; the
coordinator orchestrates the cascade and then **independently verifies**
that no residue remains.  Verification is the point — a deletion that
reports success without being checked is how data quietly survives.

Two rules the design enforces:

  - **Audit survives, content does not.**  Audit-layer records are
    tombstoned rather than removed, and a tombstone carries a digest of
    what was deleted, never the data.
  - **Partial is not success.**  If any layer fails or leaves residue,
    the receipt is PARTIAL or FAILED, never COMPLETE.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from common.contracts.base import Principal, Provenance
from common.contracts.erasure import (
    ErasureLayer,
    ErasureReceipt,
    ErasureRequest,
    ErasureStatus,
    LayerResult,
    Tombstone,
)


class ErasureError(Exception):
    pass


def content_digest(payload: object) -> str:
    """Stable digest of erased content, for tombstones."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LayerHandler(Protocol):
    """What each subsystem must provide to participate in erasure."""

    def find(self, subject: str) -> List[Tuple[str, object]]:
        """Return ``(record_id, content)`` pairs belonging to the subject."""

    def erase(self, subject: str) -> int:
        """Remove the subject's records. Returns the count removed."""


class CallableLayerHandler:
    """Adapter turning two functions into a LayerHandler."""

    __slots__ = ("_find", "_erase", "_tombstone")

    def __init__(
        self,
        find: Callable[[str], List[Tuple[str, object]]],
        erase: Callable[[str], int],
        tombstone: bool = False,
    ) -> None:
        self._find = find
        self._erase = erase
        self._tombstone = tombstone

    def find(self, subject: str) -> List[Tuple[str, object]]:
        return self._find(subject)

    def erase(self, subject: str) -> int:
        return self._erase(subject)

    @property
    def tombstones(self) -> bool:
        return self._tombstone


class ErasureCoordinator:
    """Orchestrates and verifies subject-scoped erasure.

    Parameters:
        principal: owning principal
    """

    def __init__(self, principal: Principal) -> None:
        self._principal = principal
        self._handlers: Dict[ErasureLayer, CallableLayerHandler] = {}
        self._tombstones: List[Tombstone] = []
        self._receipts: List[ErasureReceipt] = []

    def register_layer(
        self,
        layer: ErasureLayer,
        find: Callable[[str], List[Tuple[str, object]]],
        erase: Callable[[str], int],
        tombstone: bool = False,
    ) -> None:
        """Register a subsystem's deletion handler.

        ``tombstone=True`` marks an audit-allowed layer: records are
        recorded as tombstones before removal so the audit trail keeps
        proof the data existed.
        """
        if layer in self._handlers:
            raise ErasureError(f"layer already registered: {layer.value}")
        self._handlers[layer] = CallableLayerHandler(find, erase, tombstone)

    # ── Execution ───────────────────────────────────────────────────

    def erase(
        self,
        subject_identity: str,
        reason: str,
        requested_by: str,
        layers: Optional[List[ErasureLayer]] = None,
    ) -> ErasureReceipt:
        if not subject_identity or not subject_identity.strip():
            raise ErasureError("erasure requires a subject identity")
        if not requested_by or not requested_by.strip():
            raise ErasureError("erasure cannot be requested anonymously")
        if not reason or not reason.strip():
            raise ErasureError("erasure requires a reason")

        target_layers = layers or list(self._handlers)
        unregistered = [l for l in target_layers if l not in self._handlers]
        if unregistered:
            raise ErasureError(
                f"no handler for: {', '.join(l.value for l in unregistered)}"
            )

        request = ErasureRequest(
            subject_identity=subject_identity,
            reason=reason,
            requested_by=requested_by,
            layers=target_layers,
            principal=self._principal,
            purpose="erasure",
            provenance=Provenance(source=f"erasure:{requested_by}"),
        )

        results: List[LayerResult] = []
        total_erased = 0
        total_tombstones = 0

        for layer in target_layers:
            handler = self._handlers[layer]
            result = LayerResult(
                layer=layer,
                principal=self._principal,
                purpose="erasure_layer",
                provenance=Provenance(source=f"erasure_layer:{layer.value}"),
            )
            try:
                found = handler.find(subject_identity)
                result.records_examined = len(found)

                if handler.tombstones:
                    for record_id, content in found:
                        self._tombstones.append(Tombstone(
                            layer=layer,
                            original_id=record_id,
                            content_digest=content_digest(content),
                            erased_at=datetime.now(timezone.utc),
                            erasure_request_id=request.id,
                            principal=self._principal,
                            purpose="tombstone",
                            provenance=Provenance(source="erasure_coordinator"),
                        ))
                        result.tombstones_created += 1

                result.records_erased = handler.erase(subject_identity)
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"

            result.digest = result._make_digest()
            results.append(result)
            total_erased += result.records_erased
            total_tombstones += result.tombstones_created

        residue = self.verify_erased(subject_identity, target_layers)
        for result in results:
            result.residue_found = [
                r for r in residue if r.startswith(f"{result.layer.value}:")
            ]

        errored = any(r.error for r in results)
        if errored:
            status = ErasureStatus.FAILED
        elif residue:
            status = ErasureStatus.PARTIAL
        else:
            status = ErasureStatus.COMPLETE

        receipt = ErasureReceipt(
            request_id=request.id,
            subject_identity=subject_identity,
            status=status,
            layer_results=results,
            total_erased=total_erased,
            total_tombstones=total_tombstones,
            verified=not residue and not errored,
            verification_residue=residue,
            completed_at=datetime.now(timezone.utc),
            principal=self._principal,
            purpose="erasure_receipt",
            provenance=Provenance(
                source="erasure_coordinator",
                upstream_ids=[request.id],
            ),
        )
        self._receipts.append(receipt)
        return receipt

    # ── Verification ────────────────────────────────────────────────

    def verify_erased(
        self,
        subject_identity: str,
        layers: Optional[List[ErasureLayer]] = None,
    ) -> List[str]:
        """Re-query every layer and report anything still present.

        Independent of the erase path deliberately: a handler reporting
        success is not evidence that the data is gone.
        """
        residue: List[str] = []
        for layer in (layers or list(self._handlers)):
            handler = self._handlers.get(layer)
            if handler is None:
                continue
            try:
                for record_id, _ in handler.find(subject_identity):
                    residue.append(f"{layer.value}:{record_id}")
            except Exception as exc:
                residue.append(f"{layer.value}:<verification-failed:{exc}>")
        return residue

    # ── Inspection ──────────────────────────────────────────────────

    def tombstones_for(self, request_id: str) -> List[Tombstone]:
        return [t for t in self._tombstones if t.erasure_request_id == request_id]

    @property
    def tombstones(self) -> List[Tombstone]:
        return list(self._tombstones)

    @property
    def receipts(self) -> List[ErasureReceipt]:
        return list(self._receipts)

    def registered_layers(self) -> List[ErasureLayer]:
        return sorted(self._handlers, key=lambda l: l.value)
