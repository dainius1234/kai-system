"""Capability-specific signed release bundles.

A release bundle authorises one capability at one autonomy level for one
code revision.  The signature covers the payload, so changing the
capability, the level, the revision or the expiry after signing
invalidates it.

Binding to a code revision is the point: shipping new code does not
inherit the previous revision's release authority.  A rebuild must be
signed again.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import AutonomyLevel, ReleaseBundle


class ReleaseBundleError(Exception):
    pass


def _payload(
    capability: str,
    code_revision: str,
    autonomy_level: AutonomyLevel,
    domains: List[str],
    valid_until: datetime,
) -> str:
    return json.dumps(
        {
            "capability": capability,
            "code_revision": code_revision,
            "autonomy_level": int(autonomy_level),
            "domains": sorted(domains),
            "valid_until": valid_until.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class ReleaseBundleService:
    """Signs and verifies capability release bundles.

    Parameters:
        principal: owning principal
        signing_key: HMAC key; never leaves this service
    """

    def __init__(self, principal: Principal, signing_key: bytes) -> None:
        if not signing_key:
            raise ReleaseBundleError("signing key must not be empty")
        self._principal = principal
        self._key = signing_key
        self._bundles: Dict[str, ReleaseBundle] = {}

    def sign(
        self,
        capability: str,
        code_revision: str,
        autonomy_level: AutonomyLevel,
        domains: List[str],
        signed_by: str,
        valid_for_seconds: int = 86400,
    ) -> ReleaseBundle:
        if not capability or not capability.strip():
            raise ReleaseBundleError("bundle must name a capability")
        if not code_revision or not code_revision.strip():
            raise ReleaseBundleError("bundle must bind to a code revision")
        if not signed_by or not signed_by.strip():
            raise ReleaseBundleError("bundle cannot be signed anonymously")
        if not domains:
            raise ReleaseBundleError("bundle must name at least one domain")
        if valid_for_seconds < 1:
            raise ReleaseBundleError("bundle validity must be positive")

        now = datetime.now(timezone.utc)
        valid_until = now + timedelta(seconds=valid_for_seconds)
        payload = _payload(
            capability, code_revision, autonomy_level, domains, valid_until
        )
        signature = hmac.new(
            self._key, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        bundle = ReleaseBundle(
            capability=capability,
            code_revision=code_revision,
            autonomy_level=autonomy_level,
            domains=sorted(domains),
            signature=signature,
            signed_by=signed_by,
            signed_at=now,
            valid_until=valid_until,
            principal=self._principal,
            purpose="release_bundle",
            provenance=Provenance(source=f"release_service:{signed_by}"),
        )
        self._bundles[bundle.id] = bundle
        return bundle

    def verify(
        self,
        bundle: ReleaseBundle,
        capability: str,
        code_revision: str,
        domain: str,
    ) -> Tuple[bool, str]:
        """Whether this bundle authorises this capability on this revision."""
        if bundle.revoked:
            return False, "bundle revoked"

        now = datetime.now(timezone.utc)
        if now >= bundle.valid_until:
            return False, "bundle expired"

        expected = hmac.new(
            self._key,
            _payload(
                bundle.capability,
                bundle.code_revision,
                bundle.autonomy_level,
                bundle.domains,
                bundle.valid_until,
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, bundle.signature):
            return False, "signature invalid — bundle was modified after signing"

        if bundle.capability != capability:
            return False, (
                f"capability mismatch: bundle is for '{bundle.capability}', "
                f"not '{capability}'"
            )

        if bundle.code_revision != code_revision:
            return False, (
                f"revision mismatch: bundle is for '{bundle.code_revision}', "
                f"not '{code_revision}'"
            )

        if domain not in bundle.domains:
            return False, f"bundle does not cover domain '{domain}'"

        return True, "valid"

    def revoke(self, bundle_id: str, reason: str) -> ReleaseBundle:
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            raise ReleaseBundleError(f"bundle not found: {bundle_id}")
        bundle.revoked = True
        bundle.digest = bundle._make_digest()
        return bundle

    def get(self, bundle_id: str) -> Optional[ReleaseBundle]:
        return self._bundles.get(bundle_id)

    def list_bundles(
        self,
        capability: Optional[str] = None,
        code_revision: Optional[str] = None,
    ) -> List[ReleaseBundle]:
        results = list(self._bundles.values())
        if capability is not None:
            results = [b for b in results if b.capability == capability]
        if code_revision is not None:
            results = [b for b in results if b.code_revision == code_revision]
        return sorted(results, key=lambda b: b.signed_at)
