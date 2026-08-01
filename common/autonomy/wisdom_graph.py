"""Wisdom graph with explicit lineage and contradiction tracking.

Every node records what it was derived from and which evidence backs it.
A node backed only by non-qualifying evidence carries no confidence,
which stops a chain of self-generated inferences from accumulating
authority through sheer repetition.

Contradictions are recorded rather than resolved.  Two nodes that
disagree both stay in the graph, linked, until superseding evidence
arrives.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import WisdomNode
from common.autonomy.evidence_service import EvidenceService


class WisdomError(Exception):
    pass


class WisdomGraph:
    """Lineage-tracked knowledge graph over graded evidence."""

    def __init__(self, principal: Principal, evidence: EvidenceService) -> None:
        self._principal = principal
        self._evidence = evidence
        self._nodes: Dict[str, WisdomNode] = {}

    def add(
        self,
        statement: str,
        domain: str,
        evidence_ids: Optional[List[str]] = None,
        derived_from: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> WisdomNode:
        if not statement or not statement.strip():
            raise WisdomError("wisdom node must carry a statement")
        if not 0.0 <= confidence <= 1.0:
            raise WisdomError(f"confidence out of range: {confidence}")

        ev_ids = list(evidence_ids or [])
        for ev_id in ev_ids:
            if self._evidence.get(ev_id) is None:
                raise WisdomError(f"unknown evidence: {ev_id}")

        for parent_id in derived_from or []:
            if parent_id not in self._nodes:
                raise WisdomError(f"unknown parent node: {parent_id}")

        # A node with no qualifying evidence behind it holds no confidence,
        # however confident the caller claims to be.
        effective_confidence = (
            confidence if self._has_qualifying_support(ev_ids, derived_from or [])
            else 0.0
        )

        node = WisdomNode(
            statement=statement,
            domain=domain,
            derived_from=list(derived_from or []),
            evidence_ids=ev_ids,
            confidence=effective_confidence,
            principal=self._principal,
            purpose="wisdom",
            provenance=Provenance(
                source="wisdom_graph",
                upstream_ids=ev_ids + list(derived_from or []),
            ),
        )
        self._nodes[node.id] = node
        return node

    def _has_qualifying_support(
        self,
        evidence_ids: List[str],
        derived_from: List[str],
    ) -> bool:
        for ev_id in evidence_ids:
            record = self._evidence.get(ev_id)
            if record is not None and record.grade.qualifies():
                return True

        # Inherit support transitively, but only through parents that
        # themselves hold confidence.
        for parent_id in derived_from:
            parent = self._nodes.get(parent_id)
            if parent is not None and parent.confidence > 0.0:
                return True

        return False

    def record_contradiction(self, node_a_id: str, node_b_id: str) -> None:
        node_a = self._nodes.get(node_a_id)
        node_b = self._nodes.get(node_b_id)
        if node_a is None:
            raise WisdomError(f"unknown node: {node_a_id}")
        if node_b is None:
            raise WisdomError(f"unknown node: {node_b_id}")
        if node_a_id == node_b_id:
            raise WisdomError("a node cannot contradict itself")

        if node_b_id not in node_a.contradicts:
            node_a.contradicts.append(node_b_id)
            node_a.digest = node_a._make_digest()
        if node_a_id not in node_b.contradicts:
            node_b.contradicts.append(node_a_id)
            node_b.digest = node_b._make_digest()

    def supersede(self, old_node_id: str, new_node_id: str) -> None:
        old = self._nodes.get(old_node_id)
        new = self._nodes.get(new_node_id)
        if old is None:
            raise WisdomError(f"unknown node: {old_node_id}")
        if new is None:
            raise WisdomError(f"unknown node: {new_node_id}")

        old.superseded_by = new_node_id
        old.digest = old._make_digest()

    def lineage(self, node_id: str) -> List[WisdomNode]:
        """All ancestors of a node, nearest first, cycle-safe."""
        node = self._nodes.get(node_id)
        if node is None:
            raise WisdomError(f"unknown node: {node_id}")

        seen: Set[str] = {node_id}
        chain: List[WisdomNode] = []
        frontier = list(node.derived_from)
        while frontier:
            current_id = frontier.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)
            parent = self._nodes.get(current_id)
            if parent is not None:
                chain.append(parent)
                frontier.extend(parent.derived_from)
        return chain

    def get(self, node_id: str) -> Optional[WisdomNode]:
        return self._nodes.get(node_id)

    def contradictions(self) -> List[tuple[str, str]]:
        pairs: Set[tuple[str, str]] = set()
        for node in self._nodes.values():
            for other_id in node.contradicts:
                pairs.add(tuple(sorted((node.id, other_id))))
        return sorted(pairs)

    def active_nodes(self, domain: Optional[str] = None) -> List[WisdomNode]:
        results = [n for n in self._nodes.values() if n.superseded_by is None]
        if domain is not None:
            results = [n for n in results if n.domain == domain]
        return sorted(results, key=lambda n: n.created_at)
