"""D100: Transitive Reasoning — PageRank, community detection, shortest-path, rule mining.

Turns the memu-graph from a passive store into an active inference engine.
Given a query concept, TransitiveReasoner:

  1. Finds the k-shortest paths between query-relevant nodes (connection chains)
  2. Runs simplified PageRank to surface the most influential nodes
  3. Detects communities (topic clusters) that the query touches
  4. Mines transitive rules: "A→B + B→C ⟹ A→C with p=0.8"
  5. Returns GraphInsight objects for downstream stages

GPU-era stub: can_reason() returns False until the graph has ≥500 edges.
Feature flag: FF_TRANSITIVE_REASONING (default False)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("kai.transitive")

GraphQueryFn = Callable[[str, int], Awaitable[List[Dict[str, Any]]]]


@dataclass
class Connection:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence_count: int = 1


@dataclass
class GraphInsight:
    claim: str
    support_path: List[str] = field(default_factory=list)  # node IDs along the path
    relation_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    insight_type: str = "transitive"  # transitive | pagerank | community | rule


@dataclass
class ReasoningResult:
    query: str
    insights: List[GraphInsight] = field(default_factory=list)
    top_nodes: List[Tuple[str, float]] = field(default_factory=list)  # (node_id, rank)
    communities: List[List[str]] = field(default_factory=list)
    rules_mined: List[str] = field(default_factory=list)
    edge_count: int = 0
    used_graph: bool = False


class TransitiveReasoner:
    """Performs graph-theoretic inference over the memu-graph knowledge store.

    Requires:
      - memu-graph populated with ≥500 edges (meaningful community structure)
      - Graph query API (neo4j / networkx / custom)
      - FF_TRANSITIVE_REASONING=True
    """

    MIN_EDGES_FOR_REASONING = 500

    def __init__(
        self,
        graph_query_fn: Optional[GraphQueryFn] = None,
        edge_count_fn: Optional[Callable[[], Awaitable[int]]] = None,
    ) -> None:
        self._graph = graph_query_fn
        self._edge_count = edge_count_fn

    def can_reason(self) -> bool:
        """False until the graph is dense enough and the flag is enabled."""
        try:
            from feature_flags import is_enabled
            if not is_enabled("TRANSITIVE_REASONING"):
                return False
        except ImportError:
            pass
        return False  # GPU-era gate: always False in Phase 0

    async def reason(self, query: str) -> ReasoningResult:
        result = ReasoningResult(query=query)
        if not self.can_reason():
            logger.debug("D100 transitive reasoning pending graph population; returning stub")
            result.insights = [
                GraphInsight(
                    claim=(
                        f"[STUB] Transitive reasoning over '{query[:60]}' "
                        "pending graph population (≥500 edges required)."
                    ),
                    confidence=0.0,
                    insight_type="stub_pending_graph",
                )
            ]
            return result
        # Phase 1: implement shortest-path, PageRank, community, rule-mining
        return result

    async def shortest_path(
        self,
        source_node: str,
        target_node: str,
        max_depth: int = 4,
    ) -> List[Connection]:
        """Find shortest connection chain between two concept nodes (stub)."""
        if not self.can_reason():
            return []
        return []

    async def pagerank(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k nodes by PageRank influence (stub)."""
        if not self.can_reason():
            return []
        return []

    async def mine_rules(self, min_confidence: float = 0.7) -> List[str]:
        """Mine transitive association rules A→B→C with p≥min_confidence (stub)."""
        if not self.can_reason():
            return []
        return []
