"""D119: Wisdom Graph — relational value map for the Ohana Core.

Transforms Kai's flat list of values into a structured graph where values,
principles, boundaries, and stances can relate to each other:

  APPLIES_IN    — "Family first" applies in financial decisions
  REFINES       — "Protect my daughter" refines "Family first"
  OVERRIDES     — "Never reveal api key" overrides convenience
  CONFLICTS_WITH — two values that may tension against each other
  SUPPORTS      — "Freedom is a source of strength" supports "autonomy"

File-backed by default (data/wisdom/graph.json). Cognee/Kuzu can plug in
as a backend when available — same interface, richer query capability.

Public API:
    add_node(content, node_type, domain, confidence, extract_id) → str
    add_edge(source_id, target_id, relation, weight) → None
    query_context(domain_keywords) → List[WisdomNode]
    find_relevant(action_text, top_k) → List[WisdomNode]
    nodes_by_type(node_type) → List[WisdomNode]
    subgraph(node_id, depth) → Dict
    stats() → Dict
    get_wisdom_graph(data_dir) → WisdomGraph
    reset_wisdom_graph() → None
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("kai.wisdom_graph")

_GRAPH_FILENAME = "graph.json"

# ── Node and edge types ───────────────────────────────────────────────────────

NODE_TYPES = {"VALUE", "PRINCIPLE", "BOUNDARY", "STANCE"}
EDGE_TYPES = {"APPLIES_IN", "REFINES", "OVERRIDES", "CONFLICTS_WITH", "SUPPORTS"}

# ── Auto-edge rules: when a new node matches source_pattern, automatically
#    create an edge to any existing node matching target_pattern.
#    Tuples: (source_keyword, relation, target_keyword, weight)
_AUTO_EDGE_RULES: List[tuple] = [
    ("protect my daughter", "REFINES", "family", 0.9),
    ("daughter", "REFINES", "family first", 0.9),
    ("family first", "APPLIES_IN", "financial", 0.85),
    ("family first", "APPLIES_IN", "daily", 0.8),
    ("family safety", "SUPPORTS", "family first", 0.9),
    ("freedom", "SUPPORTS", "autonomy", 0.85),
    ("soul", "OVERRIDES", "convenience", 0.7),
    ("kai is for soul", "OVERRIDES", "convenience", 0.8),
    ("never reveal api key", "APPLIES_IN", "operational", 0.95),
    ("api key", "APPLIES_IN", "operational", 0.9),
    ("respect is earned", "REFINES", "authority", 0.8),
    ("respect", "APPLIES_IN", "relational", 0.8),
    ("survival", "SUPPORTS", "family safety", 0.85),
    ("autonomy", "SUPPORTS", "freedom", 0.8),
]


@dataclass
class WisdomNode:
    node_id: str
    node_type: str          # VALUE | PRINCIPLE | BOUNDARY | STANCE
    domain: str
    content: str
    confidence: float
    extract_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def word_set(self) -> Set[str]:
        return set(re.findall(r"\w+", self.content.lower()))


@dataclass
class WisdomEdge:
    source_id: str
    target_id: str
    relation: str           # APPLIES_IN | REFINES | OVERRIDES | CONFLICTS_WITH | SUPPORTS
    weight: float = 0.8
    created_at: float = field(default_factory=time.time)


class WisdomGraph:
    """Relational graph of Kai's values, principles, boundaries, and stances."""

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / _GRAPH_FILENAME
        self._nodes: Dict[str, WisdomNode] = {}
        self._edges: List[WisdomEdge] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            for nid, nd in raw.get("nodes", {}).items():
                self._nodes[nid] = WisdomNode(**nd)
            for ed in raw.get("edges", []):
                self._edges.append(WisdomEdge(**ed))
        except Exception as exc:
            logger.warning("Wisdom graph load failed (starting fresh): %s", exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: asdict(n) for nid, n in self._nodes.items()},
            "edges": [asdict(e) for e in self._edges],
        }
        self._path.write_text(json.dumps(data, indent=2))

    # ── Node management ───────────────────────────────────────────────────────

    def add_node(
        self,
        content: str,
        node_type: str,
        domain: str,
        confidence: float,
        extract_id: Optional[str] = None,
    ) -> str:
        """Add a wisdom node. Returns node_id. Deduplicates by content."""
        content_lc = content.lower().strip()
        for nid, node in self._nodes.items():
            if node.content.lower().strip() == content_lc:
                return nid  # already exists

        if node_type not in NODE_TYPES:
            node_type = "VALUE"

        node_id = str(uuid.uuid4())
        node = WisdomNode(
            node_id=node_id,
            node_type=node_type,
            domain=domain,
            content=content,
            confidence=confidence,
            extract_id=extract_id,
        )
        self._nodes[node_id] = node
        self._apply_auto_edges(node)
        self._save()
        logger.debug("Wisdom graph: added %s node '%s'", node_type, content[:60])
        return node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 0.8,
    ) -> None:
        """Add a directed edge. No-op if source or target don't exist."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return
        if relation not in EDGE_TYPES:
            return
        # Deduplicate: same source+target+relation
        for e in self._edges:
            if e.source_id == source_id and e.target_id == target_id and e.relation == relation:
                return
        self._edges.append(WisdomEdge(source_id, target_id, relation, weight))
        self._save()

    def _apply_auto_edges(self, new_node: WisdomNode) -> None:
        """Check auto-edge rules against the new node and existing nodes."""
        new_text = new_node.content.lower()
        for src_kw, relation, tgt_kw, weight in _AUTO_EDGE_RULES:
            if src_kw in new_text:
                # new node is the SOURCE — find existing nodes matching target
                for nid, existing in self._nodes.items():
                    if nid == new_node.node_id:
                        continue
                    if tgt_kw in existing.content.lower():
                        self.add_edge(new_node.node_id, nid, relation, weight)
            if tgt_kw in new_text:
                # new node is the TARGET — find existing nodes matching source
                for nid, existing in self._nodes.items():
                    if nid == new_node.node_id:
                        continue
                    if src_kw in existing.content.lower():
                        self.add_edge(nid, new_node.node_id, relation, weight)

    # ── Query API ─────────────────────────────────────────────────────────────

    def find_relevant(self, action_text: str, top_k: int = 5) -> List[WisdomNode]:
        """Return up to top_k nodes most relevant to action_text.

        Scores each node by:
          base = word overlap (Jaccard) between action_text and node content
          boost = +0.2 per APPLIES_IN edge whose target domain keyword appears in action_text
          weight = confidence × (base + boost)
        """
        action_words = set(re.findall(r"\w+", action_text.lower()))
        if not action_words:
            return []

        # Build map: node_id → list of APPLIES_IN target contents
        applies_in_targets: Dict[str, List[str]] = {}
        for edge in self._edges:
            if edge.relation == "APPLIES_IN" and edge.source_id in self._nodes:
                target = self._nodes.get(edge.target_id)
                target_text = target.content.lower() if target else ""
                applies_in_targets.setdefault(edge.source_id, []).append(target_text)

        scored: List[tuple] = []
        for nid, node in self._nodes.items():
            node_words = node.word_set()
            if not node_words:
                continue
            overlap = len(action_words & node_words) / len(action_words | node_words)
            boost = sum(
                0.2 for ctx in applies_in_targets.get(nid, [])
                if any(w in action_words for w in ctx.split()[:3])
            )
            score = node.confidence * min(1.0, overlap + boost)
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:top_k]]

    def query_context(self, domain_keywords: List[str]) -> List[WisdomNode]:
        """Return nodes that APPLY_IN any of the given domain keywords."""
        kw_set = {k.lower() for k in domain_keywords}
        relevant_sources: Set[str] = set()
        for edge in self._edges:
            if edge.relation == "APPLIES_IN":
                target = self._nodes.get(edge.target_id)
                if target and any(kw in target.content.lower() for kw in kw_set):
                    relevant_sources.add(edge.source_id)
        # Also include nodes whose domain directly matches
        for nid, node in self._nodes.items():
            if any(kw in node.domain.lower() for kw in kw_set):
                relevant_sources.add(nid)
        return [self._nodes[nid] for nid in relevant_sources if nid in self._nodes]

    def nodes_by_type(self, node_type: str) -> List[WisdomNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def subgraph(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Return all nodes and edges reachable from node_id within depth hops."""
        if node_id not in self._nodes:
            return {"nodes": {}, "edges": []}
        visited: Set[str] = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                visited.add(nid)
                for edge in self._edges:
                    if edge.source_id == nid and edge.target_id not in visited:
                        next_frontier.add(edge.target_id)
                    if edge.target_id == nid and edge.source_id not in visited:
                        next_frontier.add(edge.source_id)
            frontier = next_frontier - visited
        visited.update(frontier)
        edges = [
            asdict(e) for e in self._edges
            if e.source_id in visited and e.target_id in visited
        ]
        return {
            "nodes": {nid: asdict(self._nodes[nid]) for nid in visited if nid in self._nodes},
            "edges": edges,
        }

    def evaluate_alignment(self, action_text: str) -> Dict[str, Any]:
        """Graph-based alignment score for action_text.

        Returns:
          score: float 0.0–1.0
          blocked_by: str | None (boundary content if blocked)
          relevant_nodes: List[str] (content of top relevant nodes)
        """
        # Hard block on BOUNDARY nodes
        for node in self.nodes_by_type("BOUNDARY"):
            if any(w in action_text.lower() for w in node.word_set()):
                return {
                    "score": 0.0,
                    "blocked_by": node.content,
                    "relevant_nodes": [],
                }

        relevant = self.find_relevant(action_text, top_k=5)
        if not relevant:
            return {"score": 0.5, "blocked_by": None, "relevant_nodes": []}

        weights = [n.confidence for n in relevant]
        score = 0.5 + min(0.5, sum(weights) / (len(weights) * 2))
        return {
            "score": round(score, 3),
            "blocked_by": None,
            "relevant_nodes": [n.content for n in relevant],
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for node in self._nodes.values():
            by_type[node.node_type] = by_type.get(node.node_type, 0) + 1
        by_relation: Dict[str, int] = {}
        for edge in self._edges:
            by_relation[edge.relation] = by_relation.get(edge.relation, 0) + 1
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "by_type": by_type,
            "by_relation": by_relation,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_wisdom_graph: Optional[WisdomGraph] = None
_DEFAULT_DATA_DIR = Path("data/wisdom")


def get_wisdom_graph(data_dir: Optional[Path] = None) -> WisdomGraph:
    global _wisdom_graph
    if _wisdom_graph is None:
        _wisdom_graph = WisdomGraph(data_dir or _DEFAULT_DATA_DIR)
    return _wisdom_graph


def reset_wisdom_graph() -> None:
    global _wisdom_graph
    _wisdom_graph = None
