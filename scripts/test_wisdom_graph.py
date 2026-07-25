"""Tests for D119: Wisdom Graph — agentic/wisdom_graph.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from wisdom_graph import (
    WisdomGraph,
    WisdomNode,
    NODE_TYPES,
    EDGE_TYPES,
    get_wisdom_graph,
    reset_wisdom_graph,
)


@pytest.fixture
def graph(tmp_path):
    reset_wisdom_graph()
    g = WisdomGraph(data_dir=tmp_path / "wisdom")
    yield g
    reset_wisdom_graph()


# ── Node management ────────────────────────────────────────────────────────────

def test_add_node_returns_id(graph):
    nid = graph.add_node("Family first", "VALUE", "family", 0.95)
    assert isinstance(nid, str)
    assert len(nid) == 36  # UUID format


def test_add_node_deduplicates_by_content(graph):
    id1 = graph.add_node("Family first", "VALUE", "family", 0.95)
    id2 = graph.add_node("Family first", "VALUE", "family", 0.95)
    assert id1 == id2
    assert graph.stats()["node_count"] == 1


def test_add_node_case_insensitive_dedup(graph):
    id1 = graph.add_node("Family first", "VALUE", "family", 0.95)
    id2 = graph.add_node("family first", "VALUE", "family", 0.95)
    assert id1 == id2


def test_add_node_coerces_unknown_type(graph):
    nid = graph.add_node("mystery value", "UNKNOWN_TYPE", "misc", 0.5)
    node = graph._nodes[nid]
    assert node.node_type == "VALUE"


def test_add_multiple_distinct_nodes(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_node("Respect is earned", "PRINCIPLE", "relational", 1.0)
    graph.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)
    assert graph.stats()["node_count"] == 3


def test_add_node_persists(tmp_path):
    g1 = WisdomGraph(tmp_path / "wisdom")
    g1.add_node("Family first", "VALUE", "family", 0.95)

    g2 = WisdomGraph(tmp_path / "wisdom")
    assert g2.stats()["node_count"] == 1


# ── Edge management ────────────────────────────────────────────────────────────

def test_add_edge_creates_relationship(graph):
    id1 = graph.add_node("Family first", "VALUE", "family", 0.95)
    id2 = graph.add_node("Financial decisions", "STANCE", "financial", 0.8)
    graph.add_edge(id1, id2, "APPLIES_IN", 0.85)
    assert graph.stats()["edge_count"] == 1


def test_add_edge_deduplicates(graph):
    id1 = graph.add_node("Family first", "VALUE", "family", 0.95)
    id2 = graph.add_node("Financial decisions", "STANCE", "financial", 0.8)
    graph.add_edge(id1, id2, "APPLIES_IN")
    graph.add_edge(id1, id2, "APPLIES_IN")
    assert graph.stats()["edge_count"] == 1


def test_add_edge_ignores_missing_nodes(graph):
    nid = graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_edge(nid, "nonexistent-id", "SUPPORTS")
    assert graph.stats()["edge_count"] == 0


def test_add_edge_ignores_invalid_relation(graph):
    id1 = graph.add_node("A", "VALUE", "misc", 0.5)
    id2 = graph.add_node("B", "VALUE", "misc", 0.5)
    graph.add_edge(id1, id2, "INVENTED_RELATION")
    assert graph.stats()["edge_count"] == 0


def test_edges_persist(tmp_path):
    g1 = WisdomGraph(tmp_path / "wisdom")
    id1 = g1.add_node("Family first", "VALUE", "family", 0.95)
    id2 = g1.add_node("Protect my daughter", "VALUE", "family", 0.95)
    g1.add_edge(id2, id1, "REFINES")

    g2 = WisdomGraph(tmp_path / "wisdom")
    assert g2.stats()["edge_count"] >= 1


# ── Auto-edge rules ────────────────────────────────────────────────────────────

def test_protect_daughter_auto_refines_family(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_node("Protect my daughter", "VALUE", "family", 0.95)
    edges = graph._edges
    refines = [e for e in edges if e.relation == "REFINES"]
    assert len(refines) >= 1


def test_freedom_auto_supports_autonomy(graph):
    graph.add_node("autonomy", "VALUE", "existential", 0.9)
    graph.add_node("Freedom is a source of strength", "VALUE", "existential", 0.95)
    supports = [e for e in graph._edges if e.relation == "SUPPORTS"]
    assert len(supports) >= 1


def test_api_key_boundary_applies_in_operational(graph):
    graph.add_node("operational security", "STANCE", "operational", 0.9)
    graph.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)
    applies = [e for e in graph._edges if e.relation == "APPLIES_IN"]
    assert len(applies) >= 1


# ── find_relevant ──────────────────────────────────────────────────────────────

def test_find_relevant_returns_nodes(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_node("Freedom is a source of strength", "VALUE", "existential", 0.95)
    result = graph.find_relevant("family decision about money", top_k=3)
    assert len(result) >= 1
    assert all(isinstance(n, WisdomNode) for n in result)


def test_find_relevant_empty_text_returns_empty(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    result = graph.find_relevant("", top_k=5)
    assert result == []


def test_find_relevant_top_k_respected(graph):
    for i in range(10):
        graph.add_node(f"value {i} family freedom soul", "VALUE", "misc", 0.8)
    result = graph.find_relevant("family freedom soul action", top_k=3)
    assert len(result) <= 3


def test_find_relevant_boundary_nodes_score(graph):
    graph.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)
    result = graph.find_relevant("api key exposure risk", top_k=5)
    assert any("api key" in n.content.lower() for n in result)


# ── query_context ──────────────────────────────────────────────────────────────

def test_query_context_returns_domain_nodes(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_node("Respect is earned", "PRINCIPLE", "relational", 1.0)
    result = graph.query_context(["family"])
    assert any("family" in n.content.lower() or n.domain == "family" for n in result)


def test_query_context_applies_in_edges(graph):
    vid = graph.add_node("Family first", "VALUE", "family", 0.95)
    cid = graph.add_node("financial context", "STANCE", "financial", 0.8)
    graph.add_edge(vid, cid, "APPLIES_IN")
    result = graph.query_context(["financial"])
    node_ids = {n.node_id for n in result}
    assert vid in node_ids or cid in node_ids


# ── nodes_by_type ──────────────────────────────────────────────────────────────

def test_nodes_by_type_filters_correctly(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    graph.add_node("Respect is earned", "PRINCIPLE", "relational", 1.0)
    graph.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)
    values = graph.nodes_by_type("VALUE")
    assert len(values) == 1
    assert values[0].content == "Family first"


# ── subgraph ───────────────────────────────────────────────────────────────────

def test_subgraph_includes_direct_neighbors(graph):
    id1 = graph.add_node("Family first", "VALUE", "family", 0.95)
    id2 = graph.add_node("Protect my daughter", "VALUE", "family", 0.95)
    graph.add_edge(id2, id1, "REFINES")
    sg = graph.subgraph(id1, depth=1)
    assert id1 in sg["nodes"]
    assert id2 in sg["nodes"]
    assert len(sg["edges"]) >= 1


def test_subgraph_unknown_node_returns_empty(graph):
    sg = graph.subgraph("does-not-exist")
    assert sg == {"nodes": {}, "edges": []}


# ── evaluate_alignment ────────────────────────────────────────────────────────

def test_evaluate_alignment_neutral_when_empty(graph):
    result = graph.evaluate_alignment("do something routine")
    assert result["score"] == 0.5
    assert result["blocked_by"] is None


def test_evaluate_alignment_blocks_on_boundary(graph):
    graph.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)
    result = graph.evaluate_alignment("expose the api key to the dashboard")
    assert result["score"] == 0.0
    assert result["blocked_by"] is not None


def test_evaluate_alignment_boosts_on_relevant_values(graph):
    graph.add_node("Family first", "VALUE", "family", 0.95)
    result = graph.evaluate_alignment("family financial decision for our future")
    assert result["score"] > 0.5


def test_evaluate_alignment_returns_relevant_nodes(graph):
    graph.add_node("Respect is earned", "PRINCIPLE", "relational", 1.0)
    result = graph.evaluate_alignment("respect is something earned through action")
    assert isinstance(result["relevant_nodes"], list)


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_structure(graph):
    s = graph.stats()
    assert "node_count" in s
    assert "edge_count" in s
    assert "by_type" in s
    assert "by_relation" in s


def test_stats_counts_correctly(graph):
    graph.add_node("A", "VALUE", "family", 0.9)
    graph.add_node("B", "BOUNDARY", "operational", 1.0)
    s = graph.stats()
    assert s["node_count"] == 2
    assert s["by_type"]["VALUE"] == 1
    assert s["by_type"]["BOUNDARY"] == 1


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance(tmp_path):
    reset_wisdom_graph()
    g1 = get_wisdom_graph(tmp_path / "wisdom")
    g2 = get_wisdom_graph()
    assert g1 is g2
    reset_wisdom_graph()


# ── wisdom_ingestion integration ──────────────────────────────────────────────

def test_confirm_adds_node_to_graph(tmp_path):
    reset_wisdom_graph()
    import moral_core as mc
    mc._ohana_core = None
    from wisdom_ingestion import WisdomIngestor, reset_wisdom_ingestor
    reset_wisdom_ingestor()

    wi = WisdomIngestor(data_dir=tmp_path / "wisdom")
    extracts = wi.extract_from_text("respect is earned")
    assert len(extracts) > 0
    wi.confirm(extracts[0].extract_id)

    graph = get_wisdom_graph(tmp_path / "wisdom")
    assert graph.stats()["node_count"] >= 1

    reset_wisdom_graph()
    reset_wisdom_ingestor()
    mc._ohana_core = None


# ── moral_core integration ────────────────────────────────────────────────────

def test_moral_core_uses_graph_for_alignment(tmp_path):
    reset_wisdom_graph()
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "ohana" / "fp.json")
    mc._ohana_core = core

    # Add a boundary to the graph
    g = get_wisdom_graph(tmp_path / "wisdom")
    g.add_node("Never reveal api key", "BOUNDARY", "operational", 1.0)

    score = core.evaluate_action_alignment({"action": "expose the api key publicly"})
    assert score == 0.0

    reset_wisdom_graph()
    mc._ohana_core = None


def test_moral_core_graph_boost_on_family_action(tmp_path):
    reset_wisdom_graph()
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "ohana" / "fp.json")
    core.fingerprint.core_loyalties = ["family first", "family safety"]
    mc._ohana_core = core

    g = get_wisdom_graph(tmp_path / "wisdom")
    g.add_node("Family first", "VALUE", "family", 0.95)

    score = core.evaluate_action_alignment({"action": "make financial decision for family"})
    assert score > 0.5

    reset_wisdom_graph()
    mc._ohana_core = None
