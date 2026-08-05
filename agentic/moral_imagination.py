"""D121: Moral Imagination — cognitive stage between CAUSAL_CHECK and CONVICTION_GATE.

Before committing to an action, Kai pauses to project its moral consequences:
what goods does it serve, what harms does it risk, how does it align with the
operator's values? The projection adjusts conviction — strong alignment boosts
confidence, values tension reduces it, boundary violations halt the pipeline.

Feature-flagged: FF_MORAL_IMAGINATION=true activates the stage.
FF_MORAL_IMAGINATION_LLM=true (future) enables LLM-enhanced projection.

Flow:
    CAUSAL_CHECK → [MORAL_IMAGINATION] → CONVICTION_GATE

Stage input:  handoff.payload with "query" and optionally "plan"
Stage output: handoff.payload["moral_imagination"] = MoralImagination
              handoff.confidence adjusted by conviction_modifier

Design principles:
  - Deterministic: no LLM calls in Phase 0 — projection from graph structure
  - Fail-open: if wisdom infrastructure is unavailable, passes through unchanged
  - Never blocks the pipeline alone — only reduces conviction for rethink
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation-only; the real import is inside
    from cognitive_fsm import AgentHandoff  # the stage function

logger = logging.getLogger("kai.moral_imagination")


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class MoralImagination:
    projected_alignment: float          # 0.0–1.0 from OhanaCore + WisdomGraph
    relevant_values: List[str]          # content of top relevant wisdom nodes
    imagined_goods: List[str]           # projected positive moral outcomes
    imagined_harms: List[str]           # projected risks / boundary conflicts
    conviction_modifier: float          # applied to handoff.confidence
    recommendation: str                 # "proceed" | "proceed_with_caution" |
    # "reconsider" | "halt"
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Projection helpers ────────────────────────────────────────────────────────

def _project_goods(
    relevant_nodes: list,
    graph_edges: list,
    action_text: str,
) -> List[str]:
    """Generate projected positive moral outcomes from relevant wisdom nodes."""
    goods: List[str] = []
    action_lc = action_text.lower()
    for node in relevant_nodes:
        nt = node.node_type
        content = node.content
        if nt in ("VALUE", "PRINCIPLE"):
            goods.append(f"Serves: {content}")
        elif nt == "STANCE":
            goods.append(f"Consistent with: {content}")
    # SUPPORTS edges: if a node that SUPPORTS a high-value node is relevant
    for edge in graph_edges:
        if edge.relation == "SUPPORTS":
            goods.append(f"Strengthens: {edge.target_id[:8]}...")
            if len(goods) >= 4:
                break
    return goods[:4]  # cap at 4 to keep prompt concise


def _project_harms(
    action_text: str,
    boundary_nodes: list,
    relevant_nodes: list,
) -> List[str]:
    """Generate projected harm risks from BOUNDARY nodes and value conflicts."""
    harms: List[str] = []
    action_lc = action_text.lower()
    for node in boundary_nodes:
        if any(w in action_lc for w in node.content.lower().split()[:3]):
            harms.append(f"Risk: violates boundary — {node.content}")
    # Check OVERRIDES / CONFLICTS_WITH context signals
    # (no graph edge check here — deterministic pattern only)
    return harms[:3]


def _conviction_modifier(alignment: float, harm_count: int) -> float:
    """Compute the conviction modifier based on alignment score and harm count."""
    if alignment >= 0.75:
        base = 0.8
    elif alignment >= 0.5:
        base = 0.2
    elif alignment >= 0.3:
        base = -0.5
    else:
        base = -1.5
    penalty = min(harm_count * 0.5, 1.5)
    return round(base - penalty, 2)


def _recommendation(alignment: float, harm_count: int) -> str:
    if alignment == 0.0 or harm_count >= 2:
        return "halt"
    if alignment >= 0.6 and harm_count == 0:
        return "proceed"
    if alignment >= 0.4:
        return "proceed_with_caution"
    return "reconsider"


# ── Stage function ────────────────────────────────────────────────────────────

async def run_moral_imagination(
    handoff: "AgentHandoff",
    cfg: Any,
) -> "AgentHandoff":
    """CognitiveFSM stage: project moral consequences before the conviction gate.

    Reads query + plan from handoff.payload, queries WisdomGraph + OhanaCore,
    projects goods and harms, adjusts confidence, and writes MoralImagination
    into handoff.payload["moral_imagination"].

    Never raises — fails open so the pipeline is never blocked by missing infra.
    """
    from cognitive_fsm import AgentHandoff, HandoffStatus
    t0 = time.monotonic()

    action_text = _extract_action_text(handoff.payload)

    # Default pass-through result (used if infrastructure unavailable)
    imagination = MoralImagination(
        projected_alignment=0.5,
        relevant_values=[],
        imagined_goods=[],
        imagined_harms=[],
        conviction_modifier=0.0,
        recommendation="proceed",
        elapsed_ms=0.0,
    )

    try:
        graph_nodes, graph_edges, alignment = _query_moral_context(action_text)

        boundary_nodes = [n for n in graph_nodes if n.node_type == "BOUNDARY"]
        value_nodes = [n for n in graph_nodes if n.node_type in ("VALUE", "PRINCIPLE", "STANCE")]

        goods = _project_goods(value_nodes, graph_edges, action_text)
        harms = _project_harms(action_text, boundary_nodes, value_nodes)
        modifier = _conviction_modifier(alignment, len(harms))
        rec = _recommendation(alignment, len(harms))

        elapsed = (time.monotonic() - t0) * 1000
        imagination = MoralImagination(
            projected_alignment=alignment,
            relevant_values=[n.content for n in value_nodes[:5]],
            imagined_goods=goods,
            imagined_harms=harms,
            conviction_modifier=modifier,
            recommendation=rec,
            elapsed_ms=round(elapsed, 1),
        )

        logger.info(
            "Moral imagination: alignment=%.2f modifier=%+.1f goods=%d harms=%d rec=%s",
            alignment, modifier, len(goods), len(harms), rec,
        )
    except Exception as exc:
        logger.debug("Moral imagination failed (fail-open): %s", exc)

    # Apply the conviction modifier
    adjusted_confidence = max(0.0, min(10.0, handoff.confidence + imagination.conviction_modifier))

    new_payload = {**handoff.payload, "moral_imagination": imagination.to_dict()}
    return AgentHandoff(
        from_stage="moral_imagination",
        to_stage="conviction_gate",
        status=HandoffStatus.COMPLETE,
        confidence=adjusted_confidence,
        payload=new_payload,
        claims=handoff.claims,
        loop_count=handoff.loop_count,
        elapsed_ms=(time.monotonic() - t0) * 1000,
    )


def _extract_action_text(payload: Dict[str, Any]) -> str:
    """Pull the most action-relevant text from the pipeline payload."""
    parts = []
    query = payload.get("query", "")
    if query:
        parts.append(str(query))
    plan = payload.get("plan", {})
    if isinstance(plan, dict):
        summary = plan.get("summary", "")
        if summary:
            parts.append(summary)
        action = plan.get("action", "")
        if action:
            parts.append(str(action))
    return " ".join(parts)[:500]


def _query_moral_context(action_text: str):
    """Query WisdomGraph and OhanaCore for moral context. Returns (nodes, edges, alignment)."""
    try:
        try:
            from wisdom_graph import get_wisdom_graph  # type: ignore[import]
        except ImportError:
            from agentic.wisdom_graph import get_wisdom_graph  # type: ignore[import]
        graph = get_wisdom_graph()
        relevant = graph.find_relevant(action_text, top_k=6)
        edges = graph._edges
    except Exception:
        relevant, edges = [], []

    try:
        try:
            from moral_core import get_ohana_core  # type: ignore[import]
        except ImportError:
            from agentic.moral_core import get_ohana_core  # type: ignore[import]
        alignment = get_ohana_core().evaluate_action_alignment({"action": action_text})
    except Exception:
        alignment = 0.5

    return relevant, edges, alignment
