"""Tests for D102: Global Workspace Consciousness.

Covers:
  - WorkspaceBid dataclass: fields and defaults
  - ConsciousMoment dataclass: fields and defaults
  - GlobalWorkspace: submit_bid, select_winner, broadcast, subscribe/unsubscribe,
    get_stream, get_latest_moment, can_operate(), progress(), stream_length
  - Singleton: get_global_workspace returns same instance
"""
import sys
import time
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _import_gw():
    sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
    import global_workspace
    return global_workspace


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

def test_global_workspace_importable():
    gw = _import_gw()
    assert hasattr(gw, "WorkspaceBid")
    assert hasattr(gw, "ConsciousMoment")
    assert hasattr(gw, "GlobalWorkspace")
    assert hasattr(gw, "get_global_workspace")


# ---------------------------------------------------------------------------
# WorkspaceBid
# ---------------------------------------------------------------------------

def test_workspace_bid_required_fields():
    gw = _import_gw()
    bid = gw.WorkspaceBid(
        module="perception",
        content="AQ spike detected — 142 µg/m³",
        urgency=0.9,
        relevance=0.7,
        surprise=0.8,
        confidence=0.95,
        emotional_salience=0.6,
    )
    assert bid.module == "perception"
    assert bid.urgency == 0.9
    assert bid.surprise == 0.8


def test_workspace_bid_timestamp_auto():
    gw = _import_gw()
    before = time.time()
    bid = gw.WorkspaceBid(
        module="hypothesis_engine",
        content="Gap detected: sleep quality untracked",
        urgency=0.3,
        relevance=0.5,
        surprise=0.2,
        confidence=0.7,
        emotional_salience=0.1,
    )
    assert bid.timestamp >= before


def test_workspace_bid_zero_scores_valid():
    gw = _import_gw()
    bid = gw.WorkspaceBid(
        module="causal_model",
        content="Simulation ready",
        urgency=0.0,
        relevance=0.0,
        surprise=0.0,
        confidence=0.0,
        emotional_salience=0.0,
    )
    assert bid.confidence == 0.0


# ---------------------------------------------------------------------------
# ConsciousMoment
# ---------------------------------------------------------------------------

def test_conscious_moment_required_fields():
    gw = _import_gw()
    import uuid
    moment = gw.ConsciousMoment(
        timestamp=time.time(),
        content="I notice the AQ is unusually high.",
        source_module="perception",
        salience_score=0.92,
        broadcast_id=str(uuid.uuid4()),
    )
    assert moment.source_module == "perception"
    assert moment.salience_score == 0.92


def test_conscious_moment_defaults():
    gw = _import_gw()
    import uuid
    moment = gw.ConsciousMoment(
        timestamp=time.time(),
        content="Testing defaults",
        source_module="test",
        salience_score=0.5,
        broadcast_id=str(uuid.uuid4()),
    )
    assert moment.context == {}
    assert moment.emotional_valence == 0.0


def test_conscious_moment_negative_valence():
    gw = _import_gw()
    import uuid
    moment = gw.ConsciousMoment(
        timestamp=time.time(),
        content="High-risk trading signal detected.",
        source_module="causal_model",
        salience_score=0.85,
        broadcast_id=str(uuid.uuid4()),
        emotional_valence=-0.7,
    )
    assert moment.emotional_valence == -0.7


# ---------------------------------------------------------------------------
# GlobalWorkspace — Phase 0 stub behaviour
# ---------------------------------------------------------------------------

def test_global_workspace_can_operate_false():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.can_operate() is False


def test_global_workspace_select_winner_none():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.select_winner() is None


def test_global_workspace_get_stream_empty():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.get_stream() == []
    assert ws.get_stream(limit=100) == []


def test_global_workspace_get_latest_moment_none():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.get_latest_moment() is None


def test_global_workspace_stream_length_zero():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.stream_length() == 0


def test_global_workspace_submit_bid_no_exception():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    bid = gw.WorkspaceBid(
        module="memory",
        content="Reminds me of last Tuesday",
        urgency=0.2,
        relevance=0.6,
        surprise=0.1,
        confidence=0.8,
        emotional_salience=0.3,
    )
    ws.submit_bid(bid)  # should not raise


def test_global_workspace_broadcast_no_exception():
    gw = _import_gw()
    import uuid
    ws = gw.GlobalWorkspace()
    moment = gw.ConsciousMoment(
        timestamp=time.time(),
        content="Broadcast test moment",
        source_module="test",
        salience_score=0.5,
        broadcast_id=str(uuid.uuid4()),
    )
    ws.broadcast(moment)  # should not raise


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe
# ---------------------------------------------------------------------------

def test_global_workspace_subscribe_increments_count():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    assert ws.subscriber_count() == 0
    ws.subscribe("memory", lambda m: None)
    assert ws.subscriber_count() == 1
    ws.subscribe("debate_engine", lambda m: None)
    assert ws.subscriber_count() == 2


def test_global_workspace_unsubscribe_decrements_count():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    ws.subscribe("perception", lambda m: None)
    ws.unsubscribe("perception")
    assert ws.subscriber_count() == 0


def test_global_workspace_unsubscribe_unknown_no_exception():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    ws.unsubscribe("nonexistent_module")  # should not raise


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_global_workspace_progress_dict():
    gw = _import_gw()
    ws = gw.GlobalWorkspace()
    prog = ws.progress()
    assert "can_operate" in prog
    assert "subscribers" in prog
    assert "stream_length" in prog
    assert "cycle_ms" in prog
    assert prog["can_operate"] is False
    assert prog["stream_length"] == 0


def test_global_workspace_custom_cycle_ms():
    gw = _import_gw()
    ws = gw.GlobalWorkspace(cycle_ms=50.0)
    assert ws.progress()["cycle_ms"] == 50.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_global_workspace_returns_global_workspace():
    gw = _import_gw()
    gw._workspace = None  # reset for clean test
    ws = gw.get_global_workspace()
    assert isinstance(ws, gw.GlobalWorkspace)


def test_get_global_workspace_is_singleton():
    gw = _import_gw()
    gw._workspace = None
    ws1 = gw.get_global_workspace()
    ws2 = gw.get_global_workspace()
    assert ws1 is ws2


def test_get_global_workspace_retains_subscribers():
    gw = _import_gw()
    gw._workspace = None
    ws = gw.get_global_workspace()
    ws.subscribe("causal_model", lambda m: None)
    ws2 = gw.get_global_workspace()
    assert ws2.subscriber_count() == 1
