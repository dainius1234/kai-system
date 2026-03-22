"""H3b: LangGraph Checkpoint Engine — time-travel debug & state snapshots.

Tests cover:
  - Checkpoint creation and persistence
  - Listing and loading checkpoints
  - Diff between two checkpoints
  - Rollback / restore semantics
  - Cap enforcement (max 30)
  - Edge cases (missing, duplicate labels, empty state)
  - Serialization round-trip
  - Endpoint integration (dream auto-checkpoint, recover auto-checkpoint)
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import pytest

# ── path setup ───────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "langgraph"))

# Use temp dir for checkpoints so tests don't interfere
_test_dir = tempfile.mkdtemp(prefix="kai_checkpoint_test_")
os.environ.setdefault("CHECKPOINT_DIR", _test_dir)
os.environ.setdefault("REDIS_URL", "")

from langgraph.kai_config import (
    Checkpoint,
    CHECKPOINT_DIR,
    CHECKPOINT_MAX,
    create_checkpoint,
    list_checkpoints,
    load_checkpoint,
    diff_checkpoints,
    delete_checkpoint,
    _checkpoint_path,
)


@pytest.fixture(autouse=True)
def clean_checkpoint_dir():
    """Ensure a clean checkpoint dir for each test."""
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    yield
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)


# ─────────────────────────────────────────────────────────────────────
#  Checkpoint creation
# ─────────────────────────────────────────────────────────────────────

class TestCreateCheckpoint:
    def test_basic_create(self):
        cp = create_checkpoint(
            label="test-basic",
            trigger="manual",
            breaker_states={"memu": {"state": "closed", "failures": 0}},
            guard_states={"memu": {"error_rate": 0.01}},
            budget_state={"budget_ok": True, "error_rate": 0.0},
        )
        assert cp.checkpoint_id
        assert cp.label == "test-basic"
        assert cp.trigger == "manual"
        assert cp.timestamp > 0

    def test_checkpoint_has_unique_id(self):
        cp1 = create_checkpoint("a", "manual", {}, {}, {})
        cp2 = create_checkpoint("b", "manual", {}, {}, {})
        assert cp1.checkpoint_id != cp2.checkpoint_id

    def test_checkpoint_persisted_to_disk(self):
        cp = create_checkpoint("disk-test", "auto", {}, {}, {})
        path = _checkpoint_path(cp.checkpoint_id)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["label"] == "disk-test"

    def test_conviction_overrides_captured(self):
        cp = create_checkpoint(
            "override-test", "manual", {}, {}, {},
            conviction_overrides=["test rule", "another rule"],
        )
        assert cp.conviction_overrides == ["test rule", "another rule"]


# ─────────────────────────────────────────────────────────────────────
#  Listing checkpoints
# ─────────────────────────────────────────────────────────────────────

class TestListCheckpoints:
    def test_empty_list(self):
        result = list_checkpoints()
        assert result == []

    def test_list_returns_newest_first(self):
        cp1 = create_checkpoint("first", "manual", {}, {}, {})
        time.sleep(0.05)  # ensure different mtime
        cp2 = create_checkpoint("second", "manual", {}, {}, {})
        cps = list_checkpoints()
        assert len(cps) == 2
        assert cps[0]["checkpoint_id"] == cp2.checkpoint_id
        assert cps[1]["checkpoint_id"] == cp1.checkpoint_id

    def test_list_respects_limit(self):
        for i in range(5):
            create_checkpoint(f"cp-{i}", "manual", {}, {}, {})
            time.sleep(0.02)
        cps = list_checkpoints(limit=3)
        assert len(cps) == 3

    def test_list_includes_metadata(self):
        create_checkpoint("meta-test", "pre_recover", {}, {}, {})
        cps = list_checkpoints()
        assert cps[0]["label"] == "meta-test"
        assert cps[0]["trigger"] == "pre_recover"
        assert "iso_time" in cps[0]
        assert "timestamp" in cps[0]


# ─────────────────────────────────────────────────────────────────────
#  Loading checkpoints
# ─────────────────────────────────────────────────────────────────────

class TestLoadCheckpoint:
    def test_load_existing(self):
        cp = create_checkpoint("load-me", "manual", {"k": "v"}, {}, {})
        loaded = load_checkpoint(cp.checkpoint_id)
        assert loaded is not None
        assert loaded.label == "load-me"
        assert loaded.breakers == {"k": "v"}

    def test_load_missing_returns_none(self):
        assert load_checkpoint("nonexistent-id") is None

    def test_roundtrip_all_fields(self):
        cp = create_checkpoint(
            label="full-roundtrip",
            trigger="post_dream",
            breaker_states={"memu": {"state": "open", "failures": 5, "opened_at": 1234.5}},
            guard_states={"memu": {"error_rate": 0.08, "state": "warn"}},
            budget_state={"budget_ok": False, "error_rate": 0.12},
            conviction_overrides=["override_a"],
        )
        loaded = load_checkpoint(cp.checkpoint_id)
        assert loaded.trigger == "post_dream"
        assert loaded.breakers["memu"]["state"] == "open"
        assert loaded.error_guards["memu"]["error_rate"] == 0.08
        assert loaded.error_budget["budget_ok"] is False
        assert loaded.conviction_overrides == ["override_a"]


# ─────────────────────────────────────────────────────────────────────
#  Diff checkpoints
# ─────────────────────────────────────────────────────────────────────

class TestDiffCheckpoints:
    def test_identical_checkpoints_no_changes(self):
        state = {"memu": {"state": "closed", "failures": 0}}
        cp1 = create_checkpoint("a", "manual", state, {}, {})
        cp2 = create_checkpoint("b", "manual", state, {}, {})
        d = diff_checkpoints(cp1, cp2)
        assert d["changed_fields"] == 0
        assert d["changes"] == {}

    def test_breaker_state_change_detected(self):
        cp1 = create_checkpoint("before", "manual",
                                {"memu": {"state": "closed", "failures": 0}}, {}, {})
        cp2 = create_checkpoint("after", "manual",
                                {"memu": {"state": "open", "failures": 3}}, {}, {})
        d = diff_checkpoints(cp1, cp2)
        assert d["changed_fields"] >= 1
        assert "breaker.memu" in d["changes"]
        assert d["changes"]["breaker.memu"]["before"]["state"] == "closed"
        assert d["changes"]["breaker.memu"]["after"]["state"] == "open"

    def test_guard_change_detected(self):
        cp1 = create_checkpoint("g1", "manual", {}, {"memu": {"rate": 0.01}}, {})
        cp2 = create_checkpoint("g2", "manual", {}, {"memu": {"rate": 0.09}}, {})
        d = diff_checkpoints(cp1, cp2)
        assert "guard.memu" in d["changes"]

    def test_budget_change_detected(self):
        cp1 = create_checkpoint("b1", "manual", {}, {}, {"budget_ok": True})
        cp2 = create_checkpoint("b2", "manual", {}, {}, {"budget_ok": False})
        d = diff_checkpoints(cp1, cp2)
        assert "error_budget" in d["changes"]

    def test_conviction_override_diff(self):
        cp1 = create_checkpoint("o1", "manual", {}, {}, {},
                                conviction_overrides=["rule_a"])
        cp2 = create_checkpoint("o2", "manual", {}, {}, {},
                                conviction_overrides=["rule_a", "rule_b"])
        d = diff_checkpoints(cp1, cp2)
        assert "conviction_overrides" in d["changes"]
        assert "rule_b" in d["changes"]["conviction_overrides"]["added"]
        assert d["changes"]["conviction_overrides"]["removed"] == []

    def test_time_delta_computed(self):
        cp1 = create_checkpoint("t1", "manual", {}, {}, {})
        time.sleep(0.1)
        cp2 = create_checkpoint("t2", "manual", {}, {}, {})
        d = diff_checkpoints(cp1, cp2)
        assert d["time_delta_seconds"] > 0
        assert d["checkpoint_a"] == cp1.checkpoint_id
        assert d["checkpoint_b"] == cp2.checkpoint_id


# ─────────────────────────────────────────────────────────────────────
#  Delete checkpoint
# ─────────────────────────────────────────────────────────────────────

class TestDeleteCheckpoint:
    def test_delete_existing(self):
        cp = create_checkpoint("del-me", "manual", {}, {}, {})
        assert delete_checkpoint(cp.checkpoint_id) is True
        assert load_checkpoint(cp.checkpoint_id) is None

    def test_delete_nonexistent(self):
        assert delete_checkpoint("no-such-id") is False


# ─────────────────────────────────────────────────────────────────────
#  Cap enforcement
# ─────────────────────────────────────────────────────────────────────

class TestCapEnforcement:
    def test_cap_at_max(self):
        for i in range(CHECKPOINT_MAX + 5):
            create_checkpoint(f"cap-{i}", "auto", {}, {}, {})
            time.sleep(0.01)
        files = list(CHECKPOINT_DIR.glob("*.json"))
        assert len(files) <= CHECKPOINT_MAX

    def test_oldest_removed_first(self):
        cps = []
        for i in range(CHECKPOINT_MAX + 3):
            cp = create_checkpoint(f"order-{i}", "auto", {}, {}, {})
            cps.append(cp)
            time.sleep(0.01)
        # The first few should be gone
        assert load_checkpoint(cps[0].checkpoint_id) is None
        assert load_checkpoint(cps[1].checkpoint_id) is None
        # The last one should still exist
        assert load_checkpoint(cps[-1].checkpoint_id) is not None


# ─────────────────────────────────────────────────────────────────────
#  Serialization
# ─────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_roundtrip(self):
        cp = create_checkpoint("ser-test", "manual",
                               {"a": 1}, {"b": 2}, {"c": 3},
                               conviction_overrides=["x"])
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.checkpoint_id == cp.checkpoint_id
        assert restored.label == cp.label
        assert restored.breakers == {"a": 1}
        assert restored.conviction_overrides == ["x"]

    def test_to_dict_json_serializable(self):
        cp = create_checkpoint("json-test", "auto", {"k": [1, 2]}, {}, {})
        s = json.dumps(cp.to_dict())
        assert isinstance(s, str)

    def test_iso_time_format(self):
        cp = create_checkpoint("iso-test", "manual", {}, {}, {})
        assert "T" in cp.iso_time
        assert len(cp.iso_time) > 10


# ─────────────────────────────────────────────────────────────────────
#  Edge cases
# ─────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_state(self):
        cp = create_checkpoint("empty", "manual", {}, {}, {})
        loaded = load_checkpoint(cp.checkpoint_id)
        assert loaded.breakers == {}

    def test_special_chars_in_label(self):
        cp = create_checkpoint("lbl/with:specials!", "manual", {}, {}, {})
        assert cp.label == "lbl/with:specials!"
        loaded = load_checkpoint(cp.checkpoint_id)
        assert loaded is not None

    def test_checkpoint_id_is_filesystem_safe(self):
        cp = create_checkpoint("safe-check", "manual", {}, {}, {})
        # ID should contain only alphanumeric, dashes, underscores
        import re
        assert re.match(r'^[\w\-]+$', cp.checkpoint_id)

    def test_concurrent_create_no_collision(self):
        """Multiple rapid creates should all persist."""
        cps = [create_checkpoint(f"rapid-{i}", "auto", {}, {}, {}) for i in range(10)]
        ids = {cp.checkpoint_id for cp in cps}
        assert len(ids) == 10

    def test_from_dict_missing_fields(self):
        """Checkpoint.from_dict handles partial data gracefully."""
        cp = Checkpoint.from_dict({"checkpoint_id": "test-partial"})
        assert cp.checkpoint_id == "test-partial"
        assert cp.label == ""
        assert cp.breakers == {}
        assert cp.conviction_overrides == []


# ─────────────────────────────────────────────────────────────────────
#  Integration: auto-checkpoint in /recover and /dream
# ─────────────────────────────────────────────────────────────────────

class TestAutoCheckpointIntegration:
    """Verify the wiring that creates checkpoints from endpoints.

    These test the checkpoint functions called by app.py, not the
    HTTP endpoints themselves (those need the running server).
    """

    def test_pre_recover_checkpoint_pattern(self):
        """Simulate what /recover does: create pre_recover checkpoint."""
        cp = create_checkpoint(
            label="pre-recover",
            trigger="pre_recover",
            breaker_states={"memu": {"state": "open", "failures": 3}},
            guard_states={"memu": {"error_rate": 0.09}},
            budget_state={"budget_ok": True},
            conviction_overrides=[],
        )
        assert cp.trigger == "pre_recover"
        loaded = load_checkpoint(cp.checkpoint_id)
        assert loaded.breakers["memu"]["state"] == "open"

    def test_post_dream_checkpoint_pattern(self):
        """Simulate what /dream does: create post_dream checkpoint."""
        cp = create_checkpoint(
            label="post-dream-abc12345",
            trigger="post_dream",
            breaker_states={"memu": {"state": "closed"}},
            guard_states={},
            budget_state={"budget_ok": True},
        )
        assert cp.trigger == "post_dream"
        assert "post-dream" in cp.label

    def test_pre_restore_checkpoint_pattern(self):
        """Simulate what /checkpoint/{id}/restore does before rollback."""
        target = create_checkpoint("target-state", "manual",
                                   {"memu": {"state": "open"}}, {}, {})
        # Pre-restore checkpoint
        pre = create_checkpoint(
            label=f"pre-restore-to-{target.checkpoint_id[:16]}",
            trigger="pre_restore",
            breaker_states={"memu": {"state": "closed"}},
            guard_states={},
            budget_state={},
        )
        assert pre.trigger == "pre_restore"
        assert target.checkpoint_id[:16] in pre.label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
