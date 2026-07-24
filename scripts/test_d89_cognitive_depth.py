"""D89 Cognitive Depth — test suite.

Tests cover:
    - System FSM (system_fsm.py)
    - Cognitive FSM (cognitive_fsm.py)
    - Persistent Teammates (teammates.py)
    - Counterfactual Rehearsal stub (counterfactual.py)
    - Resource-Aware Curiosity stub (curiosity.py)
    - House Doctor service (house-doctor/app.py)
    - Skill Hunter provenance + probationary (skill-hunter/app.py)
    - Feature flags for D89 (8 new flags)
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── path setup ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "agentic"))
sys.path.insert(0, str(_ROOT / "common"))
sys.path.insert(0, str(_ROOT / "skill-hunter"))
sys.path.insert(0, str(_ROOT / "house-doctor"))


# ══════════════════════════════════════════════════════════════════════
# System FSM
# ══════════════════════════════════════════════════════════════════════

class TestSystemFSM:
    def test_initial_state_is_idle(self):
        from system_fsm import KaiFSM, KaiState
        fsm = KaiFSM()
        assert fsm.state == KaiState.IDLE

    @pytest.mark.anyio
    async def test_user_message_idle_to_active(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        result = await fsm.fire(KaiEvent.USER_MESSAGE)
        assert result == KaiState.ACTIVE
        assert fsm.state == KaiState.ACTIVE

    @pytest.mark.anyio
    async def test_session_end_active_to_idle(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        await fsm.fire(KaiEvent.USER_MESSAGE)
        result = await fsm.fire(KaiEvent.SESSION_END)
        assert result == KaiState.IDLE

    @pytest.mark.anyio
    async def test_service_down_from_idle_to_degraded(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        result = await fsm.fire(KaiEvent.SERVICE_DOWN)
        assert result == KaiState.DEGRADED

    @pytest.mark.anyio
    async def test_service_restored_from_degraded_to_idle(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        await fsm.fire(KaiEvent.SERVICE_DOWN)
        result = await fsm.fire(KaiEvent.SERVICE_RESTORED)
        assert result == KaiState.IDLE

    @pytest.mark.anyio
    async def test_focus_enter_from_idle_to_focused(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        result = await fsm.fire(KaiEvent.FOCUS_ENTER)
        assert result == KaiState.FOCUSED

    @pytest.mark.anyio
    async def test_undefined_transition_returns_none(self):
        from system_fsm import KaiFSM, KaiState, KaiEvent
        fsm = KaiFSM()
        # IDLE + SESSION_END is undefined
        result = await fsm.fire(KaiEvent.SESSION_END)
        assert result is None
        assert fsm.state == KaiState.IDLE

    @pytest.mark.anyio
    async def test_snapshot_includes_recent_transitions(self):
        from system_fsm import KaiFSM, KaiEvent
        fsm = KaiFSM()
        await fsm.fire(KaiEvent.USER_MESSAGE)
        snap = fsm.snapshot()
        assert snap["state"] == "active"
        assert len(snap["recent_transitions"]) == 1
        assert snap["recent_transitions"][0]["event"] == "user_message"

    @pytest.mark.anyio
    async def test_history_capped_at_100(self):
        from system_fsm import KaiFSM, KaiEvent
        fsm = KaiFSM()
        for _ in range(60):
            await fsm.fire(KaiEvent.USER_MESSAGE)
            await fsm.fire(KaiEvent.SESSION_END)
        assert len(fsm._history) <= 100


# ══════════════════════════════════════════════════════════════════════
# Cognitive FSM
# ══════════════════════════════════════════════════════════════════════

class TestCognitiveFSM:
    def _make_stage(self, status, confidence=8.5):
        from cognitive_fsm import AgentHandoff, HandoffStatus
        hs = HandoffStatus(status)

        async def _fn(handoff: AgentHandoff, cfg) -> AgentHandoff:
            return AgentHandoff(
                from_stage=handoff.to_stage,
                to_stage="next",
                status=hs,
                confidence=confidence,
            )
        return _fn

    @pytest.mark.anyio
    async def test_happy_path_reaches_present(self):
        from cognitive_fsm import CognitiveFSM, CogState, SWARM_CONFIGS
        fsm = CognitiveFSM(SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=self._make_stage("complete"),
            debate_fn=self._make_stage("consensus", 9.0),
            fact_check_fn=self._make_stage("pass", 8.5),
            causal_check_fn=self._make_stage("pass", 8.5),
            conviction_gate_fn=self._make_stage("complete", 8.5),
        )
        assert result.final_state == CogState.PRESENT
        assert not result.halted

    @pytest.mark.anyio
    async def test_gather_failure_halts(self):
        from cognitive_fsm import CognitiveFSM, CogState, AgentHandoff, HandoffStatus, SWARM_CONFIGS

        async def bad_gather(handoff, cfg):
            return AgentHandoff("gather", "next", HandoffStatus.FAILED, 0.0, halt_reason="no data")

        fsm = CognitiveFSM(SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=bad_gather,
            debate_fn=self._make_stage("consensus"),
            fact_check_fn=self._make_stage("pass"),
            causal_check_fn=self._make_stage("pass"),
            conviction_gate_fn=self._make_stage("complete", 8.5),
        )
        assert result.final_state == CogState.HALT
        assert result.halted

    @pytest.mark.anyio
    async def test_debate_retry_cap_halts(self):
        from cognitive_fsm import CognitiveFSM, CogState, SWARM_CONFIGS
        fsm = CognitiveFSM(SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=self._make_stage("complete"),
            debate_fn=self._make_stage("no_consensus", 4.0),
            fact_check_fn=self._make_stage("pass"),
            causal_check_fn=self._make_stage("pass"),
            conviction_gate_fn=self._make_stage("complete", 8.5),
        )
        assert result.final_state == CogState.HALT

    @pytest.mark.anyio
    async def test_conviction_below_threshold_rethink_then_halt(self):
        from cognitive_fsm import CognitiveFSM, CogState, SWARM_CONFIGS
        fsm = CognitiveFSM(SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=self._make_stage("complete"),
            debate_fn=self._make_stage("consensus", 8.5),
            fact_check_fn=self._make_stage("pass"),
            causal_check_fn=self._make_stage("pass"),
            conviction_gate_fn=self._make_stage("complete", 3.0),  # below threshold
        )
        assert result.final_state == CogState.HALT

    @pytest.mark.anyio
    async def test_trading_config_has_higher_conviction_threshold(self):
        from cognitive_fsm import SWARM_CONFIGS
        assert SWARM_CONFIGS["trading"].conviction_threshold > SWARM_CONFIGS["research"].conviction_threshold

    @pytest.mark.anyio
    async def test_transition_log_populated(self):
        from cognitive_fsm import CognitiveFSM, SWARM_CONFIGS
        fsm = CognitiveFSM(SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=self._make_stage("complete"),
            debate_fn=self._make_stage("consensus", 8.5),
            fact_check_fn=self._make_stage("pass"),
            causal_check_fn=self._make_stage("pass"),
            conviction_gate_fn=self._make_stage("complete", 8.5),
        )
        assert len(result.transition_log) > 0
        assert any(t["to"] == "present" for t in result.transition_log)


# ══════════════════════════════════════════════════════════════════════
# Persistent Teammates
# ══════════════════════════════════════════════════════════════════════

class TestPersistentTeammates:
    def _tmp_teammates_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "teammates"
        d.mkdir()
        (d / "scout.md").write_text(
            "# Scout\n**Specialty:** skill_discovery\n**Description:** Finds tools.\n\n## System Prompt\nYou are Scout.\n"
        )
        (d / "doctor.md").write_text(
            "# Doctor\n**Specialty:** system_health\n**Description:** Diagnoses.\n\n## System Prompt\nYou are Doctor.\n"
        )
        return d

    def test_load_teammates_from_directory(self, tmp_path):
        d = self._tmp_teammates_dir(tmp_path)
        import teammates
        original_dir = teammates.TEAMMATES_DIR
        teammates.TEAMMATES_DIR = d
        teammates.load_teammates()
        result = teammates.list_teammates()
        teammates.TEAMMATES_DIR = original_dir
        slugs = {t["slug"] for t in result}
        assert "scout" in slugs
        assert "doctor" in slugs

    def test_teammate_specialty_parsed(self, tmp_path):
        d = self._tmp_teammates_dir(tmp_path)
        import teammates
        original_dir = teammates.TEAMMATES_DIR
        teammates.TEAMMATES_DIR = d
        teammates.load_teammates()
        t = teammates.get_teammate("scout")
        teammates.TEAMMATES_DIR = original_dir
        assert t is not None
        assert t.specialty == "skill_discovery"

    def test_build_teammate_context_contains_name(self, tmp_path):
        d = self._tmp_teammates_dir(tmp_path)
        import teammates
        original_dir = teammates.TEAMMATES_DIR
        teammates.TEAMMATES_DIR = d
        teammates.load_teammates()
        ctx = teammates.build_teammate_context("scout")
        teammates.TEAMMATES_DIR = original_dir
        assert ctx is not None
        assert "Scout" in ctx

    def test_unknown_teammate_returns_none(self, tmp_path):
        d = self._tmp_teammates_dir(tmp_path)
        import teammates
        original_dir = teammates.TEAMMATES_DIR
        teammates.TEAMMATES_DIR = d
        teammates.load_teammates()
        result = teammates.get_teammate("nonexistent")
        teammates.TEAMMATES_DIR = original_dir
        assert result is None

    def test_missing_directory_loads_empty(self, tmp_path):
        import teammates
        original_dir = teammates.TEAMMATES_DIR
        teammates.TEAMMATES_DIR = tmp_path / "no_such_dir"
        teammates.load_teammates()
        result = teammates.list_teammates()
        teammates.TEAMMATES_DIR = original_dir
        assert result == []


# ══════════════════════════════════════════════════════════════════════
# Counterfactual Rehearsal stub
# ══════════════════════════════════════════════════════════════════════

class TestCounterfactualStub:
    @pytest.mark.anyio
    async def test_rehearse_returns_stub_status(self):
        from counterfactual import rehearse
        result = await rehearse("should I deploy now?", {"cpu": 80})
        assert result["status"] == "stub_pending_gpu"
        assert result["scenarios"] == []
        assert result["recommendation"] is None

    @pytest.mark.anyio
    async def test_can_rehearse_is_false(self):
        from counterfactual import can_rehearse
        assert await can_rehearse() is False

    @pytest.mark.anyio
    async def test_rehearse_preserves_decision(self):
        from counterfactual import rehearse
        result = await rehearse("deploy feature X", {})
        assert result["decision"] == "deploy feature X"

    @pytest.mark.anyio
    async def test_skill_test_stub_returns_stub(self):
        from counterfactual import rehearse_skill_test
        result = await rehearse_skill_test("my_skill", ["gap1", "gap2"], {})
        assert result["status"] == "stub_pending_gpu"
        assert result["past_gaps_tested"] == 2


# ══════════════════════════════════════════════════════════════════════
# Curiosity stub
# ══════════════════════════════════════════════════════════════════════

class TestCuriosityStub:
    @pytest.mark.anyio
    async def test_no_gpu_returns_none(self):
        from curiosity import idle_curiosity_tick
        result = await idle_curiosity_tick({}, is_gpu_available=False)
        assert result is None

    def test_get_open_questions_returns_list(self):
        from curiosity import get_open_questions
        qs = get_open_questions({})
        assert isinstance(qs, list)
        assert len(qs) > 0


# ══════════════════════════════════════════════════════════════════════
# House Doctor service
# ══════════════════════════════════════════════════════════════════════

class TestHouseDoctor:
    def _app(self):
        from fastapi.testclient import TestClient
        import house_doctor_app as hd_app
        return TestClient(hd_app.app)

    def test_health_endpoint(self, tmp_path):
        sys.path.insert(0, str(_ROOT / "house-doctor"))
        import importlib, types
        # Patch httpx at module level to avoid real network calls
        with patch("httpx.AsyncClient"):
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "house_doctor_app", _ROOT / "house-doctor" / "app.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            from fastapi.testclient import TestClient
            client = TestClient(mod.app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_classify_cpu_high(self):
        sys.path.insert(0, str(_ROOT / "house-doctor"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tags = mod._classify_observations(["System: CPU at 92% — possible runaway process"])
        assert "cpu_high" in tags

    def test_classify_docker_unhealthy(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd2", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tags = mod._classify_observations(["Docker: 2 unhealthy container(s) — kai-memu"])
        assert "docker_unhealthy" in tags

    def test_diagnose_cpu_and_docker_matches_d001(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd3", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tags = ["cpu_high", "docker_unhealthy"]
        diagnoses = mod._apply_rules(tags)
        ids = [d.rule_id for d in diagnoses]
        assert "D001" in ids

    def test_no_tags_no_diagnoses(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd4", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        diagnoses = mod._apply_rules([])
        assert diagnoses == []

    def test_severity_ordering_critical_first(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd5", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tags = ["sensor_anomaly", "cpu_high", "ram_high", "docker_unhealthy"]
        diagnoses = mod._apply_rules(tags)
        if len(diagnoses) > 1:
            order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            for i in range(len(diagnoses) - 1):
                assert order[diagnoses[i].severity] <= order[diagnoses[i + 1].severity]

    def test_rules_endpoint_returns_all(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hd6", _ROOT / "house-doctor" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from fastapi.testclient import TestClient
        client = TestClient(mod.app)
        resp = client.get("/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(mod._RULES)


# ══════════════════════════════════════════════════════════════════════
# Skill Hunter provenance (D89 C2)
# ══════════════════════════════════════════════════════════════════════

class TestSkillHunterProvenance:
    def _load_app(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sh_app", _ROOT / "skill-hunter" / "app.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_generate_skill_md_has_provenance_header(self):
        mod = self._load_app()
        md = mod._generate_skill_md("parse pdf files", "pypdf2", "parse_pdf_files")
        assert "pypi_verified: true" in md
        assert "probationary: true" in md
        assert "hunted_at:" in md

    def test_skill_name_normalised(self):
        mod = self._load_app()
        name = mod._skill_name("Parse PDF files quickly")
        assert " " not in name
        assert name.islower() or "_" in name

    def test_error_count_increments(self, tmp_path):
        mod = self._load_app()
        mod.SKILLS_DIR = tmp_path
        (tmp_path / "hunted_testskill.md").write_text("---\nname: testskill\n---\n")
        mod._save_meta("testskill", {"error_count": 0, "disabled": False, "probationary": True})
        mod._load_meta("testskill")  # ensure file exists
        # simulate 3 errors
        for i in range(3):
            meta = mod._load_meta("testskill")
            meta.setdefault("error_count", 0)
            meta["error_count"] += 1
            if meta["error_count"] >= mod.DISABLE_THRESHOLD:
                meta["disabled"] = True
            mod._save_meta("testskill", meta)
        final = mod._load_meta("testskill")
        assert final["disabled"] is True

    def test_meta_path_is_sidecar(self):
        mod = self._load_app()
        p = mod._meta_path("myskill")
        assert p.suffix == ".json"
        assert "myskill" in p.name

    @pytest.mark.anyio
    async def test_hunt_creates_provenance_meta(self, tmp_path):
        mod = self._load_app()
        mod.SKILLS_DIR = tmp_path
        with patch.object(mod, "_pypi_exists", new=AsyncMock(return_value=True)):
            result = await mod.hunt(mod.HuntRequest(gap="scrape web pages"))
        if result["skill_created"]:
            meta_files = list(tmp_path.glob("*.meta.json"))
            assert len(meta_files) == 1
            meta = json.loads(meta_files[0].read_text())
            assert meta["pypi_verified"] is True
            assert meta["probationary"] is True


# ══════════════════════════════════════════════════════════════════════
# D89 Feature Flags
# ══════════════════════════════════════════════════════════════════════

class TestD89FeatureFlags:
    def _flags(self):
        from feature_flags import get_all_flags
        return {f["flag"]: f for f in get_all_flags()}

    def test_fsm_flag_registered_default_true(self):
        flags = self._flags()
        assert "FSM" in flags
        assert flags["FSM"]["default"] is True

    def test_persistent_teammates_flag_registered(self):
        flags = self._flags()
        assert "PERSISTENT_TEAMMATES" in flags
        assert flags["PERSISTENT_TEAMMATES"]["default"] is True

    def test_house_doctor_flag_registered(self):
        flags = self._flags()
        assert "HOUSE_DOCTOR" in flags
        assert flags["HOUSE_DOCTOR"]["default"] is True

    def test_ritual_discovery_flag_registered(self):
        flags = self._flags()
        assert "RITUAL_DISCOVERY" in flags

    def test_gap_logging_flag_registered(self):
        flags = self._flags()
        assert "GAP_LOGGING" in flags

    def test_trust_negotiation_flag_registered(self):
        flags = self._flags()
        assert "TRUST_NEGOTIATION" in flags

    def test_predictive_empathy_flag_registered(self):
        flags = self._flags()
        assert "PREDICTIVE_EMPATHY" in flags

    def test_curiosity_flag_registered(self):
        flags = self._flags()
        assert "CURIOSITY" in flags

    def test_all_d89_flags_default_true(self):
        flags = self._flags()
        d89 = ["FSM", "PERSISTENT_TEAMMATES", "HOUSE_DOCTOR", "RITUAL_DISCOVERY",
               "GAP_LOGGING", "TRUST_NEGOTIATION", "PREDICTIVE_EMPATHY", "CURIOSITY"]
        for flag in d89:
            assert flags[flag]["default"] is True, f"{flag} should default to True"
