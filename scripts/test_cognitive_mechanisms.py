"""Tests for D88 — 8 cognitive mechanisms.

Covers:
 M1 — anomaly detection with rolling baselines
 M2 — self-capability map (/introspect/capabilities endpoint)
 M3 — cross-service correlation
 M4 — world model persistence (feature flag gate)
 M5 — sensory learning / pattern detection
 M6 — skill hunter service
 M7 — proactive scheduling (calendar + sensor fusion)
 M8 — reactive skill acquisition (capability gap → skill hunter)
"""
from __future__ import annotations

import importlib.util
import sys
import types
from collections import deque
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ── Loader helpers ────────────────────────────────────────────────────

def _stub_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _load_agentic():
    """Load agentic/app.py with all heavy dependencies stubbed."""
    stubs = {
        "router": _stub_module(
            "router",
            RouteDecision=MagicMock(route="GENERAL_CHAT", confidence=0.5, bypass_llm=False, matched_keywords=[], reason=""),
            classify=MagicMock(return_value=MagicMock(route="GENERAL_CHAT", confidence=0.5, bypass_llm=False, matched_keywords=[], reason="")),
            dispatch_route=AsyncMock(return_value=None),
            load_skills=MagicMock(return_value=[]),
            list_skills=MagicMock(return_value=[]),
            match_skill=MagicMock(return_value=None),
            unload_skill=MagicMock(return_value=False),
            prune_stale_skills=MagicMock(return_value=[]),
            scan_skill_md=MagicMock(return_value={"safe": True}),
        ),
        "planner": _stub_module("planner", gather_context=AsyncMock(), build_enriched_plan=MagicMock(), predict_next_request=MagicMock(), pre_fetch_predicted_context=AsyncMock()),
        "adversary": _stub_module("adversary", challenge_plan=AsyncMock(return_value=MagicMock(score=8.0)), verdict_to_plan_metadata=MagicMock(return_value={})),
        "tree_search": _stub_module("tree_search", tree_search=AsyncMock()),
        "priority_queue": _stub_module("priority_queue", get_queue=MagicMock(return_value=MagicMock(stats=MagicMock(return_value=MagicMock(pending=0, active=0, total_processed=0, avg_wait_ms=0.0))))),
        "model_selector": _stub_module("model_selector", select_model=MagicMock(return_value="Ollama"), list_models=MagicMock(return_value=[]), get_profile=MagicMock(return_value=None)),
        "conviction": _stub_module("conviction", build_plan=MagicMock(), detect_self_deception=MagicMock(), low_conviction_feedback=MagicMock(), score_conviction=MagicMock(return_value=8.0)),
        "kai_config": _stub_module("kai_config", build_saver=MagicMock(return_value=MagicMock(recall=MagicMock(return_value=[]))), classify_failure=MagicMock(), extract_metacognitive_rule=MagicMock(), extract_preference=MagicMock(), FailureClass=MagicMock(), compute_learning_value=MagicMock(), capture_snapshot=MagicMock(), save_snapshot=MagicMock(), create_checkpoint=MagicMock(), list_checkpoints=MagicMock(return_value=[]), load_checkpoint=MagicMock(), diff_checkpoints=MagicMock(), delete_checkpoint=MagicMock()),
        "common.auth": _stub_module("common.auth", sign_gate_request=MagicMock(), sign_gate_request_bundle=MagicMock()),
        "common.feature_flags": _stub_module("common.feature_flags", is_enabled=MagicMock(return_value=True), get_all_flags=MagicMock(return_value=[])),
        "common.llm": _stub_module("common.llm", LLMRouter=MagicMock(return_value=MagicMock(available=[], stream=AsyncMock())), llm_warmup=AsyncMock()),
        "common.runtime": _stub_module("common.runtime", AuditStream=MagicMock(return_value=MagicMock(log=MagicMock())), CircuitBreaker=MagicMock(return_value=MagicMock(allow=MagicMock(return_value=True), record_success=MagicMock(), record_failure=MagicMock(), snapshot=MagicMock(return_value={}), failures=0, state="closed", opened_at=0.0)), ErrorBudget=MagicMock(return_value=MagicMock(record=MagicMock(), snapshot=MagicMock(return_value={}))), ErrorBudgetCircuitBreaker=MagicMock(return_value=MagicMock(allow=MagicMock(return_value=True), record=MagicMock(), snapshot=MagicMock(return_value={}))), INJECTION_RE=MagicMock(search=MagicMock(return_value=None)), detect_device=MagicMock(return_value="cpu"), sanitize_string=lambda x: x, setup_json_logger=MagicMock(return_value=MagicMock(info=MagicMock(), warning=MagicMock(), debug=MagicMock(), error=MagicMock()))),
        "common.self_emp_advisor": _stub_module("common.self_emp_advisor", advise=MagicMock(return_value=[]), load_expenses=MagicMock(return_value=[]), load_income_total=MagicMock(return_value=0.0), thresholds=MagicMock(return_value={})),
        "common.model_registry": _stub_module("common.model_registry", context_budget=MagicMock(return_value=3072), count_tokens=MagicMock(return_value=1)),
        "fastapi": _stub_module("fastapi", FastAPI=MagicMock(return_value=MagicMock(middleware=MagicMock(return_value=lambda f: f), get=MagicMock(return_value=lambda f: f), post=MagicMock(return_value=lambda f: f))), HTTPException=Exception, Request=MagicMock()),
        "fastapi.responses": _stub_module("fastapi.responses", StreamingResponse=MagicMock()),
        "pydantic": _stub_module("pydantic", BaseModel=object),
        "httpx": _stub_module("httpx", AsyncClient=MagicMock(), HTTPError=Exception),
    }
    for name, mod in stubs.items():
        sys.modules[name] = mod

    spec = importlib.util.spec_from_file_location("agentic_app", ROOT / "agentic" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── M1: Anomaly Detection with Baselines ─────────────────────────────

class TestAnomalyDetection:
    def setup_method(self):
        self.mod = _load_agentic()
        self.mod._sensor_baselines.clear()

    def test_baseline_returns_none_under_6_readings(self):
        for i in range(5):
            z = self.mod._update_baseline("cpu", float(i * 10))
        assert z is None

    def test_baseline_returns_zscore_after_6_readings(self):
        for i in range(6):
            self.mod._update_baseline("cpu", 50.0)
        # 7th call: 6 prior readings → computes z against mean=50.0, std=0 → returns 0.0
        z = self.mod._update_baseline("cpu", 50.0)
        assert z is not None
        assert z == pytest.approx(0.0, abs=0.01)  # same value → z=0

    def test_anomaly_detected_on_spike(self):
        for i in range(20):
            self.mod._update_baseline("cpu", float(28 + (i % 5)))  # varies 28–32
        z = self.mod._update_baseline("cpu", 95.0)
        assert z is not None
        assert z > 2.0

    def test_normal_variation_not_flagged(self):
        for i in range(20):
            self.mod._update_baseline("cpu", float(30 + (i % 3)))
        z = self.mod._update_baseline("cpu", 32.0)
        assert z is not None
        assert abs(z) < 2.0

    def test_multiple_metrics_tracked_independently(self):
        # Feed varied readings so std > 0 for both metrics
        for i in range(30):
            self.mod._update_baseline("cpu_indep", float(45 + (i % 10)))   # varies 45–54
            self.mod._update_baseline("ram_indep", float(78 + (i % 5)))    # varies 78–82
        z_cpu_normal = self.mod._update_baseline("cpu_indep", 50.0)  # mid-range → not anomalous
        z_ram_spike = self.mod._update_baseline("ram_indep", 130.0)  # massive spike → anomalous
        assert z_cpu_normal is not None and abs(z_cpu_normal) < 2.0
        assert z_ram_spike is not None and z_ram_spike > 2.0
        assert "cpu_indep" in self.mod._sensor_baselines
        assert "ram_indep" in self.mod._sensor_baselines

    def test_window_bounded_by_baseline_window(self):
        for i in range(100):
            self.mod._update_baseline("cpu", float(i))
        assert len(self.mod._sensor_baselines["cpu"]) == self.mod._BASELINE_WINDOW

    def test_zero_std_returns_zero_not_nan(self):
        for _ in range(10):
            self.mod._update_baseline("constant", 42.0)
        z = self.mod._update_baseline("constant", 42.0)
        assert z == pytest.approx(0.0)


# ── M3: Cross-Service Correlation ─────────────────────────────────────

class TestCrossServiceCorrelation:
    def setup_method(self):
        self.mod = _load_agentic()

    def test_no_correlation_on_single_observation(self):
        obs = ["Docker: 1 unhealthy container(s) — kai-memu"]
        result = self.mod._correlate_observations(obs)
        assert result == []

    def test_cpu_and_docker_flagged(self):
        obs = [
            "System: CPU at 92% — possible runaway process",
            "Docker: 2 unhealthy container(s) — kai-agentic, kai-redis",
        ]
        result = self.mod._correlate_observations(obs)
        assert any("resource pressure" in r for r in result)

    def test_ram_and_docker_flagged(self):
        obs = [
            "System: RAM at 93% — memory pressure",
            "Docker: 1 unhealthy container(s) — kai-redis",
        ]
        result = self.mod._correlate_observations(obs)
        assert any("memory leak" in r for r in result)

    def test_cpu_and_ram_together(self):
        obs = [
            "System: CPU at 92% — possible runaway process",
            "System: RAM at 93% — memory pressure",
        ]
        result = self.mod._correlate_observations(obs)
        assert any("runaway" in r or "contention" in r for r in result)

    def test_git_and_email_flagged(self):
        obs = [
            "Git: 3 repo(s) with uncommitted changes",
            "Email: 12 unread message(s) (was 4)",
        ]
        result = self.mod._correlate_observations(obs)
        assert any("mid-flow" in r or "mid-task" in r for r in result)

    def test_unrelated_observations_not_correlated(self):
        obs = [
            "Weather: 15°C, partly cloudy",
            "News: 3 new headlines",
        ]
        result = self.mod._correlate_observations(obs)
        assert result == []


# ── M4: World Model Persistence flag ──────────────────────────────────

class TestWorldModelPersistence:
    def setup_method(self):
        self.mod = _load_agentic()

    def test_flag_name_registered(self):
        import common.feature_flags as ff
        # Reload to get real module
        spec = importlib.util.spec_from_file_location("ff_real", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "WORLD_MODEL_PERSISTENCE" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["WORLD_MODEL_PERSISTENCE"][1] is True  # default on

    def test_anomaly_detection_flag_registered(self):
        spec = importlib.util.spec_from_file_location("ff_real2", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "ANOMALY_DETECTION" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["ANOMALY_DETECTION"][1] is True

    def test_proactive_scheduling_flag_registered(self):
        spec = importlib.util.spec_from_file_location("ff_real3", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "PROACTIVE_SCHEDULING" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["PROACTIVE_SCHEDULING"][1] is True

    def test_skill_hunter_flag_registered(self):
        spec = importlib.util.spec_from_file_location("ff_real4", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "SKILL_HUNTER" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["SKILL_HUNTER"][1] is True

    def test_sensory_learning_flag_registered(self):
        spec = importlib.util.spec_from_file_location("ff_real5", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "SENSORY_LEARNING" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["SENSORY_LEARNING"][1] is True


# ── M5: Sensory Learning / Pattern Detection ──────────────────────────

class TestSensoryLearning:
    def setup_method(self):
        self.mod = _load_agentic()
        self.mod._observation_history.clear()

    def test_no_pattern_on_empty_history(self):
        obs = ["Docker: 1 unhealthy container(s) — kai-redis"]
        patterns = self.mod._detect_sensor_patterns(obs)
        assert patterns == []

    def test_no_pattern_when_appears_twice(self):
        obs = ["Docker: 1 unhealthy container(s) — kai-redis"]
        self.mod._observation_history.append(obs)
        self.mod._observation_history.append(obs)
        patterns = self.mod._detect_sensor_patterns(obs)
        assert patterns == []

    def test_pattern_detected_at_three_occurrences(self):
        obs = ["Docker: 1 unhealthy container(s) — kai-redis"]
        for _ in range(3):
            self.mod._observation_history.append(obs)
        patterns = self.mod._detect_sensor_patterns(obs)
        assert len(patterns) >= 1
        assert any("docker_unhealthy" in p for p in patterns)

    def test_pattern_reports_count(self):
        obs = ["System: CPU at 92% — possible runaway process"]
        for _ in range(5):
            self.mod._observation_history.append(obs)
        patterns = self.mod._detect_sensor_patterns(obs)
        assert any("5/10" in p or "cpu_high" in p for p in patterns)

    def test_history_deque_bounded_to_10(self):
        obs = ["Git: 2 repo(s) with uncommitted changes"]
        for i in range(15):
            self.mod._observation_history.append([f"obs_{i}"])
        assert len(self.mod._observation_history) == 10

    def test_different_types_tracked_independently(self):
        cpu_obs = ["System: CPU at 92% — possible runaway process"]
        email_obs = ["Email: 5 unread message(s) (was 2)"]
        for _ in range(4):
            self.mod._observation_history.append(cpu_obs)
        patterns = self.mod._detect_sensor_patterns(cpu_obs)
        assert any("cpu_high" in p for p in patterns)
        patterns_email = self.mod._detect_sensor_patterns(email_obs)
        assert not any("cpu_high" in p for p in patterns_email)


# ── M6: Skill Hunter Service ──────────────────────────────────────────

class TestSkillHunterService:
    def _load_hunter(self):
        for mod_name in list(sys.modules.keys()):
            if "skill_hunter" in mod_name:
                del sys.modules[mod_name]
        stubs = {
            "fastapi": _stub_module("fastapi", FastAPI=MagicMock(return_value=MagicMock(get=MagicMock(return_value=lambda f: f), post=MagicMock(return_value=lambda f: f))), HTTPException=Exception),
            "pydantic": _stub_module("pydantic", BaseModel=object),
            "httpx": _stub_module("httpx", AsyncClient=MagicMock()),
        }
        for name, mod in stubs.items():
            sys.modules.setdefault(name, mod)
        spec = importlib.util.spec_from_file_location("skill_hunter", ROOT / "skill-hunter" / "app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_keyword_extraction_removes_stopwords(self):
        mod = self._load_hunter()
        kws = mod._extract_keywords("can you help me parse pdf files")
        assert "can" not in kws
        assert "pdf" in kws
        assert "parse" in kws

    def test_keyword_extraction_min_length(self):
        mod = self._load_hunter()
        kws = mod._extract_keywords("do web scraping")
        assert "do" not in kws
        assert "web" in kws

    def test_candidate_packages_for_pdf(self):
        mod = self._load_hunter()
        candidates = mod._candidate_packages(["pdf"])
        assert any("pypdf" in p or "pdfminer" in p or "reportlab" in p for p in candidates)

    def test_candidate_packages_for_nlp(self):
        mod = self._load_hunter()
        candidates = mod._candidate_packages(["nlp"])
        assert any(p in candidates for p in ["nltk", "spacy", "textblob"])

    def test_candidate_packages_bounded_to_8(self):
        mod = self._load_hunter()
        candidates = mod._candidate_packages(["pdf", "excel", "data", "web", "image", "audio", "ml", "graph", "crypto"])
        assert len(candidates) <= 8

    def test_skill_name_normalised(self):
        mod = self._load_hunter()
        name = mod._skill_name("Parse PDF files for me!")
        assert re.match(r"^[a-z0-9_]+$", name)
        assert len(name) <= 30

    def test_generate_skill_md_contains_package(self):
        mod = self._load_hunter()
        md = mod._generate_skill_md("parse pdf files", "pypdf2")
        assert "pypdf2" in md
        assert "parse pdf files" in md

    @pytest.mark.anyio
    async def test_hunt_returns_not_created_on_no_match(self, tmp_path, monkeypatch):
        mod = self._load_hunter()
        mod.SKILLS_DIR = tmp_path
        monkeypatch.setattr(mod, "_pypi_exists", AsyncMock(return_value=False))
        req = MagicMock()
        req.gap = "xyzzy nonexistent capability"
        result = await mod.hunt(req)
        assert result["skill_created"] is False

    @pytest.mark.anyio
    async def test_hunt_creates_skill_file_on_match(self, tmp_path, monkeypatch):
        mod = self._load_hunter()
        mod.SKILLS_DIR = tmp_path
        monkeypatch.setattr(mod, "_pypi_exists", AsyncMock(return_value=True))
        req = MagicMock()
        req.gap = "parse pdf files"
        result = await mod.hunt(req)
        assert result["skill_created"] is True
        assert (tmp_path / f"hunted_{result['skill_name']}.md").exists()

    @pytest.mark.anyio
    async def test_list_hunted_skills(self, tmp_path):
        mod = self._load_hunter()
        mod.SKILLS_DIR = tmp_path
        (tmp_path / "hunted_test_skill.md").write_text("# test", encoding="utf-8")
        result = await mod.list_hunted_skills()
        assert result["count"] == 1
        assert "hunted_test_skill" in result["skills"]


import re  # needed inside class methods above


# ── M7: Proactive Scheduling (calendar + sensor fusion) ───────────────

class TestProactiveScheduling:
    def setup_method(self):
        self.mod = _load_agentic()

    def test_flag_registered_and_default_true(self):
        spec = importlib.util.spec_from_file_location("ff_sched", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert "PROACTIVE_SCHEDULING" in ff_mod._REGISTRY
        assert ff_mod._REGISTRY["PROACTIVE_SCHEDULING"][1] is True

    def test_calendar_url_constant_exists(self):
        assert hasattr(self.mod, "CALENDAR_URL")
        assert "calendar" in self.mod.CALENDAR_URL.lower()


# ── M8: Reactive Skill Acquisition ───────────────────────────────────

class TestReactiveSkillAcquisition:
    def setup_method(self):
        self.mod = _load_agentic()

    def test_hunt_skill_for_gap_function_exists(self):
        assert hasattr(self.mod, "_hunt_skill_for_gap")
        import asyncio
        assert asyncio.iscoroutinefunction(self.mod._hunt_skill_for_gap)

    def test_skill_hunter_url_constant(self):
        assert hasattr(self.mod, "SKILL_HUNTER_URL")
        assert "skill-hunter" in self.mod.SKILL_HUNTER_URL or "8045" in self.mod.SKILL_HUNTER_URL

    def test_skill_hunter_flag_gates_hunt(self):
        spec = importlib.util.spec_from_file_location("ff_m8", ROOT / "common" / "feature_flags.py")
        ff_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ff_mod)
        assert ff_mod.is_enabled("SKILL_HUNTER") is True  # default on

    def test_introspect_capabilities_endpoint_registered(self):
        # The endpoint should be defined in the module
        assert hasattr(self.mod, "introspect_capabilities")


# ── M2: Self-Capability Map ───────────────────────────────────────────

class TestSelfCapabilityMap:
    def setup_method(self):
        self.mod = _load_agentic()

    def test_introspect_endpoint_exists(self):
        assert hasattr(self.mod, "introspect_capabilities")

    def test_baselines_tracked_is_list(self):
        assert isinstance(self.mod._sensor_baselines, dict)

    def test_observation_history_is_deque(self):
        assert isinstance(self.mod._observation_history, deque)
        assert self.mod._observation_history.maxlen == 10

    def test_baseline_window_constant(self):
        assert self.mod._BASELINE_WINDOW == 48
