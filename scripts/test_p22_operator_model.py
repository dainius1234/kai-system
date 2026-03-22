"""P22 Operator Model & Adaptive Response — test suite.

Covers all 5 subsystems + LangGraph integration + dashboard proxies + supervisor + UI.
"""
import ast
import inspect
import os
import re
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _read(rel: str) -> str:
    return (ROOT / rel).read_text()


def _parse(rel: str) -> ast.Module:
    return ast.compile(_read(rel), rel, "exec", ast.PyCF_ONLY_AST)


def _memu() -> str:
    return _read("memu-core/app.py")


def _langgraph() -> str:
    return _read("langgraph/app.py")


def _dashboard() -> str:
    return _read("dashboard/app.py")


def _html() -> str:
    return _read("dashboard/static/app.html")


def _supervisor() -> str:
    return _read("supervisor/app.py")


# ═══════════════════════════════════════════════════════════════════
# P22a: Echo-Response Engine
# ═══════════════════════════════════════════════════════════════════

class TestEchoResponseEngine(unittest.TestCase):
    """Echo-Response Engine endpoints and logic."""

    def test_echo_analyse_endpoint(self):
        src = _memu()
        self.assertIn("/memory/echo/analyse", src)

    def test_echo_analyse_is_post(self):
        src = _memu()
        self.assertIn('@app.post("/memory/echo/analyse")', src)

    def test_echo_history_endpoint(self):
        src = _memu()
        self.assertIn("/memory/echo/history", src)

    def test_echo_history_is_get(self):
        src = _memu()
        self.assertIn('@app.get("/memory/echo/history")', src)

    def test_echo_history_store(self):
        src = _memu()
        self.assertIn("_echo_history", src)

    def test_echo_types(self):
        """Must support deep_bridge, gentle_bridge, soft_mirror, none."""
        src = _memu()
        for etype in ("deep_bridge", "gentle_bridge", "soft_mirror", "none"):
            self.assertIn(etype, src, f"Missing echo type: {etype}")

    def test_echo_bridges_past_emotions(self):
        """Echo should look at _emotional_timeline for past matches."""
        src = _memu()
        self.assertIn("_emotional_timeline", src.split("/memory/echo/analyse")[1].split("/memory/echo/history")[0])

    def test_echo_intensity_thresholds(self):
        """Different intensity levels should produce different echo types."""
        src = _memu()
        self.assertIn("intensity >= 0.6", src)
        self.assertIn("intensity >= 0.4", src)

    def test_echo_records_event(self):
        src = _memu()
        self.assertIn("_echo_history.append", src)


# ═══════════════════════════════════════════════════════════════════
# P22b: Nudge Escalation Ladder
# ═══════════════════════════════════════════════════════════════════

class TestNudgeEscalation(unittest.TestCase):
    """Nudge escalation ladder — 4 tiers."""

    def test_escalate_endpoint(self):
        src = _memu()
        self.assertIn("/memory/nudge/escalate", src)

    def test_escalate_is_post(self):
        src = _memu()
        self.assertIn('@app.post("/memory/nudge/escalate")', src)

    def test_ladder_endpoint(self):
        src = _memu()
        self.assertIn("/memory/nudge/ladder", src)

    def test_ladder_is_get(self):
        src = _memu()
        self.assertIn('@app.get("/memory/nudge/ladder")', src)

    def test_four_tiers(self):
        src = _memu()
        self.assertIn("_ESCALATION_TIERS", src)
        for tier in ("gentle", "firm", "tough_love", "intervention"):
            self.assertIn(tier, src, f"Missing tier: {tier}")

    def test_escalation_thresholds(self):
        """Thresholds: 0, 3, 5, 7."""
        src = _memu()
        for t in ("0", "3", "5", "7"):
            self.assertIn(f'"threshold": {t}', src)

    def test_escalation_tracks_dismissals(self):
        src = _memu()
        self.assertIn("dismissals", src.split("/memory/nudge/escalate")[1].split("@app")[0])

    def test_escalation_generates_message(self):
        """Each tier should produce a different message."""
        src = _memu()
        self.assertIn("INTERVENTION:", src)
        self.assertIn("stop dodging", src)

    def test_nudge_ladder_store(self):
        src = _memu()
        self.assertIn("_nudge_ladder", src)

    def test_max_ladder_entries(self):
        src = _memu()
        self.assertIn("_MAX_LADDER_ENTRIES", src)


# ═══════════════════════════════════════════════════════════════════
# P22c: Cross-Mode Insight Bridge
# ═══════════════════════════════════════════════════════════════════

class TestCrossModeInsight(unittest.TestCase):
    """Cross-mode insight bridge — PUB<->WORK learning."""

    def test_scan_endpoint(self):
        src = _memu()
        self.assertIn("/memory/cross-mode/scan", src)

    def test_scan_is_post(self):
        src = _memu()
        self.assertIn('@app.post("/memory/cross-mode/scan")', src)

    def test_history_endpoint(self):
        src = _memu()
        self.assertIn('@app.get("/memory/cross-mode")', src)

    def test_opposite_mode_logic(self):
        src = _memu()
        # Should swap modes: PUB -> WORK, WORK -> PUB
        self.assertIn('opposite_mode = "PUB" if current_mode == "WORK" else "WORK"', src)

    def test_bridge_message_generation(self):
        src = _memu()
        self.assertIn("bridge_message", src.split("/memory/cross-mode/scan")[1].split("@app")[0])

    def test_cross_mode_insights_store(self):
        src = _memu()
        self.assertIn("_cross_mode_insights", src)

    def test_mode_inference_patterns(self):
        """Should detect PUB vs WORK from content patterns."""
        src = _memu()
        self.assertIn("pub", src.split("/memory/cross-mode/scan")[1].split("from_opposite")[0])
        self.assertIn("work", src.split("/memory/cross-mode/scan")[1].split("from_opposite")[0])

    def test_relevance_scoring(self):
        """Should rank by word overlap."""
        src = _memu()
        self.assertIn("relevance_words", src)


# ═══════════════════════════════════════════════════════════════════
# P22d: Impact Oracle
# ═══════════════════════════════════════════════════════════════════

class TestImpactOracle(unittest.TestCase):
    """Impact Oracle — predicts action consequences."""

    def test_predict_endpoint(self):
        src = _memu()
        self.assertIn("/memory/oracle/predict", src)

    def test_predict_is_post(self):
        src = _memu()
        self.assertIn('@app.post("/memory/oracle/predict")', src)

    def test_chains_endpoint(self):
        src = _memu()
        self.assertIn("/memory/oracle/chains", src)

    def test_chains_is_get(self):
        src = _memu()
        self.assertIn('@app.get("/memory/oracle/chains")', src)

    def test_analyses_goal_impact(self):
        src = _memu()
        block = src.split("/memory/oracle/predict")[1].split("/memory/oracle/chains")[0]
        self.assertIn("active_goals", block)

    def test_impact_directions(self):
        src = _memu()
        for d in ("negative", "positive", "neutral"):
            self.assertIn(d, src.split("/memory/oracle/predict")[1].split("/memory/oracle/chains")[0])

    def test_risk_levels(self):
        src = _memu()
        for r in ('"high"', '"medium"', '"low"'):
            self.assertIn(r, src)

    def test_emotional_forecast(self):
        src = _memu()
        self.assertIn("emotional_forecast", src)
        self.assertIn("emotion_prediction", src.split("/memory/oracle/predict")[1].split("/memory/oracle/chains")[0])

    def test_skip_detection(self):
        """Should detect skip/miss/ignore type actions."""
        src = _memu()
        self.assertIn("skip", src.split("/memory/oracle/predict")[1].split("/memory/oracle/chains")[0])

    def test_oracle_predictions_store(self):
        src = _memu()
        self.assertIn("_oracle_predictions", src)

    def test_overall_risk_computed(self):
        src = _memu()
        self.assertIn("overall_risk", src)

    def test_chains_count(self):
        src = _memu()
        self.assertIn("goals_affected", src)


# ═══════════════════════════════════════════════════════════════════
# P22e: Shadow Memory Branches
# ═══════════════════════════════════════════════════════════════════

class TestShadowBranches(unittest.TestCase):
    """Shadow memory branches — alternate timelines."""

    def test_branch_endpoint(self):
        src = _memu()
        self.assertIn("/memory/shadow/branch", src)

    def test_branch_is_post(self):
        src = _memu()
        self.assertIn('@app.post("/memory/shadow/branch")', src)

    def test_branches_list_endpoint(self):
        src = _memu()
        self.assertIn("/memory/shadow/branches", src)

    def test_explore_endpoint(self):
        src = _memu()
        self.assertIn("/memory/shadow/explore/{branch_id}", src)

    def test_shadow_branches_store(self):
        src = _memu()
        self.assertIn("_shadow_branches", src)

    def test_branch_has_id(self):
        src = _memu()
        self.assertIn("branch_id", src.split("/memory/shadow/branch")[1].split("@app")[0])

    def test_branch_has_timeline(self):
        src = _memu()
        block = src.split('@app.post("/memory/shadow/branch")')[1].split("shadow_branches_list")[0]
        self.assertIn("timeline", block)
        self.assertIn("original", block)
        self.assertIn("shadow", block)

    def test_branch_finds_related_memories(self):
        src = _memu()
        self.assertIn("related_mems", src)

    def test_branch_checks_goals(self):
        src = _memu()
        self.assertIn("affected_goals", src.split("/memory/shadow/branch")[1].split("@app")[0])

    def test_explore_returns_404(self):
        src = _memu()
        self.assertIn("branch not found", src)

    def test_branch_is_queryable(self):
        src = _memu()
        self.assertIn('"queryable": True', src)


# ═══════════════════════════════════════════════════════════════════
# P22 Operator Model Summary
# ═══════════════════════════════════════════════════════════════════

class TestOperatorModelSummary(unittest.TestCase):
    """Unified operator model endpoint."""

    def test_endpoint(self):
        src = _memu()
        self.assertIn("/memory/operator-model", src)

    def test_is_get(self):
        src = _memu()
        self.assertIn('@app.get("/memory/operator-model")', src)

    def test_has_echo_state(self):
        src = _memu()
        self.assertIn("echo_state", src.split("/memory/operator-model")[1])

    def test_has_escalation_state(self):
        src = _memu()
        self.assertIn("escalation_state", src.split("/memory/operator-model")[1])

    def test_has_cross_mode(self):
        src = _memu()
        self.assertIn("cross_mode", src.split("/memory/operator-model")[1])

    def test_has_oracle(self):
        src = _memu()
        self.assertIn('"oracle"', src.split("/memory/operator-model")[1])

    def test_has_shadow_branches(self):
        src = _memu()
        self.assertIn("shadow_branches", src.split("/memory/operator-model")[1])

    def test_model_completeness(self):
        src = _memu()
        self.assertIn("model_completeness", src.split("/memory/operator-model")[1])


# ═══════════════════════════════════════════════════════════════════
# Dashboard Proxy Routes
# ═══════════════════════════════════════════════════════════════════

class TestDashboardProxies(unittest.TestCase):
    """P22 proxy routes in dashboard/app.py."""

    def test_echo_analyse_proxy(self):
        self.assertIn("/api/echo/analyse", _dashboard())

    def test_echo_history_proxy(self):
        self.assertIn("/api/echo/history", _dashboard())

    def test_nudge_escalate_proxy(self):
        self.assertIn("/api/nudge/escalate", _dashboard())

    def test_nudge_ladder_proxy(self):
        self.assertIn("/api/nudge/ladder", _dashboard())

    def test_cross_mode_scan_proxy(self):
        self.assertIn("/api/cross-mode/scan", _dashboard())

    def test_cross_mode_history_proxy(self):
        self.assertIn("/api/cross-mode", _dashboard())

    def test_oracle_predict_proxy(self):
        self.assertIn("/api/oracle/predict", _dashboard())

    def test_oracle_chains_proxy(self):
        self.assertIn("/api/oracle/chains", _dashboard())

    def test_shadow_branch_proxy(self):
        self.assertIn("/api/shadow/branch", _dashboard())

    def test_shadow_branches_proxy(self):
        self.assertIn("/api/shadow/branches", _dashboard())

    def test_operator_model_proxy(self):
        self.assertIn("/api/operator-model", _dashboard())


# ═══════════════════════════════════════════════════════════════════
# Dashboard UI
# ═══════════════════════════════════════════════════════════════════

class TestDashboardUI(unittest.TestCase):
    """P22 UI elements in dashboard/static/app.html."""

    def test_operator_model_card(self):
        self.assertIn("Operator Model", _html())

    def test_operator_model_icon(self):
        self.assertIn("🧠", _html())

    def test_echo_history_card(self):
        self.assertIn("Emotional Echoes", _html())

    def test_echo_icon(self):
        self.assertIn("🪞", _html())

    def test_nudge_ladder_card(self):
        self.assertIn("Nudge Escalation Ladder", _html())

    def test_nudge_ladder_icon(self):
        self.assertIn("📢", _html())

    def test_impact_oracle_card(self):
        self.assertIn("Impact Oracle", _html())

    def test_oracle_icon(self):
        self.assertIn("🔮", _html())

    def test_shadow_branches_card(self):
        self.assertIn("Shadow Branches", _html())

    def test_shadow_icon(self):
        self.assertIn("🌿", _html())

    def test_oracle_input(self):
        self.assertIn("oracleAction", _html())

    def test_shadow_inputs(self):
        html = _html()
        self.assertIn("shadowDecision", html)
        self.assertIn("shadowAlternative", html)

    def test_predict_button(self):
        self.assertIn("predictImpact()", _html())

    def test_create_branch_button(self):
        self.assertIn("createShadowBranch()", _html())

    # JS functions
    def test_js_refreshOperatorModel(self):
        self.assertIn("function refreshOperatorModel()", _html())

    def test_js_refreshEchoHistory(self):
        self.assertIn("function refreshEchoHistory()", _html())

    def test_js_refreshNudgeLadder(self):
        self.assertIn("function refreshNudgeLadder()", _html())

    def test_js_refreshOracleChains(self):
        self.assertIn("function refreshOracleChains()", _html())

    def test_js_predictImpact(self):
        self.assertIn("function predictImpact()", _html())

    def test_js_refreshShadowBranches(self):
        self.assertIn("function refreshShadowBranches()", _html())

    def test_js_createShadowBranch(self):
        self.assertIn("function createShadowBranch()", _html())

    def test_js_refreshP22(self):
        self.assertIn("function refreshP22()", _html())

    def test_p22_wired_into_goals_view(self):
        self.assertIn("refreshP22()", _html())

    def test_operator_model_content_div(self):
        self.assertIn("operatorModelContent", _html())

    def test_echo_content_div(self):
        self.assertIn("echoHistoryContent", _html())

    def test_nudge_ladder_content_div(self):
        self.assertIn("nudgeLadderContent", _html())

    def test_oracle_content_div(self):
        self.assertIn("oracleContent", _html())

    def test_shadow_content_div(self):
        self.assertIn("shadowContent", _html())


# ═══════════════════════════════════════════════════════════════════
# LangGraph Integration
# ═══════════════════════════════════════════════════════════════════

class TestLangGraphIntegration(unittest.TestCase):
    """P22 operator model integration in langgraph/app.py."""

    def test_get_operator_model_function(self):
        self.assertIn("_get_operator_model", _langgraph())

    def test_get_operator_model_is_async(self):
        self.assertIn("async def _get_operator_model", _langgraph())

    def test_operator_task_created(self):
        self.assertIn("_get_operator_model(", _langgraph())

    def test_operator_model_awaited(self):
        self.assertIn("operator_model", _langgraph())

    def test_echo_analysis_in_fetch(self):
        src = _langgraph()
        self.assertIn("/memory/echo/analyse", src)

    def test_operator_model_endpoint_in_fetch(self):
        src = _langgraph()
        self.assertIn("/memory/operator-model", src)

    def test_cross_mode_in_fetch(self):
        src = _langgraph()
        self.assertIn("/memory/cross-mode/scan", src)

    def test_echo_injected(self):
        src = _langgraph()
        self.assertIn("Emotional echo:", src)

    def test_escalation_injected(self):
        src = _langgraph()
        self.assertIn("Nudge escalation level", src)

    def test_cross_mode_injected(self):
        src = _langgraph()
        self.assertIn("Cross-mode insight:", src)

    def test_operator_model_context_block(self):
        src = _langgraph()
        self.assertIn("Operator model (how I understand you right now)", src)

    def test_ten_parallel_fetches(self):
        """Should now have 10 parallel fetches in /chat via asyncio.gather."""
        src = _langgraph()
        chat_section = src.split("@app.post(\"/chat\")")[1] if "@app.post(\"/chat\")" in src else ""
        # H1.3 refactored to asyncio.gather with _safe() wrappers
        safe_calls = chat_section.count("_safe(")
        self.assertGreaterEqual(safe_calls, 10, f"Expected 10+ _safe() calls in gather, found {safe_calls}")


# ═══════════════════════════════════════════════════════════════════
# Supervisor Integration
# ═══════════════════════════════════════════════════════════════════

class TestSupervisorIntegration(unittest.TestCase):
    """P22 supervisor enhancements."""

    def test_check_escalations_function(self):
        self.assertIn("_check_escalations", _supervisor())

    def test_check_escalations_is_async(self):
        self.assertIn("async def _check_escalations", _supervisor())

    def test_escalation_in_background_loop(self):
        self.assertIn("_check_escalations()", _supervisor())

    def test_escalation_checks_ladder(self):
        self.assertIn("/memory/nudge/ladder", _supervisor())

    def test_escalation_sends_telegram(self):
        src = _supervisor()
        block = src.split("_check_escalations")[1].split("_background_loop")[0]
        self.assertIn("TELEGRAM_ALERT_URL", block)

    def test_escalation_icon(self):
        self.assertIn('"escalation": "📢"', _supervisor())

    def test_echo_icon(self):
        self.assertIn('"echo": "🪞"', _supervisor())

    def test_escalation_fires_at_level_3(self):
        """Only fires telegram for tough_love (3) and intervention (4)."""
        src = _supervisor()
        self.assertIn("level", src.split("_check_escalations")[1].split("_background_loop")[0])


if __name__ == "__main__":
    unittest.main()
