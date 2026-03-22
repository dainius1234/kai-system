"""P21 Proactive Agent Loop — test suite.

Tests: scheduled tasks, reminders, briefings, action registry, agent summary,
       dashboard proxies, dashboard UI, langgraph integration, supervisor integration.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── helpers ──────────────────────────────────────────────────────────

def _load_memu_module():
    """Import memu-core/app.py with mocked dependencies."""
    with patch.dict(os.environ, {"PORT": "8001", "VECTOR_STORE": "memory"}):
        # stub heavy deps
        for mod in [
            "common.runtime", "common.auth", "common.llm", "common.rate_limit",
            "common.policy", "common.self_emp_advisor", "pydantic",
        ]:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

        # ensure pydantic BaseModel exists
        pydantic_mock = sys.modules["pydantic"]
        pydantic_mock.BaseModel = type("BaseModel", (), {})
        pydantic_mock.Field = lambda *a, **kw: None

        runtime_mock = sys.modules["common.runtime"]
        runtime_mock.detect_device.return_value = "cpu"
        runtime_mock.setup_json_logger.return_value = MagicMock()
        runtime_mock.sanitize_string = lambda s: s
        runtime_mock.AuditStream = MagicMock
        runtime_mock.ErrorBudget = MagicMock

        spec = __import__("importlib").util.spec_from_file_location(
            "memu_app", str(ROOT / "memu-core" / "app.py"),
        )
        mod = __import__("importlib").util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass
        return mod


def _read_file(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


MEMU_SRC = _read_file("memu-core/app.py")
LANG_SRC = _read_file("langgraph/app.py")
DASH_SRC = _read_file("dashboard/app.py")
HTML_SRC = _read_file("dashboard/static/app.html")
SUP_SRC = _read_file("supervisor/app.py")


# ═══════════════════════════════════════════════════════════════════
# Test classes
# ═══════════════════════════════════════════════════════════════════


class TestActionRegistry(unittest.TestCase):
    """P21a: Action discovery."""

    def test_actions_endpoint_exists(self):
        self.assertIn("/memory/actions", MEMU_SRC)

    def test_has_multiple_actions(self):
        self.assertIn("_ACTION_REGISTRY", MEMU_SRC)
        self.assertIn("set_reminder", MEMU_SRC)
        self.assertIn("morning_briefing", MEMU_SRC)
        self.assertIn("evening_checkin", MEMU_SRC)
        self.assertIn("check_conscience", MEMU_SRC)
        self.assertIn("record_gratitude", MEMU_SRC)

    def test_action_has_endpoint(self):
        self.assertIn('"endpoint"', MEMU_SRC)
        self.assertIn('"description"', MEMU_SRC)

    def test_action_count_minimum(self):
        count = MEMU_SRC.count('"action":')
        self.assertGreaterEqual(count, 10)


class TestScheduledTasks(unittest.TestCase):
    """P21b: Scheduled tasks engine."""

    def test_schedule_task_endpoint(self):
        self.assertIn("POST /memory/schedule/task", MEMU_SRC.replace("@app.post", "POST"))
        self.assertIn("/memory/schedule/task", MEMU_SRC)

    def test_list_tasks_endpoint(self):
        self.assertIn("/memory/schedule/tasks", MEMU_SRC)

    def test_cancel_task_endpoint(self):
        self.assertIn("/memory/schedule/task/{task_id}/cancel", MEMU_SRC)

    def test_fire_task_endpoint(self):
        self.assertIn("/memory/schedule/task/{task_id}/fire", MEMU_SRC)

    def test_due_tasks_endpoint(self):
        self.assertIn("/memory/schedule/due", MEMU_SRC)

    def test_valid_frequencies(self):
        self.assertIn("_VALID_FREQUENCIES", MEMU_SRC)
        self.assertIn("daily", MEMU_SRC)
        self.assertIn("weekly", MEMU_SRC)
        self.assertIn("monthly", MEMU_SRC)
        self.assertIn("hourly", MEMU_SRC)

    def test_task_has_title(self):
        self.assertIn('"title"', MEMU_SRC)

    def test_task_has_frequency(self):
        self.assertIn('"frequency"', MEMU_SRC)

    def test_max_scheduled_cap(self):
        self.assertIn("_MAX_SCHEDULED", MEMU_SRC)

    def test_task_deactivated_on_once_fire(self):
        # When frequency is "once" and fired, task should be deactivated
        self.assertIn('task["active"] = False', MEMU_SRC)


class TestReminders(unittest.TestCase):
    """P21c: Reminder system."""

    def test_set_reminder_endpoint(self):
        self.assertIn("/memory/reminders/set", MEMU_SRC)

    def test_list_reminders_endpoint(self):
        self.assertIn('"/memory/reminders"', MEMU_SRC)

    def test_due_reminders_endpoint(self):
        self.assertIn("/memory/reminders/due", MEMU_SRC)

    def test_fire_reminder_endpoint(self):
        self.assertIn("/memory/reminders/{reminder_id}/fire", MEMU_SRC)

    def test_cancel_reminder_endpoint(self):
        self.assertIn("/memory/reminders/{reminder_id}/cancel", MEMU_SRC)

    def test_reminder_has_text(self):
        self.assertIn('"text":', MEMU_SRC)

    def test_reminder_has_fire_at(self):
        self.assertIn('"fire_at"', MEMU_SRC)

    def test_reminder_repeat_support(self):
        self.assertIn('"repeat"', MEMU_SRC)

    def test_max_reminders_cap(self):
        self.assertIn("_MAX_REMINDERS", MEMU_SRC)

    def test_default_fire_at_1_hour(self):
        # Default fire_at is 1 hour from now
        self.assertIn("timedelta(hours=1)", MEMU_SRC)


class TestMorningBriefing(unittest.TestCase):
    """P21d: Morning briefing and evening check-in."""

    def test_morning_briefing_endpoint(self):
        self.assertIn("/memory/briefing/morning", MEMU_SRC)

    def test_evening_checkin_endpoint(self):
        self.assertIn("/memory/briefing/evening", MEMU_SRC)

    def test_briefing_history_endpoint(self):
        self.assertIn("/memory/briefing/history", MEMU_SRC)

    def test_morning_has_goals_section(self):
        self.assertIn('"goals"', MEMU_SRC)
        self.assertIn("Active Goals", MEMU_SRC)

    def test_morning_has_reminders_section(self):
        self.assertIn("Upcoming Reminders", MEMU_SRC)

    def test_morning_has_emotional_arc(self):
        self.assertIn("emotional_arc", MEMU_SRC)
        self.assertIn("Yesterday's Emotional Arc", MEMU_SRC)

    def test_morning_has_nudges_section(self):
        self.assertIn("Things To Keep In Mind", MEMU_SRC)

    def test_morning_has_scheduled_section(self):
        self.assertIn("Scheduled For Today", MEMU_SRC)

    def test_evening_has_activity_count(self):
        self.assertIn("Today's Activity", MEMU_SRC)

    def test_evening_has_reflection_prompt(self):
        self.assertIn("What went well today", MEMU_SRC)

    def test_evening_has_tomorrow_preview(self):
        self.assertIn("Tomorrow's Schedule", MEMU_SRC)

    def test_time_of_day_greeting(self):
        self.assertIn("Good morning", MEMU_SRC)
        self.assertIn("Afternoon check-in", MEMU_SRC)
        self.assertIn("Evening briefing", MEMU_SRC)

    def test_briefing_log_exists(self):
        self.assertIn("_briefing_log", MEMU_SRC)


class TestAgentSummary(unittest.TestCase):
    """P21e: Agent summary endpoint."""

    def test_summary_endpoint(self):
        self.assertIn("/memory/agent/summary", MEMU_SRC)

    def test_summary_has_capabilities(self):
        self.assertIn('"capabilities"', MEMU_SRC)

    def test_summary_has_scheduled(self):
        self.assertIn('"scheduled_tasks"', MEMU_SRC)

    def test_summary_has_reminders_count(self):
        self.assertIn('"pending"', MEMU_SRC)
        self.assertIn('"due_now"', MEMU_SRC)

    def test_summary_has_briefings_count(self):
        self.assertIn('"briefings_generated"', MEMU_SRC)


class TestDashboardProxies(unittest.TestCase):
    """P21: Dashboard proxy routes."""

    def test_actions_proxy(self):
        self.assertIn("/api/actions", DASH_SRC)

    def test_schedule_task_proxy(self):
        self.assertIn("/api/schedule/task", DASH_SRC)

    def test_schedule_tasks_list_proxy(self):
        self.assertIn("/api/schedule/tasks", DASH_SRC)

    def test_cancel_task_proxy(self):
        self.assertIn("/api/schedule/task/{task_id}/cancel", DASH_SRC)

    def test_set_reminder_proxy(self):
        self.assertIn("/api/reminders/set", DASH_SRC)

    def test_list_reminders_proxy(self):
        self.assertIn('"/api/reminders"', DASH_SRC)

    def test_cancel_reminder_proxy(self):
        self.assertIn("/api/reminders/{reminder_id}/cancel", DASH_SRC)

    def test_morning_briefing_proxy(self):
        self.assertIn("/api/briefing/morning", DASH_SRC)

    def test_evening_checkin_proxy(self):
        self.assertIn("/api/briefing/evening", DASH_SRC)

    def test_agent_summary_proxy(self):
        self.assertIn("/api/agent/summary", DASH_SRC)


class TestDashboardUI(unittest.TestCase):
    """P21: Dashboard Goals view enhancements."""

    def test_reminders_card(self):
        self.assertIn("remindersContent", HTML_SRC)

    def test_reminder_input(self):
        self.assertIn("reminderText", HTML_SRC)

    def test_reminder_time_input(self):
        self.assertIn("reminderTime", HTML_SRC)

    def test_reminder_repeat_select(self):
        self.assertIn("reminderRepeat", HTML_SRC)

    def test_set_reminder_button(self):
        self.assertIn("setReminder()", HTML_SRC)

    def test_scheduled_card(self):
        self.assertIn("scheduledContent", HTML_SRC)

    def test_schedule_task_button(self):
        self.assertIn("scheduleTask()", HTML_SRC)

    def test_agent_summary_card(self):
        self.assertIn("agentSummaryContent", HTML_SRC)

    def test_morning_briefing_button(self):
        self.assertIn("triggerBriefing('morning')", HTML_SRC)

    def test_evening_checkin_button(self):
        self.assertIn("triggerBriefing('evening')", HTML_SRC)

    # JS functions
    def test_refresh_reminders_function(self):
        self.assertIn("async function refreshReminders()", HTML_SRC)

    def test_set_reminder_function(self):
        self.assertIn("async function setReminder()", HTML_SRC)

    def test_cancel_reminder_function(self):
        self.assertIn("async function cancelReminder(", HTML_SRC)

    def test_refresh_scheduled_function(self):
        self.assertIn("async function refreshScheduled()", HTML_SRC)

    def test_schedule_task_function(self):
        self.assertIn("async function scheduleTask()", HTML_SRC)

    def test_cancel_scheduled_function(self):
        self.assertIn("async function cancelScheduled(", HTML_SRC)

    def test_refresh_agent_summary_function(self):
        self.assertIn("async function refreshAgentSummary()", HTML_SRC)

    def test_trigger_briefing_function(self):
        self.assertIn("async function triggerBriefing(", HTML_SRC)

    def test_refresh_p21_function(self):
        self.assertIn("function refreshP21()", HTML_SRC)

    def test_p21_wired_into_goals(self):
        self.assertIn("refreshP21()", HTML_SRC)

    def test_reminder_icon(self):
        self.assertIn("⏰", HTML_SRC)

    def test_scheduled_icon(self):
        self.assertIn("📅", HTML_SRC)

    def test_agent_icon(self):
        self.assertIn("🤖", HTML_SRC)


class TestLangGraphIntegration(unittest.TestCase):
    """P21: LangGraph 9th parallel fetch."""

    def test_get_agent_context_exists(self):
        self.assertIn("_get_agent_context", LANG_SRC)

    def test_agent_context_is_async(self):
        self.assertIn("async def _get_agent_context", LANG_SRC)

    def test_agent_task_created(self):
        self.assertIn("agent_task = asyncio.create_task(_get_agent_context())", LANG_SRC)

    def test_agent_ctx_awaited(self):
        self.assertIn("agent_ctx = await agent_task", LANG_SRC)

    def test_agent_context_injected(self):
        self.assertIn("Agent capabilities & schedule", LANG_SRC)

    def test_fetches_due_tasks(self):
        self.assertIn("/memory/schedule/due", LANG_SRC)

    def test_fetches_due_reminders(self):
        self.assertIn("/memory/reminders/due", LANG_SRC)

    def test_fetches_agent_summary(self):
        self.assertIn("/memory/agent/summary", LANG_SRC)


class TestSupervisorIntegration(unittest.TestCase):
    """P21: Supervisor fires due items."""

    def test_fire_due_items_function(self):
        self.assertIn("_fire_due_items", SUP_SRC)

    def test_fire_due_items_is_async(self):
        self.assertIn("async def _fire_due_items", SUP_SRC)

    def test_fire_due_in_background_loop(self):
        self.assertIn("await _fire_due_items()", SUP_SRC)

    def test_fires_reminders(self):
        self.assertIn("/memory/reminders/due", SUP_SRC)

    def test_fires_scheduled_tasks(self):
        self.assertIn("/memory/schedule/due", SUP_SRC)

    def test_marks_reminders_fired(self):
        self.assertIn("/memory/reminders/", SUP_SRC)
        self.assertIn("/fire", SUP_SRC)

    def test_marks_tasks_fired(self):
        self.assertIn("/memory/schedule/task/", SUP_SRC)

    def test_scheduled_icon_in_supervisor(self):
        self.assertIn('"scheduled_task"', SUP_SRC)
        self.assertIn('"briefing"', SUP_SRC)

    def test_reminder_telegram_icon(self):
        self.assertIn("*Reminder:*", SUP_SRC)

    def test_scheduled_telegram_icon(self):
        self.assertIn("*Scheduled:*", SUP_SRC)


if __name__ == "__main__":
    unittest.main()
