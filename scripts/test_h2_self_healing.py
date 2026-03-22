"""H2 Self-Healing & Resilience Sprint — regression tests.

Validates the dual-layer resilience architecture:
  Layer 1 (Process): deep /health, /recover endpoints, resilient_call
  Layer 2 (System):  supervisor recovery actions, TaskWatchdog, fleet history

Tests verify code patterns exist in the right places and the shared
resilience primitives work correctly.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


COMMON_RESILIENCE = _read("common/resilience.py")
SUPERVISOR_SRC = _read("supervisor/app.py")
MEMU_SRC = _read("memu-core/app.py")
EXEC_SRC = _read("executor/app.py")
LANG_SRC = _read("langgraph/app.py")
TOOL_GATE_SRC = _read("tool-gate/app.py")
HEARTBEAT_SRC = _read("heartbeat/app.py")
DASHBOARD_SRC = _read("dashboard/app.py")


# ═══════════════════════════════════════════════════════════════════
# Shared Resilience Primitives (common/resilience.py)
# ═══════════════════════════════════════════════════════════════════

class TestResilienceModule(unittest.TestCase):

    def test_resilient_call_defined(self):
        self.assertIn("async def resilient_call(", COMMON_RESILIENCE)

    def test_resilient_call_has_retry(self):
        self.assertIn("retries", COMMON_RESILIENCE)
        self.assertIn("backoff", COMMON_RESILIENCE)

    def test_resilient_call_has_circuit_breaker(self):
        self.assertIn("cb.allow()", COMMON_RESILIENCE)
        self.assertIn("cb.record_success()", COMMON_RESILIENCE)
        self.assertIn("cb.record_failure()", COMMON_RESILIENCE)

    def test_service_health_class(self):
        self.assertIn("class ServiceHealth:", COMMON_RESILIENCE)

    def test_service_health_probe(self):
        self.assertIn("async def probe(self)", COMMON_RESILIENCE)

    def test_task_watchdog_class(self):
        self.assertIn("class TaskWatchdog:", COMMON_RESILIENCE)

    def test_task_watchdog_methods(self):
        self.assertIn("def heartbeat(self, task_name", COMMON_RESILIENCE)
        self.assertIn("def frozen(self)", COMMON_RESILIENCE)
        self.assertIn("def snapshot(self)", COMMON_RESILIENCE)

    def test_get_breaker_snapshot(self):
        self.assertIn("def get_breaker_snapshot()", COMMON_RESILIENCE)


class TestResilienceRuntime(unittest.TestCase):
    """Actually run the resilience primitives, not just pattern-match."""

    def test_task_watchdog_heartbeat_and_frozen(self):
        from common.resilience import TaskWatchdog
        wd = TaskWatchdog(stale_seconds=0.1)
        wd.heartbeat("loop_a")
        self.assertEqual(wd.frozen(), [])
        time.sleep(0.2)
        self.assertEqual(wd.frozen(), ["loop_a"])

    def test_task_watchdog_snapshot(self):
        from common.resilience import TaskWatchdog
        wd = TaskWatchdog(stale_seconds=60.0)
        wd.heartbeat("bg")
        snap = wd.snapshot()
        self.assertIn("bg", snap)
        self.assertFalse(snap["bg"]["frozen"])

    def test_service_health_probe(self):
        from common.resilience import ServiceHealth

        async def _ok():
            return True

        async def _fail():
            raise RuntimeError("boom")

        sh = ServiceHealth(service_name="test-svc")
        sh.register("dep_ok", _ok)
        sh.register("dep_fail", _fail)
        result = asyncio.get_event_loop().run_until_complete(sh.probe())
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["checks"]["dep_ok"], "ok")
        self.assertTrue(result["checks"]["dep_fail"].startswith("fail"))

    def test_resilient_call_fallback_on_bad_url(self):
        from common.resilience import resilient_call
        result = asyncio.get_event_loop().run_until_complete(
            resilient_call("GET", "http://127.0.0.1:1/nonexistent",
                           retries=1, backoff=0.01, fallback={"fallback": True})
        )
        self.assertEqual(result, {"fallback": True})


# ═══════════════════════════════════════════════════════════════════
# Layer 2: Supervisor Self-Heal (supervisor/app.py)
# ═══════════════════════════════════════════════════════════════════

class TestSupervisorSelfHeal(unittest.TestCase):

    def test_imports_task_watchdog(self):
        self.assertIn("from common.resilience import TaskWatchdog", SUPERVISOR_SRC)

    def test_watchdog_instantiated(self):
        self.assertIn("_watchdog = TaskWatchdog(", SUPERVISOR_SRC)

    def test_recovery_actions_registry(self):
        self.assertIn("RECOVERY_ACTIONS", SUPERVISOR_SRC)
        self.assertIn("/recover", SUPERVISOR_SRC)

    def test_attempt_recovery_function(self):
        self.assertIn("async def _attempt_recovery(", SUPERVISOR_SRC)

    def test_recovery_cooldown(self):
        self.assertIn("RECOVERY_COOLDOWN", SUPERVISOR_SRC)

    def test_fleet_history_tracking(self):
        self.assertIn("_fleet_history", SUPERVISOR_SRC)
        self.assertIn("FLEET_HISTORY_MAX", SUPERVISOR_SRC)

    def test_background_loop_beats_watchdog(self):
        loop_section = SUPERVISOR_SRC.split("async def _background_loop")[1].split("@app.")[0]
        self.assertIn('_watchdog.heartbeat("main_loop")', loop_section)

    def test_health_checks_frozen_tasks(self):
        health_section = SUPERVISOR_SRC.split('def health')[1].split("@app.")[0]
        self.assertIn("_watchdog.frozen()", health_section)

    def test_health_returns_degraded_on_frozen(self):
        health_section = SUPERVISOR_SRC.split('def health')[1].split("@app.")[0]
        self.assertIn('"degraded"', health_section)

    def test_fleet_history_endpoint(self):
        self.assertIn("/fleet/history", SUPERVISOR_SRC)

    def test_watchdog_endpoint(self):
        self.assertIn("/watchdog", SUPERVISOR_SRC)

    def test_manual_recover_endpoint(self):
        self.assertIn('/recover/{service_name}', SUPERVISOR_SRC)

    def test_sweep_records_history(self):
        sweep = SUPERVISOR_SRC.split("async def _sweep")[1].split("async def _")[0]
        self.assertIn("_fleet_history.append(", sweep)

    def test_sweep_triggers_recovery(self):
        sweep = SUPERVISOR_SRC.split("async def _sweep")[1].split("async def _")[0]
        self.assertIn("_attempt_recovery(", sweep)

    def test_check_service_detects_degraded(self):
        check = SUPERVISOR_SRC.split("async def _check_service")[1].split("async def _")[0]
        self.assertIn('"degraded"', check)


# ═══════════════════════════════════════════════════════════════════
# Layer 1: Deep /health on Core Services
# ═══════════════════════════════════════════════════════════════════

class TestDeepHealthEndpoints(unittest.TestCase):

    def test_memu_health_is_deep(self):
        health = MEMU_SRC.split("def health")[1].split("@app.")[0]
        self.assertIn("checks", health)
        self.assertIn("degraded", health)

    def test_memu_has_recover(self):
        self.assertIn("/recover", MEMU_SRC)
        self.assertIn("async def recover", MEMU_SRC)

    def test_executor_health_is_deep(self):
        health = EXEC_SRC.split("def health")[1].split("@app.")[0]
        self.assertIn("checks", health)
        self.assertIn("disk_space", health)
        self.assertIn("degraded", health)

    def test_executor_has_recover(self):
        self.assertIn("async def recover", EXEC_SRC)

    def test_langgraph_health_is_deep(self):
        health = LANG_SRC.split("def health")[1].split("@app.")[0]
        self.assertIn("degraded", health)
        self.assertIn("MEMU_BREAKER", health)

    def test_langgraph_has_recover(self):
        self.assertIn("async def recover", LANG_SRC)
        self.assertIn("breakers_reset", LANG_SRC)

    def test_toolgate_health_is_deep(self):
        health = TOOL_GATE_SRC.split("def health")[1].split("@app.")[0]
        self.assertIn("redis", health)
        self.assertIn("ledger", health)
        self.assertIn("degraded", health)

    def test_toolgate_has_recover(self):
        self.assertIn("async def recover", TOOL_GATE_SRC)

    def test_heartbeat_health_is_deep(self):
        health = HEARTBEAT_SRC.split("def health")[1].split("@app.")[0]
        self.assertIn("checks", health)
        self.assertIn("stale", health)
        self.assertIn("degraded", health)

    def test_heartbeat_has_recover(self):
        self.assertIn("async def recover", HEARTBEAT_SRC)


# ═══════════════════════════════════════════════════════════════════
# Resilient Inter-Service Calls
# ═══════════════════════════════════════════════════════════════════

class TestResilientCalls(unittest.TestCase):

    def test_executor_imports_resilient_call(self):
        self.assertIn("from common.resilience import resilient_call", EXEC_SRC)

    def test_executor_uses_resilient_call(self):
        self.assertIn("resilient_call(", EXEC_SRC)

    def test_dashboard_imports_resilient_call(self):
        self.assertIn("from common.resilience import resilient_call", DASHBOARD_SRC)

    def test_dashboard_proxy_uses_resilient_call(self):
        proxy_section = DASHBOARD_SRC.split("_proxy_get")[1].split("@app.")[0]
        self.assertIn("resilient_call(", proxy_section)

    def test_dashboard_proxy_post_uses_resilient_call(self):
        proxy_section = DASHBOARD_SRC.split("_proxy_post")[1].split("@app.")[0]
        self.assertIn("resilient_call(", proxy_section)


if __name__ == "__main__":
    unittest.main()
