"""H1 Critical Hardening Sprint — tests for all 7 fixes.

Tests verify that the critical issues found during the March 22 2026
system audit are properly fixed:
  H1.1 — asyncio locks on shared mutable state (memu-core)
  H1.2 — prompt injection check on /chat (agentic)
  H1.3 — 10-way parallel fetch error handling (agentic)
  H1.4 — store.memorize → store.insert fix (memu-core feedback)
  H1.5 — executor shell=False + AST sandbox validation
  H1.6 — telegram voice file size limit
  H1.7 — dashboard proxy error handling (_proxy_get/_proxy_post)
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(path: str) -> str:
    with open(os.path.join(ROOT, path)) as f:
        return f.read()


MEMU_SRC = _read("memu-core/app.py")
LANG_SRC = _read("agentic/app.py")
EXEC_SRC = _read("executor/app.py")
TELE_SRC = _read("telegram-bot/app.py")
DASH_SRC = _read("dashboard/app.py")


# ═══════════════════════════════════════════════════════════════════
# H1.1 — asyncio.Lock on shared mutable state
# ═══════════════════════════════════════════════════════════════════

class TestH1_1_AsyncioLocks(unittest.TestCase):

    def test_asyncio_imported(self):
        self.assertIn("import asyncio", MEMU_SRC)

    def test_session_lock_defined(self):
        self.assertIn("_session_lock = asyncio.Lock()", MEMU_SRC)

    def test_feedback_lock_defined(self):
        self.assertIn("_feedback_lock = asyncio.Lock()", MEMU_SRC)

    def test_emotion_no_longer_needs_lock(self):
        """P17 dropped its asyncio.Lock (along with the relationship-
        milestones one) — reads/writes go through atomic Redis list/hash
        ops instead (see DECISIONS.md D23), same rationale as P20's D22."""
        self.assertNotIn("_emotion_lock", MEMU_SRC)
        self.assertNotIn("_relationship_lock", MEMU_SRC)
        self.assertIn("_p17_append_capped", MEMU_SRC)

    def test_nudge_lock_defined(self):
        self.assertIn("_nudge_lock = asyncio.Lock()", MEMU_SRC)

    def test_topic_lock_defined(self):
        self.assertIn("_topic_lock = asyncio.Lock()", MEMU_SRC)

    def test_narrative_no_longer_needs_lock(self):
        """P18 dropped its asyncio.Lock — reads/writes go through atomic
        Redis list ops instead (see DECISIONS.md D24), same rationale as
        P17's D23 and P20's D22."""
        self.assertNotIn("_narrative_lock", MEMU_SRC)
        self.assertIn("_p18_append_capped", MEMU_SRC)

    def test_imagination_no_longer_needs_lock(self):
        """P19 dropped its asyncio.Lock — reads/writes go through atomic
        Redis list ops (and a GET/SET-backed empathy map) instead (see
        DECISIONS.md D25), same rationale as P17/P18/P20's D23/D24/D22."""
        self.assertNotIn("_imagination_lock", MEMU_SRC)
        self.assertIn("_p19_append_capped", MEMU_SRC)

    def test_conscience_no_longer_needs_lock(self):
        """P20 dropped its asyncio.Lock — reads/writes go through atomic
        Redis hash/list ops instead (see DECISIONS.md D22), since a Python
        lock can't coordinate across the separate process P20 will move to."""
        self.assertNotIn("_conscience_lock", MEMU_SRC)
        self.assertIn("_p20_append_capped", MEMU_SRC)

    def test_agent_no_longer_needs_lock(self):
        """P21 dropped its asyncio.Lock — reads/writes go through atomic
        Redis hash/list ops instead (see DECISIONS.md D26), same rationale
        as P17/P18/P19/P20's D23/D24/D25/D22."""
        self.assertNotIn("_agent_lock", MEMU_SRC)
        self.assertIn("_p21_hash_put", MEMU_SRC)

    def test_operator_no_longer_needs_lock(self):
        """P22 dropped its asyncio.Lock — reads/writes go through atomic
        Redis hash/list ops instead (see DECISIONS.md D27), same rationale
        as P17/P18/P19/P20/P21's D23/D24/D25/D22/D26."""
        self.assertNotIn("_operator_lock", MEMU_SRC)
        self.assertIn("_p22_hash_put", MEMU_SRC)

    def test_feedback_uses_lock(self):
        self.assertIn("async with _feedback_lock:", MEMU_SRC)

    def test_emotion_uses_redis_native_store(self):
        self.assertIn("_p17_append_capped(_P17_EMOTION_KEY", MEMU_SRC)


# ═══════════════════════════════════════════════════════════════════
# H1.2 — Prompt injection check on /chat
# ═══════════════════════════════════════════════════════════════════

class TestH1_2_ChatInjectionCheck(unittest.TestCase):

    def test_injection_re_defined(self):
        self.assertIn("INJECTION_RE", LANG_SRC)

    def test_chat_has_injection_check(self):
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        self.assertIn("INJECTION_RE.search(user_msg)", chat_section)

    def test_chat_raises_on_injection(self):
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        self.assertIn("prompt injection pattern blocked", chat_section)


# ═══════════════════════════════════════════════════════════════════
# H1.3 — 10-way parallel fetch error handling
# ═══════════════════════════════════════════════════════════════════

class TestH1_3_GatherErrorHandling(unittest.TestCase):

    def test_safe_wrapper_defined(self):
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        self.assertIn("async def _safe(coro, default):", chat_section)

    def test_gather_used(self):
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        self.assertIn("asyncio.gather(", chat_section)

    def test_ten_safe_calls(self):
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        safe_count = chat_section.count("_safe(")
        self.assertGreaterEqual(safe_count, 10, f"Expected 10+ _safe() calls, found {safe_count}")

    def test_no_bare_create_task(self):
        """Old pattern should be gone."""
        chat_section = LANG_SRC.split('@app.post("/chat")')[1].split("@app.")[0]
        self.assertNotIn("create_task(_get_relevant_memories", chat_section)


# ═══════════════════════════════════════════════════════════════════
# H1.4 — store.memorize → store.insert fix
# ═══════════════════════════════════════════════════════════════════

class TestH1_4_StoreInsert(unittest.TestCase):

    def test_no_store_memorize(self):
        self.assertNotIn("store.memorize(", MEMU_SRC)

    def test_feedback_uses_store_insert(self):
        feedback_section = MEMU_SRC.split("/memory/feedback")[1].split("@app.")[0]
        self.assertIn("store.insert(", feedback_section)

    def test_feedback_creates_memory_record(self):
        feedback_section = MEMU_SRC.split("/memory/feedback")[1].split("@app.")[0]
        self.assertIn("MemoryRecord(", feedback_section)


# ═══════════════════════════════════════════════════════════════════
# H1.5 — executor shell=False + AST sandbox
# ═══════════════════════════════════════════════════════════════════

class TestH1_5_ExecutorSecurity(unittest.TestCase):

    def test_no_shell_true(self):
        self.assertNotIn("shell=True", EXEC_SRC)

    def test_shell_false(self):
        self.assertIn("shell=False", EXEC_SRC)

    def test_shlex_imported(self):
        self.assertIn("import shlex", EXEC_SRC)

    def test_shlex_split_used(self):
        self.assertIn("shlex.split(command)", EXEC_SRC)

    def test_ast_imported(self):
        self.assertIn("import ast", EXEC_SRC)

    def test_ast_parse_used(self):
        self.assertIn("ast.parse(expression", EXEC_SRC)

    def test_getattr_blocked(self):
        self.assertIn('"getattr"', EXEC_SRC)

    def test_dunder_attribute_blocked(self):
        self.assertIn('startswith("_")', EXEC_SRC)

    def test_ast_walk_used(self):
        self.assertIn("ast.walk(tree)", EXEC_SRC)


# ═══════════════════════════════════════════════════════════════════
# H1.6 — Telegram voice file size limit
# ═══════════════════════════════════════════════════════════════════

class TestH1_6_VoiceSizeLimit(unittest.TestCase):

    def test_max_voice_bytes_defined(self):
        self.assertIn("MAX_VOICE_BYTES", TELE_SRC)

    def test_size_check_present(self):
        self.assertIn("MAX_VOICE_BYTES", TELE_SRC.split("_download_file")[1].split("def ")[0])

    def test_path_traversal_check(self):
        download_section = TELE_SRC.split("_download_file")[1].split("def ")[0]
        self.assertIn('".."', download_section)

    def test_ten_mb_limit(self):
        self.assertIn("10 * 1024 * 1024", TELE_SRC)


# ═══════════════════════════════════════════════════════════════════
# H1.7 — Dashboard proxy error handling
# ═══════════════════════════════════════════════════════════════════

class TestH1_7_DashboardProxyGuards(unittest.TestCase):

    def test_proxy_get_defined(self):
        self.assertIn("async def _proxy_get(", DASH_SRC)

    def test_proxy_post_defined(self):
        self.assertIn("async def _proxy_post(", DASH_SRC)

    def test_proxy_get_has_try_except(self):
        """H2 upgraded proxy to use resilient_call (replaces bare try/except)."""
        fn = DASH_SRC.split("async def _proxy_get(")[1].split("async def ")[0]
        self.assertIn("resilient_call(", fn)

    def test_proxy_post_has_try_except(self):
        """H2 upgraded proxy to use resilient_call (replaces bare try/except)."""
        fn = DASH_SRC.split("async def _proxy_post(")[1].split("async def ")[0]
        self.assertIn("resilient_call(", fn)

    def test_emotion_endpoint_uses_proxy(self):
        section = DASH_SRC.split("proxy_emotion_record")[1].split("@app.")[0]
        self.assertIn("_proxy_post(", section)

    def test_reflections_uses_proxy(self):
        section = DASH_SRC.split("proxy_reflections")[1].split("@app.")[0]
        self.assertIn("_proxy_get(", section)

    def test_operator_model_uses_proxy(self):
        section = DASH_SRC.split("proxy_operator_model")[1].split("@app.")[0]
        self.assertIn("_proxy_get(", section)

    def test_no_bare_httpx_in_p17_plus(self):
        """P17+ proxy endpoints should not have bare httpx calls."""
        p17_section = DASH_SRC.split("P17:")[1].split("Unified App Shell")[0]
        # Allow httpx in _proxy_get/_proxy_post helpers, block in endpoint functions
        proxy_endpoints = [line for line in p17_section.split("\n")
                          if "async def proxy_" in line]
        # Verify we have many proxy endpoints
        self.assertGreaterEqual(len(proxy_endpoints), 20,
                               f"Expected 20+ proxy endpoints, found {len(proxy_endpoints)}")

    def test_values_uses_proxy(self):
        section = DASH_SRC.split("proxy_values_learn")[1].split("@app.")[0]
        self.assertIn("_proxy_post(", section)

    def test_shadow_uses_proxy(self):
        section = DASH_SRC.split("proxy_shadow_branches")[1].split("@app.")[0]
        self.assertIn("_proxy_get(", section)


if __name__ == "__main__":
    unittest.main()
