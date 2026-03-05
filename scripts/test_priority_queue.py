"""Tests for HP5 Priority Queue."""
import asyncio
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from priority_queue import Priority, PriorityQueue, QueueEntry, QueueStats, get_queue


class TestPriority(unittest.TestCase):
    def test_ordering(self):
        self.assertLess(Priority.CHAT, Priority.RUN)
        self.assertLess(Priority.RUN, Priority.BACKGROUND)
        self.assertLess(Priority.BACKGROUND, Priority.BATCH)

    def test_values(self):
        self.assertEqual(Priority.CHAT, 0)
        self.assertEqual(Priority.BATCH, 3)


class TestQueueEntry(unittest.TestCase):
    def test_comparison_by_priority(self):
        e1 = QueueEntry(priority=Priority.CHAT, submitted_at=1.0, task_id="a")
        e2 = QueueEntry(priority=Priority.RUN, submitted_at=1.0, task_id="b")
        self.assertTrue(e1 < e2)

    def test_comparison_by_time_on_tie(self):
        e1 = QueueEntry(priority=Priority.CHAT, submitted_at=1.0, task_id="a")
        e2 = QueueEntry(priority=Priority.CHAT, submitted_at=2.0, task_id="b")
        self.assertTrue(e1 < e2)


class TestPriorityQueue(unittest.TestCase):
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_submit_and_get_result(self):
        q = PriorityQueue(max_concurrent=2)
        async def work():
            return 42
        result = self._run(q.submit(Priority.CHAT, work))
        self.assertEqual(result, 42)

    def test_stats_after_processing(self):
        q = PriorityQueue(max_concurrent=2)
        async def work():
            return "done"
        self._run(q.submit(Priority.CHAT, work))
        s = q.stats()
        self.assertEqual(s.total_processed, 1)
        self.assertEqual(s.active, 0)

    def test_concurrent_limit(self):
        q = PriorityQueue(max_concurrent=1)
        execution_order = []

        async def task(name, delay):
            execution_order.append(f"{name}-start")
            await asyncio.sleep(delay)
            execution_order.append(f"{name}-end")
            return name

        async def run_both():
            t1 = asyncio.create_task(q.submit(Priority.CHAT, task, "a", 0.01))
            t2 = asyncio.create_task(q.submit(Priority.CHAT, task, "b", 0.01))
            await asyncio.gather(t1, t2)

        self._run(run_both())
        # With max_concurrent=1, tasks should not overlap
        self.assertEqual(q.stats().total_processed, 2)

    def test_auto_task_id(self):
        q = PriorityQueue()
        async def work():
            return True
        self._run(q.submit(Priority.RUN, work))
        self.assertEqual(q.stats().total_processed, 1)

    def test_exception_propagation(self):
        q = PriorityQueue()
        async def failing():
            raise ValueError("test error")
        with self.assertRaises(ValueError):
            self._run(q.submit(Priority.CHAT, failing))
        # Should still count as processed
        self.assertEqual(q.stats().total_processed, 1)

    def test_multiple_priorities(self):
        q = PriorityQueue(max_concurrent=4)
        results = []
        async def work(label):
            results.append(label)
            return label

        async def run_all():
            tasks = [
                q.submit(Priority.BATCH, work, "batch"),
                q.submit(Priority.CHAT, work, "chat"),
                q.submit(Priority.RUN, work, "run"),
            ]
            await asyncio.gather(*tasks)

        self._run(run_all())
        self.assertEqual(len(results), 3)
        self.assertIn("chat", results)
        self.assertIn("run", results)
        self.assertIn("batch", results)


class TestGetQueue(unittest.TestCase):
    def test_singleton(self):
        import priority_queue
        priority_queue._default_queue = None  # reset
        q1 = get_queue(max_concurrent=2)
        q2 = get_queue()
        self.assertIs(q1, q2)


class TestQueueStats(unittest.TestCase):
    def test_stats_structure(self):
        s = QueueStats(pending=0, active=1, total_processed=5, avg_wait_ms=1.5, by_priority={})
        self.assertEqual(s.pending, 0)
        self.assertEqual(s.active, 1)
        self.assertEqual(s.total_processed, 5)


if __name__ == "__main__":
    unittest.main()
