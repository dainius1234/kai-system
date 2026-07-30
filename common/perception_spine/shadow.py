"""Shadow perception runner — polls sensors and journals events without side effects.

The shadow runner operates alongside the existing Cortex data path.
It validates that every sensor reading can be wrapped in a PerceptionEvent,
journals accepted events, and logs rejections — but does NOT feed events
into any downstream consumer.  This is the UH-2 shadow mode.

Usage:
    runner = ShadowPerceptionRunner(journal_path="/data/perception/journal.jsonl")
    await runner.run_once()       # single poll cycle
    await runner.run_loop()       # continuous background loop
    runner.report()               # print shadow comparison stats
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from common.contracts.base import Principal, Provenance
from common.contracts.perception import EventSource, PerceptionEvent

from common.perception_spine.adapters import ADAPTER_REGISTRY
from common.perception_spine.ingress import IngressVerdict, PerceptionIngress
from common.perception_spine.journal import EventJournal

logger = logging.getLogger("perception-spine")


SENSOR_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "weather":   ("WEATHER_SERVICE_URL",  "http://weather-service:8039/summary"),
    "calendar":  ("CALENDAR_SERVICE_URL", "http://calendar-service:8043/summary"),
    "docker":    ("DOCKER_WATCHER_URL",   "http://docker-watcher:8041/summary"),
    "git":       ("GIT_WATCHER_URL",      "http://git-watcher:8044/summary"),
    "system":    ("SYSMETRICS_URL",       "http://sysmetrics:8035/snapshot"),
    "screen":    ("SCREEN_WATCHER_URL",   "http://screen-watcher:8036/status"),
    "clipboard": ("CLIPBOARD_SERVICE_URL","http://clipboard-service:8024/latest"),
}


class ShadowPerceptionRunner:
    """Polls sensor services, wraps data in PerceptionEvents, journals them."""

    def __init__(
        self,
        journal_path: str = "/data/perception/journal.jsonl",
        principal_identity: str = "kai",
        principal_role: str = "system",
        refresh_interval: int = 60,
    ) -> None:
        self._principal = Principal(identity=principal_identity, role=principal_role)
        self._journal = EventJournal(journal_path)
        self._ingress = PerceptionIngress(
            journal=self._journal,
            principal=self._principal,
            freshness_seconds=int(os.getenv("PERCEPTION_FRESHNESS_SECONDS", "600")),
        )
        self._interval = refresh_interval
        self._cycle_count = 0
        self._cycle_stats: List[Dict[str, Any]] = []
        self._running = False

        self._endpoints: Dict[str, str] = {}
        for sensor, (env_key, default) in SENSOR_ENDPOINTS.items():
            base_url = os.getenv(env_key, default.rsplit("/", 1)[0])
            path = "/" + default.rsplit("/", 1)[1]
            self._endpoints[sensor] = base_url + path

    async def _fetch(self, url: str, timeout: float = 3.0) -> Optional[Dict]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                if r.status_code == 200:
                    return r.json()
        except Exception as exc:
            logger.debug("sensor fetch failed: %s — %s", url, exc)
        return None

    async def run_once(self) -> Dict[str, Any]:
        """Execute one poll cycle across all sensors.  Returns cycle stats."""
        self._cycle_count += 1
        cycle_start = time.monotonic()
        results: Dict[str, Any] = {
            "cycle": self._cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sensors_polled": 0,
            "events_accepted": 0,
            "events_stale": 0,
            "events_duplicate": 0,
            "events_rejected": 0,
            "sensors_unavailable": 0,
            "details": {},
        }

        fetch_coros = {
            sensor: self._fetch(url) for sensor, url in self._endpoints.items()
        }

        raw_data: Dict[str, Optional[Dict]] = {}
        for sensor, coro in fetch_coros.items():
            raw_data[sensor] = await coro

        for sensor, data in raw_data.items():
            results["sensors_polled"] += 1

            if data is None:
                results["sensors_unavailable"] += 1
                results["details"][sensor] = "unavailable"
                continue

            adapter = ADAPTER_REGISTRY.get(sensor)
            if adapter is None:
                results["details"][sensor] = "no_adapter"
                continue

            event = adapter(data, self._principal)
            if event is None:
                results["details"][sensor] = "adapter_returned_none"
                continue

            ingress_result = self._ingress.submit(event)
            verdict = ingress_result.verdict.value
            results["details"][sensor] = verdict

            if ingress_result.verdict == IngressVerdict.ACCEPTED:
                results["events_accepted"] += 1
            elif ingress_result.verdict == IngressVerdict.ACCEPTED_STALE:
                results["events_stale"] += 1
            elif ingress_result.verdict == IngressVerdict.REJECTED_DUPLICATE:
                results["events_duplicate"] += 1
            else:
                results["events_rejected"] += 1

        results["duration_ms"] = round((time.monotonic() - cycle_start) * 1000, 1)
        self._cycle_stats.append(results)

        logger.info(
            "shadow cycle %d: %d accepted, %d stale, %d dup, %d rejected, "
            "%d unavailable (%.1fms)",
            self._cycle_count,
            results["events_accepted"],
            results["events_stale"],
            results["events_duplicate"],
            results["events_rejected"],
            results["sensors_unavailable"],
            results["duration_ms"],
        )
        return results

    async def run_loop(self) -> None:
        """Run polling loop until stopped."""
        self._running = True
        logger.info(
            "shadow perception runner starting (interval=%ds, sensors=%d)",
            self._interval,
            len(self._endpoints),
        )
        while self._running:
            try:
                await self.run_once()
            except Exception:
                logger.exception("shadow cycle error")
            await asyncio.sleep(self._interval)

    def stop(self) -> None:
        self._running = False

    @property
    def journal(self) -> EventJournal:
        return self._journal

    @property
    def ingress(self) -> PerceptionIngress:
        return self._ingress

    def report(self) -> Dict[str, Any]:
        ingress_stats = self._ingress.stats
        return {
            "total_cycles": self._cycle_count,
            "journal_entries": self._journal.count(),
            "ingress_stats": ingress_stats,
            "last_cycle": self._cycle_stats[-1] if self._cycle_stats else None,
        }
