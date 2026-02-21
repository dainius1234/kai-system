from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Protocol


class Saver(Protocol):
    def save_episode(self, payload: Dict[str, Any]) -> None: ...
    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]: ...
    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int: ...


class InMemorySaver:
    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        self._archive: Dict[str, List[Dict[str, Any]]] = {}

    def save_episode(self, payload: Dict[str, Any]) -> None:
        key = payload["user_id"]
        self._store.setdefault(key, []).insert(0, dict(payload))

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        min_ts = time.time() - (days * 24 * 3600)
        return [e for e in self._store.get(user_id, []) if float(e.get("ts", 0)) >= min_ts][:200]

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int:
        max_age = time.time() - (days * 24 * 3600)
        moved = 0
        kept: List[Dict[str, Any]] = []
        for episode in self._store.get(user_id, []):
            too_old = float(episode.get("ts", 0)) < max_age
            low_score = float(episode.get("outcome_score", 0)) < score_threshold
            if too_old and low_score:
                self._archive.setdefault(user_id, []).insert(0, dict(episode))
                moved += 1
            else:
                kept.append(episode)
        self._store[user_id] = kept
        return moved


class RedisSaver:
    def __init__(self, redis_url: str) -> None:
        import redis

        self.redis = redis.from_url(redis_url, decode_responses=True)

    def save_episode(self, payload: Dict[str, Any]) -> None:
        key = f"episodes:{payload['user_id']}"
        self.redis.lpush(key, json.dumps(payload))

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        key = f"episodes:{user_id}"
        min_ts = time.time() - (days * 24 * 3600)
        items = self.redis.lrange(key, 0, 200)
        out: List[Dict[str, Any]] = []
        for raw in items:
            episode = json.loads(raw)
            if float(episode.get("ts", 0)) >= min_ts:
                out.append(episode)
        return out

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int:
        key = f"episodes:{user_id}"
        archive = f"episodes:archive:{user_id}"
        max_age = time.time() - (days * 24 * 3600)
        moved = 0
        kept: List[str] = []
        for raw in self.redis.lrange(key, 0, 1000):
            episode = json.loads(raw)
            too_old = float(episode.get("ts", 0)) < max_age
            low_score = float(episode.get("outcome_score", 0)) < score_threshold
            if too_old and low_score:
                self.redis.lpush(archive, raw)
                moved += 1
            else:
                kept.append(raw)
        pipe = self.redis.pipeline()
        pipe.delete(key)
        if kept:
            pipe.rpush(key, *reversed(kept))
        pipe.execute()
        return moved


def build_saver() -> Saver:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    try:
        saver = RedisSaver(redis_url=redis_url)
        saver.redis.ping()
        return saver
    except Exception:
        return InMemorySaver()
