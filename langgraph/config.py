from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Protocol

import redis


class EpisodeSaver(Protocol):
    def save_episode(self, payload: Dict[str, Any]) -> None: ...

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]: ...

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int: ...


class RedisSaver:
    def __init__(self, redis_url: str) -> None:
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


class InMemorySaver:
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Dict[str, Any]]] = {}
        self._archive: Dict[str, List[Dict[str, Any]]] = {}

    def save_episode(self, payload: Dict[str, Any]) -> None:
        user_id = str(payload.get("user_id", "keeper"))
        bucket = self._episodes.setdefault(user_id, [])
        bucket.insert(0, dict(payload))

    def recall(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        min_ts = time.time() - (days * 24 * 3600)
        return [e for e in self._episodes.get(user_id, []) if float(e.get("ts", 0)) >= min_ts][:200]

    def decay(self, user_id: str, days: int = 30, score_threshold: float = 0.2) -> int:
        max_age = time.time() - (days * 24 * 3600)
        moved = 0
        kept: List[Dict[str, Any]] = []
        archive = self._archive.setdefault(user_id, [])
        for episode in self._episodes.get(user_id, []):
            too_old = float(episode.get("ts", 0)) < max_age
            low_score = float(episode.get("outcome_score", 0)) < score_threshold
            if too_old and low_score:
                archive.append(episode)
                moved += 1
            else:
                kept.append(episode)
        self._episodes[user_id] = kept
        return moved


class ChecksummedSpoolSaver(InMemorySaver):
    def __init__(self, spool_path: str) -> None:
        super().__init__()
        self.spool_path = Path(spool_path)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_from_spool()

    def _line_for(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        checksum = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return json.dumps({"checksum": checksum, "payload": payload}, sort_keys=True)

    def _load_from_spool(self) -> None:
        if not self.spool_path.exists():
            return
        for line in self.spool_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                payload = obj["payload"]
                checksum = str(obj.get("checksum", ""))
                raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
                if checksum != expected:
                    continue
                super().save_episode(payload)
            except Exception:
                continue

    def save_episode(self, payload: Dict[str, Any]) -> None:
        line = self._line_for(payload)
        with self.spool_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        super().save_episode(payload)


def build_saver() -> EpisodeSaver:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    prefer_memory = os.getenv("EPISODE_STORE", "redis").lower() == "memory"
    spool_path = os.getenv("EPISODE_SPOOL_PATH", "/tmp/langgraph_episode_spool.log")
    if prefer_memory:
        return ChecksummedSpoolSaver(spool_path=spool_path)

    try:
        saver = RedisSaver(redis_url=redis_url)
        saver.redis.ping()
        return saver
    except Exception:
        return ChecksummedSpoolSaver(spool_path=spool_path)
