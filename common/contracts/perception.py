"""Perception contracts — events entering the system from external sources."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from common.contracts.base import ContractBase, RiskTier


class EventSource(str, Enum):
    TELEGRAM = "telegram"
    CAMERA = "camera"
    AUDIO = "audio"
    MARKET = "market"
    WEATHER = "weather"
    CALENDAR = "calendar"
    SCREEN = "screen"
    CLIPBOARD = "clipboard"
    EMAIL = "email"
    NEWS = "news"
    DOCKER = "docker"
    GIT = "git"
    SYSTEM = "system"
    MANUAL = "manual"


class PerceptionEvent(ContractBase):
    """An event entering the system from a perception source.

    This is the canonical ingress contract. All sensor data, user input,
    and external signals must be wrapped in a PerceptionEvent before
    entering the processing pipeline.
    """

    event_type: str
    source_type: EventSource
    payload: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    risk_tier: RiskTier = RiskTier.OBSERVE
    tags: List[str] = Field(default_factory=list)
    raw_hash: Optional[str] = None
    source_timestamp: Optional[datetime] = None
    duplicate_of: Optional[str] = None
    stale: bool = False
