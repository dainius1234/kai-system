from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict

CACHE_PATH = Path(os.getenv("MARKET_CACHE_PATH", "data/self-emp/Accounting/market_price_cache.json"))
PETROL_URL = os.getenv("PETROL_API_URL", "https://example.invalid/petrol")
GROCERY_URL = os.getenv("GROCERY_API_URL", "https://example.invalid/grocery")


def _fetch_json(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def _default_cache() -> Dict[str, Any]:
    return {
        "updated_at": int(time.time()),
        "petrol": {"price_gbp_per_l": 1.50, "trend": "+0.05 tomorrow"},
        "grocery": {"basket_change_pct": 0.02, "trend": "stable"},
        "note": "offline fallback cache",
    }


def refresh_cache() -> Dict[str, Any]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_cache()
    try:
        payload["petrol"] = _fetch_json(PETROL_URL)
    except Exception:
        pass
    try:
        payload["grocery"] = _fetch_json(GROCERY_URL)
    except Exception:
        pass
    payload["updated_at"] = int(time.time())
    CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return refresh_cache()
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return refresh_cache()
