from __future__ import annotations

import json

from common.market_cache import refresh_cache

if __name__ == "__main__":
    print(json.dumps(refresh_cache(), indent=2))
