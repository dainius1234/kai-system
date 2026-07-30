"""UH-6: Paper-trading vertical slice.

Wires UH-1 through UH-5 into a complete pipeline that exercises
perception → world state → proposal → policy → approval →
capability → execution → verification with no direct mutation path.
"""

from common.vertical_slice.paper_trade_slice import (
    PaperTradeSlice,
    SliceResult,
    SliceStage,
)

__all__ = [
    "PaperTradeSlice",
    "SliceResult",
    "SliceStage",
]
