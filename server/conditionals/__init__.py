"""Conditional orders tied to manually-drawn trendlines.

A user draws a line on the chart. The system:
  1. Stores the line (existing drawings infra)
  2. Computes pattern stats against the historical DB
  3. User configures a conditional order (offset, size, stop, TP, expiry)
  4. Watcher polls live price; when the line is touched, triggers:
     - Telegram alert (always)
     - Optional real exchange submit
  5. User sees all this in a visible "pending orders" panel

The conditional is a VIRTUAL order — it does not sit on the exchange until
the trigger fires. This avoids cancel/replace storms from the line's slope
changing the trigger price every bar.
"""
from .types import (
    ConditionalOrder,
    ConditionalEvent,
    ConditionalStatus,
    TriggerConfig,
    OrderConfig,
)
from .store import ConditionalOrderStore, now_ts

__all__ = [
    "ConditionalOrder",
    "ConditionalEvent",
    "ConditionalStatus",
    "TriggerConfig",
    "OrderConfig",
    "ConditionalOrderStore",
    "now_ts",
]
