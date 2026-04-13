"""File-based store for ConditionalOrder, with atomic writes + locking.

One JSON file: data/conditional_orders.json
Format: list of ConditionalOrder dicts.

Locking is intra-process (threading.Lock). For multi-process safety use
orchestrator-style file lock if needed later — for now conditional orders
are only touched by the server process.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.config import PROJECT_ROOT
from .types import (
    ConditionalEvent,
    ConditionalOrder,
    ConditionalStatus,
    OrderConfig,
    TriggerConfig,
)

DEFAULT_PATH = PROJECT_ROOT / "data" / "conditional_orders.json"


def now_ts() -> int:
    return int(time.time())


class ConditionalOrderStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def list_all(
        self,
        *,
        status: ConditionalStatus | None = None,
        symbol: str | None = None,
    ) -> list[ConditionalOrder]:
        with self._lock:
            items = self._read_all()
        if status:
            items = [i for i in items if i.status == status]
        if symbol:
            items = [i for i in items if i.symbol == symbol]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items

    def get(self, conditional_id: str) -> ConditionalOrder | None:
        with self._lock:
            items = self._read_all()
        for it in items:
            if it.conditional_id == conditional_id:
                return it
        return None

    def create(self, order: ConditionalOrder) -> ConditionalOrder:
        with self._lock:
            items = self._read_all()
            # Dedup: same conditional_id → reject
            if any(i.conditional_id == order.conditional_id for i in items):
                raise ValueError(f"conditional already exists: {order.conditional_id}")
            # Append a 'created' event
            order.events.append(ConditionalEvent(
                ts=now_ts(), kind="created",
                message=f"Created from manual_line={order.manual_line_id}, "
                        f"side={order.side}, mode={order.order.exchange_mode}, "
                        f"submit_to_exchange={order.order.submit_to_exchange}",
            ))
            items.append(order)
            self._write_all(items)
        return order

    def update(self, order: ConditionalOrder) -> ConditionalOrder:
        with self._lock:
            items = self._read_all()
            order.updated_at = now_ts()
            replaced = False
            for idx, it in enumerate(items):
                if it.conditional_id == order.conditional_id:
                    items[idx] = order
                    replaced = True
                    break
            if not replaced:
                raise ValueError(f"conditional not found: {order.conditional_id}")
            self._write_all(items)
        return order

    def append_event(
        self,
        conditional_id: str,
        event: ConditionalEvent,
    ) -> ConditionalOrder | None:
        with self._lock:
            items = self._read_all()
            for idx, it in enumerate(items):
                if it.conditional_id == conditional_id:
                    it.events.append(event)
                    it.updated_at = now_ts()
                    # Keep only last 200 events per conditional to avoid unbounded growth
                    if len(it.events) > 200:
                        it.events = it.events[-200:]
                    items[idx] = it
                    self._write_all(items)
                    return it
        return None

    def set_status(
        self,
        conditional_id: str,
        status: ConditionalStatus,
        *,
        reason: str = "",
    ) -> ConditionalOrder | None:
        with self._lock:
            items = self._read_all()
            for idx, it in enumerate(items):
                if it.conditional_id == conditional_id:
                    it.status = status
                    it.updated_at = now_ts()
                    if status == "triggered":
                        it.triggered_at = now_ts()
                    elif status == "cancelled":
                        it.cancelled_at = now_ts()
                        it.cancel_reason = reason
                    items[idx] = it
                    self._write_all(items)
                    return it
        return None

    def delete(self, conditional_id: str) -> bool:
        with self._lock:
            items = self._read_all()
            kept = [i for i in items if i.conditional_id != conditional_id]
            if len(kept) == len(items):
                return False
            self._write_all(kept)
        return True

    # ─────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────
    def _read_all(self) -> list[ConditionalOrder]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(raw, list):
            return []
        out: list[ConditionalOrder] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                out.append(self._dict_to_order(item))
            except (TypeError, KeyError, ValueError) as e:
                # Skip malformed but log
                print(f"[conditional_store] skip malformed: {e}", flush=True)
                continue
        return out

    def _write_all(self, items: list[ConditionalOrder]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [self._order_to_dict(i) for i in items]
        # Atomic write: tmp + replace
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.path)

    def _order_to_dict(self, o: ConditionalOrder) -> dict[str, Any]:
        """Serialize ConditionalOrder + nested dataclasses to JSON-safe dict."""
        from dataclasses import asdict as _asdict
        d = _asdict(o)
        return d

    def _dict_to_order(self, d: dict[str, Any]) -> ConditionalOrder:
        """Inverse of _order_to_dict. Reconstructs nested dataclasses."""
        trigger = TriggerConfig(**d.get("trigger", {}))
        order_cfg = OrderConfig(**d.get("order", {}))
        events = [ConditionalEvent(**e) for e in d.get("events", [])]
        return ConditionalOrder(
            conditional_id=d["conditional_id"],
            manual_line_id=d["manual_line_id"],
            symbol=d["symbol"],
            timeframe=d["timeframe"],
            side=d["side"],
            t_start=d["t_start"],
            t_end=d["t_end"],
            price_start=d["price_start"],
            price_end=d["price_end"],
            pattern_stats_at_create=d.get("pattern_stats_at_create", {}),
            trigger=trigger,
            order=order_cfg,
            status=d["status"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            triggered_at=d.get("triggered_at"),
            cancelled_at=d.get("cancelled_at"),
            cancel_reason=d.get("cancel_reason", ""),
            exchange_order_id=d.get("exchange_order_id"),
            fill_price=d.get("fill_price"),
            fill_qty=d.get("fill_qty"),
            events=events,
            last_poll_ts=d.get("last_poll_ts"),
            last_market_price=d.get("last_market_price"),
            last_line_price=d.get("last_line_price"),
            last_distance_atr=d.get("last_distance_atr"),
        )


__all__ = ["ConditionalOrderStore", "DEFAULT_PATH", "now_ts"]
