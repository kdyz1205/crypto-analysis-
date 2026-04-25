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


# 2026-04-25: ConditionalOrderStore is now an alias for the SQLite-backed
# implementation. The legacy JSON-file class below is retained as
# `_LegacyJsonConditionalOrderStore` and selected via the env var
# COND_STORE_BACKEND=json (used by tests + as an emergency revert switch).
#
# Migration: SqliteConditionalOrderStore.__init__ auto-imports the JSON
# file on first run, then renames it to .json.bak.
import os as _os_for_backend


def _resolve_backend():
    backend = _os_for_backend.environ.get("COND_STORE_BACKEND", "sqlite").lower()
    if backend == "json":
        return _LegacyJsonConditionalOrderStore
    from .sqlite_store import SqliteConditionalOrderStore
    return SqliteConditionalOrderStore


class _LegacyJsonConditionalOrderStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_PATH
        self._lock = threading.Lock()
        # 2026-04-24: in-memory mirror to avoid the 614 KB JSON read on
        # every list_all / get / mutation. Polling (decision_rail 30s,
        # conditional_panel 10s) was producing 4-12 full deserializes /
        # min when the file already had 96+ conds. Now the cache serves
        # reads in O(1) and is invalidated when the file mtime changes
        # (handles external edits e.g. manual cleanup scripts) or when
        # we mutate via _write_all. Writes still go to disk atomically;
        # crash safety unchanged.
        self._cache: list[ConditionalOrder] | None = None
        self._cache_mtime: float = 0.0

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    def list_all(
        self,
        *,
        status: ConditionalStatus | None = None,
        symbol: str | None = None,
        manual_line_id: str | None = None,
    ) -> list[ConditionalOrder]:
        with self._lock:
            items = self._read_all()
        if status:
            items = [i for i in items if i.status == status]
        if symbol:
            items = [i for i in items if i.symbol == symbol]
        if manual_line_id:
            items = [i for i in items if i.manual_line_id == manual_line_id]
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
                    elif status == "filled":
                        if it.triggered_at is None:
                            it.triggered_at = now_ts()
                    elif status == "cancelled":
                        it.cancelled_at = now_ts()
                        it.cancel_reason = reason
                    items[idx] = it
                    self._write_all(items)
                    return it
        return None

    def set_status_if(
        self,
        conditional_id: str,
        *,
        from_status: ConditionalStatus,
        to_status: ConditionalStatus,
        reason: str = "",
    ) -> ConditionalOrder | None:
        """Atomic compare-and-swap on status. Transitions only if the
        current status matches `from_status`; otherwise returns None
        without mutating.

        CRITICAL USE: code paths that "cancel + spawn_reverse" (and any
        other one-shot side effect) MUST use this, NOT set_status, to
        avoid double-firing when two loops race on the same cond.
        Example: _maybe_replan's line-broken cancel AND reconcile's
        "order gone" path can both see status=triggered in the same
        tick. Without CAS, both would fire set_status + spawn_reverse,
        giving the user 2 reverse positions on 1 invalidation.

        Returns the updated cond (now at to_status), or None if the
        from_status guard failed. Callers MUST check the return value
        before executing any side effect (reverse spawn, event append,
        etc.).
        """
        with self._lock:
            items = self._read_all()
            for idx, it in enumerate(items):
                if it.conditional_id == conditional_id:
                    if it.status != from_status:
                        return None
                    it.status = to_status
                    it.updated_at = now_ts()
                    if to_status == "triggered":
                        it.triggered_at = now_ts()
                    elif to_status == "filled":
                        if it.triggered_at is None:
                            it.triggered_at = now_ts()
                    elif to_status == "cancelled":
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
        # Cache hit when file's mtime hasn't changed since last load.
        # Returns a SHALLOW COPY so callers mutating the list don't
        # corrupt our cache (writes go through _write_all anyway).
        if not self.path.exists():
            self._cache = []
            self._cache_mtime = 0.0
            return []
        try:
            current_mtime = self.path.stat().st_mtime
        except OSError:
            current_mtime = 0.0
        if self._cache is not None and current_mtime == self._cache_mtime:
            return list(self._cache)   # shallow copy

        # Miss: parse from disk + repopulate cache
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._cache = []
            self._cache_mtime = current_mtime
            return []
        if not isinstance(raw, list):
            self._cache = []
            self._cache_mtime = current_mtime
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
        self._cache = list(out)
        self._cache_mtime = current_mtime
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
        # Update in-memory mirror to match what we just persisted.
        # Use the new mtime so next _read_all serves from cache.
        try:
            self._cache_mtime = self.path.stat().st_mtime
        except OSError:
            self._cache_mtime = 0.0
        self._cache = list(items)   # shallow copy

    def _order_to_dict(self, o: ConditionalOrder) -> dict[str, Any]:
        """Serialize ConditionalOrder + nested dataclasses to JSON-safe dict.

        2026-04-24: added defensive fallback. A prior bug report caught
        `dataclasses._asdict_inner` recursing to Python's limit on some
        in-flight cond (pre-persist), likely because something placed a
        cyclic structure into ConditionalEvent.extra (dict[str, Any]).
        When that happens, fall back to a manual serializer that skips
        `extra` entirely so the rest of the cond can still be stored +
        returned to the client.
        """
        from dataclasses import asdict as _asdict, is_dataclass
        try:
            return _asdict(o)
        except RecursionError:
            print(
                f"[store] _order_to_dict RecursionError on {getattr(o, 'conditional_id', '?')} "
                f"— falling back to shallow serialization (events[].extra dropped)",
                flush=True,
            )
            # Manual shallow serializer. Dataclasses → vars-like dict, but
            # for ConditionalEvent we DROP extra (the likely cycle source)
            # and keep only the scalar fields.
            def _safe(obj, depth: int = 0) -> Any:
                if depth > 20:
                    return "<truncated_depth>"
                if is_dataclass(obj) and not isinstance(obj, type):
                    out = {}
                    for f in obj.__dataclass_fields__.values():
                        val = getattr(obj, f.name, None)
                        if f.name == "extra":
                            out[f.name] = {}  # drop potentially-cyclic extra
                        else:
                            out[f.name] = _safe(val, depth + 1)
                    return out
                if isinstance(obj, dict):
                    return {k: _safe(v, depth + 1) for k, v in obj.items()
                            if not isinstance(v, type) and k != "extra"}
                if isinstance(obj, (list, tuple)):
                    return [_safe(v, depth + 1) for v in obj]
                if isinstance(obj, (str, int, float, bool)) or obj is None:
                    return obj
                return str(obj)
            return _safe(o)

    def _dict_to_order(self, d: dict[str, Any]) -> ConditionalOrder:
        """Inverse of _order_to_dict. Reconstructs nested dataclasses.
        Filters keys defensively so schema drift in the JSON file (extra
        fields from older or newer code versions) doesn't silently drop
        the entire conditional via TypeError."""
        from dataclasses import fields as _dc_fields

        def _filter(payload: dict, dc_cls) -> dict:
            valid = {f.name for f in _dc_fields(dc_cls)}
            return {k: v for k, v in (payload or {}).items() if k in valid}

        trigger = TriggerConfig(**_filter(d.get("trigger", {}), TriggerConfig))
        order_cfg = OrderConfig(**_filter(d.get("order", {}), OrderConfig))
        events = [
            ConditionalEvent(**_filter(e, ConditionalEvent))
            for e in d.get("events", [])
        ]
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
            extend_left=bool(d.get("extend_left", False)),
            extend_right=bool(d.get("extend_right", True)),
        )


# 2026-04-25: public symbol resolves to the active backend at import time.
# Default backend is SQLite; set COND_STORE_BACKEND=json to revert.
ConditionalOrderStore = _resolve_backend()


__all__ = [
    "ConditionalOrderStore",
    "_LegacyJsonConditionalOrderStore",  # exposed for tests
    "DEFAULT_PATH",
    "now_ts",
]
