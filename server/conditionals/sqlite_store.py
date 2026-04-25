"""SQLite-backed ConditionalOrderStore.

2026-04-25: replaces the JSON-file primary store. Same public API as
ConditionalOrderStore (server/conditionals/store.py) so it's a drop-in
backend swap — all five callers (watcher, routers/conditionals,
routers/drawings ×2, routers/live_execution) work unchanged.

WHY:
    The JSON store rewrote the entire 614 KB file on every mutation
    (set_status_if, append_event, etc.). With 192 conds and 5-10
    writes/min during active trading, that's 50-100 ms of disk I/O
    per write × ~600 writes/hour = a meaningful chunk of CPU + disk.
    SQLite indexed UPDATEs cost ~1 ms each. Reads also drop from
    13 ms parse to indexed lookup.

SCHEMA:
    table conditionals(
        conditional_id  TEXT PRIMARY KEY,
        manual_line_id  TEXT NOT NULL,
        symbol          TEXT NOT NULL,
        status          TEXT NOT NULL,
        created_at      INTEGER NOT NULL,
        updated_at      INTEGER NOT NULL,
        data            TEXT NOT NULL    -- JSON of full ConditionalOrder
    )

The data column stores the full serialized ConditionalOrder so we
preserve every field without normalizing the schema. Indexed columns
(status, symbol, manual_line_id) accelerate the common filter queries.

DURABILITY:
    WAL journal_mode for concurrent reads + crash safety. Each mutation
    is wrapped in a transaction. SQLite's atomic-replace semantics
    match the existing JSON tmp+rename pattern.

BACKUP:
    On first startup, the existing JSON file is migrated to SQLite and
    renamed to *.json.bak so the user can revert if needed.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict
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

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "conditional_orders.db"
DEFAULT_JSON_PATH = PROJECT_ROOT / "data" / "conditional_orders.json"


def now_ts() -> int:
    return int(time.time())


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conditionals (
    conditional_id  TEXT PRIMARY KEY,
    manual_line_id  TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    data            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cond_status         ON conditionals(status);
CREATE INDEX IF NOT EXISTS idx_cond_symbol         ON conditionals(symbol);
CREATE INDEX IF NOT EXISTS idx_cond_manual_line    ON conditionals(manual_line_id);
CREATE INDEX IF NOT EXISTS idx_cond_created_at     ON conditionals(created_at);
"""


class SqliteConditionalOrderStore:
    """SQLite-backed replacement for ConditionalOrderStore.

    Same public API — five methods used externally (list_all, get,
    create, update, append_event, set_status, set_status_if, delete).
    """

    def __init__(self, db_path: Path | None = None,
                 json_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.json_path = json_path or DEFAULT_JSON_PATH
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        first_time = not self.db_path.exists()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # WAL mode = better concurrency for reader+writer mixes;
            # also matches our "atomic transaction" intent.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # acceptable for non-financial-ledger uses
            conn.commit()

        # First-time migration from JSON if the db is fresh and JSON has data.
        if first_time and self.json_path.exists():
            try:
                self._migrate_from_json()
            except Exception as exc:
                # Don't crash startup — the user can still operate, just
                # without the historical conds. Print loudly so they see it.
                print(f"[sqlite_store] migration from JSON FAILED: {exc}", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Connection helper
    # ─────────────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because the watcher runs in a separate
        # asyncio task and our HTTP handlers may run in different threads
        # depending on FastAPI worker config. Safety still maintained
        # by self._lock around all mutations.
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
            timeout=30.0,           # wait up to 30s for write lock
        )
        conn.row_factory = sqlite3.Row
        return conn

    # ─────────────────────────────────────────────────────────────
    # One-time migration from JSON
    # ─────────────────────────────────────────────────────────────
    def _migrate_from_json(self) -> None:
        try:
            raw = json.loads(self.json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[sqlite_store] JSON unreadable, skipping migration: {exc}", flush=True)
            return
        if not isinstance(raw, list):
            return
        n = 0
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            try:
                for item in raw:
                    if not isinstance(item, dict):
                        continue
                    try:
                        cond = self._dict_to_order(item)
                    except (TypeError, KeyError, ValueError) as e:
                        print(f"[sqlite_store] skipping malformed cond during migration: {e}", flush=True)
                        continue
                    conn.execute(
                        """INSERT OR REPLACE INTO conditionals
                           (conditional_id, manual_line_id, symbol, status,
                            created_at, updated_at, data)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            cond.conditional_id,
                            cond.manual_line_id,
                            cond.symbol,
                            cond.status,
                            cond.created_at,
                            cond.updated_at,
                            json.dumps(self._order_to_dict(cond), ensure_ascii=False),
                        ),
                    )
                    n += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        # Backup the JSON so accidental re-init doesn't double-migrate.
        bak = self.json_path.with_suffix(self.json_path.suffix + ".bak")
        try:
            self.json_path.rename(bak)
            print(f"[sqlite_store] migrated {n} conds from {self.json_path.name} → {self.db_path.name}; "
                  f"original moved to {bak.name}", flush=True)
        except OSError as exc:
            print(f"[sqlite_store] migrated {n} conds; could not rename JSON to .bak: {exc}", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Public API (same shape as ConditionalOrderStore)
    # ─────────────────────────────────────────────────────────────
    def list_all(
        self,
        *,
        status: ConditionalStatus | None = None,
        symbol: str | None = None,
        manual_line_id: str | None = None,
    ) -> list[ConditionalOrder]:
        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if manual_line_id:
            clauses.append("manual_line_id = ?")
            params.append(manual_line_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT data FROM conditionals {where} ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get(self, conditional_id: str) -> ConditionalOrder | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM conditionals WHERE conditional_id = ?",
                (conditional_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_order(row)

    def create(self, order: ConditionalOrder) -> ConditionalOrder:
        order.events.append(ConditionalEvent(
            ts=now_ts(), kind="created",
            message=f"Created from manual_line={order.manual_line_id}, "
                    f"side={order.side}, mode={order.order.exchange_mode}, "
                    f"submit_to_exchange={order.order.submit_to_exchange}",
        ))
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM conditionals WHERE conditional_id = ?",
                (order.conditional_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError(f"conditional already exists: {order.conditional_id}")
            conn.execute(
                """INSERT INTO conditionals
                   (conditional_id, manual_line_id, symbol, status,
                    created_at, updated_at, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                self._cond_to_row_tuple(order),
            )
        return order

    def update(self, order: ConditionalOrder) -> ConditionalOrder:
        order.updated_at = now_ts()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """UPDATE conditionals
                   SET manual_line_id = ?, symbol = ?, status = ?,
                       created_at = ?, updated_at = ?, data = ?
                   WHERE conditional_id = ?""",
                (
                    order.manual_line_id, order.symbol, order.status,
                    order.created_at, order.updated_at,
                    json.dumps(self._order_to_dict(order), ensure_ascii=False),
                    order.conditional_id,
                ),
            )
            if cur.rowcount == 0:
                raise ValueError(f"conditional not found: {order.conditional_id}")
        return order

    def append_event(
        self,
        conditional_id: str,
        event: ConditionalEvent,
    ) -> ConditionalOrder | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            try:
                row = conn.execute(
                    "SELECT data FROM conditionals WHERE conditional_id = ?",
                    (conditional_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                cond = self._row_to_order(row)
                cond.events.append(event)
                cond.updated_at = now_ts()
                # Cap events list at 200 to avoid unbounded growth
                if len(cond.events) > 200:
                    cond.events = cond.events[-200:]
                conn.execute(
                    "UPDATE conditionals SET updated_at = ?, data = ? WHERE conditional_id = ?",
                    (
                        cond.updated_at,
                        json.dumps(self._order_to_dict(cond), ensure_ascii=False),
                        conditional_id,
                    ),
                )
                conn.execute("COMMIT")
                return cond
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def set_status(
        self,
        conditional_id: str,
        status: ConditionalStatus,
        *,
        reason: str = "",
    ) -> ConditionalOrder | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            try:
                row = conn.execute(
                    "SELECT data FROM conditionals WHERE conditional_id = ?",
                    (conditional_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                cond = self._row_to_order(row)
                cond = self._apply_status_change(cond, status, reason)
                conn.execute(
                    """UPDATE conditionals
                       SET status = ?, updated_at = ?, data = ?
                       WHERE conditional_id = ?""",
                    (
                        cond.status, cond.updated_at,
                        json.dumps(self._order_to_dict(cond), ensure_ascii=False),
                        conditional_id,
                    ),
                )
                conn.execute("COMMIT")
                return cond
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def set_status_if(
        self,
        conditional_id: str,
        *,
        from_status: ConditionalStatus,
        to_status: ConditionalStatus,
        reason: str = "",
    ) -> ConditionalOrder | None:
        """Atomic CAS on status. Returns the updated cond if the
        from_status guard matched; None otherwise. Critical for
        avoiding double-fire side effects (cancel + spawn_reverse)
        when two reconcile paths see the same triggered cond.

        The CAS is enforced at the SQL level via WHERE status = ?
        in the UPDATE. SQLite's row-level locking + our self._lock
        + transaction wrapping guarantee no torn read.
        """
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN")
            try:
                row = conn.execute(
                    "SELECT data FROM conditionals WHERE conditional_id = ? AND status = ?",
                    (conditional_id, from_status),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    return None
                cond = self._row_to_order(row)
                cond = self._apply_status_change(cond, to_status, reason)
                # Conditional UPDATE with status guard — defense in depth
                # vs the SELECT-then-UPDATE race.
                cur = conn.execute(
                    """UPDATE conditionals
                       SET status = ?, updated_at = ?, data = ?
                       WHERE conditional_id = ? AND status = ?""",
                    (
                        cond.status, cond.updated_at,
                        json.dumps(self._order_to_dict(cond), ensure_ascii=False),
                        conditional_id, from_status,
                    ),
                )
                if cur.rowcount == 0:
                    conn.execute("ROLLBACK")
                    return None
                conn.execute("COMMIT")
                return cond
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def delete(self, conditional_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conditionals WHERE conditional_id = ?",
                (conditional_id,),
            )
            return cur.rowcount > 0

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _apply_status_change(
        cond: ConditionalOrder,
        to_status: ConditionalStatus,
        reason: str,
    ) -> ConditionalOrder:
        cond.status = to_status
        cond.updated_at = now_ts()
        if to_status == "triggered":
            cond.triggered_at = now_ts()
        elif to_status == "filled":
            if cond.triggered_at is None:
                cond.triggered_at = now_ts()
        elif to_status == "cancelled":
            cond.cancelled_at = now_ts()
            cond.cancel_reason = reason
        return cond

    def _cond_to_row_tuple(self, cond: ConditionalOrder) -> tuple:
        return (
            cond.conditional_id,
            cond.manual_line_id,
            cond.symbol,
            cond.status,
            cond.created_at,
            cond.updated_at,
            json.dumps(self._order_to_dict(cond), ensure_ascii=False),
        )

    def _row_to_order(self, row: sqlite3.Row | dict) -> ConditionalOrder:
        data_str = row["data"] if isinstance(row, sqlite3.Row) else row.get("data", "{}")
        return self._dict_to_order(json.loads(data_str))

    def _order_to_dict(self, o: ConditionalOrder) -> dict[str, Any]:
        from dataclasses import asdict as _asdict, is_dataclass
        try:
            return _asdict(o)
        except RecursionError:
            print(
                f"[sqlite_store] _order_to_dict RecursionError on "
                f"{getattr(o, 'conditional_id', '?')} — falling back to shallow",
                flush=True,
            )
            def _safe(obj, depth: int = 0) -> Any:
                if depth > 20:
                    return "<truncated_depth>"
                if is_dataclass(obj) and not isinstance(obj, type):
                    out = {}
                    for f in obj.__dataclass_fields__.values():
                        val = getattr(obj, f.name, None)
                        if f.name == "extra":
                            out[f.name] = {}
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


__all__ = [
    "SqliteConditionalOrderStore",
    "DEFAULT_DB_PATH",
    "DEFAULT_JSON_PATH",
    "now_ts",
]
