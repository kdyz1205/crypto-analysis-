"""Tests for SqliteConditionalOrderStore — same operations as the JSON
store but persisted via SQLite. Each test uses a tmp_path-isolated db
so they don't share state.

Coverage:
  - create + get
  - list_all (no filter, status/symbol/manual_line_id filters, combinations)
  - update
  - append_event (with 200-event cap)
  - set_status (status side effects: triggered_at, cancelled_at, cancel_reason)
  - set_status_if (CAS guard preserved)
  - delete
  - migration from JSON file (one-time, idempotent on second run)
  - concurrency: parallel set_status_if from threads doesn't double-fire
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.conditionals.sqlite_store import SqliteConditionalOrderStore
from server.conditionals.types import (
    ConditionalOrder, ConditionalEvent, OrderConfig, TriggerConfig,
)


def _make_cond(cid: str, *, status: str = "pending", symbol: str = "TESTUSDT",
               manual_line_id: str = "manual-TEST-1h-resistance-100-200",
               created_at: int = 1777000000) -> ConditionalOrder:
    return ConditionalOrder(
        conditional_id=cid,
        manual_line_id=manual_line_id,
        symbol=symbol,
        timeframe="1h",
        side="resistance",
        status=status,
        trigger=TriggerConfig(),
        order=OrderConfig(direction="long", exchange_mode="paper"),
        t_start=100, t_end=200,
        price_start=100.0, price_end=200.0,
        pattern_stats_at_create=None,
        created_at=created_at,
        updated_at=created_at,
        events=[],
    )


def _store(tmp_path):
    return SqliteConditionalOrderStore(
        db_path=tmp_path / "conds.db",
        json_path=tmp_path / "conds.json",  # nonexistent → no migration
    )


# ─── Basic CRUD ────────────────────────────────────────────────────

def test_create_and_get(tmp_path):
    store = _store(tmp_path)
    cond = _make_cond("c1")
    store.create(cond)
    fetched = store.get("c1")
    assert fetched is not None
    assert fetched.conditional_id == "c1"
    assert fetched.symbol == "TESTUSDT"


def test_create_rejects_duplicate(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    with pytest.raises(ValueError):
        store.create(_make_cond("c1"))


def test_create_appends_created_event(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    cond = store.get("c1")
    assert any(e.kind == "created" for e in cond.events)


def test_get_nonexistent_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get("nonexistent") is None


def test_update(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    cond = store.get("c1")
    cond.fill_price = 123.45
    store.update(cond)
    assert store.get("c1").fill_price == 123.45


def test_delete(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    assert store.delete("c1") is True
    assert store.get("c1") is None
    assert store.delete("c1") is False  # second call: nothing to delete


# ─── list_all + filters ────────────────────────────────────────────

def test_list_all_no_filter(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", created_at=1000))
    store.create(_make_cond("c2", created_at=2000))
    store.create(_make_cond("c3", created_at=1500))
    items = store.list_all()
    # Sorted by created_at DESC
    assert [i.conditional_id for i in items] == ["c2", "c3", "c1"]


def test_list_all_status_filter(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", status="pending"))
    store.create(_make_cond("c2", status="triggered"))
    store.create(_make_cond("c3", status="filled"))
    triggered = store.list_all(status="triggered")
    assert [c.conditional_id for c in triggered] == ["c2"]


def test_list_all_symbol_filter(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", symbol="ZECUSDT"))
    store.create(_make_cond("c2", symbol="HYPEUSDT"))
    zec = store.list_all(symbol="ZECUSDT")
    assert [c.conditional_id for c in zec] == ["c1"]


def test_list_all_manual_line_filter(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", manual_line_id="line-A"))
    store.create(_make_cond("c2", manual_line_id="line-B"))
    a = store.list_all(manual_line_id="line-A")
    assert [c.conditional_id for c in a] == ["c1"]


def test_list_all_combined_filters(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", symbol="ZECUSDT", status="triggered"))
    store.create(_make_cond("c2", symbol="HYPEUSDT", status="triggered"))
    store.create(_make_cond("c3", symbol="ZECUSDT", status="filled"))
    zec_triggered = store.list_all(symbol="ZECUSDT", status="triggered")
    assert [c.conditional_id for c in zec_triggered] == ["c1"]


# ─── append_event ──────────────────────────────────────────────────

def test_append_event(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    store.append_event("c1", ConditionalEvent(ts=1, kind="exchange_acked", message="hi"))
    cond = store.get("c1")
    kinds = [e.kind for e in cond.events]
    assert "exchange_acked" in kinds


def test_append_event_caps_at_200(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    for i in range(250):
        store.append_event("c1", ConditionalEvent(ts=i, kind="ping", message=f"#{i}"))
    cond = store.get("c1")
    assert len(cond.events) == 200
    # Last event should be the most recent we appended
    assert cond.events[-1].message == "#249"


def test_append_event_nonexistent_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.append_event("nope", ConditionalEvent(ts=1, kind="x")) is None


# ─── status transitions ───────────────────────────────────────────

def test_set_status_triggers_triggered_at(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    cond = store.set_status("c1", "triggered")
    assert cond.status == "triggered"
    assert cond.triggered_at is not None


def test_set_status_cancelled_records_reason(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1"))
    cond = store.set_status("c1", "cancelled", reason="line broken")
    assert cond.status == "cancelled"
    assert cond.cancel_reason == "line broken"
    assert cond.cancelled_at is not None


def test_set_status_if_succeeds_when_from_matches(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", status="pending"))
    result = store.set_status_if("c1", from_status="pending", to_status="triggered")
    assert result is not None
    assert result.status == "triggered"


def test_set_status_if_fails_when_from_mismatches(tmp_path):
    store = _store(tmp_path)
    store.create(_make_cond("c1", status="pending"))
    # Try to CAS from "triggered" but it's actually "pending"
    result = store.set_status_if("c1", from_status="triggered", to_status="filled")
    assert result is None
    # Status unchanged
    assert store.get("c1").status == "pending"


def test_set_status_if_concurrent_only_one_winner(tmp_path):
    """Two threads try to CAS the same cond from 'triggered' to 'cancelled'.
    Exactly ONE must win — the other must return None. This is the
    safety guarantee that prevents double-spawn-reverse on cond races.
    """
    store = _store(tmp_path)
    store.create(_make_cond("c1", status="triggered"))

    winners = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        result = store.set_status_if(
            "c1", from_status="triggered", to_status="cancelled",
            reason="race test",
        )
        if result is not None:
            winners.append(result.conditional_id)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(winners) == 1, (
        f"CAS race produced {len(winners)} winners — expected exactly 1. "
        f"This breaks the 'no double-fire' safety guarantee."
    )


# ─── Migration from JSON ───────────────────────────────────────────

def test_first_time_migration_from_json(tmp_path):
    """If db is fresh and JSON exists, migrate on init + rename JSON to .bak."""
    json_path = tmp_path / "conds.json"
    db_path = tmp_path / "conds.db"
    # Seed JSON with two conds
    raw = [
        {
            "conditional_id": "c1", "manual_line_id": "L1", "symbol": "ZECUSDT",
            "timeframe": "1h", "side": "resistance", "status": "pending",
            "t_start": 100, "t_end": 200, "price_start": 100.0, "price_end": 200.0,
            "pattern_stats_at_create": {},
            "trigger": {}, "order": {"direction": "long", "exchange_mode": "paper"},
            "created_at": 1000, "updated_at": 1000, "events": [],
        },
        {
            "conditional_id": "c2", "manual_line_id": "L2", "symbol": "HYPEUSDT",
            "timeframe": "4h", "side": "support", "status": "triggered",
            "t_start": 100, "t_end": 200, "price_start": 50.0, "price_end": 60.0,
            "pattern_stats_at_create": {},
            "trigger": {}, "order": {"direction": "long", "exchange_mode": "live"},
            "created_at": 2000, "updated_at": 2000, "events": [],
        },
    ]
    json_path.write_text(json.dumps(raw), encoding="utf-8")

    store = SqliteConditionalOrderStore(db_path=db_path, json_path=json_path)
    items = store.list_all()
    ids = sorted(i.conditional_id for i in items)
    assert ids == ["c1", "c2"]
    # JSON should be backed up
    bak = json_path.with_suffix(json_path.suffix + ".bak")
    assert bak.exists()
    assert not json_path.exists()


def test_second_init_does_not_re_migrate(tmp_path):
    """If db already exists, don't touch JSON or re-migrate."""
    json_path = tmp_path / "conds.json"
    db_path = tmp_path / "conds.db"
    # First init: migrate
    json_path.write_text(json.dumps([{
        "conditional_id": "c1", "manual_line_id": "L", "symbol": "TESTUSDT",
        "timeframe": "1h", "side": "resistance", "status": "pending",
        "t_start": 100, "t_end": 200, "price_start": 100.0, "price_end": 200.0,
        "pattern_stats_at_create": {},
        "trigger": {}, "order": {"direction": "long", "exchange_mode": "paper"},
        "created_at": 1000, "updated_at": 1000, "events": [],
    }]), encoding="utf-8")
    SqliteConditionalOrderStore(db_path=db_path, json_path=json_path)
    bak = json_path.with_suffix(json_path.suffix + ".bak")
    assert bak.exists()

    # Second init: db already exists → no migration. Backup unchanged.
    bak_size_before = bak.stat().st_size
    store2 = SqliteConditionalOrderStore(db_path=db_path, json_path=json_path)
    assert len(store2.list_all()) == 1
    assert bak.stat().st_size == bak_size_before


# ─── Round-trip preservation (serialization fidelity) ────────────

def test_round_trip_preserves_all_fields(tmp_path):
    """Every field on ConditionalOrder must survive create → get."""
    store = _store(tmp_path)
    cond = _make_cond("c1")
    cond.exchange_order_id = "12345"
    cond.fill_price = 100.5
    cond.fill_qty = 0.1
    cond.last_market_price = 102.3
    cond.cancel_reason = "test"
    store.create(cond)
    out = store.get("c1")
    assert out.exchange_order_id == "12345"
    assert out.fill_price == 100.5
    assert out.fill_qty == 0.1
    assert out.last_market_price == 102.3
    assert out.cancel_reason == "test"
