"""Verify the in-memory cache on ConditionalOrderStore (added 2026-04-24)
doesn't go stale — invalidates on:
  1. Our own _write_all calls (cache updated to match)
  2. External file edits (mtime changes)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 2026-04-25: these tests exercise the JSON-file cache (mtime invalidation
# etc.), which is now the LEGACY backend (SQLite is default). Point at the
# legacy class explicitly so the tests still validate the JSON store —
# in case anyone reverts via COND_STORE_BACKEND=json.
from server.conditionals.store import _LegacyJsonConditionalOrderStore as ConditionalOrderStore
from server.conditionals.types import (
    ConditionalOrder, OrderConfig, TriggerConfig,
)


def _make_cond(cid: str, status: str = "pending") -> ConditionalOrder:
    return ConditionalOrder(
        conditional_id=cid,
        manual_line_id="manual-TEST-1h-resistance-100-200",
        symbol="TESTUSDT",
        timeframe="1h",
        side="resistance",
        status=status,
        trigger=TriggerConfig(),
        order=OrderConfig(direction="long", exchange_mode="paper"),
        t_start=100, t_end=200,
        price_start=100.0, price_end=200.0,
        pattern_stats_at_create=None,
        created_at=1777000000,
        updated_at=1777000000,
        events=[],
    )


def test_cache_returns_same_data_on_warm_read(tmp_path):
    p = tmp_path / "conds.json"
    store = ConditionalOrderStore(path=p)
    cond = _make_cond("c1")
    store.create(cond)

    a = store.list_all()
    b = store.list_all()
    assert len(a) == len(b) == 1
    assert a[0].conditional_id == b[0].conditional_id == "c1"


def test_cache_invalidates_on_external_file_edit(tmp_path):
    """If something else modifies the file (e.g. a manual cleanup
    script), the next _read_all must refresh — must not serve the
    stale cache."""
    p = tmp_path / "conds.json"
    store = ConditionalOrderStore(path=p)
    store.create(_make_cond("c1"))
    assert len(store.list_all()) == 1

    # External edit: rewrite file with different content. Bump mtime
    # to ensure the cache invalidates (filesystems may have mtime
    # resolution issues — sleep is the safest cross-platform way).
    time.sleep(0.05)
    raw = json.loads(p.read_text(encoding="utf-8"))
    raw.append({
        **raw[0],
        "conditional_id": "c2",
    })
    p.write_text(json.dumps(raw), encoding="utf-8")
    # Force mtime advance — Windows can otherwise share the same mtime
    # if writes happen too fast.
    new_mtime = time.time()
    os.utime(p, (new_mtime, new_mtime))

    items = store.list_all()
    ids = {i.conditional_id for i in items}
    assert ids == {"c1", "c2"}, f"cache served stale data — got {ids}"


def test_cache_updates_after_our_own_write(tmp_path):
    """Our own _write_all must refresh the cache so subsequent
    list_all returns the new state without re-parsing the file."""
    p = tmp_path / "conds.json"
    store = ConditionalOrderStore(path=p)
    store.create(_make_cond("c1"))
    store.create(_make_cond("c2"))
    items_a = store.list_all()
    assert len(items_a) == 2

    # Mutation: status change via set_status_if
    winner = store.set_status_if(
        "c1", from_status="pending", to_status="cancelled",
        reason="test cache invalidation",
    )
    assert winner is not None

    items_b = store.list_all()
    by_id = {i.conditional_id: i.status for i in items_b}
    assert by_id["c1"] == "cancelled"
    assert by_id["c2"] == "pending"


def test_cache_returns_shallow_copy_so_caller_mutations_dont_corrupt(tmp_path):
    p = tmp_path / "conds.json"
    store = ConditionalOrderStore(path=p)
    store.create(_make_cond("c1"))
    items_a = store.list_all()
    items_a.clear()   # caller mutates the returned list

    items_b = store.list_all()
    assert len(items_b) == 1, "caller mutation leaked into cache"
    assert items_b[0].conditional_id == "c1"


def test_cache_handles_missing_file(tmp_path):
    """If file doesn't exist, list_all returns empty + caches that."""
    p = tmp_path / "missing.json"
    store = ConditionalOrderStore(path=p)
    assert store.list_all() == []
    assert store.list_all() == []   # second call uses cache, not new probe
