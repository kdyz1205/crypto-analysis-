"""Test the reconcile-aware replan path that fixes the 2026-04-24 ZEC bug.

Bug recap:
    User moves a manual line. The line has an attached conditional whose
    LOCAL status is 'filled' but whose Bitget plan-order is still LIVE.
    Old `force_replan_line` only scans status='triggered' — misses the
    drifted cond — line move silently fails to cancel+replace the Bitget
    order.

The fix promotes filled/cancelled conds back to 'triggered' when Bitget
confirms the order is still live, then flags them for the watcher's
existing replan path.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.mark.asyncio
async def test_reconcile_promotes_drifted_filled_cond_to_triggered():
    """Cond locally 'filled' but Bitget still has it live → must promote
    back to triggered + flag for replan."""
    from server.conditionals import watcher

    line_id = "manual-TESTUSDT-1h-resistance-100-200"
    cond_id = "cond_test_drifted"

    # Build a fake "filled" cond with a bogus exchange_order_id.
    fake_cond = MagicMock()
    fake_cond.conditional_id = cond_id
    fake_cond.manual_line_id = line_id
    fake_cond.status = "filled"
    fake_cond.exchange_order_id = "1431407611887972352"
    fake_cond.order = MagicMock()
    fake_cond.order.exchange_mode = "live"

    fake_store = MagicMock()
    fake_store.list_all = MagicMock(return_value=[fake_cond])
    fake_store.set_status_if = MagicMock(return_value="winner_id")
    fake_store.append_event = MagicMock()

    # Bitget says the order IS still pending.
    fake_adapter = MagicMock()
    fake_adapter.has_api_keys.return_value = True
    fake_adapter._bitget_request = AsyncMock(return_value={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "1431407611887972352", "symbol": "TESTUSDT"}]},
    })

    with patch.object(watcher, "_store", fake_store), \
         patch("server.execution.live_adapter.LiveExecutionAdapter", return_value=fake_adapter), \
         patch.object(watcher, "_force_replan_set", set()) as fake_set:
        result = await watcher.force_replan_line_with_reconcile(line_id)

    assert result["recovered"] == 1, f"expected 1 recovery, got: {result}"
    assert result["flagged"] == 1
    assert result["skipped_terminated"] == 0
    # Cond was promoted back to triggered
    fake_store.set_status_if.assert_called_once()
    args = fake_store.set_status_if.call_args
    assert args.kwargs["from_status"] == "filled"
    assert args.kwargs["to_status"] == "triggered"
    # Audit event was appended
    fake_store.append_event.assert_called_once()


@pytest.mark.asyncio
async def test_reconcile_skips_truly_terminated_cond():
    """Cond locally 'filled' AND Bitget no longer has the order → leave
    alone, don't promote to triggered. Prevents double-placing on a
    genuinely-completed trade."""
    from server.conditionals import watcher

    line_id = "manual-TESTUSDT-1h-resistance-100-200"

    fake_cond = MagicMock()
    fake_cond.conditional_id = "cond_done"
    fake_cond.manual_line_id = line_id
    fake_cond.status = "filled"
    fake_cond.exchange_order_id = "999999"
    fake_cond.order = MagicMock()
    fake_cond.order.exchange_mode = "live"

    fake_store = MagicMock()
    fake_store.list_all = MagicMock(return_value=[fake_cond])

    fake_adapter = MagicMock()
    fake_adapter.has_api_keys.return_value = True
    fake_adapter._bitget_request = AsyncMock(return_value={
        "code": "00000",
        "data": {"entrustedList": []},  # Bitget confirms: no live order
    })

    with patch.object(watcher, "_store", fake_store), \
         patch("server.execution.live_adapter.LiveExecutionAdapter", return_value=fake_adapter), \
         patch.object(watcher, "_force_replan_set", set()):
        result = await watcher.force_replan_line_with_reconcile(line_id)

    assert result["recovered"] == 0
    assert result["flagged"] == 0
    assert result["skipped_terminated"] == 1
    # Status should NOT have been mutated
    fake_store.set_status_if.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_still_flags_already_triggered_cond():
    """Pre-existing behavior preserved: locally-triggered conds get
    flagged for replan WITHOUT needing Bitget verification."""
    from server.conditionals import watcher

    line_id = "manual-TESTUSDT-1h-resistance-100-200"

    fake_cond = MagicMock()
    fake_cond.conditional_id = "cond_already_triggered"
    fake_cond.manual_line_id = line_id
    fake_cond.status = "triggered"
    fake_cond.exchange_order_id = "12345"
    fake_cond.order = MagicMock()
    fake_cond.order.exchange_mode = "live"

    fake_store = MagicMock()
    fake_store.list_all = MagicMock(return_value=[fake_cond])

    fake_adapter = MagicMock()
    fake_adapter.has_api_keys.return_value = True
    fake_adapter._bitget_request = AsyncMock(return_value={
        "code": "00000", "data": {"entrustedList": []},
    })

    with patch.object(watcher, "_store", fake_store), \
         patch("server.execution.live_adapter.LiveExecutionAdapter", return_value=fake_adapter), \
         patch.object(watcher, "_force_replan_set", set()) as fake_set:
        result = await watcher.force_replan_line_with_reconcile(line_id)

    assert result["flagged"] == 1
    assert result["recovered"] == 0
    assert "cond_already_triggered" in fake_set


@pytest.mark.asyncio
async def test_reconcile_handles_bitget_api_failure_gracefully():
    """If Bitget query fails, fall back to original behavior (only flag
    locally-triggered conds). Never blow up the line move."""
    from server.conditionals import watcher

    line_id = "manual-TESTUSDT-1h-resistance-100-200"

    triggered_cond = MagicMock()
    triggered_cond.conditional_id = "cond_a"
    triggered_cond.status = "triggered"
    triggered_cond.exchange_order_id = "111"
    triggered_cond.order = MagicMock()
    triggered_cond.order.exchange_mode = "live"

    filled_cond = MagicMock()
    filled_cond.conditional_id = "cond_b"
    filled_cond.status = "filled"
    filled_cond.exchange_order_id = "222"
    filled_cond.order = MagicMock()
    filled_cond.order.exchange_mode = "live"

    fake_store = MagicMock()
    fake_store.list_all = MagicMock(return_value=[triggered_cond, filled_cond])

    fake_adapter = MagicMock()
    fake_adapter.has_api_keys.return_value = True
    fake_adapter._bitget_request = AsyncMock(side_effect=Exception("Bitget down"))

    with patch.object(watcher, "_store", fake_store), \
         patch("server.execution.live_adapter.LiveExecutionAdapter", return_value=fake_adapter), \
         patch.object(watcher, "_force_replan_set", set()) as fake_set:
        result = await watcher.force_replan_line_with_reconcile(line_id)

    # Triggered cond still flagged (original behavior preserved)
    assert "cond_a" in fake_set
    assert result["flagged"] == 1
    # Filled cond NOT promoted (we couldn't verify with Bitget)
    assert "cond_b" not in fake_set
    assert result["recovered"] == 0
    # Errors logged
    assert any("bitget" in e.lower() or "down" in e.lower() for e in result["errors"])
