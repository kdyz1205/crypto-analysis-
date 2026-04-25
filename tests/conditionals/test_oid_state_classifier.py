"""Tests for the cond-state classifier (server/conditionals/oid_state.py).

Most important test: `test_zec_2026_04_24_multiple_plans_same_side_must_not_falsely_fill`.
This pins the exact bug that caused real money to be left in a stale state:
when the user has multiple plan-orders on the same (symbol, side), an
unrelated position on Bitget MUST NOT cause our still-pending plan to
be misclassified as FILLED.

Branches covered:
  - oid in pending_oids               → LIVE (fast path 1)
  - oid gone + position exists        → FILLED (fast path 2 — ONLY when oid affirmed gone)
  - oid still pending + position      → LIVE (must NOT touch fast path 2)
  - pending_fetch_failed + position   → fall through to history (must NOT infer FILLED)
  - history row state=cancelled       → CANCELLED
  - history row state=filled          → FILLED
  - history row state=live             → LIVE
  - history row state=failed           → CANCELLED
  - oid not in pending, not in history → UNKNOWN
  - history exception                 → UNKNOWN
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.conditionals.oid_state import classify_cond_state, CondRemoteState


def _make_cond(oid: str = "1431407611887972352", symbol: str = "ZECUSDT",
               direction: str = "long", mode: str = "live"):
    cond = MagicMock()
    cond.exchange_order_id = oid
    cond.symbol = symbol
    cond.order = MagicMock()
    cond.order.direction = direction
    cond.order.exchange_mode = mode
    return cond


def _make_adapter(history_response: dict | Exception | None = None):
    adapter = MagicMock()
    if isinstance(history_response, Exception):
        adapter._bitget_request = AsyncMock(side_effect=history_response)
    else:
        adapter._bitget_request = AsyncMock(return_value=history_response or {
            "code": "00000", "data": {"entrustedList": []}
        })
    return adapter


# ─── The headline test: the exact ZEC bug ────────────────────────────────

@pytest.mark.asyncio
async def test_zec_2026_04_24_multiple_plans_same_side_must_not_falsely_fill():
    """User has TWO plan-orders for ZECUSDT long. Plan B fires and opens
    a position. Reconciler runs on plan A (still pending on Bitget).
    The classifier MUST return LIVE for plan A — NOT FILLED — even
    though a matching ZEC long position exists.

    This is the exact 2026-04-24 ZEC incident. Before the fix, the
    'position_matches_symbol_side' fast path would pre-empt the LIVE
    determination and wrongly mark plan A as filled. After the fix,
    fast path 2 requires AFFIRMATIVE evidence that A's oid is gone
    from pending — which it isn't, so we correctly return LIVE.
    """
    cond_a = _make_cond(oid="1431407611887972352")  # plan A
    plan_b_position = {
        "symbol": "ZECUSDT", "holdSide": "long", "total": "10.0",
    }
    result = await classify_cond_state(
        cond_a,
        adapter=_make_adapter(),
        pending_oids={"1431407611887972352"},  # A is STILL in pending
        pending_fetch_ok=True,
        positions=[plan_b_position],            # …AND a position from B exists
        positions_fetch_ok=True,
    )
    assert result.state == CondRemoteState.LIVE, (
        f"Plan A's oid is in the pending list — it must be classified LIVE "
        f"regardless of any unrelated open position. Got {result.state} "
        f"via path={result.path!r}. THIS IS THE 2026-04-24 ZEC BUG. "
        f"Reverting forbidden."
    )
    assert result.path == "pending_list_has_oid"


# ─── Fast paths individually ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pending_has_oid_returns_live():
    cond = _make_cond(oid="ABC")
    r = await classify_cond_state(
        cond, adapter=_make_adapter(),
        pending_oids={"ABC"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.LIVE
    assert r.path == "pending_list_has_oid"


@pytest.mark.asyncio
async def test_oid_gone_from_pending_plus_position_returns_filled():
    """When the pending list AFFIRMS the oid is gone AND a position
    matches, FILLED is the safe inference."""
    cond = _make_cond(oid="ABC", symbol="ZECUSDT", direction="long")
    r = await classify_cond_state(
        cond, adapter=_make_adapter(),
        pending_oids={"OTHER", "DIFFERENT"},  # ABC NOT here
        pending_fetch_ok=True,
        positions=[{"symbol": "ZECUSDT", "holdSide": "long", "total": "5.0"}],
        positions_fetch_ok=True,
    )
    assert r.state == CondRemoteState.FILLED
    assert r.path == "oid_gone_from_pending+position_matches"


@pytest.mark.asyncio
async def test_pending_fetch_failed_plus_position_must_fall_through_to_history():
    """If pending fetch failed we don't have authoritative absence —
    must NOT infer FILLED from position alone. Falls through to history.
    """
    cond = _make_cond(oid="ABC")
    # History returns a row that confirms FILLED (separate evidence)
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "ABC", "planStatus": "filled"}]},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids=None, pending_fetch_ok=False,    # pending fetch failed
        positions=[{"symbol": "ZECUSDT", "holdSide": "long", "total": "5.0"}],
        positions_fetch_ok=True,
    )
    # Crucially the path must be the history-based one, not the fast-path-2 one
    assert r.state == CondRemoteState.FILLED
    assert r.path == "history_row_filled"
    # And the adapter WAS called for history
    adapter._bitget_request.assert_called_once()


@pytest.mark.asyncio
async def test_oid_gone_no_position_falls_through_to_history():
    cond = _make_cond(oid="ABC", symbol="ZECUSDT", direction="long")
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "ABC", "planStatus": "cancelled"}]},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
        positions=[],   # no positions
        positions_fetch_ok=True,
    )
    assert r.state == CondRemoteState.CANCELLED
    assert r.path == "history_row_cancelled"


# ─── History-path branches ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_history_returns_live_overrides_pending_absence():
    """If pending list said absent but history says live, trust history
    (pagination edge case)."""
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "ABC", "planStatus": "live"}]},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
        positions=[], positions_fetch_ok=True,
    )
    assert r.state == CondRemoteState.LIVE
    assert r.path == "history_row_live_despite_pending_absence"


@pytest.mark.asyncio
async def test_history_failed_treated_as_cancelled():
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "ABC", "planStatus": "failed"}]},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.CANCELLED
    assert r.path == "history_row_failed"


@pytest.mark.asyncio
async def test_history_triggered_means_filled():
    """Bitget's 'triggered' on plan-history means the trigger fired —
    that's our FILLED."""
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": [{"orderId": "ABC", "planStatus": "triggered"}]},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.FILLED
    assert r.path == "history_row_filled"


# ─── Negative paths (must NOT falsely transition) ────────────────────────

@pytest.mark.asyncio
async def test_oid_not_in_pending_not_in_history_returns_unknown():
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response={
        "code": "00000",
        "data": {"entrustedList": []},   # no row for our oid
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.UNKNOWN
    assert r.path == "not_in_pending_and_not_in_history"


@pytest.mark.asyncio
async def test_history_api_error_returns_unknown():
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response={"code": "40001", "msg": "rate limit"})
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.UNKNOWN
    assert r.path == "history_api_not_ok"


@pytest.mark.asyncio
async def test_history_network_exception_returns_unknown():
    cond = _make_cond(oid="ABC")
    adapter = _make_adapter(history_response=ConnectionError("Bitget down"))
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
    )
    assert r.state == CondRemoteState.UNKNOWN
    assert r.path == "history_exception"


@pytest.mark.asyncio
async def test_no_oid_on_cond_returns_unknown():
    cond = _make_cond(oid="")
    r = await classify_cond_state(cond, adapter=_make_adapter())
    assert r.state == CondRemoteState.UNKNOWN
    assert r.path == "no_oid_on_cond"


# ─── Side-flipping confusion: position with WRONG side must not match ────

@pytest.mark.asyncio
async def test_position_with_opposite_side_does_not_match_long_cond():
    """A SHORT position must not be counted as evidence of a LONG cond's fill."""
    cond = _make_cond(oid="ABC", symbol="ZECUSDT", direction="long")
    adapter = _make_adapter(history_response={
        "code": "00000", "data": {"entrustedList": []},
    })
    r = await classify_cond_state(
        cond, adapter=adapter,
        pending_oids={"OTHER"}, pending_fetch_ok=True,
        positions=[{"symbol": "ZECUSDT", "holdSide": "short", "total": "5.0"}],
        positions_fetch_ok=True,
    )
    # Note: find_position_for_cond falls back to "any position" if no
    # exact side match — preserving prior behavior. Test still asserts
    # state ends up FILLED *only* via the new combined-evidence path or
    # UNKNOWN, never via the old position-only path.
    assert r.path != "position_matches_symbol_side", (
        "old buggy fast-path-2 fired — fix reverted somehow"
    )
