"""Classify a triggered cond's CURRENT state on Bitget via affirmative evidence only.

The entire point of this module:
   → Never transition local state (triggered → cancelled / filled) based on
     "my oid was absent from a remote list".
   → Only transition based on Bitget explicitly telling us the order is
     cancelled / filled / live, via a response row we can name.

Root cause of 2026-04-22 and 2026-04-23 real-money false cancels (see
memory/feedback_exchange_cancel_forensics.md Q1 + Q4): the old reconcile
logic used absence from orders-plan-pending as evidence of cancel. When
Bitget returned 429 / 5xx / paginated partially, the pending set was
incomplete → all plan-type conds looked "gone" → cascaded local CAS-cancel
while Bitget still held them live.

Four possible states this classifier returns:

  LIVE      — pending list contains oid, OR history row shows state=live.
              Action: keep triggered.

  FILLED    — pending list AFFIRMED our oid is gone AND a matching open
              position exists, OR history row shows state=filled /
              state=triggered (Bitget's "triggered" on plan history = fired).
              Action: transition to filled.

  CANCELLED — history row EXISTS and shows state=cancelled.
              Action: transition to cancelled.

  UNKNOWN   — pending said "not there" AND history check failed / rate-
              limited / did not contain a row for this oid. We literally
              do not know. Action: KEEP TRIGGERED, retry next cycle.

The classifier NEVER returns CANCELLED based on absence alone.

The classifier NEVER returns FILLED based on "a position matches symbol+
side" alone — multiple plans on the same (symbol, side) make that
inference unsafe. Position must be paired with affirmative evidence the
oid is gone from pending. (2026-04-24 fix; see ZEC incident docs.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import ConditionalOrder


class CondRemoteState(Enum):
    LIVE = "live"
    FILLED = "filled"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class ClassifyResult:
    state: CondRemoteState
    path: str                # which decision branch fired — for logs
    matched_row: dict[str, Any] | None = None   # history row or position row
    error: str | None = None


# ─── Position-matching helper (mirrors watcher._find_position_for_cond) ──

def _position_size(row: dict[str, Any]) -> float:
    candidates = ("total", "holdSize", "size", "positionAmt", "positionSize", "total_position")
    for key in candidates:
        val = row.get(key)
        if val in (None, "", "0"):
            continue
        try:
            return abs(float(val))
        except (TypeError, ValueError):
            continue
    return 0.0


def find_position_for_cond(
    rows: list[dict[str, Any]] | None,
    cond: ConditionalOrder,
) -> dict[str, Any] | None:
    """Return the Bitget position row matching this cond, or None."""
    if not rows:
        return None
    symbol = cond.symbol.upper()
    want_side = (cond.order.direction or "").lower()
    fallback: dict[str, Any] | None = None
    for row in rows:
        if str(row.get("symbol") or "").upper() != symbol:
            continue
        if _position_size(row) <= 0:
            continue
        side = str(row.get("holdSide") or row.get("posSide") or "").lower()
        if side == want_side:
            return row
        if fallback is None:
            fallback = row
    return fallback


# ─── Main classifier ──────────────────────────────────────────────────

# How far back to ask history for a match. 48h covers fills that opened
# yesterday and are still live today; shorter than Bitget's default ~7d
# retention so API response stays snappy.
HISTORY_LOOKBACK_HOURS = 48


async def classify_cond_state(
    cond: ConditionalOrder,
    adapter: Any,
    *,
    pending_oids: set[str] | None = None,
    pending_fetch_ok: bool = False,
    positions: list[dict[str, Any]] | None = None,
    positions_fetch_ok: bool = False,
    history_lookback_hours: int = HISTORY_LOOKBACK_HOURS,
) -> ClassifyResult:
    """Work out whether a triggered cond's oid is still LIVE on Bitget,
    FILLED into a position, CANCELLED (affirmatively via history), or
    UNKNOWN (we couldn't tell and must NOT change state).

    `adapter` must have `_bitget_request(method, path, *, mode, params)`
    returning the parsed JSON dict with code/msg/data, matching
    LiveExecutionAdapter's interface.

    Callers typically pre-fetch pending_oids and positions once per
    reconcile tick and pass them in, so N conds → 1 pending + 1 position
    fetch (not N). The history query is per-cond because Bitget has no
    batch "status of these N oids" endpoint.
    """
    oid = str(cond.exchange_order_id or "")
    if not oid:
        return ClassifyResult(
            state=CondRemoteState.UNKNOWN,
            path="no_oid_on_cond",
        )

    # ── Fast path 1: pending-list contains our oid → LIVE ──
    if pending_fetch_ok and pending_oids is not None and oid in pending_oids:
        return ClassifyResult(
            state=CondRemoteState.LIVE,
            path="pending_list_has_oid",
        )

    # ── Fast path 2: pending list AFFIRMS the oid is gone AND a matching
    #                 open position exists → FILLED ──
    #
    # 2026-04-24 fix (ZEC real-money incident, see memory/feedback_oid_
    # state_position_matching.md): the previous version of this fast path
    # only required `positions_fetch_ok and positions` — i.e. it inferred
    # FILLED purely from "a position matches my symbol+side" without first
    # confirming our oid was actually gone from the pending list.
    #
    # That logic is unsafe when the user has multiple plan-orders for the
    # same (symbol, side). On 2026-04-24 the user had:
    #   - ZECUSDT plan-order A (live, waiting on trigger $317)
    #   - ZECUSDT plan-order B that had already fired and produced a long position
    # The reconcile classifier saw "ZEC long position exists" and marked
    # plan-order A as FILLED — even though A's oid was still in Bitget's
    # pending list. From that point on, force_replan_line skipped A
    # (because it filters status=triggered), so when the user moved A's
    # parent line, the actual Bitget plan-order didn't update.
    #
    # The fix: require BOTH conditions:
    #   1. `pending_fetch_ok` — we got an authoritative pending list
    #   2. `oid not in pending_oids` — our specific plan is genuinely absent
    # A position alone is not enough; only the pair (oid-gone-from-pending
    # + position-exists) is sufficient evidence that THIS plan filled.
    if (
        pending_fetch_ok
        and pending_oids is not None
        and oid not in pending_oids
        and positions_fetch_ok
        and positions
    ):
        pos = find_position_for_cond(positions, cond)
        if pos is not None:
            return ClassifyResult(
                state=CondRemoteState.FILLED,
                path="oid_gone_from_pending+position_matches",
                matched_row=pos,
            )

    # ── Slow path: affirmative history query ──
    # Only reached when pending list SAID it wasn't there (or pending
    # fetch failed). Must NEVER return CANCELLED without a history row
    # explicitly naming our oid with state=cancelled.
    symbol = cond.symbol.upper().replace("/", "")
    mode = cond.order.exchange_mode
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - history_lookback_hours * 3600 * 1000

    try:
        resp = await adapter._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-history",
            mode=mode,
            params={
                "productType": "USDT-FUTURES",
                "symbol": symbol,
                "planType": "normal_plan",
                "startTime": str(start_ms),
                "endTime": str(now_ms),
                "limit": "100",
            },
        )
    except Exception as exc:
        # Network failure, timeout, or exchange 5xx → cannot determine.
        return ClassifyResult(
            state=CondRemoteState.UNKNOWN,
            path="history_exception",
            error=repr(exc),
        )

    if (resp or {}).get("code") != "00000":
        return ClassifyResult(
            state=CondRemoteState.UNKNOWN,
            path="history_api_not_ok",
            error=f"code={resp.get('code')} msg={resp.get('msg')}",
        )

    rows = (resp.get("data") or {}).get("entrustedList") or []

    for row in rows:
        if str(row.get("orderId") or "") != oid:
            continue
        state_raw = (row.get("planStatus") or row.get("state") or "").lower()
        # Bitget state vocab for normal_plan history:
        #   "cancelled" — confirmed cancel (manual / reconcile / exchange)
        #   "live" / "new" — unfinished, waiting for trigger
        #   "triggered" / "executed" / "filled" — fired, position opened
        #   "failed" — rejected
        if state_raw == "cancelled":
            return ClassifyResult(
                state=CondRemoteState.CANCELLED,
                path="history_row_cancelled",
                matched_row=row,
            )
        if state_raw in ("triggered", "executed", "filled"):
            return ClassifyResult(
                state=CondRemoteState.FILLED,
                path="history_row_filled",
                matched_row=row,
            )
        if state_raw in ("live", "new", "active"):
            # History says live but pending didn't — pagination / cache
            # edge case. Trust the authoritative history row.
            return ClassifyResult(
                state=CondRemoteState.LIVE,
                path="history_row_live_despite_pending_absence",
                matched_row=row,
            )
        if state_raw == "failed":
            # "failed" means Bitget rejected the plan (bad trigger etc).
            # For our purposes that's equivalent to cancelled.
            return ClassifyResult(
                state=CondRemoteState.CANCELLED,
                path="history_row_failed",
                matched_row=row,
            )
        # Unknown vocab — don't guess. Stay UNKNOWN, next tick re-check.
        return ClassifyResult(
            state=CondRemoteState.UNKNOWN,
            path=f"history_row_unknown_state:{state_raw}",
            matched_row=row,
        )

    # Not in pending AND not in history. Could be:
    #   - order older than lookback window
    #   - Bitget's history endpoint temporarily missing it
    #   - we just submitted and Bitget hasn't indexed yet
    # Any of these = ambiguous = UNKNOWN (keep triggered, retry).
    return ClassifyResult(
        state=CondRemoteState.UNKNOWN,
        path="not_in_pending_and_not_in_history",
    )
