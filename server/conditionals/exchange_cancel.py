from __future__ import annotations

from typing import Any

from .types import ConditionalOrder


def _mode_for_exchange(cond: ConditionalOrder) -> str:
    mode = (cond.order.exchange_mode if cond.order else "live") or "live"
    return "demo" if mode == "paper" else mode


def _oid_set(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows or []:
        oid = str(row.get("orderId") or row.get("order_id") or "")
        if oid:
            out.add(oid)
    return out


async def cancel_exchange_order_for_cond(
    cond: ConditionalOrder,
    *,
    adapter: Any | None = None,
) -> dict[str, Any]:
    """Cancel a conditional's live Bitget order before local state changes.

    Returns ok=True only when Bitget confirmed a cancel, or when a fresh
    exchange-truth query shows the order is already absent. Callers should not
    mark local state cancelled when ok=False.
    """
    oid = str(cond.exchange_order_id or "")
    if not oid:
        return {
            "ok": True,
            "attempted": False,
            "reason": "no_exchange_order_id",
            "attempts": [],
        }

    if adapter is None:
        from server.execution.live_adapter import LiveExecutionAdapter

        adapter = LiveExecutionAdapter()

    if not adapter.has_api_keys():
        return {
            "ok": False,
            "attempted": False,
            "reason": "api_keys_missing",
            "order_id": oid,
            "attempts": [],
        }

    symbol = cond.symbol.upper().replace("/", "")
    mode = _mode_for_exchange(cond)
    attempts: list[dict[str, Any]] = []

    try:
        regular = await adapter.cancel_order(symbol, oid, mode)
        attempts.append({"path": "regular", "result": regular})
        if (regular or {}).get("ok"):
            return {
                "ok": True,
                "attempted": True,
                "reason": "regular_cancel_ok",
                "order_id": oid,
                "mode": mode,
                "attempts": attempts,
            }
    except Exception as exc:
        attempts.append({"path": "regular", "error": repr(exc)})

    try:
        plan = await adapter.cancel_plan_order_any_type(symbol, oid, mode)
        attempts.append({"path": "plan", "result": plan})
        if (plan or {}).get("ok"):
            return {
                "ok": True,
                "attempted": True,
                "reason": "plan_cancel_ok",
                "order_id": oid,
                "mode": mode,
                "attempts": attempts,
            }
    except Exception as exc:
        attempts.append({"path": "plan", "error": repr(exc)})

    pending_oids: set[str] = set()
    pending_fetch_ok = False
    pending_errors: list[str] = []
    try:
        pending_oids |= _oid_set(await adapter.get_pending_orders(mode, symbol=symbol))
        pending_fetch_ok = True
    except Exception as exc:
        pending_errors.append(f"regular_pending={exc!r}")
    try:
        pending_oids |= _oid_set(
            await adapter.get_pending_plan_orders(mode, plan_type="normal_plan", symbol=symbol)
        )
        pending_fetch_ok = True
    except Exception as exc:
        pending_errors.append(f"plan_pending={exc!r}")

    if pending_fetch_ok and oid not in pending_oids:
        return {
            "ok": True,
            "attempted": True,
            "reason": "already_absent_from_bitget",
            "order_id": oid,
            "mode": mode,
            "attempts": attempts,
            "pending_checked": True,
        }

    return {
        "ok": False,
        "attempted": True,
        "reason": "bitget_cancel_not_confirmed",
        "order_id": oid,
        "mode": mode,
        "attempts": attempts,
        "pending_checked": pending_fetch_ok,
        "pending_errors": pending_errors,
    }
