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

    # ── Fallback: both cancel paths failed. Confirm "already gone" via
    # AFFIRMATIVE Bitget evidence, never via absence inference. Using
    # orders-plan-history so we can see the row's final state.
    #
    # Old logic said "not in pending list → already gone" — that was
    # the same class of bug as the reconcile 429 false-cancel. If the
    # pending fetch hits 429, oid not in pending_oids is meaningless.
    # Now we only return ok=True when Bitget's own history row says
    # the oid is cancelled/filled, or when the pending list CONTAINS
    # the oid (we can see it's live, so cancel really did fail).
    import time
    history_checked = False
    history_error: str | None = None
    history_state: str | None = None
    try:
        now_ms = int(time.time() * 1000)
        resp = await adapter._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-history",
            mode=mode,
            params={
                "productType": "USDT-FUTURES",
                "symbol": symbol,
                "planType": "normal_plan",
                "startTime": str(now_ms - 48 * 3600 * 1000),
                "endTime": str(now_ms),
                "limit": "100",
            },
        )
        if resp.get("code") == "00000":
            history_checked = True
            rows = (resp.get("data") or {}).get("entrustedList") or []
            for row in rows:
                if str(row.get("orderId") or "") != oid:
                    continue
                history_state = (row.get("planStatus") or row.get("state") or "").lower()
                break
        else:
            history_error = f"code={resp.get('code')} msg={resp.get('msg')}"
    except Exception as exc:
        history_error = repr(exc)

    if history_state in ("cancelled", "filled", "triggered", "executed", "failed"):
        return {
            "ok": True,
            "attempted": True,
            "reason": f"already_{history_state}_per_history",
            "order_id": oid,
            "mode": mode,
            "attempts": attempts,
            "history_checked": history_checked,
            "history_state": history_state,
        }

    return {
        "ok": False,
        "attempted": True,
        "reason": "bitget_cancel_not_confirmed_history_unknown",
        "order_id": oid,
        "mode": mode,
        "attempts": attempts,
        "history_checked": history_checked,
        "history_state": history_state,
        "history_error": history_error,
    }
