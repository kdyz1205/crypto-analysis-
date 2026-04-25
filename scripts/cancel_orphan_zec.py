"""Cancel the orphan ZEC plan-order discovered 2026-04-24:
    orderId=1431407611887972352  symbol=ZECUSDT  side=BUY trigger=$317.12

User explicitly authorized cancelling this single order. Will NOT touch
the support line itself (manual_line_id stays in store).

Steps:
  1. Cancel on Bitget (POST /api/v2/mix/order/cancel-plan-order)
  2. Verify it's gone from Bitget pending list
  3. Update local cond status filled→cancelled with audit reason
"""
from __future__ import annotations
import asyncio, sys, io, os
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

ORDER_ID = "1431407611887972352"
SYMBOL = "ZECUSDT"


async def main() -> int:
    from server.execution.live_adapter import LiveExecutionAdapter
    a = LiveExecutionAdapter()
    if not a.has_api_keys():
        print("FAIL: API keys not loaded"); return 1

    # 1. Pre-cancel verify it IS still live (don't try to cancel a phantom)
    pre = await a._bitget_request(
        "GET", "/api/v2/mix/order/orders-plan-pending", mode="live",
        params={"productType": "usdt-futures", "planType": "normal_plan"},
    )
    rows = (pre.get("data") or {}).get("entrustedList") or []
    target = [r for r in rows if str(r.get("orderId")) == ORDER_ID]
    if not target:
        print(f"NOTE: orderId {ORDER_ID} not in pending list — already cancelled or filled.")
        # Still update local store
    else:
        print(f"[pre] confirmed live: trigger={target[0].get('triggerPrice')} "
              f"size={target[0].get('size')} side={target[0].get('side')}")

    # 2. Cancel
    if target:
        body = {
            "symbol": SYMBOL, "productType": "usdt-futures",
            "planType": "normal_plan",
            "orderIdList": [{"orderId": ORDER_ID}],
        }
        cancel_resp = await a._bitget_request(
            "POST", "/api/v2/mix/order/cancel-plan-order",
            mode="live", body=body,
        )
        if cancel_resp.get("code") != "00000":
            print(f"FAIL: Bitget cancel rejected: {cancel_resp}"); return 2
        print("[cancel] OK")

        # 3. Verify gone
        await asyncio.sleep(2)
        post = await a._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-pending", mode="live",
            params={"productType": "usdt-futures", "planType": "normal_plan"},
        )
        rows2 = (post.get("data") or {}).get("entrustedList") or []
        still = [r for r in rows2 if str(r.get("orderId")) == ORDER_ID]
        if still:
            print(f"WARN: order still showing in pending list 2s after cancel"); return 3
        print("[verify] OK: order gone from Bitget pending list")

    # 4. Update local cond store
    try:
        from server.conditionals import store as _cond_store
        from server.conditionals.types import ConditionalEvent
        from server.conditionals.store import now_ts
        store = _cond_store._instance if hasattr(_cond_store, "_instance") else None
        if store is None:
            from server.conditionals.store import ConditionalStore as _CS
            store = _CS()
        # Find cond with this exchange_order_id
        all_conds = store.list_all()
        target_cond = next((c for c in all_conds if str(getattr(c, "exchange_order_id", "")) == ORDER_ID), None)
        if not target_cond:
            print(f"NOTE: no local cond with exchange_order_id={ORDER_ID}; nothing to update locally")
            return 0
        winner = store.set_status_if(
            target_cond.conditional_id,
            from_status=str(target_cond.status),
            to_status="cancelled",
            reason="manual cancel via cancel_orphan_zec.py — user authorized 2026-04-24",
        )
        if winner is not None:
            store.append_event(target_cond.conditional_id, ConditionalEvent(
                ts=now_ts(), kind="cancelled",
                message=f"manual user-authorized cancel of orphan Bitget order {ORDER_ID}",
            ))
            print(f"[local] cond {target_cond.conditional_id}: {target_cond.status}→cancelled")
        else:
            print(f"[local] CAS lost — cond {target_cond.conditional_id} status changed concurrently; skipped")
    except Exception as exc:
        print(f"[local] update failed: {exc}")
        return 4

    print("\n== ORPHAN ORDER CANCELLED ==")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
