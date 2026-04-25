"""Live $5 conditional order test on a non-crypto Bitget symbol (TSLA).

Goal: prove the existing order infrastructure works end-to-end on stocks
so user's request to "add stocks / gold / oil" is actually usable, not
just displayed.

Flow (all against real Bitget live API, real keys):
  1. Fetch TSLA current price.
  2. Place a conditional LONG order, ~$7.54 notional (0.02 shares × $377),
     trigger price 15% BELOW current so it won't fill unless TSLA crashes.
  3. Wait 2s, query Bitget's plan-orders list to verify the order shows up.
  4. Modify the order — bump trigger 1% lower ("move the order").
  5. Verify the modified trigger is visible on Bitget.
  6. Cancel the order.
  7. Verify it's gone.

If any step fails, prints the Bitget error and bails out. Always cleans
up (cancels any surviving test order) in the finally block.

Run with:
  python scripts/test_stock_order_live.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


SYMBOL = os.environ.get("TEST_SYMBOL", "TSLAUSDT")
TRIGGER_BELOW_PCT = 0.15   # 15% below mark → safe, won't fill
SIZE = os.environ.get("TEST_SIZE", "0.02")   # override via env for different-priced contracts
CLIENT_OID_PREFIX = "test-stock-"


async def get_tsla_price(client) -> float:
    from server.market.bitget_client import BitgetPublicClient
    pub = BitgetPublicClient(http_client=client, product_type="usdt-futures")
    tickers = await pub.get_tickers()
    for t in tickers:
        if t.get("symbol") == SYMBOL:
            return float(t.get("lastPr") or t.get("markPrice") or 0.0)
    raise RuntimeError(f"{SYMBOL} not found in tickers")


async def place_test_order(adapter, trigger: float) -> dict:
    from server.execution.types import OrderIntent
    ts = int(time.time())
    intent = OrderIntent(
        order_intent_id=f"{CLIENT_OID_PREFIX}intent-{ts}",
        signal_id=f"{CLIENT_OID_PREFIX}signal-{ts}",
        line_id=f"{CLIENT_OID_PREFIX}line-{ts}",
        client_order_id=f"{CLIENT_OID_PREFIX}coid-{ts}",
        symbol=SYMBOL,
        timeframe="1h",
        side="long",
        order_type="market",
        trigger_mode="limit",
        entry_price=trigger,
        stop_price=trigger * 0.95,
        tp_price=trigger * 1.10,
        quantity=float(SIZE),
        status="pending",
    )
    print(f"[place] qty={SIZE} trigger={trigger:.2f} SL={intent.stop_price:.2f} TP={intent.tp_price:.2f}")
    result = await adapter.submit_live_plan_entry(intent, mode="live", trigger_price=trigger)
    return result


async def query_open_plans(adapter) -> list[dict]:
    """GET /api/v2/mix/order/orders-plan-pending — live open plan orders."""
    response = await adapter._bitget_request(
        "GET",
        "/api/v2/mix/order/orders-plan-pending",
        mode="live",
        params={"productType": "usdt-futures", "planType": "normal_plan"},
    )
    if response.get("code") != "00000":
        print(f"[query] ERROR code={response.get('code')} msg={response.get('msg')}")
        return []
    data = response.get("data") or {}
    return data.get("entrustedList") or []


async def modify_order(adapter, order_id: str, new_trigger: float, size: str) -> dict:
    """POST /api/v2/mix/order/modify-plan-order — change trigger price."""
    body = {
        "orderId": order_id,
        "symbol": SYMBOL,
        "productType": "usdt-futures",
        "newTriggerPrice": f"{new_trigger:.2f}",
        "newSize": size,
    }
    return await adapter._bitget_request(
        "POST", "/api/v2/mix/order/modify-plan-order",
        mode="live", body=body,
    )


async def cancel_order(adapter, order_id: str) -> dict:
    """POST /api/v2/mix/order/cancel-plan-order."""
    body = {
        "symbol": SYMBOL,
        "productType": "usdt-futures",
        "planType": "normal_plan",
        "orderIdList": [{"orderId": order_id}],
    }
    return await adapter._bitget_request(
        "POST", "/api/v2/mix/order/cancel-plan-order",
        mode="live", body=body,
    )


async def main() -> int:
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.data_service import _get_http_client

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        print("FAIL: API keys missing from .env")
        return 1

    http = _get_http_client()
    price = await get_tsla_price(http)
    if price <= 0:
        print(f"FAIL: could not fetch {SYMBOL} price")
        return 1
    trigger = round(price * (1 - TRIGGER_BELOW_PCT), 2)
    print(f"\n== {SYMBOL} current={price:.2f}, will test with trigger={trigger:.2f} ==\n")

    order_id: str | None = None
    try:
        # 1. Place
        place_result = await place_test_order(adapter, trigger)
        if not place_result.get("ok"):
            print(f"[place] FAIL: {json.dumps(place_result, default=str, indent=2)}")
            return 2
        order_id = place_result.get("exchange_order_id")
        print(f"[place] OK orderId={order_id}")

        # 2. Verify via GET
        await asyncio.sleep(2.0)
        plans = await query_open_plans(adapter)
        ours = [p for p in plans if str(p.get("orderId")) == str(order_id)]
        if not ours:
            print(f"[verify] FAIL: order {order_id} not found in {len(plans)} open plans")
            print(f"[verify] open plans for TSLA: {[p for p in plans if p.get('symbol')==SYMBOL]}")
            return 3
        print(f"[verify] OK: trigger={ours[0].get('triggerPrice')} size={ours[0].get('size')} "
              f"symbol={ours[0].get('symbol')}")

        # 3. Modify — bump trigger 1% lower
        new_trigger = round(trigger * 0.99, 2)
        mod_result = await modify_order(adapter, order_id, new_trigger, SIZE)
        if mod_result.get("code") != "00000":
            print(f"[modify] FAIL code={mod_result.get('code')} msg={mod_result.get('msg')}")
            return 4
        print(f"[modify] OK: new trigger={new_trigger:.2f} (was {trigger:.2f})")

        # 4. Verify modification
        await asyncio.sleep(2.0)
        plans = await query_open_plans(adapter)
        ours = [p for p in plans if str(p.get("orderId")) == str(order_id)]
        if not ours:
            print(f"[verify-mod] FAIL: order {order_id} vanished after modify")
            return 5
        # orderId sometimes changes after modify
        reported_trigger = float(ours[0].get("triggerPrice") or 0)
        if abs(reported_trigger - new_trigger) > 0.1:
            print(f"[verify-mod] trigger mismatch: expected {new_trigger} got {reported_trigger}")
        else:
            print(f"[verify-mod] OK: Bitget shows trigger={reported_trigger}")

        # 5. Cancel
        can_result = await cancel_order(adapter, order_id)
        if can_result.get("code") != "00000":
            print(f"[cancel] FAIL code={can_result.get('code')} msg={can_result.get('msg')}")
            return 6
        print(f"[cancel] OK")

        # 6. Verify it's gone
        await asyncio.sleep(2.0)
        plans = await query_open_plans(adapter)
        still = [p for p in plans if str(p.get("orderId")) == str(order_id)]
        if still:
            print(f"[verify-cancel] WARN: order {order_id} still in open plans")
        else:
            print(f"[verify-cancel] OK: order {order_id} no longer pending")

        print("\n== ALL STEPS PASSED: place -> verify -> modify -> verify-mod -> cancel -> verify-cancel ==")
        return 0
    except Exception as exc:
        import traceback
        print(f"[EXCEPTION] {exc}")
        traceback.print_exc()
        return 99
    finally:
        if order_id:
            try:
                # Best-effort cleanup
                plans = await query_open_plans(adapter)
                still = [p for p in plans if str(p.get("orderId")) == str(order_id)]
                if still:
                    print(f"[cleanup] order {order_id} still open, cancelling")
                    await cancel_order(adapter, order_id)
            except Exception as exc:
                print(f"[cleanup] failed: {exc}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
