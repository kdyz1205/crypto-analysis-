"""Verify whether oid 1431109453823860736 (our 'cancelled' HYPE cond) is
actually still LIVE on Bitget. If it is, reconcile produced a false
positive and we should NOT have marked the cond cancelled."""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OID = "1431109453823860736"

async def main():
    from server.execution.live_adapter import LiveExecutionAdapter
    ad = LiveExecutionAdapter()

    print("=" * 80)
    print(f"Step 1: current pending plan orders (any symbol) — is {OID} there?")
    print("=" * 80)
    rows = []
    found = None
    for plan_type in ("normal_plan", "profit_loss", "pos_profit", "pos_loss", "profit_plan", "loss_plan", "track_plan"):
        resp = await ad._bitget_request(
            "GET", "/api/v2/mix/order/orders-plan-pending",
            mode="live",
            params={"productType": "USDT-FUTURES", "planType": plan_type},
        )
        these = (resp.get("data") or {}).get("entrustedList") or []
        print(f"planType={plan_type}: code={resp.get('code')}  rows={len(these)}")
        for r in these:
            oid = str(r.get("orderId", ""))
            marker = "  <<< OUR OID" if oid == OID else ""
            print(f"  {oid}  {r.get('symbol')}  {r.get('side')}  size={r.get('size')} "
                  f"trig={r.get('triggerPrice')}  state={r.get('planStatus') or r.get('state')}"
                  f"  planType={plan_type}{marker}")
            if oid == OID:
                found = r
            rows.append(r)

    print()
    print(f"Total pending plan rows across all plan types: {len(rows)}")

    # Also query HYPE-specific plan history for the last hour to see if the oid was recently cancelled
    print()
    print("=" * 80)
    print(f"Step 2: HYPE plan history (last 2 hours)")
    print("=" * 80)
    import time
    now_ms = int(time.time() * 1000)
    resp = await ad._bitget_request(
        "GET", "/api/v2/mix/order/orders-plan-history",
        mode="live",
        params={
            "productType": "USDT-FUTURES",
            "symbol": "HYPEUSDT",
            "planType": "normal_plan",
            "startTime": str(now_ms - 2 * 3600 * 1000),
            "endTime": str(now_ms),
            "limit": "50",
        },
    )
    hist = (resp.get("data") or {}).get("entrustedList") or []
    print(f"code={resp.get('code')}  HYPE plan history rows: {len(hist)}")
    for r in hist:
        oid = str(r.get("orderId", ""))
        marker = "  <<< OUR OID" if oid == OID else ""
        print(f"  {oid}  state={r.get('planStatus') or r.get('state')}  "
              f"size={r.get('size')}  trig={r.get('triggerPrice')}  "
              f"cTime={r.get('cTime')}  uTime={r.get('uTime')}{marker}")

    print()
    print("=" * 80)
    if found:
        print(f"VERDICT: oid {OID} IS LIVE on Bitget right now.")
        print("Our cond.status = cancelled is a FALSE POSITIVE.")
        print()
        print("Full record:")
        for k, v in sorted(found.items()):
            print(f"  {k}: {v}")
    else:
        print(f"VERDICT: oid {OID} is NOT in Bitget's pending list.")
        print("Reconcile was right to mark it cancelled.")

asyncio.run(main())
