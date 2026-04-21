"""One-off: check if HYPEUSDT position has SL/TP plans on Bitget."""
import sys, os, json, asyncio
from pathlib import Path

env_path = Path(__file__).resolve().parents[1] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def main():
    from server.execution.live_adapter import LiveExecutionAdapter
    a = LiveExecutionAdapter()
    print("has_api_keys:", a.has_api_keys())
    resp = await a._bitget_request(
        "GET", "/api/v2/mix/order/orders-plan-pending",
        mode="live",
        params={
            "symbol": "HYPEUSDT",
            "productType": "USDT-FUTURES",
            "planType": "profit_loss",
        },
    )
    print("code:", resp.get("code"))
    d = resp.get("data") or {}
    if isinstance(d, dict):
        rows = d.get("entrustedList") or d.get("list") or []
    else:
        rows = d
    print(f"HYPE SL/TP plans: {len(rows)}")
    for r in rows[:10]:
        oid = str(r.get("orderId", ""))[-10:]
        print(
            f"  planType={r.get('planType')}  trigger={r.get('triggerPrice')}  "
            f"side={r.get('side')}  holdSide={r.get('holdSide')}  oid=...{oid}"
        )


asyncio.run(main())
