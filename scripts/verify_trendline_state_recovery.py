"""P3 deep-review harness: simulate restart recovery without duplicate orders."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import server.execution.live_adapter as live_adapter
from server.execution.types import OrderIntent
from server.strategy import mar_bb_runner as runner
from server.strategy import trendline_order_manager as tom


class FakeAdapter:
    intents: list[OrderIntent] = []

    def has_api_keys(self) -> bool:
        return True

    async def _bitget_request(self, method: str, path: str, *, mode: str, body=None, params=None):
        return {"code": "00000", "data": {"orderId": "fake"}}

    async def get_open_position_symbols(self, mode: str) -> set[str]:
        return {"LINKUSDT"}

    async def cancel_plan_order(self, symbol: str, order_id: str, mode: str, *, plan_type: str = "normal_plan"):
        return {"ok": True, "order_id": order_id, "plan_type": plan_type}

    async def submit_live_plan_entry(self, intent: OrderIntent, mode: str, trigger_price: float):
        self.intents.append(intent)
        return {"ok": True, "exchange_order_id": "duplicate-should-not-happen"}


async def main() -> None:
    original_active_file = tom.ACTIVE_LINES_FILE
    original_adapter = live_adapter.LiveExecutionAdapter
    original_params = dict(runner._trendline_params)
    FakeAdapter.intents.clear()

    tmp = PROJECT_ROOT / "data" / "codex_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
            tom.ACTIVE_LINES_FILE = tmp / "p3_trendline_active_orders.json"
            live_adapter.LiveExecutionAdapter = FakeAdapter  # type: ignore[assignment]
            now = time.time()
            tom._save_active([
                tom.ActiveLineOrder(
                    symbol="LINKUSDT",
                    timeframe="1h",
                    kind="support",
                    slope=0.0,
                    intercept=100.0,
                    anchor1_bar=1,
                    anchor2_bar=5,
                    bar_count=10,
                    current_projected_price=100.0,
                    limit_price=100.2,
                    stop_price=100.0,
                    tp_price=103.206,
                    exchange_order_id="existing-plan",
                    created_ts=now - 7200,
                    last_updated_ts=now - 3700,
                    status="placed",
                )
            ])

            # Restart means memory is empty, but Bitget says the symbol is held.
            runner._trendline_params.clear()
            result = await tom.update_trendline_orders(
                [],
                current_bar_index=-1,
                cfg={
                    "buffer_pct": 0.10,
                    "rr": 15.0,
                    "prices": {"LINKUSDT": 101.0},
                    "leverage": 30,
                    "equity": 1000.0,
                    "risk_pct": 0.01,
                    "max_position_pct": 0.50,
                    "mode": "demo",
                    "held_symbols": {"LINKUSDT"},
                    "tf_risk": {"1h": 0.015},
                    "tf_buffer": {"1h": 0.20},
                },
            )
            active_after = tom._load_active()
            ao = active_after[0]
            runner.register_trendline_params(
                ao.symbol,
                slope=ao.slope,
                intercept=ao.intercept,
                entry_bar=ao.bar_count - 1,
                entry_price=ao.limit_price,
                side="long",
                tf=ao.timeframe,
                created_ts=now - 3600,
                tp_price=ao.tp_price,
                last_sl_set=ao.stop_price,
            )

            print(json.dumps({
                "update_result": result,
                "active_status_after_restart_sync": ao.status,
                "restored_params": runner._trendline_params.get("LINKUSDT"),
                "duplicate_orders_submitted": len(FakeAdapter.intents),
            }, indent=2, sort_keys=True))
    finally:
        tom.ACTIVE_LINES_FILE = original_active_file
        live_adapter.LiveExecutionAdapter = original_adapter  # type: ignore[assignment]
        runner._trendline_params.clear()
        runner._trendline_params.update(original_params)


if __name__ == "__main__":
    asyncio.run(main())
