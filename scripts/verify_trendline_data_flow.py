"""P0 deep-review harness: trace trendline scan -> order -> fill -> SL move.

This is intentionally local by default. It does not call Bitget or place real
orders. Use it to print the exact numeric path for one LINKUSDT 1h example.
"""
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


class FakeBitgetAdapter:
    intents: list[dict] = []
    sl_updates: list[dict] = []

    def has_api_keys(self) -> bool:
        return True

    async def _bitget_request(self, method: str, path: str, *, mode: str, body=None, params=None):
        return {"code": "00000", "data": {"orderId": "fake-cancel"}}

    async def get_open_position_symbols(self, mode: str) -> set[str]:
        return set()

    async def cancel_plan_order(self, symbol: str, order_id: str, mode: str, *, plan_type: str = "normal_plan"):
        return {"ok": True, "order_id": order_id, "plan_type": plan_type}

    async def get_position_sl_trigger_price(self, symbol, hold_side, mode="demo", *, entry_price=None):
        if self.sl_updates:
            return float(self.sl_updates[-1]["new_sl"])
        return None

    async def submit_live_plan_entry(self, intent: OrderIntent, mode: str, trigger_price: float):
        record = {
            "symbol": intent.symbol,
            "tf": intent.timeframe,
            "side": intent.side,
            "order_type": intent.order_type,
            "trigger_price": trigger_price,
            "entry_price": intent.entry_price,
            "stop_price": intent.stop_price,
            "tp_price": intent.tp_price,
            "quantity": intent.quantity,
            "mode": mode,
        }
        self.intents.append(record)
        return {"ok": True, "exchange_order_id": "fake-plan-1", "request_body_excerpt": record}

    async def update_position_sl_tp(self, symbol, hold_side, new_sl=None, new_tp=None, mode="demo"):
        record = {"symbol": symbol, "hold_side": hold_side, "new_sl": new_sl, "new_tp": new_tp, "mode": mode}
        self.sl_updates.append(record)
        return {
            "ok": True,
            "new_sl": str(new_sl),
            "actual_sl_after": float(new_sl),
            "sl_verified": True,
            "updates": ["sl_ok"],
            "cancelled_order_ids": ["fake-old-sl"],
        }


async def main() -> None:
    original_active_file = tom.ACTIVE_LINES_FILE
    original_adapter = live_adapter.LiveExecutionAdapter
    original_positions = runner._get_bitget_positions
    original_params = dict(runner._trendline_params)
    FakeBitgetAdapter.intents.clear()
    FakeBitgetAdapter.sl_updates.clear()

    tmp = PROJECT_ROOT / "data" / "codex_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
            tom.ACTIVE_LINES_FILE = tmp / "p0_trendline_active_orders.json"
            tom.ACTIVE_LINES_FILE.write_text("[]", encoding="utf-8")
            live_adapter.LiveExecutionAdapter = FakeBitgetAdapter  # type: ignore[assignment]

            signal = {
                "symbol": "LINKUSDT",
                "timeframe": "1h",
                "kind": "support",
                "slope": 0.1,
                "intercept": 99.1,
                "bar_count": 10,
                "anchor1_bar": 1,
                "anchor2_bar": 5,
            }
            cfg = {
                "buffer_pct": 0.10,
                "rr": 15.0,
                "prices": {"LINKUSDT": 101.0},
                "leverage": 30,
                "equity": 1000.0,
                "risk_pct": 0.01,
                "max_position_pct": 0.50,
                "mode": "demo",
                "tf_risk": {"1h": 0.015},
                "tf_buffer": {"1h": 0.20},
            }

            order_result = await tom.update_trendline_orders([signal], current_bar_index=9, cfg=cfg)
            active = tom._load_active()
            ao = active[0]

            runner._trendline_params.clear()
            runner.register_trendline_params(
                ao.symbol,
                slope=ao.slope,
                intercept=ao.intercept,
                entry_bar=ao.bar_count - 1,
                entry_price=ao.limit_price,
                side="long",
                tf=ao.timeframe,
                created_ts=int(time.time()) - 3600,
                tp_price=ao.tp_price,
                last_sl_set=ao.stop_price,
            )

            async def fake_positions():
                return ([{
                    "symbol": "LINKUSDT",
                    "holdSide": "long",
                    "total": "1",
                    "averageOpenPrice": str(ao.limit_price),
                }], True)

            runner._get_bitget_positions = fake_positions  # type: ignore[assignment]
            sl_updates = await runner._update_trailing_stops({"mode": "demo"})

            buffer = 0.20 / 100.0
            backtest_formula = {
                "line_price": 100.0,
                "entry": 100.0 * (1.0 + buffer),
                "stop": 100.0,
                "target": 100.0 * (1.0 + buffer) * (1.0 + buffer * 15.0),
                "risk_fraction": buffer,
            }

            print(json.dumps({
                "scan_detection": signal,
                "order_manager_result": order_result,
                "plan_order": FakeBitgetAdapter.intents[-1],
                "fill_registration": runner._trendline_params.get("LINKUSDT"),
                "sl_move_count": sl_updates,
                "sl_move": FakeBitgetAdapter.sl_updates[-1] if FakeBitgetAdapter.sl_updates else None,
                "backtest_formula": backtest_formula,
            }, indent=2, sort_keys=True))
    finally:
        tom.ACTIVE_LINES_FILE = original_active_file
        live_adapter.LiveExecutionAdapter = original_adapter  # type: ignore[assignment]
        runner._get_bitget_positions = original_positions  # type: ignore[assignment]
        runner._trendline_params.clear()
        runner._trendline_params.update(original_params)


if __name__ == "__main__":
    asyncio.run(main())
