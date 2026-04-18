"""P2 deep-review harness: print buffer/price/quantity unit conversions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.execution.live_adapter import LiveExecutionAdapter
from server.strategy.trendline_order_manager import _buffer_fraction_for_tf, _qty_for_risk


def main() -> None:
    cfg = {
        "buffer_pct": 0.10,
        "tf_buffer": {"5m": 0.05, "15m": 0.10, "1h": 0.20, "4h": 0.30},
        "tf_risk": {"5m": 0.003, "15m": 0.007, "1h": 0.015, "4h": 0.030},
        "equity": 1000.0,
        "leverage": 30,
        "max_position_pct": 0.50,
    }
    rows = []
    for tf in ("5m", "15m", "1h", "4h"):
        buffer_fraction = _buffer_fraction_for_tf(tf, cfg)
        line = 100.0
        entry = line * (1 + buffer_fraction)
        stop = line
        qty, risk_usd, capped = _qty_for_risk(
            equity=cfg["equity"],
            risk_pct=cfg["tf_risk"][tf],
            entry_price=entry,
            stop_price=stop,
            leverage=cfg["leverage"],
            max_position_pct=cfg["max_position_pct"],
        )
        rows.append({
            "tf": tf,
            "buffer_config_percent": cfg["tf_buffer"][tf],
            "buffer_fraction": buffer_fraction,
            "entry": entry,
            "stop": stop,
            "stop_distance": abs(entry - stop),
            "risk_pct": cfg["tf_risk"][tf],
            "risk_usd": risk_usd,
            "quantity": qty,
            "notional": qty * entry,
            "capped": capped,
        })

    adapter = LiveExecutionAdapter.__new__(LiveExecutionAdapter)
    price_precision = {
        "tickSz_0.01": adapter._normalize_price(100.129, {"tickSz": "0.01"}),
        "raw_pricePlace_3_endStep_5": adapter._normalize_price(
            50.1239, {"pricePlace": "3", "priceEndStep": "5"}
        ),
        "legacy_pricePlace_tick": adapter._normalize_price(50.1239, {"pricePlace": "0.001"}),
    }
    print(json.dumps({"buffer_and_quantity": rows, "price_precision": price_precision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
