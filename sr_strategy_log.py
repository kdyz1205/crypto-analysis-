"""
S/R Strategy Monitor — Continuous scanner across all timeframes.
Logs every detection cycle to sr_strategy.log.
Runs independently, no web server needed.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

# Load .env
try:
    from dotenv import load_dotenv
    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env):
        load_dotenv(_env, override=True)
except ImportError:
    pass

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

from server.strategy.pivots import detect_pivots
from server.strategy.zones import detect_horizontal_zones
from server.strategy.trendlines import detect_trendlines
from server.strategy.signals import generate_signals
from server.strategy.zone_signals import generate_zone_signals
from server.strategy.config import StrategyConfig
from server.data_service import get_ohlcv_with_df
import pandas as pd

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sr_strategy.log")
SYMBOL = "HYPEUSDT"
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
SCAN_INTERVAL = 60  # seconds between full scans


async def scan_all_timeframes():
    cfg = StrategyConfig(pivot_left=3, pivot_right=3, lookback_bars=500)
    log = open(LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    cycle = 0

    print(f"=== S/R Strategy Monitor ===", flush=True)
    print(f"Symbol: {SYMBOL}", flush=True)
    print(f"Timeframes: {TIMEFRAMES}", flush=True)
    print(f"Scan interval: {SCAN_INTERVAL}s", flush=True)
    print(f"Log: {LOG_PATH}", flush=True)
    print(flush=True)

    while True:
        cycle += 1
        now = datetime.now().isoformat()
        total_signals = 0

        for tf in TIMEFRAMES:
            try:
                df, _ = await get_ohlcv_with_df(SYMBOL, tf, None, 365, history_mode="fast_window")
                pdf = df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
                pdf = pdf.rename(columns={"open_time": "timestamp"})
                pdf["timestamp"] = pdf["timestamp"].map(lambda v: int(pd.Timestamp(v).timestamp()))
                for c in ("open", "high", "low", "close", "volume"):
                    pdf[c] = pd.to_numeric(pdf[c])
                pdf = pdf.reset_index(drop=True)

                close = float(pdf.iloc[-1]["close"])

                # M1: Structure extraction
                pivots = detect_pivots(pdf, cfg)

                # M1: Trendlines
                detection = detect_trendlines(pdf, pivots, cfg, symbol=SYMBOL, timeframe=tf)

                # M2: Horizontal zones
                zones = detect_horizontal_zones(pdf, pivots, cfg, symbol=SYMBOL, timeframe=tf, max_zones_per_side=3)

                # M3: Trendline signals
                line_sigs = generate_signals(pdf, detection.active_lines, cfg)

                # M3: Zone signals
                zone_sigs = generate_zone_signals(pdf, zones, cfg, symbol=SYMBOL, timeframe=tf)

                all_sigs = line_sigs + zone_sigs
                total_signals += len(all_sigs)

                # Log summary
                entry = {
                    "time": now,
                    "cycle": cycle,
                    "symbol": SYMBOL,
                    "timeframe": tf,
                    "close": close,
                    "pivots": len(pivots),
                    "trendlines": len(detection.active_lines),
                    "zones": len(zones),
                    "zone_details": [
                        {"side": z.side, "center": z.price_center, "touches": z.touches, "strength": z.strength}
                        for z in zones
                    ],
                    "signals": [
                        {"type": s.signal_type, "dir": s.direction, "entry": s.entry_price,
                         "sl": s.stop_price, "tp": s.tp_price, "rr": round(s.risk_reward, 2)}
                        for s in all_sigs
                    ],
                    "signal_count": len(all_sigs),
                }
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # Print signals to console
                if all_sigs:
                    for s in all_sigs:
                        print(f"  [{now[:19]}] {tf:4s} {s.signal_type:25s} entry=${s.entry_price:.4f} SL=${s.stop_price:.4f} TP=${s.tp_price:.4f} RR={s.risk_reward:.1f}", flush=True)

            except Exception as e:
                log.write(json.dumps({"time": now, "cycle": cycle, "tf": tf, "error": str(e)}) + "\n")

        # Status line
        print(f"[Cycle {cycle:4d}] {now[:19]} | {SYMBOL} | {total_signals} signals across {len(TIMEFRAMES)} TFs", flush=True)

        await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    try:
        asyncio.run(scan_all_timeframes())
    except KeyboardInterrupt:
        print("\nStopped.")
