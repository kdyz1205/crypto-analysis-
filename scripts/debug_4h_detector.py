"""Why is 4h generate_signals returning 0 across 10 coins × 500 bars?
Trace: how many swings? how many lines? how many approach-near? how many fill?"""
from __future__ import annotations
import sys, asyncio
sys.path.insert(0, '.')
import numpy as np


async def main():
    from server.strategy.mar_bb_runner import _fetch_bars
    from research.trendline_strategy import (
        generate_signals, build_trendlines, find_swing_highs, find_swing_lows, DEFAULT_CONFIG
    )
    cfg = DEFAULT_CONFIG
    for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT','DOGEUSDT']:
        for tf in ['15m','1h','4h']:
            days = 7 if tf=='15m' else (21 if tf=='1h' else 84)
            bars = await _fetch_bars(sym, tf, days)
            if not bars:
                print(f'{sym} {tf}: no bars'); continue
            h, l, c = bars['h'], bars['l'], bars['c']
            n = len(c)
            sh = find_swing_highs(h, cfg['swing_lookback'])
            sl = find_swing_lows(l, cfg['swing_lookback'])
            lines = build_trendlines(h, l, c, cfg)
            sigs, _, _, _, _ = generate_signals(bars['o'], h, l, c, bars['v'], cfg)
            total_sigs = int(np.sum(np.abs(sigs) > 0))
            # Signal counts by bar — which proportion of active lines get close to price at end?
            print(f'{sym} {tf}: N={n}  swing_H={len(sh)}  swing_L={len(sl)}  lines={len(lines)}  sigs={total_sigs}')


asyncio.run(main())
