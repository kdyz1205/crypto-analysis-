"""Test approach_pct values 1.0 / 2.0 / 3.0 on 4h — confirm wider is the fix."""
from __future__ import annotations
import sys, asyncio
sys.path.insert(0, '.')
import numpy as np


async def main():
    from server.strategy.mar_bb_runner import _fetch_bars
    from research.trendline_strategy import generate_signals, DEFAULT_CONFIG
    for sym in ['BTCUSDT','ETHUSDT','SOLUSDT']:
        for tf in ['1h', '4h']:
            days = 21 if tf == '1h' else 84
            bars = await _fetch_bars(sym, tf, days)
            if not bars:
                print(f'{sym} {tf}: no bars'); continue
            for approach in [1.0, 2.0, 3.0, 5.0]:
                cfg = {**DEFAULT_CONFIG, 'approach_pct': approach}
                sigs, _, _, _, _ = generate_signals(
                    bars['o'], bars['h'], bars['l'], bars['c'], bars['v'], cfg
                )
                total = int(np.sum(np.abs(sigs) > 0))
                print(f'  {sym} {tf} approach={approach}%: sigs={total}')
            print()


asyncio.run(main())
