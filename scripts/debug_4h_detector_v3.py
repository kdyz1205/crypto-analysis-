"""Try relaxing max_penetrations / max_post_penetrations on 4h."""
from __future__ import annotations
import sys, asyncio
sys.path.insert(0, '.')
import numpy as np


async def main():
    from server.strategy.mar_bb_runner import _fetch_bars
    from research.trendline_strategy import generate_signals, build_trendlines, DEFAULT_CONFIG
    for sym in ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT']:
        bars = await _fetch_bars(sym, '4h', 84)
        if not bars:
            print(f'{sym} 4h: no bars'); continue
        h, l, c = bars['h'], bars['l'], bars['c']
        for variant in [
            {'name': 'default', 'swing_lookback':10, 'max_penetrations':2, 'max_post_penetrations':0},
            {'name': 'relax_post_1', 'swing_lookback':10, 'max_penetrations':2, 'max_post_penetrations':1},
            {'name': 'relax_post_3', 'swing_lookback':10, 'max_penetrations':2, 'max_post_penetrations':3},
            {'name': 'wider_swing', 'swing_lookback':5, 'max_penetrations':2, 'max_post_penetrations':1},
            {'name': 'all_loose', 'swing_lookback':5, 'max_penetrations':5, 'max_post_penetrations':5},
        ]:
            cfg = {**DEFAULT_CONFIG, **{k:v for k,v in variant.items() if k != 'name'}}
            lines = build_trendlines(h, l, c, cfg)
            sigs, _, _, _, _ = generate_signals(bars['o'], h, l, c, bars['v'], cfg)
            total = int(np.sum(np.abs(sigs) > 0))
            print(f'{sym} 4h {variant["name"]:<12}: lines={len(lines):3d}  sigs={total}')
        print()


asyncio.run(main())
