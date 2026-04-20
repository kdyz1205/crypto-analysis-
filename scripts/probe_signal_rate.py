"""Probe: how often does trendline generate_signals fire across the window?
Run to see if 0-signal-today is a model-gate issue, a detector issue, or just
last-bar-rarity (signal is WINDOW-wide but _check_trendline_signal only takes
the final bar)."""
from __future__ import annotations
import sys, asyncio
sys.path.insert(0, '.')
import numpy as np


async def main():
    from server.strategy.mar_bb_runner import _fetch_bars
    from research.trendline_strategy import generate_signals, DEFAULT_CONFIG
    symbols = ['BTCUSDT','ETHUSDT','SOLUSDT','HYPEUSDT','DOGEUSDT',
               'BNBUSDT','XRPUSDT','ADAUSDT','LINKUSDT','ARBUSDT']
    for sym in symbols:
        for tf in ['15m', '1h', '4h']:
            days = 7 if tf == '15m' else (21 if tf == '1h' else 84)
            try:
                bars = await _fetch_bars(sym, tf, days)
                if not bars:
                    print(f'  {sym} {tf}: no bars')
                    continue
                sigs, entries, sls, tps, lines = generate_signals(
                    bars['o'], bars['h'], bars['l'], bars['c'], bars['v'], DEFAULT_CONFIG
                )
                total = int(np.sum(np.abs(sigs) > 0))
                longs = int(np.sum(sigs == 1))
                shorts = int(np.sum(sigs == -1))
                last_sig = int(sigs[-1]) if len(sigs) else 0
                n = len(bars['c'])
                print(f'  {sym} {tf}: N={n} total={total} ({longs}L/{shorts}S) last_bar={last_sig}')
            except Exception as e:
                print(f'  {sym} {tf} ERR: {type(e).__name__}: {str(e)[:80]}')


asyncio.run(main())
