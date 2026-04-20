"""Probe: run _check_trendline_signal on ARBUSDT with full runner context,
gate enabled, see why runner isn't picking it up."""
from __future__ import annotations
import sys, asyncio
sys.path.insert(0, '.')


async def main():
    from server.strategy.mar_bb_runner import _fetch_bars, _check_trendline_signal, DEFAULT_RUNNER_CFG
    cfg_gate_on = {**DEFAULT_RUNNER_CFG}
    cfg_gate_off = {**DEFAULT_RUNNER_CFG, 'model_gate': {**DEFAULT_RUNNER_CFG['model_gate'], 'enabled': False}}
    for tf in ['15m', '1h']:
        bars = await _fetch_bars('ARBUSDT', tf, 7 if tf == '15m' else 21)
        if not bars:
            print(f'ARB {tf}: no bars')
            continue
        p_on = _check_trendline_signal('ARBUSDT', tf, bars, cfg_gate_on)
        p_off = _check_trendline_signal('ARBUSDT', tf, bars, cfg_gate_off)
        print(f'ARBUSDT {tf}:')
        print(f'  gate ON : {p_on}')
        print(f'  gate OFF: {p_off}')


asyncio.run(main())
