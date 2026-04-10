# Validated Trendline Runtime Preset

This document records the current trendline strategy preset that has been
verified through the real `strategy -> paper execution -> runtime` chain.

## Scope

This is **not** a claim that the current auto trendline strategy is profitable
on every symbol or timeframe.

This is the narrower statement that the current implementation has a
backtest-positive preset on a specific 1h high-beta market cluster.

After the Phase A touch / invalidation tightening pass, `ENSOUSDT` no longer
stays inside the currently validated cluster and should not be treated as part
of the passing preset scope.

## Preset

- Trigger modes: `pre_limit`
- Lookback bars: `80`
- Strategy window bars: `100`
- Timeframe: `1h`
- Execution path: `build_latest_snapshot -> paper execution engine -> runtime tick`

## Verified Market Cluster

- `HYPEUSDT`
- `RIVERUSDT`

## Verification Command

```powershell
python scripts/verify_trendline_runtime_preset.py
```

The verification script currently checks that the combined preset run:

- produces at least 6 closed trades
- finishes with positive total PnL
- finishes with profit factor greater than `1.5`

## Latest Observed Result

Latest passing run on this branch:

- trade count: `11`
- total pnl: `301.6216`
- return pct: `1.5081`
- win rate pct: `63.64`
- profit factor: `3.4909`
- max drawdown pct: `0.5991`

## Important Boundary

Broader 1h baskets that include symbols such as `XRPUSDT`, `SOLUSDT`, or
`1INCHUSDT` can still degrade sharply under the same logic.

So the current productized conclusion is:

- the pipeline is validated
- the preset is validated
- the validated scope is symbol-cluster specific

Do not treat this preset as a universal cross-market trendline system without
additional symbol selection or further regime filtering.
