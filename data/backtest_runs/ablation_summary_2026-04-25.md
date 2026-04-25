# Ablation Report — 2026-04-25

3-variant comparison on **BTCUSDT 5m**, start_bar=1000, predict_every=200,
hold_bars=20, min_confidence=0.55. Same training pool (3000 max records,
50x manual oversample, 2 epochs, d_model=64) — only the trendline-stream
flags differ.

| Variant | Streams | val_loss | trades | hit% | avg/trade | cum | Sharpe | win_RR | L/S | stop/exp |
|---|---|---|---|---|---|---|---|---|---|---|
| rule_only | rule | 11.42 | 47 | 51.1% | -0.085% | **-4.01%** | -1.07 | 0.57 | 39/8 | 13/34 |
| raw_only | raw 36-d | 11.43 | 48 | 33.3% | -0.059% | **-2.82%** | -0.67 | 1.46 | 23/25 | 10/38 |
| **all** | rule+learned+raw | 12.02 | 52 | 51.9% | +0.067% | **+3.51%** | **+0.80** | 1.33 | 7/45 | 0/52 |

## Findings

1. **Multi-stream wins** — only the `all` variant produced a positive cumulative
   return on this slice. Rule-only and raw-only both lost money. This is
   the strongest argument for the 3-stream design.

2. **Stream-specific direction bias**:
   - rule_only: heavily LONG-biased (39 long / 8 short) — learns to predict
     "support bounce" because that's the dominant pattern in `data/patterns/*.jsonl`.
   - raw_only: balanced (23/25) but hit rate collapses to 33% — without the
     coarse rule axes the raw 36-d features can't separate role/direction.
   - all: heavily SHORT-biased (7/45) — the learned VQ-VAE stream apparently
     contributed a "break-of-resistance" signal that rule and raw missed.

3. **val_loss is a poor proxy for backtest PnL** — `all` has the WORST val_loss
   (12.02) but the BEST cumulative return. Loss measures next-token prediction
   accuracy; PnL depends on directional bet quality + stop-buffer calibration.

4. **No stops triggered for `all`** (0/52 vs 10-13 for the other two). Either
   the buffer suggestions are conservative or the LONG bets in this period
   never hit stops. Inspect at scale.

## Caveats

- 47-52 trades is a tiny sample. Confidence intervals on hit_rate are wide.
- Single symbol / single timeframe. Need multi-symbol replication.
- Model is 6M params trained on 4k examples. Way undersized.
- BTCUSDT 5m over the test period was likely choppy — directional bias matters less.

## Next test cycle

After scaling per ROADMAP T1:
1. 50M+ trendlines (sweep_sr_params_full.py)
2. Train d_model=256, 50M params, 5 epochs
3. Re-run this 3-variant comparison on a 6-month BTCUSDT slice
4. Add ETHUSDT / SOLUSDT for symbol generalisation check

## Reproducible commands

```bash
# Train + backtest each variant (same slice):
python -m trendline_tokenizer.backtest.run_ablation \
  --symbols BTCUSDT ETHUSDT HYPEUSDT \
  --timeframes 5m 15m 1h \
  --max-records 3000 --epochs 2 --d-model 64 \
  --variants rule_only raw_only all \
  --backtest-symbol BTCUSDT --backtest-timeframe 5m \
  --start-bar 1000 --predict-every 200 \
  --hold-bars 20 --min-confidence 0.55
```

## Artifacts

- `checkpoints/fusion/abl-rule_only-1777128387/`
- `checkpoints/fusion/abl-raw_only-1777128494/`
- `checkpoints/fusion/abl-all-1777129011/`
- `data/backtest_runs/BTCUSDT_5m_1777129179/` (rule_only)
- `data/backtest_runs/BTCUSDT_5m_1777129247/` (raw_only)
- `data/backtest_runs/ablation_1777129084.txt` (raw report)
