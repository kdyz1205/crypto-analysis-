# Screenshot and Training Notes

Date: 2026-04-17

This note captures what I learned from the trading-related screenshots in
`C:\Users\alexl\Pictures\Screenshots` and how it should map into the
trendline model/order logic. I skipped unrelated screenshots such as Claude API
errors, Copilot path errors, and non-trading UI.

## Training Run

Command used for the full available pattern set:

```powershell
python scripts\train_trendline_quality.py --allow-missing --epochs 8 --batch-size 8192
```

Training used:

- Rows: 253251
- Features: 24
- Device: CUDA
- Torch: 2.6.0+cu124
- GPU available: True
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT, HYPEUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, PEPEUSDT, SUIUSDT
- Timeframes: 4h, 1h, 15m, 5m
- Missing files: LINKUSDT_5m, SUIUSDT_5m
- Checkpoint: `checkpoints/trendline_quality/best_model.pt`

Walk-forward results:

| Fold | Train Rows | Val Rows | AUC | Accuracy | RMSE |
|---:|---:|---:|---:|---:|---:|
| 1 | 123308 | 25751 | 0.9017 | 0.8234 | 4.4368 |
| 2 | 149059 | 26214 | 0.9140 | 0.8444 | 4.3071 |
| 3 | 175273 | 26014 | 0.9084 | 0.8680 | 5.4318 |
| 4 | 201287 | 26362 | 0.9077 | 0.8567 | 4.4503 |
| 5 | 227649 | 25602 | 0.9294 | 0.8669 | 3.4353 |

Best fold: 5, AUC: 0.9293599910603298.

## Screenshot Evidence

### FILUSDT 5m support long

Files:

- `trendline1.png`
- `Screenshot 2026-04-16 031547.png`

Observed behavior:

- Yellow ascending support line is drawn through prior lows and projected
  forward.
- The order is placed at the projected retest area, not after chasing the
  bounce.
- TradingView long-position tool shows a tight stop below the line and a much
  larger target above. The visible risk/reward label is about 29.4R.

Learned rule:

- For support, draw from obvious swing lows, wait for first retest, place a
  long entry near the projected line, put SL just below the line plus buffer,
  and TP above using at least the configured RR target.

### FILUSDT 5m resistance short

Files:

- `112233.png`
- `Screenshot 2026-04-16 031129.png`
- `Screenshot 2026-04-16 030941.png`

Observed behavior:

- Yellow descending resistance line is drawn through prior highs and projected
  into the current bar area.
- Short-position tool is placed around the projected retest.
- Stop is above the resistance line; target is lower. Visible RR examples are
  around 12R to 13R.

Learned rule:

- For resistance, draw from obvious swing highs, wait for retest from below,
  use short entry near or slightly below projected line, put SL above the line,
  and TP below.

### BTCUSDT 4h macro resistance short

Files:

- `Screenshot 2026-04-15 153402.png`
- `Screenshot 2026-04-15 153413.png`

Observed behavior:

- Macro descending resistance is anchored on major swing highs.
- A lower rising support line forms a wedge/channel context.
- Short-position tool is placed at the resistance retest, with SL above the
  macro resistance and TP lower inside the structure.

Learned rule:

- 4h macro lines are important filters. A lower-timeframe entry should respect
  the 4h structure instead of treating every local touch equally.

### HYPEUSDT 4h manual line workflow

Files:

- `Screenshot 2026-04-14 004534.png`
- `Screenshot 2026-04-14 135836.png`
- `Screenshot 2026-04-14 134928.png`
- `Screenshot 2026-04-14 121913.png`
- `Screenshot 2026-04-14 121905.png`

Observed behavior:

- The Trading OS UI keeps a right-side list of manual lines under "my drawn
  lines".
- Lines have start price, end price, timestamp, support/resistance type, and
  a delete/select action.
- Some screenshots show selected line metadata such as support/resistance,
  confirmed points, touch count, stability, and bounce probability.
- Blue anchor points on the chart show that the useful line is defined by
  explicit swing anchors, not by an unconstrained regression fit.

Learned rule:

- Manual user drawings are high-value labels. They should be captured with
  anchors, projected price, touch count, reaction strength, stability, and
  whether the user chose to place an order.

### Trading OS condition orders

Files:

- `Screenshot 2026-04-15 203104.png`
- `Screenshot 2026-04-14 134928.png`

Observed behavior:

- Condition cards include symbol, timeframe, bounce direction, long/short,
  status, order mode, line status, RR, Bitget id, log button, and delete button.
- Many cards show `mkt -- line -- untested` and `RR 2`.

Learned rule:

- The system needs line-linked order state: symbol/timeframe/side, line id,
  tested vs untested, Bitget id, RR, and cancellation status.
- Duplicate protection must key off both open positions and existing active
  line-linked plan orders.

### Bitget TP/SL and position evidence

Files:

- `Screenshot 2026-04-15 214106.png`
- `Screenshot 2026-04-16 191201.png`

Observed values:

- HYPEUSDT TP/SL rows show qty 6.98.
- SL row: market partial stop loss, trigger `last price <= 44.82 USDT`,
  execution price market, waiting.
- TP row: market partial take profit, trigger `last price >= 45.359 USDT`,
  execution price market, waiting.
- LINKUSDT mobile position screenshot shows an active short around entry 9.601,
  mark/current around 9.446, with whole-position TP/SL editing visible.

Learned rule:

- Entry plans and post-fill TP/SL are different order families.
- Entry should be a line-linked plan entry. After fill, TP/SL should attach to
  the actual position and be verified from the exchange order list.
- Cancelling or replacing TP/SL must identify the right `planType`.

## Learned Policy

Trendline quality:

- Use swing pivots, not random candle bodies.
- Prefer at least two clear anchors, and strongly prefer three or more touches
  or reactions.
- A line is most useful before or at first retest. It is lower value after a
  large move away from the line.
- Support lines connect lows and are traded long on retest from above.
- Resistance lines connect highs and are traded short on retest from below.
- Multi-timeframe context matters: 4h macro line first, 5m/15m timing second.
- Do not place a fresh line order if the same symbol already has an open
  position or an active unfilled plan order.

Order placement:

- Long support retest: entry near projected support, SL just below line plus
  buffer, TP above by configured RR or structure target.
- Short resistance retest: entry near projected resistance, SL just above line
  plus buffer, TP below by configured RR or structure target.
- Entry order should preserve the chosen trigger/limit price. It should not
  silently become a market entry.
- Post-fill SL/TP should be position-attached and verified against exchange
  state.

## Limits of Screenshot Learning

The screenshots are useful for learning user intent and rule preferences. They
are not enough by themselves for exact numeric PyTorch labels unless the image
pixels are mapped back to chart time/price coordinates.

Best path for quantitative learning:

1. Capture manual line events from Trading OS into `data/user_drawings_ml.jsonl`.
2. Store exact anchor timestamps/prices, support/resistance type, selected or
   rejected status, and whether the user clicked order placement.
3. Join those records with OHLCV features at draw time.
4. Train a preference model that scores algorithm lines against user-drawn
   lines.

