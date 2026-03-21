# Crypto-Analysis Autoresearch

Autonomous strategy optimization for the crypto-analysis platform.
Inspired by Karpathy's autoresearch: modify → verify → keep/discard → repeat.

## Setup

The repo is a TradingView-style crypto technical analysis platform with:
- `server/backtest_service.py` — MFI/MA strategy + backtester (the file you optimize)
- `server/data_service.py` — OHLCV data from OKX/Binance (DO NOT MODIFY)
- `sr_patterns.py` — Pattern detection (DO NOT MODIFY in this loop)
- `autoresearch/autoresearch.py` — The autonomous experiment runner

## What You CAN Do

Modify the **strategy parameters and logic** in `server/backtest_service.py`:
- BacktestParams defaults (mfi_period, ma_fast, ema_span, ma_slow, atr_period, atr_sl_mult, bb_period, bb_std)
- Entry conditions (the stacking rule: Price > MFI > MA_fast > EMA > MA_slow)
- Exit logic (stop-loss, take-profit, trailing stops)
- Add new indicators (RSI, MACD, Bollinger width, etc.)
- Add new exit strategies (time-based, volatility-based, etc.)

## What You CANNOT Do

- Modify `server/data_service.py` (fixed data pipeline)
- Modify the scoring function in `autoresearch/autoresearch.py`
- Install new packages beyond what's in `requirements.txt`
- Look ahead in backtest (no future data leakage)

## The Goal

**Maximize the composite score** (higher is better):

```
score = sharpe × √(trade_factor) − drawdown_penalty

where:
  sharpe = mean(pnl%) / std(pnl%)
  trade_factor = min(num_trades / 20, 1.0)
  drawdown_penalty = max(0, max_drawdown% − 10) × 0.05
```

Hard cutoffs (score = -999):
- < 3 trades
- max_drawdown > 50%

## Experiment Loop

LOOP FOREVER:

1. Read current state of `server/backtest_service.py`
2. Propose a single focused change (parameter tweak, new indicator, exit rule, etc.)
3. Implement the change
4. git commit
5. Run: `python autoresearch/autoresearch.py > autoresearch/run.log 2>&1`
6. Read results: `grep "^score:\|^sharpe:\|^total_pnl:" autoresearch/run.log`
7. If score improved → keep. If score equal/worse → git reset to previous commit.
8. Log results to `autoresearch/results.tsv`
9. Repeat

## Research Priority

High probability:
- ATR stop-loss multiplier tuning (currently 1.0, try 1.5–3.0)
- Bollinger band width/std tuning
- Adding RSI filter (avoid overbought entries)
- Adding MACD confirmation signal
- Multi-timeframe momentum confirmation

Medium probability:
- Trailing stop instead of fixed SL
- Time-based exit (max hold period)
- Volatility-adjusted position sizing
- EMA crossover exit signals

Exploratory:
- Ensemble voting (multiple signals must agree)
- Regime detection (trending vs ranging market)
- Volume profile confirmation
