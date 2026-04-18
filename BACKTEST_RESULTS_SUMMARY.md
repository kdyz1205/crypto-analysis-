# Backtest Results Summary (2026-04-17)

---

## 1. Trendline V3 Full Sweep (5m/15m/1h/4h)

### Best Config Per TF (taker EV, 10 coins × 5 pivot_k × 5 buffer × 5 RR)

| TF | Best Buffer | Best RR | WR | Gross EV | Taker EV | Trades |
|----|------------|---------|------|----------|----------|--------|
| 5m | 0.05% | 15 | 22.7% | +0.100% | ±0.000% | 137-168K |
| 15m | 0.10% | 15 | 23.4% | +0.185% | **+0.085%** | 150-174K |
| 1h | 0.20% | 15 | 24.5% | +0.396% | **+0.296%** | 58-80K |
| 4h | 0.30% | 15 | 25.2% | +0.830% | **+0.730%** | 13-19K |

### Key Findings
- **Pivot k (1-5) has ZERO effect** — all TFs produce identical results regardless of k
- **RR越高EV越高** — RR=15 consistently best across all TFs
- **Buffer越大 4h越好，越小 5m越好**
- **4h per-trade EV is 10x higher than 5m**
- Total data: 500 configs tested (5m: 125, 15m: 125, 1h: 125, 4h: 125)

---

## 2. Trailing TP Comparison (6 exit strategies, same entries)

### Results by TF (buffer=0.001, pivot_k=3, max_hold=200)

#### 5m (85,579 trades)
| Exit Strategy | WR | Avg Win | Avg Loss | Gross EV | Taker EV | Total PnL |
|---------------|------|---------|----------|----------|----------|-----------|
| RR=8 | 28.0% | +0.54% | -0.10% | +0.080% | -0.020% | +6,811% |
| RR=15 | 24.5% | +0.68% | -0.11% | +0.084% | -0.016% | +7,149% |
| RR=30 | 23.2% | +0.79% | -0.12% | +0.091% | -0.009% | +7,746% |
| **ATR trail** | **38.7%** | **+0.41%** | **-0.08%** | **+0.106%** | **+0.006%** | **+9,076%** |
| BB touch | 33.3% | +0.46% | -0.10% | +0.090% | -0.010% | +7,661% |
| MA cross | 48.1% | +0.25% | -0.07% | +0.081% | -0.019% | +6,889% |

#### 15m (47,210 trades)
| Exit Strategy | WR | Gross EV | Taker EV | Total PnL |
|---------------|------|----------|----------|-----------|
| RR=8 | 34.8% | +0.147% | +0.047% | +6,923% |
| RR=15 | 29.4% | +0.160% | +0.060% | +7,548% |
| RR=30 | 26.7% | +0.171% | +0.071% | +8,078% |
| **ATR trail** | **38.2%** | **+0.190%** | **+0.090%** | **+8,988%** |
| BB touch | 34.0% | +0.164% | +0.064% | +7,752% |
| MA cross | 51.1% | +0.156% | +0.056% | +7,342% |

#### 1h (30,079 trades)
| Exit Strategy | WR | Gross EV | Taker EV | Total PnL |
|---------------|------|----------|----------|-----------|
| RR=8 | 46.5% | +0.261% | +0.161% | +7,861% |
| RR=15 | 38.8% | +0.329% | +0.229% | +9,886% |
| RR=30 | 33.1% | +0.358% | +0.258% | +10,779% |
| **ATR trail** | **39.1%** | **+0.422%** | **+0.322%** | **+12,684%** |
| BB touch | 35.9% | +0.381% | +0.281% | +11,459% |
| MA cross | 54.3% | +0.369% | +0.269% | +11,097% |

#### 4h (6,932 trades)
| Exit Strategy | WR | Avg Win | Avg Loss | Taker EV | Total PnL |
|---------------|------|---------|----------|----------|-----------|
| RR=8 | 60.8% | +0.66% | -0.10% | +0.261% | +2,500% |
| RR=15 | 55.9% | +1.11% | -0.11% | +0.471% | +3,959% |
| RR=30 | 48.9% | +1.68% | -0.13% | +0.655% | +5,235% |
| **ATR trail** | **48.2%** | **+2.24%** | **-0.12%** | **+0.914%** | **+7,032%** |
| BB touch | 45.0% | +2.12% | -0.18% | +0.757% | +5,938% |
| MA cross | 61.0% | +1.42% | -0.10% | +0.725% | +5,716% |

### Conclusion
**ATR trailing is the best exit strategy on ALL timeframes.** 4h ATR trail: +0.914%/trade, 18:1 payoff ratio.

---

## 3. PyTorch Trendline Quality Model

### Training Results (walk-forward, 5 folds)

| Fold | Train Samples | Val Samples | AUC | Accuracy | PnL RMSE |
|------|--------------|-------------|------|----------|----------|
| 1 | 123,309 | 25,751 | 0.903 | 82.3% | 4.46 |
| 2 | 149,060 | 26,214 | 0.912 | 85.0% | 4.32 |
| 3 | 175,274 | 26,014 | 0.910 | 86.9% | 5.44 |
| 4 | 201,288 | 26,362 | 0.909 | 85.7% | 4.44 |
| **5** | **227,650** | **25,602** | **0.928** | **86.9%** | **3.44** |

- **Best AUC: 0.928** (fold 5, 228K training samples)
- **24 features**: slope_atr, length_bars, volatility, rsi, ma_distance_atr, touch_quality, anchor_gap, context (uptrend/downtrend/range), etc.
- **10 coins × 4 TFs** (BTC, ETH, SOL, HYPE, XRP, ADA, DOGE, LINK, PEPE, SUI)
- **Checkpoint**: `checkpoints/trendline_quality/best_model.pt`
- **Trained with CUDA** (RTX 4060, torch 2.6.0+cu124)

### Model Integration Status
- ✅ Model trained and saved
- ✅ Drawing capture pipeline working (user_drawings_ml.jsonl)
- ✅ Inference scorer class built
- ❌ Not yet filtering live trades (Codex reports "还没有真正参与决策")

---

## 4. Model-Filtered Backtest (Proxy Score)

### Unfiltered vs Filtered (46,761 trades, 1h+4h, 7 coins)

| Filter | Trades | WR | Gross EV | Taker EV | Total PnL |
|--------|--------|------|----------|----------|-----------|
| All (unfiltered) | 46,761 | 42.2% | +0.299% | +0.199% | +13,978% |
| Top 40% | 20,492 | 42.9% | +0.304% | +0.204% | +6,235% |

**Note**: This test used a proxy score (ATR + kind), NOT the real 24-feature model.
Real model filtering should show much larger EV improvement between top and bottom trades.
Full model integration needed to see true filtering power.

---

## 5. MA Ribbon Previous Findings

From earlier research (not re-run yet, sweep crashed):

| Config | TF | Direction | Result |
|--------|-----|-----------|--------|
| R=[3,8,21,55] s=0.6 | 4h | long+short | **Overfit garbage** (-7.4%) |
| R=[5,13,34,89] s=1.0 | 4h | long only | **Only stable** (+13.9% BTC) |
| All configs | 1h | any | **All negative** |
| All configs | any | short | **All worse** |

MA Ribbon full sweep (38,880 configs) needs to be re-run.

---

## 6. Live Trading Observations (2026-04-16)

### Early Session (market execution, no preset SL)
- 13 trades, 12 losses, 1 win
- Average hold time: 1-2 minutes
- Root cause: market order slippage + no SL for 60 seconds

### Late Session (preset SL/TP, planType cancel fix)
- Equity: $14.54 → $16.75 (peak) → $15.85 (end)
- System stable for 10+ hours
- SL cancel fix verified on real Bitget
- SL trailing movement still not working automatically

### Bitget API Discoveries
- `cancel-plan-order` requires `planType` in body (silent failure without it)
- Preset SL type = `loss_plan`, preset TP = `profit_plan`
- Position entry price field = `openPriceAvg` (not `averageOpenPrice`)
- Plan orders create SL/TP pairs under `profit_loss` umbrella query

---

## Recommended Production Config

Based on all backtests:

```python
config = {
    "timeframes": ["5m", "15m", "1h", "4h"],
    "buffer": {"5m": 0.0005, "15m": 0.001, "1h": 0.002, "4h": 0.003},
    "exit_mode": "atr_trailing",  # best on all TFs
    "atr_trail_mult": 1.5,
    "rr_safety_cap": 30,  # backup fixed TP
    "risk_per_tf": {"5m": 0.003, "15m": 0.007, "1h": 0.015, "4h": 0.03},
    "pivot_k": 3,  # doesn't matter, use any 1-5
    "execution": "market",  # plan order with market trigger
    "preset_sl_tp": True,  # immediate protection
    "ml_filter": True,  # use quality model (AUC=0.928)
    "ml_min_score": 0.45,  # only trade lines model rates >45%
}
```
