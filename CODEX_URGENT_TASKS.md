# URGENT TASKS FOR CODEX

## Critical Discovery (2026-04-16 22:00)

The cancel-plan-order API requires `planType` in the body to actually work.
Without it, returns success but `successList` is empty (silent failure).

### Proven working cancel:
```python
# This WORKS:
await adapter._bitget_request("POST", "/api/v2/mix/order/cancel-plan-order",
    mode="live", body={
        "symbol": sym, "productType": "USDT-FUTURES",
        "orderId": oid,
        "planType": "loss_plan"  # <-- MUST include this!
    })
# Response: successList: [{orderId}] ✅

# This FAILS SILENTLY:
await adapter._bitget_request("POST", "/api/v2/mix/order/cancel-plan-order",
    mode="live", body={
        "symbol": sym, "productType": "USDT-FUTURES",
        "orderId": oid,
        # No planType → successList: [] (nothing cancelled)
    })
```

### Bitget order sub-types (under "profit_loss" umbrella query):
- `pos_loss` = 仓位止损 (position SL, from place-tpsl-order)
- `pos_profit` = 仓位止盈 (position TP, from place-tpsl-order)  
- `loss_plan` = 部分止损 (preset SL from plan order trigger)
- `profit_plan` = 部分止盈 (preset TP from plan order trigger)

### Position entry price field: `openPriceAvg` (NOT `averageOpenPrice`)

---

## Task 1: Fix trailing SL to ACTUALLY MOVE at bar boundaries

The trailing SL has never successfully moved in live. Problems:
1. `bars_since` calculation uses `opened_ts` which is wrong after restart
2. SL projection value doesn't match actual line position
3. The update only runs once then stops

Requirements:
- Every 5m bar (for 5m positions), cancel old SL + place new SL at updated line projection
- Every 1h bar (for 1h positions), same
- Verify on real Bitget that trigger price actually changed
- Print: `[trailing] MOVED SL {sym} {old_sl} -> {new_sl} bars={bars}`

## Task 2: Stop placing plan orders on symbols that already have positions

Current bug: MSFT has a position AND a new plan order (normal_plan trigger=421.18).
Fix: Before placing any new plan order, check if the symbol already has an open position.

## Task 3: Complete PyTorch training

Run `scripts/train_trendline_quality.py` to completion:
- Collect data from all 4 TFs (4h, 1h, 15m, 5m) × 10 coins
- Train 5-fold walk-forward
- Save checkpoint to `checkpoints/trendline_quality/best_model.pt`
- Report AUC per fold

## Task 4: Install CUDA PyTorch

The machine has RTX 4060 Laptop GPU. Current PyTorch is CPU-only.
Install: `pip install torch --index-url https://download.pytorch.org/whl/cu124`
Verify: `torch.cuda.is_available()` returns True

## How to verify Task 1 works:
```python
# 1. Check position exists
# 2. Note current SL trigger price on Bitget
# 3. Wait for bar boundary (e.g., 5m = next :x0 or :x5)
# 4. Check SL trigger price changed
# 5. Print old vs new
```
