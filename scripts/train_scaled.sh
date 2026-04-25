#!/usr/bin/env bash
# Train the fusion model on the post-sweep scaled pool.
#
# Run AFTER scale_patterns_sweep.py has finished writing data/patterns_sweep/.
# Trains on a mix of the scaled auto pool + 50x manual gold + outcomes.
#
# d_model=128 (vs 64-96 in earlier smoke runs) + 4 layers + 5 epochs is the
# first run that has any chance of producing a model with predictive value.
set -e

PY="C:/Users/alexl/AppData/Local/Programs/Python/Python312/python.exe"

cd "$(dirname "$0")/.."

echo "[train_scaled] sweep dir contents:"
ls -la data/patterns_sweep/ 2>/dev/null | head -5
echo ""

"$PY" -u -m trendline_tokenizer.training.train_fusion \
    --symbols ADAUSDT BTCUSDT DOGEUSDT ETHUSDT HYPEUSDT LINKUSDT PEPEUSDT SOLUSDT SUIUSDT TAOUSDT XRPUSDT \
    --timeframes 5m 15m 1h 4h 1d \
    --max-records 30000 \
    --epochs 5 \
    --batch-size 32 \
    --d-model 128 \
    --n-layers-price 4 \
    --n-layers-token 3 \
    --n-layers-fusion 2 \
    --manual-oversample 50 \
    "$@" \
    | tee data/logs/train_scaled_$(date +%Y%m%d_%H%M%S).log
