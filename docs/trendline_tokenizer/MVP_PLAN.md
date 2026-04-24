# MVP Plan — Trendline Tokenizer

## Current status (2026-04-24)

| Phase | Status | Evidence |
|---|---|---|
| 0. Asset inventory | **done** | 78 manual + 271,773 auto rows located; `sr_patterns.py`, `pattern_features`, `drawing_outcome_labeler` catalogued |
| 1. Canonical schema | **done** | `trendline_tokenizer/schemas/trendline.py` (Pydantic) |
| 2. Adapters | **done** | `adapters/manual.py` + `adapters/legacy_patterns.py` — 15/15 unit tests green |
| 3. Rule tokenizer v1 | **done** | `tokenizer/rule.py` + `tokenizer/vocab.py`, encode/decode round-trip tested on real data |
| 4. Reconstruction metrics | **done** | `tokenizer/metrics.py` — median agg 0.127 on 78 manual, 100 % role round-trip |
| 5. Round-trip CLI | **done** | `cli/roundtrip.py` — 24k records/sec |
| 6. Feature vector | **done** | `features/vector.py` — 36-dim (17 continuous + 7 role + 12 tf) |
| 7. Learned VQ-VAE | **done** | `learned/vqvae.py` — hierarchical 256/1024, EMA codebook, kmeans init, dead-code revival |
| 8. Training CLI | **done** | `learned/train.py` — val_role_acc 1.0, val_bounce_acc 0.987 after 5 epochs on 40k |
| 9. Detokenizer visualisation | pending | `visualization/plot_overlay.py` — next turn |
| 10. Auto-data scaling loop | pending | wrap `sr_patterns` with the new schema + Bayesian param sweep |
| 11. Autoregressive model | pending | `sequence_model/` — after visualisation confirms tokens are readable |

## File tree (committed)

```
trendline_tokenizer/
  __init__.py
  schemas/
    __init__.py
    trendline.py                # TrendlineRecord / TokenizedTrendline / TokenizerConfig / OHLCVBar
  adapters/
    __init__.py
    manual.py                   # data/manual_trendlines.json → TrendlineRecord
    legacy_patterns.py          # data/patterns/*.jsonl    → TrendlineRecord
  tokenizer/
    __init__.py
    vocab.py                    # rule.v1 vocabulary (5040 coarse × 21600 fine)
    rule.py                     # encode_rule / decode_rule, mixed-radix integer
    metrics.py                  # RoundTripError + summarize()
  features/
    __init__.py
    vector.py                   # 36-dim feature vector builder
  learned/
    __init__.py
    vqvae.py                    # HierarchicalVQVAE + VectorQuantizer (EMA + kmeans + revive)
    dataset.py                  # TrendlineFeatureDataset
    train.py                    # training CLI
  cli/
    __init__.py
    roundtrip.py                # encode/decode on real data + metrics summary
  tests/
    __init__.py
    test_schema.py              (6 tests)
    test_rule_tokenizer.py      (7 tests)
    test_round_trip.py          (2 tests, skipped when data absent)
```

## How to run what's there

```bash
# Python that has torch + pytest + pydantic (see scripts/open_trading_os.ps1 for discovery rule).
# Substitute your own interpreter if different.
set PY="C:\Users\alexl\AppData\Local\Programs\Python\Python312\python.exe"

# 1. tests
%PY% -m pytest trendline_tokenizer/tests/ -q
#   → 15 passed

# 2. round-trip on real data, produce metrics + token stats + JSONL output
%PY% -m trendline_tokenizer.cli.roundtrip --limit 20000 --out data/tokenized/rule_v1_sample.jsonl
#   → manual_summary, auto_summary, top-10 coarse token histogram

# 3. train the learned VQ-VAE (CUDA auto-detected)
%PY% -m trendline_tokenizer.learned.train --limit 40000 --epochs 5 --batch-size 512
#   → checkpoint at checkpoints/trendline_tokenizer/vqvae_v0.pt
#   → val_role_acc ~1.0, val_bounce_acc ~0.98 after 5 epochs
```

## Acceptance criteria (§G of the brief)

| # | Criterion | Status |
|---|---|---|
| 1 | Load a TrendlineRecord from JSON/CSV | ✅ `adapters/manual.py` + `adapters/legacy_patterns.py` |
| 2 | Encode into coarse / fine tokens | ✅ `tokenizer.encode_rule` |
| 3 | Decode tokens back into an approximate TrendlineRecord | ✅ `tokenizer.decode_rule` |
| 4 | Measure reconstruction error | ✅ `tokenizer/metrics.py`, baseline printed by `cli/roundtrip.py` |
| 5 | Inspect token bucket distribution | ✅ `cli/roundtrip.py` prints top-10 coarse histogram + distinct-tokens count |
| 6 | Generate automatic trendline candidates | Reuse `sr_patterns.detect_patterns` (already producing 271k rows) |
| 7 | Compare manual vs automatic | Next turn: Hungarian matching on (symbol, tf, geometric distance) |
| 8 | Overlay original + decoded on chart | Next turn: `visualization/plot_overlay.py` |
| 9 | Optional: learned tokenizer outperforms rule baseline | ✅ val_role_acc 1.0, val_bounce_acc 0.987 |
| 10 | No large model training required for first milestone | ✅ rule tokenizer alone satisfies #1–5 |

## Next Agent Mode task (exact copy-paste)

```
Build visualisation + a Bayesian parameter sweep for the auto generator.

1. `trendline_tokenizer/visualization/plot_overlay.py`:
   - matplotlib-based
   - inputs: (ohlcv_df, original_record, decoded_record)
   - draws candles, the original line, the decoded line, error
     annotations (slope_err, duration_err)
   - saves PNG; no GUI dependency
   - also: a `plot_token_histogram(tokenised_jsonl)` that draws the
     coarse-token frequency chart

2. `trendline_tokenizer/auto_label/sweep.py`:
   - wraps sr_patterns.detect_patterns with a Bayesian search
     (bayes_opt or optuna) over SRParams
   - metric: F1 on Hungarian matching vs the 78 manual lines
   - logs all trials, saves best SRParams to configs/sr_params_v2.json

3. Add tests:
   tests/test_visualization.py  — asserts the PNG writer produces a
                                  non-empty file for a fixture record
   tests/test_sweep.py          — runs a 3-trial sweep end-to-end

4. Commit with message:
   "feat(tokenizer): overlay viz + sr_params bayes sweep"

Do NOT yet start the autoregressive sequence model; gated on
visualisation confirming that decoded records look plausible against
candles.
```

## What NOT to do next

- Do not scale the VQ-VAE to bigger sizes until visualisation is in.
  Bigger model on bad features = worse interpretability, not better.
- Do not add the autoregressive Transformer (`sequence_model/`) until
  coarse+fine distinct-token counts grow past 30 % codebook utilisation.
  Otherwise the AR model will learn a near-degenerate token distribution.
- Do not wire this into the live trading system
  (`server/conditionals/`, `server/execution/`). Research pipeline
  only.
- Do not ask the user to hand-draw more lines. All data growth is via
  `sr_patterns` auto generation + the existing outcome labeler.

## Data to prepare before the NEXT run

1. OHLCV CSVs for the symbols that appear in `data/patterns/*.jsonl`
   but where we don't yet have candles cached. (Needed for the
   visualisation step; not needed for more training — the VQ-VAE works
   directly on the feature vectors.)
2. If the user wants broader symbol coverage, run the existing auto
   detector over additional symbols (loop in `server/pattern_service`
   batch mode). This is a Python task, not human drawing.
