# Trendline Tokenizer — Engineering Design (Brownfield)

> Kronos-inspired hierarchical tokenizer where the tokenized object is a
> **trendline / market-structure segment**, not a raw candle.
>
> **This is NOT greenfield.** The repo already contains an auto
> detector (`sr_patterns.py`), 271k structured pattern rows
> (`data/patterns/*.jsonl`), 78 manual gold lines
> (`data/manual_trendlines.json`), a feature extractor
> (`server/pattern_features.TrendLineFeatures`), and a trained bounce
> classifier (`checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt`).
> The design below REUSES those and builds the tokenizer on top.
> Human labor stays minimal.

## Asset inventory (scanned 2026-04-24)

```
sr_patterns.py                                       auto-detector (SRParams, detect_patterns, PatternResult, TrianglePattern)
server/strategy/pivots.py                            pivot detection (local high/low with left/right window)
server/pattern_features.py                           TrendLineFeatures dataclass (slope, touch_count, volatility_norm, direction, length_bars, price_span, line_type)
server/pattern_service.py                            wraps detect_patterns → JSON for frontend
server/routers/patterns.py                           HTTP API
server/strategy/drawing_learner.py                   ML loop for drawing quality
server/strategy/drawing_outcome_labeler.py           labels auto lines with bounce/break outcome
server/drawings/store.py                             ManualTrendlineStore (JSON file)
data/manual_trendlines.json                          78 manual gold lines
data/patterns/*.jsonl                                271,773 auto pattern rows across {ADA,BTC,DOGE,XRP,...} × {5m,15m,1h,4h,1d}
checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt    pre-trained scorer, AUC 0.928 on bounce prediction
agent/pattern_driven.py                              existing agent consuming patterns
```

## This folder contains

| File | Purpose |
|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Full design mapped to the existing assets. Sections A–E of the brief (architecture, schemas, rule tokenizer, learned tokenizer, AR model). |
| [`MVP_PLAN.md`](./MVP_PLAN.md) | Concrete file tree, staged milestones with tests, and the **exact first Agent Mode task**. Sections F–G. |

## Non-negotiable constraints

1. **Reuse `sr_patterns.detect_patterns`** as the first auto generator.
   Do not rewrite a detector from scratch. Improve it via scoring +
   backtest + RL after the tokenizer round-trip works.
2. **Use the 78 manual lines as the gold validation set.** Do not ask
   the user to draw thousands of lines.
3. **Don't touch `server/conditionals/*`, `server/execution/*`, or
   frontend live-trading code.** This project is orthogonal.
4. **No learned tokenizer until rule-based encode/decode/visualise
   round-trip passes** on both manual and auto data.
5. **Every trendline serialisable as JSON/CSV** via one canonical
   schema (defined in `ARCHITECTURE.md` §B).
6. **Fail the milestone if acceptance tests don't run.** No tests = not
   done.

## Entry point

1. Read `ARCHITECTURE.md` once, end-to-end.
2. Read `MVP_PLAN.md` Phase 0 + Phase 1.
3. Run the **first Agent Mode task** at the bottom of `MVP_PLAN.md`.
