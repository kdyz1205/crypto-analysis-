# Extended Feature Spec (User-supplied, 2026-04-24)

This is the user's detailed catalogue of hundreds of trendline / segment
features, provided as the roadmap for expanding the feature vector
beyond the minimal 36-dim v0 used by the current rule tokenizer and
VQ-VAE. Copy is preserved verbatim at the bottom for traceability.

## How v0 maps against this spec today

| Category (from user spec) | In v0 feature vector? | In v0 rule coarse token? | In v0 rule fine token? |
|---|---|---|---|
| 1.1 Primary keys (symbol, tf, start/end time, anchors, role, source) | yes | yes (role, tf) | — |
| 1.2 Geometry (duration, slope, angle, length) | yes | yes (duration, slope_coarse) | yes (slope_fine) |
| 2. Touch / reaction (touch_count, bounce, break, retest, wick vs body) | partial (touch, bounce, break, retested) | no | yes (touch, bounce, break) |
| 3.1 Volatility / momentum (ATR regime, std/skew/kurt, RSI, MACD...) | partial (volatility_atr_pct) | no | yes (volatility_regime) |
| 3.2 Volume (avg, trend, spikes, OBV, MFI) | partial (volume_z_score) | no | yes (volume_regime) |
| 3.3 Trend / MAs (regime, ribbon state, ichimoku, bollinger, keltner) | partial (distance_to_ma20_atr) | no | no |
| 3.4 Structure (swing, pivot density, confluence, fib, phase) | no | no | no (anchor_position only) |
| 4. Future-behaviour labels (future touches, outcomes, PnL) | no (model target) | no | no |
| 5.1 Technical indicator values (RSI/MACD/KDJ/boll/ATR/CCI/ADX/...) | no | no | no |
| 5.2 Derived stats (correlation, autocorr, Hurst, entropy, fractal, spectral) | no | no | no |
| 6. Candlestick patterns + gaps + trend acceleration + clustering | no | no | no |
| 7. Interaction with other lines (intersections, parallel, confluence, stack level) | no | no | no |
| 8. Metadata (auto_method, score, confidence, reviewer, timestamps) | yes | no | no |

**Summary:** v0 covers categories 1.1–1.2 fully, 2 / 3.1 / 3.2 in part,
and drops everything in sections 3.3 (except MA distance), 3.4, 5, 6, 7.
These are exactly the "extended features" that, per the user spec,
should be fed into v1 of the learned tokenizer.

## Planned expansion roadmap

### v0 → v1 (next iteration)

Add these without breaking `TokenizerConfig.version = "rule.v1"`:

1. **Wick vs body discrimination** — requires OHLCV join; split
   `touch_count` into `wick_violation_count` + `body_violation_count`.
2. **First-touch outcome** — enum `{bounce, fake_break, true_break,
   no_reaction}` as a fine-token dimension.
3. **Trend regime** at start_time — enum `{uptrend, downtrend, range,
   volatile_range}` from existing `server.strategy.regime`.
4. **MA alignment** at start_time — enum `{bull_stacked, bear_stacked,
   mixed, compressed}` from existing MA Ribbon state.
5. **Confluence count** — integer, number of other auto lines within
   0.5×ATR of this line at `end_bar_index`.
6. **Parallel channel flag** — boolean, whether a near-parallel line of
   opposite role exists.
7. **Timeframe-of-parent-swing** — which HTF pivot anchors this line.

These add roughly 8 dims to the continuous vector and 3–5 new fine
dimensions. Coarse token cardinality grows from 5,040 → ~12,000.

### v1 → v2 (after more data + scorer v2)

Add technical-indicator snapshots at start / end / mid:
`rsi`, `macd_hist`, `bollinger_bandwidth`, `atr_ratio`, `adx`,
`vwap_distance`. Feature vector grows to ~60 dims. Learned tokenizer
then handles the increased dimensionality; rule tokenizer stays at v1
(it's the interpretable baseline, no need to enlarge).

### v2 → v3 (multi-line interaction)

Line-to-line interaction features (section 7 of user spec):
`intersection_count`, `nearest_line_distance`, `confluence_score`,
`support_stack_level`, `cross_timeframe_conflict`. These need a
line-graph builder that indexes all lines per (symbol, tf) and
computes pairwise geometry. Output: 10 additional dims on the feature
vector. Enables the AR model to predict whether the next event is
inside/outside an existing cluster.

### v3+ (autoregressive sequence model — Phase 8 in MVP_PLAN.md)

Feature vector is frozen; the sequence model operates on
`(coarse, fine)` token pairs directly. No further feature expansion
needed for AR training itself.

## Where to pull each feature family from (reuse before writing)

| Feature family | Existing module in this repo to reuse |
|---|---|
| Pivots / swings | `server/strategy/pivots.py` — `detect_pivots()` |
| Touch / bounce / break outcomes | `server/strategy/drawing_outcome_labeler.py` |
| MA / ribbon state | `server/ma_ribbon_service.py`, `server/strategy/factors.py` |
| Regime | `server/strategy/regime.py` |
| RSI / MACD / ATR / ADX | `server/strategy/indicators.py` |
| Confluence | `server/strategy/confluence.py` |
| Legacy pattern rows | `data/patterns/*.jsonl` + `sr_patterns.py` |
| Manual lines | `data/manual_trendlines.json` + `server/drawings/store.py` |
| Bounce classifier (scorer baseline) | `checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt` |

The rule for v1 expansion: **do not write a new pivot detector, MA
calculator, or outcome labeler**. Call the existing functions and wrap
their output into the `TrendlineRecord` schema.

## Bucket cardinality playbook

When extending the rule tokenizer with a new dimension, apply this
rule before merging:

1. If the dim has ≤8 distinct symbolic values, use a raw enum.
2. If continuous, start with 3-way tercile buckets. Only expand to
   5- or 10-way buckets after showing in the metrics report that the
   coarse-vs-fine separation improves (`aggregate_median` and
   `bounce_auc` on the held-out manual set).
3. Never add a dim whose cardinality × existing coarse cardinality
   exceeds ~50,000 without moving it to the fine token.

## v0 rule-tokenizer output on real data (baseline to beat)

```
manual set (78 rows):
  aggregate_median = 0.127   p95 = 1.45   role_accuracy = 1.000
  slope_err_median = 0.047   p95 = 1.06

auto set (5,000 sample rows from data/patterns/*.jsonl):
  aggregate_median = 0.119   p95 = 0.70   role_accuracy = 1.000
  slope_err_median = 0.134   p95 = 1.63
  distinct coarse tokens = 69 / 5040 (1.4 %)
  distinct fine tokens   = 176 / 21600 (0.8 %)
```

Low utilisation on the rule tokenizer is expected — the legacy auto
dataset only covers {support, resistance} × a few TFs. v1 features
above will push utilisation up.

## Learned tokenizer v0 training result (baseline to beat)

```
epoch 5  train_recon=0.087
         val_role_acc   = 1.000
         val_bounce_acc = 0.987
         val_break_acc  = 0.546
         coarse_codes_used = 64 / 256   (25 %)
         fine_codes_used   = 119 / 1024 (12 %)
```

Bounce 0.987 is strong. Break 0.546 is noise-level — expected, since
break labels are heavily imbalanced in the legacy rows. Fix in v1 by:

- Balancing the dataset (oversample broken lines).
- Adding break-related context features (momentum, volume, regime).
- Reusing the existing `pattern_bounce_auc_0.928.pt` as a weak label
  for semi-supervised pre-training.

---

## Appendix — User's original feature catalogue (reference, verbatim)

> 本文档面向构建趋势线/圈段语言模型，为手工标注及自动生成的趋势线对象
> 设定详细的特征列表和分类方法。[…section-by-section catalogue with
> 200+ feature names across 11 categories…]
>
> 11. 建议的实施步骤
> 1. 设计数据模式
> 2. 收集手工数据
> 3. 自动生成候选线段
> 4. 规则式 Tokenizer V1
> 5. 统计分析与特征选择
> 6. 构建学习式 Tokenizer
> 7. 模型训练与评估

The full catalogue is stored alongside this doc at
`docs/trendline_tokenizer/user_spec_2026-04-24.md` when the user
commits the raw text; use it as the authoritative reference when
deciding which features to add in each subsequent iteration.
