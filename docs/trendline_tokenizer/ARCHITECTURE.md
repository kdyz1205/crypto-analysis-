# ARCHITECTURE — Trendline Tokenizer

## A. How Kronos maps to this project

Kronos tokenises each K-line as `(coarse_token, fine_token)` via a VQ-VAE
over continuous OHLCV feature vectors, then trains a decoder-only
Transformer over those paired tokens to forecast future candles.

We apply the same pattern with the tokenised object swapped out:

| Kronos | This project |
|---|---|
| 1 candle | 1 trendline / market-structure segment |
| OHLCV vector (5-dim) | `TrendlineRecord` feature vector (~35-dim, see §B) |
| Coarse codebook ~256 | Coarse codebook (role × direction × tf × duration × slope-coarse) |
| Fine codebook ~1024 | Fine codebook (slope-fine × touch × bounce × break × anchor × vol × volume) |
| VQ-VAE encode/decode | Same; decoder reconstructs the full record, including role + bounce/break heads |
| AR Transformer over candle tokens | AR Transformer over trendline-event tokens (next segment, next break, next bounce) |

**Why trendline-level tokens beat candle-level tokens for this user:**
candles are a commodity; Kronos already does them. The user's trading
edge is line-level structure (support/resistance/channels/wedges). A
model that learns to predict "next resistance break on 4h" is
dramatically more trade-relevant than "next candle close".

## B. Canonical data schema

One record type. All manual, auto, and derived lines serialise into it
for every tool downstream. No Python-only types; JSON/CSV exchangeable.

### `TrendlineRecord`

```python
# trendline_tokenizer/schemas/trendline.py
from typing import Literal, Optional
from pydantic import BaseModel, Field

LineRole = Literal[
    "support", "resistance",
    "channel_upper", "channel_lower",
    "wedge_side", "triangle_side",
    "unknown",
]
Direction = Literal["up", "down", "flat"]
LabelSource = Literal["manual", "auto", "auto_approved", "auto_rejected"]

class TrendlineRecord(BaseModel):
    # ── identity ─────────────────────────────────────────
    id: str                               # stable hash of (symbol, tf, anchors)
    symbol: str
    exchange: str = "bitget"
    timeframe: str                        # "1m", "5m", "15m", "1h", "4h", "1d", "1w"

    # ── geometry (enough to re-draw on any candle chart) ─
    start_time: int                       # unix seconds
    end_time: int
    start_bar_index: int                  # bar index within the symbol+tf series
    end_bar_index: int
    start_price: float
    end_price: float
    extend_right: bool = False
    extend_left: bool = False

    # ── classification ───────────────────────────────────
    line_role: LineRole
    direction: Direction

    # ── structural behaviour ─────────────────────────────
    touch_count: int = Field(ge=0)
    rejection_strength_atr: Optional[float] = None     # max excursion vs line / ATR
    bounce_after: Optional[bool] = None
    bounce_strength_atr: Optional[float] = None
    break_after: Optional[bool] = None
    break_distance_atr: Optional[float] = None
    retested_after_break: Optional[bool] = None

    # ── context (for tokeniser buckets) ──────────────────
    volatility_atr_pct: Optional[float] = None          # ATR / price
    volume_z_score: Optional[float] = None
    distance_to_ma20_atr: Optional[float] = None
    distance_to_recent_high_atr: Optional[float] = None
    distance_to_recent_low_atr: Optional[float] = None

    # ── provenance ───────────────────────────────────────
    label_source: LabelSource
    auto_method: Optional[str] = None                   # e.g. "sr_patterns.v1"
    score: Optional[float] = None                       # detector confidence
    confidence: Optional[float] = None                  # human or classifier confidence
    quality_flags: list[str] = Field(default_factory=list)
    created_at: int
    notes: str = ""
```

### `TokenizedTrendline`

```python
class TokenizedTrendline(BaseModel):
    record_id: str
    coarse_token_id: int
    fine_token_id: int
    tokenizer_version: str                # "rule.v1" or "learned.v0.3"
    reconstruction_error: Optional[float] = None
```

### `TokenizerConfig` (versioned; every change bumps the version)

```python
class BucketSpec(BaseModel):
    name: str
    kind: Literal["enum", "linear", "log", "quantile"]
    edges: Optional[list[float]] = None   # for linear/log/quantile
    values: Optional[list[str]] = None    # for enum

class TokenizerConfig(BaseModel):
    version: str
    coarse_dims: list[BucketSpec]         # dimensions composing coarse id
    fine_dims: list[BucketSpec]           # dimensions composing fine id
    feature_vector_keys: list[str]        # which TrendlineRecord fields feed into the learned tokenizer
    normalization: dict[str, dict]        # per-field normalisation params
```

### `OHLCVBar` (trivial — matches existing repo format)

```python
class OHLCVBar(BaseModel):
    time: int                             # bar_open_ts, unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
```

### Mapping existing assets to this schema (no data loss)

| Source | Fields that already map cleanly | Fields we compute fresh |
|---|---|---|
| `data/manual_trendlines.json` (78 rows) | `symbol`, `timeframe`, `side` → `line_role`, `t_start` → `start_time`, `price_start`, `price_end`, `extend_*`, `notes`, `created_at` | `start_bar_index`, `end_bar_index`, `direction`, `touch_count`, `rejection_strength_atr`, bounce/break outcomes, volatility/volume context |
| `data/patterns/*.jsonl` (271k rows) | raw pattern dicts from `sr_patterns` — touches, slope, anchors | Re-derive `line_role` from pattern type (support/resistance/channel/wedge/triangle) |
| `server/pattern_features.TrendLineFeatures` | slope, touch_count, volatility_norm, direction, length_bars, price_span, line_type | direct adapter; keep as reference |

A deterministic converter in `trendline_tokenizer/adapters/` does this
mapping ONCE and emits versioned Parquet files. Downstream code never
re-parses raw legacy formats.

## C. Rule-based tokenizer (Phase 2)

Built first because it is transparent, debuggable, and produces a
baseline that every future learned tokenizer must beat on real metrics
(reconstruction error, bounce/break predictive AUC).

### Coarse token — 5 dimensions

| Dim | Values | Cardinality |
|---|---|---|
| `line_role` | support / resistance / channel_upper / channel_lower / wedge_side / triangle_side / unknown | 7 |
| `direction` | up / down / flat | 3 |
| `timeframe` | 1m 3m 5m 15m 30m 1h 2h 4h 6h 12h 1d 1w | 12 |
| `duration_bucket` | short (≤16 bars) / medium (17–48) / long (49–128) / very_long (>128) | 4 |
| `slope_coarse` | flat (\|log_slope\| < 0.002) / low (<0.005) / mid (<0.015) / high (<0.04) / very_high (≥0.04) | 5 |

Coarse vocabulary size: 7 × 3 × 12 × 4 × 5 = **5,040**. Bounded, flat, enumerable.

Encoding: multiplicative mixed-radix. Given (r, d, t, dur, slp) indices,
`coarse_id = r*(3*12*4*5) + d*(12*4*5) + t*(4*5) + dur*5 + slp`. Decoding
is the inverse — no hash tables needed, fully deterministic.

### Fine token — 7 dimensions

| Dim | Values | Cardinality |
|---|---|---|
| `slope_fine` | 10 quantile buckets within the `slope_coarse` bin (per-TF quantiles) | 10 |
| `touch_count` | 1 / 2 / 3 / 4 / 5+ | 5 |
| `bounce_strength` | none / weak (<0.5 ATR) / medium (<1.5 ATR) / strong (≥1.5 ATR) | 4 |
| `break_status` | none / touched / broken_weak / broken_retested | 4 |
| `anchor_position` | early / middle / late (within recent N bars) | 3 |
| `volatility_regime` | low / normal / high (ATR/price terciles) | 3 |
| `volume_regime` | low / normal / high (volume z-score terciles) | 3 |

Fine vocabulary: 10 × 5 × 4 × 4 × 3 × 3 × 3 = **21,600**.

### Encode / decode API

```python
def encode_rule(record: TrendlineRecord, cfg: TokenizerConfig) -> TokenizedTrendline: ...
def decode_rule(tok: TokenizedTrendline, cfg: TokenizerConfig) -> TrendlineRecord:
    """Returns a RECONSTRUCTED record. Bucket midpoints fill the
    continuous fields (slope, touch count, etc.). Geometry
    (start/end_price, bar indices) is reconstructed from the coarse
    slope bucket + TF + duration; it will not match the original
    exactly — that's the expected reconstruction error."""
```

### Reconstruction-error metric (locks in before any learned work)

For each record:
- `slope_err = |log(decoded.slope / original.slope)|`
- `role_err = 0 if decoded.role == original.role else 1`
- `duration_err = |decoded.duration - original.duration| / original.duration`
- `touch_err = |decoded.touch_count - original.touch_count|`
- `aggregate = 0.4*slope_err + 0.3*role_err + 0.2*duration_err + 0.1*touch_err`

Track median and p95 on (manual gold set) and (auto test slice). Any
learned tokenizer later must beat these numbers before it ships.

## D. Learned tokenizer (Phase 7 — gated on rule round-trip passing)

### Input feature vector (~35 dims)

Normalise per (symbol, timeframe) window so the model sees scale-free
geometry:

```
normalized_start_price        # (start_price - window_mean) / window_std
normalized_end_price
normalized_start_bar_index    # relative to window [0, 1]
normalized_end_bar_index
duration_bars_log             # log1p(end_bar_index - start_bar_index)
slope                         # log-price per bar
angle_rad                     # atan(slope * chart_aspect)
touch_count                   # raw int (clip at 8)
rejection_strength_atr        # clip [0, 5]
break_distance_atr            # signed, clip [-5, 5]
bounce_strength_atr           # clip [0, 5]
retested_after_break          # 0/1
volatility_atr_pct
volume_z_score
distance_to_ma20_atr
distance_to_recent_high_atr
distance_to_recent_low_atr
role_onehot[7]
timeframe_onehot[12]
```

### Model — hierarchical VQ-VAE

```
encoder:   MLP(35 → 128 → 128) with GELU
coarse_q:  VectorQuantizer(codebook_size=256, dim=128, β=0.25)
coarse_emb = coarse_q(encoder_out)
fine_q:    VectorQuantizer(codebook_size=1024, dim=128, β=0.25, input = encoder_out − sg(coarse_emb))
decoder_in = coarse_emb + fine_residual_emb
decoder:   MLP(128 → 128 → 35) + heads:
    role_head         → 7-class softmax
    bounce_head       → 2-class
    break_head        → 2-class
```

The "coarse captures broad structure, fine captures residual geometry"
property falls out of subtracting `sg(coarse_emb)` before the fine
quantiser — identical to Kronos's staged quantisation.

### Losses

```
loss =
    1.0 * L1(decoded_feat, target_feat)
  + 0.5 * CE(role_head,   target_role)
  + 0.3 * BCE(bounce_head, target_bounce)
  + 0.3 * BCE(break_head,  target_break)
  + 0.25 * (||encoder_out - sg(coarse_emb)||² + ||sg(encoder_out) - coarse_emb||²)
  + 0.25 * (fine analog)
  + 0.01 * entropy_regularizer(codebook_usage)    # against collapse
```

### Training data requirement

From the asset inventory we already have:

- **271,773** auto rows across multiple symbols/TFs → plenty for a
  small VQ-VAE.
- **78** manual rows → gold validation split.

Before training ships, expand auto data to **~1M rows** using the
existing `sr_patterns` detector with broader sweep params (longer
history, more symbols). This is an overnight batch, not a manual task.

### Validation metrics (must beat rule-tokenizer baseline)

1. Reconstruction error (same aggregate as rule tokenizer, lower is better).
2. Role classification accuracy on decoded output.
3. Bounce/break AUC on the held-out slice (compare against the existing
   `pattern_bounce_auc_0.928.pt` scorer as the ceiling).
4. Codebook utilisation: ≥80 % of codes used on the validation set
   (collapse detection).
5. Round-trip stability: re-encoding a decoded record must give the
   same token ≥95 % of the time.

## E. Autoregressive trendline event model (Phase 8 — after D)

### Sequence construction

For each (symbol, timeframe), order all records by `end_time`. Each
position is a 2-token tuple `(coarse, fine)`. A special `[EVENT]`
meta-token marks line-break and line-bounce events (inferred from
downstream candles after `end_time`).

```
seq = [
  (coarse_5040, fine_12345),  # new line formed
  (EVENT_BOUNCE_STRONG,  EVENT_NULL),
  (coarse_4812, fine_8876),   # new channel upper
  (EVENT_BREAK_RETESTED, EVENT_NULL),
  ...
]
```

Context window: 64 trendline tokens ≈ 128 positions (pair tokens).

### Model

Decoder-only Transformer, d_model=256, 6 layers, 8 heads, rotary
positional embeddings on the token-position axis. Separate softmax
heads for coarse-next and fine-next (shared trunk).

### Prediction targets (what the user reads)

- next `line_role` distribution
- next direction
- probability of bounce / break in the next N bars
- expected time-to-event (quantile regression)

### Explicitly NOT outputs of this model

- Order size, leverage, entry / SL / TP — those belong to the live
  trading system and remain untouched.

## Improvement loop for the auto generator (Phase 3 — before training)

1. Run `sr_patterns.detect_patterns` with the current `SRParams`.
2. Use `server/strategy/drawing_outcome_labeler.py` to label each auto
   line with (bounced, broke, time_to_outcome).
3. Train a scorer on (features → outcome) — reuse the existing
   `checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt` or fit a
   fresh XGBoost/small MLP.
4. Filter auto candidates: keep top-K per (symbol, tf, window) by scorer
   probability.
5. Compare filtered auto lines vs the 78 manual lines using Hungarian
   matching on (symbol, tf, geometric distance). Report precision /
   recall / slope MAE.
6. Sweep `SRParams` with Bayesian optimisation on the metric from (5).
   **No RL until this converges** — Bayes is enough when you have 78
   gold points and a fast-to-evaluate detector.
7. RL / evolutionary search (later): reward = realised bounce/break
   outcome + matches-gold bonus. Wire through `server/strategy/evolution`
   if the user wants to promote this.

This whole loop is scripted. The user does not hand-draw more lines
during this phase.
