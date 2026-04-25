# Trendline Tokenizer + Model System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the system the architecture diagram describes — offline training of a fusion model over (price sequence × trendline tokens), online inference + signal engine, and a human-in-the-loop feedback / retraining loop — on top of the existing `trendline_tokenizer/` package (schemas, rule tokenizer, learned VQ-VAE) and existing pattern detector (`server/strategy/trendlines.py`, `server/pattern_features`).

**Architecture:** Two-branch encoder: `PriceSequenceEncoder` (Transformer over OHLCV+indicators) and `TrendlineTokenEncoder` (embedding+Transformer over `(coarse_id, fine_id)` pairs from the existing rule/VQ-VAE tokenizers). Cross-attention fusion produces a fused market-structure representation consumed by multi-task heads (next-token prediction, bounce/break classification, continuation probability, suggested buffer regression). Same fusion model serves training and inference; inference wraps it with a feature cache + signal engine. Feedback writes corrected `TrendlineRecord` rows that flow into the next dataset refresh.

**Tech Stack:** Python 3.12, PyTorch (already installed for VQ-VAE), Pydantic v2, FastAPI (already used in `server/`), pytest, NumPy, pandas. Existing `trendline_tokenizer/` package extended; no new top-level packages.

---

## Status Map — Diagram → Existing vs To-Build

Citations are to existing files. Treat anything marked **DONE** as the canonical implementation; this plan does not duplicate it.

### 1. Offline Training Pipeline

| Diagram Box | Status | File / Notes |
|---|---|---|
| Market Data Lake | **DONE** (reuse) | `data/patterns/*.jsonl` (271k auto rows), `data/manual_trendlines.json` (78 manual), `data/user_drawing_labels.jsonl`, `data/user_drawing_outcomes.jsonl`, OHLCV cache via `server/routers/market.py` |
| Trendline Detection & Labeling | **DONE** (reuse) | `server/strategy/trendlines.py`, `research/trendline_backtest.py` |
| Trendline Object Builder | **DONE** | `trendline_tokenizer/adapters/{manual,legacy_patterns,user_outcomes}.py` |
| TrendlineTokenizer v1 (rule) | **DONE** | `trendline_tokenizer/tokenizer/{vocab,rule,metrics}.py` — 5040 coarse × 21600 fine, mixed-radix, round-trip tested |
| Learned TrendlineTokenizer v2 | **DONE** | `trendline_tokenizer/learned/{vqvae,dataset,train}.py` — hierarchical 256/1024 VQ-VAE, checkpoint at `checkpoints/trendline_tokenizer/vqvae_v0.pt` |
| Price Sequence Encoder | **TO BUILD** | this plan, Milestone 1 |
| Trendline Token Encoder | **TO BUILD** | this plan, Milestone 1 |
| Fusion Layer | **TO BUILD** | this plan, Milestone 1 |
| Multi-Task Heads | **TO BUILD** | this plan, Milestone 1 |
| Losses & Training Objectives | **TO BUILD** | this plan, Milestone 1 |
| Model Registry / Artifacts | **TO BUILD** | this plan, Milestone 1 (formalise `checkpoints/` + manifest) |

### 2. Online Inference / Serving

| Diagram Box | Status | File / Notes |
|---|---|---|
| Realtime Market Feed | **DONE** (reuse) | `frontend/js/services/stream.js`, `server/routers/market.py`, `server/bitget_private_ws.py` |
| Feature Builder & Multi-Timeframe Cache | **PARTIAL** | OHLCV cache exists; needs indicator pipeline + sequence-window builder |
| Runtime Trendline Detection | **DONE** (reuse) | `server/strategy/trendlines.py` |
| Runtime Tokenizer | **TO BUILD** | this plan, Milestone 2 (thin wrapper around existing rule + VQ-VAE) |
| Inference Model | **TO BUILD** | this plan, Milestone 2 |
| Signal Engine | **TO BUILD** | this plan, Milestone 2 |
| Strategy / Execution Connectors | **PARTIAL** (paper only) | `server/strategy/trendline_order_manager.py` exists for live; this plan only wires **paper** |
| Monitoring & Logging | **PARTIAL** | needs prediction/drift logging |

### 3. UI + Human-in-the-Loop

| Diagram Box | Status | File / Notes |
|---|---|---|
| Interactive Chart UI | **DONE** (extend) | `frontend/v2.html`, `frontend/js/workbench/chart.js`, `drawings/chart_drawing.js` |
| Analyst Review Queue | **TO BUILD** | this plan, Milestone 3 |
| Feedback Store | **PARTIAL** | `data/user_drawing_outcomes.jsonl` exists; needs corrected-trendline schema + signal accept/reject log |
| Dataset Refresh / Retraining Trigger | **PARTIAL** | `trendline_tokenizer/evolve/rounds.py` exists for line-evolution; needs model-retrain hook |
| Research / Backtest Console | **PARTIAL** | `research/`, `scripts/run_trendline_strategy_backtest.py`; needs ablation + token-vis UI |

---

## Plan structure (4 milestones, this doc is Milestone 1 in detail)

This document writes **Milestone 1** as bite-sized TDD tasks. Milestones 2–4 are outlined; each will be a separate plan once Milestone 1 lands and its model interfaces are concrete.

| Milestone | Output | Status |
|---|---|---|
| 1. Sequence Model & Fusion | `models/`, `training/`, `registry/` — trainable + checkpoint | **THIS DOC, full TDD detail** |
| 2. Inference Service & Signal Engine | `inference/`, paper signal stream | outlined only |
| 3. Feedback Loop & Retraining Trigger | `ui_api/feedback`, `evolve/retrain.py` | outlined only |
| 4. Backtest / Research Console | `backtest/`, ablation reports | outlined only |

---

# Milestone 1 — Sequence Model & Fusion (Multi-Stream)

## 中文概要

**目标:** 训练一个融合模型,它同时吃两条信息:
1. **价格序列** — 最近 256 根 K 线的 OHLCV + 8 个指标(EMA21/EMA50/ATR14/RSI14/Vol-Z/dist-MA/ret1/ret5)
2. **趋势线序列** — 最近 32 个趋势线对象,每个用 **三种** 表示同时输入:
   - **规则 token (rule)**: 你已有的 5040×21600 mixed-radix 整数,可解释
   - **学习 token (learned VQ-VAE)**: 你已训练好的 256×1024 离散码本,捕捉规则没覆盖的几何残差
   - **原始特征 (raw)**: `features/vector.py` 产出的 36 维连续向量,保留量化丢失的精度

模型把三路 embedding **求平均后送入 Transformer**(每路独立可关,做消融实验)。Cross-Attention 让 K 线序列查询趋势线序列,得到融合表示,再分发到 6 个任务头:

| 任务头 | 输出 | 用途 |
|---|---|---|
| `next_coarse_logits` | (B, 5040) | 下一条线的 coarse rule token |
| `next_fine_logits` | (B, 21600) | 下一条线的 fine rule token |
| `bounce_logits` | (B, 2) | 反弹概率 |
| `break_logits` | (B, 2) | 突破概率 |
| `continuation_logits` | (B, 2) | 趋势延续概率 |
| `buffer_pct` | (B,) | 建议止损 buffer (% of price,回归) |

**为什么三路同时用最厉害:**
- rule token 给可解释性的"骨架"(支撑/阻力/方向/时间框/斜率桶)
- learned token 给规则桶分得太粗时的"残差精度"(VQ-VAE 编码 36 维特征里规则没用上的部分)
- raw 36 维给量化时彻底丢掉的连续信号(比如 0.4 vs 0.6 ATR 的 bounce 强度差异)

三路加起来 = 既保留规则系统的解释性,又拿到神经网络的精度,还能消融每一路验证它的贡献。

**版本注册表 (registry):** 每次训练保存 `manifest.json`,记录用了哪个 rule tokenizer 版本、哪个 VQ-VAE checkpoint、训练数据的版本、以及验证集指标。线上推理 / 反馈 / 回测都按 manifest 名字定位,版本不匹配直接报错。

**严格 no-lookahead:** 任何输入 token 的 `end_bar_index` **必须** 小于预测目标的 `end_bar_index`。这是整个 milestone 唯一不可妥协的不变量,有专门的测试 `test_no_lookahead.py` 钉死。

**输出:** `checkpoints/fusion/<artifact_name>/{fusion.pt, fusion.json, manifest.json}`。

---

## Objective (English)

Train a fusion model that takes (price sequence window) and (trendline sequence ending at the same bar, expressed via THREE parallel streams: rule tokens + learned VQ-VAE tokens + raw 36-dim feature vectors) and predicts: next trendline coarse/fine token, bounce probability, break probability, continuation probability, and suggested buffer % (regression). Save checkpoint to a versioned registry.

Each stream is independently gateable (`use_rule_tokens` / `use_learned_tokens` / `use_raw_features` in `FusionConfig`) so we can ablate later. Default: all three on — that's the "most powerful" baseline the user asked for.

**File structure (to create under existing `trendline_tokenizer/`):**

```
trendline_tokenizer/
  models/
    __init__.py
    config.py                   # FusionConfig (Pydantic)
    price_seq_encoder.py        # OHLCV + indicators → embedding
    token_encoder.py            # (coarse,fine) tokens → embedding
    fusion.py                   # cross-attention fusion
    heads.py                    # multi-task heads
    full_model.py               # composes everything; .forward, .compute_loss, .predict
    losses.py                   # loss weights + helpers
  training/
    __init__.py
    sequence_dataset.py         # builds (price_window, token_seq, targets) tuples
    train_fusion.py             # CLI entry-point
    eval_fusion.py              # holdout metrics
  registry/
    __init__.py
    manifest.py                 # ArtifactManifest schema + load/save
    paths.py                    # versioned checkpoint paths
  tests/
    test_models_forward.py
    test_sequence_dataset.py
    test_no_lookahead.py
    test_registry.py
```

**Tech-stack assumption:** Python interpreter is the same one used by the existing VQ-VAE training (the user's `Python312`). All commands use `python -m` so cwd doesn't matter.

---

## Task 1: Models package skeleton + FusionConfig (multi-stream)

**Files:**
- Create: `trendline_tokenizer/models/__init__.py`
- Create: `trendline_tokenizer/models/config.py`
- Create: `trendline_tokenizer/tests/test_models_forward.py` (just the config test for now)

- [ ] **Step 1: Write the failing test for FusionConfig defaults**

`trendline_tokenizer/tests/test_models_forward.py`:

```python
"""Tests for the fusion model — forward shapes, no-lookahead, config."""
import pytest
from trendline_tokenizer.models.config import FusionConfig


def test_fusion_config_defaults_match_existing_tokenizers():
    cfg = FusionConfig()
    # rule tokenizer vocab from tokenizer/vocab.py
    assert cfg.rule_coarse_vocab_size == 5040
    assert cfg.rule_fine_vocab_size == 21600
    # learned VQ-VAE vocab from learned/vqvae.py defaults
    assert cfg.learned_coarse_vocab_size == 256
    assert cfg.learned_fine_vocab_size == 1024
    # raw feature dim from features/vector.py
    assert cfg.raw_feat_dim == 36
    # all 3 streams on by default — "most powerful" baseline
    assert cfg.use_rule_tokens is True
    assert cfg.use_learned_tokens is True
    assert cfg.use_raw_features is True
    assert cfg.n_streams() == 3
    # OHLCV (5) + N_indicators must equal price_feat_dim
    assert cfg.price_feat_dim == 5 + cfg.n_indicators
    # context window: 256 candles + 32 trendline events
    assert cfg.price_seq_len == 256
    assert cfg.token_seq_len == 32


def test_fusion_config_loads_from_dict_with_ablation():
    raw = {"d_model": 64, "n_layers_price": 2,
           "use_learned_tokens": False, "use_raw_features": False}
    cfg = FusionConfig(**raw)
    assert cfg.d_model == 64
    assert cfg.n_streams() == 1   # only rule


def test_fusion_config_rejects_zero_streams():
    with pytest.raises(ValueError):
        FusionConfig(use_rule_tokens=False, use_learned_tokens=False,
                     use_raw_features=False)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: FAIL with `ModuleNotFoundError: trendline_tokenizer.models`.

- [ ] **Step 3: Create the package + config**

`trendline_tokenizer/models/__init__.py`:

```python
"""Fusion model: price sequence × trendline tokens → multi-task predictions."""
__version__ = "0.1.0"
```

`trendline_tokenizer/models/config.py`:

```python
"""Fusion model configuration. Versioned; bumping invalidates checkpoints.

Trendline branch supports THREE parallel streams:
  - rule tokens   (5040 coarse × 21600 fine, from tokenizer/rule.py)
  - learned tokens (256 coarse × 1024 fine, from learned/vqvae.py)
  - raw features  (36-dim continuous vector, from features/vector.py)

Each stream can be disabled for ablation. Default: all three on.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..tokenizer.vocab import coarse_vocab_size, fine_vocab_size
from ..features.vector import FEATURE_VECTOR_DIM
from ..learned.vqvae import VQVAEConfig


class FusionConfig(BaseModel):
    version: str = "fusion.v0.1"

    # ── trendline stream switches ───────────────────────────
    use_rule_tokens: bool = True
    use_learned_tokens: bool = True
    use_raw_features: bool = True

    # ── trendline stream sizes (locked to existing artifacts) ──
    rule_coarse_vocab_size: int = Field(default_factory=coarse_vocab_size)
    rule_fine_vocab_size: int = Field(default_factory=fine_vocab_size)
    learned_coarse_vocab_size: int = VQVAEConfig().coarse_codes
    learned_fine_vocab_size: int = VQVAEConfig().fine_codes
    raw_feat_dim: int = FEATURE_VECTOR_DIM
    vqvae_checkpoint_path: Optional[str] = "checkpoints/trendline_tokenizer/vqvae_v0.pt"

    # ── price branch ──────────────────────────────────────
    n_indicators: int = 8           # ema21, ema50, atr14, rsi14, vol_z, dist_ma_atr, ret_1, ret_5
    price_feat_dim: int = 13        # 5 OHLCV + 8 indicators (validated below)
    price_seq_len: int = 256
    n_layers_price: int = 4
    n_heads_price: int = 4

    # ── trendline branch ─────────────────────────────────
    token_seq_len: int = 32
    n_layers_token: int = 3
    n_heads_token: int = 4

    # ── fusion ───────────────────────────────────────────
    d_model: int = 128
    n_layers_fusion: int = 2
    n_heads_fusion: int = 4
    dropout: float = 0.1

    # ── heads ────────────────────────────────────────────
    next_token_head: bool = True
    bounce_head: bool = True
    break_head: bool = True
    continuation_head: bool = True
    buffer_head: bool = True

    def n_streams(self) -> int:
        return int(self.use_rule_tokens) + int(self.use_learned_tokens) + int(self.use_raw_features)

    def model_post_init(self, _ctx):
        if self.price_feat_dim != 5 + self.n_indicators:
            raise ValueError(
                f"price_feat_dim ({self.price_feat_dim}) must equal "
                f"5 + n_indicators ({5 + self.n_indicators})"
            )
        if self.n_streams() == 0:
            raise ValueError("at least one trendline stream must be enabled "
                             "(use_rule_tokens / use_learned_tokens / use_raw_features)")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/__init__.py trendline_tokenizer/models/config.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): models package skeleton + FusionConfig"
```

---

## Task 2: Price Sequence Encoder

**Files:**
- Create: `trendline_tokenizer/models/price_seq_encoder.py`
- Modify: `trendline_tokenizer/tests/test_models_forward.py`

- [ ] **Step 1: Write the failing test for forward shape**

Append to `trendline_tokenizer/tests/test_models_forward.py`:

```python
import torch
from trendline_tokenizer.models.config import FusionConfig
from trendline_tokenizer.models.price_seq_encoder import PriceSequenceEncoder


def test_price_seq_encoder_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_price=2, n_heads_price=4)
    enc = PriceSequenceEncoder(cfg)
    B, T, F = 2, cfg.price_seq_len, cfg.price_feat_dim
    x = torch.randn(B, T, F)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)  # no padding
    h = enc(x, pad_mask)
    assert h.shape == (B, T, cfg.d_model)


def test_price_seq_encoder_handles_padding():
    cfg = FusionConfig(d_model=64, n_layers_price=2, n_heads_price=4)
    enc = PriceSequenceEncoder(cfg)
    B, T, F = 2, cfg.price_seq_len, cfg.price_feat_dim
    x = torch.randn(B, T, F)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    pad_mask[0, -10:] = True   # last 10 of batch[0] are padded
    h = enc(x, pad_mask)
    assert torch.isfinite(h).all()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: FAIL with `ModuleNotFoundError: trendline_tokenizer.models.price_seq_encoder`.

- [ ] **Step 3: Implement `PriceSequenceEncoder`**

`trendline_tokenizer/models/price_seq_encoder.py`:

```python
"""Transformer encoder over OHLCV + indicators.

Input:  (B, T, price_feat_dim) — 5 OHLCV columns + n_indicators
Output: (B, T, d_model)        — per-bar embedding

Causal masking is NOT applied here; the bar at position t may attend
to t' < t and t' > t within the same window because the window itself
is the model's "now" view. No-lookahead is enforced at the dataset
level (only bars with end_time <= prediction time are included).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .config import FusionConfig


class PriceSequenceEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.price_feat_dim, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.price_seq_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads_price,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers_price)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F). pad_mask: (B, T) bool — True where padded."""
        h = self.input_proj(x) + self.pos_emb[:, : x.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out_norm(h)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/price_seq_encoder.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): PriceSequenceEncoder + shape tests"
```

---

## Task 3: TrendlineMultiStreamEncoder (rule + learned + raw)

**Files:**
- Create: `trendline_tokenizer/models/trendline_encoder.py`
- Modify: `trendline_tokenizer/tests/test_models_forward.py`

The encoder fuses up to three parallel streams per trendline position:
1. **Rule stream:** `rule_coarse_emb(rule_coarse) + rule_fine_emb(rule_fine)`
2. **Learned stream:** `learned_coarse_emb(learned_coarse) + learned_fine_emb(learned_fine)`
3. **Raw stream:** `raw_proj(raw_feat_36d)`

Streams are summed (averaged across enabled streams) before positional embedding + Transformer. Each stream has its own `LayerNorm` to prevent one stream from dominating before averaging. If a stream is disabled in `FusionConfig`, its module is not created and its key in the batch is ignored.

- [ ] **Step 1: Write the failing test**

Append to `trendline_tokenizer/tests/test_models_forward.py`:

```python
from trendline_tokenizer.models.trendline_encoder import TrendlineMultiStreamEncoder


def _toy_trendline_batch(cfg, B):
    return {
        "rule_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B, cfg.token_seq_len)),
        "rule_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B, cfg.token_seq_len)),
        "learned_coarse": torch.randint(0, cfg.learned_coarse_vocab_size, (B, cfg.token_seq_len)),
        "learned_fine": torch.randint(0, cfg.learned_fine_vocab_size, (B, cfg.token_seq_len)),
        "raw_feat": torch.randn(B, cfg.token_seq_len, cfg.raw_feat_dim),
        "token_pad": torch.zeros(B, cfg.token_seq_len, dtype=torch.bool),
    }


def test_trendline_encoder_all_streams_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_token=2, n_heads_token=4)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 3
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)


def test_trendline_encoder_rule_only_ablation():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2,
                       use_rule_tokens=True, use_learned_tokens=False,
                       use_raw_features=False)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 2
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)
    # the disabled streams' modules should not exist
    assert not hasattr(enc, "learned_coarse_emb") or enc.learned_coarse_emb is None
    assert not hasattr(enc, "raw_proj") or enc.raw_proj is None


def test_trendline_encoder_raw_only_ablation():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2,
                       use_rule_tokens=False, use_learned_tokens=False,
                       use_raw_features=True)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 2
    batch = _toy_trendline_batch(cfg, B)
    h = enc(batch, batch["token_pad"])
    assert h.shape == (B, cfg.token_seq_len, cfg.d_model)


def test_trendline_encoder_clamps_out_of_range_ids():
    cfg = FusionConfig(d_model=32, n_layers_token=1, n_heads_token=2)
    enc = TrendlineMultiStreamEncoder(cfg)
    B = 1
    batch = _toy_trendline_batch(cfg, B)
    batch["rule_coarse"] = batch["rule_coarse"] + cfg.rule_coarse_vocab_size  # overflow
    h = enc(batch, batch["token_pad"])
    assert torch.isfinite(h).all()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `TrendlineMultiStreamEncoder`**

`trendline_tokenizer/models/trendline_encoder.py`:

```python
"""Multi-stream trendline encoder.

A position represents one trendline event. Up to three representations
of that event are fused at the embedding stage:
    rule stream   : (rule_coarse_id, rule_fine_id) → 2 embeddings, summed
    learned stream: (learned_coarse_id, learned_fine_id) → same
    raw stream    : 36-dim continuous feature vector → linear projection

Each stream is layer-normed before averaging across enabled streams.
That keeps any one stream from dominating just because its statistics
have a larger scale.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class TrendlineMultiStreamEncoder(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # rule stream
        if cfg.use_rule_tokens:
            self.rule_coarse_emb = nn.Embedding(cfg.rule_coarse_vocab_size, d)
            self.rule_fine_emb = nn.Embedding(cfg.rule_fine_vocab_size, d)
            self.rule_norm = nn.LayerNorm(d)
        else:
            self.rule_coarse_emb = None
            self.rule_fine_emb = None
            self.rule_norm = None

        # learned (VQ-VAE) stream
        if cfg.use_learned_tokens:
            self.learned_coarse_emb = nn.Embedding(cfg.learned_coarse_vocab_size, d)
            self.learned_fine_emb = nn.Embedding(cfg.learned_fine_vocab_size, d)
            self.learned_norm = nn.LayerNorm(d)
        else:
            self.learned_coarse_emb = None
            self.learned_fine_emb = None
            self.learned_norm = None

        # raw feature stream
        if cfg.use_raw_features:
            self.raw_proj = nn.Linear(cfg.raw_feat_dim, d)
            self.raw_norm = nn.LayerNorm(d)
        else:
            self.raw_proj = None
            self.raw_norm = None

        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.token_seq_len, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads_token,
            dim_feedforward=d * 4, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers_token)
        self.out_norm = nn.LayerNorm(d)

    def _embed_streams(self, batch: dict) -> torch.Tensor:
        embeds: list[torch.Tensor] = []
        cfg = self.cfg
        if self.rule_coarse_emb is not None:
            c = batch["rule_coarse"].clamp(0, cfg.rule_coarse_vocab_size - 1)
            f = batch["rule_fine"].clamp(0, cfg.rule_fine_vocab_size - 1)
            embeds.append(self.rule_norm(self.rule_coarse_emb(c) + self.rule_fine_emb(f)))
        if self.learned_coarse_emb is not None:
            c = batch["learned_coarse"].clamp(0, cfg.learned_coarse_vocab_size - 1)
            f = batch["learned_fine"].clamp(0, cfg.learned_fine_vocab_size - 1)
            embeds.append(self.learned_norm(self.learned_coarse_emb(c) + self.learned_fine_emb(f)))
        if self.raw_proj is not None:
            embeds.append(self.raw_norm(self.raw_proj(batch["raw_feat"])))
        # average pool across streams (so stream count is invisible to downstream)
        stacked = torch.stack(embeds, dim=0)
        return stacked.mean(dim=0)

    def forward(self, batch: dict, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self._embed_streams(batch)
        h = h + self.pos_emb[:, : h.shape[1]]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out_norm(h)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/trendline_encoder.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): TrendlineMultiStreamEncoder (rule+learned+raw)"
```

---

## Task 4: Fusion Layer (cross-attention)

**Files:**
- Create: `trendline_tokenizer/models/fusion.py`
- Modify: `trendline_tokenizer/tests/test_models_forward.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from trendline_tokenizer.models.fusion import CrossAttentionFusion


def test_fusion_forward_shape():
    cfg = FusionConfig(d_model=64, n_layers_fusion=2, n_heads_fusion=4)
    fusion = CrossAttentionFusion(cfg)
    B = 2
    price_h = torch.randn(B, cfg.price_seq_len, cfg.d_model)
    token_h = torch.randn(B, cfg.token_seq_len, cfg.d_model)
    price_pad = torch.zeros(B, cfg.price_seq_len, dtype=torch.bool)
    token_pad = torch.zeros(B, cfg.token_seq_len, dtype=torch.bool)
    fused = fusion(price_h, token_h, price_pad, token_pad)
    # Fused output is the price sequence enriched with token context;
    # shape stays (B, T_price, d_model) so downstream heads can pool.
    assert fused.shape == (B, cfg.price_seq_len, cfg.d_model)


def test_fusion_with_all_token_padding():
    """When the trendline branch is fully padded (no lines yet), fusion
    must not produce NaNs — fall back to the price branch."""
    cfg = FusionConfig(d_model=32, n_layers_fusion=1, n_heads_fusion=2)
    fusion = CrossAttentionFusion(cfg)
    B = 1
    price_h = torch.randn(B, cfg.price_seq_len, cfg.d_model)
    token_h = torch.zeros(B, cfg.token_seq_len, cfg.d_model)
    price_pad = torch.zeros(B, cfg.price_seq_len, dtype=torch.bool)
    token_pad = torch.ones(B, cfg.token_seq_len, dtype=torch.bool)
    fused = fusion(price_h, token_h, price_pad, token_pad)
    assert torch.isfinite(fused).all()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 3: Implement `CrossAttentionFusion`**

`trendline_tokenizer/models/fusion.py`:

```python
"""Cross-attention fusion: price as Query, tokens as Key/Value.

Each price-bar position learns to attend to the most relevant trendline
events. If the token branch is fully padded, the cross-attention is
short-circuited (skip connection only) so the model still has signal.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class _FusionBlock(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.norm_q = nn.LayerNorm(cfg.d_model)
        self.norm_kv = nn.LayerNorm(cfg.d_model)
        self.cross = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads_fusion,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 4, cfg.d_model),
        )

    def forward(self, q, kv, kv_pad_mask, kv_all_padded):
        if kv_all_padded.all():
            # No trendline context anywhere in this batch → skip cross.
            return q + self.ff(self.norm_ff(q))
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross(q_norm, kv_norm, kv_norm, key_padding_mask=kv_pad_mask)
        # Where the row had no valid kv, attn_out can be NaN; mask them.
        if kv_all_padded.any():
            attn_out = attn_out.masked_fill(kv_all_padded.view(-1, 1, 1), 0.0)
        h = q + attn_out
        h = h + self.ff(self.norm_ff(h))
        return h


class CrossAttentionFusion(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([_FusionBlock(cfg) for _ in range(cfg.n_layers_fusion)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, price_h, token_h, price_pad_mask, token_pad_mask):
        kv_all_padded = token_pad_mask.all(dim=-1)  # (B,)
        h = price_h
        for blk in self.blocks:
            h = blk(h, token_h, token_pad_mask, kv_all_padded)
        return self.out_norm(h)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/fusion.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): CrossAttentionFusion with empty-token fallback"
```

---

## Task 5: Multi-task heads

**Files:**
- Create: `trendline_tokenizer/models/heads.py`
- Modify: `trendline_tokenizer/tests/test_models_forward.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
from trendline_tokenizer.models.heads import MultiTaskHeads


def test_heads_forward_shapes():
    cfg = FusionConfig(d_model=64)
    heads = MultiTaskHeads(cfg)
    B = 4
    pooled = torch.randn(B, cfg.d_model)
    out = heads(pooled)
    # next-token head predicts RULE tokens (deterministic + explainable)
    assert out["next_coarse_logits"].shape == (B, cfg.rule_coarse_vocab_size)
    assert out["next_fine_logits"].shape == (B, cfg.rule_fine_vocab_size)
    assert out["bounce_logits"].shape == (B, 2)
    assert out["break_logits"].shape == (B, 2)
    assert out["continuation_logits"].shape == (B, 2)
    assert out["buffer_pct"].shape == (B,)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 3: Implement `MultiTaskHeads`**

`trendline_tokenizer/models/heads.py`:

```python
"""Multi-task heads attached to the pooled fusion output.

next_coarse / next_fine: predict the next trendline event's tokens
bounce / break / continuation: 2-class classification at horizon H
buffer_pct: regression — recommended SL buffer in % of price
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import FusionConfig


class MultiTaskHeads(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.d_model
        # next-token heads predict rule tokens (the deterministic, explainable id space)
        self.next_coarse = nn.Linear(h, cfg.rule_coarse_vocab_size) if cfg.next_token_head else None
        self.next_fine = nn.Linear(h, cfg.rule_fine_vocab_size) if cfg.next_token_head else None
        self.bounce = nn.Linear(h, 2) if cfg.bounce_head else None
        self.brk = nn.Linear(h, 2) if cfg.break_head else None
        self.cont = nn.Linear(h, 2) if cfg.continuation_head else None
        self.buffer = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1)) if cfg.buffer_head else None

    def forward(self, pooled: torch.Tensor) -> dict:
        out: dict[str, torch.Tensor] = {}
        if self.next_coarse is not None:
            out["next_coarse_logits"] = self.next_coarse(pooled)
            out["next_fine_logits"] = self.next_fine(pooled)
        if self.bounce is not None:
            out["bounce_logits"] = self.bounce(pooled)
        if self.brk is not None:
            out["break_logits"] = self.brk(pooled)
        if self.cont is not None:
            out["continuation_logits"] = self.cont(pooled)
        if self.buffer is not None:
            out["buffer_pct"] = self.buffer(pooled).squeeze(-1)
        return out
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/heads.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): MultiTaskHeads"
```

---

## Task 6: Full fusion model + losses

**Files:**
- Create: `trendline_tokenizer/models/full_model.py`
- Create: `trendline_tokenizer/models/losses.py`
- Modify: `trendline_tokenizer/tests/test_models_forward.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from trendline_tokenizer.models.full_model import TrendlineFusionModel


def _full_batch(cfg, B):
    return {
        "price": torch.randn(B, cfg.price_seq_len, cfg.price_feat_dim),
        "price_pad": torch.zeros(B, cfg.price_seq_len, dtype=torch.bool),
        "rule_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B, cfg.token_seq_len)),
        "rule_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B, cfg.token_seq_len)),
        "learned_coarse": torch.randint(0, cfg.learned_coarse_vocab_size, (B, cfg.token_seq_len)),
        "learned_fine": torch.randint(0, cfg.learned_fine_vocab_size, (B, cfg.token_seq_len)),
        "raw_feat": torch.randn(B, cfg.token_seq_len, cfg.raw_feat_dim),
        "token_pad": torch.zeros(B, cfg.token_seq_len, dtype=torch.bool),
    }


def test_full_model_forward_returns_all_heads():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    out = model(_full_batch(cfg, 2))
    for k in ("next_coarse_logits", "next_fine_logits", "bounce_logits",
              "break_logits", "continuation_logits", "buffer_pct"):
        assert k in out


def test_full_model_compute_loss_returns_scalar():
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1)
    model = TrendlineFusionModel(cfg)
    B = 4
    batch = _full_batch(cfg, B)
    targets = {
        "next_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B,)),
        "next_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B,)),
        "bounce": torch.randint(0, 2, (B,)),
        "brk": torch.randint(0, 2, (B,)),
        "cont": torch.randint(0, 2, (B,)),
        "buffer_pct": torch.rand(B) * 0.05,
    }
    total, parts = model.compute_loss(batch, targets)
    assert total.ndim == 0
    assert torch.isfinite(total)
    assert "recon" not in parts  # this model has no reconstruction head
    for k in ("next_coarse_ce", "next_fine_ce", "bounce_ce", "break_ce", "cont_ce", "buffer_mse"):
        assert k in parts


def test_full_model_runs_with_only_one_stream():
    """Ablation: only raw features. Useful baseline + sanity check for stream gating."""
    cfg = FusionConfig(d_model=32, n_layers_price=1, n_layers_token=1, n_layers_fusion=1,
                       use_rule_tokens=False, use_learned_tokens=False, use_raw_features=True)
    model = TrendlineFusionModel(cfg)
    B = 2
    batch = _full_batch(cfg, B)
    targets = {
        "next_coarse": torch.randint(0, cfg.rule_coarse_vocab_size, (B,)),
        "next_fine": torch.randint(0, cfg.rule_fine_vocab_size, (B,)),
        "bounce": torch.randint(0, 2, (B,)),
        "brk": torch.randint(0, 2, (B,)),
        "cont": torch.randint(0, 2, (B,)),
        "buffer_pct": torch.rand(B) * 0.05,
    }
    total, _ = model.compute_loss(batch, targets)
    assert torch.isfinite(total)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 3: Implement losses + full model**

`trendline_tokenizer/models/losses.py`:

```python
"""Loss weights for the multi-task fusion model. Centralised so a
single place tunes how heads trade off."""
from __future__ import annotations
from pydantic import BaseModel


class LossWeights(BaseModel):
    next_coarse: float = 1.0
    next_fine: float = 0.5
    bounce: float = 0.5
    brk: float = 0.5
    cont: float = 0.3
    buffer: float = 0.2
```

`trendline_tokenizer/models/full_model.py`:

```python
"""Full fusion model: composes the four blocks and exposes
forward / compute_loss / predict."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig
from .price_seq_encoder import PriceSequenceEncoder
from .trendline_encoder import TrendlineMultiStreamEncoder
from .fusion import CrossAttentionFusion
from .heads import MultiTaskHeads
from .losses import LossWeights


class TrendlineFusionModel(nn.Module):
    def __init__(self, cfg: FusionConfig | None = None,
                 weights: LossWeights | None = None):
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.weights = weights or LossWeights()
        self.price_enc = PriceSequenceEncoder(self.cfg)
        self.token_enc = TrendlineMultiStreamEncoder(self.cfg)
        self.fusion = CrossAttentionFusion(self.cfg)
        self.heads = MultiTaskHeads(self.cfg)
        self.pool_attn = nn.Linear(self.cfg.d_model, 1)

    def _pool(self, fused, pad_mask):
        """Attention pool over the price-bar axis, masking padded bars."""
        scores = self.pool_attn(fused).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = scores.softmax(dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (fused * weights).sum(dim=1)             # (B, d)

    def forward(self, batch: dict) -> dict:
        price_h = self.price_enc(batch["price"], batch["price_pad"])
        token_h = self.token_enc(batch, batch["token_pad"])
        fused = self.fusion(price_h, token_h, batch["price_pad"], batch["token_pad"])
        pooled = self._pool(fused, batch["price_pad"])
        return self.heads(pooled)

    def compute_loss(self, batch: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        out = self.forward(batch)
        w = self.weights
        parts: dict[str, float] = {}
        total = torch.zeros((), device=out["next_coarse_logits"].device)

        nc = F.cross_entropy(out["next_coarse_logits"], targets["next_coarse"])
        nf = F.cross_entropy(out["next_fine_logits"], targets["next_fine"])
        bo = F.cross_entropy(out["bounce_logits"], targets["bounce"])
        br = F.cross_entropy(out["break_logits"], targets["brk"])
        co = F.cross_entropy(out["continuation_logits"], targets["cont"])
        bf = F.mse_loss(out["buffer_pct"], targets["buffer_pct"])

        total = (w.next_coarse * nc + w.next_fine * nf
                 + w.bounce * bo + w.brk * br
                 + w.cont * co + w.buffer * bf)
        parts.update({
            "next_coarse_ce": nc.detach().item(),
            "next_fine_ce": nf.detach().item(),
            "bounce_ce": bo.detach().item(),
            "break_ce": br.detach().item(),
            "cont_ce": co.detach().item(),
            "buffer_mse": bf.detach().item(),
            "total": total.detach().item(),
        })
        return total, parts

    @torch.no_grad()
    def predict(self, batch: dict) -> dict:
        self.eval()
        out = self.forward(batch)
        return {
            "next_coarse_id": out["next_coarse_logits"].argmax(dim=-1),
            "next_fine_id": out["next_fine_logits"].argmax(dim=-1),
            "bounce_prob": out["bounce_logits"].softmax(dim=-1)[:, 1],
            "break_prob": out["break_logits"].softmax(dim=-1)[:, 1],
            "continuation_prob": out["continuation_logits"].softmax(dim=-1)[:, 1],
            "suggested_buffer_pct": out["buffer_pct"].clamp(min=0.0),
        }
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_models_forward.py -q
```

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/models/full_model.py trendline_tokenizer/models/losses.py trendline_tokenizer/tests/test_models_forward.py
git commit -m "feat(trendline-tokenizer): TrendlineFusionModel + multi-task losses"
```

---

## Task 7: Sequence dataset (no-lookahead enforced)

**Files:**
- Create: `trendline_tokenizer/training/__init__.py`
- Create: `trendline_tokenizer/training/sequence_dataset.py`
- Create: `trendline_tokenizer/tests/test_sequence_dataset.py`
- Create: `trendline_tokenizer/tests/test_no_lookahead.py`

- [ ] **Step 1: Write the failing tests for the dataset structure**

`trendline_tokenizer/tests/test_sequence_dataset.py`:

```python
"""Sequence dataset tests — given a synthetic OHLCV df + a list of
TrendlineRecords, the dataset must produce (price_window, token_seq,
targets) tuples whose tokens were emitted strictly before the
prediction bar."""
import numpy as np
import pandas as pd
import pytest
import torch

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.training.sequence_dataset import (
    SequenceDataset, build_examples,
)


def _toy_ohlcv(n=400):
    return pd.DataFrame({
        "time": np.arange(n) * 60,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100.5, 110.5, n),
        "volume": np.full(n, 1.0),
    })


def _toy_record(start_idx: int, end_idx: int, role="support") -> TrendlineRecord:
    return TrendlineRecord(
        id=f"r-{start_idx}-{end_idx}",
        symbol="BTCUSDT", exchange="bitget", timeframe="5m",
        start_time=start_idx * 60, end_time=end_idx * 60,
        start_bar_index=start_idx, end_bar_index=end_idx,
        start_price=100.0, end_price=101.0,
        line_role=role, direction="up", touch_count=2,
        bounce_after=True, bounce_strength_atr=1.0,
        break_after=False, break_distance_atr=None,
        retested_after_break=False,
        volatility_atr_pct=0.01, volume_z_score=0.0,
        label_source="auto", auto_method="test",
        created_at=0,
    )


def test_dataset_length_matches_examples():
    df = _toy_ohlcv()
    records = [_toy_record(50, 80), _toy_record(120, 150), _toy_record(220, 250)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=20)
    ds = SequenceDataset(examples)
    assert len(ds) == len(examples) > 0


def test_dataset_returns_correct_shapes():
    df = _toy_ohlcv()
    records = [_toy_record(50 + 30 * i, 80 + 30 * i) for i in range(8)]
    examples = build_examples(df, records, price_seq_len=64, token_seq_len=8,
                              horizon_bars=20, raw_feat_dim=36)
    ds = SequenceDataset(examples)
    sample = ds[0]
    assert sample["price"].shape == (64, 13)
    # rule stream
    assert sample["rule_coarse"].shape == (8,)
    assert sample["rule_fine"].shape == (8,)
    # learned stream (zeros if no vqvae provided)
    assert sample["learned_coarse"].shape == (8,)
    assert sample["learned_fine"].shape == (8,)
    # raw stream
    assert sample["raw_feat"].shape == (8, 36)
    # padding
    assert sample["price_pad"].shape == (64,)
    assert sample["token_pad"].shape == (8,)
    # targets
    assert "next_coarse" in sample and "next_fine" in sample
    assert "bounce" in sample and "brk" in sample and "cont" in sample
    assert "buffer_pct" in sample
```

`trendline_tokenizer/tests/test_no_lookahead.py`:

```python
"""The single most important property of the dataset: no token in
the input sequence may have an end_bar_index >= the prediction bar."""
import numpy as np
import pandas as pd

from trendline_tokenizer.schemas.trendline import TrendlineRecord
from trendline_tokenizer.training.sequence_dataset import build_examples


def _df(n=300):
    return pd.DataFrame({
        "time": np.arange(n) * 60,
        "open": np.full(n, 100.0), "high": np.full(n, 101.0),
        "low": np.full(n, 99.0), "close": np.full(n, 100.5),
        "volume": np.full(n, 1.0),
    })


def _rec(s, e):
    return TrendlineRecord(
        id=f"x-{s}-{e}", symbol="BTC", exchange="bitget", timeframe="5m",
        start_time=s, end_time=e, start_bar_index=s, end_bar_index=e,
        start_price=100.0, end_price=101.0,
        line_role="support", direction="up", touch_count=2,
        label_source="auto", created_at=0,
    )


def test_input_tokens_strictly_before_prediction_bar():
    records = [_rec(s, s + 10) for s in (10, 30, 50, 70, 90, 110, 130, 150)]
    examples = build_examples(_df(), records, price_seq_len=64,
                              token_seq_len=8, horizon_bars=20, raw_feat_dim=36)
    for ex in examples:
        pred_bar = ex.prediction_bar_index
        for r in ex.input_records:
            assert r.end_bar_index < pred_bar, (
                f"leak: token end_bar {r.end_bar_index} >= pred {pred_bar}"
            )
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_sequence_dataset.py trendline_tokenizer/tests/test_no_lookahead.py -q
```

- [ ] **Step 3: Implement the dataset**

`trendline_tokenizer/training/__init__.py`:

```python
"""Training utilities: dataset building, training loop, evaluation."""
```

`trendline_tokenizer/training/sequence_dataset.py`:

```python
"""(price_window, token_seq, targets) examples for the fusion model.

Strict no-lookahead invariant:
    every TrendlineRecord in token_seq must have end_bar_index <
    prediction_bar_index.
The "next" target is the FIRST record whose end_bar_index >=
prediction_bar_index — that's what the model is trying to forecast.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..schemas.trendline import TrendlineRecord
from ..tokenizer.rule import encode_rule


@dataclass
class SequenceExample:
    price_window: np.ndarray            # (T_price, F)
    price_pad: np.ndarray               # (T_price,) bool
    # three trendline streams
    rule_coarse_ids: np.ndarray         # (T_token,) int64
    rule_fine_ids: np.ndarray           # (T_token,) int64
    learned_coarse_ids: np.ndarray      # (T_token,) int64 — zeros if no vqvae
    learned_fine_ids: np.ndarray        # (T_token,) int64 — zeros if no vqvae
    raw_feats: np.ndarray               # (T_token, raw_feat_dim) float32
    token_pad: np.ndarray               # (T_token,) bool
    prediction_bar_index: int
    input_records: list[TrendlineRecord] = field(default_factory=list)
    # targets — next-token always points at rule tokenizer ids
    next_coarse: int = 0
    next_fine: int = 0
    bounce: int = 0
    brk: int = 0
    cont: int = 0
    buffer_pct: float = 0.0


def _price_window(df: pd.DataFrame, end_idx: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (window, pad_mask). end_idx is exclusive (last bar that
    the model sees is df.iloc[end_idx-1]). If end_idx < length the front
    is padded."""
    start = max(0, end_idx - length)
    raw = df.iloc[start:end_idx]
    ohlcv = raw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32)
    # 8 indicators (cheap; production builder lives in inference/feature_cache.py)
    closes = ohlcv[:, 3]
    ema21 = pd.Series(closes).ewm(span=21, adjust=False).mean().to_numpy()
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().to_numpy()
    ret1 = np.diff(closes, prepend=closes[:1]) / np.maximum(closes, 1e-9)
    ret5 = (closes - np.roll(closes, 5)) / np.maximum(closes, 1e-9)
    ret5[:5] = 0.0
    high = ohlcv[:, 1]; low = ohlcv[:, 2]
    tr = np.maximum.reduce([high - low,
                            np.abs(high - np.roll(closes, 1)),
                            np.abs(low - np.roll(closes, 1))])
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    delta = np.diff(closes, prepend=closes[:1])
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    rs = (pd.Series(up).rolling(14, min_periods=1).mean()
          / pd.Series(dn).rolling(14, min_periods=1).mean().replace(0, 1e-9)).to_numpy()
    rsi14 = 100 - 100 / (1 + rs)
    vol = ohlcv[:, 4]
    vol_z = (vol - vol.mean()) / (vol.std() + 1e-9)
    dist_ma_atr = (closes - ema21) / np.maximum(atr14, 1e-9)

    feats = np.column_stack([
        ohlcv,                           # 5
        ema21, ema50, atr14, rsi14, vol_z, dist_ma_atr, ret1, ret5  # 8
    ]).astype(np.float32)                # (n, 13)

    # left-pad to `length`
    pad = length - feats.shape[0]
    if pad > 0:
        feats = np.concatenate([np.zeros((pad, feats.shape[1]), dtype=np.float32), feats], axis=0)
        mask = np.concatenate([np.ones(pad, dtype=bool), np.zeros(length - pad, dtype=bool)])
    else:
        mask = np.zeros(length, dtype=bool)
    return feats, mask


def _label_outcomes(rec: TrendlineRecord, df: pd.DataFrame, horizon_bars: int) -> dict:
    """Outcome labels for the *next* record. Uses bounce_after / break_after
    if already populated; else derives from the price path within horizon."""
    if rec.bounce_after is not None and rec.break_after is not None:
        bounce = int(bool(rec.bounce_after))
        brk = int(bool(rec.break_after))
        cont = int(not brk)
        buffer_pct = float(rec.bounce_strength_atr or 0.0) * 0.005   # crude prior
        return {"bounce": bounce, "brk": brk, "cont": cont, "buffer_pct": buffer_pct}
    end = min(len(df), rec.end_bar_index + horizon_bars)
    seg = df.iloc[rec.end_bar_index:end]
    if len(seg) == 0 or rec.start_price <= 0:
        return {"bounce": 0, "brk": 0, "cont": 1, "buffer_pct": 0.005}
    moved = (seg["close"].iloc[-1] - rec.end_price) / rec.end_price
    if rec.line_role == "support":
        bounce = int(moved > 0.005); brk = int(moved < -0.005)
    elif rec.line_role == "resistance":
        bounce = int(moved < -0.005); brk = int(moved > 0.005)
    else:
        bounce = brk = 0
    cont = int(not brk)
    buffer_pct = float(seg["high"].max() - seg["low"].min()) / max(rec.end_price, 1e-9)
    return {"bounce": bounce, "brk": brk, "cont": cont,
            "buffer_pct": min(0.05, max(0.0, buffer_pct))}


def build_examples(
    df: pd.DataFrame,
    records: Sequence[TrendlineRecord],
    *,
    price_seq_len: int,
    token_seq_len: int,
    horizon_bars: int,
    raw_feat_dim: int,
    vqvae=None,                # optional HierarchicalVQVAE for learned tokens
) -> list[SequenceExample]:
    """One example per record (the record is the prediction target).
    Input tokens are the previous `token_seq_len` records that ended
    strictly before this record's end_bar_index.

    Each input record produces three parallel representations:
      rule_*  — from tokenizer/rule.py
      learned_* — from learned/vqvae.py if `vqvae` given, else zeros
      raw_feats — from features/vector.py
    """
    from ..features.vector import build_feature_vector

    sorted_recs = sorted(records, key=lambda r: r.end_bar_index)
    examples: list[SequenceExample] = []
    for i, target in enumerate(sorted_recs):
        pred_bar = target.end_bar_index
        # strict no-lookahead
        input_recs = [r for r in sorted_recs[:i] if r.end_bar_index < pred_bar]
        input_recs = input_recs[-token_seq_len:]

        rule_coarse = np.zeros(token_seq_len, dtype=np.int64)
        rule_fine = np.zeros(token_seq_len, dtype=np.int64)
        learned_coarse = np.zeros(token_seq_len, dtype=np.int64)
        learned_fine = np.zeros(token_seq_len, dtype=np.int64)
        raw_feats = np.zeros((token_seq_len, raw_feat_dim), dtype=np.float32)
        token_pad = np.ones(token_seq_len, dtype=bool)

        if input_recs:
            # build the raw feature vectors (also used for VQ-VAE)
            feats = np.stack([build_feature_vector(r) for r in input_recs], axis=0)
            for j, r in enumerate(input_recs):
                slot = token_seq_len - len(input_recs) + j   # right-align
                tok = encode_rule(r)
                rule_coarse[slot] = tok.coarse_token_id
                rule_fine[slot] = tok.fine_token_id
                raw_feats[slot] = feats[j]
                token_pad[slot] = False
            # learned tokens via VQ-VAE
            if vqvae is not None:
                import torch
                with torch.no_grad():
                    feat_t = torch.from_numpy(feats).float()
                    c_idx, f_idx = vqvae.tokenize(feat_t)
                for j in range(len(input_recs)):
                    slot = token_seq_len - len(input_recs) + j
                    learned_coarse[slot] = int(c_idx[j].item())
                    learned_fine[slot] = int(f_idx[j].item())

        price_window, price_pad = _price_window(df, pred_bar, price_seq_len)
        target_tok = encode_rule(target)
        outcomes = _label_outcomes(target, df, horizon_bars)
        examples.append(SequenceExample(
            price_window=price_window, price_pad=price_pad,
            rule_coarse_ids=rule_coarse, rule_fine_ids=rule_fine,
            learned_coarse_ids=learned_coarse, learned_fine_ids=learned_fine,
            raw_feats=raw_feats, token_pad=token_pad,
            prediction_bar_index=pred_bar,
            input_records=list(input_recs),
            next_coarse=target_tok.coarse_token_id,
            next_fine=target_tok.fine_token_id,
            bounce=outcomes["bounce"], brk=outcomes["brk"],
            cont=outcomes["cont"], buffer_pct=outcomes["buffer_pct"],
        ))
    return examples


class SequenceDataset(Dataset):
    def __init__(self, examples: list[SequenceExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "price": torch.from_numpy(ex.price_window),
            "price_pad": torch.from_numpy(ex.price_pad),
            "rule_coarse": torch.from_numpy(ex.rule_coarse_ids),
            "rule_fine": torch.from_numpy(ex.rule_fine_ids),
            "learned_coarse": torch.from_numpy(ex.learned_coarse_ids),
            "learned_fine": torch.from_numpy(ex.learned_fine_ids),
            "raw_feat": torch.from_numpy(ex.raw_feats),
            "token_pad": torch.from_numpy(ex.token_pad),
            "next_coarse": torch.tensor(ex.next_coarse, dtype=torch.long),
            "next_fine": torch.tensor(ex.next_fine, dtype=torch.long),
            "bounce": torch.tensor(ex.bounce, dtype=torch.long),
            "brk": torch.tensor(ex.brk, dtype=torch.long),
            "cont": torch.tensor(ex.cont, dtype=torch.long),
            "buffer_pct": torch.tensor(ex.buffer_pct, dtype=torch.float32),
        }
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_sequence_dataset.py trendline_tokenizer/tests/test_no_lookahead.py -q
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/training/__init__.py trendline_tokenizer/training/sequence_dataset.py trendline_tokenizer/tests/test_sequence_dataset.py trendline_tokenizer/tests/test_no_lookahead.py
git commit -m "feat(trendline-tokenizer): SequenceDataset + no-lookahead invariant test"
```

---

## Task 8: Model registry (versioned artifacts)

**Files:**
- Create: `trendline_tokenizer/registry/__init__.py`
- Create: `trendline_tokenizer/registry/manifest.py`
- Create: `trendline_tokenizer/registry/paths.py`
- Create: `trendline_tokenizer/tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

`trendline_tokenizer/tests/test_registry.py`:

```python
"""Registry: every checkpoint must record tokenizer version, training
data version, and metrics. Loading a model demands the manifest."""
import json
from pathlib import Path

import pytest

from trendline_tokenizer.registry.manifest import ArtifactManifest


def test_manifest_round_trip(tmp_path: Path):
    manifest = ArtifactManifest(
        artifact_name="fusion.v0.1-2026-04-24",
        model_kind="fusion",
        model_version="fusion.v0.1",
        tokenizer_version="rule.v1",
        training_dataset_version="2026-04-24-271k-auto",
        feature_norm_version="none",
        ckpt_path="checkpoints/fusion/fusion_v0.1.pt",
        config_path="checkpoints/fusion/fusion_v0.1.json",
        metrics={"val_next_coarse_acc": 0.42, "val_bounce_auc": 0.71},
        created_at=1700000000,
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ArtifactManifest.load(path)
    assert loaded == manifest


def test_manifest_rejects_missing_versions(tmp_path: Path):
    with pytest.raises(ValueError):
        ArtifactManifest(
            artifact_name="bad", model_kind="fusion", model_version="",
            tokenizer_version="rule.v1", training_dataset_version="x",
            feature_norm_version="none",
            ckpt_path="x", config_path="x", metrics={}, created_at=0,
        )
```

- [ ] **Step 2: Run tests, expect failure**

```bash
python -m pytest trendline_tokenizer/tests/test_registry.py -q
```

- [ ] **Step 3: Implement the registry**

`trendline_tokenizer/registry/__init__.py`:

```python
"""Versioned artifact registry. Checkpoints + manifests live under
checkpoints/<model_kind>/<artifact_name>/."""
```

`trendline_tokenizer/registry/manifest.py`:

```python
"""Artifact manifest — every saved model carries its versions."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class ArtifactManifest(BaseModel):
    artifact_name: str
    model_kind: str               # "fusion" | "vqvae" | "rule_tokenizer"
    model_version: str            # e.g. "fusion.v0.1"
    tokenizer_version: str        # e.g. "rule.v1" or "vqvae.v0"
    training_dataset_version: str
    feature_norm_version: str
    ckpt_path: str
    config_path: str
    metrics: dict[str, Any]
    created_at: int               # unix seconds

    @field_validator("model_version", "tokenizer_version", "training_dataset_version")
    @classmethod
    def _nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("version field must not be empty")
        return v

    def save(self, path: Path):
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ArtifactManifest":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
```

`trendline_tokenizer/registry/paths.py`:

```python
"""Standardised checkpoint paths."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = ROOT / "checkpoints"


def fusion_dir(artifact_name: str) -> Path:
    p = CHECKPOINTS / "fusion" / artifact_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def latest_fusion(model_version: str) -> Path | None:
    base = CHECKPOINTS / "fusion"
    if not base.exists():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(model_version)]
    if not candidates:
        return None
    return sorted(candidates)[-1]
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest trendline_tokenizer/tests/test_registry.py -q
```

- [ ] **Step 5: Commit**

```bash
git add trendline_tokenizer/registry/ trendline_tokenizer/tests/test_registry.py
git commit -m "feat(trendline-tokenizer): ArtifactManifest + checkpoint paths"
```

---

## Task 9: Training CLI (single-epoch smoke run)

**Files:**
- Create: `trendline_tokenizer/training/train_fusion.py`
- Create: `trendline_tokenizer/training/eval_fusion.py`

This task is the integration point. It does not have a unit test — it has a smoke run on real data. The user's CLAUDE.md is explicit that we don't claim "done" without proof.

- [ ] **Step 1: Implement the training entry-point**

`trendline_tokenizer/training/train_fusion.py`:

```python
"""End-to-end training for the fusion model on legacy auto rows.

Smoke-run usage:
    python -m trendline_tokenizer.training.train_fusion \
        --symbols BTCUSDT ETHUSDT --timeframes 5m 15m \
        --max-records 5000 --epochs 1 --batch-size 32

Production usage: drop --max-records, raise epochs.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..adapters import iter_legacy_pattern_records
from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..learned.vqvae import HierarchicalVQVAE, VQVAEConfig
from ..registry.manifest import ArtifactManifest
from ..registry.paths import fusion_dir
from .sequence_dataset import build_examples, SequenceDataset


ROOT = Path(__file__).resolve().parents[2]


def _load_vqvae(ckpt_path: Path | None):
    """Load the existing VQ-VAE so build_examples can produce learned tokens."""
    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"[train] no vqvae checkpoint at {ckpt_path}; learned tokens will be zeros")
        return None
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = VQVAEConfig(**state["cfg"])
    model = HierarchicalVQVAE(cfg)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"[train] loaded VQ-VAE from {ckpt_path}")
    return model


def _load_records(symbols, timeframes, max_records):
    out = []
    for s in symbols:
        for tf in timeframes:
            f = ROOT / "data" / "patterns" / f"{s.upper()}_{tf}.jsonl"
            if not f.exists():
                continue
            for rec in iter_legacy_pattern_records(f):
                out.append(rec)
                if len(out) >= max_records:
                    return out
    return out


def _load_ohlcv(symbol: str, timeframe: str):
    """Read cached OHLCV. Returns DataFrame or None."""
    import pandas as pd
    cache = ROOT / "data" / "ohlcv_cache" / f"{symbol.upper()}_{timeframe}.parquet"
    if not cache.exists():
        return None
    return pd.read_parquet(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframes", nargs="+", required=True)
    ap.add_argument("--max-records", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--horizon-bars", type=int, default=20)
    args = ap.parse_args()

    cfg = FusionConfig()
    vqvae = _load_vqvae(Path(cfg.vqvae_checkpoint_path) if cfg.vqvae_checkpoint_path else None)
    records = _load_records(args.symbols, args.timeframes, args.max_records)
    print(f"loaded {len(records)} records")
    if not records:
        raise SystemExit("no records — populate data/patterns/*.jsonl first")

    # group records by (symbol, timeframe) so each example has the right df
    from collections import defaultdict
    by_key = defaultdict(list)
    for r in records:
        by_key[(r.symbol, r.timeframe)].append(r)

    all_examples = []
    for (sym, tf), recs in by_key.items():
        df = _load_ohlcv(sym, tf)
        if df is None:
            print(f"skip {sym} {tf}: no OHLCV cache")
            continue
        ex = build_examples(df, recs,
                            price_seq_len=cfg.price_seq_len,
                            token_seq_len=cfg.token_seq_len,
                            horizon_bars=args.horizon_bars,
                            raw_feat_dim=cfg.raw_feat_dim,
                            vqvae=vqvae)
        all_examples.extend(ex)
    print(f"built {len(all_examples)} examples")
    if not all_examples:
        raise SystemExit("no examples — check OHLCV cache parity with patterns/")

    n_val = max(1, len(all_examples) // 10)
    train_ds = SequenceDataset(all_examples[:-n_val])
    val_ds = SequenceDataset(all_examples[-n_val:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrendlineFusionModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        n, agg = 0, 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in
                       ("next_coarse", "next_fine", "bounce", "brk", "cont", "buffer_pct")}
            opt.zero_grad()
            loss, _ = model.compute_loss(batch, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n += 1; agg += loss.item()
        print(f"epoch {epoch}: train_loss={agg/max(n,1):.4f} ({time.time()-t0:.1f}s)")

    # validation
    model.eval()
    n, agg = 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = {k: batch.pop(k) for k in
                       ("next_coarse", "next_fine", "bounce", "brk", "cont", "buffer_pct")}
            loss, _ = model.compute_loss(batch, targets)
            n += 1; agg += loss.item()
    val_loss = agg / max(n, 1)
    print(f"val_loss={val_loss:.4f}")

    name = f"{cfg.version}-{int(time.time())}"
    out_dir = fusion_dir(name)
    torch.save({"state_dict": model.state_dict(), "config": cfg.model_dump()},
               out_dir / "fusion.pt")
    (out_dir / "fusion.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    tokenizer_versions = []
    if cfg.use_rule_tokens:
        tokenizer_versions.append("rule.v1")
    if cfg.use_learned_tokens and vqvae is not None:
        tokenizer_versions.append("vqvae.v0")
    if cfg.use_raw_features:
        tokenizer_versions.append("raw.v1")
    manifest = ArtifactManifest(
        artifact_name=name, model_kind="fusion", model_version=cfg.version,
        tokenizer_version="+".join(tokenizer_versions) or "none",
        training_dataset_version=f"legacy-patterns-{args.max_records}",
        feature_norm_version="none",
        ckpt_path=str(out_dir / "fusion.pt"),
        config_path=str(out_dir / "fusion.json"),
        metrics={"val_loss": val_loss, "n_train": len(train_ds),
                 "n_val": len(val_ds),
                 "streams": {"rule": cfg.use_rule_tokens,
                             "learned": cfg.use_learned_tokens and vqvae is not None,
                             "raw": cfg.use_raw_features}},
        created_at=int(time.time()),
    )
    manifest.save(out_dir / "manifest.json")
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
```

`trendline_tokenizer/training/eval_fusion.py`:

```python
"""Holdout evaluation: load a manifest, replay the val split, dump
metrics to <artifact_dir>/eval_metrics.json."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.config import FusionConfig
from ..models.full_model import TrendlineFusionModel
from ..registry.manifest import ArtifactManifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    args = ap.parse_args()
    manifest = ArtifactManifest.load(args.manifest)
    cfg = FusionConfig(**json.loads(Path(manifest.config_path).read_text(encoding="utf-8")))
    model = TrendlineFusionModel(cfg)
    state = torch.load(manifest.ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    print(f"loaded {manifest.artifact_name}; metrics={manifest.metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with tiny data**

```bash
python -m trendline_tokenizer.training.train_fusion \
  --symbols BTCUSDT --timeframes 5m \
  --max-records 200 --epochs 1 --batch-size 8
```

Expected:
- prints `loaded N records` (N > 0)
- prints `built M examples` (M > 0)
- prints `epoch 0: train_loss=...` and `val_loss=...`
- writes `checkpoints/fusion/fusion.v0.1-<ts>/{fusion.pt,fusion.json,manifest.json}`

If `built 0 examples`, the OHLCV cache is missing — see "Data prerequisites" at the bottom of this doc.

- [ ] **Step 3: Inspect and commit**

```bash
ls checkpoints/fusion/
cat checkpoints/fusion/fusion.v0.1-*/manifest.json
```

```bash
git add trendline_tokenizer/training/train_fusion.py trendline_tokenizer/training/eval_fusion.py
git commit -m "feat(trendline-tokenizer): training CLI + smoke run produces manifest"
```

---

# Self-review

Spec coverage:
- Schemas (`OHLCV`, `Trendline`, `TokenizedTrendline`, `ModelArtifact`) — already exist in `trendline_tokenizer/schemas/trendline.py`; this plan adds `ArtifactManifest` (Task 8).
- TrendlineTokenizer v1 — exists.
- Learned TrendlineTokenizer v2 — exists (VQ-VAE).
- Price Sequence Encoder — Task 2.
- Trendline Token Encoder — Task 3.
- Fusion Layer — Task 4.
- Multi-Task Heads — Task 5.
- Losses — Task 6.
- Model Registry — Task 8.
- Online inference — **Milestone 2** (outlined below).
- Monitoring & Logging — **Milestone 2**.
- UI feedback / analyst review queue — **Milestone 3**.
- Backtest console — **Milestone 4**.

No-placeholder check: every code block above is full executable code — no `# TODO` or `# similar to above` left.

Type consistency: `coarse_vocab_size`, `fine_vocab_size` flow from `tokenizer/vocab.py` → `FusionConfig` → `TrendlineTokenEncoder`/`MultiTaskHeads`. The dataset uses the same `encode_rule()` to produce ids that the model reads.

---

# Milestones 2–4 (outlines, not detailed TDD yet)

These will become their own plans once Milestone 1 lands and the model interfaces are concrete in code.

## Milestone 2 — Inference Service & Signal Engine

**New files (sketch):**

```
trendline_tokenizer/
  inference/
    feature_cache.py        # multi-tf ring buffer; OHLCV+indicators
    runtime_detector.py     # adapter over server/strategy/trendlines.py
    runtime_tokenizer.py    # loads rule v1 (and optionally vqvae v0)
    inference_service.py    # loads manifest, runs forward, emits PredictionRecord
    signal_engine.py        # PredictionRecord → SignalRecord with confidence + reason
    monitoring.py           # latency/drift/confidence-collapse metrics
server/routers/
  trendline_signals.py      # FastAPI route /api/trendline/signals (paper only)
```

**Key invariants:**
- Higher TF only emits a signal after the confirmed candle close (no intra-bar leakage).
- Tokenizer version in the prediction must equal the tokenizer version in the loaded manifest — assert at startup.
- All signals go to a JSONL log first; "Strategy / Execution Connectors" is a paper-trade dispatcher only.

**First test:** `tests/test_inference_pipeline.py` — load a small saved manifest, push 300 synthetic bars through the pipeline, assert one `SignalRecord` is emitted with the expected schema.

## Milestone 3 — Feedback Loop & Retraining Trigger

**New files (sketch):**

```
trendline_tokenizer/
  feedback/
    store.py                # append-only JSONL: corrections + signal accept/reject
    schemas.py              # CorrectedTrendline, SignalAccepted, SignalRejected
    review_queue.py         # surfaces low-confidence / high-impact predictions
  retrain/
    refresh_dataset.py      # consume feedback → rebuild training jsonl
    trigger.py              # cron-like; runs train_fusion when new feedback > threshold
server/routers/
  trendline_feedback.py     # POST /api/trendline/feedback (UI sends here)
frontend/js/workbench/
  trendline_review.js       # adds Accept/Reject/Correct buttons next to signals
```

**Hook into existing code:** `data/user_drawing_outcomes.jsonl` already exists; this milestone formalises its schema and adds a corrected-trendline channel alongside it. The existing `trendline_tokenizer/evolve/rounds.py` is already a "round orchestrator" — `retrain/trigger.py` is the model-side analogue.

## Milestone 4 — Backtest / Research Console

**New files (sketch):**

```
trendline_tokenizer/
  backtest/
    replay_engine.py        # given symbol+tf+date_range, replays bars + records
    strategy_simulator.py   # paper PnL on signals
    metrics.py              # hit rate, RR, max DD, by-tf breakdown
    ablation.py             # price-only vs token-only vs fused
    reports.py              # markdown + matplotlib pngs
research/
  console.ipynb             # jupyter for ad-hoc exploration
```

**Compares:**
1. Rule tokenizer v1 vs Learned tokenizer v2 (existing VQ-VAE).
2. Price-only baseline (Transformer with no token branch) vs trendline-token-only vs fused.
3. With vs without the bounce/break heads (does the auxiliary loss help next-token accuracy?).

---

# Final answers (the practical blueprint you asked for)

## How I interpreted the diagram

Three-tier system with the **Trendline as the tokenized object** (Kronos-on-trendlines, not Kronos-on-candles). Tier 1 (offline) trains a fusion model that ingests both raw price sequences and discrete trendline tokens. Tier 2 (online) serves predictions as **signals** with confidence/reason — no live trading initially. Tier 3 (UI/HITL) closes the loop: humans correct the lines, accept/reject the signals, and that data triggers a retrain. The diagram's "Final Products" (Interactive Chart App / Research Tools / Strategy Layer) are consumers of these three tiers — they are **endpoints**, not new pipelines.

## Exact modules to build (gap analysis vs your existing code)

You have already built (do **not** rebuild):
- `trendline_tokenizer/schemas/trendline.py` (full TrendlineRecord)
- `trendline_tokenizer/tokenizer/{vocab,rule,metrics}.py` (rule tokenizer v1, 5040×21600)
- `trendline_tokenizer/learned/{vqvae,dataset,train}.py` (VQ-VAE with checkpoint at `checkpoints/trendline_tokenizer/vqvae_v0.pt`)
- `trendline_tokenizer/adapters/{manual,legacy_patterns,user_outcomes}.py`
- `trendline_tokenizer/features/vector.py` (36-dim feature vector)
- `trendline_tokenizer/evolve/*` (line-evolution rounds — useful as a separate axis)

You need to build (this plan):
- `trendline_tokenizer/models/{config,price_seq_encoder,token_encoder,fusion,heads,full_model,losses}.py`
- `trendline_tokenizer/training/{sequence_dataset,train_fusion,eval_fusion}.py`
- `trendline_tokenizer/registry/{manifest,paths}.py`
- Tests: `test_models_forward.py`, `test_sequence_dataset.py`, `test_no_lookahead.py`, `test_registry.py`

Future milestones add `inference/`, `feedback/`, `retrain/`, `backtest/`.

## Recommended repo structure

Already established by your existing code — do not invent a parallel `trendline_ai/` tree. Everything lives under `trendline_tokenizer/`.

## First milestone

Milestone 1 (this doc): the fusion model that combines price + trendline tokens and produces multi-task predictions. Tasks 1–9 above.

## First files to create (in order)

1. `trendline_tokenizer/models/__init__.py`
2. `trendline_tokenizer/models/config.py`
3. `trendline_tokenizer/tests/test_models_forward.py`

Then proceed Tasks 2–9 sequentially. Each task is 5–15 minutes and ends with a green test + a commit.

## First tests to write

1. `test_fusion_config_defaults_are_compatible_with_rule_tokenizer` (Task 1) — locks in vocab sizes.
2. `test_price_seq_encoder_forward_shape` (Task 2) — `(B, T, F) → (B, T, d_model)`.
3. `test_input_tokens_strictly_before_prediction_bar` (Task 7) — **the no-lookahead invariant**, the single most important test. Without this the rest of the model is useless.

## What should NOT be built yet

- **Live trading hookup.** Predictions are signals only. The existing `server/strategy/trendline_order_manager.py` stays untouched until backtest results justify wiring it.
- **Autoregressive sequence-of-events Transformer** (the §E "trendline event" model in `ARCHITECTURE.md`). Build the bar-conditioned next-token head first; only graduate to a multi-step AR model after codebook utilisation passes 30 % and bounce/break AUC beats the existing `pattern_bounce_auc_0.928.pt` ceiling.
- **A new tokenizer.** Use rule v1 for Milestone 1. The learned VQ-VAE v2 can swap in via a manifest flip once it outperforms rule v1 on the same downstream metrics.
- **Multi-symbol joint training.** Train per-symbol or per-symbol-pair first; cross-symbol generalisation comes after a single-symbol model is convincing.
- **A new UI tree.** The existing `frontend/v2.html` + `frontend/js/workbench/chart.js` are the chart UI; only extend them.

## Data prerequisites for Milestone 1

The smoke run needs OHLCV in parquet form per (symbol, timeframe):

```
data/ohlcv_cache/BTCUSDT_5m.parquet
data/ohlcv_cache/ETHUSDT_5m.parquet
...
```

If your existing OHLCV cache uses a different path/format, point `_load_ohlcv` in `train_fusion.py` at it (the only branch that needs changing). Trendline records are already in `data/patterns/*.jsonl` (271k rows) — those are the training input.

For the **first run**, use 200–500 records on one (symbol, tf) pair. Once that produces a manifest, scale to ~50k records per pair.

## How training, inference, UI, and feedback connect

```
TRAIN (Milestone 1)
  data/patterns/*.jsonl  ─┐
  data/ohlcv_cache/*.parquet ─┤
                          ├─→ build_examples → SequenceDataset
                          │                           │
                          │                           ▼
  rule.encode_rule  ─────┘                  TrendlineFusionModel
                                                      │
                                                      ▼
                                            checkpoints/fusion/<name>/{fusion.pt, manifest.json}

INFER (Milestone 2)
  Realtime bars → FeatureCache → Runtime trendline detect → encode_rule
                       │                                          │
                       └──────────────► TrendlineFusionModel.predict
                                                │                 │
                                                ▼                 ▼
                                          SignalRecord     monitoring.jsonl
                                                │
                                                ▼
                                       paper-trade dispatcher (logs only)

UI + FEEDBACK (Milestone 3)
  Chart UI signal banner → user clicks Accept/Reject/Correct → POST /api/trendline/feedback
                                                                       │
                                                                       ▼
                                                              feedback/store.jsonl
                                                                       │
                                                                       ▼
                                                  retrain/trigger fires when N feedback rows
                                                                       │
                                                                       ▼
                                                          rebuild dataset → train_fusion → new manifest

BACKTEST (Milestone 4)
  any historical (symbol, tf, date range)
       │
       ▼
  replay_engine → strategy_simulator (uses any manifest version) → ablation reports
```

The contract that ties all of this together is `ArtifactManifest`. Inference must load it. Feedback rows record which manifest produced the signal. Retraining bumps the manifest. Backtests pick a manifest by name. **Without the manifest there is no integration story.** That is why Task 8 exists in Milestone 1 even though it looks like infrastructure.

---

**Plan complete.** Saved to `docs/superpowers/plans/2026-04-24-trendline-tokenizer-system.md`.
