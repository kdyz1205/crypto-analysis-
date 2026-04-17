"""PyTorch model gate for live trendline orders.

The two checkpoints have different jobs:
  - primary_trade_model: V3 passive-limit trade outcome model (AUC ~= 0.825).
  - aux_line_quality_model: manual/pattern line-quality model (AUC ~= 0.928).

Live decision order is intentionally:
  1. Aux line-quality filter.
  2. Primary trade-outcome filter.

If a model cannot be loaded, the gate fails open by default and annotates the
reason. That keeps live scanning from crashing because of a local checkpoint or
torch environment issue.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY_TRADE_MODEL = Path(
    os.environ.get(
        "TRENDLINE_PRIMARY_TRADE_MODEL",
        "C:/Users/alexl/trading-system/checkpoints/trendline_quality/v3_trade_outcome_auc_0.825.pt",
    )
)
AUX_LINE_QUALITY_MODEL = Path(
    os.environ.get(
        "TRENDLINE_AUX_LINE_QUALITY_MODEL",
        "C:/Users/alexl/Desktop/crypto-analysis-/checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt",
    )
)

TF_CODE = {"5m": 0.0, "15m": 1.0, "1h": 2.0, "4h": 3.0, "1d": 4.0}


@dataclass
class ModelGateResult:
    accepted: bool
    reason: str
    trade_win_prob: float | None = None
    trade_pred_pnl: float | None = None
    line_quality_prob: float | None = None
    line_quality_value: float | None = None
    primary_trade_model: str = str(PRIMARY_TRADE_MODEL)
    aux_line_quality_model: str = str(AUX_LINE_QUALITY_MODEL)


@dataclass
class _LoadedModel:
    kind: str
    path: Path
    mtime: float
    model: Any
    mean: np.ndarray
    scale: np.ndarray
    symbols: list[str]


_MODEL_CACHE: dict[str, _LoadedModel | None] = {}
_MODEL_CACHE_CHECK_TS: dict[str, float] = {}
_CACHE_TTL_SECONDS = 30.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    out = np.full(len(close), np.nan, dtype=float)
    for i in range(period - 1, len(close)):
        out[i] = np.nanmean(tr[i - period + 1 : i + 1])
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return out
    csum = np.cumsum(np.insert(values, 0, 0.0))
    out[period - 1 :] = (csum[period:] - csum[:-period]) / period
    return out


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < period:
        return out
    for i in range(period - 1, len(values)):
        out[i] = np.std(values[i - period + 1 : i + 1])
    return out


def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) <= period:
        return 50.0
    delta = np.diff(close[-(period + 1) :])
    gain = np.maximum(delta, 0).mean()
    loss = np.maximum(-delta, 0).mean()
    if loss <= 1e-12:
        return 100.0
    rs = gain / loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _trend_context(close: np.ndarray) -> str:
    if len(close) < 60:
        return "range"
    ema21 = _ema(close, 21)[-1]
    ema55 = _ema(close, 55)[-1]
    if ema21 > ema55:
        return "uptrend"
    if ema21 < ema55:
        return "downtrend"
    return "range"


def _load_model(kind: str, path: Path) -> _LoadedModel | None:
    now = time.time()
    cached = _MODEL_CACHE.get(kind)
    last_check = _MODEL_CACHE_CHECK_TS.get(kind, 0.0)
    if now - last_check < _CACHE_TTL_SECONDS:
        return cached
    _MODEL_CACHE_CHECK_TS[kind] = now

    if not path.exists():
        _MODEL_CACHE[kind] = None
        return None

    mtime = path.stat().st_mtime
    if cached is not None and cached.mtime == mtime:
        return cached

    try:
        import torch
        import torch.nn as nn

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if kind == "trade":
            class TradeOutcomeNet(nn.Module):
                def __init__(self, n_features: int, hidden_dims: list[int], pnl_scale: float):
                    super().__init__()
                    layers = []
                    prev = n_features
                    for dim in hidden_dims:
                        layers += [nn.Linear(prev, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(0.0)]
                        prev = dim
                    self.trunk = nn.Sequential(*layers)
                    self.cls_head = nn.Sequential(nn.Linear(prev, 16), nn.GELU(), nn.Dropout(0.0), nn.Linear(16, 1))
                    self.reg_head = nn.Sequential(nn.Linear(prev, 16), nn.GELU(), nn.Dropout(0.0), nn.Linear(16, 1))
                    self.pnl_scale = pnl_scale

                def forward(self, x):
                    h = self.trunk(x)
                    return self.cls_head(h).squeeze(-1), torch.tanh(self.reg_head(h)).squeeze(-1) * self.pnl_scale

            cfg = ckpt.get("config") or {}
            n_features = int(ckpt.get("n_features") or 24)
            model = TradeOutcomeNet(
                n_features=n_features,
                hidden_dims=list(cfg.get("hidden_dims") or [128, 64, 32]),
                pnl_scale=float(cfg.get("pnl_scale") or 0.10),
            )
            model.load_state_dict(ckpt["model_state"])
            mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
            scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
            symbols: list[str] = []
        else:
            class LineQualityNet(nn.Module):
                def __init__(self, n_features: int):
                    super().__init__()
                    self.backbone = nn.Sequential(
                        nn.Linear(n_features, 96),
                        nn.ReLU(),
                        nn.Dropout(0.0),
                        nn.Linear(96, 48),
                        nn.ReLU(),
                    )
                    self.cls_head = nn.Linear(48, 1)
                    self.reg_head = nn.Linear(48, 1)

                def forward(self, x):
                    h = self.backbone(x)
                    return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)

            mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
            scale = np.array(ckpt.get("scaler_std", ckpt.get("scaler_scale")), dtype=np.float32)
            model = LineQualityNet(len(mean))
            model.load_state_dict(ckpt["model_state_dict"])
            symbols = [str(s).upper() for s in ckpt.get("symbols", [])]

        model.eval()
        loaded = _LoadedModel(kind=kind, path=path, mtime=mtime, model=model, mean=mean, scale=scale, symbols=symbols)
        _MODEL_CACHE[kind] = loaded
        return loaded
    except Exception as exc:
        print(f"[trendline_model_gate] failed to load {kind} model {path}: {exc}", flush=True)
        _MODEL_CACHE[kind] = None
        return None


def _score(loaded: _LoadedModel, features: list[float]) -> tuple[float, float]:
    import torch

    x = np.array(features, dtype=np.float32)
    if len(x) != len(loaded.mean):
        raise ValueError(f"{loaded.kind} feature length {len(x)} != model length {len(loaded.mean)}")
    scale = np.where(np.abs(loaded.scale) < 1e-12, 1.0, loaded.scale)
    x = np.nan_to_num((x - loaded.mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        logits, reg = loaded.model(torch.FloatTensor(x[None, :]))
        prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)[0]
        value = reg.cpu().numpy().reshape(-1)[0]
    return float(prob), float(value)


def _market_values(bars: dict) -> dict[str, float]:
    high = np.asarray(bars["h"], dtype=float)
    low = np.asarray(bars["l"], dtype=float)
    close = np.asarray(bars["c"], dtype=float)
    vol = np.asarray(bars.get("v", np.ones_like(close)), dtype=float)
    if len(close) < 2:
        return {}

    atr_arr = _atr(high, low, close)
    atr = _safe_float(atr_arr[-1], max(float(close[-1]) * 0.01, 1e-9))
    bb_ma = _rolling_mean(close, 21)
    bb_sd = _rolling_std(close, 21)
    bb_upper = bb_ma + 2.1 * bb_sd
    bb_lower = bb_ma - 2.1 * bb_sd
    bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_ma[-1] if math.isfinite(bb_ma[-1]) and abs(bb_ma[-1]) > 1e-12 else 0.0
    bb_pct = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if math.isfinite(bb_upper[-1] - bb_lower[-1]) and abs(bb_upper[-1] - bb_lower[-1]) > 1e-12 else 0.5
    vol_ma = _rolling_mean(vol, 20)
    vol_ratio = vol[-1] / vol_ma[-1] if math.isfinite(vol_ma[-1]) and vol_ma[-1] > 0 else 1.0
    ema5 = _ema(close, 5)
    ema8 = _ema(close, 8)
    ema21 = _ema(close, 21)
    ema55 = _ema(close, 55)
    ribbon_spread = abs(ema5[-1] - ema55[-1]) / close[-1] if close[-1] > 0 else 0.0
    bull = ema5[-1] > ema8[-1] > ema21[-1] > ema55[-1]
    bear = ema5[-1] < ema8[-1] < ema21[-1] < ema55[-1]
    ribbon_score = 1.0 if bull else (-1.0 if bear else 0.0)
    log_ret = np.diff(np.log(np.maximum(close, 1e-12)))
    realized_vol = float(np.std(log_ret[-20:])) if len(log_ret) >= 20 else 0.0

    def ret(horizon: int) -> float:
        if len(close) <= horizon or close[-horizon - 1] <= 0:
            return 0.0
        return float(close[-1] / close[-horizon - 1] - 1.0)

    return {
        "close": float(close[-1]),
        "atr": float(atr),
        "atr_pct": float(atr / close[-1]) if close[-1] > 0 else 0.0,
        "bb_width": float(bb_width),
        "bb_pct": float(bb_pct),
        "vol_ratio": float(vol_ratio),
        "ribbon_spread": float(ribbon_spread),
        "ribbon_score": float(ribbon_score),
        "realized_vol": realized_vol,
        "ret_1": ret(1),
        "ret_4": ret(4),
        "ret_12": ret(12),
        "ret_24": ret(24),
        "rsi": _rsi(close),
        "ma_distance_atr": float((close[-1] - ema21[-1]) / atr) if atr > 0 else 0.0,
        "context": _trend_context(close),
    }


def _trade_features(symbol: str, tf: str, kind: str, line_info: dict, bars: dict) -> list[float]:
    m = _market_values(bars)
    current_bar = len(bars["c"]) - 1
    i1 = int(line_info.get("i1") or line_info.get("anchor1_bar") or 0)
    i2 = int(line_info.get("i2") or line_info.get("anchor2_bar") or current_bar)
    p1 = _safe_float(line_info.get("p1") or line_info.get("anchor1_price"))
    p2 = _safe_float(line_info.get("p2") or line_info.get("anchor2_price"))
    slope = _safe_float(line_info.get("slope"))
    atr = max(_safe_float(m.get("atr")), 1e-9)
    close = _safe_float(m.get("close"))
    line_at_bar = slope * current_bar + _safe_float(line_info.get("intercept"))

    timestamp = time.time()
    hour = (timestamp / 3600.0) % 24.0
    dow = int(timestamp / 86400.0 + 4) % 7
    anchor_dist_bars = max(0, i2 - i1)
    anchor_dist_pct = (p2 - p1) / p1 if p1 > 0 else 0.0
    max_bounce = 0.0
    try:
        high = np.asarray(bars["h"], dtype=float)
        low = np.asarray(bars["l"], dtype=float)
        if kind == "support":
            max_bounce = float(np.nanmax(high[max(i2, 0) :] - line_at_bar) / atr)
        else:
            max_bounce = float(np.nanmax(line_at_bar - low[max(i2, 0) :]) / atr)
    except Exception:
        max_bounce = 0.0

    return [
        1.0 if kind == "support" else 0.0,
        math.log1p(anchor_dist_bars),
        anchor_dist_pct,
        slope / atr,
        0.0,
        0.0,
        math.log1p(max(0, current_bar - i2)),
        max_bounce,
        _safe_float(m.get("atr_pct")),
        _safe_float(m.get("bb_width")),
        _safe_float(m.get("bb_pct"), 0.5),
        _safe_float(m.get("vol_ratio"), 1.0),
        _safe_float(m.get("ribbon_spread")),
        _safe_float(m.get("ribbon_score")),
        _safe_float(m.get("realized_vol")),
        _safe_float(m.get("ret_1")),
        _safe_float(m.get("ret_4")),
        _safe_float(m.get("ret_12")),
        _safe_float(m.get("ret_24")),
        math.sin(2 * math.pi * hour / 24.0),
        math.cos(2 * math.pi * hour / 24.0),
        math.sin(2 * math.pi * dow / 7.0),
        math.cos(2 * math.pi * dow / 7.0),
        (close - line_at_bar) / atr if atr > 0 else 0.0,
    ]


def _aux_features(symbol: str, tf: str, kind: str, line_info: dict, bars: dict, loaded: _LoadedModel | None) -> list[float]:
    m = _market_values(bars)
    current_bar = len(bars["c"]) - 1
    i1 = int(line_info.get("i1") or line_info.get("anchor1_bar") or 0)
    i2 = int(line_info.get("i2") or line_info.get("anchor2_bar") or current_bar)
    p1 = max(_safe_float(line_info.get("p1") or line_info.get("anchor1_price")), 1e-12)
    p2 = max(_safe_float(line_info.get("p2") or line_info.get("anchor2_price")), 1e-12)
    anchor_gap = max(0, i2 - i1)
    anchor_distance_pct = (p2 - p1) / p1
    anchor_slope_pct = anchor_distance_pct / max(anchor_gap, 1)
    detected_at = float(current_bar)
    time_position = current_bar / max(len(bars["c"]) - 1, 1)
    rsi = _safe_float(m.get("rsi"), 50.0)
    volatility = _safe_float(m.get("realized_vol"))
    context = str(m.get("context") or "range")
    symbol_index = -1.0
    if loaded and loaded.symbols:
        try:
            symbol_index = loaded.symbols.index(symbol.upper()) / max(len(loaded.symbols) - 1, 1)
        except ValueError:
            symbol_index = -1.0

    return [
        _safe_float(line_info.get("slope")) / max(_safe_float(m.get("atr")), 1e-9),
        float(anchor_gap),
        volatility,
        rsi,
        _safe_float(m.get("ma_distance_atr")),
        0.75,
        float(anchor_gap),
        math.log1p(anchor_gap),
        anchor_distance_pct,
        anchor_slope_pct,
        math.log(p1),
        math.log(p2),
        detected_at / max(detected_at + 1000.0, 1.0),
        min(max(time_position, 0.0), 1.0),
        TF_CODE.get(tf, -1.0),
        1.0 if kind == "support" else 0.0,
        1.0 if kind == "resistance" else 0.0,
        1.0 if context == "downtrend" else 0.0,
        1.0 if context == "range" else 0.0,
        1.0 if context == "uptrend" else 0.0,
        symbol_index,
        1.0 if volatility >= 0.03 else 0.0,
        1.0 if rsi >= 70.0 else 0.0,
        1.0 if rsi <= 30.0 else 0.0,
    ]


def score_trendline_gate(
    *,
    symbol: str,
    timeframe: str,
    kind: str,
    line_info: dict,
    bars: dict,
    cfg: dict,
) -> ModelGateResult:
    gate_cfg = cfg.get("model_gate") or {}
    if not gate_cfg.get("enabled", True):
        return ModelGateResult(accepted=True, reason="disabled")

    min_trade = float(gate_cfg.get("min_trade_win_prob", 0.50))
    min_line = float(gate_cfg.get("min_line_quality_prob", 0.50))
    fail_open = bool(gate_cfg.get("fail_open", True))
    primary_path = Path(gate_cfg.get("primary_trade_model") or PRIMARY_TRADE_MODEL)
    aux_path = Path(gate_cfg.get("aux_line_quality_model") or AUX_LINE_QUALITY_MODEL)

    result = ModelGateResult(
        accepted=True,
        reason="accepted",
        primary_trade_model=str(primary_path),
        aux_line_quality_model=str(aux_path),
    )

    trade_model = _load_model("trade", primary_path)
    aux_model = _load_model("aux", aux_path)
    missing = []

    try:
        if aux_model is not None:
            p, v = _score(aux_model, _aux_features(symbol, timeframe, kind, line_info, bars, aux_model))
            result.line_quality_prob = p
            result.line_quality_value = v
            if p < min_line:
                result.accepted = False
                result.reason = f"line_quality_prob {p:.3f} < {min_line:.3f}"
                return result
        else:
            missing.append("aux_line_quality_model")

        if trade_model is not None:
            p, v = _score(trade_model, _trade_features(symbol, timeframe, kind, line_info, bars))
            result.trade_win_prob = p
            result.trade_pred_pnl = v
            if p < min_trade:
                result.accepted = False
                result.reason = f"trade_win_prob {p:.3f} < {min_trade:.3f}"
                return result
        else:
            missing.append("primary_trade_model")

        if missing:
            result.reason = f"fail_open_missing:{','.join(missing)}" if fail_open else f"missing:{','.join(missing)}"
            result.accepted = fail_open
        return result
    except Exception as exc:
        result.reason = f"fail_open_error:{exc}" if fail_open else f"model_error:{exc}"
        result.accepted = fail_open
        return result


def result_dict(result: ModelGateResult) -> dict:
    return asdict(result)
