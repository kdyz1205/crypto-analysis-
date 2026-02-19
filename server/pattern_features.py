"""
Pattern → Feature → Backtest → Performance.

Turn trendlines from "drawn lines" into computable feature vectors, so we can:
- Define similarity (same-class) between structures
- Backtest with frozen-at-t, no look-ahead
- Compare success rate across similar historical structures
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


# ── 1) Feature vector: 趋势线「变成数字」 ─────────────────────────────────────

@dataclass
class TrendLineFeatures:
    """One trendline as a comparable feature vector."""
    timeframe: str
    n_points: int          # 构成该线的 K 线数量 (x2 - x1 + 1)
    start_time: Any        # 起点时间 (for display/grouping)
    end_time: Any
    slope: float           # Δprice / Δbar (per bar)
    touch_count: int
    volatility_norm: float # slope normalized by ATR (e.g. slope / avg_atr)
    direction: int         # 1 = 上升, -1 = 下降
    length_bars: int
    price_span: float      # |y2 - y1|
    line_type: str         # 'support' | 'resistance'

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "n_points": self.n_points,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "slope": self.slope,
            "touch_count": self.touch_count,
            "volatility_norm": self.volatility_norm,
            "direction": self.direction,
            "length_bars": self.length_bars,
            "price_span": self.price_span,
            "line_type": self.line_type,
        }


def _dict_to_features(d: dict[str, Any]) -> TrendLineFeatures | None:
    """Reconstruct TrendLineFeatures from to_dict() for same-class comparison."""
    if not d or "slope" not in d:
        return None
    return TrendLineFeatures(
        timeframe=str(d.get("timeframe", "")),
        n_points=int(d.get("n_points", 0)),
        start_time=d.get("start_time"),
        end_time=d.get("end_time"),
        slope=float(d["slope"]),
        touch_count=int(d.get("touch_count", 0)),
        volatility_norm=float(d.get("volatility_norm", 0)),
        direction=int(d.get("direction", 0)),
        length_bars=int(d.get("length_bars", 0)),
        price_span=float(d.get("price_span", 0)),
        line_type=str(d.get("line_type", "resistance")),
    )


def extract_features(
    line: Any,  # TrendLine with x1, y1, x2, y2, slope, touches, line_type
    timeframe: str,
    bar_times: list,
    atr_array: np.ndarray | None = None,
) -> TrendLineFeatures | None:
    """
    Convert one TrendLine to a feature vector.
    bar_times[i] = open_time at bar i; atr_array same length as bars.
    """
    x1, x2 = int(line.x1), int(line.x2)
    if x1 < 0 or x2 >= len(bar_times) or x2 <= x1:
        return None
    n_points = x2 - x1 + 1
    length_bars = n_points
    y1, y2 = float(line.y1), float(line.y2)
    price_span = abs(y2 - y1)
    slope = float(line.slope)
    touch_count = int(getattr(line, "touches", 0))
    line_type = getattr(line, "line_type", "resistance")

    # volatility_norm: slope per bar, normalized by ATR so comparable across symbols/timeframes
    avg_atr = np.nan
    if atr_array is not None and x1 < len(atr_array) and x2 < len(atr_array):
        seg = atr_array[x1 : x2 + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) > 0:
            avg_atr = float(np.mean(seg))
    if avg_atr is not None and not np.isnan(avg_atr) and avg_atr > 1e-12:
        volatility_norm = slope / avg_atr
    else:
        volatility_norm = slope / (abs(y1) * 0.01) if abs(y1) > 1e-12 else 0.0

    direction = 1 if slope > 0 else (-1 if slope < 0 else 0)
    start_time = bar_times[x1]
    end_time = bar_times[x2]

    return TrendLineFeatures(
        timeframe=timeframe,
        n_points=n_points,
        start_time=start_time,
        end_time=end_time,
        slope=slope,
        touch_count=touch_count,
        volatility_norm=volatility_norm,
        direction=direction,
        length_bars=length_bars,
        price_span=price_span,
        line_type=line_type,
    )


# ── 2) Similarity: 「相似趋势线」的可计算标准 ───────────────────────────────

DEFAULT_SIMILARITY_WEIGHTS = {
    "slope": 0.3,       # w1: relative slope difference
    "length": 0.25,     # w2: relative length difference
    "n_points": 0.2,    # w3: absolute n_points difference (normalized)
    "volatility_norm": 0.25,  # w4: volatility_norm difference
}


def similarity(a: TrendLineFeatures, b: TrendLineFeatures, weights: dict | None = None) -> float:
    """
    Distance between two trendline feature vectors. 0 = identical; larger = more different.
    Same timeframe is required for meaningful comparison; caller can filter.
    """
    if a.timeframe != b.timeframe:
        return float("inf")
    w = weights or DEFAULT_SIMILARITY_WEIGHTS
    d = 0.0
    # Relative slope difference (avoid div by zero)
    sa, sb = abs(a.slope), abs(b.slope)
    ref_s = max(sa, sb, 1e-12)
    d += w.get("slope", 0.3) * abs(sa - sb) / ref_s
    # Relative length
    ref_l = max(a.length_bars, b.length_bars, 1)
    d += w.get("length", 0.25) * abs(a.length_bars - b.length_bars) / ref_l
    # n_points (normalized by max)
    ref_n = max(a.n_points, b.n_points, 1)
    d += w.get("n_points", 0.2) * abs(a.n_points - b.n_points) / ref_n
    # volatility_norm
    ref_v = max(abs(a.volatility_norm), abs(b.volatility_norm), 1e-12)
    d += w.get("volatility_norm", 0.25) * abs(a.volatility_norm - b.volatility_norm) / ref_v
    return d


def is_same_class(a: TrendLineFeatures, b: TrendLineFeatures, epsilon: float = 0.3) -> bool:
    """Similarity < ε → 认为是「同一类」趋势线."""
    return a.timeframe == b.timeframe and similarity(a, b) < epsilon


# ── 3) Backtest: 冻结在 t，向前看 N 根，严格成功率 ───────────────────────────

# Timeframe → (min_bars, max_bars) for forward look
FORWARD_BARS_BY_TF = {
    "1m": (4, 12),
    "5m": (4, 12),
    "15m": (4, 8),
    "1h": (3, 6),
    "4h": (2, 5),
    "1d": (2, 5),
}


@dataclass
class PatternBacktestConfig:
    """Success/fail definition for trendline backtest."""
    forward_bars_min: int = 4
    forward_bars_max: int = 8
    success_atr_mult: float = 1.0   # 向突破方向达到 k×ATR 算成功
    fail_atr_mult: float = -0.5     # 反向超过此 ATR 算失败（可选）
    require_same_direction: bool = True  # 只统计与趋势线方向一致的突破


def run_trendline_backtest(
    df: pl.DataFrame,
    interval: str,
    get_patterns_at_t: callable,  # (df, replay_idx) -> PatternResult with support_lines, resistance_lines
    config: PatternBacktestConfig | None = None,
) -> dict[str, Any]:
    """
    At each t (with enough history and future bars), freeze structure from [t-n, t],
    get trendlines, then look forward N bars (timeframe-bound). No look-ahead.
    """
    cfg = config or PatternBacktestConfig()
    n = len(df)
    try:
        from sr_patterns import calculate_atr
        if "atr" not in df.columns:
            df = calculate_atr(df, 14)
    except Exception:
        pass
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    closes = df["close"].to_numpy().astype(float)
    atr = df["atr"].to_numpy().astype(float)
    bar_times = df["open_time"].to_list()

    fb_min, fb_max = FORWARD_BARS_BY_TF.get(interval, (4, 8))
    warmup = 100  # avoid early bars with too few points
    signals = []
    for t in range(warmup, n - fb_max - 1):
        result = get_patterns_at_t(df, t)
        if result is None:
            continue
        for line in (result.support_lines or []) + (result.resistance_lines or []):
            feats = extract_features(line, interval, bar_times, atr)
            if feats is None:
                continue
            # Forward: next fb_max bars
            entry_price = closes[t]
            atr_t = atr[t] if t < len(atr) and not np.isnan(atr[t]) else (entry_price * 0.01)
            success_thresh = entry_price + feats.direction * cfg.success_atr_mult * atr_t
            exit_idx = min(t + fb_max, n - 1)
            exit_price = closes[exit_idx]
            # Simple success: price moved in direction of line by at least success_atr_mult * ATR
            if feats.direction != 0:
                hit_success = (feats.direction == 1 and exit_price >= success_thresh) or (
                    feats.direction == -1 and exit_price <= success_thresh
                )
            else:
                hit_success = abs(exit_price - entry_price) >= cfg.success_atr_mult * atr_t
            ret_pct = (exit_price - entry_price) / entry_price * 100
            signals.append({
                "t": t,
                "start_time": str(bar_times[t]),
                "line_type": feats.line_type,
                "direction": feats.direction,
                "slope": feats.slope,
                "n_points": feats.n_points,
                "volatility_norm": feats.volatility_norm,
                "entry": entry_price,
                "exit": exit_price,
                "return_pct": round(ret_pct, 4),
                "success": hit_success,
                "forward_bars": exit_idx - t,
                "x1": int(line.x1),
                "y1": float(line.y1),
                "x2": int(line.x2),
                "y2": float(line.y2),
            })

    if not signals:
        return {"interval": interval, "total_signals": 0, "signals": [], "success_rate_pct": None}
    success_count = sum(1 for s in signals if s["success"])
    return {
        "interval": interval,
        "total_signals": len(signals),
        "success_rate_pct": round(success_count / len(signals) * 100, 1),
        "signals": signals,
        "sample_signals": signals[:30],
        "by_line_type": _aggregate_by(signals, "line_type"),
        "by_direction": _aggregate_by(signals, "direction"),
    }


def _aggregate_by(signals: list[dict], key: str) -> dict[str, dict]:
    from collections import defaultdict
    groups = defaultdict(list)
    for s in signals:
        k = s.get(key, "?")
        groups[str(k)].append(s)
    out = {}
    for k, lst in groups.items():
        n_ok = sum(1 for x in lst if x.get("success"))
        out[k] = {"count": len(lst), "success_rate_pct": round(n_ok / len(lst) * 100, 1)}
    return out


def user_line_to_features(x1: int, y1: float, x2: int, y2: float, timeframe: str, bar_times: list, atr: Any = None) -> TrendLineFeatures | None:
    """Build TrendLineFeatures from user-drawn (x1,y1,x2,y2) for similarity search."""
    if x2 <= x1 or x1 < 0 or x2 >= len(bar_times):
        return None
    slope = (y2 - y1) / (x2 - x1)
    try:
        from types import SimpleNamespace
        line = SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2, slope=slope, touches=0, line_type="support")
        return extract_features(line, timeframe, bar_times, atr)
    except Exception:
        return None


def _signal_to_features(s: dict, timeframe: str) -> TrendLineFeatures:
    """Build minimal TrendLineFeatures from a backtest signal for similarity()."""
    return TrendLineFeatures(
        timeframe=timeframe,
        n_points=int(s.get("n_points", 0)),
        start_time=s.get("start_time", ""),
        end_time=s.get("start_time", ""),
        slope=float(s.get("slope", 0)),
        touch_count=0,
        volatility_norm=float(s.get("volatility_norm", 0)),
        direction=int(s.get("direction", 0)),
        length_bars=int(s.get("n_points", 0)),
        price_span=0.0,
        line_type=str(s.get("line_type", "resistance")),
    )


def current_vs_history(
    current_features: list[TrendLineFeatures],
    historical_signals: list[dict],
    timeframe: str,
    epsilon: float = 0.35,
) -> dict[str, Any]:
    """
    For each current trendline feature, find historical signals in the same class (similarity < epsilon)
    and compute their success rate. Returns per-line stats and overall aggregate.
    """
    if not historical_signals:
        n = len(current_features)
        return {
            "current": [
                {"feature": f.to_dict(), "similar_count": 0, "success_rate_pct": None, "similar_line_indices": []}
                for f in current_features
            ],
            "overall_similar_count": 0,
            "overall_success_rate_pct": None,
        }
    out_current = []
    all_similar_success = []
    for feat in current_features:
        similar = []
        for s in historical_signals:
            try:
                s_feat = _signal_to_features(s, timeframe)
                if is_same_class(feat, s_feat, epsilon):
                    similar.append(s)
            except Exception:
                continue
        if similar:
            n_ok = sum(1 for x in similar if x.get("success"))
            rate = round(n_ok / len(similar) * 100, 1)
            out_current.append({
                "feature": feat.to_dict(),
                "similar_count": len(similar),
                "success_rate_pct": rate,
            })
            all_similar_success.extend([x.get("success") for x in similar])
        else:
            out_current.append({
                "feature": feat.to_dict(),
                "similar_count": 0,
                "success_rate_pct": None,
            })

    # Per-line: which other current lines are in the same class (for frontend highlighting)
    features_list = [c["feature"] for c in out_current]
    for i, feat_dict in enumerate(features_list):
        fi = _dict_to_features(feat_dict)
        similar_indices = []
        if fi:
            for j in range(len(features_list)):
                if i == j:
                    continue
                fj = _dict_to_features(features_list[j])
                if fj and is_same_class(fi, fj, epsilon):
                    similar_indices.append(j)
        out_current[i]["similar_line_indices"] = similar_indices

    overall_rate = None
    if all_similar_success:
        overall_rate = round(sum(all_similar_success) / len(all_similar_success) * 100, 1)
    return {
        "current": out_current,
        "overall_similar_count": len(all_similar_success),
        "overall_success_rate_pct": overall_rate,
    }
