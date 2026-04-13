"""Pattern Engine — historical pattern matching for 2-touch trendlines.

Core idea: Don't wait for 3-touch confirmation. Instead:
1. Any 2-touch line is a HYPOTHESIS
2. Extract a feature vector describing the structure
3. Search historical data for SIMILAR structures
4. Compute what ACTUALLY happened next (did 3rd touch occur? bounce? break?)
5. Output probabilities → expected value calculation

This turns S/R from rule-based to evidence-based.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

DATA_ROOT = Path(__file__).parent.parent / "data" / "patterns"


# ── Feature Vector ──────────────────────────────────────────────────────

@dataclass
class PatternFeatures:
    """Normalized feature vector describing a 2-touch trendline structure."""
    slope_atr: float = 0.0          # slope per bar, normalized by ATR
    length_bars: int = 0            # bars between anchor1 and anchor2
    volatility: float = 0.0         # ATR / close at anchor2
    trend_context: str = "range"    # uptrend | downtrend | range
    side: str = ""                  # support | resistance
    rsi: float = 50.0               # RSI at anchor2
    ma_distance_atr: float = 0.0    # (close - EMA50) / ATR
    # Round 1/10 #2: touch_quality was hardcoded to 1.0 (constant column),
    # contributing 0 variance to KNN distance and 0 correlation in
    # learn_feature_weights → 0 weight. Dead feature dim. Field kept for
    # backward-compat on serialized records but NOT included in to_vector.
    touch_quality: float = 1.0
    symbol: str = ""
    timeframe: str = ""

    def to_vector(self) -> list[float]:
        """Convert to numerical vector for distance calculations.
        Categorical fields are encoded as multiple dims.

        NOTE: touch_quality was removed from the vector (it was a constant
        and added zero signal). The vector is now 9-dim, not 10. Anything
        that hardcodes the dim count must be updated in lockstep — see
        _NUMERICAL_IDX / _CATEGORICAL_IDX below.
        """
        trend_up = 1.0 if self.trend_context == "uptrend" else 0.0
        trend_down = 1.0 if self.trend_context == "downtrend" else 0.0
        trend_range = 1.0 if self.trend_context == "range" else 0.0
        support = 1.0 if self.side == "support" else 0.0
        return [
            self.slope_atr,
            math.log1p(self.length_bars),
            self.volatility,
            trend_up,
            trend_down,
            trend_range,
            support,
            self.rsi / 100.0,
            self.ma_distance_atr,
        ]


@dataclass
class PatternOutcome:
    """What actually happened after the 2-touch structure was formed."""
    third_touch: bool = False       # did a 3rd confirming touch occur within lookahead?
    third_touch_bars_later: int = -1
    bounced: bool = False           # did price bounce > 1 ATR from line within lookahead?
    bounce_magnitude_atr: float = 0.0
    broke: bool = False             # did price break through line decisively?
    break_bars_later: int = -1
    fake_break: bool = False        # broke then returned within N bars
    max_return_atr: float = 0.0     # best move in the favorable direction
    max_drawdown_atr: float = 0.0   # worst move against


@dataclass
class PatternRecord:
    """One historical pattern instance with features + outcome."""
    pattern_id: str = ""
    features: PatternFeatures = field(default_factory=PatternFeatures)
    outcome: PatternOutcome = field(default_factory=PatternOutcome)
    anchor1_idx: int = 0
    anchor2_idx: int = 0
    anchor1_price: float = 0.0
    anchor2_price: float = 0.0
    detected_at_bar: int = 0
    # Time-based split: which slice of history does this pattern belong to
    # Used to prevent future-leakage during KNN: match current pattern only against earlier splits.
    split_bucket: str = ""      # "train" | "val" | "test"
    time_position: float = 0.0  # 0.0 = oldest, 1.0 = newest


# ── Feature Extraction ─────────────────────────────────────────────────

def extract_features(
    df,
    anchor1_idx: int,
    anchor1_price: float,
    anchor2_idx: int,
    anchor2_price: float,
    side: str,
    symbol: str = "",
    timeframe: str = "",
) -> PatternFeatures:
    """Extract a feature vector from a 2-touch structure."""
    from .types import new_id  # noqa
    import numpy as np
    import pandas as pd

    # ATR at anchor2
    atr = _calc_atr(df, period=14)
    atr_val = float(atr.iloc[anchor2_idx]) if anchor2_idx < len(atr) else 0.01
    atr_val = max(atr_val, 0.01)

    length = anchor2_idx - anchor1_idx
    if length <= 0:
        return PatternFeatures(symbol=symbol, timeframe=timeframe, side=side)

    # Slope per bar, normalized by ATR
    slope_per_bar = (anchor2_price - anchor1_price) / length
    slope_atr = slope_per_bar / atr_val

    close2 = float(df.iloc[anchor2_idx]["close"])
    volatility = atr_val / close2 if close2 > 0 else 0.0

    # Trend context via EMA50
    ema = _calc_ema(df["close"], 50)
    ema_val = float(ema.iloc[anchor2_idx]) if anchor2_idx < len(ema) else close2
    ma_distance_atr = (close2 - ema_val) / atr_val

    # Trend classification: look at EMA slope + price position
    ema_start = float(ema.iloc[max(0, anchor2_idx - 20)])
    ema_slope_pct = (ema_val - ema_start) / ema_start if ema_start > 0 else 0.0
    if ema_slope_pct > 0.02 and close2 > ema_val:
        trend = "uptrend"
    elif ema_slope_pct < -0.02 and close2 < ema_val:
        trend = "downtrend"
    else:
        trend = "range"

    # RSI at anchor2
    rsi = _calc_rsi(df["close"], 14)
    rsi_val = float(rsi.iloc[anchor2_idx]) if anchor2_idx < len(rsi) and not pd.isna(rsi.iloc[anchor2_idx]) else 50.0

    return PatternFeatures(
        slope_atr=round(slope_atr, 4),
        length_bars=int(length),
        volatility=round(volatility, 4),
        trend_context=trend,
        side=side,
        rsi=round(rsi_val, 1),
        ma_distance_atr=round(ma_distance_atr, 2),
        touch_quality=1.0,
        symbol=symbol,
        timeframe=timeframe,
    )


def _calc_atr(df, period: int = 14):
    """Simple ATR via rolling mean of true range."""
    import pandas as pd
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def _calc_ema(series, period: int):
    return series.ewm(span=period, adjust=False).mean()


def _calc_rsi(series, period: int = 14):
    import pandas as pd
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


# ── Outcome Observation ────────────────────────────────────────────────

def observe_outcome(
    df,
    anchor1_idx: int,
    anchor1_price: float,
    anchor2_idx: int,
    anchor2_price: float,
    side: str,
    lookahead_bars: int = 50,
) -> PatternOutcome:
    """Given a 2-touch line, scan forward N bars and see what happened."""
    length = anchor2_idx - anchor1_idx
    if length <= 0:
        return PatternOutcome()

    slope = (anchor2_price - anchor1_price) / length
    total_bars = len(df)
    end_idx = min(anchor2_idx + lookahead_bars, total_bars - 1)

    atr = _calc_atr(df, 14)
    anchor_atr = float(atr.iloc[anchor2_idx]) if anchor2_idx < len(atr) else 0.01
    anchor_atr = max(anchor_atr, 0.01)

    # Tolerance for "touching" the line (proportional to ATR)
    touch_tol = anchor_atr * 0.3
    # Break threshold: decisively through the line
    break_tol = anchor_atr * 1.0

    third_touch = False
    third_touch_bar = -1
    broke = False
    break_bar = -1
    fake_break = False
    bounced = False
    bounce_magnitude = 0.0
    max_favorable = 0.0
    max_adverse = 0.0

    for i in range(anchor2_idx + 3, end_idx + 1):  # skip a few bars after anchor2
        line_val = anchor2_price + slope * (i - anchor2_idx)
        bar = df.iloc[i]
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])

        if side == "support":
            # Third touch: low comes close to line but close stays above
            if not third_touch and abs(low - line_val) <= touch_tol and close >= line_val - touch_tol:
                third_touch = True
                third_touch_bar = i - anchor2_idx

            # Break: close decisively below
            if not broke and close < line_val - break_tol:
                broke = True
                break_bar = i - anchor2_idx

            # Bounce magnitude: how far above line did price go
            favorable = (high - line_val) / anchor_atr
            max_favorable = max(max_favorable, favorable)
            if favorable > 1.0:
                bounced = True
                bounce_magnitude = max(bounce_magnitude, favorable)

            # Adverse: how far below line
            adverse = (line_val - low) / anchor_atr
            max_adverse = max(max_adverse, adverse)
        else:  # resistance
            if not third_touch and abs(high - line_val) <= touch_tol and close <= line_val + touch_tol:
                third_touch = True
                third_touch_bar = i - anchor2_idx

            if not broke and close > line_val + break_tol:
                broke = True
                break_bar = i - anchor2_idx

            favorable = (line_val - low) / anchor_atr
            max_favorable = max(max_favorable, favorable)
            if favorable > 1.0:
                bounced = True
                bounce_magnitude = max(bounce_magnitude, favorable)

            adverse = (high - line_val) / anchor_atr
            max_adverse = max(max_adverse, adverse)

    # Fake break detection: broke AND then returned to other side within 5 bars
    if broke and break_bar >= 0:
        recovery_end = min(anchor2_idx + break_bar + 1 + 5, total_bars - 1)
        for j in range(anchor2_idx + break_bar + 1, recovery_end + 1):
            line_val_j = anchor2_price + slope * (j - anchor2_idx)
            close_j = float(df.iloc[j]["close"])
            if side == "support" and close_j > line_val_j + touch_tol:
                fake_break = True
                break
            if side == "resistance" and close_j < line_val_j - touch_tol:
                fake_break = True
                break

    return PatternOutcome(
        third_touch=third_touch,
        third_touch_bars_later=third_touch_bar,
        bounced=bounced,
        bounce_magnitude_atr=round(bounce_magnitude, 2),
        broke=broke,
        break_bars_later=break_bar,
        fake_break=fake_break,
        max_return_atr=round(max_favorable, 2),
        max_drawdown_atr=round(max_adverse, 2),
    )


# ── Historical Scan (build database) ───────────────────────────────────

def _time_bucket(position: float) -> str:
    """60/20/20 split by time position (0=oldest, 1=newest)."""
    if position < 0.6:
        return "train"
    if position < 0.8:
        return "val"
    return "test"


def scan_historical_patterns(
    df,
    symbol: str,
    timeframe: str,
    pivot_window: int = 3,
    max_anchor_distance: int = 100,
    lookahead_bars: int = 50,
) -> list[PatternRecord]:
    """Scan a historical dataframe for all 2-touch structures and observe outcomes.

    For each pair of pivot lows (support candidate) or pivot highs (resistance candidate),
    extract features + outcome, store as PatternRecord.

    Each record is tagged with a time split (train/val/test) for leak-free matching.
    """
    from .types import new_id

    total = len(df)
    if total < pivot_window * 3:
        return []

    # Find pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    for i in range(pivot_window, total - pivot_window):
        h = float(df.iloc[i]["high"])
        l = float(df.iloc[i]["low"])
        left_highs = [float(df.iloc[j]["high"]) for j in range(i - pivot_window, i)]
        right_highs = [float(df.iloc[j]["high"]) for j in range(i + 1, i + pivot_window + 1)]
        left_lows = [float(df.iloc[j]["low"]) for j in range(i - pivot_window, i)]
        right_lows = [float(df.iloc[j]["low"]) for j in range(i + 1, i + pivot_window + 1)]

        if h > max(left_highs) and h >= max(right_highs):
            pivot_highs.append((i, h))
        if l < min(left_lows) and l <= min(right_lows):
            pivot_lows.append((i, l))

    records = []

    # Support lines: pairs of pivot lows where 2nd > 1st (ascending)
    for i in range(len(pivot_lows)):
        for j in range(i + 1, len(pivot_lows)):
            a1_idx, a1_price = pivot_lows[i]
            a2_idx, a2_price = pivot_lows[j]
            if a2_idx - a1_idx > max_anchor_distance:
                break
            if a2_idx - a1_idx < 5:
                continue
            if a2_price <= a1_price:
                continue  # only ascending support
            features = extract_features(df, a1_idx, a1_price, a2_idx, a2_price, "support", symbol, timeframe)
            outcome = observe_outcome(df, a1_idx, a1_price, a2_idx, a2_price, "support", lookahead_bars)
            time_pos = a2_idx / max(total, 1)
            records.append(PatternRecord(
                pattern_id=new_id(),
                features=features,
                outcome=outcome,
                anchor1_idx=a1_idx,
                anchor2_idx=a2_idx,
                anchor1_price=a1_price,
                anchor2_price=a2_price,
                detected_at_bar=a2_idx,
                time_position=round(time_pos, 4),
                split_bucket=_time_bucket(time_pos),
            ))

    # Resistance lines: pairs of pivot highs where 2nd < 1st (descending)
    for i in range(len(pivot_highs)):
        for j in range(i + 1, len(pivot_highs)):
            a1_idx, a1_price = pivot_highs[i]
            a2_idx, a2_price = pivot_highs[j]
            if a2_idx - a1_idx > max_anchor_distance:
                break
            if a2_idx - a1_idx < 5:
                continue
            if a2_price >= a1_price:
                continue  # only descending resistance
            features = extract_features(df, a1_idx, a1_price, a2_idx, a2_price, "resistance", symbol, timeframe)
            outcome = observe_outcome(df, a1_idx, a1_price, a2_idx, a2_price, "resistance", lookahead_bars)
            time_pos = a2_idx / max(total, 1)
            records.append(PatternRecord(
                pattern_id=new_id(),
                features=features,
                outcome=outcome,
                anchor1_idx=a1_idx,
                anchor2_idx=a2_idx,
                anchor1_price=a1_price,
                anchor2_price=a2_price,
                detected_at_bar=a2_idx,
                time_position=round(time_pos, 4),
                split_bucket=_time_bucket(time_pos),
            ))

    return records


def save_patterns(records: list[PatternRecord], symbol: str, timeframe: str):
    """Persist pattern records to disk."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATA_ROOT / f"{symbol}_{timeframe}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return str(path)


def load_patterns(symbol: str, timeframe: str) -> list[PatternRecord]:
    """Load historical patterns from disk."""
    path = DATA_ROOT / f"{symbol}_{timeframe}.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(_dict_to_record(data))
        except Exception:
            pass
    return records


def _record_to_dict(r: PatternRecord) -> dict:
    return {
        "pattern_id": r.pattern_id,
        "features": asdict(r.features),
        "outcome": asdict(r.outcome),
        "anchor1_idx": r.anchor1_idx,
        "anchor2_idx": r.anchor2_idx,
        "anchor1_price": r.anchor1_price,
        "anchor2_price": r.anchor2_price,
        "detected_at_bar": r.detected_at_bar,
        "split_bucket": r.split_bucket,
        "time_position": r.time_position,
    }


def append_pattern_record(symbol: str, timeframe: str, record: PatternRecord):
    """Append a single pattern record to the database file.
    Used by the closed-loop writeback from backtest/live results.
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATA_ROOT / f"{symbol}_{timeframe}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_record_to_dict(record)) + "\n")


def writeback_trade_outcome(
    symbol: str,
    timeframe: str,
    pattern_id: str,
    strategy_id: str,
    actual_outcome: dict,
) -> dict:
    """Write back actual trading result as a new pattern observation.

    Given a pattern that was traded, record what actually happened in reality.
    This feeds the pattern database with real-world outcomes (not just backtested).

    actual_outcome should include:
    - profit_atr: actual profit/loss in ATR units
    - max_return_atr: best point during the trade
    - max_drawdown_atr: worst point during the trade
    - hit_stop: bool
    - hit_target: bool
    - bars_in_trade: int
    """
    # Load the original pattern
    database = load_patterns(symbol, timeframe)
    original = None
    for r in database:
        if r.pattern_id == pattern_id:
            original = r
            break
    if not original:
        return {"ok": False, "error": f"pattern {pattern_id} not found in {symbol}_{timeframe}"}

    # Create a new outcome-based record: copy features, override outcome with real result
    from .types import new_id
    real_outcome = PatternOutcome(
        third_touch=actual_outcome.get("hit_target", False),
        third_touch_bars_later=actual_outcome.get("bars_in_trade", -1),
        bounced=actual_outcome.get("profit_atr", 0) > 0,
        bounce_magnitude_atr=round(actual_outcome.get("max_return_atr", 0), 2),
        broke=actual_outcome.get("hit_stop", False),
        break_bars_later=actual_outcome.get("bars_in_trade", -1) if actual_outcome.get("hit_stop") else -1,
        fake_break=False,
        max_return_atr=round(actual_outcome.get("max_return_atr", 0), 2),
        max_drawdown_atr=round(actual_outcome.get("max_drawdown_atr", 0), 2),
    )

    new_record = PatternRecord(
        pattern_id=new_id(),
        features=original.features,  # same structure
        outcome=real_outcome,
        anchor1_idx=original.anchor1_idx,
        anchor2_idx=original.anchor2_idx,
        anchor1_price=original.anchor1_price,
        anchor2_price=original.anchor2_price,
        detected_at_bar=original.detected_at_bar,
        time_position=1.0,  # newest (real trade)
        split_bucket="live",  # tag as real-world (outside train/val/test)
    )
    append_pattern_record(symbol, timeframe, new_record)

    return {
        "ok": True,
        "new_pattern_id": new_record.pattern_id,
        "source_pattern_id": pattern_id,
        "strategy_id": strategy_id,
        "real_outcome": asdict(real_outcome),
    }


def _dict_to_record(d: dict) -> PatternRecord:
    return PatternRecord(
        pattern_id=d.get("pattern_id", ""),
        features=PatternFeatures(**d.get("features", {})),
        outcome=PatternOutcome(**d.get("outcome", {})),
        anchor1_idx=d.get("anchor1_idx", 0),
        anchor2_idx=d.get("anchor2_idx", 0),
        anchor1_price=d.get("anchor1_price", 0.0),
        anchor2_price=d.get("anchor2_price", 0.0),
        detected_at_bar=d.get("detected_at_bar", 0),
        split_bucket=d.get("split_bucket", ""),
        time_position=d.get("time_position", 0.0),
    )


# ── PCA Dimensionality Reduction (pure numpy via SVD) ─────────────────
#
# Standard PCA: center data, compute covariance (or use SVD directly),
# keep top K eigenvectors. Reduces noise, finds structural directions.
#
# This also powers autoencoder-style anomaly detection: project down, project back up,
# large reconstruction error = pattern doesn't fit the main structural basis.

_PCA_CACHE: dict[str, dict] = {}


def fit_pca(database: list[PatternRecord], n_components: int = 5) -> dict:
    """Fit PCA on feature vectors, return components + mean + std for reuse.

    Uses SVD decomposition: X = U @ diag(S) @ V.T
    Principal components are rows of V.T (top n_components).
    """
    import numpy as np

    if len(database) < n_components + 1:
        return {"fitted": False, "reason": "insufficient samples"}

    # Build matrix and normalize each column
    X = np.array([r.features.to_vector() for r in database], dtype=np.float64)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-9] = 1.0
    X_std = (X - means) / stds

    # SVD: X_std = U @ diag(S) @ Vt
    # The principal components are rows of Vt, ordered by singular value
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)

    # Keep top n_components
    components = Vt[:n_components]  # shape: (n_components, n_features)
    explained_var = (S[:n_components] ** 2) / (X_std.shape[0] - 1)
    total_var = (S ** 2).sum() / (X_std.shape[0] - 1)
    explained_ratio = explained_var / total_var if total_var > 0 else np.zeros_like(explained_var)

    return {
        "fitted": True,
        "n_components": n_components,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "components": components.tolist(),
        "explained_variance_ratio": explained_ratio.tolist(),
        "cumulative_variance": float(explained_ratio.sum()),
    }


def pca_transform(vector: list[float], pca: dict) -> list[float]:
    """Project a feature vector into the PCA latent space."""
    import numpy as np
    if not pca.get("fitted"):
        return vector[:pca.get("n_components", len(vector))]
    means = np.array(pca["means"])
    stds = np.array(pca["stds"])
    components = np.array(pca["components"])  # (n_components, n_features)
    v = (np.array(vector) - means) / stds
    return (components @ v).tolist()


def pca_reconstruct(vector: list[float], pca: dict) -> tuple[list[float], float]:
    """Project down to latent, project back, return (reconstructed, error).

    Large reconstruction error means the vector doesn't fit the principal structure
    — it's an anomaly relative to the data's main modes of variation.
    """
    import numpy as np
    if not pca.get("fitted"):
        return vector, 0.0
    means = np.array(pca["means"])
    stds = np.array(pca["stds"])
    components = np.array(pca["components"])

    v = (np.array(vector) - means) / stds
    latent = components @ v           # project down
    v_reconstructed = components.T @ latent  # project back up
    # De-normalize
    orig_space = v_reconstructed * stds + means
    # Reconstruction error in normalized space (more stable)
    error = float(np.linalg.norm(v - v_reconstructed))
    return orig_space.tolist(), error


# ── Outcome-based Metric Learning ──────────────────────────────────────
#
# Simple but effective: learn which features are most predictive of outcomes.
# For each feature, compute its correlation with target variable (e.g. p_bounce).
# Features that strongly predict outcome get higher weight in the distance metric.
#
# This is a lightweight form of metric learning — no neural network needed.

_LEARNED_WEIGHTS_CACHE: dict[str, list[float]] = {}


def learn_feature_weights(database: list[PatternRecord], target: str = "bounced") -> list[float]:
    """Learn per-feature importance weights from historical outcomes.

    For each feature dimension, compute the absolute correlation with the target.
    Features that are more predictive of outcome get higher weights.

    target options:
      - "bounced": did price bounce > 1 ATR
      - "broke": did price break decisively
      - "third_touch": did 3rd confirming touch occur
      - "profitable": max_return_atr > max_drawdown_atr
    """
    if len(database) < 20:
        return [1.0] * 10  # not enough data — use uniform weights

    import numpy as np

    vectors = np.array([r.features.to_vector() for r in database])

    def outcome_value(r: PatternRecord) -> float:
        if target == "bounced":
            return 1.0 if r.outcome.bounced else 0.0
        if target == "broke":
            return 1.0 if r.outcome.broke else 0.0
        if target == "third_touch":
            return 1.0 if r.outcome.third_touch else 0.0
        if target == "profitable":
            return 1.0 if r.outcome.max_return_atr > r.outcome.max_drawdown_atr else 0.0
        if target == "expected_value":
            # regression target: return - drawdown
            return r.outcome.max_return_atr - r.outcome.max_drawdown_atr
        return 0.0

    targets = np.array([outcome_value(r) for r in database])

    # If target has no variance, can't learn weights
    if np.std(targets) < 1e-6:
        return [1.0] * vectors.shape[1]

    # Pearson correlation per feature dim
    n_dims = vectors.shape[1]
    weights = []
    for i in range(n_dims):
        col = vectors[:, i]
        if np.std(col) < 1e-6:
            weights.append(0.0)  # constant feature, no predictive value
            continue
        corr = np.corrcoef(col, targets)[0, 1]
        if np.isnan(corr):
            weights.append(0.0)
        else:
            weights.append(abs(corr))

    # Normalize so sum = n_dims (uniform = [1,1,...,1])
    total = sum(weights)
    if total < 1e-6:
        return [1.0] * n_dims
    scaled = [w * n_dims / total for w in weights]
    return scaled


def _learned_distance(vec_a: list[float], vec_b: list[float], stds: list[float], weights: list[float]) -> float:
    """Weighted whitened euclidean, with per-dimension importance from learned weights."""
    total = 0.0
    for i in range(len(vec_a)):
        diff = (vec_a[i] - vec_b[i]) / stds[i]
        total += weights[i] * diff * diff
    return total ** 0.5


def learn_metric_from_pairs(
    database: list[PatternRecord],
    positive_pairs: list[tuple[str, str]],
    negative_pairs: list[tuple[str, str]],
    learning_rate: float = 0.1,
    iterations: int = 50,
) -> list[float]:
    """Lightweight contrastive metric learning (Siamese-style without neural network).

    Goal: find feature weights such that:
      - distance(pos_a, pos_b) is SMALL (positive pairs)
      - distance(neg_a, neg_b) is LARGE (negative pairs)

    Method: gradient descent on margin loss.
    L = sum(d_pos^2) - sum(max(0, margin - d_neg)^2)

    Per-feature gradient:
      d/dw_i [d^2] = (a_i - b_i)^2 / std_i^2

    For positive pairs we MINIMIZE this → decrease w_i if difference is large
    For negative pairs we MAXIMIZE distance → increase w_i where difference is large
    """
    if not positive_pairs and not negative_pairs:
        return [1.0] * 10

    import numpy as np

    # Build pattern_id → vector lookup
    by_id: dict[str, list[float]] = {}
    for r in database:
        by_id[r.pattern_id] = r.features.to_vector()

    # Filter pairs to those we have data for
    pos = [(by_id[a], by_id[b]) for a, b in positive_pairs if a in by_id and b in by_id]
    neg = [(by_id[a], by_id[b]) for a, b in negative_pairs if a in by_id and b in by_id]

    if not pos and not neg:
        return [1.0] * 10

    _, stds_list = _compute_feature_stats(database)
    stds = np.array(stds_list)
    n_dims = len(stds)
    weights = np.ones(n_dims)
    margin = 2.0

    for it in range(iterations):
        grad = np.zeros(n_dims)

        # Positive pairs: minimize distance → gradient pulls weights down on dims where pos pairs differ
        for va, vb in pos:
            diff = (np.array(va) - np.array(vb)) / stds
            grad += diff * diff  # positive contribution → gradient ascent reduces weights

        # Negative pairs: maximize distance up to margin
        for va, vb in neg:
            diff = (np.array(va) - np.array(vb)) / stds
            d = np.sqrt(np.sum(weights * diff * diff))
            if d < margin:
                # Margin not met → push weights up where dims differ
                grad -= diff * diff

        # Update with learning rate, keep weights non-negative
        weights = weights - learning_rate * grad / max(len(pos) + len(neg), 1)
        weights = np.clip(weights, 0.01, 10.0)

    # Normalize to sum = n_dims
    total = weights.sum()
    if total > 1e-6:
        weights = weights * n_dims / total
    return weights.tolist()


# ── Similarity Search (multi-metric, normalized, leak-free) ────────────

# Indices in feature vector: [slope_atr, log_length, volatility, trend_up, trend_down, trend_range, support, rsi, ma_dist, touch_quality]
_NUMERICAL_IDX = [0, 1, 2, 7, 8]     # slope, length, vol, rsi, ma_dist (touch_quality removed)
_CATEGORICAL_IDX = [3, 4, 5, 6]      # trend flags + support flag


def _compute_feature_stats(database: list[PatternRecord]) -> tuple[list[float], list[float]]:
    """Compute per-feature mean and std from database for normalization.
    Used for whitened / z-score distance.
    """
    if not database:
        return [0.0] * 10, [1.0] * 10
    vectors = [r.features.to_vector() for r in database]
    n_dims = len(vectors[0])
    means = [sum(v[i] for v in vectors) / len(vectors) for i in range(n_dims)]
    stds = []
    for i in range(n_dims):
        var = sum((v[i] - means[i]) ** 2 for v in vectors) / len(vectors)
        std = max(var ** 0.5, 1e-6)
        stds.append(std)
    return means, stds


def _weighted_euclidean(vec_a: list[float], vec_b: list[float], stds: list[float]) -> float:
    """Euclidean with each dimension normalized by std dev (whitened distance)."""
    total = 0.0
    for i in range(len(vec_a)):
        diff = (vec_a[i] - vec_b[i]) / stds[i]
        total += diff * diff
    return total ** 0.5


def _cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """1 - cosine similarity. Better for angular similarity."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 2.0
    return 1.0 - (dot / (norm_a * norm_b))


def _hybrid_distance(vec_a: list[float], vec_b: list[float], stds: list[float]) -> float:
    """Weighted combination of numerical (whitened euclidean) + categorical (hamming)."""
    # Numerical part: whitened euclidean on continuous dims
    num_sum = 0.0
    for i in _NUMERICAL_IDX:
        diff = (vec_a[i] - vec_b[i]) / stds[i]
        num_sum += diff * diff
    num_dist = num_sum ** 0.5

    # Categorical part: hamming (number of mismatches) normalized
    cat_mismatches = sum(1 for i in _CATEGORICAL_IDX if abs(vec_a[i] - vec_b[i]) > 0.5)
    cat_dist = cat_mismatches / len(_CATEGORICAL_IDX)

    # Weighted combination: 70% numerical, 30% categorical
    return 0.7 * num_dist + 0.3 * cat_dist * 5.0  # scale categorical up


def find_similar(
    current: PatternFeatures,
    database: list[PatternRecord],
    k: int = 30,
    same_side_only: bool = True,
    metric: str = "learned",  # "learned" | "hybrid" | "euclidean" | "cosine"
    max_time_position: float | None = None,  # leak prevention: only earlier patterns
    learned_weights: list[float] | None = None,
    exclude_anchors: tuple[int, int] | None = None,  # (a1, a2) of the query line
    min_required: int = 10,
) -> list[tuple[PatternRecord, float]]:
    """Find K nearest historical patterns using selected distance metric.

    - Features are normalized by per-dimension std dev (whitened — not full
      z-score but distance-equivalent since means cancel in differences).
    - metric="learned" uses outcome-correlated feature weights (default)
    - Numerical vs categorical features get separate treatment in hybrid mode
    - max_time_position prevents matching future patterns (information leak)
    - exclude_anchors prevents self-matching when the current line is itself
      in the database (e.g. found via prior writeback). Records with the
      same symbol+timeframe+side and anchors within ±2 bars are skipped.
    - K is adaptively capped to max(5, len(candidates) // 3) so very small
      candidate pools don't return overconfident "k=30" matches.
    - Returns empty if fewer than `min_required` candidates pass filters
      (sample too small for any meaningful K-NN claim).
    """
    if not database:
        return []

    # Filter eligible candidates
    candidates = []
    for r in database:
        if same_side_only and r.features.side != current.side:
            continue
        if max_time_position is not None and r.time_position > max_time_position:
            continue
        # Self-exclusion: same symbol+timeframe AND anchors within 2 bars
        if exclude_anchors is not None:
            if (r.features.symbol == current.symbol
                    and r.features.timeframe == current.timeframe
                    and abs(getattr(r, "anchor1_idx", -999) - exclude_anchors[0]) <= 2
                    and abs(getattr(r, "anchor2_idx", -999) - exclude_anchors[1]) <= 2):
                continue
        candidates.append(r)

    if len(candidates) < min_required:
        return []

    # Adaptive K cap — never return more than 1/3 of candidates
    k = max(1, min(k, max(5, len(candidates) // 3)))

    # Compute normalization stats from filtered candidates
    _, stds = _compute_feature_stats(candidates)
    current_vec = current.to_vector()

    # For learned metric: derive weights from same-side candidates' outcomes
    if metric == "learned":
        if learned_weights is None:
            learned_weights = learn_feature_weights(candidates, target="profitable")

    scored: list[tuple[PatternRecord, float]] = []
    for r in candidates:
        vec = r.features.to_vector()
        if metric == "euclidean":
            dist = _weighted_euclidean(current_vec, vec, stds)
        elif metric == "cosine":
            dist = _cosine_distance(current_vec, vec)
        elif metric == "learned":
            dist = _learned_distance(current_vec, vec, stds, learned_weights)
        else:  # hybrid
            dist = _hybrid_distance(current_vec, vec, stds)
        scored.append((r, dist))

    scored.sort(key=lambda x: x[1])
    return scored[:k]


MIN_SAMPLE_SIZE = 20  # below this, confidence drops hard


def _compute_split_stats(samples: list[PatternRecord]) -> dict:
    """Compute probability stats for a single split bucket."""
    if not samples:
        return {"n": 0, "p_bounce": 0.0, "p_break": 0.0, "avg_ret": 0.0, "avg_dd": 0.0}
    n = len(samples)
    bounces = sum(1 for r in samples if r.outcome.bounced)
    breaks = sum(1 for r in samples if r.outcome.broke)
    avg_ret = sum(r.outcome.max_return_atr for r in samples) / n
    avg_dd = sum(r.outcome.max_drawdown_atr for r in samples) / n
    return {
        "n": n,
        "p_bounce": bounces / n,
        "p_break": breaks / n,
        "avg_ret": avg_ret,
        "avg_dd": avg_dd,
    }


def compute_outcome_stats(similar: list[tuple[PatternRecord, float]]) -> dict:
    """Aggregate outcomes with confidence scoring, cross-split validation, overfit detection."""
    if not similar:
        return {
            "sample_size": 0,
            "p_third_touch": 0.0, "p_bounce": 0.0, "p_break": 0.0, "p_fake_break": 0.0,
            "avg_return_atr": 0.0, "avg_drawdown_atr": 0.0,
            "expected_value": 0.0, "confidence": 0.0,
            "overfit_flag": "insufficient_samples",
            "split_breakdown": {}, "trustworthiness": "none",
        }

    records = [r for r, _ in similar]
    n = len(records)

    # Overall stats
    third_touches = sum(1 for r in records if r.outcome.third_touch)
    bounces = sum(1 for r in records if r.outcome.bounced)
    breaks = sum(1 for r in records if r.outcome.broke)
    fake_breaks = sum(1 for r in records if r.outcome.fake_break)
    avg_ret = sum(r.outcome.max_return_atr for r in records) / n
    avg_dd = sum(r.outcome.max_drawdown_atr for r in records) / n
    p_bounce = bounces / n
    p_break = breaks / n

    # EV calculation: separate bounce-only outcomes vs break outcomes
    # A realistic trade scales in on bounce signal and stops out on break
    # So favorable = avg_return WHEN bounced (not overall)
    # Adverse = stop loss = typical 1 ATR, not max historical dd
    bounce_records = [r for r in records if r.outcome.bounced and not r.outcome.broke]
    break_records = [r for r in records if r.outcome.broke and not r.outcome.bounced]
    pure_bounce_return = sum(r.outcome.max_return_atr for r in bounce_records) / max(len(bounce_records), 1)
    # Use a realistic stop assumption: 1.5 ATR (typical structure stop), not worst-case
    assumed_stop = 1.5
    # Probability of pure-bounce (win) vs break (lose)
    p_win = len(bounce_records) / n
    p_lose = len(break_records) / n
    ev = p_win * pure_bounce_return - p_lose * assumed_stop

    # Cross-split validation: compare stats across train/val/test buckets
    train = [r for r in records if r.split_bucket == "train"]
    val = [r for r in records if r.split_bucket == "val"]
    test = [r for r in records if r.split_bucket == "test"]
    train_stats = _compute_split_stats(train)
    val_stats = _compute_split_stats(val)
    test_stats = _compute_split_stats(test)

    # Overfit detection: does bounce probability differ across splits?
    overfit_flag = "stable"
    if train_stats["n"] >= 5 and test_stats["n"] >= 5:
        train_p = train_stats["p_bounce"]
        test_p = test_stats["p_bounce"]
        # If train >> test by 20%+, it's overfitting
        if train_p > test_p + 0.2:
            overfit_flag = "overfit_detected"
        elif abs(train_p - test_p) > 0.3:
            overfit_flag = "high_variance"
        elif test_p > train_p + 0.1:
            overfit_flag = "improving"
    elif n < MIN_SAMPLE_SIZE:
        overfit_flag = "insufficient_samples"

    # Confidence score: combines sample size + split agreement
    # sqrt(n/MIN) capped at 1.0, then multiplied by split agreement
    sample_conf = min(1.0, (n / MIN_SAMPLE_SIZE) ** 0.5)
    split_conf = 1.0
    if train_stats["n"] >= 5 and test_stats["n"] >= 5:
        split_conf = max(0.0, 1.0 - abs(train_stats["p_bounce"] - test_stats["p_bounce"]))
    confidence = round(sample_conf * split_conf, 3)

    # Trustworthiness label
    if confidence >= 0.7 and overfit_flag == "stable":
        trust = "high"
    elif confidence >= 0.4:
        trust = "medium"
    else:
        trust = "low"

    return {
        "sample_size": n,
        "p_third_touch": round(third_touches / n, 3),
        "p_bounce": round(p_bounce, 3),
        "p_break": round(p_break, 3),
        "p_fake_break": round(fake_breaks / n, 3),
        "avg_return_atr": round(avg_ret, 2),
        "avg_drawdown_atr": round(avg_dd, 2),
        "expected_value": round(ev, 2),
        "confidence": confidence,
        "overfit_flag": overfit_flag,
        "trustworthiness": trust,
        "split_breakdown": {
            "train": {"n": train_stats["n"], "p_bounce": round(train_stats["p_bounce"], 3), "ev": round(train_stats["p_bounce"] * train_stats["avg_ret"] - train_stats["p_break"] * train_stats["avg_dd"], 2)},
            "val":   {"n": val_stats["n"],   "p_bounce": round(val_stats["p_bounce"], 3),   "ev": round(val_stats["p_bounce"] * val_stats["avg_ret"] - val_stats["p_break"] * val_stats["avg_dd"], 2)},
            "test":  {"n": test_stats["n"],  "p_bounce": round(test_stats["p_bounce"], 3),  "ev": round(test_stats["p_bounce"] * test_stats["avg_ret"] - test_stats["p_break"] * test_stats["avg_dd"], 2)},
        },
    }


# ── Lightweight density-based clustering (no sklearn dependency) ───────

def find_clusters(
    database: list[PatternRecord],
    eps: float = 1.5,
    min_samples: int = 10,
) -> list[dict]:
    """Simple DBSCAN-style density clustering on normalized feature vectors.

    Returns list of clusters, each with:
      - cluster_id
      - center_features (average)
      - sample_count
      - p_bounce, p_break, avg_ev (aggregate outcome)
      - label: "typical" or "rare"
    """
    if len(database) < min_samples * 2:
        return []

    _, stds = _compute_feature_stats(database)
    vectors = [r.features.to_vector() for r in database]
    n = len(vectors)

    # For each point, find neighbors within eps
    labels = [-1] * n  # -1 = unassigned
    cluster_id = 0

    def neighbors(i):
        out = []
        for j in range(n):
            if i == j:
                continue
            dist = _weighted_euclidean(vectors[i], vectors[j], stds)
            if dist <= eps:
                out.append(j)
        return out

    for i in range(n):
        if labels[i] != -1:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            labels[i] = -2  # noise
            continue
        # Start new cluster
        labels[i] = cluster_id
        stack = list(nbrs)
        while stack:
            j = stack.pop()
            if labels[j] == -2:
                labels[j] = cluster_id
            if labels[j] != -1:
                continue
            labels[j] = cluster_id
            j_nbrs = neighbors(j)
            if len(j_nbrs) >= min_samples:
                stack.extend(j_nbrs)
        cluster_id += 1

    # Aggregate clusters
    clusters = []
    for cid in range(cluster_id):
        members = [database[i] for i in range(n) if labels[i] == cid]
        if not members:
            continue
        bounces = sum(1 for r in members if r.outcome.bounced)
        breaks = sum(1 for r in members if r.outcome.broke)
        avg_ret = sum(r.outcome.max_return_atr for r in members) / len(members)
        avg_dd = sum(r.outcome.max_drawdown_atr for r in members) / len(members)
        p_bounce = bounces / len(members)
        p_break = breaks / len(members)
        ev = p_bounce * avg_ret - p_break * avg_dd

        # Center = element-wise mean of feature vectors
        member_vecs = [r.features.to_vector() for r in members]
        center = [sum(v[k] for v in member_vecs) / len(member_vecs) for k in range(len(member_vecs[0]))]

        clusters.append({
            "cluster_id": cid,
            "sample_count": len(members),
            "center": [round(c, 3) for c in center],
            "p_bounce": round(p_bounce, 3),
            "p_break": round(p_break, 3),
            "avg_return_atr": round(avg_ret, 2),
            "avg_drawdown_atr": round(avg_dd, 2),
            "expected_value": round(ev, 2),
            "label": "typical" if len(members) >= n * 0.1 else "rare",
        })

    clusters.sort(key=lambda c: -c["sample_count"])
    return clusters


def detect_anomaly(
    current: PatternFeatures,
    database: list[PatternRecord],
    threshold: float = 3.0,
    use_pca: bool = True,
) -> dict:
    """Check if the current structure is unusual compared to the database.

    Three-layer anomaly detection:
    1. Z-score per feature (which dimensions are outliers)
    2. Nearest-neighbor distance (distance to closest historical pattern)
    3. PCA reconstruction error (does it fit the principal structural modes?)

    Returns an anomaly score combining all three.
    """
    if not database:
        return {"is_anomaly": False, "nearest_distance": None, "outlier_features": [], "pca_error": None}

    means, stds = _compute_feature_stats(database)
    current_vec = current.to_vector()

    # Layer 1: Z-score per feature dim
    z_scores = [(current_vec[i] - means[i]) / stds[i] for i in range(len(current_vec))]
    outliers = []
    feature_names = ["slope_atr", "log_length", "volatility", "trend_up", "trend_down", "trend_range", "support", "rsi", "ma_dist"]
    for i, z in enumerate(z_scores):
        if abs(z) > threshold:
            outliers.append({"feature": feature_names[i], "z_score": round(z, 2)})

    # Layer 2: nearest neighbor distance
    same_side = [r for r in database if r.features.side == current.side]
    if not same_side:
        return {
            "is_anomaly": True, "nearest_distance": None,
            "outlier_features": outliers, "pca_error": None,
            "reason": "no same-side samples",
        }
    min_dist = min(_weighted_euclidean(current_vec, r.features.to_vector(), stds) for r in same_side)

    # Layer 3: PCA reconstruction error
    pca_error = None
    pca_anomaly = False
    if use_pca and len(same_side) >= 20:
        pca = fit_pca(same_side, n_components=min(5, len(current_vec) - 1))
        if pca.get("fitted"):
            _, pca_error = pca_reconstruct(current_vec, pca)
            # Compute baseline: average reconstruction error across database
            db_errors = []
            for r in same_side[:100]:  # sample to keep it fast
                _, e = pca_reconstruct(r.features.to_vector(), pca)
                db_errors.append(e)
            import numpy as np
            mean_err = float(np.mean(db_errors))
            std_err = float(np.std(db_errors))
            # Anomaly if reconstruction error > mean + 2*std
            if std_err > 1e-6:
                error_z = (pca_error - mean_err) / std_err
                pca_anomaly = error_z > 2.0
            pca_error = round(float(pca_error), 3)

    # Combine all three signals
    is_anomaly = (min_dist > threshold) or (len(outliers) > 0) or pca_anomaly

    return {
        "is_anomaly": is_anomaly,
        "nearest_distance": round(min_dist, 3),
        "outlier_features": outliers,
        "pca_error": pca_error,
        "pca_anomaly": pca_anomaly,
    }


# ── Exploration: Find Under-sampled High-EV Clusters ──────────────────
#
# Instead of full RL, use a simple explore/exploit rule:
# - Find clusters with HIGH EV but LOW sample count → "exploration priorities"
# - Find clusters with HIGH EV and LOTS of samples → "exploitation ready"
# - System can bias new data collection toward exploration targets


def discover_exploration_targets(
    database: list[PatternRecord],
    min_samples: int = 5,
    max_samples_for_explore: int = 40,
    min_ev: float = 0.5,
) -> dict:
    """Find clusters that are HIGH EV but UNDER-SAMPLED.

    These are exploration priorities — we have some evidence they work but not enough.
    Returns {explore_targets, exploit_targets, total_clusters}.
    """
    clusters = find_clusters(database, eps=1.5, min_samples=min_samples)

    exploration = []
    exploitation = []
    for c in clusters:
        n = c["sample_count"]
        ev = c["expected_value"]
        if ev >= min_ev:
            if n <= max_samples_for_explore:
                c["target_type"] = "explore"
                c["priority"] = round(ev / max(1, n ** 0.5), 3)
                exploration.append(c)
            else:
                c["target_type"] = "exploit"
                c["priority"] = round(ev * min(1.0, n / max_samples_for_explore), 3)
                exploitation.append(c)

    exploration.sort(key=lambda c: -c["priority"])
    exploitation.sort(key=lambda c: -c["priority"])

    return {
        "explore_targets": exploration,
        "exploit_targets": exploitation,
        "total_clusters": len(clusters),
    }


# ── Main match entry point ─────────────────────────────────────────────

def match_pattern(
    df,
    anchor1_idx: int,
    anchor1_price: float,
    anchor2_idx: int,
    anchor2_price: float,
    side: str,
    symbol: str,
    timeframe: str,
    k: int = 30,
    metric: str = "learned",
    current_time_position: float | None = None,
    use_human_labels: bool = True,
) -> dict:
    """Given a 2-touch line, return historical probability stats.

    Full pipeline:
    1. Extract normalized features
    2. Check if structure is anomalous (z-score + nearest neighbor + PCA reconstruction)
    3. Learn feature weights — from human labels if available, else from outcomes
    4. Find K nearest using learned weighted distance
    5. Compute stats with cross-split validation + confidence scoring
    6. Surface "explore vs exploit" classification from clustering
    7. Return all evidence for decision making
    """
    # Extract current features
    current = extract_features(df, anchor1_idx, anchor1_price, anchor2_idx, anchor2_price, side, symbol, timeframe)

    # Load historical database
    database = load_patterns(symbol, timeframe)

    if not database:
        return {
            "current_features": asdict(current),
            "stats": compute_outcome_stats([]),
            "similar_count": 0,
            "database_size": 0,
            "anomaly": {"is_anomaly": False, "nearest_distance": None, "outlier_features": [], "pca_error": None},
            "metric": metric,
            "weights_source": "none",
            "top_matches": [],
        }

    # Self-exclusion: prevent the query line from matching its own prior
    # writeback record in the DB. find_similar uses (symbol, timeframe,
    # anchor1_idx, anchor2_idx) within ±2 bars to identify self.
    self_exclusion = (int(anchor1_idx), int(anchor2_idx))

    # Pre-filter the database by side + time-position so that learned
    # weights are trained on the SAME pool find_similar will query —
    # avoids the "circular reasoning" of training on the full DB and
    # querying back against the full DB (Round 9/10 bug #3).
    eligible: list = []
    for r in database:
        if r.features.side != side:
            continue
        if current_time_position is not None and r.time_position > current_time_position:
            continue
        # Self-exclusion at training time too — don't let a previously
        # written-back copy of this exact line teach the model that
        # "this exact line is profitable".
        if (r.features.symbol == symbol
                and r.features.timeframe == timeframe
                and abs(getattr(r, "anchor1_idx", -999) - self_exclusion[0]) <= 2
                and abs(getattr(r, "anchor2_idx", -999) - self_exclusion[1]) <= 2):
            continue
        eligible.append(r)

    # Layer 1: Anomaly detection (uses full DB — anomaly is "is this
    # structure unusual relative to ALL history?", which is a separate
    # question from "what happened to similar lines on this side?")
    anomaly = detect_anomaly(current, database, use_pca=True)

    # Layer 2: Determine feature weights from the FILTERED pool only
    weights_source = "outcome"
    learned_weights = None
    if use_human_labels:
        try:
            from .pattern_labels import get_positive_pairs, get_negative_pairs
            pos = get_positive_pairs(symbol, timeframe)
            neg = get_negative_pairs(symbol, timeframe)
            if len(pos) + len(neg) >= 5:
                learned_weights = learn_metric_from_pairs(database, pos, neg)
                weights_source = f"human_labels (pos={len(pos)}, neg={len(neg)})"
        except Exception:
            pass

    if learned_weights is None and eligible:
        # Train on filtered eligible (NOT the full database) to avoid
        # circular reasoning — Round 10 #3.
        learned_weights = learn_feature_weights(eligible, target="profitable")

    # Layer 3: Find similar (leak-free + self-excluded)
    similar = find_similar(
        current, database, k=k,
        same_side_only=True,
        metric=metric,
        max_time_position=current_time_position,
        learned_weights=learned_weights,
        exclude_anchors=self_exclusion,  # Round 10 #1: actually pass it
    )

    # Layer 4: Stats with cross-split validation
    stats = compute_outcome_stats(similar)

    # Layer 5: Cluster context — which cluster does this structure belong to?
    cluster_info = None
    try:
        clusters = find_clusters(database, eps=1.5, min_samples=10)
        if clusters:
            # Find nearest cluster center
            _, stds = _compute_feature_stats(database)
            current_vec = current.to_vector()
            best = None
            for c in clusters:
                d = _weighted_euclidean(current_vec, c["center"], stds)
                if best is None or d < best[1]:
                    best = (c, d)
            if best:
                c, d = best
                cluster_info = {
                    "cluster_id": c["cluster_id"],
                    "label": c["label"],
                    "sample_count": c["sample_count"],
                    "p_bounce": c["p_bounce"],
                    "p_break": c["p_break"],
                    "expected_value": c["expected_value"],
                    "distance_to_center": round(d, 3),
                }
    except Exception:
        pass

    return {
        "current_features": asdict(current),
        "stats": stats,
        "similar_count": len(similar),
        "database_size": len(database),
        "anomaly": anomaly,
        "cluster_context": cluster_info,
        "metric": metric,
        "weights_source": weights_source,
        "feature_weights": [round(w, 3) for w in learned_weights],
        "top_matches": [
            {
                "pattern_id": r.pattern_id,
                "distance": round(d, 3),
                "split": r.split_bucket,
                "features": asdict(r.features),
                "outcome": asdict(r.outcome),
            }
            for r, d in similar[:5]
        ],
    }
