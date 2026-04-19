from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "HYPEUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "PEPEUSDT",
    "SUIUSDT",
)
DEFAULT_TIMEFRAMES = ("4h", "1h", "15m", "5m")
PATTERN_DIR = PROJECT_ROOT / "data" / "patterns"
USER_DRAWINGS_FILE = PROJECT_ROOT / "data" / "user_drawings_ml.jsonl"
USER_LABELS_FILE = PROJECT_ROOT / "data" / "user_drawing_labels.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "trendline_quality"

BASE_PATTERN_WEIGHT = 1.0
USER_DRAWING_WEIGHT = 2.0
USER_DRAWING_OUTCOME_WEIGHT = 4.0
USER_ORDER_INTENT_WEIGHT = 8.0
USER_ORDER_OUTCOME_WEIGHT = 10.0

FEATURE_NAMES = [
    "slope_atr",
    "length_bars",
    "volatility",
    "rsi",
    "ma_distance_atr",
    "touch_quality",
    "anchor_gap_bars",
    "anchor_gap_log",
    "anchor_distance_pct",
    "anchor_slope_pct_per_bar",
    "anchor1_log_price",
    "anchor2_log_price",
    "detected_at_bar_norm",
    "time_position",
    "timeframe_code",
    "side_support",
    "side_resistance",
    "context_downtrend",
    "context_range",
    "context_uptrend",
    "symbol_code",
    "is_high_vol",
    "rsi_overbought",
    "rsi_oversold",
    "touch_count",
    "recent_touch_count",
    "near_miss_count",
    "body_violation_count",
    "wick_rejection_count",
    "wick_rejection_ratio",
    "avg_rejection_atr",
    "max_rejection_atr",
    "last_touch_age_bars",
    "line_age_bars",
    "line_span_bars",
    "distance_to_line_atr",
    "wrong_side_close",
    "htf_confluence_score",
]

TF_CODE = {"5m": 0.0, "15m": 1.0, "1h": 2.0, "4h": 3.0}


@dataclass
class Dataset:
    x: np.ndarray
    y_cls: np.ndarray
    y_reg: np.ndarray
    sample_weight: np.ndarray
    time_position: np.ndarray
    records: list[dict[str, Any]]
    missing: list[str]
    user_rows: int = 0


class TrendlineQualityNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(n_features, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(48, 1)
        self.reg_head = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _feature_vector(row: dict[str, Any], symbol_index: dict[str, int]) -> list[float]:
    features = row.get("features") or {}
    outcome = row.get("outcome") or {}
    symbol = str(features.get("symbol") or row.get("symbol") or "").upper()
    timeframe = str(features.get("timeframe") or row.get("timeframe") or "")
    side = str(features.get("side") or "").lower()
    context = str(features.get("trend_context") or "range").lower()

    anchor1_idx = int(_safe_float(row.get("anchor1_idx")))
    anchor2_idx = int(_safe_float(row.get("anchor2_idx")))
    anchor_gap = max(0, anchor2_idx - anchor1_idx)
    anchor1_price = max(_safe_float(row.get("anchor1_price")), 1e-12)
    anchor2_price = max(_safe_float(row.get("anchor2_price")), 1e-12)
    anchor_distance_pct = (anchor2_price - anchor1_price) / anchor1_price
    anchor_slope_pct_per_bar = anchor_distance_pct / max(anchor_gap, 1)
    detected_at = max(_safe_float(row.get("detected_at_bar")), 0.0)
    time_position = min(max(_safe_float(row.get("time_position")), 0.0), 1.0)
    max_return = _safe_float(outcome.get("max_return_atr"))
    max_drawdown = _safe_float(outcome.get("max_drawdown_atr"))
    rsi = _safe_float(features.get("rsi"), 50.0)
    volatility = _safe_float(features.get("volatility"))

    return [
        _safe_float(features.get("slope_atr")),
        _safe_float(features.get("length_bars")),
        volatility,
        rsi,
        _safe_float(features.get("ma_distance_atr")),
        _safe_float(features.get("touch_quality")),
        float(anchor_gap),
        math.log1p(anchor_gap),
        anchor_distance_pct,
        anchor_slope_pct_per_bar,
        math.log(anchor1_price),
        math.log(anchor2_price),
        detected_at / max(detected_at + 1000.0, 1.0),
        time_position,
        TF_CODE.get(timeframe, -1.0),
        1.0 if side == "support" else 0.0,
        1.0 if side == "resistance" else 0.0,
        1.0 if context == "downtrend" else 0.0,
        1.0 if context == "range" else 0.0,
        1.0 if context == "uptrend" else 0.0,
        symbol_index.get(symbol, -1) / max(len(symbol_index) - 1, 1),
        1.0 if volatility >= 0.03 else 0.0,
        1.0 if rsi >= 70.0 else 0.0,
        1.0 if rsi <= 30.0 else 0.0,
        _safe_float(features.get("touch_count"), _safe_float(features.get("confirming_touch_count"))),
        _safe_float(features.get("recent_touch_count"), _safe_float(features.get("recent_bar_touch_count"))),
        _safe_float(features.get("near_miss_count")),
        _safe_float(features.get("body_violation_count")),
        _safe_float(features.get("wick_rejection_count"), _safe_float(features.get("rejection_count"))),
        _safe_float(features.get("wick_rejection_ratio")),
        _safe_float(features.get("avg_rejection_atr")),
        _safe_float(features.get("max_rejection_atr"), max_return),
        _safe_float(features.get("last_touch_age_bars"), _safe_float(features.get("bars_since_touch"))),
        _safe_float(features.get("line_age_bars")),
        _safe_float(features.get("line_span_bars"), _safe_float(features.get("length_bars"))),
        _safe_float(features.get("distance_to_line_atr")),
        _safe_float(features.get("wrong_side_close")),
        _safe_float(features.get("htf_confluence_score")),
    ]


def _label(row: dict[str, Any]) -> tuple[float, float]:
    outcome = row.get("outcome") or {}
    bounced = 1.0 if bool(outcome.get("bounced")) else 0.0
    max_return = _safe_float(outcome.get("max_return_atr"))
    max_drawdown = _safe_float(outcome.get("max_drawdown_atr"))
    quality = max_return - 0.5 * max_drawdown
    return bounced, quality


def _load_user_outcome_labels() -> dict[str, dict[str, Any]]:
    if not USER_LABELS_FILE.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with USER_LABELS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            manual_line_id = str(row.get("manual_line_id") or "")
            if not manual_line_id:
                continue
            current = labels.get(manual_line_id)
            if current is None or _safe_float(row.get("expected_realized_r"), -999.0) >= _safe_float(current.get("expected_realized_r"), -999.0):
                labels[manual_line_id] = row
    return labels


def _load_user_preference_rows(symbols: list[str], timeframes: list[str]) -> list[dict[str, Any]]:
    if not USER_DRAWINGS_FILE.exists():
        return []
    symbol_set = {s.upper() for s in symbols}
    tf_set = set(timeframes)
    grouped: dict[str, dict[str, Any]] = {}
    outcome_labels = _load_user_outcome_labels()

    def _prefer_drawing(current: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
        if current is None:
            return candidate
        cur_has_features = bool(current.get("features"))
        cand_has_features = bool(candidate.get("features"))
        if cand_has_features and not cur_has_features:
            return candidate
        if cand_has_features == cur_has_features and _safe_float(candidate.get("ts")) >= _safe_float(current.get("ts")):
            return candidate
        return current

    with USER_DRAWINGS_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            symbol = str(row.get("symbol") or "").upper()
            timeframe = str(row.get("timeframe") or "")
            manual_line_id = str(row.get("manual_line_id") or "")
            if not manual_line_id and row.get("event") == "user_drawing":
                manual_line_id = (
                    f"legacy_{symbol}_{timeframe}_"
                    f"{row.get('t_start')}_{row.get('t_end')}_"
                    f"{row.get('price_start')}_{row.get('price_end')}"
                )
            if not manual_line_id or symbol not in symbol_set or timeframe not in tf_set:
                continue
            bucket = grouped.setdefault(manual_line_id, {})
            event = row.get("event")
            if event == "user_drawing":
                bucket["drawing"] = _prefer_drawing(bucket.get("drawing"), row)
            elif event == "user_order_intent":
                bucket["order"] = row

    out: list[dict[str, Any]] = []
    for manual_line_id, bucket in grouped.items():
        base = dict(bucket.get("drawing") or bucket.get("order") or {})
        if not base:
            continue
        base["manual_line_id"] = manual_line_id
        base["_source"] = "user_preference"
        base["_has_order_intent"] = bool(bucket.get("order"))
        base["_outcome_label"] = outcome_labels.get(manual_line_id)
        out.append(base)
    return out


def _user_feature_vector(row: dict[str, Any], symbol_index: dict[str, int]) -> list[float]:
    features = row.get("features") or {}
    symbol = str(row.get("symbol") or "").upper()
    timeframe = str(row.get("timeframe") or "")
    side = str(row.get("side") or row.get("kind") or "").lower()
    close = max(_safe_float(features.get("close") or row.get("line_price") or row.get("entry_price") or row.get("price_end")), 1e-12)
    atr = _safe_float(features.get("atr"))
    atr_pct = _safe_float(features.get("atr_pct"))
    length_bars = max(_safe_float(row.get("bars_between"), 1.0), 1.0)
    price_start = max(_safe_float(row.get("price_start") or row.get("line_price") or row.get("entry_price")), 1e-12)
    price_end = max(_safe_float(row.get("price_end") or row.get("line_price") or row.get("entry_price") or price_start), 1e-12)
    anchor_distance_pct = _safe_float(
        row.get("anchor_distance_pct"),
        (price_end - price_start) / price_start if price_start > 0 else 0.0,
    )
    slope_per_bar = _safe_float(row.get("slope_per_bar"), (price_end - price_start) / length_bars)
    slope_atr = slope_per_bar / atr if atr > 0 else anchor_distance_pct / length_bars
    dist_pct = _safe_float(row.get("dist_to_line_pct"))
    ma_distance_atr = (dist_pct * close / atr) if atr > 0 else dist_pct
    rsi = _safe_float(features.get("rsi"), 50.0)
    touch_quality = 1.0 if row.get("_has_order_intent") else 0.75
    time_position = 0.49

    return [
        slope_atr,
        length_bars,
        atr_pct,
        rsi,
        ma_distance_atr,
        touch_quality,
        length_bars,
        math.log1p(length_bars),
        anchor_distance_pct,
        anchor_distance_pct / max(length_bars, 1.0),
        math.log(price_start),
        math.log(price_end),
        0.99,
        time_position,
        TF_CODE.get(timeframe, -1.0),
        1.0 if side == "support" else 0.0,
        1.0 if side == "resistance" else 0.0,
        0.0,
        1.0,
        0.0,
        symbol_index.get(symbol, -1) / max(len(symbol_index) - 1, 1),
        1.0 if atr_pct >= 0.03 else 0.0,
        1.0 if rsi >= 70.0 else 0.0,
        1.0 if rsi <= 30.0 else 0.0,
        _safe_float(features.get("touch_count")),
        _safe_float(features.get("recent_touch_count")),
        _safe_float(features.get("near_miss_count")),
        _safe_float(features.get("body_violation_count")),
        _safe_float(features.get("wick_rejection_count")),
        _safe_float(features.get("wick_rejection_ratio")),
        _safe_float(features.get("avg_rejection_atr")),
        _safe_float(features.get("max_rejection_atr")),
        _safe_float(features.get("last_touch_age_bars"), -1.0),
        _safe_float(features.get("line_age_bars"), length_bars),
        _safe_float(features.get("line_span_bars"), length_bars),
        _safe_float(features.get("distance_to_line_atr"), ma_distance_atr),
        _safe_float(features.get("wrong_side_close")),
        _safe_float(features.get("htf_confluence_score")),
    ]


def _user_sample_weight(row: dict[str, Any], outcome_label: dict[str, Any]) -> float:
    has_order_intent = bool(row.get("_has_order_intent"))
    has_outcome = bool(outcome_label)
    if has_order_intent and has_outcome:
        return USER_ORDER_OUTCOME_WEIGHT
    if has_order_intent:
        return USER_ORDER_INTENT_WEIGHT
    if has_outcome:
        return USER_DRAWING_OUTCOME_WEIGHT
    return USER_DRAWING_WEIGHT


def load_dataset(symbols: list[str], timeframes: list[str]) -> Dataset:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    symbol_index = {symbol: idx for idx, symbol in enumerate(symbols)}
    for symbol in symbols:
        for timeframe in timeframes:
            path = PATTERN_DIR / f"{symbol}_{timeframe}.jsonl"
            if not path.exists():
                missing.append(f"{symbol}_{timeframe}")
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    row["symbol"] = symbol
                    row["timeframe"] = timeframe
                    rows.append(row)

    x_rows: list[list[float]] = []
    y_cls: list[float] = []
    y_reg: list[float] = []
    sample_weight: list[float] = []
    time_position: list[float] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        features = _feature_vector(row, symbol_index)
        if len(features) != len(FEATURE_NAMES):
            raise RuntimeError(f"feature count mismatch: {len(features)} != {len(FEATURE_NAMES)}")
        label, quality = _label(row)
        x_rows.append(features)
        y_cls.append(label)
        y_reg.append(quality)
        sample_weight.append(BASE_PATTERN_WEIGHT)
        time_position.append(min(max(_safe_float(row.get("time_position")), 0.0), 1.0))
        kept.append(row)

    user_rows = _load_user_preference_rows(symbols, timeframes)
    for row in user_rows:
        features = _user_feature_vector(row, symbol_index)
        if len(features) != len(FEATURE_NAMES):
            raise RuntimeError(f"user feature count mismatch: {len(features)} != {len(FEATURE_NAMES)}")
        outcome_label = row.get("_outcome_label") or {}
        x_rows.append(features)
        if outcome_label:
            y_cls.append(1.0 if _safe_float(outcome_label.get("label_trade_win")) >= 0.5 else 0.0)
            y_reg.append(_safe_float(outcome_label.get("expected_realized_r")))
        else:
            y_cls.append(1.0)
            y_reg.append(1.0 if row.get("_has_order_intent") else 0.6)
        sample_weight.append(_user_sample_weight(row, outcome_label))
        time_position.append(0.49)
        kept.append(row)

    if not x_rows:
        raise RuntimeError("no training rows loaded")
    order = np.argsort(np.asarray(time_position, dtype=np.float32))
    return Dataset(
        x=np.asarray(x_rows, dtype=np.float32)[order],
        y_cls=np.asarray(y_cls, dtype=np.float32)[order],
        y_reg=np.asarray(y_reg, dtype=np.float32)[order],
        sample_weight=np.asarray(sample_weight, dtype=np.float32)[order],
        time_position=np.asarray(time_position, dtype=np.float32)[order],
        records=[kept[int(i)] for i in order],
        missing=missing,
        user_rows=len(user_rows),
    )


async def build_missing(symbols: list[str], timeframes: list[str], days: int) -> dict[str, Any]:
    from tools.pattern_batch import build_batch

    missing_by_tf: dict[str, list[str]] = {}
    for symbol in symbols:
        for timeframe in timeframes:
            path = PATTERN_DIR / f"{symbol}_{timeframe}.jsonl"
            if not path.exists():
                missing_by_tf.setdefault(timeframe, []).append(symbol)

    results: dict[str, Any] = {}
    for timeframe, tf_symbols in missing_by_tf.items():
        results[timeframe] = await build_batch(
            tf_symbols,
            [timeframe],
            days=days,
            skip_existing=True,
        )
    return results


def _standardize(train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def train_fold(
    dataset: Dataset,
    fold: int,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    train_x_raw = dataset.x[train_mask]
    val_x_raw = dataset.x[val_mask]
    train_x, val_x, mean, std = _standardize(train_x_raw, val_x_raw)
    train_y_cls = dataset.y_cls[train_mask]
    train_y_reg = dataset.y_reg[train_mask]
    train_weight = dataset.sample_weight[train_mask]
    val_y_cls = dataset.y_cls[val_mask]
    val_y_reg = dataset.y_reg[val_mask]

    model = TrendlineQualityNet(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")

    train_ds = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y_cls),
        torch.from_numpy(train_y_reg),
        torch.from_numpy(train_weight),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb_cls, yb_reg, wb in loader:
            xb = xb.to(device)
            yb_cls = yb_cls.to(device)
            yb_reg = yb_reg.to(device)
            wb = wb.to(device)
            logits, reg = model(xb)
            raw_loss = bce(logits, yb_cls) + 0.15 * mse(reg, yb_reg)
            loss = (raw_loss * wb).sum() / wb.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(val_x).to(device)
        logits, reg = model(val_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        reg_pred = reg.detach().cpu().numpy()
    auc = _auc(val_y_cls, probs)
    reg_rmse = float(np.sqrt(np.mean((reg_pred - val_y_reg) ** 2)))
    cls_acc = float(((probs >= 0.5).astype(np.float32) == val_y_cls).mean())
    return {
        "fold": fold,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "auc": auc,
        "accuracy": cls_acc,
        "reg_rmse": reg_rmse,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": mean.astype(np.float32),
        "scaler_std": std.astype(np.float32),
    }


def walk_forward_masks(time_position: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    masks = []
    edges = np.linspace(0.50, 1.00, folds + 1)
    for i in range(folds):
        val_start = edges[i]
        val_end = edges[i + 1]
        train_mask = time_position < val_start
        if i == folds - 1:
            val_mask = (time_position >= val_start) & (time_position <= val_end)
        else:
            val_mask = (time_position >= val_start) & (time_position < val_end)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        masks.append((train_mask, val_mask))
    return masks


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Train PyTorch trendline quality model.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--timeframes", default=",".join(DEFAULT_TIMEFRAMES))
    parser.add_argument("--build-missing", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]

    if args.build_missing:
        build_result = await build_missing(symbols, timeframes, args.days)
        print(json.dumps({"build_missing": build_result}, indent=2, default=str)[:4000])

    dataset = load_dataset(symbols, timeframes)
    if dataset.missing and not args.allow_missing:
        raise RuntimeError(
            "missing pattern files: "
            + ", ".join(dataset.missing)
            + " (rerun with --build-missing or --allow-missing)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] rows={len(dataset.x)} user_rows={dataset.user_rows} features={len(FEATURE_NAMES)} device={device}")
    print(
        "[train] weights="
        f"pattern={BASE_PATTERN_WEIGHT} drawing={USER_DRAWING_WEIGHT} "
        f"drawing_outcome={USER_DRAWING_OUTCOME_WEIGHT} "
        f"order_intent={USER_ORDER_INTENT_WEIGHT} "
        f"order_outcome={USER_ORDER_OUTCOME_WEIGHT}"
    )
    if dataset.missing:
        print(f"[train] missing={dataset.missing}")

    fold_results = []
    best: dict[str, Any] | None = None
    for fold, (train_mask, val_mask) in enumerate(walk_forward_masks(dataset.time_position, args.folds), start=1):
        result = train_fold(
            dataset,
            fold,
            train_mask,
            val_mask,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        fold_results.append({k: v for k, v in result.items() if k not in {"state_dict", "scaler_mean", "scaler_std"}})
        print(
            f"[train] fold={fold} train={result['train_rows']} val={result['val_rows']} "
            f"auc={result['auc']:.4f} acc={result['accuracy']:.4f} rmse={result['reg_rmse']:.4f}",
            flush=True,
        )
        if best is None or (not math.isnan(result["auc"]) and result["auc"] > best["auc"]):
            best = result

    if best is None:
        raise RuntimeError("no valid walk-forward folds")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    payload = {
        "model_state_dict": best["state_dict"],
        "scaler_mean": torch.from_numpy(best["scaler_mean"]),
        "scaler_std": torch.from_numpy(best["scaler_std"]),
        "feature_names": FEATURE_NAMES,
        "symbols": symbols,
        "timeframes": timeframes,
        "fold_metrics": fold_results,
        "best_fold": int(best["fold"]),
        "best_auc": float(best["auc"]),
        "sample_weight_policy": {
            "base_pattern": BASE_PATTERN_WEIGHT,
            "user_drawing": USER_DRAWING_WEIGHT,
            "user_drawing_outcome": USER_DRAWING_OUTCOME_WEIGHT,
            "user_order_intent": USER_ORDER_INTENT_WEIGHT,
            "user_order_outcome": USER_ORDER_OUTCOME_WEIGHT,
        },
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    torch.save(payload, checkpoint_path)
    print(f"[train] saved={checkpoint_path}")
    print(json.dumps({"fold_auc": [r["auc"] for r in fold_results], "best_auc": best["auc"]}, indent=2))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
