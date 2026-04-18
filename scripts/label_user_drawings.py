from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.data_service import get_ohlcv_with_df
from server.strategy.drawing_outcome_labeler import ReplayConfig, simulate_line_outcome

USER_DRAWINGS_FILE = PROJECT_ROOT / "data" / "user_drawings_ml.jsonl"
OUTCOMES_FILE = PROJECT_ROOT / "data" / "user_drawing_outcomes.jsonl"
LABELS_FILE = PROJECT_ROOT / "data" / "user_drawing_labels.jsonl"


def _load_latest_drawings(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    grouped: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "user_drawing":
                continue
            manual_line_id = str(row.get("manual_line_id") or "")
            if not manual_line_id:
                continue
            previous = grouped.get(manual_line_id)
            if previous is None or float(row.get("ts") or 0) >= float(previous.get("ts") or 0):
                grouped[manual_line_id] = row
    return list(grouped.values())


def _polars_to_pandas(polars_df) -> pd.DataFrame:
    pdf = polars_df.select(["open_time", "open", "high", "low", "close", "volume"]).to_pandas()
    pdf = pdf.rename(columns={"open_time": "timestamp"})
    return pdf


def _float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


async def _load_1m(symbol: str, days: int, history_mode: str) -> pd.DataFrame:
    polars_df, _ = await get_ohlcv_with_df(
        symbol,
        "1m",
        None,
        days=days,
        history_mode=history_mode,
        include_price_precision=False,
        include_render_payload=False,
    )
    if polars_df is None or polars_df.is_empty():
        return pd.DataFrame()
    return _polars_to_pandas(polars_df)


def _best_label(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    filled = [row for row in rows if row.get("filled") and row.get("status") in {"closed", "open_at_end"}]
    if not filled:
        return None
    best = max(filled, key=lambda row: float(row.get("realized_r") or -999.0))
    cfg = best.get("config") or {}
    return {
        "event": "user_drawing_label",
        "manual_line_id": best.get("manual_line_id"),
        "symbol": best.get("symbol"),
        "timeframe": best.get("timeframe"),
        "side": best.get("side"),
        "label_trade_win": int(best.get("label_trade_win") or 0),
        "expected_realized_r": float(best.get("realized_r") or 0.0),
        "best_buffer_pct": float(cfg.get("buffer_pct") or 0.0),
        "best_rr": float(cfg.get("rr") or 0.0),
        "best_trailing_enabled": bool(cfg.get("trailing_enabled")),
        "best_exit_reason": best.get("exit_reason"),
        "best_status": best.get("status"),
        "mfe_r": float(best.get("mfe_r") or 0.0),
        "mae_r": float(best.get("mae_r") or 0.0),
        "walking_stop_updates": int(best.get("walking_stop_updates") or 0),
    }


async def async_main() -> int:
    parser = argparse.ArgumentParser(description="Replay user-drawn trendlines and label outcomes.")
    parser.add_argument("--input", default=str(USER_DRAWINGS_FILE))
    parser.add_argument("--outcomes", default=str(OUTCOMES_FILE))
    parser.add_argument("--labels", default=str(LABELS_FILE))
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol filter.")
    parser.add_argument("--timeframes", default="", help="Optional comma-separated timeframe filter.")
    parser.add_argument("--buffers", default="0.0003,0.0005,0.001,0.002")
    parser.add_argument("--rrs", default="2,4,8,15,25,50")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--history-mode", default="fast_window", choices=["fast_window", "full_history"])
    parser.add_argument("--max-tf-bars", type=int, default=20)
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--max-lines", type=int, default=0)
    args = parser.parse_args()

    drawings = _load_latest_drawings(Path(args.input))
    symbols = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
    timeframes = {tf.strip() for tf in args.timeframes.split(",") if tf.strip()}
    if symbols:
        drawings = [row for row in drawings if str(row.get("symbol") or "").upper() in symbols]
    if timeframes:
        drawings = [row for row in drawings if str(row.get("timeframe") or "") in timeframes]
    if args.max_lines > 0:
        drawings = drawings[-args.max_lines :]

    buffers = _float_list(args.buffers)
    rrs = _float_list(args.rrs)
    configs = [
        ReplayConfig(buffer_pct=buffer, rr=rr, max_tf_bars=args.max_tf_bars, fee_bps=args.fee_bps)
        for buffer in buffers
        for rr in rrs
    ]

    by_symbol: dict[str, pd.DataFrame] = {}
    outcome_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for drawing in drawings:
        symbol = str(drawing.get("symbol") or "").upper()
        if symbol not in by_symbol:
            print(f"[label] loading {symbol} 1m days={args.days} mode={args.history_mode}", flush=True)
            by_symbol[symbol] = await _load_1m(symbol, args.days, args.history_mode)
        per_line = [simulate_line_outcome(drawing, by_symbol[symbol], cfg) for cfg in configs]
        best = _best_label(per_line)
        best_key = None
        if best is not None:
            best_key = (best["best_buffer_pct"], best["best_rr"], best["best_trailing_enabled"])
            label_rows.append(best)
        for row in per_line:
            cfg = row.get("config") or {}
            row["is_best_config"] = bool(best_key and (
                float(cfg.get("buffer_pct") or 0.0),
                float(cfg.get("rr") or 0.0),
                bool(cfg.get("trailing_enabled")),
            ) == best_key)
            outcome_rows.append(row)
        print(
            f"[label] {symbol} {drawing.get('timeframe')} {drawing.get('manual_line_id')} "
            f"configs={len(per_line)} best_R={(best or {}).get('expected_realized_r')}",
            flush=True,
        )

    outcomes_path = Path(args.outcomes)
    labels_path = Path(args.labels)
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, default=str) + "\n" for row in outcome_rows),
        encoding="utf-8",
    )
    labels_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, default=str) + "\n" for row in label_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "drawings": len(drawings),
        "configs": len(configs),
        "outcome_rows": len(outcome_rows),
        "label_rows": len(label_rows),
        "outcomes": str(outcomes_path),
        "labels": str(labels_path),
    }, indent=2))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
