"""Render a PNG snapshot of the chart at trade-trigger time.

Used by `_write_trade_snapshot` (watcher.py) to persist a visual record
next to the JSONL line. Useful for:

  - Human retrospective ("what did the chart look like when I fired
    trade #17?")
  - Future vision/VLA models that want image + metadata pairs
  - Generated reports

Implementation: matplotlib (server-side, no browser). Deliberately simple:
  - Last N candles as candlesticks (green/red bodies + wicks)
  - The user's drawn line (as a yellow line extending across the frame)
  - Horizontal markers for entry / stop / tp
  - No indicator overlays (MA, volume) in v1 — add later if useful

Path: data/logs/trade_snapshots/{SYMBOL}/{trade_id}.png

Non-blocking: any failure is swallowed by caller. Rendering takes 200-400ms
per call in practice; fine at the trigger rate we expect.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Use the non-interactive Agg backend so this works without a display.
os.environ.setdefault("MPLBACKEND", "Agg")


def _project_root() -> Path:
    try:
        from server.utils.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[2]


async def render_trade_snapshot_png(cond: Any, snapshot: dict[str, Any], n_bars: int = 120) -> Path | None:
    """Render a PNG of the chart around trigger time.

    Args:
        cond: ConditionalOrder-like object (needs symbol, timeframe, line anchors)
        snapshot: The trade snapshot dict (for trade_id + params)
        n_bars: How many bars before trigger to show

    Returns path to the written PNG, or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Rectangle
    except ImportError:
        return None

    symbol = cond.symbol
    timeframe = cond.timeframe
    trade_id = snapshot.get("trade_id")
    if not trade_id:
        return None

    # Fetch the candles. Best-effort.
    try:
        from server.data_service import get_ohlcv_with_df
        polars_df, _ = await get_ohlcv_with_df(
            symbol, timeframe, None, days=30,
            history_mode="fast_window",
            include_price_precision=False,
            include_render_payload=False,
        )
    except Exception:
        return None
    if polars_df is None or polars_df.is_empty():
        return None

    pdf = polars_df.tail(n_bars).to_pandas()
    if pdf.empty:
        return None

    # Coerce timestamps to matplotlib datenums
    import pandas as pd
    try:
        times = pd.to_datetime(pdf["open_time"]).astype("int64") // 10**9
    except Exception:
        times = pdf["open_time"].astype("int64")
    times_list = times.tolist()
    opens = pdf["open"].astype(float).tolist()
    highs = pdf["high"].astype(float).tolist()
    lows = pdf["low"].astype(float).tolist()
    closes = pdf["close"].astype(float).tolist()

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5), dpi=110, facecolor="#0e141f")
    ax.set_facecolor("#0e141f")

    # Bar width in x units (times are seconds since epoch; width = half bar size)
    if len(times_list) >= 2:
        bar_w = (times_list[1] - times_list[0]) * 0.7
    else:
        bar_w = 60.0

    for i, (t, o, h, l, c) in enumerate(zip(times_list, opens, highs, lows, closes)):
        up = c >= o
        color = "#00e676" if up else "#ff5252"
        # Wick
        ax.plot([t, t], [l, h], color=color, linewidth=0.9, zorder=2)
        # Body
        body_bot = min(o, c)
        body_h = max(abs(c - o), (h - l) * 0.01)
        rect = Rectangle((t - bar_w / 2, body_bot), bar_w, body_h,
                         facecolor=color, edgecolor=color, zorder=3)
        ax.add_patch(rect)

    # User line (extend across visible range)
    try:
        t_start = float(cond.t_start)
        t_end = float(cond.t_end)
        p_start = float(cond.price_start)
        p_end = float(cond.price_end)
    except Exception:
        t_start = t_end = p_start = p_end = None

    if None not in (t_start, t_end, p_start, p_end) and t_end > t_start:
        # Compute slope and extend to visible range
        slope = (p_end - p_start) / (t_end - t_start)
        x_min = min(times_list[0], t_start)
        x_max = max(times_list[-1], t_end)
        y_min = p_start + slope * (x_min - t_start)
        y_max = p_start + slope * (x_max - t_start)
        ax.plot([x_min, x_max], [y_min, y_max], color="#fbbf24", linewidth=2.0, zorder=5, label="line")
        # Anchor dots
        ax.scatter([t_start, t_end], [p_start, p_end], color="#fbbf24",
                   s=40, zorder=6, edgecolors="white", linewidths=1.2)

    # Entry / stop / tp markers as horizontal dashed lines spanning the last few bars
    params = snapshot.get("trade_params") or {}
    for key, color, label in (
        ("entry_price", "#38bdf8", "entry"),
        ("stop_price",  "#ff5252", "stop"),
        ("tp_price",    "#00e676", "tp"),
    ):
        v = params.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if v <= 0:
            continue
        ax.axhline(v, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        # Small label at the right edge
        ax.text(times_list[-1], v, f" {label} {v:.4f}",
                color=color, fontsize=8, va="center")

    # Styling
    ax.set_title(
        f"{symbol} {timeframe} · {snapshot.get('direction','?').upper()} "
        f"· touch #{snapshot.get('touch_number','?')} · {trade_id}",
        color="#d8dde8", fontsize=10, pad=10,
    )
    ax.tick_params(colors="#8a95a6", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2a3548")
    ax.grid(color="#1d2537", linewidth=0.5, alpha=0.6)

    # Format x axis as hh:mm
    def _fmt(x, pos):
        from datetime import datetime, timezone as _tz
        try:
            return datetime.fromtimestamp(x, tz=_tz.utc).strftime("%m-%d %H:%M")
        except Exception:
            return ""
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()

    # Write file
    out_dir = _project_root() / "data" / "logs" / "trade_snapshots" / symbol.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{trade_id}.png"
    try:
        fig.savefig(out_file, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return out_file


__all__ = ["render_trade_snapshot_png"]
