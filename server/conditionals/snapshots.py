"""Trade-snapshot storage and outcome backfill.

Storage model — two append-only JSONL files per symbol per month:

    data/logs/trade_snapshots/{SYMBOL}/{YYYYMM}.jsonl
    data/logs/trade_snapshots/{SYMBOL}/{YYYYMM}.outcomes.jsonl

The first (trades.jsonl, produced by watcher._write_trade_snapshot) is
written once when a conditional triggers. It is IMMUTABLE — we never
rewrite it. Each line contains full context at entry time:
  - line geometry, pattern stats, market context
  - trade params (entry/stop/tp/qty/leverage)
  - outcome.status == "pending"

The second (outcomes.jsonl, produced by backfill_pending_outcomes) is
written by a background job when trades close. Each line is one outcome
record keyed by `trade_id`:
  - exit_price, exit_ts, exit_reason, pnl, pnl_pct, mae_pct, mfe_pct

Reading: load both files, build a dict[trade_id -> latest outcome], and
join them when serving or training. Latest wins (a trade can be updated
multiple times as we refine MAE/MFE during open; final update is the
close).

Why split? JSONL is append-only. Rewriting the trades file to update an
outcome would require loading and rewriting the whole file on every
update — cheap at 100 rows, expensive at 100k. Splitting lets both files
stay append-only, and a join at read time is O(rows).

This module also provides:
  - backfill_pending_outcomes(): the watcher-callable job that polls
    Bitget for closed positions and emits outcome records
  - iter_snapshots_joined(): reader that yields (trade, outcome) tuples
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


def _project_root() -> Path:
    try:
        from server.utils.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except Exception:
        return Path(__file__).resolve().parents[2]


def _snapshots_dir() -> Path:
    return _project_root() / "data" / "logs" / "trade_snapshots"


def _paths_for(symbol: str, ts: int) -> tuple[Path, Path]:
    """(trades_file, outcomes_file) for a given symbol + timestamp."""
    ym = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m")
    d = _snapshots_dir() / symbol.upper()
    return d / f"{ym}.jsonl", d / f"{ym}.outcomes.jsonl"


def write_trade(snapshot: dict[str, Any]) -> Path:
    """Append a trade snapshot. Called at trigger time. Returns file path."""
    symbol = snapshot.get("symbol") or "UNKNOWN"
    ts = int(snapshot.get("ts") or 0)
    trades_file, _ = _paths_for(symbol, ts)
    trades_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trades_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return trades_file


def append_outcome(trade_id: str, symbol: str, ts_created: int, outcome: dict[str, Any]) -> Path:
    """Append an outcome record keyed by trade_id.

    Called by the backfill job when a trade closes (or progresses — we
    can also write progress updates for MAE/MFE tracking during open).
    Latest wins on read.
    """
    _, outcomes_file = _paths_for(symbol, ts_created)
    outcomes_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "trade_id": trade_id,
        "updated_at": int(datetime.now(tz=timezone.utc).timestamp()),
        **outcome,
    }
    with open(outcomes_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return outcomes_file


def iter_trades(symbol: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw trade snapshot dicts. Optionally filter by symbol."""
    root = _snapshots_dir()
    if not root.exists():
        return
    sym_dirs = [root / symbol.upper()] if symbol else [p for p in root.iterdir() if p.is_dir()]
    for d in sym_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.jsonl")):
            if f.name.endswith(".outcomes.jsonl"):
                continue
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue


def iter_outcomes(symbol: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw outcome records. Optionally filter by symbol."""
    root = _snapshots_dir()
    if not root.exists():
        return
    sym_dirs = [root / symbol.upper()] if symbol else [p for p in root.iterdir() if p.is_dir()]
    for d in sym_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.outcomes.jsonl")):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue


def latest_outcomes_by_trade(symbol: str | None = None) -> dict[str, dict[str, Any]]:
    """Build {trade_id -> latest outcome record} from the outcomes stream."""
    out: dict[str, dict[str, Any]] = {}
    for rec in iter_outcomes(symbol):
        tid = rec.get("trade_id")
        if not tid:
            continue
        prev = out.get(tid)
        if prev is None or int(rec.get("updated_at") or 0) >= int(prev.get("updated_at") or 0):
            out[tid] = rec
    return out


def iter_snapshots_joined(symbol: str | None = None) -> Iterator[dict[str, Any]]:
    """Yield trades with their latest outcome merged into snapshot['outcome'].

    If a trade has no outcome record, the inline outcome (from the
    original trade JSONL) is left as-is.
    """
    outcomes = latest_outcomes_by_trade(symbol)
    for trade in iter_trades(symbol):
        tid = trade.get("trade_id")
        if tid and tid in outcomes:
            merged = dict(trade)
            # Drop meta fields (trade_id, updated_at) from the outcome record
            # so it reads cleanly as a flat outcome block.
            clean = {k: v for k, v in outcomes[tid].items()
                     if k not in ("trade_id", "updated_at")}
            merged_outcome = dict(trade.get("outcome") or {})
            merged_outcome.update(clean)
            merged["outcome"] = merged_outcome
            yield merged
        else:
            yield trade


def list_pending_trades(symbol: str | None = None) -> list[dict[str, Any]]:
    """Trades whose joined outcome is still pending (i.e. no outcome record yet)."""
    pending: list[dict[str, Any]] = []
    for snap in iter_snapshots_joined(symbol):
        outcome = snap.get("outcome") or {}
        if (outcome.get("status") or "pending") == "pending":
            pending.append(snap)
    return pending


# ─────────────────────────────────────────────────────────────
# Backfill job — polls closed trades + MAE/MFE
# ─────────────────────────────────────────────────────────────
async def backfill_pending_outcomes() -> int:
    """Look at every pending trade, try to determine its outcome.

    Strategy (best-effort — data quality depends on Bitget retention and
    our own K-line coverage):

      1. For each pending trade, fetch 1m OHLCV from entry_ts to now.
      2. If price has hit TP first  → outcome=win,  exit=tp_price, reason=tp_hit
         If price has hit SL first  → outcome=loss, exit=stop_price, reason=sl_hit
         If neither hit yet         → update MAE/MFE only, keep status pending
      3. Compute MAE (worst drawdown) and MFE (best run) relative to entry
         in terms of ACCOUNT PnL percentage (uses leverage).
      4. Append outcome record (latest wins).

    Returns number of outcome records written.
    """
    trades = list_pending_trades()
    if not trades:
        return 0

    written = 0
    try:
        from server.data_service import get_ohlcv_with_df
    except Exception:
        # data_service unavailable — skip entirely
        return 0

    now = int(datetime.now(tz=timezone.utc).timestamp())

    for trade in trades:
        try:
            params = trade.get("trade_params") or {}
            entry = params.get("entry_price")
            stop = params.get("stop_price")
            tp = params.get("tp_price")
            leverage = float(params.get("leverage") or 1.0)
            direction = (trade.get("direction") or "").lower()
            if entry is None or stop is None or not direction:
                continue

            symbol = trade.get("symbol")
            trade_id = trade.get("trade_id")
            ts_created = int(trade.get("ts") or 0)
            if not (symbol and trade_id and ts_created):
                continue

            # Fetch 1m bars from entry to now (cap 7d to avoid huge loads)
            elapsed_sec = now - ts_created
            if elapsed_sec < 60:
                continue  # not enough time to even have one bar
            days = max(1, min(7, int(elapsed_sec / 86400) + 1))
            try:
                polars_df, _ = await get_ohlcv_with_df(
                    symbol, "1m", None, days,
                    history_mode="fast_window",
                    include_price_precision=False,
                    include_render_payload=False,
                )
            except Exception:
                continue
            if polars_df is None or polars_df.is_empty():
                continue

            pdf = polars_df.to_pandas()
            # Filter bars at/after entry
            pdf = pdf[pdf["open_time"].astype(int) // 1_000_000_000 >= ts_created] \
                if pdf["open_time"].dtype.kind == "M" else pdf
            if pdf.empty:
                continue

            highs = pdf["high"].astype(float).tolist()
            lows = pdf["low"].astype(float).tolist()
            closes = pdf["close"].astype(float).tolist()
            # Derive bar timestamps as seconds-since-epoch
            if "open_time" in pdf.columns:
                raw = pdf["open_time"].tolist()
                if raw and hasattr(raw[0], "timestamp"):
                    bar_ts = [int(v.timestamp()) for v in raw]
                else:
                    bar_ts = [int(v) for v in raw]
            else:
                bar_ts = [ts_created + 60 * i for i in range(len(closes))]

            # Track MAE / MFE and detect first hit of tp or sl
            is_long = direction == "long"
            mae_price = entry    # worst price (MAE)
            mfe_price = entry    # best  price
            hit_tp = False
            hit_sl = False
            exit_price = None
            exit_ts = None
            exit_reason = None

            for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
                # MAE / MFE update
                if is_long:
                    if l < mae_price:
                        mae_price = l
                    if h > mfe_price:
                        mfe_price = h
                else:
                    if h > mae_price:
                        mae_price = h
                    if l < mfe_price:
                        mfe_price = l

                # Hit detection — conservative: a bar counts as a hit if
                # its high>=tp (long) / low<=tp (short) etc. If BOTH hit
                # in the same bar, we pessimistically assume SL hit first.
                if is_long:
                    sl_hit_here = l <= stop
                    tp_hit_here = h >= tp if tp else False
                else:
                    sl_hit_here = h >= stop
                    tp_hit_here = l <= tp if tp else False

                if sl_hit_here and tp_hit_here:
                    hit_sl = True
                    exit_price = stop
                    exit_reason = "sl_hit_pessimistic"
                    exit_ts = bar_ts[i] if i < len(bar_ts) else None
                    break
                if sl_hit_here:
                    hit_sl = True
                    exit_price = stop
                    exit_reason = "sl_hit"
                    exit_ts = bar_ts[i] if i < len(bar_ts) else None
                    break
                if tp_hit_here:
                    hit_tp = True
                    exit_price = tp
                    exit_reason = "tp_hit"
                    exit_ts = bar_ts[i] if i < len(bar_ts) else None
                    break

            # Compute PnL in account-percentage terms
            def _pct_from_entry(price: float) -> float:
                if entry == 0:
                    return 0.0
                base = (price - entry) / entry * 100.0
                if not is_long:
                    base = -base
                return base * leverage  # leverage-amplified

            mae_pct = _pct_from_entry(mae_price)
            mfe_pct = _pct_from_entry(mfe_price)

            if hit_tp or hit_sl:
                final_pct = _pct_from_entry(exit_price)
                status = "win" if final_pct > 0 else ("breakeven" if final_pct == 0 else "loss")
                outcome = {
                    "status": status,
                    "exit_price": float(exit_price),
                    "exit_ts": int(exit_ts) if exit_ts else None,
                    "exit_reason": exit_reason,
                    "pnl_pct": round(final_pct, 4),
                    "mae_pct": round(mae_pct, 4),
                    "mfe_pct": round(mfe_pct, 4),
                }
            else:
                # Still open — progress update only (status stays pending)
                outcome = {
                    "status": "pending",
                    "mae_pct": round(mae_pct, 4),
                    "mfe_pct": round(mfe_pct, 4),
                }

            append_outcome(trade_id, symbol, ts_created, outcome)
            written += 1
        except Exception as e:
            print(f"[snapshots.backfill] trade {trade.get('trade_id')}: {e}", flush=True)
            continue

    return written


__all__ = [
    "write_trade",
    "append_outcome",
    "iter_trades",
    "iter_outcomes",
    "latest_outcomes_by_trade",
    "iter_snapshots_joined",
    "list_pending_trades",
    "backfill_pending_outcomes",
]
