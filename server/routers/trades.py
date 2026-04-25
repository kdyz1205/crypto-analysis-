"""Manual-line trade history endpoint.

Consolidates data from multiple sources to build a spreadsheet-friendly
row per CLOSED trade on a user-drawn line:

  - `data/user_drawings_ml.jsonl`   : position_closed_from_drawing events
                                       + user_drawing snapshots (features at
                                       creation time: RSI, ATR, ribbon score,
                                       bb_pct, trend_context, etc.)
  - `data/conditional_orders.json`  : cond-level metadata, event log (created,
                                       exchange_submitted, replanned, cancelled)
                                       used to derive bars-to-fill and
                                       replan_count.
  - `data/user_drawing_labels.jsonl`: user's optional ML labels
                                       (⭐好 / 〜一般 / ✗差)

Each returned row aims for ~30-50 feature columns for offline Excel /
pandas / ML workflows.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api", tags=["trades"])


_DATA = Path(__file__).resolve().parents[2] / "data"
_ML_FILE = _DATA / "user_drawings_ml.jsonl"
_LABELS_FILE = _DATA / "user_drawing_labels.jsonl"
_COND_FILE = _DATA / "conditional_orders.json"


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _safe_pct(numer: float, denom: float) -> float | None:
    try:
        if not denom:
            return None
        return float(numer) / float(denom)
    except Exception:
        return None


def _build_drawing_index(closes: list[dict]) -> dict[str, dict]:
    """Map manual_line_id -> richest 'user_drawing' snapshot (most-recent
    non-deleted) so every close row gets access to the line's features
    (RSI, ATR, BB%, ribbon_score, etc.) at or near creation time."""
    wanted_ids = {c.get("manual_line_id") for c in closes if c.get("manual_line_id")}
    idx: dict[str, dict] = {}
    for row in _iter_jsonl(_ML_FILE):
        if row.get("event") != "user_drawing":
            continue
        mid = row.get("manual_line_id")
        if mid not in wanted_ids:
            continue
        # Prefer latest 'created'/'updated' snapshot over 'deleted'
        stage = row.get("capture_stage") or row.get("stage") or ""
        if stage == "deleted" and mid in idx:
            continue
        idx[mid] = row  # later rows override earlier
    return idx


def _build_cond_index() -> dict[str, dict]:
    """Map manual_line_id -> cond dict (latest if multiple)."""
    idx: dict[str, dict] = {}
    try:
        arr = json.loads(_COND_FILE.read_text(encoding="utf-8"))
        if not isinstance(arr, list):
            return idx
    except Exception:
        return idx
    for cond in arr:
        mid = cond.get("manual_line_id")
        if not mid:
            continue
        prev = idx.get(mid)
        # keep the MOST RECENT cond per line_id
        if prev is None or (cond.get("created_at", 0) > prev.get("created_at", 0)):
            idx[mid] = cond
    return idx


def _build_labels_index() -> dict[str, str]:
    """Map manual_line_id -> latest user quality label."""
    idx: dict[str, str] = {}
    for row in _iter_jsonl(_LABELS_FILE):
        mid = row.get("manual_line_id")
        lbl = _normalize_user_label(row)
        if mid and lbl is not None:
            idx[mid] = lbl
    return idx


def _normalize_user_label(row: dict[str, Any]) -> str | None:
    """Normalize historical + current label payloads into one display field.

    Compatibility:
      - new manual label API: {quality: "good"|"bad"|"mediocre"}
      - older optimizer labels: {label_trade_win: 1|0}
      - oldest free-form rows:   {label: "..."}
    """
    quality = row.get("quality")
    if quality not in (None, ""):
        return str(quality)

    label_trade_win = row.get("label_trade_win")
    if label_trade_win not in (None, ""):
        try:
            win = int(label_trade_win)
            return "win" if win > 0 else "loss"
        except Exception:
            return str(label_trade_win)

    label = row.get("label")
    if label not in (None, ""):
        return str(label)

    return None


def _derive_cond_stats(cond: dict) -> dict[str, Any]:
    """Pull structural data from a cond record without repeating the join."""
    out: dict[str, Any] = {
        "cond_id": cond.get("conditional_id"),
        "cond_status": cond.get("status"),
        "exchange_order_id": cond.get("exchange_order_id"),
        "created_at": cond.get("created_at"),
        "triggered_at": cond.get("triggered_at"),
        "cancelled_at": cond.get("cancelled_at"),
        "cancel_reason": cond.get("cancel_reason"),
    }
    ord_ = cond.get("order") or {}
    out["tolerance_pct_of_line"] = ord_.get("tolerance_pct_of_line")
    out["stop_offset_pct_of_line"] = ord_.get("stop_offset_pct_of_line")
    out["rr_target"] = ord_.get("rr_target")
    out["notional_usd"] = ord_.get("notional_usd")
    out["equity_pct"] = ord_.get("equity_pct")
    out["leverage"] = ord_.get("leverage")
    out["direction"] = ord_.get("direction")
    out["order_kind"] = ord_.get("order_kind")
    out["reverse_enabled"] = ord_.get("reverse_enabled")

    events = cond.get("events") or []
    replan_count = sum(
        1 for e in events
        if e.get("kind") == "exchange_submitted"
        and "replan" in (e.get("message") or "").lower()
    )
    out["replan_count"] = replan_count

    # Earliest exchange_submitted ts - created_at → seconds-to-first-place
    placed = [e for e in events if e.get("kind") == "exchange_submitted"]
    if placed and cond.get("created_at"):
        out["seconds_to_first_place"] = max(
            0, int(placed[0].get("ts", 0) - cond["created_at"])
        )
    return out


def _line_features(dr: dict) -> dict[str, Any]:
    """Project the rich feature block out of a user_drawing snapshot."""
    out: dict[str, Any] = {
        "line_kind": dr.get("kind"),  # support / resistance
        "line_side": dr.get("side"),
        "line_slope": dr.get("slope"),
        "line_slope_per_bar": dr.get("slope_per_bar"),
        "line_span_bars": (dr.get("features") or {}).get("line_span_bars"),
        "line_age_bars": (dr.get("features") or {}).get("line_age_bars"),
        "anchor_distance_pct": dr.get("anchor_distance_pct"),
        "projected_price": dr.get("projected_price"),
        "dist_to_line_pct": dr.get("dist_to_line_pct"),
        "touch_count": dr.get("touch_count"),
        "recent_touch_count": dr.get("recent_touch_count"),
        "body_violation_count": dr.get("body_violation_count"),
        "wick_rejection_count": dr.get("wick_rejection_count"),
        "distance_to_line_atr": dr.get("distance_to_line_atr"),
        "htf_confluence_score": dr.get("htf_confluence_score"),
    }
    feat = dr.get("features") or {}
    for k in (
        "close", "atr", "atr_pct", "bb_width", "bb_pct", "ribbon_score",
        "ribbon_spread", "vol_ratio", "rsi", "ret_1", "ret_4", "ret_12",
        "trend_context", "wrong_side_close", "htf_ribbon_score",
        "htf_trend_context", "avg_rejection_atr", "max_rejection_atr",
        "wick_rejection_ratio", "last_touch_age_bars", "near_miss_count",
    ):
        if k in feat:
            out[f"feat_{k}"] = feat[k]
    return out


@router.get("/trades/manual-history")
async def api_manual_trade_history(
    limit: int = Query(200, ge=1, le=2000),
    symbol: str | None = Query(None),
):
    """Return flat rows of CLOSED manual-line trades, newest first.

    Each row is a spreadsheet-friendly dict combining close event +
    the line's feature snapshot + cond metadata.
    """
    # Pass 1: collect all close events
    closes: list[dict] = []
    for row in _iter_jsonl(_ML_FILE):
        if row.get("event") != "position_closed_from_drawing":
            continue
        if symbol and (row.get("symbol") or "").upper() != symbol.upper():
            continue
        closes.append(row)
    # Sort newest first, apply limit
    closes.sort(key=lambda r: r.get("ts", 0), reverse=True)
    closes = closes[:limit]

    # Enrich with indexed lookups (one pass each over the source files)
    dr_idx = _build_drawing_index(closes)
    cond_idx = _build_cond_index()
    label_idx = _build_labels_index()

    rows_out: list[dict[str, Any]] = []
    for c in closes:
        mid = c.get("manual_line_id") or ""
        entry = (c.get("features_at_close") or {}).get("entry_price")
        close_price = (c.get("features_at_close") or {}).get("close_price")
        margin = (c.get("features_at_close") or {}).get("margin_used")

        # Compute price_move_pct = (close - entry) / entry * sign(direction).
        # Use 'side' (long/short) to get directional PnL % consistent with the
        # caller's intent (positive = trade went their way).
        price_move_pct = None
        try:
            if entry and close_price:
                raw = (float(close_price) - float(entry)) / float(entry)
                side = (c.get("side") or "long").lower()
                price_move_pct = raw if side == "long" else -raw
        except Exception:
            pass

        row: dict[str, Any] = {
            "ts": c.get("ts"),
            "dt": c.get("dt"),
            "symbol": c.get("symbol"),
            "timeframe": c.get("timeframe"),
            "side": c.get("side"),  # long / short
            "manual_line_id": mid,
            "entry_price": entry,
            "exit_price": close_price,
            "price_move_pct": price_move_pct,
            "pnl_usd": c.get("pnl_usd"),
            "pnl_pct": c.get("pnl_pct"),
            "close_reason": c.get("close_reason"),
            "bars_to_fill": c.get("bars_to_fill"),
            "bars_held": c.get("bars_held"),
            "margin_used": margin,
            "clientOid": c.get("clientOid"),
            "user_label": label_idx.get(mid),
        }

        # Join line features
        dr = dr_idx.get(mid)
        if dr:
            row.update(_line_features(dr))

        # Join cond metadata
        cond = cond_idx.get(mid)
        if cond:
            row.update(_derive_cond_stats(cond))

        rows_out.append(row)

    # Assemble stable column order (best-effort — clients can ignore)
    priority = [
        "dt", "symbol", "timeframe", "side", "entry_price", "exit_price",
        "price_move_pct", "pnl_usd", "pnl_pct", "close_reason",
        "bars_to_fill", "bars_held", "margin_used", "notional_usd",
        "equity_pct", "leverage", "tolerance_pct_of_line",
        "stop_offset_pct_of_line", "rr_target", "replan_count",
        "seconds_to_first_place", "user_label",
    ]
    all_keys: list[str] = []
    seen: set[str] = set()
    for p in priority:
        if p not in seen:
            all_keys.append(p)
            seen.add(p)
    for r in rows_out:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    return {
        "ok": True,
        "count": len(rows_out),
        "columns": all_keys,
        "rows": rows_out,
    }



# ─── Chart marker endpoint (2026-04-25) ─────────────────────────────────
# Returns lightweight fill/exit events for a symbol so the chart can
# render OKX/Bitget-style buy/sell arrows. Independent of TF — every
# fill shows on every chart of that symbol because the timestamp maps
# to whichever bar contains it.

@router.get("/trades/fills")
def api_trades_fills(
    symbol: str = Query(..., description="e.g. ZECUSDT"),
    days: int = Query(30, ge=1, le=730, description="lookback days"),
):
    """Return fill events for chart markers.

    Sources merged:
      1. ConditionalOrderStore: conds with status in (triggered, filled,
         cancelled) → entry markers (and triggered_at for time)
      2. user_drawings_ml.jsonl: position_closed_from_drawing events
         → exit markers with side, exit_price, pnl

    Each fill row:
      {
        "time":  int (unix sec),
        "type":  'entry' | 'exit' | 'cancel',
        "side":  'long' | 'short',          # direction of position
        "price": float,
        "qty":   float | null,
        "pnl_usd":   float | null,          # only set on exits
        "close_reason": str | null,         # 'sl' | 'tp' | 'manual' | etc
        "exchange_order_id": str | null,
        "conditional_id": str | null,
      }
    """
    import time as _time
    sym = symbol.upper().replace("/", "")
    if not sym.endswith("USDT"):
        sym += "USDT"

    cutoff = int(_time.time()) - days * 86400
    out: list[dict[str, Any]] = []

    # ── Source 1: ConditionalOrderStore (entries + cancels) ────────────
    try:
        from ..conditionals.store import ConditionalOrderStore
        store = ConditionalOrderStore()
        conds = store.list_all(symbol=sym)
        for c in conds:
            # Entry marker: when the plan-order's trigger fired and a
            # position opened. We capture this on triggered_at.
            if getattr(c, "fill_price", None) and c.fill_price > 0 \
               and getattr(c, "triggered_at", None) and c.triggered_at >= cutoff:
                out.append({
                    "time": int(c.triggered_at),
                    "type": "entry",
                    "side": c.order.direction,
                    "price": float(c.fill_price),
                    "qty": float(c.fill_qty) if c.fill_qty else None,
                    "pnl_usd": None,
                    "close_reason": None,
                    "exchange_order_id": c.exchange_order_id,
                    "conditional_id": c.conditional_id,
                })
            # Cancel marker: optional, only mark if user wants visible
            # bookkeeping. For now we DON'T render cancels on chart
            # (would clutter); telegram covers cancels separately.
    except Exception as exc:
        print(f"[trades.fills] cond store err: {exc}")

    # ── Source 2: user_drawings_ml.jsonl (position close events) ──────
    if _ML_FILE.exists():
        for row in _iter_jsonl(_ML_FILE):
            kind = row.get("kind") or row.get("event_type")
            if kind != "position_closed_from_drawing":
                continue
            if str(row.get("symbol") or "").upper() != sym:
                continue
            close_ts = row.get("close_ts") or row.get("ts")
            if not close_ts or close_ts < cutoff:
                continue
            exit_price = row.get("exit_price") or row.get("close_price")
            if not exit_price:
                continue
            out.append({
                "time": int(close_ts),
                "type": "exit",
                "side": row.get("side") or row.get("direction"),
                "price": float(exit_price),
                "qty": row.get("close_qty") or row.get("qty"),
                "pnl_usd": row.get("pnl_usd"),
                "close_reason": row.get("close_reason"),
                "exchange_order_id": row.get("close_order_id"),
                "conditional_id": row.get("conditional_id"),
            })

    # Sort by time ascending (lightweight-charts requires sorted markers)
    out.sort(key=lambda r: r["time"])
    return {"symbol": sym, "fills": out, "count": len(out)}
