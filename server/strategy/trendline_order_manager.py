"""Trendline plan-order manager — persistent lines, bar-boundary updates.

Core logic (per TRENDLINE_TRADING_RULES.md):
  1. Lines are PERSISTENT once drawn. They don't disappear between scans.
  2. Each line has a plan order on Bitget (trigger = projected line + buffer).
  3. At each TF bar boundary, recalculate projection → cancel old → place new.
  4. Between boundaries, orders are untouched.
  5. A line is only REMOVED when BROKEN (price crosses through from wrong side).
  6. When plan order triggers → Bitget opens position with SL/TP attached.

Storage: data/trendline_active_orders.json (persistent across restarts)
"""
from __future__ import annotations

import time
import json
import traceback
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

ACTIVE_LINES_FILE = Path("data/trendline_active_orders.json")
COOLDOWN_FILE = Path("data/trendline_cooldowns.json")
RUNTIME_LOG_FILE = Path("data/logs/trendline_runtime.log")


def _runtime_log(msg: str) -> None:
    """Append a human-readable log line so the user can reason about why
    the runner rejected / skipped / placed without needing uvicorn stdout
    (which isn't captured to disk in the current deployment). Tail this
    file during debugging: `tail -f data/logs/trendline_runtime.log`.
    """
    try:
        RUNTIME_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RUNTIME_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    except Exception:
        pass

_broken_cooldowns: dict[str, float] = {}

TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
TF_PRIORITY = {"4h": 4, "1h": 3, "15m": 2, "5m": 1}


@dataclass
class ActiveLineOrder:
    symbol: str
    timeframe: str
    kind: str                   # support | resistance
    slope: float
    intercept: float
    anchor1_bar: int
    anchor2_bar: int
    bar_count: int              # total bars in data when line was found
    current_projected_price: float
    limit_price: float
    stop_price: float
    tp_price: float
    exchange_order_id: str
    created_ts: float
    last_updated_ts: float
    status: str                 # placed | filled | broken | stale | closed | cancelled
    line_ref_ts: float = 0.0     # timestamp where line_ref_price was computed
    line_ref_price: float = 0.0  # projected line price at line_ref_ts
    # Absolute anchor timestamps (ms since epoch) and prices at the two bars
    # that define this line — used to render the auto-triggered line on the
    # chart after a fill. 0 = not yet captured.
    anchor1_ts: int = 0
    anchor1_price: float = 0.0
    anchor2_ts: int = 0
    anchor2_price: float = 0.0
    # Set to True once this line has been written into manual_trendlines.json
    # (with source="auto_triggered") so the user sees the line. Prevents
    # double-writes on re-scans after the transition fires.
    persisted_as_drawing: bool = False


def _cooldown_key(symbol: str, timeframe: str, kind: str) -> str:
    return f"{symbol.upper()}|{timeframe}|{kind}"


def _cooldown_expiry(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("until_ts") or value.get("expires_at") or value.get("ts")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_cooldowns(now_ts: float | None = None) -> dict[str, dict[str, Any]]:
    now = float(now_ts if now_ts is not None else time.time())
    if not COOLDOWN_FILE.exists():
        return {}
    try:
        raw = json.loads(COOLDOWN_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        until = _cooldown_expiry(value)
        if until > now:
            if isinstance(value, dict):
                out[str(key)] = {**value, "until_ts": until}
            else:
                out[str(key)] = {"until_ts": until}
    return out


def _save_cooldowns(cooldowns: dict[str, dict[str, Any]]) -> None:
    COOLDOWN_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOLDOWN_FILE.write_text(json.dumps(cooldowns, indent=2, ensure_ascii=True), encoding="utf-8")


def _cooldown_bars_from_cfg(cfg: dict) -> int:
    raw = cfg.get("trendline_cooldown_bars_after_close", cfg.get("cooldown_bars_after_loss", 4))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 4


def mark_trendline_cooldown(
    symbol: str,
    timeframe: str,
    kind: str,
    *,
    bars: int = 4,
    reason: str = "",
    now_ts: float | None = None,
) -> float:
    """Block immediate re-entry for the same symbol/timeframe/line side."""
    bars = max(0, int(bars or 0))
    if bars <= 0:
        return 0.0
    now = float(now_ts if now_ts is not None else time.time())
    until = now + bars * TF_SECONDS.get(timeframe, 900)
    cooldowns = _load_cooldowns(now)
    cooldowns[_cooldown_key(symbol, timeframe, kind)] = {
        "until_ts": until,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "kind": kind,
        "bars": bars,
        "reason": reason,
        "set_ts": now,
    }
    _save_cooldowns(cooldowns)
    return until


def _cooldown_remaining(symbol: str, timeframe: str, kind: str, now_ts: float | None = None) -> float:
    now = float(now_ts if now_ts is not None else time.time())
    cooldowns = _load_cooldowns(now)
    entry = cooldowns.get(_cooldown_key(symbol, timeframe, kind))
    if not entry:
        return 0.0
    return max(0.0, float(entry.get("until_ts") or 0.0) - now)


def _load_active() -> list[ActiveLineOrder]:
    if not ACTIVE_LINES_FILE.exists():
        return []
    try:
        with open(ACTIVE_LINES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = []
        for d in data:
            d.setdefault("bar_count", 500)
            d.setdefault("line_ref_ts", d.get("last_updated_ts") or d.get("created_ts") or 0.0)
            d.setdefault("line_ref_price", d.get("current_projected_price") or 0.0)
            result.append(ActiveLineOrder(**d))
        return result
    except Exception as exc:
        print(f"[trendline_orders] active load err: {exc}", flush=True)
        traceback.print_exc()
        return []


def _save_active(orders: list[ActiveLineOrder]):
    ACTIVE_LINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_LINES_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(o) for o in orders], f, indent=2)


MANUAL_TRENDLINES_FILE = Path("data/manual_trendlines.json")


def _load_manual_trendlines() -> list[dict]:
    if not MANUAL_TRENDLINES_FILE.exists():
        return []
    try:
        raw = json.loads(MANUAL_TRENDLINES_FILE.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else []
    except Exception:
        return []


def _save_manual_trendlines(lines: list[dict]) -> None:
    MANUAL_TRENDLINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANUAL_TRENDLINES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(MANUAL_TRENDLINES_FILE)


def _auto_line_id(order: "ActiveLineOrder") -> str:
    return (
        f"auto-{order.symbol.upper()}-{order.timeframe}-{order.kind}"
        f"-{int(order.anchor1_ts or 0)}-{int(order.anchor2_ts or 0)}"
        f"-{str(order.exchange_order_id or '')[:16]}"
    )


TF_SECONDS_MAP = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}


def _persist_auto_line(order: "ActiveLineOrder", stage: str, *, close_reason: str | None = None,
                       fill_ts: float | None = None, close_ts: float | None = None) -> bool:
    """Write an auto-triggered trendline into manual_trendlines.json so the user
    sees why the order fired. Idempotent: if the id is already there, update
    label/updated_at instead of duplicating. Honors P4 — user can still delete
    by clicking × in the UI; we never auto-expire or revive deleted lines.

    stage ∈ {'placed','filled','closed'}:
      placed  — plan landed on Bitget, not yet triggered
      filled  — plan triggered, position open
      closed  — position closed (SL/TP/line break)

    Optional kwargs let the caller surface the 3 numbers the user asked for:
      close_reason — "sl_or_tp" / "line_broken" / "manual_close" / None
      fill_ts      — epoch seconds when the plan triggered
      close_ts     — epoch seconds when the position closed

    Stored on the line for user visibility:
      bars_to_fill   = (fill_ts - placed_ts) / TF_seconds   (how many bars before trigger)
      bars_held      = (close_ts - fill_ts) / TF_seconds    (how long the position was open)
      close_reason   — mirrored into label so the chart hover shows it

    Returns True if the line was newly inserted this call.
    """
    if not order.anchor1_ts or not order.anchor2_ts:
        # Missing anchor timestamps — nothing to draw. Don't fail the caller.
        return False
    if order.anchor1_price <= 0 or order.anchor2_price <= 0:
        return False
    line_id = _auto_line_id(order)
    # Stored as UNIX SECONDS to match the convention used everywhere else in
    # manual_trendlines.json (the frontend's fmtDateShort does `new Date(ts *
    # 1000)`, so feeding it milliseconds blows up the year label).
    now_ts = int(time.time())
    lines = _load_manual_trendlines()
    # Bound anchors so t_start <= t_end in storage (chart code assumes this).
    t1, p1 = int(order.anchor1_ts), float(order.anchor1_price)
    t2, p2 = int(order.anchor2_ts), float(order.anchor2_price)
    if t2 < t1:
        t1, t2 = t2, t1
        p1, p2 = p2, p1
    # Compute bar counts for the "how many TFs did it take" labels.
    tf_secs = TF_SECONDS_MAP.get(order.timeframe) or 3600
    placed_ts = float(order.created_ts or 0)
    bars_to_fill: int | None = None
    bars_held: int | None = None
    if fill_ts and placed_ts and fill_ts > placed_ts:
        bars_to_fill = max(0, int((fill_ts - placed_ts) / tf_secs))
    if close_ts and fill_ts and close_ts > fill_ts:
        bars_held = max(0, int((close_ts - fill_ts) / tf_secs))
    # Build a label that tells the user what/when/why at a glance.
    stage_tag = {
        "placed": "placed",
        "filled": "filled",
        "closed": "closed",
    }.get(stage, stage)
    extras = []
    if bars_to_fill is not None:
        extras.append(f"+{bars_to_fill}bar→fill")
    if bars_held is not None:
        extras.append(f"{bars_held}bar held")
    if close_reason:
        # Short, human-readable close reason for the label.
        reason_short = {
            "sl_or_tp": "SL/TP",
            "line_broken": "line-break",
            "manual_close": "manual",
            "plan_triggered_and_closed": "SL/TP",
        }.get(close_reason, close_reason)
        extras.append(reason_short)
    extra_str = (" · " + " · ".join(extras)) if extras else ""
    label = f"Auto · {order.kind[:3]} · {stage_tag}{extra_str}"
    # Build a structured notes block. Each field on its own line so hover /
    # side-panel can parse if needed.
    notes_lines = [
        f"order_id={order.exchange_order_id}",
        f"entry={order.limit_price:.6f}",
        f"stop={order.stop_price:.6f}",
        f"tp={order.tp_price:.6f}",
        f"stage={stage}",
    ]
    if placed_ts:
        notes_lines.append(f"placed_ts={int(placed_ts)}")
    if fill_ts:
        notes_lines.append(f"fill_ts={int(fill_ts)}")
    if close_ts:
        notes_lines.append(f"close_ts={int(close_ts)}")
    if bars_to_fill is not None:
        notes_lines.append(f"bars_to_fill={bars_to_fill}")
    if bars_held is not None:
        notes_lines.append(f"bars_held={bars_held}")
    if close_reason:
        notes_lines.append(f"close_reason={close_reason}")
    notes = " ".join(notes_lines)
    for i, existing in enumerate(lines):
        if existing.get("manual_line_id") == line_id:
            # Update stage label + updated_at, keep user-edited fields intact.
            lines[i]["updated_at"] = now_ts
            lines[i]["label"] = label
            lines[i]["notes"] = notes
            # Carry structured metadata into the row so the panel + ML pipeline
            # can read them without string parsing.
            if fill_ts:
                lines[i]["auto_fill_ts"] = int(fill_ts)
            if close_ts:
                lines[i]["auto_close_ts"] = int(close_ts)
            if bars_to_fill is not None:
                lines[i]["auto_bars_to_fill"] = bars_to_fill
            if bars_held is not None:
                lines[i]["auto_bars_held"] = bars_held
            if close_reason:
                lines[i]["auto_close_reason"] = close_reason
            _save_manual_trendlines(lines)
            return False
    new_entry = {
        "manual_line_id": line_id,
        "symbol": order.symbol.upper(),
        "timeframe": order.timeframe,
        "side": order.kind,          # 'support' | 'resistance'
        "source": "auto_triggered",
        "t_start": t1,
        "t_end": t2,
        "price_start": p1,
        "price_end": p2,
        "extend_left": False,
        "extend_right": True,        # auto lines commonly extended forward
        "locked": True,              # prevents accidental drag/delete misfire
        "label": label,
        "notes": notes,
        "comparison_status": "uncompared",
        "override_mode": "display_only",
        "nearest_auto_line_id": None,
        "slope_diff": None,
        "projected_price_diff": None,
        "overlap_ratio": None,
        "created_at": now_ts,
        "updated_at": now_ts,
        "line_width": 1.8,
        # Trade-telemetry: how many bars to fill / how long held / why closed
        "auto_placed_ts": int(placed_ts) if placed_ts else None,
        "auto_fill_ts": int(fill_ts) if fill_ts else None,
        "auto_close_ts": int(close_ts) if close_ts else None,
        "auto_bars_to_fill": bars_to_fill,
        "auto_bars_held": bars_held,
        "auto_close_reason": close_reason,
    }
    lines.append(new_entry)
    _save_manual_trendlines(lines)
    print(
        f"[trendline_orders] auto-line persisted {line_id} stage={stage} "
        f"{order.symbol} {order.timeframe} {order.kind}",
        flush=True,
    )
    return True


def _is_bar_boundary(tf: str) -> bool:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    m, s, h = now.minute, now.second, now.hour
    if tf == "5m":
        return (m % 5 == 0) and s < 90
    elif tf == "15m":
        return (m % 15 == 0) and s < 90
    elif tf == "1h":
        return m == 0 and s < 90
    elif tf == "4h":
        return (h % 4 == 0) and m == 0 and s < 90
    elif tf == "1d":
        return h == 0 and m == 0 and s < 90
    return True


def _elapsed_tf_bars_since(previous_ts: float, tf: str, now_ts: float | None = None) -> int:
    """Return how many TF candle boundaries have passed since previous_ts."""
    if previous_ts <= 0:
        return 0
    bar_dur = TF_SECONDS.get(tf, 300)
    now_val = float(now_ts if now_ts is not None else time.time())
    return max(0, int(now_val // bar_dur) - int(float(previous_ts) // bar_dur))


def _buffer_fraction_for_tf(tf: str, cfg: dict) -> float:
    """Return buffer as a fraction. Config values are documented percentages."""
    raw_buffer_pct = float(cfg.get("buffer_pct", 0.10))
    tf_buffer_pct = float((cfg.get("tf_buffer") or {}).get(tf, raw_buffer_pct))
    return tf_buffer_pct / 100.0


def _stop_offset_fraction_for_tf(tf: str, cfg: dict) -> float:
    """Return the extra stop offset beyond the line as a fraction."""
    raw_stop_offset_pct = float(cfg.get("stop_offset_pct", 0.0))
    tf_stop_offset_pct = float((cfg.get("tf_stop_offset") or {}).get(tf, raw_stop_offset_pct))
    return max(0.0, tf_stop_offset_pct) / 100.0


def _trade_prices_for_line(kind: str, projected_price: float, tf: str, cfg: dict, rr: float) -> tuple[float, float, float, float, float]:
    """Return entry trigger, stop, target, buffer fraction, stop-offset fraction."""
    buffer_pct = _buffer_fraction_for_tf(tf, cfg)
    stop_offset_pct = _stop_offset_fraction_for_tf(tf, cfg)
    if kind == "support":
        limit_px = projected_price * (1 + buffer_pct)
        stop_px = projected_price * (1 - stop_offset_pct)
        risk = limit_px - stop_px
        tp_px = limit_px + risk * rr
    else:
        limit_px = projected_price * (1 - buffer_pct)
        stop_px = projected_price * (1 + stop_offset_pct)
        risk = stop_px - limit_px
        tp_px = limit_px - risk * rr
    return limit_px, stop_px, tp_px, buffer_pct, stop_offset_pct


def _qty_for_risk(
    *,
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    leverage: int,
    max_position_pct: float,
) -> tuple[float, float, bool]:
    stop_distance = abs(entry_price - stop_price)
    if equity <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_distance <= 0:
        return 0.0, 0.0, False

    risk_usd = equity * risk_pct
    raw_qty = risk_usd / stop_distance
    max_notional = equity * max_position_pct * leverage
    if max_notional <= 0:
        return raw_qty, risk_usd, False

    max_qty = max_notional / entry_price
    capped = raw_qty > max_qty
    return min(raw_qty, max_qty), risk_usd, capped


def _would_trigger_immediately(kind: str, current_price: float, limit_price: float) -> bool:
    """A passive line order must wait for price to come to it."""
    if current_price <= 0 or limit_price <= 0:
        return False
    if kind == "support":
        return current_price <= limit_price
    return current_price >= limit_price


def _is_through_stop(kind: str, current_price: float, stop_price: float) -> bool:
    """Return True only when the line idea is already invalidated."""
    if current_price <= 0 or stop_price <= 0:
        return False
    if kind == "support":
        return current_price <= stop_price
    return current_price >= stop_price


def _price_from_cfg(symbol: str, cfg: dict) -> float:
    prices = cfg.get("prices") or {}
    raw = prices.get(symbol) or prices.get(symbol.upper()) or prices.get(symbol.lower()) or 0
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _broken_status(kind: str, current_price: float, stop_price: float) -> str | None:
    """Return terminal status only if price has crossed the stop side."""
    if _is_through_stop(kind, current_price, stop_price):
        return "broken"
    return None


def _history_rows(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, dict):
        rows = data.get("list") or data.get("entrustedList") or data.get("orderList") or []
    else:
        rows = data or []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _row_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw in (None, "", 0, "0"):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value != 0:
            return value
    return 0.0


def _row_ts(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        raw = row.get(key)
        if raw in (None, "", 0, "0"):
            continue
        try:
            ts = int(float(raw))
        except (TypeError, ValueError):
            continue
        if ts > 10_000_000_000:
            ts //= 1000
        if ts > 0:
            return ts
    return 0


def _direction_for_order(order: ActiveLineOrder) -> str:
    return "long" if order.kind == "support" else "short"


def _history_side(row: dict[str, Any]) -> str:
    return str(row.get("holdSide") or row.get("posSide") or row.get("side") or "").lower()


async def _find_recent_closed_position(
    adapter: Any,
    order: ActiveLineOrder,
    mode: str,
    now_ts: float,
) -> dict[str, Any] | None:
    """Best-effort check for plan orders that triggered and stopped quickly.

    Bitget removes a normal_plan order as soon as it triggers. If the preset
    SL closes the position before our next 10s maintenance tick, there is no
    open position to detect, so the only exchange evidence is position
    history.
    """
    start_ms = int(max(0, float(order.created_ts) - 120) * 1000)
    end_ms = int((float(now_ts) + 60) * 1000)
    try:
        resp = await adapter._bitget_request(
            "GET",
            "/api/v2/mix/position/history-position",
            mode=mode,
            params={
                "symbol": order.symbol.upper(),
                "productType": "USDT-FUTURES",
                "startTime": str(start_ms),
                "endTime": str(end_ms),
                "limit": "20",
            },
        )
    except Exception as exc:
        print(f"[trendline_orders] history sync {order.symbol} err: {exc}", flush=True)
        return None
    if resp.get("code") != "00000":
        return None

    want_side = _direction_for_order(order)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in _history_rows(resp):
        if str(row.get("symbol") or "").upper() != order.symbol.upper():
            continue
        side = _history_side(row)
        if side and side != want_side:
            continue
        close_ts = _row_ts(row, "uTime", "updateTime", "utime", "closeTime", "cTime")
        open_ts = _row_ts(row, "openTime", "openTimestamp", "cTime", "ctime", "createdTime", "createTime")
        evidence_ts = close_ts or open_ts
        if evidence_ts <= 0:
            continue
        if evidence_ts < int(float(order.created_ts) - 120):
            continue
        candidates.append((evidence_ts, row))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _log_closed_from_history(order: ActiveLineOrder, row: dict[str, Any], reason: str) -> None:
    direction = _direction_for_order(order)
    open_price = _row_float(row, "openPriceAvg", "openAvgPrice", "averageOpenPrice")
    close_price = _row_float(row, "closePriceAvg", "closeAvgPrice", "averageClosePrice")
    pnl = _row_float(row, "netProfit", "achievedProfits", "net_pnl", "pnl")
    margin = abs(_row_float(row, "margin", "positionMargin", "openMargin"))
    pnl_pct = pnl / margin if margin > 0 else 0.0
    close_ts = _row_ts(row, "uTime", "updateTime", "utime", "closeTime", "cTime")
    open_ts = _row_ts(row, "openTime", "openTimestamp", "cTime", "ctime", "createdTime", "createTime")
    hold_seconds = max(0, close_ts - open_ts) if close_ts and open_ts else 0
    try:
        from server.strategy.trade_log import log_close
        log_close(
            str(order.exchange_order_id or ""),
            order.symbol.upper(),
            direction,
            close_price,
            pnl,
            pnl_pct,
            reason=reason,
            tf=order.timeframe,
            entry_price=open_price,
        )
    except Exception as exc:
        print(f"[trendline_orders] close log err {order.symbol}: {exc}", flush=True)
    try:
        from server.strategy.ml_trade_db import log_position_closed
        log_position_closed(
            symbol=order.symbol.upper(),
            direction=direction,
            tf=order.timeframe,
            entry_price=open_price,
            close_price=close_price,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            hold_seconds=hold_seconds,
            order_id=str(order.exchange_order_id or ""),
        )
    except Exception as exc:
        print(f"[trendline_orders] ml close log err {order.symbol}: {exc}", flush=True)


def _is_line_broken(kind: str, projected_price: float, current_close: float, atr: float) -> bool:
    """A line is broken when price closes decisively through it."""
    if projected_price <= 0:
        return True
    if kind == "support":
        return current_close < projected_price - 0.5 * atr
    else:  # resistance
        return current_close > projected_price + 0.5 * atr


async def update_trendline_orders(
    new_signals: list[dict],
    current_bar_index: int,
    cfg: dict,
):
    """Called each scan cycle.

    Two jobs:
      A. Add NEW lines (from new_signals) that we don't already track.
      B. For EXISTING lines: at bar boundary → update order coordinates.
         Between boundaries → do nothing. If broken → cancel + remove.
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"placed": 0, "updated": 0, "cancelled": 0}

    active = _load_active()
    placed = updated = cancelled = 0
    now = time.time()
    rr = cfg.get("rr", 15.0)
    leverage = int(cfg.get("leverage", 30))
    equity = float(cfg.get("equity", 50.0))
    risk_pct = float(cfg.get("risk_pct", 0.03))
    max_position_pct = float(cfg.get("max_position_pct", 0.50))
    mode = cfg.get("mode", "live")
    held_symbols = {str(s).upper() for s in (cfg.get("held_symbols") or set())}
    position_guard_ok = True
    try:
        held_symbols |= await adapter.get_open_position_symbols(mode)
    except Exception as exc:
        position_guard_ok = False
        print(f"[trendline_orders] position guard failed; skipping new orders: {exc}", flush=True)

    local_placed_ids = {
        str(o.exchange_order_id or "")
        for o in active
        if o.status == "placed" and str(o.exchange_order_id or "")
    }
    pending_plan_ids_by_symbol: dict[str, set[str]] = {}
    pending_sync_ok = False
    try:
        pending_rows = await adapter.get_pending_plan_orders(mode, plan_type="normal_plan")
        for row in pending_rows:
            row_symbol = str(row.get("symbol") or "").upper()
            row_id = str(row.get("orderId") or row.get("order_id") or "")
            if row_symbol and row_id:
                pending_plan_ids_by_symbol.setdefault(row_symbol, set()).add(row_id)
        pending_sync_ok = True
    except Exception as exc:
        print(f"[trendline_orders] pending normal_plan sync failed: {exc}", flush=True)

    if pending_sync_ok:
        for row in pending_rows:
            row_id = str(row.get("orderId") or row.get("order_id") or "")
            row_symbol = str(row.get("symbol") or "").upper()
            client_oid = str(row.get("clientOid") or "")
            if not row_id or not row_symbol or row_id in local_placed_ids:
                continue
            if not client_oid.startswith("tl_"):
                continue
            cancel_resp = await adapter.cancel_plan_order(
                row_symbol,
                row_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                cancelled += 1
                pending_plan_ids_by_symbol.get(row_symbol, set()).discard(row_id)
                print(
                    f"[trendline_orders] ORPHAN-PLAN-CANCEL {row_symbol}: "
                    f"exchange normal_plan order_id={row_id} not managed locally",
                    flush=True,
                )
            else:
                print(f"[trendline_orders] orphan plan cancel {row_symbol} failed: {cancel_resp}", flush=True)

    # Clean up regular tl_ orders left by the brief post-only implementation.
    # Automatic scanner orders must be normal_plan so they do not reserve
    # margin across the whole market.
    try:
        legacy_regular_rows = await adapter.get_pending_orders(mode)
        for row in legacy_regular_rows:
            row_id = str(row.get("orderId") or row.get("order_id") or "")
            row_symbol = str(row.get("symbol") or "").upper()
            client_oid = str(row.get("clientOid") or "")
            if not row_id or not row_symbol or not client_oid.startswith("tl_"):
                continue
            cancel_resp = await adapter.cancel_order(row_symbol, row_id, mode)
            if cancel_resp.get("ok"):
                cancelled += 1
                print(
                    f"[trendline_orders] LEGACY-REGULAR-CANCEL {row_symbol}: "
                    f"regular order_id={row_id} not allowed for automatic trendline",
                    flush=True,
                )
            else:
                print(f"[trendline_orders] legacy regular cancel {row_symbol} failed: {cancel_resp}", flush=True)
    except Exception as exc:
        print(f"[trendline_orders] legacy regular order cleanup failed: {exc}", flush=True)

    # Index only exchange-live local orders by (symbol, tf, kind). Historical
    # stale/filled records are kept for evidence but must not block fresh lines.
    inactive_orders = [o for o in active if o.status != "placed"]
    existing = {}
    for o in active:
        if o.status == "placed":
            existing[(o.symbol, o.timeframe, o.kind)] = o
    tracked_symbols = {
        o.symbol.upper()
        for o in active
        if o.status == "placed"
    } | held_symbols
    # NOTE (2026-04-19, design intent): we intentionally do NOT add
    # `pending_plan_ids_by_symbol.keys()` to tracked_symbols. The user wants
    # manual-line plans and the auto runner to coexist on the same symbol so
    # the two streams of outcomes become a side-by-side training signal —
    # "what did the human pick vs. what did the scanner pick" for the same
    # market state. See "Bidirectional learning" memory note.
    # Consequence: two plan orders (one `replan_` / `line_` / `cond_`, one
    # `tl_`) may coexist on the same symbol. On Bitget in hedge mode this is
    # fine (separate long/short hold sides). If one triggers first, the other
    # remains pending until its own trigger fires. ClientOid tagging keeps
    # the downstream attribution clean.

    # --- A. Add new lines not already tracked ---
    sorted_signals = sorted(
        new_signals,
        key=lambda s: (TF_PRIORITY.get(str(s.get("timeframe", "")), 0), str(s.get("symbol", ""))),
        reverse=True,
    )
    new_keys: set[tuple[str, str, str]] = set()
    for sig in sorted_signals:
        if not position_guard_ok:
            break
        key = (sig["symbol"], sig["timeframe"], sig["kind"])
        if key in existing:
            continue  # already tracking this line

        sym, tf, kind = key
        sym_upper = sym.upper()
        if sym_upper in tracked_symbols:
            _msg = f"SKIP {sym} {tf}: symbol already has active/order position"
            print(f"[trendline_orders] {_msg}", flush=True)
            _runtime_log(_msg)
            continue
        cooldown_left = _cooldown_remaining(sym_upper, tf, kind, now)
        if cooldown_left > 0:
            _msg = f"SKIP {sym} {tf} {kind}: cooldown active {int(cooldown_left)}s after recent close/stop"
            print(f"[trendline_orders] {_msg}", flush=True)
            _runtime_log(_msg)
            continue

        slope = sig["slope"]
        intercept = sig["intercept"]
        bar_count = sig.get("bar_count", 500)
        proj = slope * (bar_count - 1) + intercept
        if proj <= 0:
            continue

        # No cooldown needed — plan order + preset SL handles broken lines

        # Use per-TF buffer for the trigger and a tiny configurable offset
        # beyond the line for stop confirmation.
        limit_px, stop_px, tp_px, buffer_pct, stop_offset_pct = _trade_prices_for_line(kind, proj, tf, cfg, rr)
        direction = "long" if kind == "support" else "short"

        current_close = _price_from_cfg(sym_upper, cfg) or float(sig.get("entry_price", 0) or 0)
        if _is_through_stop(kind, float(current_close or 0), stop_px):
            _msg = f"SKIP {sym} {tf} {kind}: current={float(current_close):.8f} already through stop={stop_px:.8f}"
            print(f"[trendline_orders] {_msg}", flush=True)
            _runtime_log(_msg)
            continue

        per_tf_risk = float((cfg.get("tf_risk") or {}).get(tf, risk_pct))
        qty, risk_usd, capped = _qty_for_risk(
            equity=equity,
            risk_pct=per_tf_risk,
            entry_price=limit_px,
            stop_price=stop_px,
            leverage=leverage,
            max_position_pct=max_position_pct,
        )
        if qty <= 0:
            continue

        # Place a Bitget normal_plan market trigger order. This keeps the
        # automatic all-market scanner from reserving margin for every line.
        # The trigger is the line +/- buffer; once triggered, Bitget opens
        # with market execution so the bounce entry does not miss because of
        # an unfilled limit child order.
        try:
            try:
                await adapter._bitget_request(
                    "POST", "/api/v2/mix/account/set-leverage",
                    mode=mode,
                    body={"symbol": sym.upper(), "productType": "USDT-FUTURES",
                          "marginCoin": "USDT", "leverage": str(leverage)},
                )
            except Exception as exc:
                print(f"[trendline_orders] set-leverage {sym} warn: {exc}", flush=True)

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{sym}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}",
                line_id="",
                client_order_id=f"tl_{sym[:10]}_{now_ts}",
                symbol=sym.upper(), timeframe=tf, side=direction,
                order_type="market", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_market",
                created_at_bar=bar_count - 1, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode=mode, trigger_price=limit_px)

            if resp.get("ok"):
                order_id = resp.get("exchange_order_id", "")
                existing[key] = ActiveLineOrder(
                    symbol=sym, timeframe=tf, kind=kind,
                    slope=slope, intercept=intercept,
                    anchor1_bar=sig.get("anchor1_bar", 0),
                    anchor2_bar=sig.get("anchor2_bar", 0),
                    bar_count=bar_count,
                    current_projected_price=proj,
                    limit_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                    exchange_order_id=order_id,
                    created_ts=now, last_updated_ts=now, status="placed",
                    line_ref_ts=now, line_ref_price=proj,
                    anchor1_ts=int(sig.get("anchor1_ts", 0) or 0),
                    anchor2_ts=int(sig.get("anchor2_ts", 0) or 0),
                    anchor1_price=float(sig.get("anchor1_price", 0) or 0.0),
                    anchor2_price=float(sig.get("anchor2_price", 0) or 0.0),
                    persisted_as_drawing=False,
                )
                new_keys.add(key)
                tracked_symbols.add(sym_upper)
                placed += 1
                _msg = (f"NEW {direction} {sym} {tf} {kind} @ {limit_px:.6f} "
                        f"SL={stop_px:.6f} TP={tp_px:.6f} risk=${risk_usd:.4f}"
                        f"{' capped' if capped else ''}")
                print(f"[trendline_orders] {_msg}", flush=True)
                _runtime_log(_msg)
                try:
                    # Extract ML gate scores (computed upstream in mar_bb_runner
                    # when this signal was constructed) so every plan row lands
                    # in the jsonl with its predicted line/trade win probs.
                    _gate = sig.get("model_gate") or {}
                    _line_q = _gate.get("line_quality_prob")
                    _trade_w = _gate.get("trade_win_prob")
                    from server.strategy.trade_log import log_trade
                    log_trade(
                        symbol=sym, timeframe=tf, strategy="trendline",
                        direction=direction, entry_price=limit_px,
                        stop_price=stop_px, tp_price=tp_px,
                        size_usd=qty * limit_px, leverage=leverage,
                        order_id=order_id, status="plan_market_placed",
                        buffer_pct=buffer_pct, risk_usd=risk_usd,
                        line_quality_prob=_line_q, trade_win_prob=_trade_w,
                    )
                    from server.strategy.ml_trade_db import log_plan_placed
                    log_plan_placed(
                        symbol=sym, tf=tf, direction=direction, kind=kind,
                        slope=slope, intercept=intercept, bar_count=bar_count,
                        entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                        buffer_pct=buffer_pct, rr=rr, leverage=leverage,
                        quantity=qty, order_id=order_id,
                        pivot1_bar=sig.get("anchor1_bar", 0),
                        pivot2_bar=sig.get("anchor2_bar", 0),
                        risk_usd=risk_usd, capped=capped,
                        stop_offset_pct=stop_offset_pct,
                        line_quality_prob=_line_q, trade_win_prob=_trade_w,
                    )
                except Exception as exc:
                    print(f"[trendline_orders] plan log err {sym}: {exc}", flush=True)
            else:
                _msg = f"REJECTED {sym} {tf} {kind}: {resp.get('reason')}"
                print(f"[trendline_orders] {_msg}", flush=True)
                _runtime_log(_msg)
        except Exception as e:
            _msg = f"PLACE-ERR {sym} {tf} {kind}: {type(e).__name__}: {e}"
            print(f"[trendline_orders] {_msg}", flush=True)
            _runtime_log(_msg)

    # --- B. Update existing lines at bar boundaries / remove broken ---
    surviving = list(inactive_orders)
    for key, order in existing.items():
        if order.status != "placed":
            surviving.append(order)
            continue
        if key in new_keys:
            surviving.append(order)
            continue

        if order.symbol.upper() in held_symbols:
            order.status = "filled"
            try:
                # Best estimate for fill time: "now" if we have no better source.
                # The adapter's position row likely has an `openTime` / `cTime`;
                # the sync loop below sources it for closed rows; for first
                # detection as filled, approximate with the current tick.
                _fill_ts = now
                if _persist_auto_line(order, stage="filled", fill_ts=_fill_ts):
                    order.persisted_as_drawing = True
            except Exception as _exc:
                print(f"[trendline_orders] persist-auto-line err (filled) {order.symbol}: {_exc}", flush=True)
            surviving.append(order)
            print(f"[trendline_orders] FILLED-SYNC {order.symbol} {order.timeframe}: held position detected", flush=True)
            continue

        if pending_sync_ok:
            live_ids = pending_plan_ids_by_symbol.get(order.symbol.upper(), set())
            order_id = str(order.exchange_order_id or "")
            if order_id and order_id not in live_ids:
                closed_row = await _find_recent_closed_position(adapter, order, mode, now)
                if closed_row is not None:
                    order.status = "closed"
                    try:
                        # Pull timing + reason from the Bitget history-position
                        # row so the drawn line carries the full trade story.
                        _fill_ts = _row_ts(closed_row, "openTime", "ctime", "cTime", "createdTime")
                        _close_ts = _row_ts(closed_row, "closeTime", "utime", "uTime", "updateTime")
                        _reason = "plan_triggered_and_closed"
                        # If the exchange tags it differently, prefer that.
                        _raw_reason = str(closed_row.get("closeReason") or "").strip().lower()
                        if _raw_reason:
                            _reason = _raw_reason
                        _persist_auto_line(
                            order,
                            stage="closed",
                            fill_ts=_fill_ts or None,
                            close_ts=_close_ts or None,
                            close_reason=_reason,
                        )
                    except Exception as _exc:
                        print(f"[trendline_orders] persist-auto-line err (closed) {order.symbol}: {_exc}", flush=True)
                    surviving.append(order)
                    _log_closed_from_history(order, closed_row, reason="plan_triggered_and_closed")
                    mark_trendline_cooldown(
                        order.symbol,
                        order.timeframe,
                        order.kind,
                        bars=_cooldown_bars_from_cfg(cfg),
                        reason="plan_triggered_and_closed",
                        now_ts=now,
                    )
                    pnl = _row_float(closed_row, "netProfit", "achievedProfits", "net_pnl", "pnl")
                    print(
                        f"[trendline_orders] CLOSED-SYNC {order.symbol} {order.timeframe}: "
                        f"normal_plan disappeared and recent position history matched; "
                        f"PnL=${pnl:+.4f}; cooldown set",
                        flush=True,
                    )
                    continue
                order.status = "stale"
                surviving.append(order)
                print(
                    f"[trendline_orders] STALE {order.symbol} {order.timeframe}: "
                    f"exchange missing normal_plan order_id={order_id}; local line disabled",
                    flush=True,
                )
                continue

        tf = order.timeframe
        # Recalculate projection on TF candle boundaries, not "duration since
        # placement". Example: a 15m order placed at 12:38 moves at 12:45,
        # not at 12:53.
        bars_elapsed = _elapsed_tf_bars_since(order.last_updated_ts, tf, now)
        new_bar_index = order.bar_count - 1 + bars_elapsed
        proj = order.slope * new_bar_index + order.intercept

        if proj <= 0:
            continue

        # No broken-line detection needed — if price crosses through,
        # the plan order triggers and preset SL handles it automatically.

        # Recalculate target coordinates first. Broken/touched orders must be
        # cancelled immediately; only healthy orders wait for the next TF bar
        # before moving.
        limit_px, stop_px, tp_px, buffer_pct, stop_offset_pct = _trade_prices_for_line(order.kind, proj, tf, cfg, rr)
        direction = "long" if order.kind == "support" else "short"

        current_price = _price_from_cfg(order.symbol, cfg)
        broken_status = _broken_status(order.kind, current_price, stop_px)
        if broken_status:
            try:
                cancel_resp = await adapter.cancel_plan_order(
                    order.symbol.upper(),
                    order.exchange_order_id,
                    mode,
                    plan_type="normal_plan",
                )
                if cancel_resp.get("ok"):
                    order.status = broken_status
                    surviving.append(order)
                    cancelled += 1
                    mark_trendline_cooldown(
                        order.symbol,
                        tf,
                        order.kind,
                        bars=_cooldown_bars_from_cfg(cfg),
                        reason=broken_status,
                        now_ts=now,
                    )
                    print(
                        f"[trendline_orders] CANCEL {order.symbol} {tf}: "
                        f"current={current_price:.8f} already through stop={stop_px:.8f}; "
                        f"status={broken_status}",
                        flush=True,
                    )
                    continue
                print(f"[trendline_orders] cancel touched {order.symbol} failed: {cancel_resp}", flush=True)
            except Exception as e:
                print(f"[trendline_orders] cancel touched {order.symbol} err: {e}", flush=True)
            surviving.append(order)
            continue

        # Not touched/broken -> move only after at least one full TF bar elapsed.
        # This avoids missing the 90s wall-clock window and prevents repeated
        # cancel/re-place loops inside the same boundary window.
        if bars_elapsed <= 0:
            surviving.append(order)
            continue

        # Bar boundary -> cancel old + place new at updated coordinates

        # Cancel old
        try:
            cancel_resp = await adapter.cancel_plan_order(
                order.symbol.upper(),
                order.exchange_order_id,
                mode,
                plan_type="normal_plan",
            )
            if not cancel_resp.get("ok"):
                print(f"[trendline_orders] cancel {order.symbol} failed: {cancel_resp}", flush=True)
                surviving.append(order)
                continue
        except Exception as e:
            print(f"[trendline_orders] cancel {order.symbol} err: {e}", flush=True)
            surviving.append(order)
            continue

        # Place new
        try:
            per_tf_risk = float((cfg.get("tf_risk") or {}).get(tf, risk_pct))
            qty, risk_usd, capped = _qty_for_risk(
                equity=equity,
                risk_pct=per_tf_risk,
                entry_price=limit_px,
                stop_price=stop_px,
                leverage=leverage,
                max_position_pct=max_position_pct,
            )
            if qty <= 0:
                continue

            now_ts = int(time.time())
            intent = OrderIntent(
                order_intent_id=f"tl_{order.symbol}_{now_ts}",
                signal_id=f"tl_sig_{now_ts}", line_id="",
                client_order_id=f"tl_{order.symbol[:10]}_{now_ts}",
                symbol=order.symbol.upper(), timeframe=tf, side=direction,
                order_type="market", trigger_mode="plan",
                entry_price=limit_px, stop_price=stop_px, tp_price=tp_px,
                quantity=qty, status="approved",
                reason="trendline_plan_market",
                created_at_bar=new_bar_index, created_at_ts=now_ts,
            )
            resp = await adapter.submit_live_plan_entry(intent, mode=mode, trigger_price=limit_px)

            if resp.get("ok"):
                order.exchange_order_id = resp.get("exchange_order_id", "")
                order.limit_price = limit_px
                order.stop_price = stop_px
                order.tp_price = tp_px
                order.current_projected_price = proj
                order.last_updated_ts = now
                order.bar_count = new_bar_index + 1
                order.line_ref_ts = now
                order.line_ref_price = proj
                surviving.append(order)
                updated += 1
                print(f"[trendline_orders] MOVED {direction} {order.symbol} {tf} @ {limit_px:.6f} "
                      f"SL={stop_px:.6f} risk=${risk_usd:.4f}{' capped' if capped else ''}", flush=True)
            else:
                print(f"[trendline_orders] MOVE FAILED {order.symbol}: {resp.get('reason')}", flush=True)
                surviving.append(order)  # keep tracking even if place failed
        except Exception as e:
            print(f"[trendline_orders] update {order.symbol} err: {e}", flush=True)
            surviving.append(order)

    _save_active(surviving)
    return {"placed": placed, "updated": updated, "cancelled": cancelled}


async def cancel_all_trendline_plan_orders(cfg: dict, *, status: str = "cancelled") -> dict:
    """Cancel every exchange-live trendline order managed by this system.

    Used by risk halts. It cancels local active normal_plan orders plus
    exchange orphan `tl_` normal_plan orders. It also cleans up regular `tl_`
    limit orders left by the old automatic path, while leaving manual non-`tl_`
    orders untouched.
    """
    from server.execution.live_adapter import LiveExecutionAdapter

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"cancelled": 0, "failed": 0, "reason": "api_keys_missing"}

    mode = cfg.get("mode", "live")
    active = _load_active()
    cancelled = failed = 0
    cancelled_ids: set[str] = set()
    pending_plan_rows: list[dict] = []
    legacy_regular_rows: list[dict] = []

    try:
        pending_plan_rows = await adapter.get_pending_plan_orders(mode, plan_type="normal_plan")
    except Exception as exc:
        print(f"[trendline_orders] halt pending normal_plan sync failed: {exc}", flush=True)
    try:
        legacy_regular_rows = await adapter.get_pending_orders(mode)
    except Exception as exc:
        print(f"[trendline_orders] halt legacy regular sync failed: {exc}", flush=True)

    pending_ids = {
        str(row.get("orderId") or row.get("order_id") or "")
        for row in pending_plan_rows
        if str(row.get("orderId") or row.get("order_id") or "")
    }

    surviving: list[ActiveLineOrder] = []
    for order in active:
        if order.status != "placed":
            surviving.append(order)
            continue

        order_id = str(order.exchange_order_id or "")
        if pending_ids and order_id and order_id not in pending_ids:
            order.status = "stale"
            surviving.append(order)
            continue

        try:
            cancel_resp = await adapter.cancel_plan_order(
                order.symbol.upper(),
                order_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                order.status = status
                cancelled += 1
                cancelled_ids.add(order_id)
                print(
                    f"[trendline_orders] HALT-CANCEL {order.symbol} {order.timeframe}: "
                    f"order_id={order_id} status={status}",
                    flush=True,
                )
            else:
                failed += 1
                print(f"[trendline_orders] halt cancel {order.symbol} failed: {cancel_resp}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[trendline_orders] halt cancel {order.symbol} err: {exc}", flush=True)
        surviving.append(order)

    for row in pending_plan_rows:
        row_id = str(row.get("orderId") or row.get("order_id") or "")
        if not row_id or row_id in cancelled_ids:
            continue
        client_oid = str(row.get("clientOid") or "")
        if not client_oid.startswith("tl_"):
            continue
        row_symbol = str(row.get("symbol") or "").upper()
        if not row_symbol:
            continue
        try:
            cancel_resp = await adapter.cancel_plan_order(
                row_symbol,
                row_id,
                mode,
                plan_type="normal_plan",
            )
            if cancel_resp.get("ok"):
                cancelled += 1
                cancelled_ids.add(row_id)
                print(
                    f"[trendline_orders] HALT-ORPHAN-CANCEL {row_symbol}: "
                    f"order_id={row_id} status={status}",
                    flush=True,
                )
            else:
                failed += 1
                print(f"[trendline_orders] halt orphan cancel {row_symbol} failed: {cancel_resp}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[trendline_orders] halt orphan cancel {row_symbol} err: {exc}", flush=True)

    for row in legacy_regular_rows:
        row_id = str(row.get("orderId") or row.get("order_id") or "")
        if not row_id or row_id in cancelled_ids:
            continue
        client_oid = str(row.get("clientOid") or "")
        if not client_oid.startswith("tl_"):
            continue
        row_symbol = str(row.get("symbol") or "").upper()
        if not row_symbol:
            continue
        try:
            cancel_resp = await adapter.cancel_order(row_symbol, row_id, mode)
            if cancel_resp.get("ok"):
                cancelled += 1
                cancelled_ids.add(row_id)
                print(
                    f"[trendline_orders] HALT-LEGACY-REGULAR-CANCEL {row_symbol}: "
                    f"order_id={row_id} status={status}",
                    flush=True,
                )
            else:
                failed += 1
                print(f"[trendline_orders] halt legacy regular cancel {row_symbol} failed: {cancel_resp}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[trendline_orders] halt legacy regular cancel {row_symbol} err: {exc}", flush=True)

    _save_active(surviving)
    return {"cancelled": cancelled, "failed": failed, "status": status}
