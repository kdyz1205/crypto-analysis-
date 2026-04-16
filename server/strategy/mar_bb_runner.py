"""MA Ribbon + BB Exit — live scanner + executor.

Runs the V1+V3 filter strategy (research/strategy.py) across top-N Bitget
USDT perps. Fires a real Bitget market order when a signal appears, with
preset SL and TP attached so the position manages itself once filled.

Design constraints (enforced at runtime):
  - Single position at a time (user has $15 capital, can only afford one
    at a time safely). Signals ignored while any position is open.
  - Minimum notional $10 USDT (above Bitget's $5 floor) × 5x leverage.
  - Never fires on a symbol we already hold.
  - Scan interval 60s by default; each scan walks top-N symbols and
    calls Strategy.current_state on the fetched bars.
  - State persisted to data/mar_bb_state.json so a server restart
    picks up mid-position.

Run state:
  status ∈ {idle, running, paused, stopped}
  last_scan_ts, last_signal, last_error, open_position (if any)

API:
  start_runner(config) / stop_runner() / get_state()
  manual_kick() — force a scan right now without waiting for interval
"""
from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from research.strategy import Strategy, DEFAULT_CONFIG
from research.trendline_strategy import (
    generate_signals as trendline_generate_signals,
    DEFAULT_CONFIG as TRENDLINE_DEFAULT_CONFIG,
)

# Module-level state
_task: asyncio.Task | None = None
_running = False
_state_lock = asyncio.Lock()

# Runner config (defaults; can be overridden via start_runner())
DEFAULT_RUNNER_CFG = {
    "top_n": 100,
    "scan_interval_s": 60,
    # ── Multi-timeframe ──
    "timeframes": ["3m", "5m", "15m", "1h", "4h"],
    "timeframe": "1h",                       # legacy single-TF fallback
    # ── Per-TF risk config ──
    # risk_pct = max loss per single trade as % of equity
    # Adaptive risk: shorter TF = tighter SL possible = lower risk needed
    # Tighter SL + lower risk% = can use higher effective leverage safely
    "tf_risk": {
        "3m":  0.008,  # 0.8% risk — ultra-tight SL on 3m
        "5m":  0.01,   # 1.0% risk
        "15m": 0.015,  # 1.5% risk
        "1h":  0.025,  # 2.5% risk
        "4h":  0.04,   # 4% risk — widest SL, needs more room
    },
    # ── Position sizing ──
    "sizing_mode": "fixed_risk",
    "risk_pct": 0.03,               # fallback if TF not in tf_risk
    "notional_usd": 12.0,           # fallback for fixed_notional mode
    "max_position_pct": 0.50,       # max 50% equity per single position
    "leverage": 30,
    "max_concurrent_positions": 100,
    # ── Daily drawdown halt (adaptive by equity tier) ──
    # Format: [(equity_threshold, max_daily_dd_pct), ...] — checked top-down
    "daily_dd_tiers": [
        (100000, 0.03),   # $100k+: max 3% daily loss
        (25000,  0.06),   # $25k-100k: max 6%
        (5000,   0.10),   # $5k-25k: max 10%
        (2000,   0.15),   # $2k-5k: max 15%
        (1000,   0.25),   # $1k-2k: max 25%
        (500,    0.35),   # $500-1k: max 35%
        (0,      0.50),   # <$500: max 50%
    ],
    # ── Trendline reversal (breakout flip) ──
    # When a trendline trade hits SL (line broken), auto-open reverse direction.
    # Only effective on 4h — too noisy on shorter TFs (backtest verified).
    "trendline_reversal": {
        "15m": False,
        "1h":  False,
        "4h":  True,
    },
    "reversal_rr": 2.0,   # RR for the reversal trade
    # ── General ──
    "mode": "live",
    "min_bars": 100,
    "dry_run": False,
    "auto_start": True,
    "strategies": ["mar_bb", "mar_bb_v1", "trendline"],
}


def _state_file() -> Path:
    try:
        from server.utils.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT) / "data" / "mar_bb_state.json"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "mar_bb_state.json"


@dataclass
class OpenPosition:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_price: float
    tp_price: float
    size_qty: float
    notional_usd: float
    leverage: int
    bitget_order_id: str
    opened_ts: int
    timeframe: str


@dataclass
class RunnerState:
    status: Literal["idle", "running", "stopped", "error"] = "idle"
    started_at: int = 0
    last_scan_ts: int = 0
    last_scan_duration_s: float = 0.0
    scans_completed: int = 0
    signals_detected: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    last_error: str = ""
    config: dict = field(default_factory=dict)
    open_position: dict | None = None
    recent_signals: list = field(default_factory=list)     # last 20 signals


_state = RunnerState()


def _save_state() -> None:
    try:
        f = _state_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(asdict(_state), default=str), encoding="utf-8")
    except Exception as e:
        print(f"[mar_bb] state save err: {e}", flush=True)


def _load_state() -> None:
    global _state
    try:
        f = _state_file()
        if not f.exists():
            return
        data = json.loads(f.read_text(encoding="utf-8"))
        # Only restore open_position + counters + config; let status start
        # fresh so boot doesn't resume "running" if the process crashed.
        if data.get("open_position"):
            _state.open_position = data["open_position"]
        _state.scans_completed = int(data.get("scans_completed") or 0)
        _state.signals_detected = int(data.get("signals_detected") or 0)
        _state.orders_submitted = int(data.get("orders_submitted") or 0)
        _state.orders_rejected = int(data.get("orders_rejected") or 0)
    except Exception as e:
        print(f"[mar_bb] state load err: {e}", flush=True)


def get_state() -> dict:
    return asdict(_state)


# ─────────────────────────────────────────────────────────────
# Trendline reversal (breakout flip)
# ─────────────────────────────────────────────────────────────
_last_reversal_check_ts: int = 0   # avoid checking too frequently
_known_closed_ids: set = set()     # track already-processed closed positions


async def _check_and_fire_reversals(cfg: dict, equity: float) -> int:
    """
    Check Bitget for recently closed trendline positions that hit SL.
    If found and reversal is enabled for that TF, fire a reverse order.
    Returns number of reversal orders submitted.
    """
    global _last_reversal_check_ts, _known_closed_ids

    reversal_cfg = cfg.get("trendline_reversal", {})
    if not any(reversal_cfg.values()):
        return 0  # reversal disabled for all TFs

    now = int(time.time())
    if now - _last_reversal_check_ts < 30:  # check at most every 30s
        return 0
    _last_reversal_check_ts = now

    try:
        from server.strategy.mar_bb_history import fetch_position_history
        rows = await fetch_position_history(days=1, mode=cfg.get("mode", "live"))
    except Exception as e:
        print(f"[reversal] history fetch err: {e}", flush=True)
        return 0

    reversals_fired = 0
    rr = float(cfg.get("reversal_rr", 2.0))

    for row in rows:
        # Unique ID for this closed position
        pos_id = str(row.get("positionId") or row.get("uTime") or row.get("cTime") or "")
        if not pos_id or pos_id in _known_closed_ids:
            continue
        _known_closed_ids.add(pos_id)

        # Only process trendline orders (clientOid starts with "marbb_")
        client_oid = (row.get("clientOid") or row.get("clientOId") or "").lower()
        # We need to identify trendline trades — they use marbb_ prefix too currently
        # Check if it was closed by SL (stop loss)
        close_type = (row.get("closeType") or row.get("triggerType") or "").lower()
        # Bitget uses different fields; check if loss
        pnl = float(row.get("netProfit") or row.get("pnl") or 0)
        if pnl >= 0:
            continue  # not a SL hit, skip

        symbol = (row.get("symbol") or "").upper()
        side = (row.get("holdSide") or row.get("posSide") or "").lower()
        close_price = float(row.get("closeAvgPrice") or 0)
        open_price = float(row.get("openAvgPrice") or 0)

        if not symbol or not side or close_price <= 0 or open_price <= 0:
            continue

        # Determine which TF this trade was on (we don't have this directly from Bitget)
        # Use the SL distance to guess: trendline SL is typically 0.3-0.5%
        sl_dist = abs(close_price - open_price) / open_price
        if sl_dist > 0.02:  # > 2% SL distance = probably not a trendline trade
            continue

        # Check if reversal is enabled (default to 4h if we can't determine TF)
        # For now, only fire if 4h reversal is enabled
        if not reversal_cfg.get("4h", False):
            continue

        # Build reverse order
        rev_dir = "short" if side == "long" else "long"
        rev_entry = close_price  # enter at the SL price (current mark)
        sl_distance = abs(open_price - close_price)
        rev_sl = rev_entry + sl_distance if rev_dir == "short" else rev_entry - sl_distance
        rev_tp = rev_entry - rr * sl_distance if rev_dir == "short" else rev_entry + rr * sl_distance

        # Position sizing
        notional = _calc_position_size(equity, rev_entry, rev_sl, cfg) if equity > 0 else float(cfg.get("notional_usd", 12))
        qty = notional / rev_entry if rev_entry > 0 else 0

        plan = {
            "strategy": "trendline_reversal",
            "symbol": symbol,
            "timeframe": "4h",
            "direction": rev_dir,
            "entry_price": rev_entry,
            "stop_price": rev_sl,
            "tp_price": rev_tp,
            "quantity": qty,
            "notional": notional,
            "leverage": int(cfg.get("leverage", 30)),
            "mode": cfg.get("mode", "live"),
        }

        print(f"[reversal] {symbol} SL hit -> FLIP to {rev_dir} @ {rev_entry:.4f} "
              f"sl={rev_sl:.4f} tp={rev_tp:.4f} notional=${notional:.2f}", flush=True)

        if cfg.get("dry_run"):
            print(f"[reversal] DRY-RUN skipped", flush=True)
            continue

        try:
            plan = await _anchor_sl_tp_to_mark(plan)
            resp = await _submit_order(plan)
        except Exception as e:
            resp = {"ok": False, "reason": f"exception: {e}"}

        if resp.get("ok"):
            reversals_fired += 1
            _state.orders_submitted += 1
            _state.signals_detected += 1
            sig_record = {
                "ts": now, "strategy": "trendline_reversal",
                "symbol": symbol, "tf": "4h", "direction": rev_dir,
                "entry": rev_entry, "stop": rev_sl, "tp": rev_tp,
            }
            _state.recent_signals = ([sig_record] + _state.recent_signals)[:20]
            print(f"[reversal] FILLED {symbol} {rev_dir} order_id={resp.get('exchange_order_id')}", flush=True)
        else:
            _state.orders_rejected += 1
            print(f"[reversal] REJECTED {symbol}: {resp.get('reason')}", flush=True)

    # Trim known IDs to prevent unbounded growth
    if len(_known_closed_ids) > 500:
        _known_closed_ids.clear()

    return reversals_fired


# ─────────────────────────────────────────────────────────────
# Trailing SL: move SL toward entry (and past it) over time
# ─────────────────────────────────────────────────────────────
# Stores trendline params per open position for SL projection
# Key: symbol (uppercase), Value: {slope, intercept, entry_bar, entry_price, side, tf}
_trendline_params: dict[str, dict] = {}


def register_trendline_params(symbol: str, slope: float, intercept: float,
                               entry_bar: int, entry_price: float, side: str, tf: str):
    """Called when opening a trendline trade — stores line params for SL trailing."""
    _trendline_params[symbol.upper()] = {
        "slope": slope, "intercept": intercept,
        "entry_bar": entry_bar, "entry_price": entry_price,
        "side": side, "tf": tf, "opened_ts": int(time.time()),
    }


def _calc_trendline_trailing_sl(symbol: str, bars_since_entry: int, sl_pct: float = 0.004) -> float | None:
    """
    Calculate where the SL should be NOW based on the REAL trendline projection.
    Uses the actual slope and intercept from the two anchor points that formed the line.
    The projected line value = slope * (entry_bar + bars_since) + intercept.
    SL sits sl_pct below (long) or above (short) the projected line.
    Returns new SL price, or None if no trendline params stored.
    """
    params = _trendline_params.get(symbol.upper())
    if not params:
        return None

    # Real projection: the line extends forward from entry_bar
    current_bar = params["entry_bar"] + bars_since_entry
    projected_line = params["slope"] * current_bar + params["intercept"]
    if projected_line <= 0:
        return None

    if params["side"] == "long":
        # Support line: SL sits below the projected line
        return projected_line * (1 - sl_pct)
    else:
        # Resistance line: SL sits above the projected line
        return projected_line * (1 + sl_pct)


async def _update_trailing_stops(cfg: dict) -> int:
    """
    Check all open positions and move SL if it should be tighter.
    For trendline trades: follow the trendline projection.
    For ribbon trades: follow ATR trailing (already built into Bitget preset).

    Returns number of SL updates made.
    """
    if not _trendline_params:
        return 0

    bitget_positions, ok = await _get_bitget_positions()
    if not ok or not bitget_positions:
        return 0

    updates = 0
    now_ts = int(time.time())

    for pos in bitget_positions:
        symbol = (pos.get("symbol") or "").upper()
        if symbol not in _trendline_params:
            continue

        params = _trendline_params[symbol]
        side = (pos.get("holdSide") or pos.get("posSide") or "").lower()
        entry_price = float(pos.get("openAvgPrice") or params["entry_price"] or 0)
        current_sl = float(pos.get("stopLossTriggerPrice") or 0)

        if entry_price <= 0:
            continue

        # Estimate bars since entry based on time + TF
        tf = params.get("tf", "1h")
        tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
        bar_dur = tf_seconds.get(tf, 3600)
        elapsed = now_ts - params["opened_ts"]
        bars_since = max(1, elapsed // bar_dur)

        new_sl = _calc_trendline_trailing_sl(symbol, bars_since)
        if new_sl is None or new_sl <= 0:
            continue

        # Only move SL in the profitable direction (never widen it)
        if side == "long":
            if current_sl > 0 and new_sl <= current_sl:
                continue  # new SL is worse (lower), don't move
        elif side == "short":
            if current_sl > 0 and new_sl >= current_sl:
                continue  # new SL is worse (higher), don't move

        # Move it
        try:
            from server.execution.live_adapter import LiveExecutionAdapter
            adapter = LiveExecutionAdapter()
            resp = await adapter.update_position_sl_tp(
                symbol, side, new_sl=new_sl, mode=cfg.get("mode", "live"),
            )
            if resp.get("ok"):
                updates += 1
                is_profit = (side == "long" and new_sl > entry_price) or \
                            (side == "short" and new_sl < entry_price)
                label = "PROFIT-LOCK" if is_profit else "TIGHTEN"
                print(f"[trailing] {label} {symbol} {side}: SL {current_sl:.6f} -> {new_sl:.6f}"
                      f" (entry={entry_price:.6f}, bars={bars_since})", flush=True)
            else:
                print(f"[trailing] {symbol} SL update failed: {resp.get('reason')}", flush=True)
        except Exception as e:
            print(f"[trailing] {symbol} err: {e}", flush=True)

    return updates


# ─────────────────────────────────────────────────────────────
# Daily drawdown tracking
# ─────────────────────────────────────────────────────────────
_daily_equity_start: float = 0.0   # equity at start of current UTC day
_daily_date: str = ""              # "YYYY-MM-DD" of current tracking day


def _get_daily_dd_limit(equity: float, cfg: dict) -> float:
    """Return max daily drawdown % for current equity tier."""
    tiers = cfg.get("daily_dd_tiers", [(0, 0.50)])
    for threshold, dd_pct in tiers:
        if equity >= threshold:
            return dd_pct
    return 0.50  # fallback


def _check_daily_dd(equity: float, cfg: dict) -> tuple[bool, float, float]:
    """
    Check if we've exceeded daily drawdown limit.
    Returns (halted, current_dd_pct, limit_pct).
    Resets at UTC midnight.
    """
    global _daily_equity_start, _daily_date
    import datetime as _dt

    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    if today != _daily_date or _daily_equity_start <= 0:
        _daily_equity_start = equity
        _daily_date = today
        return False, 0.0, _get_daily_dd_limit(equity, cfg)

    if _daily_equity_start <= 0:
        return False, 0.0, 0.50

    dd_pct = (_daily_equity_start - equity) / _daily_equity_start
    limit = _get_daily_dd_limit(_daily_equity_start, cfg)

    if dd_pct >= limit:
        return True, dd_pct, limit
    return False, dd_pct, limit


# ─────────────────────────────────────────────────────────────
# Position sizing
# ─────────────────────────────────────────────────────────────
async def _get_equity() -> float:
    """Fetch current USDT equity from Bitget.

    The adapter's get_live_account_status returns keys like `total_equity`
    and `usdt_available` (not `equity`). Pull from whichever comes back
    first and non-zero so this works regardless of Bitget-side label drift.
    """
    try:
        from server.execution.live_adapter import LiveExecutionAdapter
        adapter = LiveExecutionAdapter()
        if not adapter.has_api_keys():
            return 0.0
        acct = await adapter.get_live_account_status(mode=_state.config.get("mode", "live"))
        if not acct.get("ok"):
            print(f"[mar_bb] equity acct ok=False reason={acct.get('reason')}", flush=True)
            return 0.0
        # Try every likely key name, pick the first non-zero.
        for key in ("total_equity", "equity", "totalEquity", "usdtEquity", "usdt_available", "available"):
            v = acct.get(key)
            if v is not None:
                try:
                    f = float(v)
                    if f > 0:
                        return f
                except (TypeError, ValueError):
                    continue
        print(f"[mar_bb] equity fetch: no non-zero value in {list(acct.keys())}", flush=True)
        return 0.0
    except Exception as e:
        print(f"[mar_bb] equity fetch err: {e}", flush=True)
        return 0.0


def _calc_position_size(equity: float, entry: float, stop: float, cfg: dict) -> float:
    """
    Calculate position notional based on sizing mode.

    fixed_risk: notional = (equity × risk_pct) / |entry-stop|/entry
                capped at equity × max_position_pct × leverage
    fixed_notional: just use cfg["notional_usd"]
    """
    if cfg.get("sizing_mode") == "fixed_notional" or equity <= 0:
        return float(cfg.get("notional_usd", 12.0))

    risk_pct = float(cfg.get("risk_pct", 0.03))
    max_pos_pct = float(cfg.get("max_position_pct", 0.50))
    leverage = int(cfg.get("leverage", 20))

    # Per-unit risk (SL distance as fraction of entry)
    if entry <= 0 or stop <= 0:
        return float(cfg.get("notional_usd", 12.0))
    sl_distance_pct = abs(entry - stop) / entry
    if sl_distance_pct < 1e-8:
        return float(cfg.get("notional_usd", 12.0))

    # Risk dollars = equity × risk_pct
    # Notional = risk_dollars / sl_distance_pct
    risk_dollars = equity * risk_pct
    notional = risk_dollars / sl_distance_pct

    # Cap: max margin = equity × max_position_pct → max notional = margin × leverage
    max_notional = equity * max_pos_pct * leverage
    notional = min(notional, max_notional)

    # Floor: Bitget minimum is $5
    notional = max(notional, 5.0)

    print(f"[mar_bb] sizing: equity=${equity:.2f} risk={risk_pct*100:.1f}%"
          f" SL_dist={sl_distance_pct*100:.2f}% -> notional=${notional:.2f}"
          f" (margin=${notional/leverage:.2f}, {notional/leverage/equity*100:.1f}% of equity)", flush=True)

    return notional


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────
async def _fetch_bars(symbol: str, tf: str, limit_days: int) -> dict | None:
    """Fetch OHLCV via the server's own get_ohlcv (hits the shared cache
    + gzip so this is cheap after first call). Returns dict with
    parallel numpy arrays for o/h/l/c/v.
    """
    try:
        from server.data_service import get_ohlcv
        data = await get_ohlcv(
            symbol, tf, end_time=None, days=limit_days,
            history_mode="fast_window",
        )
        candles = data.get("candles") or []
        if len(candles) < 50:
            return None
        o = np.array([c["open"] for c in candles], dtype=float)
        h = np.array([c["high"] for c in candles], dtype=float)
        l = np.array([c["low"] for c in candles], dtype=float)
        c = np.array([c["close"] for c in candles], dtype=float)
        vol = data.get("volume") or []
        if vol:
            v = np.array([x.get("value", 0) for x in vol], dtype=float)
        else:
            v = np.ones(len(c), dtype=float)   # fallback if volume missing
        return {"o": o, "h": h, "l": l, "c": c, "v": v}
    except Exception as e:
        print(f"[mar_bb] fetch {symbol} {tf} err: {e}", flush=True)
        return None


async def _get_top_symbols(n: int) -> list[str]:
    try:
        from server.data_service import get_top_volume_symbols
        return await get_top_volume_symbols(top_n=n)
    except Exception as e:
        print(f"[mar_bb] top symbols err: {e}", flush=True)
        return []


# ─────────────────────────────────────────────────────────────
# Position tracking (against Bitget truth)
# ─────────────────────────────────────────────────────────────
async def _get_bitget_positions() -> tuple[list[dict], bool]:
    """Return (positions_list, ok). positions is a list of dicts with
    at least 'symbol' set. ok=False means Bitget didn't answer (trust
    local cache in that case)."""
    try:
        from server.execution.live_adapter import LiveExecutionAdapter
        adapter = LiveExecutionAdapter()
        if not adapter.has_api_keys():
            return [], True
        acct = await adapter.get_live_account_status(mode=_state.config.get("mode", "live"))
        if not acct.get("ok"):
            _state.last_error = f"position_check_failed: {acct.get('reason','?')}"
            return [], False
        return (acct.get("positions") or []), True
    except Exception as e:
        print(f"[mar_bb] position fetch exception: {e}", flush=True)
        return [], False


async def _has_open_position() -> bool:
    """Does Bitget have at least one open position right now?"""
    positions, ok = await _get_bitget_positions()
    if not ok:
        return _state.open_position is not None
    return len(positions) > 0


# ─────────────────────────────────────────────────────────────
# Strategy signal → order intent
# ─────────────────────────────────────────────────────────────
def _build_order_intent_mar_bb(
    symbol: str, tf: str, state: dict, cfg: dict,
) -> dict | None:
    """MA Ribbon + BB Exit → OrderIntent. Returns None if no valid signal."""
    signal = state.get("signal")
    if signal not in (1, -1):
        return None
    close = state.get("close")
    atr_val = state.get("atr")
    if close is None or atr_val is None or close <= 0 or atr_val <= 0:
        return None

    bb_up = state.get("bb_upper")
    bb_lo = state.get("bb_lower")
    atr_mult = float(DEFAULT_CONFIG.get("atr_mult", 3.0))

    if signal == 1:
        entry = close
        stop = entry - atr_mult * atr_val
        tp = float(bb_up) if bb_up else entry * 1.02
        direction = "long"
    else:
        entry = close
        stop = entry + atr_mult * atr_val
        tp = float(bb_lo) if bb_lo else entry * 0.98
        direction = "short"

    notional = cfg.get("_notional_override") or float(cfg.get("notional_usd", 12.0))
    qty = notional / entry

    return {
        "strategy": "mar_bb",
        "symbol": symbol, "timeframe": tf,
        "direction": direction,
        "entry_price": float(entry),
        "stop_price": float(stop),
        "tp_price": float(tp),
        "quantity": qty,
        "notional": notional,
        "leverage": int(cfg["leverage"]),
        "mode": cfg.get("mode", "live"),
    }


def _check_trendline_signal(
    symbol: str, tf: str, bars: dict, cfg: dict,
) -> dict | None:
    """Trendline Bounce → OrderIntent. Runs generate_signals and picks
    the SIGNAL AT THE LAST BAR (if any).
    """
    try:
        sigs, entries, sls, tps, lines = trendline_generate_signals(
            bars["o"], bars["h"], bars["l"], bars["c"], bars["v"],
            TRENDLINE_DEFAULT_CONFIG,
        )
    except Exception as e:
        print(f"[mar_bb] trendline_signals {symbol} err: {e}", flush=True)
        return None

    i = len(bars["c"]) - 1   # latest bar
    if i < 0:
        return None
    sig = int(sigs[i]) if sigs[i] else 0
    if sig == 0:
        return None
    entry = float(entries[i]) if not np.isnan(entries[i]) else None
    stop = float(sls[i]) if not np.isnan(sls[i]) else None
    tp = float(tps[i]) if not np.isnan(tps[i]) else None
    if entry is None or stop is None or tp is None or entry <= 0:
        return None

    direction = "long" if sig == 1 else "short"
    notional = cfg.get("_notional_override") or float(cfg.get("notional_usd", 12.0))
    qty = notional / entry

    # Extract real trendline slope/intercept for trailing SL
    line_info = lines[i]  # dict with {type, i1, i2, p1, p2, slope, intercept} or None
    line_slope = float(line_info["slope"]) if line_info and "slope" in line_info else 0.0
    line_intercept = float(line_info["intercept"]) if line_info and "intercept" in line_info else 0.0
    line_entry_bar = i  # the bar index where signal fired

    return {
        "strategy": "trendline",
        "symbol": symbol, "timeframe": tf,
        "direction": direction,
        "entry_price": entry,
        "stop_price": stop,
        "tp_price": tp,
        "quantity": qty,
        "notional": notional,
        "leverage": int(cfg["leverage"]),
        "mode": cfg.get("mode", "live"),
        # Real trendline params for trailing SL
        "line_slope": line_slope,
        "line_intercept": line_intercept,
        "line_entry_bar": line_entry_bar,
    }


async def _anchor_sl_tp_to_mark(plan: dict) -> dict:
    """Rewrite plan.stop_price and plan.tp_price so they sit the SAME
    PERCENTAGE away from the CURRENT mark price, not the strategy's
    expected entry.

    Why: trendline strategy picks entry_p = projected_line × (1 - buffer)
    which can be 0.5-1% off the mark by the time a market order fills.
    If we submit the strategy's absolute SL/TP prices, the distance from
    actual fill ends up distorted (loss side stretches, profit side
    shrinks), flipping a 3:1 RR into a 0.46:1 RR.

    Fix: capture the relative SL/TP pct from the strategy's plan, fetch
    current mark, and rebuild absolute SL/TP around the mark. The
    resulting plan still risks ~$0.95 per trade regardless of fill price.
    """
    entry = float(plan.get("entry_price") or 0)
    stop = float(plan.get("stop_price") or 0)
    tp = float(plan.get("tp_price") or 0)
    if entry <= 0 or stop <= 0 or tp <= 0:
        return plan

    # Compute relative distances from strategy's intended entry
    if plan.get("direction") == "long":
        sl_rel = (entry - stop) / entry       # positive, SL is below entry
        tp_rel = (tp - entry) / entry         # positive, TP is above entry
    else:
        sl_rel = (stop - entry) / entry       # positive, SL is above entry
        tp_rel = (entry - tp) / entry         # positive, TP is below entry

    # Pull current mark from Bitget ticker (fast, no auth needed)
    try:
        from server.execution.live_adapter import _get_bitget_client
        client = _get_bitget_client()
        r = await client.get(
            "https://api.bitget.com/api/v2/mix/market/ticker",
            params={"symbol": plan["symbol"].upper(), "productType": "USDT-FUTURES"},
        )
        if r.status_code != 200:
            return plan
        data = (r.json() or {}).get("data") or []
        row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else None)
        if not row:
            return plan
        mark = float(row.get("markPrice") or row.get("lastPr") or 0)
    except Exception as e:
        print(f"[mar_bb] anchor mark fetch err: {e}", flush=True)
        return plan
    if mark <= 0:
        return plan

    # Rebuild prices anchored to the live mark
    if plan.get("direction") == "long":
        new_entry = mark
        new_stop = mark * (1 - sl_rel)
        new_tp = mark * (1 + tp_rel)
    else:
        new_entry = mark
        new_stop = mark * (1 + sl_rel)
        new_tp = mark * (1 - tp_rel)

    # Also recompute quantity so notional stays the same (sizing was
    # based on the notional, not the entry).
    notional = float(plan.get("notional") or 0)
    new_qty = (notional / new_entry) if new_entry > 0 and notional > 0 else plan.get("quantity")

    old_entry = entry
    plan["entry_price"] = new_entry
    plan["stop_price"] = new_stop
    plan["tp_price"] = new_tp
    if new_qty:
        plan["quantity"] = new_qty

    print(
        f"[mar_bb] anchored {plan['symbol']} {plan['direction']}: "
        f"strategy_entry={old_entry:.6f} -> mark={new_entry:.6f} "
        f"sl_rel={sl_rel*100:.3f}% tp_rel={tp_rel*100:.3f}% "
        f"new_sl={new_stop:.6f} new_tp={new_tp:.6f}",
        flush=True,
    )
    return plan


async def _submit_order(plan: dict) -> dict:
    """Build an OrderIntent + submit via LiveExecutionAdapter. Returns the
    adapter response dict.
    """
    from server.execution.live_adapter import LiveExecutionAdapter
    from server.execution.types import OrderIntent

    adapter = LiveExecutionAdapter()
    if not adapter.has_api_keys():
        return {"ok": False, "reason": "api_keys_missing"}

    # Set leverage first (idempotent on Bitget)
    try:
        await adapter._bitget_request(
            "POST", "/api/v2/mix/account/set-leverage",
            mode=plan["mode"],
            body={
                "symbol": plan["symbol"].upper(),
                "productType": "USDT-FUTURES",
                "marginCoin": "USDT",
                "leverage": str(plan["leverage"]),
            },
        )
    except Exception as e:
        print(f"[mar_bb] set-leverage {plan['symbol']} warn: {e}", flush=True)

    now_ts = int(time.time())
    intent = OrderIntent(
        order_intent_id=f"marbb_{plan['symbol']}_{now_ts}",
        signal_id=f"marbb_sig_{now_ts}",
        line_id="",
        client_order_id=f"marbb_{plan['symbol'][:10]}_{now_ts}",
        symbol=plan["symbol"].upper(),
        timeframe=plan["timeframe"],
        side=plan["direction"],  # type: ignore
        order_type="market",
        trigger_mode="market",
        entry_price=plan["entry_price"],
        stop_price=plan["stop_price"],
        tp_price=plan["tp_price"],
        quantity=plan["quantity"],
        status="approved",
        reason="mar_bb_runner_signal",
        created_at_bar=-1,
        created_at_ts=now_ts,
    )

    return await adapter.submit_live_entry(intent, mode=plan["mode"])


# ─────────────────────────────────────────────────────────────
# Main scan loop
# ─────────────────────────────────────────────────────────────
async def _do_scan() -> None:
    """One pass through top-N symbols × all configured TFs. Updates _state."""
    t0 = time.time()
    cfg = _state.config
    timeframes = cfg.get("timeframes") or [cfg.get("timeframe", "1h")]
    top_n = int(cfg["top_n"])
    max_concurrent = int(cfg.get("max_concurrent_positions", 10))
    tf_risk = cfg.get("tf_risk", {})

    print(f"[mar_bb] scan start: top_n={top_n} TFs={timeframes} max={max_concurrent} "
          f"leverage={cfg.get('leverage')}x", flush=True)

    # Fetch equity ONCE per scan
    equity = await _get_equity() if cfg.get("sizing_mode") == "fixed_risk" else 0
    if cfg.get("sizing_mode") == "fixed_risk":
        print(f"[mar_bb] equity=${equity:.2f}", flush=True)

    # Daily drawdown check
    dd_halted, dd_current, dd_limit = _check_daily_dd(equity, cfg)
    if dd_halted:
        _state.last_error = (
            f"DAILY DD HALT: lost {dd_current*100:.1f}% today (limit {dd_limit*100:.0f}% "
            f"for ${_daily_equity_start:.0f} tier). No new trades until UTC midnight."
        )
        print(f"[mar_bb] {_state.last_error}", flush=True)
        _state.last_scan_ts = int(time.time())
        _state.last_scan_duration_s = round(time.time() - t0, 2)
        _state.scans_completed += 1
        _save_state()
        return
    print(f"[mar_bb] daily DD: {dd_current*100:.1f}% / {dd_limit*100:.0f}% limit", flush=True)

    symbols = await _get_top_symbols(top_n)
    print(f"[mar_bb] fetched {len(symbols)} symbols", flush=True)
    if not symbols:
        _state.last_error = "no symbols fetched"
        return

    # Snapshot Bitget positions ONCE
    bitget_positions, ok = await _get_bitget_positions()
    held_symbols: set[str] = set()
    for p in bitget_positions:
        sym = (p.get("symbol") or "").upper()
        if sym:
            held_symbols.add(sym)
    slots_used = len(held_symbols)
    slots_free = max(0, max_concurrent - slots_used)
    print(f"[mar_bb] slots: {slots_used}/{max_concurrent} used, {slots_free} free", flush=True)

    tf_days = {"1m":1,"3m":2,"5m":3,"15m":7,"30m":14,"1h":21,"2h":42,"4h":84,"1d":500}

    # Strategy library: multiple ribbon configs + trendline
    RIBBON_CONFIGS = {
        "mar_bb": {
            # Champion config: R:3/8/21/55 + BB(55,0.6)
            "ma5_n": 3, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
            "bb_period": 55, "bb_std": 0.6,
            "adx_min": 25, "vol_mult": 1.2, "atr_mult": 3.0,
            "slope_threshold": 0.1, "fanning_min_pct": 0.8,
        },
        "mar_bb_v1": {
            # Original config: R:5/8/21/55 + BB(20,1.5)
            "ma5_n": 5, "ma8_n": 8, "ema21_n": 21, "ma55_n": 55,
            "bb_period": 20, "bb_std": 1.5,
            "adx_min": 25, "vol_mult": 1.2, "atr_mult": 3.0,
            "slope_threshold": 0.1, "fanning_min_pct": 0.8,
        },
    }
    ribbon_strategies = {name: Strategy(rcfg) for name, rcfg in RIBBON_CONFIGS.items()}
    enabled_strats = cfg.get("strategies") or ["mar_bb", "trendline"]
    signals_this_scan = 0
    orders_this_scan = 0

    if slots_free == 0:
        _state.last_error = (
            f"no free slots: {slots_used}/{max_concurrent} "
            f"({','.join(sorted(held_symbols))[:200]})"
        )

    # Scan each TF
    for tf in timeframes:
        if not _running or orders_this_scan >= slots_free:
            break

        days = tf_days.get(tf, 21)
        # Per-TF risk: use tf_risk[tf] if available, else fallback
        scan_risk_pct = tf_risk.get(tf, cfg.get("risk_pct", 0.03))
        scan_cfg = {**cfg, "risk_pct": scan_risk_pct, "_notional_override": None}

        scanned = 0
        for sym in symbols:
            if not _running or orders_this_scan >= slots_free:
                break
            if sym.upper() in held_symbols:
                continue

            scanned += 1
            if scanned <= 2 or scanned % 30 == 0:
                print(f"[mar_bb]   {tf} scanning {sym} ({scanned}/{len(symbols)})", flush=True)

            try:
                bars = await _fetch_bars(sym, tf, days)
            except Exception as e:
                print(f"[mar_bb] fetch {sym} {tf} EXCEPTION: {e}", flush=True)
                continue
            if bars is None or len(bars["c"]) < cfg["min_bars"]:
                continue

            # Cache last close for price alerts
            if not hasattr(_state, 'last_prices'):
                _state.last_prices = {}
            _state.last_prices[sym] = float(bars["c"][-1])

            # Run strategies — first signal wins
            plan = None

            # Try each ribbon config
            for rstrat_name, rstrat in ribbon_strategies.items():
                if rstrat_name not in enabled_strats:
                    continue
                if plan is not None:
                    break
                try:
                    state = rstrat.current_state(
                        bars["o"], bars["h"], bars["l"], bars["c"], bars["v"],
                    )
                    if state.get("signal") in (1, -1) and equity > 0:
                        entry_p = state.get("close", 0)
                        atr_val = state.get("atr", 0)
                        rcfg = RIBBON_CONFIGS[rstrat_name]
                        atr_mult = float(rcfg.get("atr_mult", 3.0))
                        stop_p = entry_p - atr_mult * atr_val if state["signal"] == 1 else entry_p + atr_mult * atr_val
                        scan_cfg["_notional_override"] = _calc_position_size(equity, entry_p, stop_p, scan_cfg)
                    plan = _build_order_intent_mar_bb(sym, tf, state, scan_cfg)
                    if plan:
                        plan["strategy"] = rstrat_name  # tag which ribbon config
                except Exception as e:
                    print(f"[mar_bb] {rstrat_name} {sym} {tf} err: {e}", flush=True)

            if plan is None and "trendline" in enabled_strats:
                try:
                    scan_cfg["_notional_override"] = None
                    plan = _check_trendline_signal(sym, tf, bars, scan_cfg)
                    if plan and equity > 0:
                        plan_notional = _calc_position_size(equity, plan["entry_price"], plan["stop_price"], scan_cfg)
                        plan["notional"] = plan_notional
                        plan["quantity"] = plan_notional / plan["entry_price"]
                    # Store real trendline slope/intercept for trailing SL
                    if plan and plan.get("strategy") == "trendline":
                        register_trendline_params(
                            sym,
                            slope=plan.get("line_slope", 0),
                            intercept=plan.get("line_intercept", 0),
                            entry_bar=plan.get("line_entry_bar", 0),
                            entry_price=plan["entry_price"],
                            side=plan["direction"], tf=tf,
                        )
                except Exception as e:
                    print(f"[mar_bb] trendline {sym} {tf} err: {e}", flush=True)

            if plan is None:
                continue

            signals_this_scan += 1
            _state.signals_detected += 1

            sig_record = {
                "ts": int(time.time()),
                "strategy": plan.get("strategy"),
                "symbol": sym,
                "tf": tf,
                "direction": plan["direction"],
                "entry": plan["entry_price"],
                "stop": plan["stop_price"],
                "tp": plan["tp_price"],
            }
            _state.recent_signals = ([sig_record] + _state.recent_signals)[:20]

            if cfg.get("dry_run"):
                print(f"[mar_bb] DRY-RUN [{plan.get('strategy')}] {sym} {plan['direction']} @ {plan['entry_price']:.4f}", flush=True)
                orders_this_scan += 1
                continue

            print(f"[mar_bb] SIGNAL [{plan.get('strategy')}] {sym} {tf} {plan['direction']} @ {plan['entry_price']:.4f}  "
                  f"sl={plan['stop_price']:.4f}  tp={plan['tp_price']:.4f}  risk={scan_risk_pct*100:.1f}%", flush=True)

            # Re-anchor SL/TP to current mark to preserve RR after slippage
            try:
                plan = await _anchor_sl_tp_to_mark(plan)
            except Exception as e:
                print(f"[mar_bb] anchor err {sym}: {e}", flush=True)

            try:
                resp = await _submit_order(plan)
            except Exception as e:
                resp = {"ok": False, "reason": f"exception: {e}"}
                traceback.print_exc()

            if resp.get("ok"):
                _state.orders_submitted += 1
                orders_this_scan += 1
                held_symbols.add(sym.upper())
                print(f"[mar_bb] FILLED {sym} {tf} order_id={resp.get('exchange_order_id')}", flush=True)
            else:
                _state.orders_rejected += 1
                _state.last_error = f"{sym}: {resp.get('reason', 'unknown')}"
                print(f"[mar_bb] REJECTED {sym}: {_state.last_error}", flush=True)

    # ── Trailing SL: move SL tighter on open trendline positions ──
    try:
        sl_updates = await _update_trailing_stops(cfg)
        if sl_updates > 0:
            print(f"[mar_bb] updated {sl_updates} trailing SL(s)", flush=True)
        # Clean up params for positions that are no longer open
        for sym in list(_trendline_params.keys()):
            if sym not in held_symbols:
                del _trendline_params[sym]
    except Exception as e:
        print(f"[mar_bb] trailing SL err: {e}", flush=True)

    # ── Trendline reversal check ──
    # After main scan, check if any trendline positions were just SL'd
    # and fire reverse orders if enabled for that TF
    try:
        rev_count = await _check_and_fire_reversals(cfg, equity)
        if rev_count > 0:
            print(f"[mar_bb] fired {rev_count} reversal(s)", flush=True)
    except Exception as e:
        print(f"[mar_bb] reversal check err: {e}", flush=True)

    _state.scans_completed += 1
    _state.last_scan_ts = int(time.time())
    _state.last_scan_duration_s = round(time.time() - t0, 2)
    _state.open_position = {
        "count": slots_used + (rev_count if 'rev_count' in dir() else 0),
        "max": max_concurrent,
        "symbols": sorted(held_symbols),
    } if held_symbols else None
    _save_state()


async def _loop() -> None:
    global _running
    print(f"[mar_bb] runner loop started, cfg={_state.config}", flush=True)
    _state.status = "running"
    _state.started_at = int(time.time())
    _save_state()
    try:
        while _running:
            try:
                await _do_scan()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _state.last_error = f"scan err: {e}"
                print(f"[mar_bb] {_state.last_error}", flush=True)
                traceback.print_exc()
            # Check trendline price alerts after each scan
            try:
                from .line_alerts import check_alerts as _check_line_alerts
                prices = {s: _state.last_prices.get(s, 0) for s in _state.last_prices} if hasattr(_state, 'last_prices') else {}
                if not prices and hasattr(_state, 'positions'):
                    pass  # no cached prices yet
                _check_line_alerts(prices)
            except Exception as e:
                print(f"[mar_bb] alert check err: {e}", flush=True)
            await asyncio.sleep(int(_state.config.get("scan_interval_s", 60)))
    except asyncio.CancelledError:
        print("[mar_bb] runner loop cancelled", flush=True)
    finally:
        _state.status = "stopped"
        _save_state()
        print("[mar_bb] runner loop exited", flush=True)


def start_runner(config: dict | None = None) -> dict:
    """Start the runner loop. Idempotent — returns current state if
    already running."""
    global _task, _running
    if _task is not None and not _task.done():
        return {"ok": True, "already_running": True, "state": get_state()}
    merged = {**DEFAULT_RUNNER_CFG, **(config or {})}
    _state.config = merged
    _state.last_error = ""
    _load_state()
    _running = True
    _task = asyncio.create_task(_loop(), name="mar_bb_runner")
    return {"ok": True, "started": True, "state": get_state()}


def stop_runner() -> dict:
    global _task, _running
    _running = False
    if _task and not _task.done():
        _task.cancel()
    _state.status = "stopped"
    _save_state()
    return {"ok": True, "state": get_state()}


async def manual_kick() -> dict:
    """Run one scan right now, outside the loop timer. Returns state."""
    if not _state.config:
        _state.config = {**DEFAULT_RUNNER_CFG}
    await _do_scan()
    return get_state()


def update_config(partial: dict) -> dict:
    """HOT-update the runner config without restarting the loop.

    The running loop reads `_state.config` at the start of each scan, so
    merging new values takes effect on the next scan (at most
    scan_interval_s seconds later). Useful for tweaking leverage,
    notional, max_concurrent, etc. mid-flight.

    Does NOT restart the loop or cancel in-flight positions. If the loop
    is not running, returns an error marker but still updates _state.config
    so a subsequent start_runner() picks up the new values.
    """
    if not isinstance(partial, dict):
        return {"ok": False, "reason": "partial must be a dict"}
    # Sanitize: drop keys not in default config
    allowed = set(DEFAULT_RUNNER_CFG.keys())
    clean = {k: v for k, v in partial.items() if k in allowed and v is not None}
    if not clean:
        return {"ok": True, "note": "no-op", "config": _state.config}
    _state.config = {**_state.config, **clean}
    _save_state()
    print(f"[mar_bb] config hot-updated: {clean}", flush=True)
    return {
        "ok": True,
        "updated_keys": sorted(clean.keys()),
        "config": _state.config,
        "running": _state.status == "running",
    }


__all__ = [
    "start_runner",
    "stop_runner",
    "get_state",
    "manual_kick",
    "update_config",
    "DEFAULT_RUNNER_CFG",
]
