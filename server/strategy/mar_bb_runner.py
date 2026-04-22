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
import datetime as _dt
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
_maintenance_task: asyncio.Task | None = None
_trailing_scheduler_task: asyncio.Task | None = None
_running = False
_runner_started_ts: float = 0.0  # set by start_runner; used to skip replay-feeding the circuit breaker with historical closes
_state_lock = asyncio.Lock()
_trendline_maintenance_lock = asyncio.Lock()

# Runner config (defaults; can be overridden via start_runner())
DEFAULT_RUNNER_CFG = {
    "top_n": 100,
    "scan_interval_s": 60,
    "maintenance_interval_s": 10,
    # ── Multi-timeframe ──
    # 5m dropped: V3 backtest shows -EV after commissions (see memory
    # "Trendline TF decision" 2026-04). 15m marginal edge +0.065% net, only
    # viable with ML gate thresholds >= 0.55. 1h/4h profitable unfiltered.
    "timeframes": ["15m", "1h", "4h"],
    "timeframe": "1h",                       # legacy single-TF fallback
    # ── Per-TF risk config (Axel's interim values, pending Kelly backtest) ──
    "tf_risk": {
        "3m":  0.003,   # 0.3%
        "5m":  0.003,   # 0.3%
        "15m": 0.007,   # 0.7%
        "1h":  0.015,   # 1.5%
        "4h":  0.030,   # 3.0%
    },
    # Stop is placed just beyond the trendline, not exactly on it.
    # Updated 2026-04-20 per user spec: typical buffer 0.03-0.05% beyond line.
    # Old 0.01% was too tight — a single tick of noise closed positions before
    # the line actually broke. Values are documented percentages: 0.04 = 0.04%.
    "stop_offset_pct": 0.04,
    "tf_stop_offset": {
        "5m": 0.03,    # smaller TFs → tighter (noise is already smaller)
        "15m": 0.04,
        "1h": 0.04,
        "4h": 0.05,    # bigger TF → larger wicks, more tolerance
    },
    "trendline_cooldown_bars_after_close": 4,
    # ── Position sizing ──
    "sizing_mode": "fixed_risk",
    "risk_pct": 0.01,               # fallback if TF not in tf_risk
    "notional_usd": 12.0,           # fallback for fixed_notional mode
    # Calibrated 2026-04-19 for small-equity ($4-10) live trading after the
    # -81% incident on 2026-04-18. Old defaults (leverage=30, concurrent=100,
    # max_pos_pct=0.50, <$500 DD 50%) were unsafe at this equity tier. See
    # data/logs/phase4_preflight.md for the live-readiness audit.
    "max_position_pct": 0.20,       # max 20% equity per single position
    "leverage": 10,                  # 10x: 10% liquidation buffer (vs 3.3% at 30x)
    "max_concurrent_positions": 3,   # $4 equity mathematically can't fund more
    # ── Daily drawdown halt (adaptive by equity tier) ──
    # Format: [(equity_threshold, max_daily_dd_pct), ...] — checked top-down
    "daily_dd_tiers": [
        (100000, 0.03),   # $100k+: max 3% daily loss
        (25000,  0.06),   # $25k-100k: max 6%
        (5000,   0.10),   # $5k-25k: max 10%
        (2000,   0.15),   # $2k-5k: max 15%
        (1000,   0.25),   # $1k-2k: max 25%
        (500,    0.35),   # $500-1k: max 35%
        (0,      0.20),   # <$500: max 20% (was 0.50 — tightened after -81% DD)
    ],
    # ── Trendline reversal (breakout flip) ──
    # When a trendline trade hits SL (line broken), auto-open reverse direction.
    # Only effective on 4h — too noisy on shorter TFs (backtest verified).
    "trendline_reversal": {
        "15m": False,
        "1h":  False,
        "4h":  False,   # disabled — no reversal trades
    },
    "reversal_rr": 2.0,   # RR for the reversal trade
    # ── General ──
    "mode": "live",
    "min_bars": 100,
    "dry_run": False,
    # auto_start=False: server restart must require a manual POST to
    # /api/mar-bb/start (or UI button). The MAR_BB_AUTOSTART env var is the
    # second guard; this default is the first. Changed from True on 2026-04-19
    # after the -81% DD incident.
    "auto_start": False,
    "strategies": ["trendline"],  # MA Ribbon disabled pending SL/TP indicator research
    "model_gate": {
        "enabled": True,
        "fail_open": True,
        # Explicit model roles. Do not replace the trade model with the AUC=0.928 line model.
        "primary_trade_model": "C:/Users/alexl/trading-system/checkpoints/trendline_quality/v3_trade_outcome_auc_0.825.pt",
        "aux_line_quality_model": "C:/Users/alexl/Desktop/crypto-analysis-/checkpoints/trendline_quality/pattern_bounce_auc_0.928.pt",
        # Global gate — applies to every TF. 0.55 picked to push 15m from
        # marginal +0.065% net EV into clearly-profitable territory while still
        # letting 1h/4h through (those are +0.18%/+0.26% net even unfiltered).
        "min_trade_win_prob": 0.55,
        "min_line_quality_prob": 0.55,
    },
}


def _state_file() -> Path:
    try:
        from server.utils.paths import PROJECT_ROOT
        return Path(PROJECT_ROOT) / "data" / "mar_bb_state.json"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "mar_bb_state.json"


def _daily_risk_file() -> Path:
    return _state_file().with_name("mar_bb_daily_risk.json")


def _trendline_params_file() -> Path:
    """Disk home for the in-memory `_trendline_params` dict.

    uvicorn reload (watchfiles on server/ + frontend/) wipes module-level
    state. Without persistence, every code edit deletes trailing-SL
    registration mid-trade and the SL never moves → stop stays at the
    price the line had at fill time (see incident 2026-04-20 RAVEUSDT,
    user report: "挂单没有移动"). Write-through on every mutation; load
    once at module import so the maintenance loop can pick up where it
    left off regardless of whether the user hit 'start runner'.
    """
    return _state_file().with_name("trendline_params.json")


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
    daily_risk: dict | None = None
    # Consecutive-loss circuit breaker (Stage 4a, 2026-04-19):
    # recent_close_outcomes tracks the most-recent close outcomes
    # ("win" | "loss"); when there are >= 3 losses in a row, trading is
    # paused until breaker_until_ts elapses (default +1h). Cleared on any win.
    recent_close_outcomes: list = field(default_factory=list)
    breaker_until_ts: float = 0.0
    breaker_reason: str = ""


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
        # Restore counters/config only. Do NOT restore open_position —
        # if the process died while the exchange closed the position, the
        # persisted dict would re-poison runner belief (ghost position).
        # The next scan re-adopts the real state from Bitget via
        # get_open_position_symbols() / _has_open_position().
        _state.open_position = None
        # Also clear recent_close_outcomes so a past losing streak can't
        # re-trigger the circuit breaker on boot. record_close_outcome()
        # already skips historical replay, but this is belt-and-suspenders.
        _state.recent_close_outcomes = []
        _state.scans_completed = int(data.get("scans_completed") or 0)
        _state.signals_detected = int(data.get("signals_detected") or 0)
        _state.orders_submitted = int(data.get("orders_submitted") or 0)
        _state.orders_rejected = int(data.get("orders_rejected") or 0)
    except Exception as e:
        print(f"[mar_bb] state load err: {e}", flush=True)


def get_state() -> dict:
    state_dict = asdict(_state)
    # Observability (P11): expose trailing-SL state so the UI / user can
    # confirm trendline_params actually persisted across uvicorn reloads,
    # and that the always-on maintenance loop is alive.
    state_dict["trendline_params_count"] = len(_trendline_params)
    state_dict["trendline_params_symbols"] = sorted(_trendline_params.keys())
    state_dict["maintenance_only_running"] = (
        _maintenance_only_task is not None and not _maintenance_only_task.done()
    )
    return state_dict


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


def _save_trendline_params() -> None:
    """Atomic write-through of `_trendline_params` to disk.

    Called after every mutation so SL trailing state survives uvicorn
    auto-reload, server restart, and crash.
    """
    try:
        f = _trendline_params_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f.suffix + ".tmp")
        tmp.write_text(json.dumps(_trendline_params, default=str), encoding="utf-8")
        tmp.replace(f)
    except Exception as exc:
        print(f"[trendline_params] save err: {exc}", flush=True)


def _load_trendline_params() -> None:
    """Restore `_trendline_params` from disk. Called at module import.

    Validates each entry has the minimum fields needed by
    `_calc_trendline_trailing_sl` and `_update_trailing_stops`. Corrupt
    or incomplete entries are dropped and the file is rewritten.
    """
    global _trendline_params
    try:
        f = _trendline_params_file()
        if not f.exists():
            return
        raw = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        cleaned: dict[str, dict] = {}
        required = {"slope", "side", "tf", "entry_price", "opened_ts"}
        for sym, params in raw.items():
            if not isinstance(params, dict):
                continue
            if not required.issubset(params.keys()):
                continue
            # Normalize numeric fields — JSON may have parsed these as
            # strings if they were written with default=str fallback.
            try:
                params["slope"] = float(params["slope"])
                params["intercept"] = float(params.get("intercept") or 0)
                params["entry_price"] = float(params["entry_price"])
                params["last_sl_set"] = float(params.get("last_sl_set") or 0)
                params["tp_price"] = float(params.get("tp_price") or 0)
                params["stop_offset_pct"] = float(params.get("stop_offset_pct") or 0)
                params["opened_ts"] = int(float(params["opened_ts"]))
                params["entry_bar"] = int(params.get("entry_bar") or 0)
                if params.get("line_ref_ts"):
                    params["line_ref_ts"] = int(float(params["line_ref_ts"]))
                if params.get("line_ref_price"):
                    params["line_ref_price"] = float(params["line_ref_price"])
            except Exception:
                continue
            cleaned[sym.upper()] = params
        _trendline_params = cleaned
        print(
            f"[trendline_params] loaded {len(cleaned)} entries from disk: "
            f"{list(cleaned.keys())}",
            flush=True,
        )
    except Exception as exc:
        print(f"[trendline_params] load err: {exc}", flush=True)


def register_trendline_params(symbol: str, slope: float, intercept: float,
                               entry_bar: int, entry_price: float, side: str,
                               tf: str, created_ts: float = 0, tp_price: float = 0,
                               last_sl_set: float = 0,
                               line_ref_ts: float = 0,
                               line_ref_price: float = 0,
                               stop_offset_pct: float = 0,
                               manual_line_id: str = "",
                               conditional_id: str = ""):
    """Called when opening a trendline trade — stores line params for SL trailing.

    `manual_line_id` / `conditional_id` are optional but important for the
    ML close-out hook: on position close we need to write back to the
    drawing that caused this trade so `user_drawings_ml.jsonl` gets a
    `closed` stage row with final PnL. Without these, the close-capture
    path has to match by Bitget order ID which fails if the exchange
    history row is missing the order ID (observed 2026-04-20 on RAVE).
    """
    key = symbol.upper()
    if key in _trendline_params:
        return
    params = {
        "slope": slope, "intercept": intercept,
        "entry_bar": entry_bar, "entry_price": entry_price,
        "side": side, "tf": tf,
        "opened_ts": int(created_ts) if created_ts > 0 else int(time.time()),
        "last_sl_set": float(last_sl_set or 0),
        "tp_price": tp_price,
        "stop_offset_pct": max(0.0, float(stop_offset_pct or 0)),
    }
    if line_ref_ts > 0 and line_ref_price > 0:
        params["line_ref_ts"] = int(line_ref_ts)
        params["line_ref_price"] = float(line_ref_price)
    if manual_line_id:
        params["manual_line_id"] = str(manual_line_id)
    if conditional_id:
        params["conditional_id"] = str(conditional_id)
    _trendline_params[key] = params
    _save_trendline_params()


def _select_active_order_for_position(active_orders: list, symbol: str):
    """Pick one local trendline order to restore trailing for an open position.

    A restart can leave historical stale/filled rows in the active-order file.
    Only live-ish rows should restore trailing, and only one row per symbol.
    """
    sym_upper = symbol.upper()
    candidates = [
        ao for ao in active_orders
        if getattr(ao, "symbol", "").upper() == sym_upper
        and getattr(ao, "status", "") in {"placed", "filled"}
    ]
    if not candidates:
        return None

    def _key(ao) -> tuple[int, float, float]:
        status_rank = 1 if getattr(ao, "status", "") == "placed" else 0
        return (
            status_rank,
            float(getattr(ao, "last_updated_ts", 0) or 0),
            float(getattr(ao, "created_ts", 0) or 0),
        )

    return max(candidates, key=_key)


def _stop_offset_pct_from_active_order(ao) -> float:
    line_price = float(
        getattr(ao, "line_ref_price", 0)
        or getattr(ao, "current_projected_price", 0)
        or 0
    )
    stop_price = float(getattr(ao, "stop_price", 0) or 0)
    if line_price <= 0 or stop_price <= 0:
        return 0.0
    if getattr(ao, "kind", "") == "support":
        return max(0.0, (line_price - stop_price) / line_price * 100.0)
    return max(0.0, (stop_price - line_price) / line_price * 100.0)


def _tf_seconds(tf: str) -> int:
    return {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 3600)


def _manual_slope_per_bar(cond) -> float:
    span = int(cond.t_end) - int(cond.t_start)
    if span <= 0:
        return 0.0
    return (float(cond.price_end) - float(cond.price_start)) / span * _tf_seconds(cond.timeframe)


def _manual_conditional_for_position(symbol: str, pos: dict):
    """Restore trailing from manual line conditionals when active-orders lack a row."""
    try:
        from server.conditionals.store import ConditionalOrderStore
        store = ConditionalOrderStore()
        rows = []
        for status in ("filled", "triggered"):
            rows.extend(store.list_all(status=status, symbol=symbol.upper()))
    except Exception as exc:
        print(f"[trailing] manual conditional lookup err {symbol}: {exc}", flush=True)
        return None
    if not rows:
        return None
    pos_side = str(pos.get("holdSide") or pos.get("posSide") or "").lower()
    if pos_side:
        rows = [row for row in rows if str(row.order.direction).lower() == pos_side] or rows
    return max(rows, key=lambda row: int(row.triggered_at or row.updated_at or row.created_at or 0))


def _register_manual_conditional_trailing(symbol: str, cond, pos: dict) -> bool:
    opened_ts = _position_open_ts(pos) or int(cond.triggered_at or time.time())
    line_ref_price = float(cond.line_price_at(opened_ts))
    entry_price = float(
        pos.get("openPriceAvg")
        or pos.get("averageOpenPrice")
        or pos.get("openAvgPrice")
        or cond.fill_price
        or line_ref_price
        or 0
    )
    if line_ref_price <= 0 or entry_price <= 0:
        return False
    stop_offset_pct = max(0.0, float(cond.order.stop_offset_pct_of_line or 0.0))
    if cond.order.direction == "long":
        last_sl_set = line_ref_price * (1.0 - stop_offset_pct / 100.0)
    else:
        last_sl_set = line_ref_price * (1.0 + stop_offset_pct / 100.0)
    register_trendline_params(
        symbol.upper(),
        slope=_manual_slope_per_bar(cond),
        intercept=0.0,
        entry_bar=0,
        entry_price=entry_price,
        side=cond.order.direction,
        tf=cond.timeframe,
        created_ts=opened_ts,
        tp_price=float(cond.order.tp_price or 0.0),
        last_sl_set=last_sl_set,
        line_ref_ts=opened_ts,
        line_ref_price=line_ref_price,
        stop_offset_pct=stop_offset_pct,
        manual_line_id=getattr(cond, "manual_line_id", "") or "",
        conditional_id=getattr(cond, "conditional_id", "") or "",
    )
    try:
        from server.conditionals.store import ConditionalOrderStore
        from server.conditionals.types import ConditionalEvent
        store = ConditionalOrderStore()
        if getattr(cond, "status", "") != "filled":
            cond.status = "filled"
            cond.fill_price = entry_price
            latest = store.get(cond.conditional_id)
            if latest is not None:
                cond.events = latest.events
            store.update(cond)
        store.append_event(cond.conditional_id, ConditionalEvent(
            ts=int(time.time()),
            kind="exchange_acked",
            price=entry_price,
            line_price=line_ref_price,
            message="open position detected; registered manual line for trailing SL",
        ))
    except Exception as exc:
        print(f"[trailing] manual conditional status update err {symbol}: {exc}", flush=True)
    return True


def _position_open_ts(pos: dict) -> int:
    """Best-effort Bitget position open timestamp in seconds."""
    # Do not use uTime/utime here: Bitget updates those on position changes,
    # which resets bars_since after every SL move and stops trailing.
    for key in ("openTime", "openTimestamp", "cTime", "ctime", "createdTime", "createTime"):
        raw = pos.get(key)
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


def _round_price(price: float) -> float:
    if price >= 1000:
        return round(price, 1)
    elif price >= 1:
        return round(price, 2)
    elif price >= 0.1:
        return round(price, 4)
    elif price >= 0.001:
        return round(price, 5)
    else:
        return round(price, 6)


def _project_line_at(symbol: str, now_ts: int | None = None, bars_since_entry: int = 0) -> float | None:
    """Project the line price at `now_ts`. Shared by SL + TP trailing."""
    params = _trendline_params.get(symbol.upper())
    if not params:
        return None
    if params.get("line_ref_ts") and params.get("line_ref_price"):
        ts = int(now_ts or time.time())
        elapsed_bars = _tf_boundaries_elapsed(params["line_ref_ts"], params.get("tf", "1h"), ts)
        projected_line = float(params["line_ref_price"]) + float(params["slope"]) * elapsed_bars
    else:
        current_bar = params["entry_bar"] + bars_since_entry
        projected_line = params["slope"] * current_bar + params["intercept"]
    return projected_line if projected_line > 0 else None


def _calc_trendline_trailing_sl(symbol: str, bars_since_entry: int, now_ts: int | None = None) -> float | None:
    params = _trendline_params.get(symbol.upper())
    if not params:
        return None
    projected_line = _project_line_at(symbol, now_ts=now_ts, bars_since_entry=bars_since_entry)
    if projected_line is None:
        return None

    offset = max(0.0, float(params.get("stop_offset_pct") or 0.0)) / 100.0
    side = str(params.get("side") or "").lower()
    if side == "long":
        projected_stop = projected_line * (1.0 - offset)
    elif side == "short":
        projected_stop = projected_line * (1.0 + offset)
    else:
        projected_stop = projected_line

    return _round_price(projected_stop)


def _calc_trendline_trailing_tp(symbol: str, bars_since_entry: int, now_ts: int | None = None) -> float | None:
    """Compute the line-trailed TP.

    Preserves the *relative* TP-to-line distance from fill time. If the
    user originally placed TP at 12% above the line (rr × (buffer+stop)),
    the TP stays at line × 1.12 as the line moves. This way TP scales
    with the line just like SL does — which is what the user asked for
    on 2026-04-20 after noticing RAVE's SL was trailing but TP wasn't.

    Returns None if there's no stored TP baseline (we refuse to invent a
    TP in thin air — the caller skips the TP update in that case).
    """
    params = _trendline_params.get(symbol.upper())
    if not params:
        return None
    baseline_tp = float(params.get("tp_price") or 0)
    baseline_line = float(params.get("line_ref_price") or 0)
    if baseline_tp <= 0 or baseline_line <= 0:
        return None
    projected_line = _project_line_at(symbol, now_ts=now_ts, bars_since_entry=bars_since_entry)
    if projected_line is None:
        return None
    multiplier = baseline_tp / baseline_line
    return _round_price(projected_line * multiplier)


def _tf_boundaries_elapsed(previous_ts: float, tf: str, now_ts: int | None = None) -> int:
    """Count completed TF candle boundaries since previous_ts."""
    if previous_ts <= 0:
        return 0
    tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    bar_dur = tf_seconds.get(tf, 3600)
    ts = int(now_ts or time.time())
    return max(0, int(ts // bar_dur) - int(float(previous_ts) // bar_dur))


async def _update_trailing_stops(cfg: dict) -> int:
    """
    Check all open positions and move SL if it should be tighter.
    For trendline trades: follow the trendline projection.
    For ribbon trades: follow ATR trailing (already built into Bitget preset).

    Returns number of SL updates made.
    """
    if not _trendline_params:
        print("[trailing] skip: no trendline params registered", flush=True)
        return 0

    bitget_positions, ok = await _get_bitget_positions()
    if not ok or not bitget_positions:
        print(f"[trailing] skip: position fetch ok={ok} positions={len(bitget_positions) if bitget_positions else 0}", flush=True)
        return 0

    updates = 0
    now_ts = int(time.time())

    for pos in bitget_positions:
        symbol = (pos.get("symbol") or "").upper()
        if symbol not in _trendline_params:
            print(f"[trailing] skip {symbol}: not a tracked trendline position", flush=True)
            continue

        params = _trendline_params[symbol]
        side = (pos.get("holdSide") or pos.get("posSide") or "").lower()
        entry_price = float(pos.get("openPriceAvg") or pos.get("averageOpenPrice") or pos.get("openAvgPrice") or params["entry_price"] or 0)
        current_sl = params.get("last_sl_set", 0.0)
        tf = params.get("tf", "1h")
        pos_open_ts = _position_open_ts(pos)
        if pos_open_ts > 0 and pos_open_ts != params.get("opened_ts"):
            params["opened_ts"] = pos_open_ts

        if entry_price <= 0:
            print(f"[trailing] skip {symbol}: entry_price unavailable in position row", flush=True)
            continue

        bars_since = _tf_boundaries_elapsed(params["opened_ts"], tf, now_ts)

        # Only update SL when a new bar has started (not every scan)
        last_bar = params.get("last_update_bar", 0)
        if bars_since <= last_bar:
            print(f"[trailing] skip {symbol}: bars_since={bars_since} last_update_bar={last_bar}", flush=True)
            continue

        new_sl = _calc_trendline_trailing_sl(symbol, bars_since, now_ts=now_ts)
        if new_sl is None or new_sl <= 0:
            print(f"[trailing] skip {symbol}: invalid projected SL {new_sl}", flush=True)
            continue

        # Only move SL in the profitable direction (never widen it)
        if side == "long":
            if current_sl > 0 and new_sl <= current_sl:
                print(f"[trailing] skip {symbol}: long SL would not tighten {current_sl:.6f}->{new_sl:.6f}", flush=True)
                continue
        elif side == "short":
            if current_sl > 0 and new_sl >= current_sl:
                print(f"[trailing] skip {symbol}: short SL would not tighten {current_sl:.6f}->{new_sl:.6f}", flush=True)
                continue
        else:
            print(f"[trailing] skip {symbol}: unknown holdSide={side}", flush=True)
            continue

        # Mark-price guard (Bitget 45122 defense). If the trailed line has
        # moved so far in our favor that the new SL would land on the wrong
        # side of mark, Bitget rejects with "Short SL price please > mark"
        # (or the LONG analogue). Even if Bitget accepted, placing SL past
        # mark would close the position instantly at the CURRENT price,
        # throwing away realised profit. Incident 2026-04-20 RAVE.
        #
        # Policy: refuse to set SL past mark. Line is meant to "hug" price;
        # if the line has run ahead of price, wait for price to catch up.
        # A future refinement can LOCK PROFIT (pull SL to entry or past-
        # entry) but for now just skip.
        mark_price = float(pos.get("markPrice") or pos.get("mark_price") or 0)
        if mark_price > 0:
            breach = (side == "long" and new_sl >= mark_price) or (side == "short" and new_sl <= mark_price)
            if breach:
                print(
                    f"[trailing] skip {symbol}: {side} new_sl={new_sl:.6f} breaches mark={mark_price:.6f} "
                    f"— would instant-fire at current price; line ran ahead of price, waiting",
                    flush=True,
                )
                # Bump last_update_bar so we don't retry every 10s until
                # the next TF boundary. If price catches up before then,
                # the stale SL from the previous bar is already tighter
                # than this one would have been, so we're not leaving
                # money on the table.
                params["last_update_bar"] = bars_since
                _save_trendline_params()
                continue

        # Same value as last time → skip
        try:
            from server.execution.live_adapter import LiveExecutionAdapter
            adapter = LiveExecutionAdapter()
            actual_current_sl = await adapter.get_position_sl_trigger_price(
                symbol,
                side,
                cfg.get("mode", "live"),
                entry_price=entry_price,
            )
            if actual_current_sl and actual_current_sl > 0:
                current_sl = actual_current_sl
                params["last_sl_set"] = actual_current_sl

            if current_sl > 0 and abs(new_sl - current_sl) <= 1e-12:
                print(f"[trailing] skip {symbol}: projected SL unchanged {current_sl:.6f}->{new_sl:.6f}", flush=True)
                continue

            # TP trailing: keep the TP-to-line multiplier constant from
            # fill time. LONG → TP up only; SHORT → TP down only. If the
            # computed TP wouldn't improve on the current TP, leave it
            # alone so we don't churn Bitget cancels for nothing.
            # (User reported 2026-04-20 that RAVE's SL trailed but TP
            # was frozen — that was `tp = None` hardcoded below.)
            new_tp: float | None = _calc_trendline_trailing_tp(symbol, bars_since, now_ts=now_ts)
            current_tp = float(params.get("last_tp_set") or 0)
            if new_tp is not None and new_tp > 0:
                if side == "long":
                    if current_tp > 0 and new_tp <= current_tp:
                        new_tp = None  # would shrink target — skip
                elif side == "short":
                    if current_tp > 0 and new_tp >= current_tp:
                        new_tp = None
            resp = await adapter.update_position_sl_tp(
                symbol, side, new_sl=new_sl, new_tp=new_tp,
                mode=cfg.get("mode", "live"),
            )
            if resp.get("ok"):
                placed_sl = float(resp.get("actual_sl_after") or resp.get("new_sl") or new_sl)
                placed_tp = float(resp.get("actual_tp_after") or resp.get("new_tp") or (new_tp or 0))
                from server.strategy.trade_log import log_sl_move
                log_sl_move(symbol, side, current_sl, placed_sl, tf=tf, bars=bars_since)
                try:
                    from server.strategy.ml_trade_db import log_sl_moved
                    log_sl_moved(
                        symbol=symbol, direction=side, tf=tf,
                        old_sl=current_sl, new_sl=placed_sl,
                        projected_line=new_sl, bars_since_entry=bars_since,
                        entry_price=entry_price,
                    )
                except Exception as exc:
                    print(f"[trailing] ml sl log err {symbol}: {exc}", flush=True)
                params["last_sl_set"] = placed_sl
                if placed_tp > 0:
                    params["last_tp_set"] = placed_tp
                params["last_update_bar"] = bars_since
                _save_trendline_params()
                updates += 1
                tp_msg = f"  TP {current_tp:.6f} -> {placed_tp:.6f}" if placed_tp > 0 else ""
                print(
                    f"[trailing] MOVED SL {symbol} {current_sl:.6f} -> {placed_sl:.6f}{tp_msg} "
                    f"bars={bars_since} tf={tf} verified={resp.get('sl_verified')}",
                    flush=True,
                )
            else:
                reason = resp.get("reason", "")
                print(f"[trailing] {symbol} SL update failed: {reason}", flush=True)
                # No manual close — preset SL on Bitget handles it automatically
        except Exception as e:
            print(f"[trailing] {symbol} err: {e}", flush=True)

    return updates


def _build_trendline_order_cfg(cfg: dict, equity: float, held_symbols: set[str] | None = None) -> dict:
    return {
        "buffer_pct": cfg.get("buffer_pct", 0.10),
        "rr": 15.0,
        "prices": getattr(_state, "last_prices", {}),
        "leverage": int(cfg.get("leverage", 30)),
        "equity": equity,
        "risk_pct": 0.01,
        "max_position_pct": float(cfg.get("max_position_pct", 0.50)),
        "mode": cfg.get("mode", "live"),
        "held_symbols": held_symbols or set(),
        "tf_risk": cfg.get("tf_risk", DEFAULT_RUNNER_CFG["tf_risk"]),
        "tf_buffer": {"5m": 0.05, "15m": 0.10, "1h": 0.20, "4h": 0.30},
        "stop_offset_pct": cfg.get("stop_offset_pct", DEFAULT_RUNNER_CFG["stop_offset_pct"]),
        "tf_stop_offset": cfg.get("tf_stop_offset", DEFAULT_RUNNER_CFG["tf_stop_offset"]),
        "trendline_cooldown_bars_after_close": cfg.get(
            "trendline_cooldown_bars_after_close",
            DEFAULT_RUNNER_CFG["trendline_cooldown_bars_after_close"],
        ),
    }


async def _sync_trendline_fills_and_update_trailing(cfg: dict) -> int:
    """Register newly filled trendline plans, then move SL on open positions."""
    try:
        from server.strategy.trendline_order_manager import _load_active, _save_active
        active_orders = _load_active()
        bitget_pos, pos_ok = await _get_bitget_positions()
        print(f"[trailing] sync check: pos_ok={pos_ok} positions={len(bitget_pos)} active_orders={len(active_orders)}", flush=True)
        if pos_ok and bitget_pos:
            held_positions_by_symbol = {}
            for p in bitget_pos:
                sym = (p.get("symbol") or "").upper()
                size = float(p.get("total") or p.get("available") or 0)
                if size > 0:
                    held_positions_by_symbol[sym] = p
            held_syms_upper = set(held_positions_by_symbol)
            print(f"[trailing] held positions: {held_syms_upper}", flush=True)
            active_dirty = False
            for sym_upper in held_syms_upper:
                if sym_upper in _trendline_params:
                    continue
                ao = _select_active_order_for_position(active_orders, sym_upper)
                if ao is None:
                    pos = held_positions_by_symbol.get(sym_upper, {})
                    manual_cond = _manual_conditional_for_position(sym_upper, pos)
                    if manual_cond is not None and _register_manual_conditional_trailing(sym_upper, manual_cond, pos):
                        print(
                            f"[trailing] registered manual line {sym_upper} "
                            f"tf={manual_cond.timeframe} cond={manual_cond.conditional_id}",
                            flush=True,
                        )
                        continue
                    print(f"[trailing] skip register {sym_upper}: no active trendline order candidate", flush=True)
                    continue
                pos = held_positions_by_symbol.get(sym_upper, {})
                opened_ts = _position_open_ts(pos) or int(time.time())
                register_trendline_params(
                    sym_upper,
                    slope=ao.slope, intercept=ao.intercept,
                    entry_bar=ao.bar_count - 1,
                    entry_price=ao.limit_price,
                    side="long" if ao.kind == "support" else "short",
                    tf=ao.timeframe,
                    created_ts=opened_ts,
                    tp_price=ao.tp_price,
                    last_sl_set=ao.stop_price,
                    line_ref_ts=getattr(ao, "line_ref_ts", 0) or ao.last_updated_ts,
                    line_ref_price=getattr(ao, "line_ref_price", 0) or ao.current_projected_price,
                    stop_offset_pct=_stop_offset_pct_from_active_order(ao),
                )
                from server.strategy.trade_log import log_fill
                log_fill(
                    ao.exchange_order_id,
                    sym_upper,
                    "long" if ao.kind == "support" else "short",
                    ao.limit_price,
                    0,
                    tf=ao.timeframe,
                    slope=ao.slope,
                    intercept=ao.intercept,
                )
                try:
                    from server.strategy.ml_trade_db import log_plan_triggered
                    log_plan_triggered(
                        ao.exchange_order_id,
                        sym_upper,
                        "long" if ao.kind == "support" else "short",
                        fill_price=ao.limit_price,
                        trigger_price=ao.limit_price,
                        tf=ao.timeframe,
                        slope=ao.slope,
                        intercept=ao.intercept,
                    )
                except Exception as exc:
                    print(f"[trailing] ml fill log err {sym_upper}: {exc}", flush=True)
                if ao.status != "filled":
                    ao.status = "filled"
                    active_dirty = True
                print(f"[trailing] registered {sym_upper} tf={ao.timeframe} tp={ao.tp_price:.6f}", flush=True)
            if active_dirty:
                _save_active(active_orders)
    except Exception as e:
        print(f"[trailing] sync err: {e}", flush=True)
        import traceback; traceback.print_exc()

    sl_updates = await _update_trailing_stops(cfg)
    if sl_updates > 0:
        print(f"[mar_bb] updated {sl_updates} trailing SL(s)", flush=True)

    try:
        fresh_pos, fresh_ok = await _get_bitget_positions()
        if fresh_ok:
            fresh_syms = {(p.get("symbol") or "").upper() for p in fresh_pos}
            for sym in list(_trendline_params.keys()):
                if sym not in fresh_syms:
                    params = _trendline_params[sym]
                    try:
                        from server.execution.live_adapter import LiveExecutionAdapter
                        adapter = LiveExecutionAdapter()
                        hist = await adapter._bitget_request(
                            "GET", "/api/v2/mix/position/history-position",
                            mode=cfg.get("mode", "live"), body=None,
                            params={"symbol": sym, "productType": "USDT-FUTURES", "limit": "5"},
                        )
                        closed_list = (hist.get("data") or {}).get("list") or []
                        if closed_list:
                            last = closed_list[0]
                            # Centralised field accessors (_bitget_fields) so a
                            # future Bitget rename only needs to update one
                            # module, not every call site.
                            from server.execution import _bitget_fields as bgf
                            pnl = bgf.realized_pnl_usd(last)
                            open_price = bgf.open_price(last)
                            close_price = bgf.close_price(last)
                            _m = bgf.margin_used(last)
                            # pnl_pct: prefer pnl/margin (return on margin =
                            # levered %). Fall back to pnl/notional (raw price
                            # move %) when margin field absent. User 2026-04-21
                            # saw all pnl_pct=0.00% because history-position
                            # rows don't include 'margin' only 'initialMargin'
                            # which is on /position endpoint.
                            if _m > 0:
                                pnl_pct = pnl / _m
                            else:
                                _nt = bgf.notional_usd(last)
                                pnl_pct = pnl / _nt if _nt > 0 else 0
                            from server.strategy.trade_log import log_close
                            log_close("", sym, params.get("side", ""), close_price, pnl, pnl_pct,
                                      reason="sl_or_tp", tf=params.get("tf", ""),
                                      entry_price=open_price)
                            # Feed the circuit breaker — if 3+ losses pile up,
                            # trading gets paused on the next scan tick.
                            try:
                                record_close_outcome(pnl, cfg=cfg)
                            except Exception as _exc:
                                print(f"[mar_bb] breaker record err: {_exc}", flush=True)
                            try:
                                from server.strategy.ml_trade_db import log_position_closed
                                log_position_closed(
                                    symbol=sym,
                                    direction=params.get("side", ""),
                                    tf=params.get("tf", ""),
                                    entry_price=open_price,
                                    close_price=close_price,
                                    pnl_usd=pnl,
                                    pnl_pct=pnl_pct,
                                    reason="sl_or_tp",
                                    hold_seconds=max(0, int(time.time()) - int(params.get("opened_ts", time.time()))),
                                )
                            except Exception as exc:
                                print(f"[trailing] ml close log err {sym}: {exc}", flush=True)
                            # Close the learning loop for manual-line orders:
                            # map Bitget clientOid -> conditional -> source drawing.
                            # Prefixes line_/replan_/cond_/rev_ all derive from
                            # a manual drawing (see mar_bb_history._tag_strategy).
                            try:
                                _coid = str(last.get("clientOid") or last.get("clientOId") or "").strip()
                                _coid_l = _coid.lower()
                                # Always grab the Bitget order id up front so
                                # capture_position_closed_from_drawing's
                                # exchange_order_id kwarg is defined on BOTH
                                # paths (with mlid from params OR looked up
                                # via the store). Previously only the lookup
                                # path set it, so the primary-path call raised
                                # UnboundLocalError and the event was silently
                                # dropped. Caught by Codex review 2026-04-20.
                                _order_id_s = str(last.get("orderId") or "")
                                # Primary: the manual_line_id we stored in
                                # _trendline_params at register time. This is
                                # the ground truth — we KNOW this position
                                # came from that drawing. Falls through to
                                # the lookup-by-orderId path only if params
                                # didn't carry it (older rows pre-2026-04-20).
                                _mlid = str(params.get("manual_line_id") or "").strip()
                                _matched_cond = None
                                if not _mlid and _coid_l.startswith(("line_", "replan_", "cond_", "rev_")):
                                    from server.conditionals import ConditionalOrderStore
                                    _cstore = ConditionalOrderStore()
                                    for _c in _cstore.list_all(symbol=sym):
                                        if _order_id_s and str(_c.exchange_order_id or "") == _order_id_s:
                                            _matched_cond = _c
                                            break
                                    if _matched_cond is None and _coid_l.startswith("replan_"):
                                        # replan_{conditional_id}_{ts}
                                        _cid = _coid.split("_", 2)[1] if _coid.count("_") >= 2 else ""
                                        if _cid:
                                            _matched_cond = _cstore.get(_cid)
                                    if _matched_cond is not None:
                                        _mlid = _matched_cond.manual_line_id
                                if _mlid:
                                        from server.strategy.drawing_learner import capture_position_closed_from_drawing
                                        _tf = str(params.get("tf", ""))
                                        _tf_sec = {"1m":60,"5m":300,"15m":900,"1h":3600,"4h":14400,"1d":86400}.get(_tf, 3600)
                                        _open_ts = int(params.get("opened_ts", 0) or 0)
                                        _held_s = max(0, int(time.time()) - _open_ts) if _open_ts else 0
                                        _bars_held = int(_held_s // _tf_sec) if _tf_sec else None
                                        _feat = {
                                            "close_price": float(close_price or 0),
                                            "entry_price": float(open_price or 0),
                                            "margin_used": float(_m or 0),
                                        }
                                        capture_position_closed_from_drawing(
                                            manual_line_id=_mlid,
                                            symbol=sym,
                                            timeframe=_tf,
                                            side=str(params.get("side", "")),
                                            pnl_usd=float(pnl),
                                            pnl_pct=float(pnl_pct),
                                            close_reason="sl_or_tp",
                                            bars_to_fill=None,
                                            bars_held=_bars_held,
                                            features_at_close=_feat,
                                            clientOid=_coid,
                                            exchange_order_id=_order_id_s,
                                        )
                                        print(
                                            f"[drawing_learner] position_closed_from_drawing "
                                            f"{sym} manual_line_id={_mlid} pnl=${pnl:+.4f}",
                                            flush=True,
                                        )
                                        # Also write a rich outcome record to
                                        # user_drawing_outcomes.jsonl so the Excel
                                        # "结果" sheet reflects REAL trades, not
                                        # just historical sims. User 2026-04-21.
                                        try:
                                            from server.strategy.drawing_learner import capture_live_outcome
                                            _qty = float(params.get("qty", 0) or 0)
                                            if _qty <= 0:
                                                from server.execution import _bitget_fields as _bgf2
                                                _qty = _bgf2.position_size(last)
                                            _init_stop = float(params.get("last_sl_set", 0) or 0)
                                            if _init_stop <= 0:
                                                _init_stop = None
                                            _init_tp = float(params.get("tp_price", 0) or 0)
                                            if _init_tp <= 0:
                                                _init_tp = None
                                            capture_live_outcome(
                                                manual_line_id=_mlid,
                                                symbol=sym,
                                                timeframe=_tf,
                                                side=str(params.get("side", "")),
                                                direction=str(params.get("side", "")),
                                                entry_ts=int(params.get("opened_ts", 0) or 0),
                                                entry_price=float(open_price or 0),
                                                exit_ts=int(time.time()),
                                                exit_price=float(close_price or 0),
                                                exit_reason="sl_or_tp",
                                                qty=_qty,
                                                leverage=float(params.get("leverage", 0) or 0),
                                                pnl_usd=float(pnl),
                                                pnl_pct=float(pnl_pct),
                                                initial_stop_price=_init_stop,
                                                initial_tp_price=_init_tp,
                                                exchange_order_id=_order_id_s,
                                            )
                                        except Exception as _exc:
                                            print(f"[drawing_learner] live outcome err {sym}: {_exc}", flush=True)
                            except Exception as exc:
                                print(f"[drawing_learner] close-link err {sym}: {exc}", flush=True)
                            print(f"[trailing] RESOLVED {sym} {params.get('side','')} PnL=${pnl:+.4f} ({pnl_pct*100:+.2f}%)", flush=True)
                            try:
                                from server.strategy.trendline_order_manager import mark_trendline_cooldown
                                side = str(params.get("side", "")).lower()
                                kind = "support" if side == "long" else "resistance"
                                mark_trendline_cooldown(
                                    sym,
                                    str(params.get("tf", "")),
                                    kind,
                                    bars=int(cfg.get("trendline_cooldown_bars_after_close", 4)),
                                    reason="position_closed",
                                )
                            except Exception as exc:
                                print(f"[trailing] cooldown set err {sym}: {exc}", flush=True)
                    except Exception as e:
                        print(f"[trailing] PnL fetch {sym}: {e}", flush=True)
                    print(f"[trailing] cleaned up {sym} (no longer in positions)", flush=True)
                    del _trendline_params[sym]
                    _save_trendline_params()
    except Exception as exc:
        print(f"[trailing] cleanup err: {exc}", flush=True)
    return sl_updates


async def _run_trendline_fast_maintenance(cfg: dict) -> dict:
    """Move existing plan orders and position SL without rescanning symbols."""
    enabled_strats = set(cfg.get("strategies") or [])
    if "trendline" not in enabled_strats:
        return {"ok": True, "skipped": "trendline_disabled"}

    # Halt guard: during a daily-DD halt we must NOT place/re-place plan
    # orders. Only the SL-trailing step keeps running — that tightens exits
    # on existing positions, which is risk-reducing, not risk-adding.
    # Without this guard (pre-2026-04-19) the maintenance loop happily kept
    # submitting new tl_ plans on 10s cadence even while the 60s scan loop
    # had already called HALT. Part of the post-mortem on the -81% incident.
    halt_active = bool((_state.daily_risk or {}).get("halted"))

    async with _trendline_maintenance_lock:
        result: dict[str, Any] = {"placed": 0, "updated": 0, "cancelled": 0}
        equity = await _get_equity() if cfg.get("sizing_mode") == "fixed_risk" else float(cfg.get("equity", 0) or 0)
        if halt_active:
            print("[maintenance] halt active — skipping plan-order moves, SL trailing still runs", flush=True)
        elif cfg.get("sizing_mode") != "fixed_risk" or equity > 0:
            try:
                from server.strategy.trendline_order_manager import update_trendline_orders
                tl_cfg = _build_trendline_order_cfg(cfg, equity, held_symbols=set())
                result = await update_trendline_orders([], current_bar_index=-1, cfg=tl_cfg)
                if result.get("updated") or result.get("cancelled"):
                    print(f"[maintenance] trendline existing orders: {result}", flush=True)
            except Exception as exc:
                print(f"[maintenance] trendline order maintenance err: {exc}", flush=True)
                traceback.print_exc()
        else:
            print("[maintenance] skip plan movement: equity unavailable", flush=True)

        try:
            sl_updates = await _sync_trendline_fills_and_update_trailing(cfg)
        except Exception as exc:
            print(f"[maintenance] trailing maintenance err: {exc}", flush=True)
            traceback.print_exc()
            sl_updates = 0
        return {"ok": True, **result, "sl_updates": sl_updates, "halt_active": halt_active}


# ─────────────────────────────────────────────────────────────
# Daily drawdown tracking
# ─────────────────────────────────────────────────────────────
_daily_equity_start: float = 0.0   # equity at start of current UTC day
_daily_date: str = ""              # "YYYY-MM-DD" of current tracking day


def _utc_day() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _load_daily_risk(today: str | None = None) -> bool:
    """Restore today's daily-DD baseline so a restart cannot reset it."""
    global _daily_equity_start, _daily_date
    day = today or _utc_day()
    try:
        path = _daily_risk_file()
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("date") != day:
            return False
        start = float(data.get("equity_start") or 0.0)
        if start <= 0:
            return False
        _daily_date = day
        _daily_equity_start = start
        _state.daily_risk = data
        return True
    except Exception as exc:
        print(f"[mar_bb] daily risk load err: {exc}", flush=True)
        return False


def _save_daily_risk(*, equity: float, dd_pct: float, limit: float, halted: bool) -> None:
    """Persist daily-DD state independently from runner status/counters."""
    if not _daily_date or _daily_equity_start <= 0:
        return
    data = {
        "date": _daily_date,
        "equity_start": _daily_equity_start,
        "last_equity": equity,
        "last_dd_pct": dd_pct,
        "limit_pct": limit,
        "halted": halted,
        "updated_ts": int(time.time()),
    }
    _state.daily_risk = data
    try:
        path = _daily_risk_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[mar_bb] daily risk save err: {exc}", flush=True)


def record_close_outcome(
    pnl: float,
    *,
    cfg: dict | None = None,
    close_ts: float | None = None,
) -> dict:
    """Feed the consecutive-loss circuit breaker with the outcome of a single
    real-time close. Returns {triggered, streak, breaker_until_ts, outcome}.

    Two guards prevent the breaker from firing on stale data:

    1. If `close_ts` is older than when the runner was last started
       (_runner_started_ts), the event is a historical replay (from
       `_sync_trendline_fills_and_update_trailing` catching up at startup)
       and MUST NOT count toward the streak. Otherwise yesterday's 59 fills
       wipe out today's budget the moment we reboot.
    2. Ditto if `close_ts` is > 30 min in the past — same intent.

    Breaker semantics (changed 2026-04-19 after user feedback):
      - Trigger: 3 consecutive real-time losses → breaker set to
        `breaker_until_ts = +inf` (indefinite).
      - Clear: ONLY via an explicit call to `resume_breaker_after_optimize()`
        (i.e. operator reviewed the losses, adjusted the model gate, and
        affirmed the algorithm has been improved). A plain time-based
        expiry is NOT the right abstraction — "6 losses" is a signal the
        algorithm is wrong, not that it needs a nap.
      - A win still auto-clears streak count (so an eventual winning trade
        resets without manual intervention), but the breaker_until_ts only
        clears via the explicit resume path.
    """
    global _state
    cfg = cfg or {}
    threshold = int(cfg.get("consecutive_loss_threshold", 3))

    now = time.time()
    close_ts = float(close_ts or now)
    started = float(_runner_started_ts or 0.0)
    # Historical replay guard — do NOT count events that predate the
    # current runner-start window, or older than 30 min regardless.
    if started > 0 and close_ts < started:
        print(f"[mar_bb] breaker skip: close_ts={close_ts:.0f} < started={started:.0f} (historical replay)", flush=True)
        return {"triggered": False, "streak": -1, "skipped": "historical_pre_start",
                "breaker_until_ts": _state.breaker_until_ts, "outcome": None}
    if close_ts < now - 1800:
        print(f"[mar_bb] breaker skip: close_ts is {int(now-close_ts)}s old (>30min)", flush=True)
        return {"triggered": False, "streak": -1, "skipped": "too_stale",
                "breaker_until_ts": _state.breaker_until_ts, "outcome": None}

    outcome = "win" if float(pnl or 0) > 0 else "loss"
    outcomes = list(_state.recent_close_outcomes or [])
    outcomes.append(outcome)
    outcomes = outcomes[-10:]
    _state.recent_close_outcomes = outcomes
    streak = 0
    for o in reversed(outcomes):
        if o == "loss":
            streak += 1
        else:
            break
    triggered = False
    if streak >= threshold:
        # Indefinite hold — cleared only via resume_breaker_after_optimize.
        _state.breaker_until_ts = float("inf")
        _state.breaker_reason = (
            f"{streak} consecutive losses — algorithm needs review. "
            "Call /api/mar-bb/breaker-resume after verifying thresholds "
            "have been tightened."
        )
        triggered = True
        print(f"[mar_bb] CIRCUIT BREAKER (indefinite): {_state.breaker_reason}", flush=True)
    elif outcome == "win":
        # A win resets the streak. It does NOT clear an already-triggered
        # indefinite breaker — operator still needs to explicitly resume.
        # (This prevents "one lucky bounce" from un-breaking a broken loop.)
        pass
    try:
        _save_state()
    except Exception:
        pass
    return {
        "triggered": triggered,
        "streak": streak,
        "breaker_until_ts": _state.breaker_until_ts,
        "outcome": outcome,
    }


def breaker_active(now_ts: float | None = None) -> tuple[bool, float]:
    """Returns (active, seconds_remaining). Indefinite breakers (until_ts=inf)
    report (True, inf) — caller should treat that as "never naturally
    recovers" and surface it as a manual-review required state."""
    now = float(now_ts if now_ts is not None else time.time())
    until = float(_state.breaker_until_ts or 0)
    if until == float("inf"):
        return True, float("inf")
    if until > now:
        return True, until - now
    return False, 0.0


def resume_breaker_after_optimize(*, confirm_code: str, model_gate_tighten: float = 0.05) -> dict:
    """Operator endpoint — clears an indefinite breaker after affirming the
    algorithm was reviewed/tightened. Auto-bumps both model-gate thresholds
    up by `model_gate_tighten` (default +0.05) as a concrete "optimized"
    delta; operator can override via explicit update-config later.

    Requires confirm_code='RESUME' to prevent accidental clears.
    """
    if (confirm_code or "").strip().upper() != "RESUME":
        return {"ok": False, "reason": "confirm_code_mismatch — send {'confirm_code': 'RESUME'}"}
    if not _state.breaker_until_ts:
        return {"ok": False, "reason": "breaker not active"}
    cfg = dict(_state.config or {})
    gate = dict(cfg.get("model_gate") or {})
    old_trade = float(gate.get("min_trade_win_prob") or 0.55)
    old_line = float(gate.get("min_line_quality_prob") or 0.55)
    new_trade = min(0.95, round(old_trade + model_gate_tighten, 3))
    new_line = min(0.95, round(old_line + model_gate_tighten, 3))
    gate["min_trade_win_prob"] = new_trade
    gate["min_line_quality_prob"] = new_line
    cfg["model_gate"] = gate
    _state.config = cfg
    before_reason = _state.breaker_reason
    _state.breaker_until_ts = 0.0
    _state.breaker_reason = ""
    _state.recent_close_outcomes = []  # fresh streak counter
    try:
        _save_state()
    except Exception:
        pass
    print(
        f"[mar_bb] breaker resumed — model_gate tightened "
        f"trade:{old_trade}->{new_trade} line:{old_line}->{new_line}",
        flush=True,
    )
    return {
        "ok": True,
        "before_reason": before_reason,
        "model_gate_before": {"trade": old_trade, "line": old_line},
        "model_gate_after": {"trade": new_trade, "line": new_line},
    }


def reset_daily_halt(*, confirm_code: str) -> dict:
    """Manually clear the daily-DD halt flag.

    This is intentionally guarded by a literal confirm code — halt exists to
    protect the user from a cascading loss; bypassing it must be a deliberate
    act. When halt is cleared, the runner's next scan will be allowed to place
    orders again.

    Returns {ok, reason, before, after}.
    """
    if (confirm_code or "").strip().upper() != "RESET":
        return {
            "ok": False,
            "reason": "confirm_code_mismatch — send {'confirm_code': 'RESET'}",
        }
    before = dict(_state.daily_risk or {})
    # Fall back to sensible defaults if no daily-risk record exists yet.
    equity = float(before.get("last_equity") or 0.0)
    dd_pct = float(before.get("last_dd_pct") or 0.0)
    limit = float(before.get("limit_pct") or 0.5)
    try:
        _save_daily_risk(equity=equity, dd_pct=dd_pct, limit=limit, halted=False)
    except Exception as exc:
        return {"ok": False, "reason": f"save failed: {exc}"}
    print(
        f"[mar_bb] daily halt reset (was dd={dd_pct*100:.1f}% limit={limit*100:.0f}%); "
        "next scan may place new orders again.",
        flush=True,
    )
    return {"ok": True, "before": before, "after": dict(_state.daily_risk or {})}


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

    today = _utc_day()
    if today != _daily_date or _daily_equity_start <= 0:
        _load_daily_risk(today)

    if today != _daily_date or _daily_equity_start <= 0:
        _daily_equity_start = equity
        _daily_date = today
        limit = _get_daily_dd_limit(equity, cfg)
        _save_daily_risk(equity=equity, dd_pct=0.0, limit=limit, halted=False)
        return False, 0.0, limit

    if _daily_equity_start <= 0:
        return False, 0.0, 0.50

    dd_pct = (_daily_equity_start - equity) / _daily_equity_start
    limit = _get_daily_dd_limit(_daily_equity_start, cfg)
    halted = dd_pct >= limit
    _save_daily_risk(equity=equity, dd_pct=dd_pct, limit=limit, halted=halted)

    if halted:
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
        t = np.array([int(c.get("timestamp") or c.get("time") or 0) for c in candles], dtype=np.int64)
        vol = data.get("volume") or []
        if vol:
            v = np.array([x.get("value", 0) for x in vol], dtype=float)
        else:
            v = np.ones(len(c), dtype=float)   # fallback if volume missing
        return {"o": o, "h": h, "l": l, "c": c, "v": v, "t": t}
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
    # V3 fix: small SL (0.3% from entry), big TP (BB or RR=8 whichever is further)
    sl_pct = 0.003  # 0.3% stop — tight, like trendline
    rr = 8.0

    if signal == 1:
        entry = close
        stop = entry * (1 - sl_pct)
        pct_tp = entry * (1 + sl_pct * rr)  # RR=8 target
        bb_tp = float(bb_up) if bb_up else pct_tp
        tp = max(pct_tp, bb_tp)  # take the FURTHER target
        direction = "long"
    else:
        entry = close
        stop = entry * (1 + sl_pct)
        pct_tp = entry * (1 - sl_pct * rr)
        bb_tp = float(bb_lo) if bb_lo else pct_tp
        tp = min(pct_tp, bb_tp)  # take the FURTHER target (lower for short)
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
    # TF-aware post-anchor penetration tolerance. The default is 0 = ANY wick
    # after anchor2 invalidates the line. On 4h, candles are so large that
    # nearly every line has a wick pokethrough → 0 signals across 500 bars on
    # 10 coins (empirically verified 2026-04-19, see debug_4h_detector_v3.py).
    # Loosen to 1 for 1h/4h while keeping the strict 0 for 5m/15m.
    tf_post_pen = {"5m": 0, "15m": 0, "1h": 1, "4h": 1}.get(tf, 0)
    tl_cfg = {**TRENDLINE_DEFAULT_CONFIG, "max_post_penetrations": tf_post_pen}
    try:
        sigs, entries, sls, tps, lines = trendline_generate_signals(
            bars["o"], bars["h"], bars["l"], bars["c"], bars["v"],
            tl_cfg,
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
    # True anchor bar indices i1/i2 from the trendline enumeration — needed
    # so `_persist_auto_line` draws the line FROM pivot1 TO pivot2, not from
    # bar 0 to signal-bar (which stretches the line across the whole window).
    line_i1 = int(line_info["i1"]) if line_info and "i1" in line_info else 0
    line_i2 = int(line_info["i2"]) if line_info and "i2" in line_info else i

    plan = {
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
        "line_i1": line_i1,
        "line_i2": line_i2,
    }

    try:
        from server.strategy.trendline_model_gate import score_trendline_gate, result_dict
        kind = "support" if direction == "long" else "resistance"
        gate = score_trendline_gate(
            symbol=symbol,
            timeframe=tf,
            kind=kind,
            line_info=line_info or {},
            bars=bars,
            cfg=cfg,
        )
        plan["model_gate"] = result_dict(gate)
        if not gate.accepted:
            print(
                f"[trendline_model_gate] REJECT {symbol} {tf} {kind}: {gate.reason} "
                f"line={gate.line_quality_prob} trade={gate.trade_win_prob}",
                flush=True,
            )
            return None
        print(
            f"[trendline_model_gate] PASS {symbol} {tf} {kind}: {gate.reason} "
            f"line={gate.line_quality_prob} trade={gate.trade_win_prob}",
            flush=True,
        )
    except Exception as e:
        print(f"[trendline_model_gate] fail-open {symbol} {tf}: {e}", flush=True)

    return plan


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
        if equity <= 0:
            _state.last_error = "equity_fetch_failed — halting scan"
            print(f"[mar_bb] HALT: equity=${equity:.2f}, skipping scan", flush=True)
            _state.scans_completed += 1
            return
        print(f"[mar_bb] equity=${equity:.2f}", flush=True)

    # Consecutive-loss circuit breaker: if N losses in a row, pause new-order
    # placement for a cooldown window. Same shape as daily halt — we skip the
    # rest of the scan (including signal detection) but let maintenance keep
    # trailing existing SL to protect still-open positions.
    brk_active, brk_remaining = breaker_active()
    if brk_active:
        _state.last_error = (
            f"CIRCUIT BREAKER: {_state.breaker_reason or 'consecutive losses'} "
            f"({int(brk_remaining)}s remaining)"
        )
        print(f"[mar_bb] {_state.last_error}", flush=True)
        _state.last_scan_ts = int(time.time())
        _state.scans_completed += 1
        _save_state()
        return

    # Daily drawdown check
    dd_halted, dd_current, dd_limit = _check_daily_dd(equity, cfg)
    if dd_halted:
        _state.last_error = (
            f"DAILY DD HALT: lost {dd_current*100:.1f}% today (limit {dd_limit*100:.0f}% "
            f"for ${_daily_equity_start:.0f} tier). No new trades until UTC midnight."
        )
        print(f"[mar_bb] {_state.last_error}", flush=True)
        # Cancel all pending auto plan orders so in-flight triggers can't fire
        # AFTER the halt boundary. Re-runs are idempotent — only tl_-prefixed
        # orders are cancelled (manual line_/cond_/replan_ are preserved).
        try:
            from server.strategy.trendline_order_manager import cancel_all_trendline_plan_orders
            cancel_result = await cancel_all_trendline_plan_orders(cfg, status="daily_halt")
            print(f"[mar_bb] daily DD halt cancelled trendline plans: {cancel_result}", flush=True)
        except Exception as exc:
            print(f"[mar_bb] daily DD halt cancel err: {exc}", flush=True)
        # Auto-flatten OPEN positions too — the -81% incident on 2026-04-18
        # showed that cancelling pending plans is not enough: positions that
        # had already filled continued to bleed via Bitget-side preset SL/TP
        # until they hit their own stops. Halt must CLOSE, not just pause.
        # Uses the same flatten-all path as the UI 紧急平仓 button.
        try:
            from ..execution.live_adapter import LiveExecutionAdapter
            from ..execution.live_engine import LiveExecutionEngine, LiveBridgeConfig
            _adapter = LiveExecutionAdapter()
            _engine = LiveExecutionEngine(
                adapter_provider=lambda: _adapter,
                config=LiveBridgeConfig.from_env(),
            )
            mode = cfg.get("mode", "live")
            account = await _adapter.get_live_account_status(mode)
            closed = 0; attempted = 0
            for pos in (account.get("positions") or []):
                sym = str(pos.get("symbol") or "").upper()
                size = float(pos.get("total") or 0)
                if not sym or size <= 0:
                    continue
                attempted += 1
                try:
                    r = await _engine.close_live_position(sym, mode=mode, confirm=True)
                    if r.get("ok"):
                        closed += 1
                    else:
                        print(f"[mar_bb] halt flatten {sym} skipped: {r.get('reason')}", flush=True)
                except Exception as exc:
                    print(f"[mar_bb] halt flatten {sym} err: {exc}", flush=True)
            print(f"[mar_bb] HALT auto-flatten: closed={closed}/{attempted} positions", flush=True)
        except Exception as exc:
            print(f"[mar_bb] halt auto-flatten failed: {exc}", flush=True)
            import traceback as _tb; _tb.print_exc()
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
    if not ok:
        _state.last_error = "position_fetch_failed — halting scan to prevent duplicate orders"
        print(f"[mar_bb] HALT: position fetch failed, skipping scan", flush=True)
        _state.scans_completed += 1
        return
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
    enabled_strats = cfg.get("strategies") or ["trendline"]
    signals_this_scan = 0
    orders_this_scan = 0
    _trendline_limit_signals = []  # collected per scan, batch-placed as limit orders

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
                        # V3: use 0.3% SL for sizing (matches _build_order_intent_mar_bb)
                        sl_pct = 0.003
                        stop_p = entry_p * (1 - sl_pct) if state["signal"] == 1 else entry_p * (1 + sl_pct)
                        scan_cfg["_notional_override"] = _calc_position_size(equity, entry_p, stop_p, scan_cfg)
                    plan = _build_order_intent_mar_bb(sym, tf, state, scan_cfg)
                    if plan:
                        plan["strategy"] = rstrat_name  # tag which ribbon config
                except Exception as e:
                    print(f"[mar_bb] {rstrat_name} {sym} {tf} err: {e}", flush=True)

            # Trendline: collect signals for LIMIT ORDER placement (not market order)
            if "trendline" in enabled_strats:
                try:
                    scan_cfg["_notional_override"] = None
                    tl_plan = _check_trendline_signal(sym, tf, bars, scan_cfg)
                    if tl_plan and tl_plan.get("strategy") == "trendline":
                        # Don't set `plan` — trendline goes through limit order path
                        _slope = tl_plan.get("line_slope", 0) or 0.0
                        _intercept = tl_plan.get("line_intercept", 0) or 0.0
                        # Use the TRUE anchor bar indices from the trendline
                        # enumeration (i1=first pivot, i2=second pivot).
                        # Falling back to 0 / line_entry_bar is wrong: it draws
                        # the line from the start of the data window to the
                        # signal bar, which can span hundreds of bars even if
                        # the real line spans only a few.
                        _a1_bar = int(tl_plan.get("line_i1", 0) or 0)
                        _a2_bar = int(tl_plan.get("line_i2") or tl_plan.get("line_entry_bar", 0) or 0)
                        _ts_arr = bars.get("t")
                        _n = len(bars["c"])
                        # Bounds-safe lookup into the bar timestamp array (ms).
                        def _bar_ts(bar):
                            if _ts_arr is None: return 0
                            i = max(0, min(int(bar), _n - 1))
                            try: return int(_ts_arr[i])
                            except Exception: return 0
                        # Projected line price at the two anchor bars — used to
                        # persist a user-visible trendline when the plan fills.
                        _a1_ts = _bar_ts(_a1_bar)
                        _a2_ts = _bar_ts(_a2_bar)
                        _a1_price = float(_slope * _a1_bar + _intercept)
                        _a2_price = float(_slope * _a2_bar + _intercept)
                        _trendline_limit_signals.append({
                            "symbol": sym,
                            "timeframe": tf,
                            "kind": "support" if tl_plan["direction"] == "long" else "resistance",
                            "slope": _slope,
                            "intercept": _intercept,
                            "anchor1_bar": _a1_bar,
                            "anchor2_bar": _a2_bar,
                            "anchor1_ts": _a1_ts,
                            "anchor2_ts": _a2_ts,
                            "anchor1_price": _a1_price,
                            "anchor2_price": _a2_price,
                            "direction": tl_plan["direction"],
                            "entry_price": tl_plan["entry_price"],
                            "stop_price": tl_plan["stop_price"],
                            "tp_price": tl_plan["tp_price"],
                            "bar_count": _n,  # actual bar count for projection
                            "model_gate": tl_plan.get("model_gate"),
                        })
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
                # Emit event for Telegram notification + log trade
                try:
                    from server.core.events import bus, Event
                    payload = {
                        "symbol": sym, "side": plan.get("direction", "long"),
                        "size_usd": plan.get("notional", 0),
                        "entry_price": plan.get("entry", 0),
                        "sl": plan.get("stop", 0), "tp": plan.get("tp", 0),
                        "strategy": plan.get("strategy", ""),
                        "timeframe": tf,
                    }
                    bus.emit(Event("position.opened", payload))
                except Exception as exc:
                    print(f"[mar_bb] event emit err {sym}: {exc}", flush=True)
                    traceback.print_exc()
                try:
                    from server.strategy.trade_log import log_trade
                    log_trade(
                        symbol=sym, timeframe=tf,
                        strategy=plan.get("strategy", ""),
                        direction=plan.get("direction", ""),
                        entry_price=plan.get("entry", 0),
                        stop_price=plan.get("stop", 0),
                        tp_price=plan.get("tp", 0),
                        size_usd=plan.get("notional", 0),
                        leverage=int(cfg.get("leverage", 20)),
                        order_id=resp.get("exchange_order_id", ""),
                    )
                except Exception as exc:
                    print(f"[mar_bb] trade log err {sym}: {exc}", flush=True)
                    traceback.print_exc()
            else:
                _state.orders_rejected += 1
                _state.last_error = f"{sym}: {resp.get('reason', 'unknown')}"
                print(f"[mar_bb] REJECTED {sym}: {_state.last_error}", flush=True)

    # ── Trendline plan orders: place new trigger-market entries ──
    if "trendline" in enabled_strats:
        try:
            from server.strategy.trendline_order_manager import update_trendline_orders
            tl_cfg = _build_trendline_order_cfg(cfg, equity, held_symbols=held_symbols)
            tl_cfg["tf_risk"] = tf_risk
            print(f"[trendline_orders] equity=${equity:.2f} signals={len(_trendline_limit_signals)} risk_pct={tl_cfg['risk_pct']}", flush=True)
            # current_bar_index = length of the OHLCV data (bar count, NOT timestamp)
            # The trendline slope/intercept are relative to bar indices 0..N
            last_bars_len = max(sig.get("bar_count", 500) for sig in _trendline_limit_signals) if _trendline_limit_signals else 0
            async with _trendline_maintenance_lock:
                tl_result = await update_trendline_orders(
                    _trendline_limit_signals,
                    current_bar_index=last_bars_len - 1,
                    cfg=tl_cfg,
                )
            if (
                tl_result.get("placed", 0) > 0
                or tl_result.get("updated", 0) > 0
                or tl_result.get("cancelled", 0) > 0
            ):
                print(f"[mar_bb] trendline plan orders: {tl_result}", flush=True)
        except Exception as e:
            print(f"[mar_bb] trendline plan orders err: {e}", flush=True)
            traceback.print_exc()

    # ── Sync filled plan orders and move trailing SL. This shares the same
    # lock as the 10s maintenance loop so active-order file updates do not
    # race with a full scan finishing at the same time.
    try:
        async with _trendline_maintenance_lock:
            await _sync_trendline_fills_and_update_trailing(cfg)
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

    # ── Level 3-5: Evolution system hooks ──
    try:
        from server.strategy.evolution import after_scan, daily_maintenance, weekly_maintenance
        import asyncio
        scan_hook = after_scan()
        if asyncio.iscoroutine(scan_hook):
            await scan_hook
        # These may be async in some versions; handle both
        dm = daily_maintenance()
        if asyncio.iscoroutine(dm):
            await dm
        wm = weekly_maintenance()
        if asyncio.iscoroutine(wm):
            await wm
    except Exception as e:
        print(f"[evolution] hook err: {e}", flush=True)

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


async def _maintenance_loop() -> None:
    """Fast loop (10s) for plan-order movement + fill detection.
    Trailing SL actually fires on bar boundary via _trailing_scheduler_loop."""
    print("[maintenance] trendline maintenance loop started", flush=True)
    try:
        while _running:
            try:
                cfg = {**DEFAULT_RUNNER_CFG, **(_state.config or {})}
                await _run_trendline_fast_maintenance(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[maintenance] loop err: {exc}", flush=True)
                traceback.print_exc()
            await asyncio.sleep(int((_state.config or DEFAULT_RUNNER_CFG).get("maintenance_interval_s", 10)))
    except asyncio.CancelledError:
        print("[maintenance] trendline maintenance loop cancelled", flush=True)
    finally:
        print("[maintenance] trendline maintenance loop exited", flush=True)


_TF_SECONDS_MAP = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400,
    "6h": 21600, "12h": 43200, "1d": 86400, "1w": 604800,
}


def _seconds_until_next_bar_for_tfs(tfs: set[str]) -> float | None:
    """Seconds until the earliest next bar boundary among given TFs.
    Returns None if no TFs. Settle buffer +1s so Bitget's candle has rolled."""
    import time as _t
    if not tfs:
        return None
    now = _t.time()
    deadlines = []
    for tf in tfs:
        bar_sec = _TF_SECONDS_MAP.get(tf)
        if bar_sec:
            next_boundary = (int(now) // bar_sec + 1) * bar_sec
            deadlines.append(next_boundary - now + 1.0)  # +1s settle
    return min(deadlines) if deadlines else None


async def _trailing_scheduler_loop() -> None:
    """Precise bar-boundary scheduler for trailing SL updates.

    User 2026-04-22 spec: "12:30 买的 1h 单, 应该在 12:59:59 等着,
    然后 13:00:01 更新 SL 跟着线移动". NOT a 10s / 1m polling loop.

    Behavior:
      1. Find all active trailing positions → get their unique TFs
      2. Compute min(next_bar_boundary_across_all_tfs) + 1s settle
      3. Sleep that long
      4. Run trailing update (cheap — skips positions whose bar
         didn't roll)
      5. Repeat

    No positions → sleep 60s and re-check. Cheap idle.
    Positions on 4h → sleep up to 4h and wake precisely at boundary.
    Mixed 1h + 4h → wakes at 1h boundaries (wins min), handles 4h
    too on the hour that coincides.
    """
    print("[trailing_sched] bar-boundary scheduler started", flush=True)
    try:
        while _running:
            try:
                # Collect unique TFs from active positions
                active_tfs: set[str] = set()
                for params in _trendline_params.values():
                    tf = str(params.get("tf", "") or "")
                    if tf in _TF_SECONDS_MAP:
                        active_tfs.add(tf)

                if not active_tfs:
                    # No trailing positions → idle poll every 60s
                    await asyncio.sleep(60)
                    continue

                sleep_s = _seconds_until_next_bar_for_tfs(active_tfs)
                if sleep_s is None or sleep_s <= 0:
                    sleep_s = 1.0
                # Cap maximum single sleep at 4h so the scheduler
                # re-evaluates if new positions on shorter TFs appear.
                sleep_s = min(sleep_s, 14400.0)
                print(f"[trailing_sched] sleeping {sleep_s:.1f}s until next bar boundary "
                      f"(active TFs: {sorted(active_tfs)})", flush=True)
                await asyncio.sleep(sleep_s)

                # Fire trailing update NOW — bar just opened
                try:
                    cfg = {**DEFAULT_RUNNER_CFG, **(_state.config or {})}
                    n = await _update_trailing_stops(cfg)
                    print(f"[trailing_sched] bar-boundary update: {n} SL change(s)", flush=True)
                except Exception as exc:
                    print(f"[trailing_sched] update err: {exc}", flush=True)
                    traceback.print_exc()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[trailing_sched] loop err: {exc}", flush=True)
                traceback.print_exc()
                await asyncio.sleep(10)  # back off on repeated errors
    except asyncio.CancelledError:
        print("[trailing_sched] bar-boundary scheduler cancelled", flush=True)
    finally:
        print("[trailing_sched] bar-boundary scheduler exited", flush=True)


def start_runner(config: dict | None = None) -> dict:
    """Start the runner loop. Idempotent — returns current state if
    already running."""
    global _task, _maintenance_task, _trailing_scheduler_task, _running, _runner_started_ts
    if _task is not None and not _task.done():
        return {"ok": True, "already_running": True, "state": get_state()}
    merged = {**DEFAULT_RUNNER_CFG, **(config or {})}
    _state.config = merged
    _state.last_error = ""
    _load_state()
    _running = True
    _runner_started_ts = time.time()
    _task = asyncio.create_task(_loop(), name="mar_bb_runner")
    _maintenance_task = asyncio.create_task(_maintenance_loop(), name="trendline_maintenance")
    # User 2026-04-22: precise bar-boundary trailing scheduler. Wakes
    # up at each TF bar boundary +1s and fires SL update.
    _trailing_scheduler_task = asyncio.create_task(_trailing_scheduler_loop(), name="trailing_scheduler")
    return {"ok": True, "started": True, "state": get_state()}


def stop_runner() -> dict:
    global _task, _maintenance_task, _trailing_scheduler_task, _running
    _running = False
    if _task and not _task.done():
        _task.cancel()
    if _maintenance_task and not _maintenance_task.done():
        _maintenance_task.cancel()
    if _trailing_scheduler_task and not _trailing_scheduler_task.done():
        _trailing_scheduler_task.cancel()
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


_maintenance_only_task: asyncio.Task | None = None


async def _maintenance_only_loop() -> None:
    """Standalone trailing-SL loop.

    Runs whenever the server is up, REGARDLESS of whether the main scan
    runner is started. Only touches existing positions (trailing SL) and
    existing line orders; never opens new trades. Yields to the main
    runner's `_maintenance_loop` while the runner is active to avoid
    racing Bitget API calls.

    Why this exists: user-placed manual line orders (via Trade Plan modal)
    depend on trailing to move SL along the drawn line. Before this loop,
    trailing only ran when the auto-scanner was on — so if the user had
    `MAR_BB_AUTOSTART=0` and drew a line manually, the SL would never
    move. See incident 2026-04-20 RAVEUSDT: stop stuck at fill-time line
    price because `_trendline_params` was wiped on uvicorn reload and
    nobody was re-registering it.
    """
    print("[trendline_maintenance_only] started", flush=True)
    try:
        while True:
            try:
                if _running:
                    # Main runner's `_maintenance_loop` is handling this —
                    # yield to avoid racing on Bitget calls.
                    pass
                else:
                    cfg = {**DEFAULT_RUNNER_CFG, **(_state.config or {})}
                    await _run_trendline_fast_maintenance(cfg)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[trendline_maintenance_only] err: {exc}", flush=True)
                traceback.print_exc()
            await asyncio.sleep(
                int((_state.config or DEFAULT_RUNNER_CFG).get("maintenance_interval_s", 10))
            )
    except asyncio.CancelledError:
        print("[trendline_maintenance_only] cancelled", flush=True)


def start_maintenance_only() -> dict:
    """Boot the always-on trailing-SL loop. Idempotent."""
    global _maintenance_only_task, _trailing_scheduler_task, _runner_started_ts
    if _maintenance_only_task is not None and not _maintenance_only_task.done():
        return {"ok": True, "already_running": True}
    _load_trendline_params()
    if _runner_started_ts == 0:
        _runner_started_ts = time.time()
    _maintenance_only_task = asyncio.create_task(
        _maintenance_only_loop(),
        name="trendline_maintenance_only",
    )
    # Bar-boundary scheduler also boots here so trailing works when
    # only the maintenance-only mode is on (not the full runner).
    if _trailing_scheduler_task is None or _trailing_scheduler_task.done():
        _trailing_scheduler_task = asyncio.create_task(
            _trailing_scheduler_loop(),
            name="trailing_scheduler",
        )
    return {"ok": True, "started": True}


# Preload persisted params at import time so any code that reads
# `_trendline_params` immediately (e.g., `/api/mar-bb/state`) sees the
# saved entries before the maintenance loop starts.
try:
    _load_trendline_params()
except Exception as _exc:
    print(f"[trendline_params] preload err: {_exc}", flush=True)


__all__ = [
    "start_runner",
    "stop_runner",
    "start_maintenance_only",
    "get_state",
    "manual_kick",
    "update_config",
    "reset_daily_halt",
    "record_close_outcome",
    "breaker_active",
    "DEFAULT_RUNNER_CFG",
]

