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
    "timeframe": "1h",
    "scan_interval_s": 60,
    # ── Position sizing ──
    "sizing_mode": "fixed_risk",    # "fixed_risk" or "fixed_notional"
    "risk_pct": 0.03,               # 3% equity risk per trade
    "notional_usd": 12.0,           # fallback for fixed_notional mode
    "max_position_pct": 0.50,       # max 50% equity per single position
    "leverage": 20,
    "max_concurrent_positions": 10,
    # ── General ──
    "mode": "live",              # or "demo" for paper trading
    "min_bars": 100,             # need at least this many for indicators
    "dry_run": False,            # if True, log signals but don't submit orders
    "auto_start": True,          # auto-boot runner on server startup
    "strategies": ["mar_bb", "trendline"],   # which strategies to scan with
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
    """One pass through top-N symbols. Updates _state."""
    t0 = time.time()
    cfg = _state.config
    tf = cfg["timeframe"]
    top_n = int(cfg["top_n"])
    max_concurrent = int(cfg.get("max_concurrent_positions", 5))

    print(f"[mar_bb] scan start: top_n={top_n} tf={tf} max={max_concurrent} "
          f"leverage={cfg.get('leverage')}x risk={cfg.get('risk_pct',0)*100:.1f}%", flush=True)

    # Fetch equity ONCE per scan for risk-based sizing
    equity = await _get_equity() if cfg.get("sizing_mode") == "fixed_risk" else 0
    if cfg.get("sizing_mode") == "fixed_risk":
        print(f"[mar_bb] equity=${equity:.2f}", flush=True)

    symbols = await _get_top_symbols(top_n)
    print(f"[mar_bb] fetched {len(symbols)} symbols", flush=True)
    if not symbols:
        _state.last_error = "no symbols fetched"
        return

    # Snapshot live Bitget positions ONCE at scan start. Use this set to:
    #   - count slots used (gate vs max_concurrent)
    #   - skip symbols we already hold (don't stack into same asset)
    bitget_positions, ok = await _get_bitget_positions()
    print(f"[mar_bb] bitget positions: ok={ok} count={len(bitget_positions)}", flush=True)
    held_symbols: set[str] = set()
    for p in bitget_positions:
        sym = (p.get("symbol") or "").upper()
        if sym:
            held_symbols.add(sym)
    slots_used = len(held_symbols)
    slots_free = max(0, max_concurrent - slots_used)
    print(f"[mar_bb] slots: used={slots_used} free={slots_free} held={sorted(held_symbols)}", flush=True)

    # Map tf → bar-window days (enough to compute MA55 + ADX + BB)
    tf_days = {
        "1m": 1, "3m": 2, "5m": 3, "15m": 7, "30m": 14,
        "1h": 21, "2h": 42, "4h": 84, "1d": 500,
    }
    days = tf_days.get(tf, 21)

    strategy = Strategy(DEFAULT_CONFIG)
    enabled_strats = cfg.get("strategies") or ["mar_bb", "trendline"]
    signals_this_scan = 0
    orders_this_scan = 0

    if slots_free == 0:
        _state.last_error = (
            f"no free slots: {slots_used}/{max_concurrent} "
            f"({','.join(sorted(held_symbols))[:200]})"
        )

    scanned = 0
    for sym in symbols:
        if not _running:
            break
        if orders_this_scan >= slots_free:
            break
        if sym.upper() in held_symbols:
            continue

        scanned += 1
        if scanned <= 3 or scanned % 20 == 0:
            print(f"[mar_bb]   scanning {sym} ({scanned}/{len(symbols)})", flush=True)

        try:
            bars = await _fetch_bars(sym, tf, days)
        except Exception as e:
            print(f"[mar_bb] fetch {sym} EXCEPTION: {e}", flush=True)
            continue
        if bars is None:
            continue
        if len(bars["c"]) < cfg["min_bars"]:
            continue

        # Run BOTH strategies on the same bars. First to produce a plan wins.
        plan = None

        if "mar_bb" in enabled_strats:
            try:
                state = strategy.current_state(
                    bars["o"], bars["h"], bars["l"], bars["c"], bars["v"],
                )
                # Pre-calculate notional for this signal's SL distance
                if state.get("signal") in (1, -1) and equity > 0:
                    entry_p = state.get("close", 0)
                    atr_val = state.get("atr", 0)
                    atr_mult = float(DEFAULT_CONFIG.get("atr_mult", 3.0))
                    stop_p = entry_p - atr_mult * atr_val if state["signal"] == 1 else entry_p + atr_mult * atr_val
                    cfg["_notional_override"] = _calc_position_size(equity, entry_p, stop_p, cfg)
                plan = _build_order_intent_mar_bb(sym, tf, state, cfg)
            except Exception as e:
                print(f"[mar_bb] mar_bb {sym} err: {e}", flush=True)

        if plan is None and "trendline" in enabled_strats:
            try:
                # For trendline, compute notional after getting signal's SL
                cfg["_notional_override"] = None  # will be set inside if signal found
                plan = _check_trendline_signal(sym, tf, bars, cfg)
                if plan and equity > 0:
                    plan_notional = _calc_position_size(equity, plan["entry_price"], plan["stop_price"], cfg)
                    plan["notional"] = plan_notional
                    plan["quantity"] = plan_notional / plan["entry_price"]
            except Exception as e:
                print(f"[mar_bb] trendline {sym} err: {e}", flush=True)

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
            orders_this_scan += 1   # count against the slot quota even in dry-run
            continue

        print(f"[mar_bb] SIGNAL [{plan.get('strategy')}] {sym} {plan['direction']} @ {plan['entry_price']:.4f}  "
              f"sl={plan['stop_price']:.4f}  tp={plan['tp_price']:.4f}", flush=True)

        # Re-anchor SL/TP to the current mark so slippage between signal
        # generation and market fill doesn't distort the intended RR ratio.
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
            held_symbols.add(sym.upper())   # don't pick same sym again this scan
            print(f"[mar_bb] FILLED {sym} order_id={resp.get('exchange_order_id')}", flush=True)
        else:
            _state.orders_rejected += 1
            _state.last_error = f"{sym}: {resp.get('reason', 'unknown')}"
            print(f"[mar_bb] REJECTED {sym}: {_state.last_error}", flush=True)

    _state.scans_completed += 1
    _state.last_scan_ts = int(time.time())
    _state.last_scan_duration_s = round(time.time() - t0, 2)
    # Mirror snapshot into state so UI can see what's held
    _state.open_position = {
        "count": slots_used,
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
