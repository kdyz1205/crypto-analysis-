"""MA-ribbon live auto-execution scanner.

Tick every 60 s:
  1. load state
  2. enforce gates (Task 9: enabled / halted / lock / DD / ramp / 25 cap)
  3. fetch live OHLCV for universe (Task 16)
  4. detect new bull/bear formations (Task 7)
  5. spawn LV1 + register LV2/LV3/LV4 pending (this Task 8)
  6. fire any due LV2/LV3/LV4 pending (this Task 8)
  7. save state

Tasks 9, 10, 16 extend this file.
"""
from __future__ import annotations
import hashlib
import logging
import time
from typing import Any

import pandas as pd

from server.strategy.ma_ribbon_auto_state import AutoState
from server.strategy.ma_ribbon_auto_adapter import Phase1Signal


_LOG = logging.getLogger(__name__)


_TF_SECONDS = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
_HIGHER_LAYERS_FOR_LV1: dict[str, str] = {"LV2": "15m", "LV3": "1h", "LV4": "4h"}


def _next_bar_close_after(ts: int, tf_seconds: int) -> int:
    """Smallest k * tf_seconds strictly > ts."""
    return (ts // tf_seconds + 1) * tf_seconds


def register_pending_higher_layers(
    sig: Phase1Signal,
    state: AutoState,
    now_utc: int,
) -> None:
    """Add this signal's LV2/LV3/LV4 to state.pending_signals.
    Caller has already spawned LV1 separately."""
    pending_layers: list[dict[str, Any]] = []
    for layer, tf in _HIGHER_LAYERS_FOR_LV1.items():
        eta = _next_bar_close_after(sig.signal_bar_ts, _TF_SECONDS[tf])
        pending_layers.append({
            "layer": layer,
            "tf": tf,
            "trigger_at_bar_close_after_ts": eta,
        })
    state.pending_signals.append({
        "signal_id": sig.signal_id,
        "symbol": sig.symbol,
        "direction": sig.direction,
        "spawned_layers": ["LV1"],
        "pending_layers": pending_layers,
        "ema21_at_signal": sig.ema21_at_signal,
        "signal_bar_ts": sig.signal_bar_ts,
        "registered_at_utc": now_utc,
    })


def ready_layers_at(state: AutoState, now_utc: int) -> list[dict[str, Any]]:
    """Return list of {signal, layer, tf} entries whose ETA has passed.
    Caller is responsible for verifying ribbon-still-aligned at the layer's TF
    before spawning, and for moving the layer from pending_layers to
    spawned_layers via remove_layer_from_pending after spawn."""
    out = []
    for s in state.pending_signals:
        for layer in s["pending_layers"]:
            if layer["trigger_at_bar_close_after_ts"] <= now_utc:
                out.append({"signal": s, "layer": layer["layer"], "tf": layer["tf"]})
    return out


def remove_layer_from_pending(state: AutoState, signal_id: str, layer: str) -> None:
    for s in state.pending_signals:
        if s["signal_id"] == signal_id:
            s["pending_layers"] = [
                l for l in s["pending_layers"] if l["layer"] != layer
            ]
            if layer not in s["spawned_layers"]:
                s["spawned_layers"].append(layer)
            break


def expire_orphans(state: AutoState, now_utc: int, max_age_seconds: int = 86400) -> None:
    state.pending_signals = [
        s for s in state.pending_signals
        if (now_utc - s.get("registered_at_utc", 0)) < max_age_seconds
    ]


# ---------------------------------------------------------------------------
# Task 9 — six independent risk gates (`can_spawn_layer`)
#
# Any one tripped blocks the spawn. Order matters for reason-string clarity:
# global state first (disabled/halted/locked), then config sanity, then
# market-state-dependent caps. A config bug should be flagged before any
# capital-dependent check so an operator can fix it without misreading.
# ---------------------------------------------------------------------------
from dataclasses import dataclass

from server.strategy.ma_ribbon_auto_state import current_ramp_cap_pct


@dataclass
class GateResult:
    ok: bool
    reason: str = ""


def _strategy_pnl_pct(state: AutoState) -> float:
    cap = state.config.strategy_capital_usd
    if cap <= 0:
        return 0.0
    realized = state.ledger.realized_pnl_usd_cumulative
    unrealized = sum(p.get("unrealized_pnl_usd", 0.0) for p in state.ledger.open_positions)
    return (realized + unrealized) / cap


def _per_symbol_risk_pct(state: AutoState, symbol: str) -> float:
    return sum(
        p.get("risk_pct", 0.0)
        for p in state.ledger.open_positions
        if p.get("symbol") == symbol
    )


def _total_open_risk_pct(state: AutoState) -> float:
    return sum(p.get("risk_pct", 0.0) for p in state.ledger.open_positions)


def can_spawn_layer(
    state: AutoState,
    symbol: str,
    layer: str,
    now_utc: int,
) -> GateResult:
    """Six independent risk gates. Any one tripped blocks the spawn.

    Order matters for the reason string clarity; a config bug should be
    flagged before any market-state-dependent check.
    """
    cfg = state.config
    if not state.enabled:
        return GateResult(False, "strategy disabled")
    if state.halted:
        return GateResult(False, f"halted: {state.halt_reason}")
    if state.locked_until_utc is not None and now_utc < state.locked_until_utc:
        return GateResult(False, f"locked until {state.locked_until_utc}")

    # Per-layer hard size cap (defends against config misedit)
    layer_risk = cfg.layer_risk_pct.get(layer)
    if layer_risk is None or layer_risk > 0.05:
        return GateResult(False, f"per_layer hard cap: {layer} risk {layer_risk}")

    if cfg.strategy_capital_usd <= 0:
        return GateResult(False, "strategy_capital_usd not configured")

    # Concurrent open-orders cap
    if len(state.ledger.open_positions) >= cfg.max_concurrent_orders:
        return GateResult(False, f"concurrent cap {cfg.max_concurrent_orders} reached")

    # DD halt
    pnl_pct = _strategy_pnl_pct(state)
    if pnl_pct <= -cfg.dd_halt_pct:
        return GateResult(False, f"DD halt: PnL {pnl_pct:.4%} <= -{cfg.dd_halt_pct:.0%}")

    # Per-symbol cap
    if _per_symbol_risk_pct(state, symbol) + layer_risk > cfg.per_symbol_risk_cap_pct:
        return GateResult(False,
            f"per_symbol cap {cfg.per_symbol_risk_cap_pct:.1%} exceeded for {symbol}")

    # Ramp-up
    ramp_cap = current_ramp_cap_pct(state, now_utc)
    if _total_open_risk_pct(state) + layer_risk > ramp_cap:
        return GateResult(False,
            f"ramp cap {ramp_cap:.1%} exceeded (current open {_total_open_risk_pct(state):.2%})")

    return GateResult(True, "")


# ---------------------------------------------------------------------------
# Task 10 — Emergency stop + scan_loop orchestrator
# ---------------------------------------------------------------------------
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from server.strategy.ma_ribbon_auto_state import save_state, load_state, StateCorruptError


_LOCK_DURATION_SECONDS = 86_400
_EMERGENCY_LOG_PATH = Path("data/logs/ma_ribbon_emergency_stop.log")
_TICK_INTERVAL_SECONDS = 60


async def flatten_all_ribbon_positions() -> dict[str, int]:
    """Cancel pending Bitget plan orders + market-close open positions for
    every conditional with lineage='ma_ribbon'. Returns counts.

    Stub for now — full implementation lives behind Task 16 wiring once the
    scanner can talk to the watcher's Bitget cancel/close helpers. Tests mock
    this function. Production-time stub returns {0, 0} so emergency_stop
    still completes safely.
    """
    return {"cancelled": 0, "closed": 0}


async def emergency_stop(state: AutoState, now_utc: int, reason: str) -> None:
    state.halted = True
    state.halt_reason = f"emergency_stop: {reason}"
    state.locked_until_utc = now_utc + _LOCK_DURATION_SECONDS
    open_snapshot = list(state.ledger.open_positions)
    state.pending_signals = []

    counts = await flatten_all_ribbon_positions()

    _EMERGENCY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts_utc": now_utc,
        "iso": datetime.fromtimestamp(now_utc, tz=timezone.utc).isoformat(),
        "reason": reason,
        "counts": counts,
        "open_positions_at_stop": open_snapshot,
    }
    with _EMERGENCY_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def scan_loop():
    """Top-level asyncio task. Catches every exception so a single bug
    doesn't take the loop down silently. PRINCIPLES P10: errors visible."""
    while True:
        try:
            await tick()
        except StateCorruptError:
            _LOG.exception("state corrupt — scanner sleeping until manual fix")
            await asyncio.sleep(_TICK_INTERVAL_SECONDS * 5)
        except Exception:  # noqa: BLE001 — top-level safety net
            _LOG.exception("scanner tick failed")
        await asyncio.sleep(_TICK_INTERVAL_SECONDS)


async def tick() -> None:
    """One scanner tick. Idempotent — safe to call multiple times.

    Flow:
      1. load state
      2. enforce gates (enabled / halted / lock / DD / ramp / 25 cap — done by can_spawn_layer)
      3. fetch live OHLCV for universe
      4. detect new bull/bear formations
      5. spawn LV1 + register pending LV2/LV3/LV4
      6. fire ready higher layers (whose ETA has passed AND ribbon still aligned)
      7. expire orphans
      8. save state
    """
    state = load_state()
    if not state.enabled or state.halted:
        save_state(state)
        return
    now_utc = int(time.time())
    if state.locked_until_utc is not None and now_utc < state.locked_until_utc:
        save_state(state)
        return

    # 3. fetch
    try:
        symbols, data = await _fetch_universe_data(state)
    except Exception as exc:  # noqa: BLE001 — surface fetch failure to state.errors_recent (P10/P11)
        _LOG.exception("scanner fetch failed: %s", exc)
        state.errors_recent.append({"ts": now_utc, "stage": "fetch", "error": str(exc)})
        state.errors_recent = state.errors_recent[-50:]
        save_state(state)
        return

    # 4. detect new formations
    new_signals = _detect_new_signals(state, data)

    # 5. spawn LV1 + register higher layers
    for sig in new_signals:
        gate = can_spawn_layer(state, sig.symbol, "LV1", now_utc)
        if not gate.ok:
            _LOG.info("LV1 gate blocked %s: %s", sig.symbol, gate.reason)
            continue
        try:
            await _spawn_layer(state, sig, layer="LV1", now_utc=now_utc)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("LV1 spawn failed for %s: %s", sig.symbol, exc)
            state.errors_recent.append({
                "ts": now_utc, "stage": "spawn_lv1",
                "symbol": sig.symbol, "error": str(exc),
            })
            state.errors_recent = state.errors_recent[-50:]
            continue
        register_pending_higher_layers(sig, state, now_utc=now_utc)
        # mark this signal-bar so we don't re-detect it next tick
        key = f"{sig.symbol}_{sig.tf}_{sig.direction}"
        state.last_processed_bar_ts[key] = sig.signal_bar_ts

    # 6. fire any due higher layers (LV2/LV3/LV4 whose ETA passed)
    for ready in ready_layers_at(state, now_utc=now_utc):
        sig_record = ready["signal"]
        layer = ready["layer"]
        tf = ready["tf"]
        df = data.get((sig_record["symbol"], tf))
        if df is None or df.empty:
            continue
        still_aligned = _is_aligned_at_last_bar(df, sig_record["direction"])
        if not still_aligned:
            _LOG.info(
                "higher-layer %s %s %s no longer aligned at %s — dropping",
                sig_record["symbol"], tf, sig_record["direction"], layer,
            )
            remove_layer_from_pending(state, sig_record["signal_id"], layer)
            continue
        gate = can_spawn_layer(state, sig_record["symbol"], layer, now_utc)
        if not gate.ok:
            _LOG.info("%s gate blocked %s: %s", layer, sig_record["symbol"], gate.reason)
            remove_layer_from_pending(state, sig_record["signal_id"], layer)
            continue
        sig = Phase1Signal(
            signal_id=sig_record["signal_id"],
            symbol=sig_record["symbol"],
            tf=tf,
            direction=sig_record["direction"],
            signal_bar_ts=int(df["timestamp"].iloc[-1]),
            next_bar_open_estimate=float(df["close"].iloc[-1]),
            ema21_at_signal=float(_compute_ema21_last(df)),
        )
        try:
            await _spawn_layer(state, sig, layer=layer, now_utc=now_utc)
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("%s spawn failed for %s: %s", layer, sig.symbol, exc)
            state.errors_recent.append({
                "ts": now_utc, "stage": f"spawn_{layer.lower()}",
                "symbol": sig.symbol, "error": str(exc),
            })
            state.errors_recent = state.errors_recent[-50:]
        remove_layer_from_pending(state, sig_record["signal_id"], layer)

    # 7. expire orphans
    expire_orphans(state, now_utc=now_utc, max_age_seconds=86_400)

    # 8. save
    save_state(state)


# ---------------------------------------------------------------------------
# Task 16 — tick() pipeline helpers
#
# Pull live data, detect, and spawn. Each helper is small and testable in
# isolation. _spawn_layer is the only one with I/O (writes to the
# conditionals store + appends to state.ledger.open_positions). Note that
# the project does NOT expose a free-function `insert_conditional` — the
# canonical path is `ConditionalOrderStore().create(cond)`. The store
# requires a non-empty `conditional_id`, which the adapter leaves blank;
# we mint one here matching the project convention `cond_<sha1[:14]>`.
# ---------------------------------------------------------------------------

async def _fetch_universe_data(state: AutoState):
    """Fetch live OHLCV for every (symbol, TF) in the configured universe.

    Returns (symbols_list, data_dict) where data_dict maps (symbol, tf) -> DataFrame.
    """
    from backtests.ma_ribbon_ema21.data_loader_async import (
        AsyncLoaderConfig, fetch_all_usdt_perp_symbols, fetch_universe_async,
    )
    import httpx
    cfg = AsyncLoaderConfig(
        pages_per_symbol=state.config.fetch_cfg.pages_per_symbol,
        concurrency=state.config.fetch_cfg.concurrency,
    )
    async with httpx.AsyncClient() as client:
        symbols = await fetch_all_usdt_perp_symbols(
            client, cfg,
            min_quote_volume_24h=state.config.universe_filter.min_volume_usd,
            product_types=tuple(state.config.universe_filter.product_types),
        )
    data = await fetch_universe_async(
        symbols=symbols, tfs=state.config.tfs, cfg=cfg,
    )
    return symbols, data


def _detect_new_signals(state: AutoState, data: dict) -> list:
    """Run bull and bear detectors over every (symbol, TF) in `data`.
    Skips bars whose ts is at or past last_processed_bar_ts for that
    (symbol, tf, direction) key."""
    from server.strategy.ma_ribbon_auto_signals import detect_new_signals_for_pair
    out = []
    for (sym, tf), df in data.items():
        for direction in state.config.directions:
            key = f"{sym}_{tf}_{direction}"
            last_ts = state.last_processed_bar_ts.get(key, 0)
            try:
                sigs = detect_new_signals_for_pair(df, sym, tf, direction, last_ts)
            except Exception as exc:  # noqa: BLE001 — never let one bad pair kill the loop (P11)
                _LOG.exception("detect failed for %s %s %s: %s", sym, tf, direction, exc)
                continue
            out.extend(sigs)
    return out


def _is_aligned_at_last_bar(df, direction: str) -> bool:
    """Re-check ribbon alignment on the most recent closed bar of this TF."""
    from server.strategy.ma_ribbon_auto_signals import _enrich, _bear_aligned
    from backtests.ma_ribbon_ema21.ma_alignment import bullish_aligned, AlignmentConfig
    enriched = _enrich(df)
    if direction == "long":
        return bool(bullish_aligned(enriched, AlignmentConfig.default()).iloc[-1])
    else:
        return bool(_bear_aligned(enriched).iloc[-1])


def _compute_ema21_last(df) -> float:
    from backtests.ma_ribbon_ema21.indicators import ema
    series = pd.Series(df["close"].astype(float).values)
    e21 = ema(series, period=21)
    return float(e21.iloc[-1])


def _mint_conditional_id(signal_id: str, layer: str) -> str:
    """Mirror the project convention `cond_<sha1[:14]>` (server/routers/conditionals.py
    `_mk_id`). Salt with time_ns so retries spawn distinct ids."""
    h = hashlib.sha1(f"{signal_id}|{layer}|{time.time_ns()}".encode()).hexdigest()
    return f"cond_{h[:14]}"


async def _spawn_layer(state: AutoState, sig: Phase1Signal, layer: str, now_utc: int) -> None:
    """Spawn one layer: build ConditionalOrder via adapter, persist via the
    canonical ConditionalOrderStore, record in ledger.open_positions for
    risk-cap accounting.

    Supervised first-cycle gate (spec §10): when state.supervised_mode is
    True, the spawn is QUEUED to state.pending_releases instead of being
    persisted. The user must call /api/ma_ribbon_auto/release_layer to push
    it through. After the first release, supervised_mode flips to False and
    subsequent calls run the full path directly.
    """
    if state.supervised_mode:
        state.pending_releases.append({
            "signal_id": sig.signal_id,
            "layer": layer,
            "tf": sig.tf,
            "symbol": sig.symbol,
            "direction": sig.direction,
            "ema21_at_signal": sig.ema21_at_signal,
            "next_bar_open_estimate": sig.next_bar_open_estimate,
            "queued_at_utc": now_utc,
        })
        _LOG.info("supervised: queued %s %s %s %s for manual release",
                  sig.symbol, sig.tf, sig.direction, layer)
        return

    from server.strategy.ma_ribbon_auto_adapter import signal_to_conditional
    cond = signal_to_conditional(sig, layer=layer, state=state, now_utc=now_utc)
    cond.conditional_id = _mint_conditional_id(sig.signal_id, layer)

    # Project convention: server/conditionals/store.py exposes
    # ConditionalOrderStore (resolves at import time to the active backend
    # — SQLite by default, JSON via COND_STORE_BACKEND=json). The spec's
    # `insert_conditional` free function does not exist in this codebase;
    # we use .create() directly, matching mar_bb_runner / drawings router.
    from server.conditionals.store import ConditionalOrderStore
    store = ConditionalOrderStore()
    store.create(cond)

    state.ledger.open_positions.append({
        "signal_id": sig.signal_id, "layer": layer, "tf": sig.tf,
        "symbol": sig.symbol, "direction": sig.direction,
        "risk_pct": state.config.layer_risk_pct[layer],
        "unrealized_pnl_usd": 0.0,
        "spawned_at_utc": now_utc,
        "conditional_id": cond.conditional_id,
    })
    _LOG.info("spawned %s %s %s %s (signal_id=%s cond_id=%s)",
              sig.symbol, sig.tf, sig.direction, layer,
              sig.signal_id[:8], cond.conditional_id)
