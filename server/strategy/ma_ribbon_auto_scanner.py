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
import logging
from typing import Any

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
