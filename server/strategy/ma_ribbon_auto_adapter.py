"""Pure translation from Phase1Signal -> ConditionalOrder. No I/O.

A `Phase1Signal` is a single ribbon-formation event for one (symbol, TF,
direction). The scanner (Tasks 7-10) produces these; this adapter converts
each one into a ConditionalOrder record that the existing watcher pipeline
can execute (Tasks 5-6 add the lineage="ma_ribbon" branches in the
watcher).

Lineage contract (spec section 'Naming conventions'):
- All MA-ribbon ConditionalOrders carry lineage="ma_ribbon".
- manual_line_id=None (ribbon orders are not anchored to a drawn line).
- side="support" is nominal placeholder; the watcher branches on lineage,
  not side, when computing SL.

Risk math (spec section 4.2 / 4.3):
- risk_usd = strategy_capital_usd * layer_risk_pct[layer].
- SL at signal time = ema21_at_signal * (1 - buffer) for long,
                    = ema21_at_signal * (1 + buffer) for short,
  where buffer = ribbon_buffer_pct[tf].
- entry_to_sl_pct = |entry - sl| / entry (fraction).
- qty_notional_usd = risk_usd / entry_to_sl_pct.

Spec ref: docs/superpowers/specs/2026-04-25-ma-ribbon-auto-execution-design.md.
"""
from __future__ import annotations
from dataclasses import dataclass

from server.strategy.ma_ribbon_auto_state import AutoState, current_ramp_cap_pct
from server.conditionals.types import ConditionalOrder, OrderConfig, TriggerConfig


@dataclass
class Phase1Signal:
    """One ribbon-formation event for a (symbol, TF, direction).

    Produced by the scanner (Task 7+). The scanner shares one signal_id
    across all 4 layers spawned from the same event, so downstream code
    can reconcile LV1..LV4 back to the original detection.

    next_bar_open_estimate is the price the scanner expects the order to
    fill at — typically the last close, since the spec wants to spawn the
    order on bar-close and have it trigger at next-bar-open. It's an
    estimate; the real fill price comes from the exchange.
    """
    signal_id: str
    symbol: str
    tf: str
    direction: str           # "long" | "short"
    signal_bar_ts: int
    next_bar_open_estimate: float
    ema21_at_signal: float


def _entry_to_sl_pct(entry: float, sl: float, direction: str) -> float:
    """Distance from entry to SL as a positive fraction.

    For long: (entry - sl) / entry; expects sl < entry.
    For short: (sl - entry) / entry; expects sl > entry.

    A negative result means SL is on the wrong side of entry — the caller
    should treat that as a build error and skip spawning.
    """
    if entry <= 0:
        raise ValueError(f"non-positive entry {entry}")
    if direction == "long":
        return (entry - sl) / entry
    elif direction == "short":
        return (sl - entry) / entry
    else:
        raise ValueError(f"unknown direction {direction!r}")


def signal_to_conditional(
    sig: Phase1Signal,
    layer: str,
    state: AutoState,
    now_utc: int,
) -> ConditionalOrder:
    """Build a ConditionalOrder for one layer of one signal.

    Raises:
        ValueError: strategy_capital_usd not set, or entry_to_sl_pct <= 0
                    (would mean SL is on the wrong side of entry).
        KeyError:   layer not in {LV1, LV2, LV3, LV4}, or sig.tf not in
                    the configured ribbon_buffer_pct dict.
    """
    cfg = state.config
    if cfg.strategy_capital_usd <= 0:
        raise ValueError("strategy_capital_usd must be > 0 to spawn orders")

    # KeyError on bad tf or bad layer is intentional — spec contract is that
    # the scanner only emits signals for configured tfs/layers. A surprise
    # key is a programming error and we want it loud, not silently defaulted.
    buffer_pct = cfg.ribbon_buffer_pct[sig.tf]
    risk_pct = cfg.layer_risk_pct[layer]
    risk_usd = cfg.strategy_capital_usd * risk_pct

    if sig.direction == "long":
        sl_at_signal = sig.ema21_at_signal * (1.0 - buffer_pct)
    elif sig.direction == "short":
        sl_at_signal = sig.ema21_at_signal * (1.0 + buffer_pct)
    else:
        raise ValueError(f"unknown direction {sig.direction!r}")

    entry_to_sl_pct = _entry_to_sl_pct(
        sig.next_bar_open_estimate, sl_at_signal, sig.direction
    )
    if entry_to_sl_pct <= 0:
        # SL on the wrong side of entry means buffer ate through entry —
        # would imply a pathological signal where ema21 +/- buffer crosses
        # the bar-open estimate. We don't size such an order.
        raise ValueError(
            f"entry_to_sl_pct must be > 0 (entry={sig.next_bar_open_estimate}, "
            f"sl={sl_at_signal}, dir={sig.direction})"
        )
    qty_notional_usd = risk_usd / entry_to_sl_pct

    ribbon_meta = {
        "signal_id":                  sig.signal_id,
        "layer":                      layer,
        "tf":                         sig.tf,
        "ribbon_buffer_pct":          buffer_pct,
        "ema21_at_signal":            sig.ema21_at_signal,
        "initial_sl_estimate":        sl_at_signal,
        "ramp_day_cap_pct_at_spawn":  current_ramp_cap_pct(state, now_utc),
        # Per spec section 'Lineage': ribbon orders explicitly OPT OUT of
        # PRINCIPLES.md P7 auto-reverse. The strategy reverses on its own
        # cadence (next ribbon event), not on every SL hit.
        "reverse_on_stop":            False,
    }

    order_cfg = OrderConfig(
        direction=sig.direction,         # type: ignore[arg-type]
        sl_logic="ribbon_ema21_trailing",
        ribbon_meta=ribbon_meta,
        risk_usd_target=risk_usd,
        qty_notional_target=qty_notional_usd,
        # Manual-line offsets are NOT used for ribbon lineage — the watcher
        # branches on sl_logic and reads ribbon_meta instead. Set to None
        # so any code that accidentally falls through the manual path
        # surfaces immediately (None will fail any arithmetic on it).
        entry_offset_points=None,
        stop_points=None,
    )

    return ConditionalOrder(
        # ID is left blank here — the scanner mints the canonical
        # conditional_id when it persists the order. Adapter is pure
        # translation and has no clock / RNG.
        conditional_id="",
        lineage="ma_ribbon",
        manual_line_id=None,
        symbol=sig.symbol,
        timeframe=sig.tf,
        # `side` is structurally required by the existing dataclass but is
        # semantically unused for ribbon lineage. Pin to "support" so
        # serialisation never produces a null. Watcher branches on lineage,
        # not side, for ribbon orders.
        side="support",
        # Line geometry is unused for ribbon orders; preserve sane defaults
        # so the rendering layer doesn't choke on None.
        t_start=sig.signal_bar_ts,
        t_end=sig.signal_bar_ts,
        price_start=sig.next_bar_open_estimate,
        price_end=sig.next_bar_open_estimate,
        pattern_stats_at_create={},
        trigger=TriggerConfig(),
        order=order_cfg,
        status="pending",
        created_at=now_utc,
        updated_at=now_utc,
    )


__all__ = ["Phase1Signal", "signal_to_conditional", "_entry_to_sl_pct"]
