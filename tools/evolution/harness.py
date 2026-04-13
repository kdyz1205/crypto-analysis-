"""Fade + flip backtest harness — walk-forward, NO look-ahead.

Core correctness rule:
  A line is defined by its FORMATION — the moment the second anchor is
  confirmed. From that moment onwards, we only know what the candles will
  do *after* it. We organically scan forward, and every time price returns
  to the line (a "touch"), that's a new setup.

What we explicitly DO NOT do:
  - We do not consult pre-computed touch_indices from production detectors
    that looked at the full dataset. That was the look-ahead leak in the
    first baseline run.
  - We do not "know" touch 3/4/5 will happen. We discover them bar-by-bar.

Multi-setup per line (the user's "3-touch 4-touch 哪个更赚" question):
  Each organic touch after the 2 anchor touches is an independent setup.
  setup_touch_number = 2 + k, where k is the kth *organic* touch discovered
  after line formation (k=1 → touch number 3 in trader terms).

Strategy (FIXED, not evolved):
  1. FADE leg: limit at line price when price returns to the line.
     Direction: support → long, resistance → short.
     Stop: stop_atr_mult × ATR on the far side of the line.
     Target: target_R × initial_risk.
  2. FLIP leg: only when FADE stopped out. Market entry at stop fill price,
     opposite direction, same tight stop, flip_target_R × risk target.

Line life cycle: a line stays active until one of:
  - max_setups_per_line setups completed
  - Candle CLOSE crosses the line by break_threshold_atr
  - max_life_bars elapsed since formation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from server.strategy.evolved.base import EvolvedLine


# ──────────────────────────────────────────────────────────────
# Harness parameters
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HarnessParams:
    stop_atr_mult: float = 0.3
    target_R: float = 2.5
    flip_target_R: float = 2.0
    entry_touch_tolerance_atr: float = 0.15
    max_hold_bars: int = 50
    fee_rt: float = 0.0006
    slippage: float = 0.0003
    # Line life cycle
    # `max_setups_per_line` is the NUMBER OF SETUPS allowed per line.
    # touch_number = 2 + setup_count, so:
    #   =1 → only touch=3 is traded
    #   =2 → touches 3, 4 are traded  ← reflection sweet spot
    #   =3 → touches 3, 4, 5 (5 is marginal/negative on v1_clean)
    max_setups_per_line: int = 2
    max_life_bars: int = 400
    break_threshold_atr: float = 0.5
    min_bars_between_touches: int = 3
    # When False, disable the flip leg entirely. Round 1 flip legs had
    # WR ≤17% against a 2R target (needs ≥33% to breakeven) → structural
    # cost. Use True to enable, False for fade-only variants.
    enable_flip: bool = True


# ──────────────────────────────────────────────────────────────
# Output schemas
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Trade:
    leg: str
    line_side: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    direction: int
    R: float
    reason: str                       # "target" | "stop" | "timeout"
    setup_touch_number: int = 0
    line_id: int = -1


@dataclass(slots=True)
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    n_lines_tested: int = 0
    n_setups_triggered: int = 0
    n_lines_with_any_trade: int = 0


# ──────────────────────────────────────────────────────────────
# ATR helper
# ──────────────────────────────────────────────────────────────
def _atr_series(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Causal ATR: no look-ahead.

    - Previous close for TR on bar 0 is NaN → TR[0] = high-low (already fine).
    - ewm(..., min_periods=period) leaves the first `period` bars as NaN.
    - We do NOT bfill (that uses future values). Instead we forward-fill
      from the first valid value, and for the pre-first-valid window we
      substitute the MEDIAN of the first `period` TR values (computed once,
      backward-looking from that window itself — still no look-ahead of
      bars > period).
    - Use 1e-6 * close as floor rather than 1e-9 to avoid phantom touches
      where tiny ATR makes tolerance collapse.
    """
    h = candles["high"].astype(float)
    l = candles["low"].astype(float)
    c = candles["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    # Fill pre-period bars with a price-scaled floor (not future data)
    floor = c * 1e-4  # 1 bp of price — conservative
    atr = atr.where(atr.notna(), floor)
    # Final safety: no zeros
    return atr.clip(lower=1e-6)


# ──────────────────────────────────────────────────────────────
# Line-touch detection (organic, no look-ahead)
# ──────────────────────────────────────────────────────────────
def _find_next_touch(
    line: EvolvedLine,
    highs, lows, atr,
    from_bar: int,
    to_bar: int,
    tolerance_atr: float,
) -> int | None:
    """Scan bars [from_bar, to_bar) for the first bar whose range includes
    the projected line price (within tolerance_atr × ATR).

    Returns the bar index of the touch or None.
    """
    for bar in range(from_bar, min(to_bar, len(highs))):
        line_price = line.price_at(bar)
        tol = tolerance_atr * atr[bar]
        # price range of the bar
        lo, hi = lows[bar], highs[bar]
        if lo <= line_price + tol and hi >= line_price - tol:
            return bar
    return None


def _line_broken_by_close(
    line: EvolvedLine,
    closes, atr,
    from_bar: int,
    to_bar: int,
    break_threshold_atr: float,
) -> int | None:
    """Return the bar index where a CLOSE cleanly broke the line (support
    closed below, or resistance closed above), else None.
    """
    for bar in range(from_bar, min(to_bar, len(closes))):
        lp = line.price_at(bar)
        thresh = break_threshold_atr * atr[bar]
        if line.side == "support":
            if closes[bar] < lp - thresh:
                return bar
        else:  # resistance
            if closes[bar] > lp + thresh:
                return bar
    return None


# ──────────────────────────────────────────────────────────────
# Core harness (walk-forward, organic touches)
# ──────────────────────────────────────────────────────────────
def run_backtest(
    candles: pd.DataFrame,
    lines: Sequence[EvolvedLine],
    params: HarnessParams | None = None,
) -> BacktestResult:
    if params is None:
        params = HarnessParams()

    candles = candles.reset_index(drop=True)
    n = len(candles)
    result = BacktestResult(n_lines_tested=len(lines or []))
    if n < 30 or not lines:
        return result

    atr = _atr_series(candles).values
    highs = candles["high"].astype(float).values
    lows = candles["low"].astype(float).values
    closes = candles["close"].astype(float).values
    opens = candles["open"].astype(float).values

    for line_idx, line in enumerate(lines):
        formation_bar = line.end_index  # second anchor = moment of formation
        if formation_bar >= n - 2:
            continue

        watch_start = formation_bar + 1
        life_end = min(n, formation_bar + params.max_life_bars + 1)
        line_trade_count = 0
        line_produced_any = False

        # Walk forward; each iteration finds the next organic touch.
        bar_cursor = watch_start
        while bar_cursor < life_end and line_trade_count < params.max_setups_per_line:
            touch_bar = _find_next_touch(
                line, highs, lows, atr,
                bar_cursor, life_end,
                params.entry_touch_tolerance_atr,
            )
            if touch_bar is None:
                break

            # Check: is the line already broken before this touch? If a
            # close broke through between bar_cursor and touch_bar, the
            # line is dead and we skip this setup.
            break_bar = _line_broken_by_close(
                line, closes, atr,
                bar_cursor, touch_bar,
                params.break_threshold_atr,
            )
            if break_bar is not None:
                break

            # Simulate fade at this touch
            fade = _simulate_fade_at(
                line, highs, lows, atr,
                touch_bar, min(touch_bar + params.max_hold_bars - 1, life_end - 1),
                params,
            )
            if fade is None:
                bar_cursor = touch_bar + max(1, params.min_bars_between_touches)
                continue

            line_trade_count += 1
            line_produced_any = True
            fade.line_id = line_idx
            fade.setup_touch_number = 2 + line_trade_count  # organic 3rd, 4th, ...
            result.trades.append(fade)
            result.n_setups_triggered += 1

            # Flip leg on stop out (only). Enter at the NEXT bar's open price,
            # not at the stale fade-stop price. This avoids the entry-bar-stop
            # death spiral where the wick of the same bar kills every flip.
            if fade.reason == "stop" and params.enable_flip:
                flip_start = fade.exit_index + 1
                if flip_start < n:
                    flip = _simulate_flip_after(
                        fade, opens, highs, lows, atr,
                        flip_start, min(flip_start + params.max_hold_bars - 1, n - 1),
                        params,
                    )
                    if flip is not None:
                        flip.line_id = line_idx
                        flip.setup_touch_number = 2 + line_trade_count
                        result.trades.append(flip)

                # A stop-out means the line was hit hard — check if it's dead.
                # If close broke through during the fade, line dies.
                died = _line_broken_by_close(
                    line, closes, atr,
                    touch_bar, fade.exit_index + 1,
                    params.break_threshold_atr,
                )
                if died is not None:
                    break

            # Resume scan after the trade + a cooldown
            bar_cursor = fade.exit_index + max(1, params.min_bars_between_touches)

        if line_produced_any:
            result.n_lines_with_any_trade += 1

    return result


# ──────────────────────────────────────────────────────────────
# Fade / Flip simulation primitives
# ──────────────────────────────────────────────────────────────
def _simulate_fade_at(
    line: EvolvedLine,
    highs, lows, atr,
    entry_bar: int,
    end_bar: int,
    params: HarnessParams,
) -> Trade | None:
    """Fade fills at the projected line price at entry_bar. Then walk forward
    bar-by-bar (INCLUDING entry_bar, pessimistically) until stop/target/timeout.
    """
    line_price = line.price_at(entry_bar)
    bar_atr = atr[entry_bar]
    direction = +1 if line.side == "support" else -1

    if direction > 0:  # long at support
        entry_price = line_price
        stop_price = line_price - params.stop_atr_mult * bar_atr
        risk = entry_price - stop_price
    else:
        entry_price = line_price
        stop_price = line_price + params.stop_atr_mult * bar_atr
        risk = stop_price - entry_price

    if risk <= 0:
        return None
    target_price = entry_price + direction * (params.target_R * risk)

    return _simulate_leg(
        leg="fade",
        line_side=line.side,
        entry_bar=entry_bar,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        direction=direction,
        highs=highs, lows=lows,
        params=params,
        end_bar=end_bar,
    )


def _simulate_flip_after(
    fade: Trade,
    opens, highs, lows, atr,
    entry_bar: int,
    end_bar: int,
    params: HarnessParams,
) -> Trade | None:
    """Flip = opposite direction, market-entered at the OPEN of the bar
    after fade stop-out.

    Using fade.exit_price (stale, from the end of the fade bar) caused a
    degenerate case where the flip's stop was 0.3 ATR from a stale anchor
    and got killed on entry_bar's wick the instant the leg was opened —
    producing the impossible 0% flip win rate observed in round 1 + 2 of
    the walk-forward baseline.

    We also start the stop/target scan at entry_bar + 1 rather than
    entry_bar, because entry_bar's open IS our entry and the rest of that
    bar's range can absolutely go either way but that's equivalent to
    immediately trading post-entry which is already counted.
    """
    if entry_bar >= len(opens):
        return None
    direction = -fade.direction
    entry_price = float(opens[entry_bar])
    bar_atr = atr[entry_bar]

    if direction > 0:
        stop_price = entry_price - params.stop_atr_mult * bar_atr
        risk = entry_price - stop_price
    else:
        stop_price = entry_price + params.stop_atr_mult * bar_atr
        risk = stop_price - entry_price

    if risk <= 0:
        return None
    target_price = entry_price + direction * (params.flip_target_R * risk)

    return _simulate_leg(
        leg="flip",
        line_side=fade.line_side,
        entry_bar=entry_bar,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        direction=direction,
        highs=highs, lows=lows,
        params=params,
        end_bar=end_bar,
    )


def _simulate_leg(
    *, leg, line_side, entry_bar, entry_price, stop_price, target_price,
    direction, highs, lows, params, end_bar,
) -> Trade:
    """Walk forward from entry_bar (inclusive) until stop/target/end.

    Pessimistic same-bar priority: if both stop and target are reachable
    within the same bar's range, STOP wins.

    end_bar is INCLUSIVE. Caller should pass (entry_bar + max_hold_bars - 1)
    to get exactly max_hold_bars of holding window including the entry bar.
    """
    exit_idx = end_bar
    exit_px = None
    reason = "timeout"

    for bar in range(entry_bar, end_bar + 1):
        hit_stop = False
        hit_target = False
        if direction > 0:
            if lows[bar] <= stop_price:
                hit_stop = True
            if highs[bar] >= target_price:
                hit_target = True
        else:
            if highs[bar] >= stop_price:
                hit_stop = True
            if lows[bar] <= target_price:
                hit_target = True

        if hit_stop:
            exit_idx, exit_px, reason = bar, stop_price, "stop"
            break
        if hit_target:
            exit_idx, exit_px, reason = bar, target_price, "target"
            break

    if exit_px is None:
        exit_px = float((highs[end_bar] + lows[end_bar]) / 2.0)

    risk = abs(entry_price - stop_price)
    raw_pnl = (exit_px - entry_price) * direction
    cost = (params.fee_rt + 2 * params.slippage) * entry_price
    net = raw_pnl - cost
    R = net / risk if risk > 0 else 0.0

    return Trade(
        leg=leg,
        line_side=line_side,
        entry_index=entry_bar,
        exit_index=exit_idx,
        entry_price=entry_price,
        exit_price=exit_px,
        stop_price=stop_price,
        target_price=target_price,
        direction=direction,
        R=R,
        reason=reason,
    )
