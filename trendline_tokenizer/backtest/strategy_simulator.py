"""Paper PnL simulator over replayed signals.

Strategy: when a BOUNCE / BREAK signal fires, open a position and hold
until either the suggested buffer is hit (stop) or N bars elapse
(time-out exit). PnL is in % of entry price - this is a sanity-check
backtest, NOT a slippage/funding-aware production simulator.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ..inference.signal_engine import SignalRecord
from .replay_engine import ReplayStep


@dataclass
class Position:
    open_bar: int
    entry_price: float
    direction: str          # "long" | "short"
    stop_pct: float         # buffer
    expiry_bar: int
    signal: SignalRecord


@dataclass
class Trade:
    open_bar: int
    close_bar: int
    direction: str
    entry: float
    exit: float
    return_pct: float
    reason: str             # "stop" | "expiry"
    signal_action: str      # BOUNCE / BREAK
    signal_confidence: float


def _close_position(pos: Position, bar: ReplayStep, reason: str) -> Trade:
    px = bar.close
    if pos.direction == "long":
        ret = (px - pos.entry_price) / pos.entry_price
    else:
        ret = (pos.entry_price - px) / pos.entry_price
    return Trade(
        open_bar=pos.open_bar, close_bar=bar.bar_index,
        direction=pos.direction, entry=pos.entry_price,
        exit=px, return_pct=ret, reason=reason,
        signal_action=pos.signal.action,
        signal_confidence=pos.signal.confidence,
    )


def simulate(
    steps: Iterable[ReplayStep],
    *,
    hold_bars: int = 20,
    min_confidence: float = 0.55,
    open_long_on: tuple[str, ...] = ("BOUNCE",),
    open_short_on: tuple[str, ...] = ("BREAK",),
) -> list[Trade]:
    """One open position max. Position closes on stop hit or expiry."""
    trades: list[Trade] = []
    open_pos: Position | None = None
    for step in steps:
        # update open position
        if open_pos is not None:
            sig = open_pos.signal
            px = step.close
            if open_pos.direction == "long":
                stop_px = open_pos.entry_price * (1 - open_pos.stop_pct)
                if px <= stop_px:
                    trades.append(_close_position(open_pos, step, "stop"))
                    open_pos = None
            else:
                stop_px = open_pos.entry_price * (1 + open_pos.stop_pct)
                if px >= stop_px:
                    trades.append(_close_position(open_pos, step, "stop"))
                    open_pos = None
            if open_pos is not None and step.bar_index >= open_pos.expiry_bar:
                trades.append(_close_position(open_pos, step, "expiry"))
                open_pos = None

        # maybe open a new position
        if open_pos is None and step.signal is not None:
            sig = step.signal
            if sig.confidence >= min_confidence:
                if sig.action in open_long_on:
                    open_pos = Position(
                        open_bar=step.bar_index, entry_price=step.close,
                        direction="long",
                        stop_pct=max(0.001, sig.suggested_buffer_pct),
                        expiry_bar=step.bar_index + hold_bars,
                        signal=sig,
                    )
                elif sig.action in open_short_on:
                    open_pos = Position(
                        open_bar=step.bar_index, entry_price=step.close,
                        direction="short",
                        stop_pct=max(0.001, sig.suggested_buffer_pct),
                        expiry_bar=step.bar_index + hold_bars,
                        signal=sig,
                    )
    return trades
