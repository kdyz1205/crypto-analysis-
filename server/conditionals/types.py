"""Dataclasses for conditional orders tied to drawn trendlines."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────
# Status transitions
#
#   pending   — watcher is polling; line not touched yet
#   triggered — line was touched, Telegram alert fired. If exchange
#               submit is configured, exchange_order_id is populated.
#   cancelled — user cancelled, or auto-cancelled by stop conditions
#               (line broken / price drifted too far / expired)
#   failed    — triggered but exchange submit errored
# ─────────────────────────────────────────────────────────────────────
ConditionalStatus = Literal["pending", "triggered", "cancelled", "failed"]


@dataclass(frozen=True, slots=True)
class TriggerConfig:
    """How the watcher decides the line has been "touched"."""
    # Tolerance as fraction of ATR at the current bar. 0.2 = within 0.2 × ATR.
    tolerance_atr: float = 0.2
    # How often the watcher polls, in seconds. Defaults tuned per TF below.
    poll_seconds: int = 60
    # Auto-cancel after this many seconds without triggering. 0 = no expiry.
    max_age_seconds: int = 48 * 3600
    # Auto-cancel if price drifts beyond this × ATR from the line.
    # Means: "the line is too far from market to reasonably trigger soon."
    max_distance_atr: float = 5.0
    # Auto-cancel if price closes through the line by >= this × ATR.
    # Means: "the line is visually broken."
    break_threshold_atr: float = 0.5


@dataclass(frozen=True, slots=True)
class OrderConfig:
    """What to do when the trigger fires."""
    # Direction — usually auto-picked from line side (support → long, resistance → short)
    direction: Literal["long", "short"]
    # Entry offset as fraction of ATR from the line. Positive = further away
    # in the "safe" direction (for support, above the line; for resistance, below).
    entry_offset_atr: float = 0.0
    # Stop distance from the line, in ATR, opposite of the "safe" direction.
    stop_atr: float = 0.3
    # Take profit — choose ONE:
    #   rr_target:   fixed R multiple (e.g. 2.0 = 2R target)
    #   tp_price:    absolute price
    # If both are None, no TP is set (manual close).
    rr_target: float | None = 2.0
    tp_price: float | None = None
    # Size — choose ONE:
    #   notional_usd:   fixed USD notional
    #   equity_pct:     percentage of total equity
    #   risk_pct:       percentage of equity to risk (respects stop distance)
    notional_usd: float | None = None
    equity_pct: float | None = None
    risk_pct: float | None = 0.005  # default 0.5% risk
    # Whether to actually submit to exchange (True) or only alert (False)
    submit_to_exchange: bool = False
    # Exchange account mode (only if submit_to_exchange=True)
    exchange_mode: Literal["paper", "live"] = "paper"


@dataclass(frozen=True, slots=True)
class ConditionalEvent:
    """One log entry on a conditional order's lifetime."""
    ts: int
    kind: Literal[
        "created", "poll", "triggered", "exchange_submitted",
        "exchange_acked", "exchange_error", "cancelled", "expired",
        "line_broken", "drifted_far", "tolerance_check",
    ]
    price: float | None = None
    line_price: float | None = None
    distance_atr: float | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionalOrder:
    """A conditional order tied to a drawn trendline.

    Mutable because status + events list evolve over time. Write through
    ConditionalOrderStore.update_status / .append_event rather than
    mutating directly.
    """
    conditional_id: str
    manual_line_id: str          # FK to ManualTrendline
    symbol: str
    timeframe: str
    side: Literal["support", "resistance"]
    # Line coords (snapshot at creation — we recompute projection each poll)
    t_start: int
    t_end: int
    price_start: float
    price_end: float

    # Pattern stats snapshot (what the user saw when they approved)
    pattern_stats_at_create: dict[str, Any]

    # Config
    trigger: TriggerConfig
    order: OrderConfig

    # State
    status: ConditionalStatus
    created_at: int
    updated_at: int
    triggered_at: int | None = None
    cancelled_at: int | None = None
    cancel_reason: str = ""

    # Fills
    exchange_order_id: str | None = None
    fill_price: float | None = None
    fill_qty: float | None = None

    # Activity log
    events: list[ConditionalEvent] = field(default_factory=list)

    # Last-known state from the watcher (for UI display)
    last_poll_ts: int | None = None
    last_market_price: float | None = None
    last_line_price: float | None = None
    last_distance_atr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # events is already dicts via asdict
        return d

    def line_price_at(self, ts: int) -> float:
        """Project the line's price at timestamp `ts`.

        Linear interpolation from (t_start, price_start) to (t_end, price_end),
        extended beyond t_end.
        """
        span = self.t_end - self.t_start
        if span <= 0:
            return self.price_start
        slope_per_sec = (self.price_end - self.price_start) / span
        return self.price_start + slope_per_sec * (ts - self.t_start)


__all__ = [
    "ConditionalStatus",
    "TriggerConfig",
    "OrderConfig",
    "ConditionalEvent",
    "ConditionalOrder",
]
